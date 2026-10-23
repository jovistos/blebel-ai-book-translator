"""Body-text damage detection and repair (stage 2).

The book this pipeline was built for came out of a PDF converter with four
kinds of damage that are *shape*-recognisable, and one repair for each:

``page_number_in_word``
    A printed page number landed inside a word, because the word was split
    across a page break: ``pažljivu psi`` / ``182 ljutnje``. Repair: join the
    fragments and drop the number, so the word becomes ``psiljutnje``.
``inline_page_number``
    A printed page number was merged into a text line instead of sitting on its
    own: ``- 2 0 "Well`` or ``... to be mean to him the - 6 8 -``. Repair: drop
    the token and keep every word on both sides.
``split_word``
    A word was cut at a line end with no hyphen: ``... i božan199`` /
    ``stvena ...``. Repair: join the two lines, dropping the printed number.
``space_before_punctuation``
    ``mišić ,`` / ``pomoć ,``. Repair: remove the space.

Each detector returns *evidence* (line numbers, offsets, the reconstructed
word) and the repair is defined as a pure string function, so a finding can be
reproduced without writing anything, and an applied repair can be re-checked
byte by byte against the original.

Findings carry a ``risk``. ``low`` means the shape can only be damage; ``high``
means a human should look (for example an acronym such as ``TxA2``).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from book_translator.core.constants import MAX_PAGE_NUMBER_DIGITS, PAGE_NUMBER_RUN_WINDOW
from book_translator.documents.bodyline import (
    count_space_before_punct,
    describes_split_number,
    has_midword_number,
    is_caption_line,
    is_word_continuation,
)
from book_translator.documents.io import load_lines
from book_translator.documents.markdown import header_of, is_page_number_line, page_number_of
from book_translator.documents.text import excerpt

#: A finding kind (also the ``fix`` name an agent may request).
PAGE_NUMBER_IN_WORD = "page_number_in_word"
INLINE_PAGE_NUMBER = "inline_page_number"
SPLIT_WORD = "split_word"
SPACE_BEFORE_PUNCTUATION = "space_before_punctuation"
#: Every damage kind the scanner knows.
DAMAGE_KINDS = (PAGE_NUMBER_IN_WORD, INLINE_PAGE_NUMBER, SPLIT_WORD, SPACE_BEFORE_PUNCTUATION)

#: Digits glued to the previous word (``us12 merena``): keep the fragments glued.
MIDWORD_RE = re.compile(r"(?<=[a-zžćčšđ])(\d{1,4})(?=\s+[a-zžćčšđ])")
#: Digits glued to the *front* of a word (``1otkucaja``): delete the run.
LEADING_GLUED_RE = re.compile(r"^\d{1,4}(?=[a-zžćčšđ])")
#: Digits printed in front of a lowercase word (``30 spreč avaju``): delete the run
#: and the space, so the two fragments of the split word are joined.
LEADING_SPACED_RE = re.compile(r"^(\d{1,4})\s+(?=[a-zžćčšđ])")

#: A page number merged into a text line, with the dashes the converter kept:
#: ``- 2 0 "Well`` at the start, ``... the - 6 8 -`` at the end, or a token in
#: the middle (``... a year. - 117 "Obviously,"``). Whitespace is required on
#: both sides so a real hyphen inside a word (``mid-1980s``) never matches.
INLINE_DASHED_PAGE_RE = re.compile(
    r"(?<!\S)(?P<token>[-–—]\s*\d(?:\s*\d){0,3}\s*[-–—]?)(?=\s|$)"
)
#: A bare number the converter left at the very start or end of a text line
#: (``... Valerie 265``). Only line edges are considered, so a number in the
#: middle of a sentence is never a candidate.
INLINE_BARE_PAGE_RE = re.compile(r"(?<!\S)(?P<token>\d(?:\s*\d){0,3})(?=\s|$)")

#: Canonical repairs: what each finding kind does, and how safe it is.
FIXES: dict[str, dict[str, str]] = {
    PAGE_NUMBER_IN_WORD: {
        "summary": "drop a printed page number printed inside a word and close the word",
        "risk": "low",
        "example": "'pažljivu psi' + '182 ljutnje' -> 'pažljivu psiljutnje'",
    },
    INLINE_PAGE_NUMBER: {
        "summary": "drop a printed page number merged into a text line, keeping the words around it",
        "risk": "medium",
        "example": "'- 2 0 \"Well,' -> '\"Well,'",
    },
    SPLIT_WORD: {
        "summary": "join a line whose last word continues on the next line",
        "risk": "low",
        "example": "'... i božan199' + 'stvena.' -> '... i božan199stvena.'",
    },
    SPACE_BEFORE_PUNCTUATION: {
        "summary": "remove whitespace before , . ; : ! ?",
        "risk": "low",
        "example": "'mišić ,' -> 'mišić,'",
    },
}


@dataclass
class Finding:
    """One piece of detected damage, with the repair that removes it."""

    kind: str
    line_number: int
    risk: str = "low"
    before: str = ""
    after: str = ""
    evidence: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def repair_page_number_in_word(
    line: str,
    extra_chars: frozenset[int] = frozenset(),
) -> str:
    """Repair a printed page number that landed *inside* a word.

    Three shapes, one idea — the digits are a page number and the word around
    them must be closed up:

    * ``... psi182 ljutnje`` → ``... psiljutnje`` — the digits and the whitespace
      after them are deleted, because the word continues on both sides of them;
    * ``1otkucaja`` → ``otkucaja`` — the digits at the start of the line fused to
      the word are deleted and the word is left intact;
    * ``30 spreč avaju`` → ``spreč avaju`` — a page number printed in front of the
      first word of a line is deleted together with the space after it, which
      leaves the sentence readable.

    That last shape is the only one where the fragments are *not* glued back
    together. ``30 spreč avaju`` is one word split by the page break, but
    ``40 godina, patila je ...`` is a real number opening a paragraph and looks
    exactly the same, so joining is not something a rule may decide — see the
    "Known limitation" section of the README.

    An inline number the author wrote (``130 otkucaja``) has whitespace before it
    and a *different* word after it, so it matches none of the rules. Offsets in
    ``extra_chars`` are skipped.
    """
    out: list[str] = []
    last = 0
    seen = False
    for m in MIDWORD_RE.finditer(line):
        if m.start(1) in extra_chars:
            continue
        seen = True
        out.append(line[last : m.start(1)])
        ws_end = m.end(1)
        while ws_end < len(line) and line[ws_end] in " \t":
            ws_end += 1
        last = ws_end
    if seen:
        out.append(line[last:])
        line = "".join(out)

    if extra_chars & {0}:
        return line
    spaced = LEADING_SPACED_RE.match(line)
    if spaced:
        # Drop the digits and the space, then let the mid-word pass above close
        # the word up (``30 spreč avaju`` → ``sprečavaju``) when the sentence
        # clearly continues from the previous line.
        rest = line[spaced.end(1) :].lstrip()
        if not rest:
            return line[spaced.end(1) :]
        return rest
    return LEADING_GLUED_RE.sub("", line, count=1)


def _only_page_decor(text: str) -> bool:
    """True when ``text`` holds no words — only page-number furniture."""
    return not text.strip("-_–—# \t")


def _inline_candidate(text: str, match: re.Match[str], signal: str) -> dict[str, Any] | None:
    """Turn one regex match into a candidate record with offsets and position."""
    start, end = match.start(), match.end()
    token = match.group("token")
    digits = re.sub(r"\D", "", token)
    if not digits or len(digits) > MAX_PAGE_NUMBER_DIGITS:
        return None
    # A bare four-digit number at a line edge is almost always a year in a date
    # header (``NOVEMBER 1 5, 2 0 0 0``), not a printed page number. A dashed
    # four-digit token is still accepted.
    if signal == "bare" and len(digits) >= MAX_PAGE_NUMBER_DIGITS:
        return None
    before = text[:start].strip()
    after = text[end:].strip()
    if before and after:
        position = "middle"
    elif after:
        position = "prefix"
    else:
        position = "suffix"
    return {
        "start": start,
        "end": end,
        "token": token.strip(),
        "page_number": int(digits),
        "position": position,
        "signal": signal,
        "before": before,
        "after": after,
    }


def inline_page_number_candidate(line: str) -> dict[str, Any] | None:
    """The first page number merged into a text line, with its exact offsets.

    Two shapes are considered: a dashed token anywhere in the line (``- 2 0
    "Well,``, ``... the - 6 8 -``, ``... a year. - 117 "Obviously,"``) and a bare
    number at one edge of the line (``... Valerie 265``). A line that is *only* a
    page number is not an inline candidate; the standalone detector handles it.
    """
    text = line.rstrip("\n")
    if not text.strip() or header_of(text):
        return None
    leading = len(text) - len(text.lstrip())
    trailing = len(text.rstrip())

    for match in INLINE_DASHED_PAGE_RE.finditer(text):
        before, after = text[: match.start()], text[match.end() :]
        if _only_page_decor(before) and _only_page_decor(after):
            continue
        candidate = _inline_candidate(text, match, "dashed")
        if candidate is not None:
            return candidate

    for match in INLINE_BARE_PAGE_RE.finditer(text):
        # A bare number is only interesting at a line edge; in the middle of a
        # sentence it is ordinary prose (``130 otkucaja``).
        if match.start() != leading and match.end() != trailing:
            continue
        before, after = text[: match.start()], text[match.end() :]
        if _only_page_decor(before) and _only_page_decor(after):
            continue
        candidate = _inline_candidate(text, match, "bare")
        if candidate is not None:
            return candidate
    return None


def repair_inline_page_number(line: str) -> str:
    """Drop the page-number token from a text line and keep every word.

    ``'- 2 0 "Well,'`` → ``'"Well,'``; ``'... the - 6 8 -'`` → ``'... the'``;
    ``'... a year. - 117 "Obviously,"'`` → ``'... a year. "Obviously,"'``. Only
    whitespace around the removed token is normalised; no word is touched.
    """
    candidate = inline_page_number_candidate(line)
    if candidate is None:
        return line
    text = line.rstrip("\n")
    before = text[: candidate["start"]].rstrip()
    after = text[candidate["end"] :].lstrip()
    if before and after:
        return f"{before} {after}"
    return before or after


def _previous_prose_line(lines: list[str], idx: int) -> tuple[int | None, str]:
    """The nearest non-blank, non-page-number line before ``idx``."""
    for i in range(idx - 1, -1, -1):
        text = lines[i].strip()
        if text and not is_page_number_line(text):
            return i + 1, text
    return None, ""


def _next_prose_line(lines: list[str], idx: int) -> tuple[int | None, str]:
    """The nearest non-blank, non-page-number line after ``idx``."""
    for i in range(idx + 1, len(lines)):
        text = lines[i].strip()
        if text and not is_page_number_line(text):
            return i + 1, text
    return None, ""


def _page_markers(lines: list[str]) -> list[tuple[int, int]]:
    """``(line_number, printed_page)`` for every standalone page-number line."""
    markers: list[tuple[int, int]] = []
    for idx, raw in enumerate(lines):
        if is_page_number_line(raw):
            page = page_number_of(raw)
            if page is not None:
                markers.append((idx + 1, page))
    return markers


def _surrounding_page_markers(
    markers: list[tuple[int, int]], line_number: int
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """The nearest printed page marker before and after ``line_number``."""
    before: tuple[int, int] | None = None
    after: tuple[int, int] | None = None
    for marker_line, page in markers:
        if marker_line < line_number:
            before = (marker_line, page)
        elif marker_line > line_number:
            after = (marker_line, page)
            break
    return before, after


def _fits_page_sequence(
    page: int,
    before: tuple[int, int] | None,
    after: tuple[int, int] | None,
) -> bool | None:
    """Whether ``page`` falls between the printed markers around it."""
    if before is not None and after is not None:
        if before[1] > after[1]:
            return None  # the sequence restarts; the bounds mean nothing
        return before[1] <= page <= after[1]
    if before is not None:
        # With only one bound, the page still has to be close to it; a bare
        # ``844`` after page 246 is a number in the text, not a page break.
        return before[1] <= page <= before[1] + PAGE_NUMBER_RUN_WINDOW
    if after is not None:
        return after[1] - PAGE_NUMBER_RUN_WINDOW <= page <= after[1]
    return None


def find_inline_page_numbers(lines: list[str]) -> list[Finding]:
    """Every page number merged into a text line, with words on both sides.

    Each finding carries the token, its position (``prefix`` / ``suffix`` /
    ``middle``), the surrounding text on the line, the nearest line before and
    after it, and — the strongest evidence — whether the number falls between
    the printed page markers around that line. That lets an agent tell a real
    page number from a year, a count or a dialogue dash without reading the
    chapter.
    """
    findings: list[Finding] = []
    markers = _page_markers(lines)
    for idx, raw in enumerate(lines):
        line = raw.rstrip("\n")
        if not line.strip() or header_of(line):
            continue
        if is_page_number_line(line) or is_caption_line(line):
            continue
        candidate = inline_page_number_candidate(line)
        if candidate is None:
            continue

        previous_number, previous_text = _previous_prose_line(lines, idx)
        next_number, next_text = _next_prose_line(lines, idx)
        before = excerpt(candidate["before"][-80:], 80)
        after = excerpt(candidate["after"][:120], 120)
        previous = excerpt(previous_text[-100:], 100)
        following = excerpt(next_text[:100], 100)
        token = candidate["token"]
        page = candidate["page_number"]
        position = candidate["position"]
        signal = candidate["signal"]
        marker_before, marker_after = _surrounding_page_markers(markers, idx + 1)
        fits = _fits_page_sequence(page, marker_before, marker_after)

        evidence = (
            f"{signal} page-number shape {token!r} merged into a text line "
            f"({position}); before={before or previous!r} "
            f"after={after or following!r}"
        )
        if marker_before is not None and marker_after is not None:
            evidence += (
                f"; printed pages nearby are {marker_before[1]} "
                f"(line {marker_before[0]}) and {marker_after[1]} "
                f"(line {marker_after[0]})"
            )
        elif marker_before is not None:
            evidence += f"; nearest printed page before is {marker_before[1]}"
        elif marker_after is not None:
            evidence += f"; nearest printed page after is {marker_after[1]}"
        else:
            evidence += "; no printed page markers nearby"
        if fits is True:
            evidence += f" - {page} fits the sequence"
        elif fits is False:
            evidence += f" - {page} does not fit the sequence"

        if fits is True:
            risk = "low" if signal == "dashed" else "medium"
            suggested = "remove"
        elif fits is False:
            risk = "medium" if signal == "dashed" else "high"
            suggested = "review"
        else:
            risk = "low" if signal == "dashed" else "medium"
            suggested = "remove" if signal == "dashed" else "review"

        findings.append(
            Finding(
                kind=INLINE_PAGE_NUMBER,
                line_number=idx + 1,
                risk=risk,
                before=before,
                after=after,
                evidence=evidence,
                extra={
                    "token": token,
                    "page_number": page,
                    "position": position,
                    "signal": signal,
                    "suggested": suggested,
                    "fits_page_sequence": fits,
                    "page_before": marker_before[1] if marker_before else None,
                    "page_before_line": marker_before[0] if marker_before else None,
                    "page_after": marker_after[1] if marker_after else None,
                    "page_after_line": marker_after[0] if marker_after else None,
                    "before_same_line": before,
                    "after_same_line": after,
                    "previous_line_number": previous_number,
                    "previous_text": previous,
                    "next_line_number": next_number,
                    "next_text": following,
                },
            )
        )
    return findings


def repair_space_before_punctuation(line: str) -> str:
    """``'mišić ,'`` → ``'mišić,'`` (never touches a leading indent)."""
    return re.sub(r"(?<=\S)[ \t]+([,.;:!?])", r"\1", line)


def repair_split_word(left: str, right: str) -> tuple[str, str]:
    """Join two lines whose last/first words are one word.

    Returns ``(left_without_newline, rest_of_right)``: the caller concatenates
    the two to get the repaired line. A single space is inserted when the
    fragments are not actually one word; when they are, they are glued.
    """
    left_body = left.rstrip("\n")
    right_body = right.lstrip("\n")
    left_stripped = left_body.rstrip()
    right_stripped = right_body.lstrip()
    if is_word_continuation(left_body, right_body):
        return left_stripped, right_stripped
    return f"{left_stripped} ", right_stripped


def scan_chapter_damage(
    path: str | Path,
    *,
    kinds: tuple[str, ...] | None = None,
    max_findings: int = 400,
) -> dict[str, Any]:
    """Every detectable body-text defect in one chapter.

    Only line numbers, short before/after fragments and the surrounding context
    of an inline page number are returned — never a paragraph. The output is
    meant to be read by an agent deciding which fixes to request, and by a
    validator checking the result.
    """
    p = Path(path).expanduser()
    lines = load_lines(p)
    wanted = set(kinds or DAMAGE_KINDS)
    findings: list[Finding] = []

    for idx, raw in enumerate(lines):
        line = raw.rstrip("\n")
        number = idx + 1
        if PAGE_NUMBER_IN_WORD in wanted:
            for hit in has_midword_number(line):
                digits = hit["digits"]
                repaired = repair_page_number_in_word(line)
                cut = hit["offset"] + len(digits)
                word = (describes_split_number(line[:cut], line[cut:]) or {}).get("word")
                findings.append(
                    Finding(
                        kind=PAGE_NUMBER_IN_WORD,
                        line_number=number,
                        before=excerpt(line[max(0, hit["offset"] - 18) : cut + 16], 56),
                        after=excerpt(
                            repaired[max(0, hit["offset"] - 18) : hit["offset"] + 16], 48
                        ),
                        evidence=(
                            f"digits {digits!r} printed between {hit['before']!r} "
                            f"and {hit['after']!r}; the word is likely "
                            f"{word!r}"
                        ),
                        extra={"offset": hit["offset"], "digits": digits, "word": word},
                    )
                )
        if SPACE_BEFORE_PUNCTUATION in wanted:
            count = count_space_before_punct(line)
            if count:
                match = re.search(r"\s+([,.;:!?])", line)
                if match is not None:
                    snippet = excerpt(line[max(0, match.start() - 18) : match.end() + 12], 48)
                    findings.append(
                        Finding(
                            kind=SPACE_BEFORE_PUNCTUATION,
                            line_number=number,
                            before=snippet,
                            after=excerpt(repair_space_before_punctuation(snippet), 48),
                            evidence=f"{count} punctuation mark(s) preceded by whitespace",
                            extra={"count": count},
                        )
                    )

    if INLINE_PAGE_NUMBER in wanted:
        findings.extend(find_inline_page_numbers(lines))

    if SPLIT_WORD in wanted:
        for idx in range(len(lines) - 1):
            left, right = lines[idx], lines[idx + 1]
            if not left.strip() or not right.strip():
                continue
            if is_word_continuation(left, right):
                joined, _ = repair_split_word(left, right)
                findings.append(
                    Finding(
                        kind=SPLIT_WORD,
                        line_number=idx + 1,
                        before=excerpt(left[-40:].rstrip(), 40),
                        after=excerpt(right.lstrip()[:40], 40),
                        evidence=(
                            "line ends in an unfinished lowercase word and the next "
                            f"line continues it ({excerpt(joined[-24:] + right.lstrip()[:16], 44)!r})"
                        ),
                        extra={"next_line_number": idx + 2},
                    )
                )

    return {
        "chapter": {"filename": p.name, "path": str(p), "line_count": len(lines)},
        "finding_count": len(findings),
        "truncated": len(findings) > max_findings,
        "by_kind": _counts(findings),
        "findings": [f.to_dict() for f in findings[:max_findings]],
    }


def _counts(findings: list[Finding]) -> dict[str, int]:
    out: dict[str, int] = {}
    for f in findings:
        out[f.kind] = out.get(f.kind, 0) + 1
    return out


def damage_digest(report: dict[str, Any], max_items: int = 25) -> str:
    """Compact text rendering of a damage report for a prompt.

    Accepts both a per-chapter report (``by_kind``, ``findings``) and a
    book-wide report (``counts``, ``examples``, ``patterns``).
    """
    chapter = report.get("chapter", {})
    by_kind = report.get("by_kind") or report.get("counts") or {}
    header = (
        f"DAMAGE REPORT {chapter.get('filename')}: {report.get('finding_count', 0)} findings "
        f"in {chapter.get('line_count', 0)} lines {by_kind}"
        if chapter
        else (
            f"CORPUS DAMAGE REPORT: {report.get('chapter_count', 0)} chapters, "
            f"{report.get('total_lines', 0)} lines, {report.get('total_findings', 0)} findings "
            f"{by_kind}"
        )
    )
    out = [header]
    out.extend(
        f"  {item['kind']:<24} line {item['line_number']:>4}: {item['evidence']}"
        for item in report.get("findings", [])[:max_items]
    )
    for name, items in (report.get("examples") or {}).items():
        out.extend(
            f"  {name:<24} {item['chapter']}:{item['line_number']} {item['text']!r}"
            for item in items[:max_items]
        )
    return "\n".join(out)


__all__ = [
    "DAMAGE_KINDS",
    "FIXES",
    "Finding",
    "INLINE_PAGE_NUMBER",
    "PAGE_NUMBER_IN_WORD",
    "SPACE_BEFORE_PUNCTUATION",
    "SPLIT_WORD",
    "damage_digest",
    "find_inline_page_numbers",
    "inline_page_number_candidate",
    "repair_inline_page_number",
    "repair_page_number_in_word",
    "repair_space_before_punctuation",
    "repair_split_word",
    "scan_chapter_damage",
]
