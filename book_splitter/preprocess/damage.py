"""Body-text damage detection and repair (stage 2).

The book this pipeline was built for came out of a PDF converter with three
kinds of damage that are *shape*-recognisable, and one repair for each:

``page_number_in_word``
    A printed page number landed inside a word, because the word was split
    across a page break: ``pažljivu psi`` / ``182 ljutnje``. Repair: join the
    fragments and drop the number, so the word becomes ``psiljutnje``.
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

from ..documents.bodyline import (
    count_space_before_punct,
    describes_split_number,
    has_midword_number,
    is_word_continuation,
)
from ..documents.io import load_lines
from ..documents.text import excerpt

#: A finding kind (also the ``fix`` name an agent may request).
PAGE_NUMBER_IN_WORD = "page_number_in_word"
SPLIT_WORD = "split_word"
SPACE_BEFORE_PUNCTUATION = "space_before_punctuation"

#: Digits glued to the previous word (``us12 merena``): keep the fragments glued.
MIDWORD_RE = re.compile(r"(?<=[a-zžćčšđ])(\d{1,4})(?=\s+[a-zžćčšđ])")
#: Digits glued to the *front* of a word (``1otkucaja``): delete the run.
LEADING_GLUED_RE = re.compile(r"^\d{1,4}(?=[a-zžćčšđ])")
#: Digits printed in front of a lowercase word (``30 spreč avaju``): delete the run
#: and the space, so the two fragments of the split word are joined.
LEADING_SPACED_RE = re.compile(r"^(\d{1,4})\s+(?=[a-zžćčšđ])")

#: Canonical repairs: what each finding kind does, and how safe it is.
FIXES: dict[str, dict[str, str]] = {
    PAGE_NUMBER_IN_WORD: {
        "summary": "drop a printed page number printed inside a word and close the word",
        "risk": "low",
        "example": "'pažljivu psi' + '182 ljutnje' -> 'pažljivu psiljutnje'",
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

    Only line numbers, the two fragments and a reconstructed word are returned —
    never a paragraph. The output is meant to be read by an agent deciding which
    fixes to request, and by a validator checking the result.
    """
    p = Path(path).expanduser()
    lines = load_lines(p)
    wanted = set(kinds or (PAGE_NUMBER_IN_WORD, SPLIT_WORD, SPACE_BEFORE_PUNCTUATION))
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
    for item in report.get("findings", [])[:max_items]:
        out.append(f"  {item['kind']:<24} line {item['line_number']:>4}: {item['evidence']}")
    for name, items in (report.get("examples") or {}).items():
        for item in items[:max_items]:
            out.append(
                f"  {name:<24} {item['chapter']}:{item['line_number']} {item['text']!r}"
            )
    return "\n".join(out)


__all__ = [
    "FIXES",
    "Finding",
    "PAGE_NUMBER_IN_WORD",
    "SPACE_BEFORE_PUNCTUATION",
    "SPLIT_WORD",
    "damage_digest",
    "repair_page_number_in_word",
    "repair_space_before_punctuation",
    "repair_split_word",
    "scan_chapter_damage",
]
