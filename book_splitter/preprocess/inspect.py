"""Header inspection for stage 2 (preprocessing).

The unit of work in stage 2 is *one chapter's headers*, never its prose. This
module turns one chapter file into a compact, content-free picture of every
header it contains:

* which line it sits on and what it says (capped),
* the first token of the next non-blank line,
* a short set of *evidence flags* (page number, caption marker, unfinished
  fragment, following lowercase continuation, sentence-ending punctuation),
* and a deterministic **recommendation** derived from those flags alone.

The recommendation is deliberately conservative and never destructive: it says
"keep", "demote to a paragraph" or "remove as a page artefact". Merging two lines
requires both sides to look like one split heading. When the evidence is thin the
recommendation is a suggestion only — the agent decides, and a validator checks
whatever it decides against the same evidence.

The important property: the cost of reviewing a chapter is proportional to its
*header count*, not to its length. A 3000-line chapter with 6 headers costs 6
records (a few hundred characters).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.constants import (
    MAX_MERGE_HEADING_CHARS,
    MAX_PLAUSIBLE_HEADING_CHARS,
    MIN_BROKEN_TAIL_LINE_CHARS,
)
from ..documents.bodyline import (
    ends_sentence,
    ends_with_broken_word,
    is_blank,
    is_caption_line,
    starts_with_lowercase,
)
from ..documents.io import load_lines
from ..documents.markdown import (
    header_body,
    header_of,
    is_all_caps,
    is_page_number_line,
    page_number_of,
)
from ..documents.text import excerpt

#: Words that start a caption rather than a heading.
CAPTION_MARKERS = (
    "slika", "slika.", "tabela", "tabla", "grafikon", "dijagram", "figure",
    "fig.", "fig", "table", "chart", "photo", "fotografija", "crtež", "crt.",
)
#: Tokens that make a heading a real structural title (never a page number).
STRUCTURAL_TOKENS = (
    "glava", "giava", "poglavlje", "chapter", "deo", "dio", "part", "uvod",
    "predgovor", "pogovor", "beleške", "belestke", "belješke", "sadržaj",
    "dodatak", "appendix", "index", "indeks", "epilog", "prolog", "zaključak",
)

#: Decision vocabulary, shared with the validator and the applier.
KEEP = "keep"
DEMOTE = "demote"
REMOVE = "remove"
MERGE_WITH_NEXT = "merge_with_next"
#: The second half of a split heading: merge (it is referenced by its partner),
#: but it is not itself the heading — the merged line is.
MERGE_DEMOTE = "merge_demote"
#: A figure/table caption: real content, but not part of the book's text flow.
REMOVE_CAPTION = "remove_caption"

DECISIONS = (KEEP, DEMOTE, REMOVE, REMOVE_CAPTION, MERGE_WITH_NEXT, MERGE_DEMOTE)
#: Decisions that delete a line instead of rewriting it.
DESTRUCTIVE_DECISIONS = (REMOVE, REMOVE_CAPTION, MERGE_WITH_NEXT)


@dataclass
class HeaderRecord:
    """Everything the pipeline knows about one header line in one chapter."""

    line_number: int
    level: int
    text: str
    preview: str = ""
    is_page_number: bool = False
    page_number: int | None = None
    has_structural_token: bool = False
    has_caption_marker: bool = False
    is_short: bool = True
    is_all_caps: bool = False
    ends_sentence: bool = False
    next_is_header: bool = False
    next_line_number: int | None = None
    next_text: str = ""
    next_starts_lowercase: bool = False
    next_is_continuation: bool = False
    next_is_long_prose: bool = False
    next_is_short_title: bool = False
    next_is_page_number: bool = False
    title_follows_label: bool = False
    next_is_blank: bool = False
    previous_is_blank: bool = True
    evidence: list[str] = field(default_factory=list)
    recommendation: str = KEEP
    risk: str = "low"

    def to_dict(self) -> dict:
        return asdict(self)


def chapter_header_lines(lines: list[str]) -> list[int]:
    """1-indexed line numbers of every Markdown ATX header in the chapter."""
    return [i for i, line in enumerate(lines, start=1) if header_of(line)]


def title_follows_label(text: str) -> bool:
    """True when a structural heading already carries its own title.

    ``"GLAVA 2"`` and ``"PRVI DEO"`` are labels whose title is expected on the
    next line; ``"GLAVA 5 STRAH OD LJUBAVI"`` and ``"BELEŠKE"`` are complete
    headings. The distinction decides whether a following short header is the
    rest of this heading or a new heading of its own.
    """
    tokens = text.split()
    if not tokens:
        return False
    head = tokens[0].strip(".:-–—").lower()
    rest = tokens[1:]
    if head in {"deo", "dio", "part", "glava", "giava", "poglavlje", "chapter"}:
        # Drop the number that belongs to the label: ``GLAVA 2``, ``DEO 1``.
        if rest and rest[0].strip(".:-–—").isdigit():
            rest = rest[1:]
        if not rest:
            return False  # a bare label: its title comes on the next line
    return True


def _next_nonblank(lines: list[str], idx0: int) -> int | None:
    """Index of the next non-blank line after ``idx0``, or ``None``."""
    for i in range(idx0 + 1, len(lines)):
        if not is_blank(lines[i]):
            return i
    return None


def _prev_is_blank(lines: list[str], idx0: int) -> bool:
    """True when the line before ``idx0`` is blank or the file starts there."""
    return idx0 == 0 or is_blank(lines[idx0 - 1])


def _next_nonblank_text(lines: list[str], idx0: int) -> tuple[int | None, str]:
    nxt = _next_nonblank(lines, idx0)
    return (nxt, lines[nxt]) if nxt is not None else (None, "")


def looks_like_page_number(text: str, level: int) -> tuple[bool, int | None]:
    """True when a header body is only a printed page number.

    ``## 18``, ``## 790``, ``18`` all qualify. A number inside a word
    (``## 2. POGLAVLJE``) does not.
    """
    if is_page_number_line(text):
        return True, page_number_of(text)
    if is_page_number_line(f"# {text}"):
        return True, page_number_of(text)
    return False, None


def classify_header(
    lines: list[str],
    line_number: int,
    *,
    next_header_line: int | None = None,
    merge_partner: int | None = None,
) -> HeaderRecord:
    """Build the evidence record for one header line.

    ``next_header_line`` is the next header in the chapter (if any), which lets
    the classifier distinguish "heading followed by body text" from "heading
    followed by a second half of the same heading".

    ``merge_partner`` is the *previous* header's line number when that header
    already claimed this one as its other half (``## GLAVA 2`` then
    ``## SEKS I SRCE``). Such a line is not an independent heading any more.
    """
    idx0 = line_number - 1
    level, text = header_of(lines[idx0]) or (0, "")
    record = HeaderRecord(line_number=line_number, level=level, text=text)
    record.preview = excerpt(text, 110)

    is_pnum, pnum = looks_like_page_number(text, level)
    record.is_page_number = is_pnum
    record.page_number = pnum
    record.is_short = len(text) <= MAX_PLAUSIBLE_HEADING_CHARS
    record.is_all_caps = is_all_caps(text)
    record.title_follows_label = title_follows_label(text)
    record.ends_sentence = ends_sentence(text)
    record.previous_is_blank = _prev_is_blank(lines, idx0)

    lowered = text.lower()
    record.has_structural_token = any(tok in lowered for tok in STRUCTURAL_TOKENS)
    record.has_caption_marker = lowered.startswith(CAPTION_MARKERS)

    nxt, nxt_raw = _next_nonblank_text(lines, idx0)
    record.next_line_number = (nxt + 1) if nxt is not None else None
    record.next_text = excerpt(nxt_raw, 110)
    record.next_is_blank = nxt is None
    if nxt is not None:
        record.next_is_header = header_of(nxt_raw) is not None
        record.next_starts_lowercase = starts_with_lowercase(nxt_raw)
        record.next_is_continuation = ends_with_broken_word(lines[idx0]) and record.next_starts_lowercase
        record.next_is_long_prose = len(nxt_raw.strip()) > MIN_BROKEN_TAIL_LINE_CHARS
        record.next_is_short_title = (
            record.next_is_header
            and len(header_body(nxt_raw)) <= MAX_MERGE_HEADING_CHARS
        )
        next_is_page, _ = looks_like_page_number(header_body(nxt_raw), 2)
        record.next_is_page_number = next_is_page

    record.evidence = _evidence(record)
    record.recommendation, record.risk = _recommend(record, merge_partner)
    return record


def _evidence(record: HeaderRecord) -> list[str]:
    """Human- and machine-readable reasons behind a recommendation."""
    flags: list[str] = []
    if record.is_page_number:
        flags.append(f"header body is only a number ({record.page_number}) - printed page number")
    if record.has_caption_marker:
        flags.append("starts with a figure/table caption marker")
    if record.ends_sentence:
        flags.append("ends with sentence punctuation - reads as a sentence, not a heading")
    if record.next_is_continuation:
        flags.append(
            "previous header ends in an unfinished word and the next line starts "
            "lowercase - the heading was split across two lines"
        )
    if record.next_starts_lowercase and not record.next_is_continuation:
        flags.append("next line starts lowercase - may be the rest of this heading")
    if record.next_is_header and record.next_is_short_title:
        flags.append("next line is another short header - may be the rest of this heading")
    if record.next_is_long_prose:
        flags.append("next non-blank line is long prose")
    if record.is_short is False:
        flags.append(f"longer than {MAX_PLAUSIBLE_HEADING_CHARS} characters")
    if record.next_is_blank:
        flags.append("nothing but blank lines follow")
    if record.is_all_caps and not flags:
        flags.append("short ALL-CAPS line - typical of a real heading")
    return flags


def _recommend(record: HeaderRecord, merge_partner: int | None = None) -> tuple[str, str]:
    """Deterministic suggestion + how risky it is to follow it blindly."""
    if record.is_page_number:
        return REMOVE, "very_low"

    # A caption is a label for a picture that is not in the Markdown at all:
    # "## Slika 7. Protok ..." is deleted, never kept as a paragraph.
    if record.has_caption_marker:
        return REMOVE_CAPTION, "low"

    # This line is the second half of a heading the previous header claimed.
    if merge_partner is not None:
        return MERGE_DEMOTE, "medium"

    # Split heading: "## GLAVA 2" + "## SEKS I SRCE", or an unfinished fragment
    # continued by the next header ("## Volja za životom i želja za smrć" + "## u").
    if record.next_is_continuation and record.next_is_header:
        return MERGE_WITH_NEXT, "medium"
    if (
        record.next_is_header
        and record.next_is_short_title
        and not record.next_is_page_number
        and record.level >= 2
        and record.has_structural_token
        and not record.title_follows_label
    ):
        # ``## GLAVA 2`` + ``## SEKS I SRCE`` is one heading. ``## GLAVA 1 LJUBAV
        # JE U SRCU ŽIVOTA`` already carries its title, so its next header is a
        # new heading rather than its other half.
        return MERGE_WITH_NEXT, "medium"

    # Body text that a page break promoted to a header.
    if not record.is_short:
        return DEMOTE, "medium"
    if record.ends_sentence and not record.has_structural_token and not record.is_all_caps:
        return DEMOTE, "medium"
    if len(record.text) <= 1 and not record.has_structural_token:
        return DEMOTE, "medium"
    if (
        record.next_is_long_prose
        and not record.has_structural_token
        and not record.is_all_caps
    ):
        return DEMOTE, "medium"

    return KEEP, "low"


def inspect_body_candidates(
    lines: list[str],
    header_lines: set[int],
    max_candidates: int = 200,
) -> list[dict[str, Any]]:
    """Page numbers and captions that the converter left as *plain* body lines.

    Some artefacts never got a ``#``: ``19`` sits alone between two paragraphs
    and ``Slika 4. Blokada ljubavi. ...`` is a caption without markup. They are
    found by shape (a line that is only digits; a line that opens with a caption
    marker and a number), so the agent can decide about them without reading any
    prose.
    """
    found: list[dict[str, Any]] = []
    for idx, raw in enumerate(lines, start=1):
        if idx in header_lines:
            continue
        text = raw.strip()
        if not text:
            continue
        if is_page_number_line(text):
            found.append(
                {
                    "line_number": idx,
                    "kind": "page_number",
                    "suggested": REMOVE,
                    "risk": "very_low",
                    "text": excerpt(text, 110),
                    "evidence": ["body line is only a number - printed page number"],
                }
            )
        elif is_caption_line(text):
            found.append(
                {
                    "line_number": idx,
                    "kind": "caption",
                    "suggested": REMOVE_CAPTION,
                    "risk": "low",
                    "text": excerpt(text, 110),
                    "evidence": [
                        "opens with a figure/table caption marker and a number",
                        "the picture itself is not in the Markdown",
                    ],
                }
            )
        if len(found) >= max_candidates:
            break
    return found


def inspect_chapter(
    path: str | Path,
    max_headers: int = 300,
    *,
    include_all: bool = False,
) -> dict[str, Any]:
    """Header-level picture of one chapter, plus a compact review list.

    ``headers`` always carries every header record (capped by ``max_headers``);
    ``body_candidates`` carries page numbers and captions that were left as plain
    body lines; ``review`` is the union of both that deserves a decision, which is
    what an agent should be shown.
    """
    p = Path(path).expanduser()
    lines = load_lines(p)
    numbers = chapter_header_lines(lines)
    records: list[HeaderRecord] = []
    merge_partner: int | None = None
    for pos, line_number in enumerate(numbers):
        next_header = numbers[pos + 1] if pos + 1 < len(numbers) else None
        record = classify_header(
            lines, line_number, next_header_line=next_header, merge_partner=merge_partner
        )
        merge_partner = line_number if record.recommendation == MERGE_WITH_NEXT else None
        records.append(record)

    counts: dict[str, int] = {}
    for record in records:
        counts[record.recommendation] = counts.get(record.recommendation, 0) + 1

    body_candidates = inspect_body_candidates(lines, set(numbers))
    review = [
        {
            "line_number": r.line_number,
            "text": r.preview,
            "suggested": r.recommendation,
            "risk": r.risk,
            "evidence": r.evidence,
            "next_line_number": r.next_line_number,
            "next_text": r.next_text,
        }
        for r in records
        if include_all or r.recommendation != KEEP
    ]
    review.extend(
        {
            "line_number": c["line_number"],
            "text": c["text"],
            "suggested": c["suggested"],
            "risk": c["risk"],
            "evidence": c["evidence"],
            "next_line_number": None,
            "next_text": "",
            "kind": c["kind"],
        }
        for c in body_candidates
    )
    review.sort(key=lambda item: item["line_number"])
    for c in body_candidates:
        counts[c["suggested"]] = counts.get(c["suggested"], 0) + 1

    return {
        "chapter": {
            "filename": p.name,
            "path": str(p),
            "line_count": len(lines),
            "bytes": p.stat().st_size,
        },
        "header_count": len(records),
        "truncated": len(records) > max_headers,
        "recommendation_counts": counts,
        "review_count": len(review),
        "body_candidates": body_candidates,
        "body_candidate_count": len(body_candidates),
        "review": review[:max_headers],
        "headers": [r.to_dict() for r in records[:max_headers]],
        "line_lengths": _line_length_profile(lines),
    }


def _line_length_profile(lines: list[str]) -> dict[str, Any]:
    """A tiny profile that proves the caller is looking at a real chapter."""
    lengths = [len(line.rstrip("\n")) for line in lines]
    non_blank = [n for n in lengths if n]
    if not non_blank:
        return {"blank_lines": len(lines), "max": 0, "mean": 0}
    return {
        "blank_lines": len(lines) - len(non_blank),
        "max": max(non_blank),
        "mean": round(sum(non_blank) / len(non_blank)),
        "over_2000_chars": sum(1 for n in non_blank if n > 2000),
    }


def chapter_digest(record: dict[str, Any]) -> str:
    """A compact text digest of an :func:`inspect_chapter` result.

    This is what goes into the prompt: one line per header that needs a decision,
    with its line number, the suggested action and the first piece of evidence.
    """
    chapter = record.get("chapter", {})
    out = [
        f"HEADER REPORT {chapter.get('filename')}: {record.get('header_count', 0)} headers "
        f"in {chapter.get('line_count', 0)} lines; "
        f"{record.get('review_count', 0)} need a decision."
    ]
    for item in record.get("review", []):
        first = (item.get("evidence") or [""])[0]
        out.append(
            f"  line {item['line_number']:>4} [{item['suggested']}/{item['risk']}] "
            f"{item['text']!r} — {first}"
        )
    return "\n".join(out)


__all__ = [
    "DECISIONS",
    "DEMOTE",
    "DESTRUCTIVE_DECISIONS",
    "HeaderRecord",
    "KEEP",
    "MERGE_DEMOTE",
    "MERGE_WITH_NEXT",
    "REMOVE",
    "REMOVE_CAPTION",
    "chapter_digest",
    "chapter_header_lines",
    "classify_header",
    "inspect_body_candidates",
    "inspect_chapter",
    "looks_like_page_number",
]
