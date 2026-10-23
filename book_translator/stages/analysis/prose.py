"""Chapter openings whose heading was lost: prose split points.

A PDF→Markdown conversion can drop a chapter heading *completely* — no empty
``##``, no double blank line, no page number survives. The chapter then starts on
a plain paragraph, and a map built only from headers silently merges it into the
previous chapter. The published contents list still says how many chapters the
book has, so the splitter has to be able to *say* that a given prose line opens a
chapter.

That is what ``role: "prose-chapter"`` is for. It is the one chapter role that
need not look like a title, so it is guarded by shape instead:

* the line must be the first line of a paragraph (the line above it is blank),
  never a line in the middle of one;
* it must be a line of text — not blank, not a lone page number, and not so
  short that it cannot be a chapter opening;
* the entry must carry ``evidence``: which published contents list (or which
  surviving headings and what they imply) says this line opens a chapter. That
  claim is not machine-checkable, so it is quoted into the validation result and
  the trace rather than trusted silently.

Only predicates live here: the agent (or a ``--map`` file) decides, the slicer
writes.
"""

from __future__ import annotations

from typing import Any

from book_translator.documents.markdown import is_page_number_line

#: The role a split point carries when it opens a chapter whose heading is gone.
PROSE_CHAPTER_ROLE = "prose-chapter"

#: Accepted spellings of that role in a submitted map.
PROSE_CHAPTER_ROLES: frozenset[str] = frozenset(
    {PROSE_CHAPTER_ROLE, "prose_chapter", "prose chapter", "prose"}
)

#: The shortest line that can plausibly open a chapter.
MIN_PROSE_START_CHARS = 12

#: …and the fewest words it may hold.
MIN_PROSE_START_WORDS = 2

#: The shortest acceptable justification for a prose chapter start.
MIN_PROSE_EVIDENCE_CHARS = 12

#: Map fields that may carry that justification, in the order they are read.
EVIDENCE_FIELDS: tuple[str, ...] = (
    "evidence",
    "reason",
    "justification",
    "why",
    "source",
    "provenance",
)


def is_prose_chapter_role(value: Any) -> bool:
    """True when a map entry declares itself as a heading-lost chapter start."""
    return str(value or "").strip().lower() in PROSE_CHAPTER_ROLES


def is_paragraph_start(lines: list[str], idx0: int) -> bool:
    """True when the line at ``idx0`` opens a paragraph (nothing above it)."""
    if not 0 <= idx0 < len(lines):
        return False
    if not lines[idx0].strip():
        return False
    return idx0 == 0 or not lines[idx0 - 1].strip()


def prose_start_issue(lines: list[str], idx0: int) -> str | None:
    """Why the line at ``idx0`` cannot open a chapter, as a short code, or ``None``.

    The codes (``blank``, ``out-of-range``, ``mid-paragraph``, ``page-number``,
    ``too-short``) are what the search tool reports per match: a whole sentence
    per match would eat the tool-result budget in a long search.
    """
    if not 0 <= idx0 < len(lines):
        return "out-of-range"
    text = lines[idx0].strip()
    if not text:
        return "blank"
    if not is_paragraph_start(lines, idx0):
        return "mid-paragraph"
    if is_page_number_line(text):
        return "page-number"
    if len(text.split()) < MIN_PROSE_START_WORDS or len(text) < MIN_PROSE_START_CHARS:
        return "too-short"
    return None


def prose_start_problem(lines: list[str], idx0: int) -> str | None:
    """Why the line at ``idx0`` cannot be a ``role=prose-chapter`` split, or ``None``.

    The returned sentence completes ``line N is marked role=prose-chapter but …``,
    so it is written for the model that has to fix its map.
    """
    issue = prose_start_issue(lines, idx0)
    if issue is None:
        return None
    text = lines[idx0].strip() if 0 <= idx0 < len(lines) else ""
    if issue == "out-of-range":
        return "the line number is outside the file"
    if issue == "blank":
        return "it is blank; a chapter starts on a line of text"
    if issue == "mid-paragraph":
        return (
            "it does not start a paragraph (the line above it is not blank), so it is "
            "somewhere inside one"
        )
    if issue == "page-number":
        return "it is only a page number"
    words = len(text.split())
    return (
        f"it is too short to open a chapter ({words} word(s), {len(text)} characters; "
        f"at least {MIN_PROSE_START_WORDS} words and {MIN_PROSE_START_CHARS} characters "
        "are required)"
    )


def entry_evidence(entry: dict[str, Any]) -> str:
    """The justification a map entry carries, from whichever field holds it."""
    for field in EVIDENCE_FIELDS:
        value = str(entry.get(field) or "").strip()
        if value:
            return value
    return ""


def prose_evidence_problem(evidence: str) -> str | None:
    """Why a prose chapter claim is not justified well enough, or ``None``."""
    text = str(evidence or "").strip()
    if len(text) < MIN_PROSE_EVIDENCE_CHARS:
        return (
            "it carries no usable `evidence`. Say which published contents list (or which "
            "surviving headings, and what they imply) makes this line a chapter opening, "
            'for example: "publisher contents list: chapter 5 <title>, missing `## 5.`".'
        )
    return None


def looks_like_prose_start(lines: list[str], idx0: int) -> bool:
    """True when the line could be used as a prose chapter start as it stands."""
    return prose_start_problem(lines, idx0) is None


__all__ = [
    "EVIDENCE_FIELDS",
    "MIN_PROSE_EVIDENCE_CHARS",
    "MIN_PROSE_START_CHARS",
    "MIN_PROSE_START_WORDS",
    "PROSE_CHAPTER_ROLE",
    "PROSE_CHAPTER_ROLES",
    "entry_evidence",
    "is_paragraph_start",
    "is_prose_chapter_role",
    "looks_like_prose_start",
    "prose_evidence_problem",
    "prose_start_issue",
    "prose_start_problem",
]
