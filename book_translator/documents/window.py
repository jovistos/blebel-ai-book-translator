"""Chapter windows: "only the first N chapters" across the whole pipeline.

Chapter splitting is the one stage that always runs over the whole book: the
chapter boundaries are the coordinate system every later stage reads, and a
partial split would make the rest of the pipeline ambiguous. Every stage *after*
splitting can be narrowed to a prefix of those chapters, and this module is the
single place that turns ``N`` into the unit count each stage understands.

``N`` counts **body chapters**, not loose book files. A split book usually starts
with ``00-front-matter.md``, and a publisher's introduction or note on the text
may be written as a numbered file even though it is front matter. Neither consumes
one of the N, so ``--first-chapters 5`` means chapters 1–5, not "front matter plus
chapters 1–4". Stage 1's explicit front/back ``kind`` is honoured when it is
present; otherwise the shared filename rule (``00-*``, ``01-introduction``,
``03-preface``, ``back-matter``) decides.

* **Chapter-based stages** (dictionary, preprocess, chunk, glossary, scan) take a
  ``limit`` that counts ``discover_chapters`` entries. The window is turned into
  the *file* count that covers the first N body chapters, so any front matter
  before chapter 1 is still included in the folder walk but does not use up a
  chapter.
* **Chunk-based stages** (chunk-glossary, translate) take a *chunk* limit. The
  window is translated by reading stage 5's manifest, taking the first N body
  chapter names in reading order, and counting the chunks that belong to them.
  That keeps the chunk prefix aligned with the chapter prefix, so a translation
  of the first N chapters is exactly the chunks of those chapters.
* **Assembly** (stage 8) applies the same body-chapter selection to the chapter
  names, so ``08-translated-chapters/`` holds exactly the chapters that were
  translated when a window is set.

The helpers are deliberately conservative: when a chunk manifest has lost its
chapter membership the chunk-based stages cannot honour a chapter window, and an
actionable :class:`~book_translator.core.errors.ToolError` is better than silently
translating the wrong prefix.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..core.constants import FRONT_MATTER_FILENAME
from ..core.errors import ToolError
from .chapters import Chapter
from .chunks import chunk_chapter_names, chunk_files

#: A limit value below one means "no limit", which is how the stage runners have
#: always read ``0`` (and how ``--limit 0`` has behaved).
Limit = int | None

#: The same matter labels stage 1 writes into its manifest, plus the filename
#: fallbacks used by the glossary stage. Kept here so every stage treats a
#: front-matter block the same way.
_FRONT_MATTER_KINDS = frozenset({"front-matter", "frontmatter", "front matter", "front_matter"})
_BACK_MATTER_KINDS = frozenset({"back-matter", "backmatter", "back matter", "back_matter"})
_BACK_MATTER_MARKERS = ("back-matter", "back_matter", "backmatter")

#: A numbered *front/back* section in an otherwise chapter-numbered folder:
#: ``01-introduction.md``, ``03-preface-to-the-first-edition.md``. Stage 1 only
#: writes one front-matter block (``00-front-matter.md``) and one optional back
#: block, so a publisher's introduction or a note on the text is written as a
#: numbered file with ``kind: chapter``. The filename is the evidence every
#: later stage can see (a translated folder no longer carries stage 1's
#: ``kind``), and the same rules are used by stage 6b and stage 9.
_FRONT_NAME_RE = re.compile(
    r"^(?:\d*[-_ ]*)?(?:predgovor|uvod|prolog|preface|foreword|introduction|"
    r"zahvaln|acknowledg|(?:a[-_ ])?note[-_ ]on[-_ ]the[-_ ]text)\b",
    re.IGNORECASE,
)
_BACK_NAME_RE = re.compile(
    r"^(?:\d*[-_ ]*)?(?:pogovor|epilog|bele[sš]ke|beleske|notes|dodatak|appendix|"
    r"o[-_ ]autoru|about[-_ ]the[-_ ]author|index|indeks)\b",
    re.IGNORECASE,
)


def matter_kind_of_name(name: str) -> str:
    """``front`` / ``back`` / ``body`` from a chapter *filename* alone.

    This is the shared name rule: the chapter window, the matter stage and the
    postprocessing stage must agree on whether a numbered file is a real chapter
    or a publisher's section. ``00-*`` is stage 1's front matter, the back
    markers are stage 1's optional back block, and a numbered preface,
    introduction or note on the text is front matter even though the slicer
    labels it ``kind: chapter``.
    """
    filename = Path(str(name or "")).name.lower()
    if not filename:
        return "body"
    if filename == FRONT_MATTER_FILENAME.lower() or "front-matter" in filename:
        return "front"
    if filename.startswith("00-"):
        return "front"
    if any(marker in filename for marker in _BACK_MATTER_MARKERS):
        return "back"
    stem = Path(filename).stem
    if _FRONT_NAME_RE.match(stem):
        return "front"
    if _BACK_NAME_RE.match(stem):
        return "back"
    return "body"


def normalize_limit(value: Any) -> Limit:
    """``None`` / ``0`` / a negative number -> ``None``; a positive int -> itself."""
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def combine_limits(*values: Any) -> Limit:
    """The tightest positive limit among ``values``; ``None`` when none is set.

    Used when a run has both a chapter window and an older unit limit: the
    smaller one wins, so ``--first-chapters 5 --limit 3`` processes three units,
    never five.
    """
    limits = [
        normalized
        for normalized in (normalize_limit(value) for value in values)
        if normalized
    ]
    return min(limits) if limits else None


def is_body_chapter_name(name: str) -> bool:
    """Whether a chapter *file* is a real chapter, not front/back matter."""
    return bool(str(name or "").strip()) and matter_kind_of_name(name) == "body"


def is_body_chapter(chapter: Chapter) -> bool:
    """Whether a discovered chapter is a real chapter, using stage 1's ``kind``."""
    kind = str(chapter.extras.get("kind") or "").strip().lower()
    if kind in _FRONT_MATTER_KINDS or kind in _BACK_MATTER_KINDS:
        return False
    return is_body_chapter_name(chapter.filename)


def body_chapter_names(names: Iterable[str]) -> list[str]:
    """The names that are real chapters, preserving reading order."""
    return [str(name) for name in names if is_body_chapter_name(str(name))]


def chapter_window_limit(
    chapters: Iterable[Chapter],
    first_chapters: Any,
    unit_limit: Any = None,
) -> Limit:
    """The file-count limit that covers the first N body chapters.

    ``chapter_window_limit([front, ch1, ch2, ch3], 2)`` returns ``3``: the folder
    walk must include three files so that two *body* chapters are covered. When
    N is not set, the older ``unit_limit`` is returned unchanged. When the book
    has fewer than N body chapters, every file is included.
    """
    ordered = list(chapters)
    first = normalize_limit(first_chapters)
    unit = normalize_limit(unit_limit)
    if first is None:
        return unit
    if not ordered:
        # Before the split has written the folder, the caller still gets a sane
        # number; it will be recomputed once the chapter files exist.
        return combine_limits(first, unit)

    seen = 0
    for index, chapter in enumerate(ordered, start=1):
        if not is_body_chapter(chapter):
            continue
        seen += 1
        if seen >= first:
            return index if unit is None else min(index, unit)
    return len(ordered) if unit is None else min(len(ordered), unit)


def chapter_window_limit_for_path(
    chapters_path: str | Path,
    first_chapters: Any,
    unit_limit: Any = None,
) -> Limit:
    """``chapter_window_limit`` over a chapters folder (or book folder).

    A missing folder is not an error here: the pipeline calls this before the
    split has run, and the raw N is the only sensible fallback. It is recomputed
    after the split writes the chapter files.
    """
    first = normalize_limit(first_chapters)
    if first is None:
        return normalize_limit(unit_limit)
    from .chapters import discover_chapters

    try:
        chapters = discover_chapters(chapters_path)
    except (ToolError, OSError):
        return combine_limits(first, unit_limit)
    return chapter_window_limit(chapters, first, unit_limit)


def chapter_files_for_window(
    chunks_path: str | Path, first_chapters: Any
) -> list[str] | None:
    """The chapter filenames a chunk-based stage may visit, or ``None`` for all.

    Stage 5's manifest is the authority. Returning the *names* rather than a
    chunk count is what keeps the window correct when front matter sits before
    chapter 1: a plain chunk prefix would start in the front matter, while this
    filter selects exactly the chunks of the chosen body chapters. ``None`` means
    no window was requested; a window always returns body-chapter names, even
    when it covers every body chapter available, so front/back matter never
    sneaks into a windowed translation.
    """
    limit = normalize_limit(first_chapters)
    if limit is None:
        return None
    names = chunk_chapter_names(chunks_path)
    if not names:
        raise ToolError(
            "cannot apply a chapter window: the chunk manifest has no chapter "
            "membership — re-run `chunk` so every chunk records its chapter"
        )
    body = body_chapter_names(names)
    if not body:
        raise ToolError(
            "cannot apply a chapter window: the chunk manifest records no body "
            "chapters — run without --first-chapters, or use a chunk-based --limit"
        )
    return body[:limit]


def chunk_limit_for_chapters(chunks_path: str | Path, first_chapters: Any) -> Limit:
    """How many chunks make up the first ``first_chapters`` body chapters.

    Returns ``None`` when no window is requested, ``0`` when there are no chunks
    at all, and every chunk when N is at least the number of body chapters.
    Raises :class:`ToolError` when a window is requested but the chunks do not
    record which chapter they belong to (or record no body chapter at all).
    """
    limit = normalize_limit(first_chapters)
    if limit is None:
        return None

    entries = chunk_files(chunks_path)
    if not entries:
        return 0
    names = chunk_chapter_names(chunks_path)
    if not names:
        raise ToolError(
            "cannot apply a chapter window: the chunk manifest has no chapter "
            "membership — re-run `chunk` so every chunk records its chapter, or "
            "use a chunk-based --limit instead"
        )
    body = body_chapter_names(names)
    if not body:
        raise ToolError(
            "cannot apply a chapter window: the chunk manifest records no body "
            "chapters — run without --first-chapters, or use a chunk-based --limit"
        )
    if limit >= len(body):
        # Asking for every body chapter means the whole folder; a book's front
        # and back matter are part of it rather than a separate prefix.
        return len(entries)

    selected = set(body[:limit])
    unassigned = [
        str(entry.get("filename") or "?")
        for entry in entries
        if not str(entry.get("chapter_filename") or "")
    ]
    if unassigned:
        raise ToolError(
            "cannot apply a chapter window: "
            f"{len(unassigned)} chunk(s) do not record a chapter "
            f"({', '.join(unassigned[:3])}…) — re-run `chunk`"
        )
    return sum(
        1 for entry in entries if str(entry.get("chapter_filename") or "") in selected
    )
