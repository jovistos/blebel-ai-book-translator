"""The four records the glossary stage writes, and the chapter selection.

``GlossaryTerm``
    one locked dictionary row (``glossary.json``);
``ChapterPlan``
    one chapter file and whether the run will ask the model about it;
``ChapterGlossary``
    one chapter's part of the run — a manifest row plus a trace-only tail;
``GlossaryResult``
    the outcome of a whole stage-4 run.

The mutable dictionary those rows belong to lives in
:mod:`book_translator.glossary.book`; the wire shape of the model's answer lives in
:mod:`book_translator.glossary.models`. All four records here expose ``to_dict()``
and survive a JSON round-trip.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from ..core.constants import (
    DEFAULT_SOURCE_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
    FRONT_MATTER_FILENAME,
)
from ..core.llm import TokenUsage
from ..documents.chapters import Chapter
from .models import GlossaryEntry, TermCategory, entry_key

if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    from .book import GlossaryBook, MergeOutcome

#: Entity kinds the stage-1 manifest uses for the two blocks around the body.
FRONT_MATTER_KINDS = frozenset({"front-matter", "frontmatter", "front matter", "front_matter"})
BACK_MATTER_KINDS = frozenset({"back-matter", "backmatter", "back matter", "back_matter"})
#: Filename fallbacks for a folder with no readable manifest.
BACK_MATTER_MARKERS = ("back-matter", "back_matter", "backmatter")

#: Monotonic clock used for the timings; injectable so tests can be exact.
Clock = Callable[[], float]


@dataclass
class GlossaryTerm:
    """One locked rendering, plus where it entered the dictionary.

    ``chapter_index`` / ``chapter_filename`` are provenance: a translator looking
    at ``glossary.json`` can go back to the chapter that introduced an entity.
    ``repeats`` is the model returning a locked term again (which the prompt tells
    it not to) — harmless, deduplicated, and worth recording because it is the
    cheapest visible sign that a chapter's answer was lazy.
    """

    term: str
    category: str
    translation: str
    chapter_index: int = 0
    chapter_filename: str = ""
    repeats: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return entry_key(self.term)

    @classmethod
    def from_entry(
        cls,
        entry: GlossaryEntry,
        *,
        chapter_index: int = 0,
        chapter_filename: str = "",
    ) -> GlossaryTerm:
        """Lock one validated model answer into a dictionary row."""
        return cls(
            term=entry.term,
            category=entry.category.value,
            translation=entry.translation,
            chapter_index=chapter_index,
            chapter_filename=chapter_filename,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "term": self.term,
            "category": self.category,
            "translation": self.translation,
            "chapter_index": self.chapter_index,
            "chapter_filename": self.chapter_filename,
            "also_returned_in": list(self.repeats),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GlossaryTerm:
        return cls(
            term=str(data.get("term") or ""),
            category=str(data.get("category") or TermCategory.TERMINOLOGY.value),
            translation=str(data.get("translation") or ""),
            chapter_index=int(data.get("chapter_index") or 0),
            chapter_filename=str(data.get("chapter_filename") or ""),
            repeats=[str(name) for name in data.get("also_returned_in") or []],
        )


@dataclass
class ChapterPlan:
    """One chapter file, and whether the run will ask the model about it.

    ``index`` is the file's position in the folder (front matter is file 1 in
    every book stage 1 writes); ``body_index`` counts only the files the model
    will actually see, so the prompt and the entries call the book's first chapter
    "chapter 1" rather than "chapter 2, after the copyright page".
    """

    chapter: Chapter
    index: int
    kind: str
    skip_reason: str | None = None
    body_index: int = 0

    @property
    def will_process(self) -> bool:
        return self.skip_reason is None

    @property
    def title(self) -> str:
        """A label for the prompt: the recorded title, else the heading line.

        The first line is used only when it is a Markdown heading, because in a
        book whose chapter headings were lost that line is ordinary prose — and
        "Chapter 7 — Marko je ušao u sobu i seo." is not a title. A manifest that
        recorded the filename as the title is ignored for the same reason: it
        would be printed twice in one prompt line.
        """
        recorded = " ".join(str(self.chapter.title or "").split())
        if recorded and recorded not in {self.chapter.filename, self.chapter.chapter_id}:
            return recorded
        first_line = " ".join(str(self.chapter.extras.get("first_line") or "").split())
        if first_line.startswith("#"):
            heading = first_line.lstrip("#").strip()
            if heading and heading not in {self.chapter.filename, self.chapter.chapter_id}:
                return heading
        return ""


def chapter_kind(chapter: Chapter) -> str:
    """``chapter``, ``front-matter`` or ``back-matter`` — from the stage-1 manifest.

    The manifest's ``kind`` is the authority when it is there. The filename
    fallback exists because the glossary must still skip the right blocks when it
    is pointed at a folder whose manifest was lost, and getting that wrong is the
    difference between a clean dictionary and one that locks a publisher's name
    into the book's terminology.
    """
    kind = str(chapter.extras.get("kind") or "").strip().lower()
    if kind in FRONT_MATTER_KINDS:
        return "front-matter"
    if kind in BACK_MATTER_KINDS:
        return "back-matter"
    name = chapter.filename.strip().lower()
    if name == FRONT_MATTER_FILENAME.lower() or name.startswith("00-"):
        return "front-matter"
    if any(marker in name for marker in BACK_MATTER_MARKERS):
        return "back-matter"
    return "chapter"


def select_chapters(
    chapters: Sequence[Chapter],
    *,
    include_front_matter: bool = False,
    include_back_matter: bool = False,
) -> list[ChapterPlan]:
    """Decide, once and visibly, which chapter files the model will see.

    The skipped files stay in the plan rather than being dropped, so the run can
    report *why* the dictionary has no entries from them — a book whose copyright
    page was mined for names is a bug someone has to be able to see.
    """
    plans: list[ChapterPlan] = []
    body_index = 0
    for index, chapter in enumerate(chapters, start=1):
        kind = chapter_kind(chapter)
        reason: str | None = None
        if kind == "front-matter" and not include_front_matter:
            reason = "front matter"
        elif kind == "back-matter" and not include_back_matter:
            reason = "back matter"
        if reason is None:
            body_index += 1
        plans.append(
            ChapterPlan(
                chapter=chapter,
                index=index,
                kind=kind,
                skip_reason=reason,
                body_index=body_index if reason is None else 0,
            )
        )
    return plans


@dataclass
class ChapterGlossary:
    """One chapter's part of the run: what it cost and what it contributed."""

    index: int
    chapter_id: str
    filename: str
    title: str = ""
    kind: str = "chapter"
    body_index: int = 0
    sentences_path: str = ""
    output_lines: int = 0
    source_chars: int = 0
    chunks: int = 0
    models_called: int = 0
    attempts: int = 0
    method: str = ""
    added: int = 0
    repeats: int = 0
    conflicts: int = 0
    skipped: bool = False
    skip_reason: str | None = None
    reused: bool = False
    ok: bool = True
    error: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    elapsed_seconds: float = 0.0
    log: list[str] = field(default_factory=list)
    added_terms: list[dict[str, Any]] = field(default_factory=list)
    conflict_detail: list[dict[str, Any]] = field(default_factory=list)
    raw_answers: list[str] = field(default_factory=list)

    #: The manifest row, in this order. ``record()`` projects these fields and
    #: adds the two computed ones, so the manifest and trace shapes cannot
    #: drift apart.
    _MANIFEST_FIELDS: tuple[str, ...] = (
        "index",
        "body_index",
        "filename",
        "chapter_id",
        "title",
        "kind",
        "sentences_path",
        "output_lines",
        "source_chars",
        "chunks",
        "models_called",
        "attempts",
        "method",
        "added",
        "repeats",
        "conflicts",
        "skipped",
        "skip_reason",
        "reused",
        "ok",
        "error",
    )
    #: Present only in the trace, never in the manifest.
    _TRACE_FIELDS: tuple[str, ...] = (
        "log",
        "added_terms",
        "conflict_detail",
        "raw_answers",
    )

    def record(self) -> dict[str, Any]:
        """The flat per-chapter record stored in the manifest."""
        data: dict[str, Any] = {
            name: getattr(self, name) for name in self._MANIFEST_FIELDS
        }
        data["tokens"] = self.usage.to_dict()
        data["elapsed_seconds"] = round(self.elapsed_seconds, 4)
        return data

    def to_dict(self) -> dict[str, Any]:
        data = self.record()
        data.update({name: list(getattr(self, name)) for name in self._TRACE_FIELDS})
        return data





#: Called once per chapter as ``(chapter_record, completed, total)``.
#: Called once per chapter as ``(chapter_record, completed, total)``.
ProgressHook = Callable[[dict[str, Any], int, int], None]
#: Called *before* a chapter's first model call, with where it sits in the run.
#: A chapter is one long request, so a caller that wants a live bar needs to know
#: what is in flight, not only what has finished.
ChapterStartHook = Callable[[dict[str, Any]], None]

@dataclass
class GlossaryResult:
    """Outcome of a whole stage-4 run."""

    ok: bool
    chapters_path: str
    sentences_dir: str
    output_dir: str
    source_language: str = DEFAULT_SOURCE_LANGUAGE
    target_language: str = DEFAULT_TARGET_LANGUAGE
    source_file: str = ""
    book_title: str = ""
    extractor: str = ""
    provider: str = ""
    model: str = ""
    chapter_count: int = 0
    body_chapter_count: int = 0
    processed_count: int = 0
    ok_count: int = 0
    skipped_count: int = 0
    reused_count: int = 0
    entry_count: int = 0
    dry_run: bool = False
    #: True when ``limit`` kept this run to a prefix of the chapters.
    limited: bool = False
    save: bool = True
    resumed: bool = False
    settings: dict[str, Any] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)
    categories: dict[str, int] = field(default_factory=dict)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    timing: dict[str, float] = field(default_factory=dict)
    chapters: list[dict[str, Any]] = field(default_factory=list)
    entries: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    glossary_path: str | None = None
    markdown_path: str | None = None
    manifest_path: str | None = None
    trace_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def __getattr__(name: str) -> Any:
    """Keep ``records.GlossaryBook`` / ``records.MergeOutcome`` importable.

    They moved to :mod:`book_translator.glossary.book`; the lazy lookup avoids a
    records <-> book import cycle while preserving the old import path.
    """
    if name in {"GlossaryBook", "MergeOutcome"}:
        from . import book as _book

        return getattr(_book, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ChapterGlossary",
    "ChapterPlan",
    "ChapterStartHook",
    "GlossaryBook",
    "GlossaryResult",
    "GlossaryTerm",
    "MergeOutcome",
    "ProgressHook",
    "chapter_kind",
    "select_chapters",
]
