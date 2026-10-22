"""Records, plans and the book-wide dictionary for the glossary stage."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from ..core.constants import (
    DEFAULT_SOURCE_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
    FRONT_MATTER_FILENAME,
)
from ..core.llm import TokenUsage
from ..documents.chapters import Chapter
from .models import GlossaryEntry, TermCategory, entry_key

#: Entity kinds the stage-1 manifest uses for the two blocks around the body.
FRONT_MATTER_KINDS = frozenset({"front-matter", "frontmatter", "front matter", "front_matter"})
BACK_MATTER_KINDS = frozenset({"back-matter", "backmatter", "back matter", "back_matter"})
#: Filename fallbacks for a folder with no readable manifest.
BACK_MATTER_MARKERS = ("back-matter", "back_matter", "backmatter")

#: Monotonic clock used for the timings; injectable so tests can be exact.
Clock = Callable[[], float]


# --------------------------------------------------------------------------- #
# the dictionary being built
# --------------------------------------------------------------------------- #
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
class MergeOutcome:
    """What one chapter's answer did to the dictionary."""

    added: list[GlossaryTerm] = field(default_factory=list)
    repeats: list[str] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.added and not self.repeats and not self.conflicts


class GlossaryBook:
    """The dictionary, in insertion order, with the consistency rules applied.

    The first rendering of a term wins, always. That is the promise the prompt
    makes to the translator, so a later chapter that re-states a locked term
    differently is recorded as a **conflict** and the dictionary is left alone —
    the model gets another chance to conform in the next chapter, but the text it
    already translated can never disagree with itself.
    """

    def __init__(self, entries: Iterable[GlossaryTerm] = ()) -> None:
        self._entries: dict[str, GlossaryTerm] = {}
        self.conflicts: list[dict[str, Any]] = []
        for entry in entries:
            if entry.key:
                self._entries[entry.key] = entry

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)

    def __iter__(self):
        return iter(self._entries.values())

    def terms(self) -> list[GlossaryTerm]:
        return list(self._entries.values())

    def lock(self, entry: GlossaryTerm) -> None:
        """Install an already-locked entry, as ``--resume`` does from disk."""
        if entry.key:
            self._entries[entry.key] = entry

    def to_list(self) -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in self._entries.values()]

    def categories(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self._entries.values():
            counts[entry.category] = counts.get(entry.category, 0) + 1
        return dict(sorted(counts.items()))

    def merge(
        self,
        entries: Iterable[GlossaryEntry],
        *,
        chapter_index: int,
        chapter_filename: str,
    ) -> MergeOutcome:
        """Fold one chapter's answer into the dictionary.

        Runs in three cases, in this order: a term nobody has seen is added with
        its provenance; a term already locked in and returned with the *same*
        rendering is a repeat (deduplicated); a term already locked in and
        returned with a *different* rendering is a conflict, the existing entry
        stands, and both renderings are recorded so the run can report where the
        model tried to drift.
        """
        outcome = MergeOutcome()
        for entry in entries:
            key = entry.key
            if not key:
                continue
            existing = self._entries.get(key)
            if existing is None:
                term = GlossaryTerm(
                    term=entry.term,
                    category=entry.category.value,
                    translation=entry.translation,
                    chapter_index=chapter_index,
                    chapter_filename=chapter_filename,
                )
                self._entries[key] = term
                outcome.added.append(term)
                continue
            if _same_rendering(existing.translation, entry.translation):
                if chapter_filename not in existing.repeats:
                    existing.repeats.append(chapter_filename)
                outcome.repeats.append(existing.term)
                continue
            conflict = {
                "term": existing.term,
                "chapter_index": chapter_index,
                "chapter_filename": chapter_filename,
                "locked": existing.translation,
                "proposed": entry.translation,
            }
            self.conflicts.append(conflict)
            outcome.conflicts.append(conflict)
        return outcome


def _same_rendering(left: str, right: str) -> bool:
    return " ".join(str(left or "").split()).casefold() == " ".join(str(right or "").split()).casefold()


# --------------------------------------------------------------------------- #
# which chapters the model is asked about
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# artifacts
# --------------------------------------------------------------------------- #
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

    def record(self) -> dict[str, Any]:
        """The flat per-chapter record stored in the manifest."""
        return {
            "index": self.index,
            "body_index": self.body_index,
            "filename": self.filename,
            "chapter_id": self.chapter_id,
            "title": self.title,
            "kind": self.kind,
            "sentences_path": self.sentences_path,
            "output_lines": self.output_lines,
            "source_chars": self.source_chars,
            "chunks": self.chunks,
            "models_called": self.models_called,
            "attempts": self.attempts,
            "method": self.method,
            "added": self.added,
            "repeats": self.repeats,
            "conflicts": self.conflicts,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "reused": self.reused,
            "ok": self.ok,
            "error": self.error,
            "tokens": self.usage.to_dict(),
            "elapsed_seconds": round(self.elapsed_seconds, 4),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.record(),
            "log": list(self.log),
            "added_terms": list(self.added_terms),
            "conflict_detail": list(self.conflict_detail),
            "raw_answers": list(self.raw_answers),
        }


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
