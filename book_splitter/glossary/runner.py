"""Stage-4 entry point: turn the reduced chapters into one book-wide glossary.

Stage 3 left a folder of first mentions — ``03-reduced-unique/`` — one sentence per
noun, in reading order, and stopped there on purpose. This is the pass that folder
was built for.

The run walks the **body** chapters from first to last, one model call per
chapter, and grows a single dictionary:

```
03-reduced-unique/01-chapter-1.md   -> call 1: the dictionary is empty, the model
                                    finds the entities and fixes a rendering
                                    for each
03-reduced-unique/02-chapter-2.md   -> call 2: the prompt now carries the whole
                                    dictionary so far, and the model is told
                                    only to add what is new
...
03-reduced-unique/41-chapter-41.md  -> call 41
                                 -> 04-glossary/glossary.json + glossary.md
```

That order is the whole design. A dictionary built chapter by chapter *with the
earlier decisions in the prompt* is what makes the renderings compatible: the
model that names a place in chapter 30 can see how the place was named in chapter
2, and it is instructed to treat those as final. A dictionary built in one pass
over the whole book would be cheaper and would drift exactly where it matters.

Front matter and back matter are **skipped by default**: the copyright page, the
table of contents and the author's note carry names (a publisher, a city, an
acknowledgement) that must not be locked into the book's terminology. The stage-1
manifest records each file's ``kind``, so the decision is read rather than guessed
— with a filename fallback (``00-front-matter.md``, ``*-back-matter.md``) for a
folder whose manifest is missing. ``include_front_matter`` /
``include_back_matter`` put either block back.

Every chapter after the first sees the dictionary *as it stands*, and the calls
are counted: ``result.tokens`` is the sum of every attempt of every chapter, which
is the number the run reports last.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.artifacts import (
    read_json,
    read_manifest,
    timestamp,
    write_json,
    write_manifest,
    write_text,
    write_trace,
)
from ..core.constants import (
    DEFAULT_GLOSSARY_ATTEMPTS,
    DEFAULT_GLOSSARY_CHUNK_CHARS,
    DEFAULT_SOURCE_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
    FRONT_MATTER_FILENAME,
    GLOSSARY_FILENAME,
    GLOSSARY_MANIFEST_FILENAME,
    GLOSSARY_MARKDOWN_FILENAME,
    GLOSSARY_TRACE_FILENAME,
)
from ..core.errors import ConfigError, ExtractionError, ToolError
from ..core.llm import TokenUsage
from ..core.run import ItemOutcome, RunContext, run_items
from ..documents.chapters import (
    Chapter,
    discover_chapters,
    glossary_dir,
    reduced_dir,
    unique_dir,
)
from ..documents.io import load_lines
from ..documents.manifest import load_split_manifest
from .extract import GlossaryExtractor, make_extractor
from .models import GlossaryEntry, TermCategory, entry_key
from .prompts import build_chapter_prompt, build_system_prompt

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


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def run_glossary(
    chapters_path: str,
    output_dir: str | None = None,
    *,
    sentences_dir: str | None = None,
    sentences: str = "unique",
    source_language: str = DEFAULT_SOURCE_LANGUAGE,
    target_language: str = DEFAULT_TARGET_LANGUAGE,
    book_title: str = "",
    extractor: GlossaryExtractor | None = None,
    settings: Any = None,
    settings_factory: Callable[[], Any] | None = None,
    max_attempts: int = DEFAULT_GLOSSARY_ATTEMPTS,
    method: str = "auto",
    max_tokens: int | None = None,
    chunk_chars: int = DEFAULT_GLOSSARY_CHUNK_CHARS,
    max_context_entries: int = 0,
    include_front_matter: bool = False,
    include_back_matter: bool = False,
    skip_if_present: bool = False,
    save: bool = True,
    dry_run: bool = False,
    limit: int | None = None,
    progress: ProgressHook | None = None,
    chapter_start: ChapterStartHook | None = None,
    clock: Clock = time.perf_counter,
) -> GlossaryResult:
    """Build the glossary of one book, one chapter at a time, first to last.

    ``sentences="unique"`` reads the first-mention reduction (the default, and the
    cheap one); ``"reduced"`` reads every entity-bearing sentence instead. An
    explicit ``sentences_dir`` overrides both. ``extractor`` is injected by the
    tests and by anything that wants a different model; when it is ``None`` the
    chat model is built from ``settings``. ``settings_factory`` defers resolving
    those settings until a chapter actually needs a model, which is what lets a
    fully resumed run finish without a provider.

    ``skip_if_present`` resumes: the dictionary and the cost of the chapters
    already recorded in ``_glossary_manifest.json`` are loaded, and the run starts
    at the first body chapter that is missing from it. That is what makes a run
    that died on chapter 30 cost the price of chapters 30..N, not of the book. A
    manifest that names a different chapter folder (raw ``01-chapters/`` vs cleaned
    ``02-preprocessed/``) is not resumed: the filenames match but the text does not.

    ``dry_run`` still calls the model — the answer *is* the artifact, so there is
    nothing to validate without asking — but writes no file.
    """
    chapters = discover_chapters(chapters_path)
    limited = bool(limit is not None and 0 < int(limit) < len(chapters))
    context = RunContext(
        clock=clock,
        dry_run=dry_run,
        settings=settings,
        limited=limited,
        progress=progress,
    )
    if limit is not None and int(limit) > 0:
        chapters = chapters[: int(limit)]
    plans = select_chapters(
        chapters,
        include_front_matter=include_front_matter,
        include_back_matter=include_back_matter,
    )
    source_dir = Path(chapters_path).expanduser()
    out_dir = Path(output_dir).expanduser() if output_dir else glossary_dir(source_dir)
    sentences_path = resolve_sentences_dir(
        source_dir, sentences_dir=sentences_dir, sentences=sentences
    )

    manifest = load_split_manifest(source_dir)
    # Stage 1 has written both keys over its lifetime: ``source_file`` in the
    # current manifest, ``source_path`` in older ones.
    source_file = str(manifest.get("source_file") or manifest.get("source_path") or "")
    title = book_title or _book_title(source_file, source_dir)

    chosen_settings = {
        "sentences": sentences,
        "max_attempts": int(max_attempts),
        "method": method,
        "chunk_chars": int(chunk_chars),
        "max_context_entries": int(max_context_entries),
        "include_front_matter": bool(include_front_matter),
        "include_back_matter": bool(include_back_matter),
        "skip_if_present": bool(skip_if_present),
        "max_tokens": int(max_tokens) if max_tokens else None,
    }

    previous_manifest = _previous_manifest(out_dir) if skip_if_present else {}
    # Resume is only valid for the same chapter folder: raw ``01-chapters/`` and
    # cleaned ``02-preprocessed/`` share filenames but not text, so a dictionary
    # built from one is not a partial version of the other. A changed source
    # clears the previous manifest and rebuilds, while an older manifest without
    # the field keeps the old resume behaviour.
    previous_source = str(previous_manifest.get("chapters_path") or "").strip()
    same_source = not previous_source or (
        Path(previous_source).expanduser().resolve() == source_dir.resolve()
    )
    if not same_source:
        previous_manifest = {}
    previous_files = {
        str(entry.get("filename")): entry
        for entry in previous_manifest.get("files") or []
        if isinstance(entry, dict) and entry.get("filename")
    }
    resumed = bool(previous_manifest)
    resume_enabled = bool(skip_if_present and not dry_run and same_source)
    book, totals, previous_conflicts, done = _resume_state(
        out_dir, previous_manifest, enabled=resume_enabled
    )
    previous_entry_count = int(previous_manifest.get("entry_count") or 0)
    if resume_enabled and previous_entry_count and not book:
        # The manifest is a record of a dictionary that existed; an empty (or
        # missing) glossary.json next to it is damage, not an instruction to throw
        # the renderings away. Writing the empty book would destroy work that a
        # translation already depends on, so the run refuses instead.
        raise ToolError(
            f"refusing to overwrite {out_dir / GLOSSARY_FILENAME}: "
            f"{out_dir / GLOSSARY_MANIFEST_FILENAME} records {previous_entry_count} "
            "entr(y/ies) but the glossary file holds none. Restore the file, or run "
            "without --resume to build a fresh dictionary."
        )

    processable = [plan for plan in plans if plan.will_process]
    pending = [plan for plan in processable if plan.chapter.filename not in done]

    if extractor is None and pending:
        if settings is None and settings_factory is not None:
            # Resolved only now: a resumed run with nothing left to mine never
            # touches the provider, so it needs neither a key nor a network.
            settings = settings_factory()
        if settings is None:
            raise ConfigError(
                "the glossary stage needs a model: pass settings (provider/model) or an extractor"
            )
        extractor = make_extractor(
            settings, max_attempts=max_attempts, method=method, max_tokens=max_tokens
        )
    extractor_name = getattr(extractor, "name", "injected") if extractor is not None else ""

    system_prompt = build_system_prompt(source_language, target_language, title)

    if pending and not dry_run and save:
        out_dir.mkdir(parents=True, exist_ok=True)

    records: list[ChapterGlossary] = []
    pending_names = {plan.chapter.filename for plan in pending}
    extras: dict[str, Any] = {"extractor_name": extractor_name}

    def build_result() -> GlossaryResult:
        """The run as it stands. Called once per chapter to snapshot the work.

        A chapter is a ~70-second call and a book is dozens of them, so the
        dictionary is not allowed to exist only in memory: after every finished
        chapter the three small artifacts are rewritten. Ctrl-C (or a killed
        terminal) then costs the chapter in flight, not the run.
        """
        errors = [f"{r.filename}: {r.error}" for r in records if r.error]
        called_chapters = sum(1 for r in records if r.models_called)
        # A provider that rejected native structured output mid-run leaves the
        # extractor in JSON mode for good; recording it here is what lets the
        # summary say so instead of leaving a run's fallback invisible.
        chosen_settings["json_only"] = bool(getattr(extractor, "json_only", False))
        return GlossaryResult(
            ok=all(r.ok for r in records),
            chapters_path=str(source_dir),
            sentences_dir=str(sentences_path),
            output_dir=str(out_dir),
            source_language=str(source_language),
            target_language=str(target_language),
            source_file=source_file,
            book_title=title,
            extractor=str(extras["extractor_name"]),
            provider=str(getattr(settings, "provider", "") or ""),
            model=str(getattr(settings, "model", "") or ""),
            chapter_count=len(plans),
            body_chapter_count=len(processable),
            processed_count=len(pending),
            ok_count=sum(
                1 for r in records if r.ok and not r.skipped and not r.reused
            ),
            skipped_count=sum(1 for r in records if r.skipped),
            reused_count=sum(1 for r in records if r.reused),
            entry_count=len(book),
            dry_run=dry_run,
            limited=limited,
            save=save,
            resumed=resumed,
            settings=chosen_settings,
            tokens={**totals.to_dict(), "chapters": called_chapters},
            categories=book.categories(),
            conflicts=previous_conflicts + list(book.conflicts),
            timing=_timing(records, run_seconds=context.elapsed()),
            chapters=[r.record() for r in records],
            entries=book.to_list(),
            errors=errors,
        )

    def snapshot() -> None:
        """Persist the dictionary built so far (never an empty one)."""
        if not save or dry_run or not book:
            return
        partial = build_result()
        with contextlib.suppress(OSError):  # a snapshot must never end the run
            write_glossary(out_dir, partial, book)
            write_glossary_markdown(out_dir, partial, book)
            write_glossary_manifest(out_dir, partial, list(records))

    processed = 0

    def _work(plan: ChapterPlan) -> ChapterGlossary:
        """One plan's outcome, with the model call isolated by ``run_items``."""
        chapter = plan.chapter
        if not plan.will_process:
            return _skipped_record(plan)
        if chapter.filename not in pending_names:
            return _reused_record(plan, previous_files.get(chapter.filename))
        if chapter_start is not None:
            with contextlib.suppress(Exception):  # a reporting hook must never break a run
                chapter_start(
                    {
                        "index": plan.index,
                        "body_index": plan.body_index,
                        "filename": chapter.filename,
                        "title": plan.title,
                        "done": processed,
                        "total": len(pending),
                    }
                )
        return _extract_chapter(
            plan,
            sentences_path=sentences_path,
            extractor=extractor,
            system_prompt=system_prompt,
            book=book,
            chunk_chars=int(chunk_chars),
            max_context_entries=int(max_context_entries),
            clock=clock,
        )

    def _failed(plan: ChapterPlan, exc: Exception) -> ChapterGlossary:
        chapter = plan.chapter
        return ChapterGlossary(
            index=plan.index,
            chapter_id=chapter.chapter_id,
            filename=chapter.filename,
            title=plan.title,
            kind=plan.kind,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    def _collect(outcome: ItemOutcome, _completed: int, _total: int) -> None:
        """Record every plan, price the ones that were asked, snapshot the book."""
        nonlocal processed, totals
        record = outcome.value
        if record is None:  # pragma: no cover - _failed always supplies a record
            return
        records.append(record)
        if record.skipped or record.reused:
            return
        totals += record.usage
        processed += 1
        context.emit(record.to_dict(), processed, len(pending))
        snapshot()

    run_items(plans, _work, on_error=_failed, on_item=_collect)

    result = build_result()
    if save and not dry_run and not book and result.processed_count and not result.ok_count:
        # Every chapter that was asked failed. The in-memory dictionary is empty,
        # and writing it would replace whatever is already on disk with nothing.
        raise ToolError(
            "every chapter failed, so there is no dictionary to write; "
            f"{out_dir / GLOSSARY_FILENAME} was left untouched"
        )
    if save and not dry_run:
        result.glossary_path = str(write_glossary(out_dir, result, book))
        result.markdown_path = str(write_glossary_markdown(out_dir, result, book))
        result.manifest_path = str(write_glossary_manifest(out_dir, result, records))
        result.trace_path = str(write_glossary_trace(out_dir, result, records))
    return context.finish(result)


# --------------------------------------------------------------------------- #
# finding the evidence
# --------------------------------------------------------------------------- #
def resolve_sentences_dir(
    chapters_path: Path, *, sentences_dir: str | None, sentences: str
) -> Path:
    """Where the sentences to mine come from: explicit, or the sibling folder.

    ``unique`` prefers ``03-reduced-unique/`` and falls back to ``03-reduced/`` when the
    first-mention folder was never written (a stage-3 run with
    ``--no-save-unique``). ``reduced`` goes straight to ``03-reduced/``. Either way
    the folder has to hold at least one chapter file, or the run says which folder
    it looked in instead of silently building an empty dictionary.
    """
    if sentences_dir:
        folder = Path(sentences_dir).expanduser()
        if not _has_markdown(folder):
            raise ToolError(f"no sentence files in {folder}")
        return folder
    wanted = str(sentences or "unique").strip().lower()
    if wanted not in {"unique", "reduced"}:
        raise ToolError(f"unknown sentences source {sentences!r}; use 'unique' or 'reduced'")
    candidates = (
        [unique_dir(chapters_path), reduced_dir(chapters_path)]
        if wanted == "unique"
        else [reduced_dir(chapters_path)]
    )
    for folder in candidates:
        if _has_markdown(folder):
            return folder
    looked = " and ".join(str(folder) for folder in candidates)
    raise ToolError(
        f"no reduced sentences found in {looked} — run `dictionary` first, "
        "or point --sentences-in at the folder that holds them"
    )


def _has_markdown(folder: Path) -> bool:
    return folder.is_dir() and any(
        path.is_file() and not path.name.startswith("_") for path in folder.glob("*.md")
    )


def _book_title(source_file: str, chapters_path: Path) -> str:
    """The book's name, for the prompt: the source filename without its suffix."""
    name = Path(source_file).name if source_file else chapters_path.parent.name
    for suffix in (".md", ".txt", ".markdown"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
    return name.replace("_", " ").strip()


def chunk_sentences(sentences: Sequence[str], limit: int) -> list[list[str]]:
    """Split a chapter's sentences into model-sized parts, on line boundaries.

    Only a chapter whose reduced file is larger than ``limit`` is split at all —
    the reduced folders are small enough that most chapters are one call. The
    limit exists so that a single enormous chapter cannot overflow a context
    window and fail: the parts are processed in order and the dictionary is
    carried from one to the next, exactly as it is carried between chapters.
    """
    lines = [line.rstrip() for line in sentences if line.strip()]
    if not lines:
        return [[]]
    if limit <= 0:
        return [lines]
    chunks: list[list[str]] = []
    current: list[str] = []
    size = 0
    for line in lines:
        if current and size + len(line) + 1 > limit:
            chunks.append(current)
            current = []
            size = 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append(current)
    return chunks


# --------------------------------------------------------------------------- #
# one chapter
# --------------------------------------------------------------------------- #
def _extract_chapter(
    plan: ChapterPlan,
    *,
    sentences_path: Path,
    extractor: GlossaryExtractor | None,
    system_prompt: str,
    book: GlossaryBook,
    chunk_chars: int,
    max_context_entries: int,
    clock: Clock,
) -> ChapterGlossary:
    """Read one reduced chapter, ask the model about it, fold the answer in.

    The dictionary is passed to the prompt *as it is at the start of each part*,
    so a long chapter's later parts see the entities its earlier parts just locked
    in — the same compatibility rule as between chapters, applied one level down.
    """
    chapter = plan.chapter
    chapter_started = clock()
    path = Path(sentences_path) / chapter.filename
    record = ChapterGlossary(
        index=plan.index,
        body_index=plan.body_index,
        chapter_id=chapter.chapter_id,
        filename=chapter.filename,
        title=plan.title,
        kind=plan.kind,
        sentences_path=str(path),
    )
    if not path.is_file():
        record.ok = False
        record.error = f"no reduced chapter at {path}"
        return record
    sentences = load_lines(path)
    chunks = chunk_sentences(sentences, chunk_chars)
    record.output_lines = len([line for line in sentences if line.strip()])
    record.source_chars = sum(len(line) for line in sentences)
    record.chunks = len(chunks)
    for part, chunk in enumerate(chunks, start=1):
        if extractor is None:  # pragma: no cover - guarded by run_glossary
            raise ConfigError("no extractor was provided for the glossary stage")
        user_prompt = build_chapter_prompt(
            chapter_index=plan.body_index or plan.index,
            chapter_filename=chapter.filename,
            chapter_title=plan.title,
            sentences=chunk,
            dictionary=book.terms(),
            max_dictionary_entries=max_context_entries,
            part=part,
            parts=len(chunks),
        )
        try:
            outcome = extractor.extract(system_prompt=system_prompt, user_prompt=user_prompt)
        except ExtractionError as exc:
            record.ok = False
            record.error = str(exc)
            record.log.append(f"part {part}/{len(chunks)}: extraction failed: {exc}")
            break
        record.models_called += 1
        record.attempts += int(outcome.attempts)
        record.method = outcome.method or record.method
        record.usage += outcome.usage
        if outcome.error:
            record.log.append(f"part {part}/{len(chunks)}: retried after {outcome.error}")
        if outcome.raw_text:
            record.raw_answers.append(outcome.raw_text)
        # Provenance is the *body* chapter number, the one the prompt used and the
        # one a translator looking at the glossary will count to.
        merge = book.merge(
            outcome.entries,
            chapter_index=plan.body_index or plan.index,
            chapter_filename=chapter.filename,
        )
        record.added += len(merge.added)
        record.repeats += len(merge.repeats)
        record.conflicts += len(merge.conflicts)
        record.added_terms.extend(term.to_dict() for term in merge.added)
        record.conflict_detail.extend(merge.conflicts)
        record.log.append(
            f"part {part}/{len(chunks)}: {len(outcome.entries)} entr"
            f"{'y' if len(outcome.entries) == 1 else 'ies'} returned -> "
            f"{len(merge.added)} new, {len(merge.repeats)} already locked, "
            f"{len(merge.conflicts)} conflicting"
        )
    record.elapsed_seconds = clock() - chapter_started
    return record


def _skipped_record(plan: ChapterPlan) -> ChapterGlossary:
    return ChapterGlossary(
        index=plan.index,
        body_index=plan.body_index,
        chapter_id=plan.chapter.chapter_id,
        filename=plan.chapter.filename,
        title=plan.title,
        kind=plan.kind,
        skipped=True,
        skip_reason=plan.skip_reason,
        log=[f"skipped: {plan.skip_reason}"],
    )


def _reused_record(plan: ChapterPlan, previous: dict[str, Any] | None) -> ChapterGlossary:
    """A chapter a resumed run did not visit, described from the old manifest."""
    record = ChapterGlossary(
        index=plan.index,
        body_index=plan.body_index,
        chapter_id=plan.chapter.chapter_id,
        filename=plan.chapter.filename,
        title=plan.title,
        kind=plan.kind,
        reused=True,
    )
    previous = previous or {}
    record.added = int(previous.get("added") or 0)
    record.repeats = int(previous.get("repeats") or 0)
    record.conflicts = int(previous.get("conflicts") or 0)
    record.output_lines = int(previous.get("output_lines") or 0)
    record.sentences_path = str(previous.get("sentences_path") or "")
    record.log.append("reused: the chapter is already in the glossary manifest")
    return record


# --------------------------------------------------------------------------- #
# resuming
# --------------------------------------------------------------------------- #
def _previous_manifest(out_dir: Path) -> dict[str, Any]:
    return read_manifest(out_dir, GLOSSARY_MANIFEST_FILENAME)


def _resume_state(
    out_dir: Path, previous_manifest: dict[str, Any], *, enabled: bool
) -> tuple[GlossaryBook, TokenUsage, list[dict[str, Any]], set[str]]:
    """The dictionary, cost and finished chapters of an earlier run.

    The entries come from ``glossary.json`` and the bookkeeping from the
    manifest, so a resumed run continues with the same locked renderings and
    reports the *whole* run's token total rather than only its own chapters.
    """
    book = GlossaryBook()
    totals = TokenUsage()
    conflicts: list[dict[str, Any]] = []
    done: set[str] = set()
    if not enabled:
        return book, totals, conflicts, done

    glossary_path = Path(out_dir) / GLOSSARY_FILENAME
    if glossary_path.is_file():
        data = read_json(glossary_path, default={})
        for entry in (data.get("entries") if isinstance(data, dict) else None) or []:
            if isinstance(entry, dict) and entry.get("term"):
                book.lock(GlossaryTerm.from_dict(entry))
    tokens = previous_manifest.get("tokens") or {}
    totals = TokenUsage(
        input_tokens=int(tokens.get("input_tokens") or 0),
        output_tokens=int(tokens.get("output_tokens") or 0),
        total_tokens=int(tokens.get("total_tokens") or 0),
        calls=int(tokens.get("calls") or 0),
    )
    for conflict in previous_manifest.get("conflicts") or []:
        if isinstance(conflict, dict):
            conflicts.append(conflict)
    for entry in previous_manifest.get("files") or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("ok") and not entry.get("skipped"):
            done.add(str(entry.get("filename") or ""))
    return book, totals, conflicts, done


# --------------------------------------------------------------------------- #
# writing the artifacts
# --------------------------------------------------------------------------- #
def glossary_payload(result: GlossaryResult, book: GlossaryBook) -> dict[str, Any]:
    """The dictionary itself — the file the translation stage consumes."""
    return {
        "stage": "glossary",
        "generated_at": timestamp(),
        "book_title": result.book_title,
        "source_file": result.source_file,
        "source_language": result.source_language,
        "target_language": result.target_language,
        "provider": result.provider,
        "model": result.model,
        "chapters_path": result.chapters_path,
        "sentences_dir": result.sentences_dir,
        "chapter_count": result.body_chapter_count,
        "entry_count": len(book),
        "categories": book.categories(),
        "tokens": result.tokens,
        "entries": book.to_list(),
    }


def write_glossary(out_dir: Path, result: GlossaryResult, book: GlossaryBook) -> Path:
    return write_json(Path(out_dir) / GLOSSARY_FILENAME, glossary_payload(result, book))


def write_glossary_markdown(out_dir: Path, result: GlossaryResult, book: GlossaryBook) -> Path:
    """The human-readable view: a table per category, in the order categories fill.

    Meant to be read in a diff or an editor while the book is being translated —
    ``glossary.json`` is the machine's copy, this is the reviewer's.
    """
    lines = [
        f"# Glossary — {result.book_title or 'the book'}",
        "",
        f"{result.source_language} → {result.target_language} · "
        f"{len(book)} entr{'y' if len(book) == 1 else 'ies'} · "
        f"built by {result.provider or '—'}/{result.model or '—'} from "
        f"{result.body_chapter_count} chapter(s)",
        "",
        f"Tokens: {result.tokens.get('total_tokens', 0):,} total "
        f"({result.tokens.get('input_tokens', 0):,} in, "
        f"{result.tokens.get('output_tokens', 0):,} out)",
        "",
    ]
    by_category: dict[str, list[GlossaryTerm]] = {}
    for entry in book:
        by_category.setdefault(entry.category, []).append(entry)
    for category, entries in by_category.items():
        lines.append(f"## {category}")
        lines.append("")
        lines.append("| Term | Translation | First seen |")
        lines.append("| --- | --- | --- |")
        for entry in entries:
            term = entry.term.replace("|", "\\|")
            translation = entry.translation.replace("|", "\\|")
            lines.append(f"| {term} | {translation} | {entry.chapter_filename} |")
        lines.append("")
    return write_text(Path(out_dir) / GLOSSARY_MARKDOWN_FILENAME, "\n".join(lines))


def write_glossary_manifest(
    out_dir: Path,
    result: GlossaryResult,
    records: Sequence[ChapterGlossary] | None = None,
) -> Path:
    """The run's bookkeeping: per chapter, what it cost and what it added."""
    if records is None:
        records = []
    payload = {
        "stage": "glossary",
        "ok": result.ok,
        "dry_run": result.dry_run,
        "limited": result.limited,
        "resumed": result.resumed,
        "book_title": result.book_title,
        "source_file": result.source_file,
        "source_language": result.source_language,
        "target_language": result.target_language,
        "provider": result.provider,
        "model": result.model,
        "extractor": result.extractor,
        "chapters_path": result.chapters_path,
        "sentences_dir": result.sentences_dir,
        "settings": result.settings,
        "chapter_count": result.chapter_count,
        "body_chapter_count": result.body_chapter_count,
        "processed_count": result.processed_count,
        "skipped_count": result.skipped_count,
        "reused_count": result.reused_count,
        "entry_count": result.entry_count,
        "categories": result.categories,
        "tokens": result.tokens,
        "timing": result.timing,
        "conflicts": result.conflicts,
        "files": [record.record() for record in records],
        "errors": result.errors,
    }
    return write_manifest(out_dir, GLOSSARY_MANIFEST_FILENAME, payload)


def write_glossary_trace(
    out_dir: Path,
    result: GlossaryResult,
    records: Sequence[ChapterGlossary] | None = None,
) -> Path:
    """The audit trail: per chapter, the entries it added and the answers it gave."""
    payload = result.to_dict()
    if records is not None:
        payload["detail"] = [record.to_dict() for record in records]
    return write_trace(out_dir, GLOSSARY_TRACE_FILENAME, payload)


def _timing(records: Sequence[ChapterGlossary], *, run_seconds: float) -> dict[str, float]:
    """Wall clock, plus how many provider requests the book actually cost.

    ``model_calls`` counts *attempts*, not chapters: a chapter that was retried
    (or that had to fall back to JSON mode after a provider refusal) made more
    than one request, and that is the number a cost estimate needs.
    """
    called = [record for record in records if record.models_called]
    model_seconds = sum(record.elapsed_seconds for record in called)
    total_attempts = sum(record.attempts for record in records)
    return {
        "run_seconds": round(float(run_seconds), 4),
        "model_seconds": round(float(model_seconds), 4),
        "seconds_per_chapter": round(model_seconds / len(called), 4) if called else 0.0,
        "model_calls": total_attempts,
    }


__all__ = [
    "BACK_MATTER_KINDS",
    "BACK_MATTER_MARKERS",
    "FRONT_MATTER_KINDS",
    "ChapterGlossary",
    "ChapterPlan",
    "ChapterStartHook",
    "Clock",
    "GlossaryBook",
    "GlossaryResult",
    "GlossaryTerm",
    "MergeOutcome",
    "ProgressHook",
    "chapter_kind",
    "chunk_sentences",
    "glossary_payload",
    "resolve_sentences_dir",
    "run_glossary",
    "select_chapters",
    "write_glossary",
    "write_glossary_manifest",
    "write_glossary_markdown",
    "write_glossary_trace",
]
