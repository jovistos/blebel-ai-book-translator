"""The glossary run body, kept separate from its public entrypoint."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from book_translator.core.constants import (
    DEFAULT_GLOSSARY_ATTEMPTS,
    DEFAULT_GLOSSARY_CHUNK_CHARS,
    DEFAULT_SOURCE_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
    GLOSSARY_FILENAME,
    GLOSSARY_MANIFEST_FILENAME,
)
from book_translator.core.errors import ConfigError, ToolError
from book_translator.core.run import ItemOutcome, RunContext, run_items
from book_translator.documents.chapters import discover_chapters, glossary_dir
from book_translator.documents.manifest import load_split_manifest

from .artifacts import (
    _book_title,
    _extract_chapter,
    _previous_manifest,
    _resume_state,
    _reused_record,
    _skipped_record,
    _timing,
    resolve_sentences_dir,
    write_glossary,
    write_glossary_csv,
    write_glossary_manifest,
    write_glossary_markdown,
    write_glossary_trace,
)
from .extract import GlossaryExtractor, make_extractor
from .prompts import build_system_prompt
from .records import (
    ChapterGlossary,
    ChapterPlan,
    ChapterStartHook,
    Clock,
    GlossaryResult,
    ProgressHook,
    select_chapters,
)


# the run
# --------------------------------------------------------------------------- #
def execute_glossary(
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
    extractor_factory=make_extractor,
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

    if dry_run:
        # The glossary entry *is* the model's answer. A dry run validates the
        # sentence folder, the chapter selection and the resumable prefix, then
        # returns without constructing an extractor or spending a token.
        return GlossaryResult(
            ok=True,
            chapters_path=str(source_dir),
            sentences_dir=str(sentences_path),
            output_dir=str(out_dir),
            source_language=str(source_language),
            target_language=str(target_language),
            source_file=source_file,
            book_title=title,
            chapter_count=len(plans),
            body_chapter_count=len(processable),
            processed_count=len(pending),
            ok_count=0,
            skipped_count=0,
            reused_count=0,
            entry_count=len(book),
            dry_run=True,
            limited=limited,
            save=save,
            resumed=resumed,
            settings=chosen_settings,
            tokens={},
            categories=book.categories(),
            conflicts=previous_conflicts,
            timing=_timing([], run_seconds=context.elapsed()),
            chapters=[],
            entries=book.to_list(),
            errors=[],
        )

    if extractor is None and pending:
        if settings is None and settings_factory is not None:
            # Resolved only now: a resumed run with nothing left to mine never
            # touches the provider, so it needs neither a key nor a network.
            settings = settings_factory()
        if settings is None:
            raise ConfigError(
                "the glossary stage needs a model: pass settings (provider/model) or an extractor"
            )
        extractor = extractor_factory(
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
            write_glossary_csv(out_dir, partial, book)
            write_glossary_markdown(out_dir, partial, book)
            write_glossary_manifest(out_dir, partial, list(records))

    processed = 0

    def _work(plan: ChapterPlan, _position: int) -> ChapterGlossary:
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
        write_glossary_csv(out_dir, result, book)
        result.markdown_path = str(write_glossary_markdown(out_dir, result, book))
        result.manifest_path = str(write_glossary_manifest(out_dir, result, records))
        result.trace_path = str(write_glossary_trace(out_dir, result, records))
    return context.finish(result)


__all__ = ["execute_glossary"]
