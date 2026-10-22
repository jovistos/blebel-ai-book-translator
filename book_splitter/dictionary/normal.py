"""The normal dictionary pass: language routing, tagger load and chapter loop."""

from __future__ import annotations

from ..core.errors import MissingDependencyError, ToolError
from ..core.run import RunContext, run_items
from ..core.store import ArtifactStore
from ..documents.chapters import Chapter
from .artifacts import entity_totals, skipped_record, timing, totals, unique_totals
from .discover import DictionaryInputs
from .language import LanguageDetector, detect_language
from .models import ChapterReduction, Clock, DictionaryResult
from .ner import (
    NerBackend,
    backend_device,
    backend_for_language,
    configure_gpu,
    make_backend,
)
from .reduce import reduce_chapter
from .result import build_dictionary_result, write_dictionary_result
from .unique import extend_seen


def run_reductions(
    inputs: DictionaryInputs,
    *,
    context: RunContext,
    store: ArtifactStore,
    language: str,
    detector: LanguageDetector | None,
    backend: NerBackend | None,
    backend_name: str | None,
    gpu: int | None,
    use_gpu: bool,
    auto_download: bool,
    min_words: int,
    require_uppercase: bool,
    keep_types: tuple[str, ...],
    drop_temporal_only: bool,
    save_reduced: bool,
    save_unique: bool,
    dry_run: bool,
    example_limit: int,
    unique_example_limit: int,
    clock: Clock,
    detect_language_fn=detect_language,
    backend_for_language_fn=backend_for_language,
    configure_gpu_fn=configure_gpu,
    make_backend_fn=make_backend,
) -> DictionaryResult:
    guess = detect_language_fn(inputs.chapters, language=language, detector=detector)
    chosen = backend_name or (
        backend.name if backend is not None else backend_for_language_fn(guess.language)
    )

    backend_load_seconds = 0.0
    resolved_gpu = None
    if backend is None:
        resolved_gpu = configure_gpu_fn(gpu, use_gpu=use_gpu)
        load_started = clock()
        try:
            backend = make_backend_fn(
                guess.language,
                use_gpu=use_gpu,
                auto_download=auto_download,
                backend=chosen,
            )
        except MissingDependencyError:
            raise
        except Exception as exc:
            raise ToolError(
                f"could not load the {chosen} models for {guess.language!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        backend_load_seconds = clock() - load_started
    chosen = backend.name
    if not dry_run:
        inputs.out_dir.mkdir(parents=True, exist_ok=True)
        if save_unique:
            inputs.unique_out_dir.mkdir(parents=True, exist_ok=True)

    records: list[ChapterReduction] = []
    pending_names = {chapter.filename for chapter in inputs.pending}
    #: The book-wide noun set, carried from chapter to chapter in reading order.
    seen: set[str] = set()
    processed = 0

    def work(chapter: Chapter, _position: int) -> ChapterReduction:
        if chapter.filename not in pending_names:
            record = skipped_record(
                chapter,
                inputs.out_dir,
                inputs.unique_out_dir if save_unique else None,
                inputs.previous_files.get(chapter.filename),
            )
            if save_unique:
                extend_seen(seen, record.noun_keys)
            return record
        try:
            return reduce_chapter(
                chapter,
                inputs.out_dir,
                unique_out_dir=inputs.unique_out_dir,
                backend=backend,
                min_words=min_words,
                require_uppercase=require_uppercase,
                keep_types=keep_types,
                drop_temporal_only=drop_temporal_only,
                example_limit=example_limit,
                save_reduced=save_reduced,
                save_unique=save_unique,
                unique_example_limit=unique_example_limit,
                seen=seen,
                dry_run=dry_run,
                clock=clock,
            )
        except Exception as exc:  # one bad chapter must not stop the book
            return ChapterReduction(
                chapter_id=chapter.chapter_id,
                filename=chapter.filename,
                source_path=chapter.path,
                title=chapter.title,
                start_line=chapter.start_line,
                end_line=chapter.end_line,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    def failed(chapter: Chapter, exc: Exception) -> ChapterReduction:
        return ChapterReduction(
            chapter_id=chapter.chapter_id,
            filename=chapter.filename,
            source_path=chapter.path,
            title=chapter.title,
            start_line=chapter.start_line,
            end_line=chapter.end_line,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    def collect(outcome, _completed: int, _total: int) -> None:
        nonlocal processed
        record = outcome.value
        if record is None:  # pragma: no cover - failed always supplies a record
            return
        records.append(record)
        if record.skipped:
            return
        processed += 1
        context.emit(record.to_dict(), processed, len(inputs.pending))

    run_items(inputs.chapters, work, on_error=failed, on_item=collect)

    errors = [
        f"{record.filename}: {record.error}"
        for record in records
        if record.error
    ]
    result = build_dictionary_result(
        records=records,
        chapters_path=str(inputs.source_dir),
        output_dir=str(inputs.out_dir),
        unique_dir=str(inputs.unique_out_dir),
        language=guess.to_dict(),
        backend=chosen,
        gpu=resolved_gpu,
        device=backend_device(backend),
        settings=inputs.settings,
        totals=totals(records),
        unique_totals=unique_totals(records),
        entities=entity_totals(records),
        timing=timing(
            records,
            backend_load_seconds=backend_load_seconds,
            run_seconds=context.elapsed(),
        ),
        errors=errors,
        dry_run=dry_run,
        limited=inputs.limited,
        save_reduced=save_reduced,
        save_unique=save_unique,
    )
    return write_dictionary_result(store, result, inputs.chapters, records)


