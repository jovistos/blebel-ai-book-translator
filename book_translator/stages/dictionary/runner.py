from __future__ import annotations

import time

from book_translator.core.constants import (
    DEFAULT_KEEP_ENTITY_TYPES,
    DEFAULT_MIN_SENTENCE_WORDS,
    DEFAULT_REQUIRE_UPPERCASE,
    REDUCED_MANIFEST_FILENAME,
    REDUCED_TRACE_FILENAME,
)
from book_translator.core.run import RunContext
from book_translator.core.store import ArtifactStore

from .artifacts import (
    skipped_record,
    write_reduced_manifest,
    write_reduced_trace,
)
from .discover import discover_dictionary
from .language import LanguageDetector, detect_language
from .models import (
    ChapterReduction,
    Clock,
    DictionaryResult,
    ProgressHook,
    TextReduction,
)
from .ner import (
    NerBackend,
    backend_for_language,
    configure_gpu,
    make_backend,
)
from .normal import run_reductions
from .reduce import (
    DEFAULT_EXAMPLE_LIMIT,
    MENTION_LIMIT,
    reduce_text,
    reduced_payload,
)
from .result import build_dictionary_result, write_dictionary_result
from .unique import DEFAULT_UNIQUE_EXAMPLES, UniqueSelection, select_unique_sentences


def run_dictionary(
    chapters_path: str,
    output_dir: str | None = None,
    *,
    unique_output_dir: str | None = None,
    language: str = "auto",
    detector: LanguageDetector | None = None,
    backend: NerBackend | None = None,
    backend_name: str | None = None,
    gpu: int | None = None,
    use_gpu: bool = True,
    auto_download: bool = True,
    min_words: int = DEFAULT_MIN_SENTENCE_WORDS,
    require_uppercase: bool = DEFAULT_REQUIRE_UPPERCASE,
    keep_types: tuple[str, ...] = DEFAULT_KEEP_ENTITY_TYPES,
    drop_temporal_only: bool = False,
    skip_if_present: bool = False,
    save_reduced: bool = True,
    save_unique: bool = True,
    dry_run: bool = False,
    limit: int | None = None,
    example_limit: int = DEFAULT_EXAMPLE_LIMIT,
    unique_example_limit: int = DEFAULT_UNIQUE_EXAMPLES,
    write_manifest: bool = True,
    progress: ProgressHook | None = None,
    clock: Clock = time.perf_counter,
) -> DictionaryResult:
    """Reduce every chapter of a stage-1 folder, one chapter at a time.

    ``skip_if_present`` reuses a chapter whose reduced file already exists and,
    when there is nothing left to reduce, returns before any model is loaded.
    Reuse is refused when the previous manifest names a different chapter folder.
    ``dry_run`` runs the same funnel but writes no chapter, manifest or trace.
    """
    inputs = discover_dictionary(
        chapters_path,
        output_dir,
        unique_output_dir,
        limit=limit,
        min_words=min_words,
        require_uppercase=require_uppercase,
        keep_types=keep_types,
        drop_temporal_only=drop_temporal_only,
        save_reduced=save_reduced,
        save_unique=save_unique,
        unique_example_limit=unique_example_limit,
        skip_if_present=skip_if_present,
        example_limit=example_limit,
        dry_run=dry_run,
    )
    context = RunContext(
        clock=clock,
        dry_run=dry_run,
        limited=inputs.limited,
        progress=progress,
    )
    store = ArtifactStore(
        inputs.out_dir,
        REDUCED_MANIFEST_FILENAME,
        REDUCED_TRACE_FILENAME,
        enabled=write_manifest and not dry_run,
    )

    if not inputs.pending:
        # Nothing left to reduce: report the previous run's own record instead of
        # loading a detector or a tagger, so a finished book needs no dependency.
        skipped_records = [
            skipped_record(
                chapter,
                inputs.out_dir,
                inputs.unique_out_dir if save_unique else None,
                inputs.previous_files.get(chapter.filename),
            )
            for chapter in inputs.chapters
        ]
        previous = inputs.previous
        result = build_dictionary_result(
            records=skipped_records,
            chapters_path=str(inputs.source_dir),
            output_dir=str(inputs.out_dir),
            unique_dir=str(inputs.unique_out_dir),
            language=dict(previous.get("language") or {}),
            backend=str(previous.get("backend") or ""),
            gpu=None,
            device="",
            settings=inputs.settings,
            totals={},
            unique_totals=(
                dict(previous.get("unique_totals") or {}) if save_unique else {}
            ),
            entities={},
            timing=dict(previous.get("timing") or {}),
            errors=[],
            dry_run=dry_run,
            limited=inputs.limited,
            save_reduced=save_reduced,
            save_unique=save_unique,
            skipped=True,
        )
        return write_dictionary_result(
            store, result, inputs.chapters, skipped_records
        )

    return run_reductions(
        inputs,
        context=context,
        store=store,
        language=language,
        detector=detector,
        backend=backend,
        backend_name=backend_name,
        gpu=gpu,
        use_gpu=use_gpu,
        auto_download=auto_download,
        min_words=min_words,
        require_uppercase=require_uppercase,
        keep_types=keep_types,
        drop_temporal_only=drop_temporal_only,
        save_reduced=save_reduced,
        save_unique=save_unique,
        dry_run=dry_run,
        example_limit=example_limit,
        unique_example_limit=unique_example_limit,
        clock=clock,
        detect_language_fn=detect_language,
        backend_for_language_fn=backend_for_language,
        configure_gpu_fn=configure_gpu,
        make_backend_fn=make_backend,
    )


__all__ = [
    "ChapterReduction",
    "Clock",
    "DEFAULT_EXAMPLE_LIMIT",
    "DEFAULT_UNIQUE_EXAMPLES",
    "DictionaryResult",
    "MENTION_LIMIT",
    "ProgressHook",
    "TextReduction",
    "UniqueSelection",
    "reduce_text",
    "reduced_payload",
    "run_dictionary",
    "select_unique_sentences",
    "write_reduced_manifest",
    "write_reduced_trace",
]
