"""Stage 7: translate every body chunk, in reading order, with the chunk's glossary.

The first six stages produce translation material. Stage 6b localizes the front
and back matter with its own agent; this is the stage that turns the remaining
body chunks into the translated book:

```
05-chunks/00001-01-chapter-1-0000.md              06-chunk-glossary/00001-….md.json
        │                                              │
        └──────────────┬───────────────────────────────┘
                       ▼
             prompt: system + glossary rows + source chunk
                       │
                       ▼   one structured-output call per chunk
             07-translations/00001-01-chapter-1-0000.md
             07-translation-prompts/00001-….md.json          (the whole input)
             07-translation-prompts/00001-….md.prompt.md     (the same, readable)
```

Front/back matter chunks (``00-front-matter.md``, a preface, a notes section…) are
filtered out unless ``include_matter=True``: stage 6b already wrote their
localized chunks to the same ``07-translations/`` folder with a different prompt and
**without** the dictionary.

**The order matters.** Chunks are visited in reading order, one model call at a
time, because each call is handed the tail of the previous translation when
continuity context is enabled and because a run that is interrupted leaves a
readable prefix of the book. The loop is a plain ``for``, not a pool: chunk 137's
text does not exist until chunk 136 has been translated.

**The glossary is per chunk.** Stage 6 wrote ``matched_terms`` for exactly the
entities a chunk mentions; that list is put in the prompt verbatim. Point
``glossary_scope="book"`` at ``glossary.json`` instead to send the whole
dictionary with every chunk (more consistent, more input tokens).

**Every call is structured and validated.** The model is asked for
``{"translation": "..."}``, Pydantic validates it, and a refusal, an empty answer,
an unchanged source or a summary is retried with the failure named. A chunk that
still fails is reported and the book continues: a hole is visible in the manifest,
an aborted run is not.

**The whole input is kept.** For every chunk, ``07-translation-prompts/<chunk>.json``
holds the system prompt, the user prompt, the serialised messages, the glossary
rows that were sent, the source chunk and the answer (or the error);
``<chunk>.prompt.md`` holds the same two prompts in a form a human can read. That
is what makes a bad translation debuggable after the fact.

``limit`` translates only the first N chunks, and ``resume`` (``skip_if_present``)
reuses the translations an earlier run already wrote — both are there so a book
can be tested and a killed run can be continued without paying twice.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ..core.constants import (
    DEFAULT_SOURCE_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
    DEFAULT_TRANSLATION_ATTEMPTS,
    DEFAULT_TRANSLATION_CONTEXT_CHARS,
    DEFAULT_TRANSLATION_MAX_TOKENS,
)
from ..core.run import RunContext
from ..documents.manifest import load_translation_manifest
from .loop import translate_chunks
from .plan import prepare_translation
from .records import (
    ChunkStartHook,
    ChunkTranslation,
    Clock,
    GlossaryRow,
    ProgressHook,
    TranslationResult,
)
from .result import build_translation_result
from .support import (
    _carried_records,
    debug_filename,
    has_chunk_glossary_files,
    load_book_glossary_rows,
    load_chunk_glossary_rows,
    preserve_trailing_newlines,
    prompt_filename,
    resolve_chunk_glossary_dir,
    resolve_chunks_dir,
    write_translation_manifest,
    write_translation_trace,
)
from .translator import ChunkTranslator, make_translator


def run_translation(
    chunks_path: str,
    output_dir: str | None = None,
    *,
    prompts_dir: str | Path | None = None,
    chunk_glossary_path: str | Path | None = None,
    glossary_path: str | Path | None = None,
    glossary_scope: str = "chunk",
    source_language: str = DEFAULT_SOURCE_LANGUAGE,
    target_language: str = DEFAULT_TARGET_LANGUAGE,
    book_title: str = "",
    translator: ChunkTranslator | None = None,
    settings: Any = None,
    settings_factory: Callable[[], Any] | None = None,
    model_identity: str = "",
    max_attempts: int = DEFAULT_TRANSLATION_ATTEMPTS,
    method: str = "auto",
    max_tokens: int | None = DEFAULT_TRANSLATION_MAX_TOKENS,
    context_chars: int = DEFAULT_TRANSLATION_CONTEXT_CHARS,
    save: bool = True,
    dry_run: bool = False,
    limit: int | None = None,
    chapter_files: Sequence[str] | None = None,
    include_matter: bool = False,
    resume: bool = False,
    write_manifest: bool = True,
    write_prompts: bool = True,
    progress: ProgressHook | None = None,
    chunk_start: ChunkStartHook | None = None,
    clock: Clock = time.perf_counter,
) -> TranslationResult:
    """Translate the chunks of one book, in reading order, one call at a time.

    ``translator`` is injectable, which is how the offline tests run the whole
    stage without a provider. A real run builds it from ``settings`` (or
    ``settings_factory``, resolved only when at least one chunk still needs a
    call, so a fully resumed run needs no key).

    ``resume`` reuses a translation an earlier run wrote when the manifest records
    it as successful, the chunk has not changed, the glossary rows, context
    window, method and model still match, and the language pair is the same.
    ``chapter_files`` restricts the run to chunks belonging to those chapter
    filenames; ``limit`` then translates only the first N of the filtered chunks.
    ``dry_run`` still calls the model but writes no file.

    Front and back matter are skipped by default: they are localized by the
    separate :mod:`book_splitter.matter_agent` stage, with a different prompt and
    without the dictionary. Pass ``include_matter=True`` only to reproduce the
    older behaviour for a book whose matter the caller has already handled.
    """
    context = RunContext(
        clock=clock,
        dry_run=dry_run,
        settings=settings,
        limited=False,
        progress=progress,
    )
    plan = prepare_translation(
        chunks_path,
        output_dir,
        prompts_dir=prompts_dir,
        chunk_glossary_path=chunk_glossary_path,
        glossary_path=glossary_path,
        glossary_scope=glossary_scope,
        source_language=source_language,
        target_language=target_language,
        book_title=book_title,
        translator=translator,
        settings=settings,
        settings_factory=settings_factory,
        model_identity=model_identity,
        max_attempts=max_attempts,
        method=method,
        max_tokens=max_tokens,
        context_chars=context_chars,
        save=save,
        dry_run=dry_run,
        limit=limit,
        chapter_files=chapter_files,
        include_matter=include_matter,
        resume=resume,
        write_manifest=write_manifest,
        write_prompts=write_prompts,
        progress=progress,
        chunk_start=chunk_start,
        clock=clock,
        context=context,
        translator_factory=make_translator,
    )
    context.limited = plan.limited
    if not plan.inputs:
        return build_translation_result(
            plan=plan,
            empty=True,
            run_seconds=context.elapsed(),
        )

    outcome = translate_chunks(plan)
    result = build_translation_result(
        plan=plan,
        outcome=outcome,
        run_seconds=context.elapsed(),
    )
    if plan.write_manifest and not plan.dry_run and plan.save:
        carried = _carried_records(
            plan.carried_from,
            {str(record.chunk.filename) for record in outcome.records},
            plan.out_dir,
        )
        result.manifest_path = str(
            write_translation_manifest(
                plan.out_dir, result, outcome.records, extra_files=carried
            )
        )
        result.trace_path = str(
            write_translation_trace(plan.out_dir, result, outcome.records)
        )
    return context.finish(result)


__all__ = [
    "ChunkStartHook",
    "ChunkTranslation",
    "GlossaryRow",
    "ProgressHook",
    "TranslationResult",
    "debug_filename",
    "has_chunk_glossary_files",
    "load_book_glossary_rows",
    "load_chunk_glossary_rows",
    "load_translation_manifest",
    "make_translator",
    "preserve_trailing_newlines",
    "prompt_filename",
    "resolve_chunk_glossary_dir",
    "resolve_chunks_dir",
    "run_translation",
    "write_translation_manifest",
    "write_translation_trace",
]
