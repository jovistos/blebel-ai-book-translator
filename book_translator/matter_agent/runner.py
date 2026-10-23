"""Entry point for the front/back matter localization stage.

The stage sits between stage 6's per-chunk glossary and stage 7's body
translation. It is deliberately *before* translation and deliberately separate:

```
05-chunks/00000-00-front-matter-0000.md
05-chunks/00001-01-predgovor-0000.md
        │
        ▼  one book-level plan (web search for an existing title, BleBel credit,
        │  omissions, formatting), then one chunk graph per matter chunk
07-translations/00000-00-front-matter-0000.md   ← stage 8 reads this as usual
06b-matter/_matter_manifest.json                 ← plan, per-chunk decisions, tokens
06b-matter-prompts/<chunk>.json / .prompt.md     ← the whole agent input per chunk
```

The translated chunks keep the chunk names, so stage 8 needs no new join rule.
Stage 7 is told to skip these chapters (see ``include_matter=False`` in
:mod:`book_translator.translation.runner`), which is what makes "front/back matter
has its own prompt and no dictionary" true end to end.

One book plan is produced first, then the chunks are localized in reading order.
The first front-matter chunk is special: the graph guarantees the localized title
and the ``BleBel`` credit are on it. An omitted chunk is written as an empty file
and recorded with its reason; the rest of the book is unaffected.

Nothing here uses the glossary. It is the one translation stage that is not
dictionary-driven.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ..core.constants import (
    DEFAULT_MATTER_ATTEMPTS,
    DEFAULT_RECURSION_LIMIT,
)
from ..core.settings import LLMSettings
from ..documents.manifest import load_matter_manifest
from .artifacts import (
    _add_usage,
    _previous_usage,
    write_matter_manifest,
    write_matter_trace,
)
from .chunks import MatterChunkRun, run_chunks
from .graph import build_matter_chunk_graph
from .models import Clock, MatterChunkResult, MatterResult, ProgressHook
from .plan import prepare_matter
from .result import finish_matter


def run_matter(
    chapters_path: str | Path,
    chunks_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    prompts_dir: str | Path | None = None,
    matter_path: str | Path | None = None,
    source_language: str = "",
    target_language: str = "",
    book_title: str = "",
    book_author: str = "",
    settings: LLMSettings | None = None,
    settings_factory: Callable[[], Any] | None = None,
    model: Any | None = None,
    book_model: Any | None = None,
    chunk_model: Any | None = None,
    max_attempts: int = DEFAULT_MATTER_ATTEMPTS,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    dry_run: bool = False,
    save: bool = True,
    resume: bool = False,
    limit: int | None = None,
    chapter_files: Sequence[str] | None = None,
    write_manifest: bool = True,
    write_prompts: bool = True,
    progress: ProgressHook | None = None,
    clock: Clock = time.perf_counter,
) -> MatterResult:
    """Localize every front/back matter chunk, after one book-level plan.

    ``book_model``/``chunk_model`` are injectable so the whole graph can be run
    offline against a scripted double. A real run builds them from ``settings``
    (or ``settings_factory``), resolved only when a model is actually needed, so
    a fully resumed run needs no API key.

    ``resume`` reuses a chunk whose manifest record is successful, whose source
    hash still matches, and whose translated file exists. It also reuses the
    previous book plan, which is what keeps the title/credit/omissions stable
    across a widened run.

    ``limit`` is a chunk limit inside the matter window, for a cheap test of the
    whole path. ``chapter_files`` restricts the run to named matter sections.
    ``dry_run`` still asks the model — the answer is the artifact — but writes no
    file, manifest or prompt record.
    """
    prepared = prepare_matter(
        chapters_path,
        chunks_path,
        output_dir,
        prompts_dir=prompts_dir,
        matter_path=matter_path,
        source_language=source_language,
        target_language=target_language,
        book_title=book_title,
        book_author=book_author,
        settings=settings,
        settings_factory=settings_factory,
        model=model,
        book_model=book_model,
        chunk_model=chunk_model,
        max_attempts=max_attempts,
        recursion_limit=recursion_limit,
        dry_run=dry_run,
        save=save,
        resume=resume,
        limit=limit,
        chapter_files=chapter_files,
        write_manifest=write_manifest,
        write_prompts=write_prompts,
    )
    if isinstance(prepared, MatterResult):
        return prepared
    plan_state = prepared
    run_started = clock()
    source_lang = plan_state.source_lang
    target_lang = plan_state.target_lang
    book = plan_state.book
    out_dir = plan_state.out_dir
    prompt_dir = plan_state.prompt_dir
    section_by_name = plan_state.section_by_name
    inputs = plan_state.inputs
    first_front = plan_state.first_front
    first_in_section = plan_state.first_in_section
    previous = plan_state.previous
    previous_files = plan_state.previous_files
    reused = plan_state.reused
    pending = plan_state.pending
    plan = plan_state.plan
    book_final = plan_state.book_final
    settings = plan_state.settings
    resume_enabled = plan_state.resume_enabled

    if not resume_enabled:
        # A plan that could not be reused invalidates the earlier chunk reuse:
        # the title/credit/omissions may have changed, so localize again.
        reused = [False] * len(inputs)
        pending = list(inputs)

    usage_tokens = _previous_usage(previous) if resume_enabled else {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "llm_calls": 0,
    }
    if book_final:
        _add_usage(usage_tokens, book_final.get("usage") or {})

    chunk_app = (
        build_matter_chunk_graph(
            settings,
            model=chunk_model if chunk_model is not None else model,
        )
        if pending
        else None
    )

    records, errors, usage_tokens = run_chunks(
        MatterChunkRun(
            inputs=inputs,
            reused=reused,
            section_by_name=section_by_name,
            book=book,
            out_dir=out_dir,
            prompt_dir=prompt_dir,
            plan=plan,
            source_lang=source_lang,
            target_lang=target_lang,
            chunk_app=chunk_app,
            max_attempts=max_attempts,
            recursion_limit=recursion_limit,
            dry_run=dry_run,
            save=save,
            write_prompts=write_prompts,
            progress=progress,
            clock=clock,
            first_front=first_front,
            first_in_section=first_in_section,
            previous_files=previous_files,
            usage_tokens=usage_tokens,
        )
    )
    return finish_matter(
        plan_state,
        records=records,
        errors=errors,
        usage_tokens=usage_tokens,
        run_seconds=clock() - run_started,
        write_manifest=write_manifest,
    )
__all__ = [
    "MatterChunkResult",
    "MatterResult",
    "ProgressHook",
    "load_matter_manifest",
    "run_matter",
    "write_matter_manifest",
    "write_matter_trace",
]
