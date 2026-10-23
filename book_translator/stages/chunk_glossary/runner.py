"""Stage 6: attach the book glossary to every chunk, in two strict phases.

Stage 5 produced a flat folder of translation-sized chunks; stage 4 produced one
book-wide dictionary. This stage is the join between them, and the whole reason it
exists is that a translator sending chunk 137 must know which of the book's locked
renderings that chunk actually mentions.

The run is deliberately **two phases**, in this order:

```
phase 1 — NER over every chunk                         (one complete pass)
    05-chunks/00000-…  -> tagger -> ["Džordž"(PER), "London"(LOC), …]
    05-chunks/00001-…  -> tagger -> ["Dženi"(PER), "Beograd"(LOC), …]
    …
    no glossary is read yet; the phase is complete before phase 2 starts

phase 2 — fuzzy retrieval for every collected term      (one complete pass)
    "Džordž" -> glossary entry "Džordž"      100.0  -> character / Džordž
    "Dženi"  -> glossary entry "Dženi"       100.0  -> character / Dženi
    "Mithril"-> best 71.4, below threshold          -> unmatched
    …
    06-chunk-glossary/<chunk>.json  (the chunk's own name, one file per chunk)
```

That ordering is the contract, not an implementation detail: the NER pass must
finish for **all** chunks before the first lookup, so the stage has one complete
picture of the book's chunk-level entities before it decides what any of them
means. The tests pin it by recording the order of the two injected doubles.

The tagger is stage 3's: the same :mod:`book_translator.stages.dictionary.ner` backends,
chosen by language, used here to *find* terms rather than to gate sentences. The
retrieval is ``rapidfuzz`` and nothing else: stage 4 already locked each
rendering, so a chunk term only needs the entry that names it. The score is a
percentage on rapidfuzz's 0..100 scale and is a command-line option (95 by
default).

The output folder mirrors the chunk folder: ``05-chunks/00001-x-0000.md`` gets
``06-chunk-glossary/00001-x-0000.md.json``. The file carries the chunk's identity
(global order, chapter, character and line spans), every NER term with its match
or near miss, and a ``matched_terms`` list a translation prompt can consume
directly.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

from book_translator.core.constants import (
    DEFAULT_CHUNK_GLOSSARY_CANDIDATES,
    DEFAULT_CHUNK_GLOSSARY_MIN_TERM_CHARS,
    DEFAULT_CHUNK_GLOSSARY_SCORER,
    DEFAULT_CHUNK_GLOSSARY_THRESHOLD,
    DEFAULT_KEEP_ENTITY_TYPES,
)
from book_translator.core.run import RunContext
from book_translator.documents.chunks import (
    ChunkInput,
    chunk_glossary_filename,
    discover_chunk_inputs,
    resolve_chunks_dir,
    resolve_glossary_path,
)
from book_translator.documents.manifest import load_chunk_glossary_manifest
from book_translator.stages.dictionary.language import LanguageDetector, detect_language
from book_translator.stages.dictionary.ner import (
    NerBackend,
    configure_gpu,
    make_backend,
)

from .artifacts import (
    chunk_glossary_payload,
    write_chunk_glossary,
    write_chunk_glossary_manifest,
    write_chunk_glossary_trace,
)
from .matching import (
    TermMatcher,
)
from .models import (
    ChunkGlossary,
    ChunkGlossaryResult,
    Clock,
    EntityTerm,
    ProgressHook,
    TermMatch,
)
from .phases import run_phases
from .result import finish_chunk_glossary
from .setup import prepare_chunks
from .terms import (
    aggregate_entities,
    collect_matched_terms,
    match_term,
)


def run_chunk_glossary(
    chunks_path: str,
    output_dir: str | None = None,
    *,
    glossary_path: str | Path | None = None,
    language: str = "auto",
    detector: LanguageDetector | None = None,
    backend: NerBackend | None = None,
    backend_name: str | None = None,
    matcher: TermMatcher | None = None,
    gpu: int | None = None,
    use_gpu: bool = True,
    auto_download: bool = True,
    keep_types: Sequence[str] = DEFAULT_KEEP_ENTITY_TYPES,
    min_term_chars: int = DEFAULT_CHUNK_GLOSSARY_MIN_TERM_CHARS,
    drop_temporal_only: bool = False,
    threshold: float = DEFAULT_CHUNK_GLOSSARY_THRESHOLD,
    scorer: str = DEFAULT_CHUNK_GLOSSARY_SCORER,
    candidates: int = DEFAULT_CHUNK_GLOSSARY_CANDIDATES,
    dry_run: bool = False,
    limit: int | None = None,
    chapter_files: Sequence[str] | None = None,
    skip_if_present: bool = False,
    write_manifest: bool = True,
    progress: ProgressHook | None = None,
    clock: Clock = time.perf_counter,
) -> ChunkGlossaryResult:
    """Build the per-chunk glossary in two passes: all NER, then all retrieval.

    ``backend`` and ``matcher`` are injectable, which is how the offline tests run
    the whole stage without torch, a model download, or a real glossary; a real
    run builds the tagger from ``language`` (detected from the chunks when
    ``language="auto"``) and the rapidfuzz matcher from ``glossary_path``.

    ``dry_run`` still runs both phases — the NER output *is* the artifact, so there
    is nothing to validate without tagging — but writes no per-chunk file,
    manifest or trace. ``chapter_files`` restricts the run to chunks belonging to
    those chapter filenames (the chapter-aligned window); ``limit`` then caps the
    filtered list, which is the cheap way to inspect a book before a full run.
    ``skip_if_present`` reuses a per-chunk file whose recorded chunk hash,
    threshold, scorer and glossary hash still match, and returns before loading a
    tagger when every chunk is covered; growing a chapter window therefore only
    tags the new chunks.
    """
    context = RunContext(clock=clock, dry_run=dry_run, limited=False, progress=progress)
    prepared = prepare_chunks(
        chunks_path,
        output_dir,
        glossary_path=glossary_path,
        language=language,
        detector=detector,
        backend=backend,
        backend_name=backend_name,
        matcher=matcher,
        gpu=gpu,
        use_gpu=use_gpu,
        auto_download=auto_download,
        keep_types=keep_types,
        min_term_chars=min_term_chars,
        drop_temporal_only=drop_temporal_only,
        threshold=threshold,
        scorer=scorer,
        candidates=candidates,
        dry_run=dry_run,
        limit=limit,
        chapter_files=chapter_files,
        skip_if_present=skip_if_present,
        write_manifest=write_manifest,
        context=context,
        clock=clock,
        make_backend_fn=make_backend,
        configure_gpu_fn=configure_gpu,
        detect_language_fn=detect_language,
    )
    if isinstance(prepared, ChunkGlossaryResult):
        return prepared
    setup = prepared
    inputs = setup.inputs
    context.limited = setup.limited
    out_dir = setup.out_dir
    resolved_glossary = setup.resolved_glossary
    glossary_sha = setup.glossary_sha
    reused_records = setup.reused_records
    pending = setup.pending
    threshold = setup.threshold
    scorer = setup.scorer
    candidate_count = setup.candidate_count
    wanted_types = setup.wanted_types
    minimum_chars = setup.minimum_chars
    matcher = setup.matcher
    backend = setup.backend
    guess = setup.guess
    chosen_backend = setup.chosen_backend

    records, errors, ner_phase_seconds, match_phase_seconds = run_phases(
        inputs=inputs,
        pending=pending,
        reused_records=reused_records,
        backend=backend,
        matcher=matcher,
        out_dir=out_dir,
        dry_run=dry_run,
        threshold=threshold,
        scorer=scorer,
        resolved_glossary=resolved_glossary,
        glossary_sha=glossary_sha,
        wanted_types=wanted_types,
        minimum_chars=minimum_chars,
        drop_temporal_only=bool(drop_temporal_only),
        candidate_count=candidate_count,
        language=guess.language,
        chosen_backend=chosen_backend,
        context=context,
        clock=clock,
    )

    return finish_chunk_glossary(
        setup,
        records=records,
        errors=errors,
        ner_phase_seconds=ner_phase_seconds,
        match_phase_seconds=match_phase_seconds,
        context=context,
        write_manifest=write_manifest,
        dry_run=dry_run,
    )


__all__ = [
    "ChunkGlossary",
    "ChunkGlossaryResult",
    "ChunkInput",
    "Clock",
    "EntityTerm",
    "ProgressHook",
    "TermMatch",
    "aggregate_entities",
    "chunk_glossary_filename",
    "chunk_glossary_payload",
    "collect_matched_terms",
    "discover_chunk_inputs",
    "load_chunk_glossary_manifest",
    "match_term",
    "resolve_chunks_dir",
    "resolve_glossary_path",
    "run_chunk_glossary",
    "write_chunk_glossary",
    "write_chunk_glossary_manifest",
    "write_chunk_glossary_trace",
]
