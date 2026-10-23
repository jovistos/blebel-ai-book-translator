"""Discovery, reuse and backend setup for the chunk-glossary stage."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from book_translator.core.constants import (
    DEFAULT_CHUNK_GLOSSARY_CANDIDATES,
    DEFAULT_CHUNK_GLOSSARY_MIN_TERM_CHARS,
    DEFAULT_CHUNK_GLOSSARY_SCORER,
    DEFAULT_CHUNK_GLOSSARY_THRESHOLD,
    DEFAULT_KEEP_ENTITY_TYPES,
)
from book_translator.core.errors import MissingDependencyError, ToolError
from book_translator.core.run import RunContext
from book_translator.documents.chapters import chunk_glossary_dir
from book_translator.documents.chunks import (
    ChunkInput,
    discover_chunk_inputs,
    resolve_glossary_path,
)
from book_translator.stages.dictionary.language import (
    LanguageDetector,
    LanguageGuess,
    detect_language,
)
from book_translator.stages.dictionary.ner import (
    NerBackend,
    backend_device,
    configure_gpu,
    make_backend,
)

from .matching import (
    RapidFuzzMatcher,
    TermMatcher,
    load_glossary,
    normalize_scorer,
    normalize_threshold,
)
from .models import ChunkGlossary, ChunkGlossaryResult
from .reuse import finish_reused
from .terms import _glossary_sha256, _int, _proxy_chapters, _reusable_chunk_glossary


@dataclass
class ChunkSetup:
    inputs: list[ChunkInput]
    limited: bool
    source_dir: Path
    out_dir: Path
    resolved_glossary: Path
    lexicon: Any
    glossary_sha: str
    reused_records: dict[str, ChunkGlossary]
    pending: list[ChunkInput]
    threshold: float
    scorer: str
    candidate_count: int
    wanted_types: tuple[str, ...]
    minimum_chars: int
    matcher: TermMatcher
    guess: Any
    chosen_backend: str
    device: str
    settings: dict[str, Any]
    backend: Any
    backend_load_seconds: float


def prepare_chunks(
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
    context: RunContext,
    clock: Any,
    make_backend_fn=make_backend,
    configure_gpu_fn=configure_gpu,
    detect_language_fn=detect_language,
) -> ChunkSetup | ChunkGlossaryResult:
    threshold = normalize_threshold(threshold)
    scorer = normalize_scorer(scorer)
    candidate_count = max(1, _int(candidates, DEFAULT_CHUNK_GLOSSARY_CANDIDATES))
    minimum_chars = max(0, _int(min_term_chars, DEFAULT_CHUNK_GLOSSARY_MIN_TERM_CHARS))
    if isinstance(keep_types, str):
        keep_types = tuple(part.strip() for part in keep_types.split(",") if part.strip())
    wanted_types = tuple(str(value).strip().upper() for value in keep_types if str(value).strip())
    if not wanted_types:
        wanted_types = tuple(DEFAULT_KEEP_ENTITY_TYPES)

    all_inputs, source_dir = discover_chunk_inputs(chunks_path)
    if chapter_files is not None:
        allowed = {str(name) for name in chapter_files}
        all_inputs = [
            chunk for chunk in all_inputs if chunk.chapter_filename in allowed
        ]
        if not all_inputs:
            raise ToolError(
                "no chunks belong to the requested chapter window "
                f"({', '.join(sorted(allowed)[:5])}…)"
            )
    limited = limit is not None and 0 < _int(limit) < len(all_inputs)
    inputs = all_inputs[: _int(limit)] if limit is not None and _int(limit) > 0 else all_inputs

    out_dir = (
        Path(output_dir).expanduser()
        if output_dir
        else chunk_glossary_dir(source_dir)
    )
    resolved_glossary = resolve_glossary_path(glossary_path, source_dir)
    lexicon = load_glossary(resolved_glossary)
    glossary_sha = _glossary_sha256(resolved_glossary)
    reuse_enabled = bool(skip_if_present and not dry_run)
    reused_records: dict[str, ChunkGlossary] = {}
    if reuse_enabled:
        for chunk in inputs:
            reusable = _reusable_chunk_glossary(
                chunk,
                out_dir,
                glossary_sha256=glossary_sha,
                threshold=threshold,
                scorer=scorer,
                keep_types=wanted_types,
                min_term_chars=minimum_chars,
                drop_temporal_only=bool(drop_temporal_only),
                candidates=candidate_count,
            )
            if reusable is not None:
                reused_records[chunk.filename] = reusable
    pending = [chunk for chunk in inputs if chunk.filename not in reused_records]

    if not pending:
        # Nothing left to tag: report the files already on disk instead of loading
        # a tagger, so a finished (or widened) window needs no model at all.
        return finish_reused(
            inputs=inputs,
            reused_records=reused_records,
            source_dir=source_dir,
            out_dir=out_dir,
            resolved_glossary=resolved_glossary,
            lexicon=lexicon,
            context=context,
            threshold=threshold,
            scorer=scorer,
            candidate_count=candidate_count,
            wanted_types=wanted_types,
            minimum_chars=minimum_chars,
            drop_temporal_only=bool(drop_temporal_only),
            limit=limit,
            chapter_files=chapter_files,
            skip_if_present=skip_if_present,
            write_manifest=write_manifest,
            dry_run=dry_run,
            limited=limited,
        )

    # Built after the resume check on purpose: a fully reused window must not
    # require rapidfuzz or a tagger, only the glossary file it already used.
    # Built before the NER phase on purpose: a missing rapidfuzz or an unreadable
    # glossary is a setup problem, and discovering it after tagging the whole
    # book would waste the expensive half of the run. It is *called* only in
    # phase 2, after every chunk has been tagged.
    matcher = matcher or RapidFuzzMatcher(
        lexicon, scorer=scorer, candidates=candidate_count
    )
    matcher_name = type(matcher).__name__

    # An injected backend already names the language it speaks; asking lingua-py
    # again would only add a dependency to a run that does not need one.
    injected_language = str(getattr(backend, "language", "") or "") if backend is not None else ""
    if injected_language and str(language or "").strip().lower() in {"", "auto"}:
        guess = LanguageGuess(
            language=injected_language,
            confidence=1.0,
            method="backend",
        )
    else:
        guess = detect_language_fn(_proxy_chapters(inputs), language=language, detector=detector)

    backend_load_started = clock()
    resolved_gpu: int | None = None
    if backend is None:
        resolved_gpu = configure_gpu_fn(gpu, use_gpu=use_gpu)
        try:
            backend = make_backend_fn(
                guess.language,
                use_gpu=use_gpu,
                auto_download=auto_download,
                backend=backend_name,
            )
        except MissingDependencyError:
            raise
        except Exception as exc:
            raise ToolError(
                f"could not load the {backend_name or guess.language!r} NER models: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    backend_load_seconds = clock() - backend_load_started
    chosen_backend = backend.name
    device = backend_device(backend)

    settings = {
        "language": guess.language,
        "language_method": guess.method,
        "backend": chosen_backend,
        "device": device,
        "gpu": resolved_gpu,
        "threshold": threshold,
        "scorer": scorer,
        "candidates": candidate_count,
        "matcher": matcher_name,
        "keep_types": list(wanted_types),
        "min_term_chars": minimum_chars,
        "drop_temporal_only": bool(drop_temporal_only),
        "limit": int(limit) if limit is not None else None,
        "chapter_files": list(chapter_files) if chapter_files is not None else None,
        "skip_if_present": bool(skip_if_present),
    }
    return ChunkSetup(
        inputs=inputs,
        limited=limited,
        source_dir=source_dir,
        out_dir=out_dir,
        resolved_glossary=resolved_glossary,
        lexicon=lexicon,
        glossary_sha=glossary_sha,
        reused_records=reused_records,
        pending=pending,
        threshold=threshold,
        scorer=scorer,
        candidate_count=candidate_count,
        wanted_types=wanted_types,
        minimum_chars=minimum_chars,
        matcher=matcher,
        guess=guess,
        chosen_backend=chosen_backend,
        device=device,
        settings=settings,
        backend=backend,
        backend_load_seconds=backend_load_seconds,
    )


__all__ = ["ChunkSetup", "prepare_chunks"]
