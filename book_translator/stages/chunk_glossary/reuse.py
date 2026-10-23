"""The fully-reused chunk-glossary window: no tagger, no matcher."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from book_translator.core.run import RunContext
from book_translator.documents.chunks import ChunkInput
from book_translator.documents.manifest import load_chunk_glossary_manifest

from .artifacts import (
    _totals,
    write_chunk_glossary_manifest,
    write_chunk_glossary_trace,
)
from .models import ChunkGlossary, ChunkGlossaryResult
from .result import build_chunk_glossary_result


def finish_reused(
    *,
    inputs: Sequence[ChunkInput],
    reused_records: dict[str, ChunkGlossary],
    source_dir: Path,
    out_dir: Path,
    resolved_glossary: Path,
    lexicon: object,
    context: RunContext,
    threshold: float,
    scorer: str,
    candidate_count: int,
    wanted_types: Sequence[str],
    minimum_chars: int,
    drop_temporal_only: bool,
    limit: int | None,
    chapter_files: Sequence[str] | None,
    skip_if_present: bool,
    write_manifest: bool,
    dry_run: bool,
    limited: bool,
) -> ChunkGlossaryResult:
    previous = load_chunk_glossary_manifest(out_dir)
    records = [reused_records[chunk.filename] for chunk in inputs]
    totals = _totals(records)
    settings = {
        "language": previous.get("language") or {},
        "backend": str(previous.get("backend") or ""),
        "device": str(previous.get("device") or ""),
        "threshold": threshold,
        "scorer": scorer,
        "candidates": candidate_count,
        "keep_types": list(wanted_types),
        "min_term_chars": minimum_chars,
        "drop_temporal_only": bool(drop_temporal_only),
        "limit": int(limit) if limit is not None else None,
        "chapter_files": list(chapter_files) if chapter_files is not None else None,
        "skip_if_present": bool(skip_if_present),
    }
    result = build_chunk_glossary_result(
        records=records,
        chunks_path=str(source_dir),
        glossary_path=str(resolved_glossary),
        output_dir=str(out_dir),
        language=dict(previous.get("language") or {}),
        backend=str(previous.get("backend") or ""),
        device=str(previous.get("device") or ""),
        entry_count=len(lexicon),  # type: ignore[arg-type]
        totals=totals,
        phases=[
            {"name": "ner", "chunks": 0, "entities": 0, "terms": 0, "seconds": 0.0},
            {
                "name": "match",
                "chunks": 0,
                "matched_terms": 0,
                "unmatched_terms": 0,
                "seconds": 0.0,
            },
        ],
        timing={
            "run_seconds": round(context.elapsed(), 4),
            "backend_load_seconds": 0.0,
            "ner_phase_seconds": 0.0,
            "match_phase_seconds": 0.0,
            "milliseconds_per_chunk": 0.0,
        },
        errors=[],
        settings=settings,
        dry_run=dry_run,
        limited=limited,
        processed_count=0,
        skipped_count=len(records),
        skipped=True,
    )
    if write_manifest and not dry_run:
        result.manifest_path = str(
            write_chunk_glossary_manifest(out_dir, result, records)
        )
        result.trace_path = str(write_chunk_glossary_trace(out_dir, result, records))
    return result


__all__ = ["finish_reused"]
