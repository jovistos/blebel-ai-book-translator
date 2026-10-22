"""The chunk-glossary result value and its one builder."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from .models import ChunkGlossary, ChunkGlossaryResult


def build_chunk_glossary_result(
    *,
    records: Sequence[ChunkGlossary],
    chunks_path: str,
    glossary_path: str,
    output_dir: str,
    language: dict[str, Any],
    backend: str,
    device: str,
    entry_count: int,
    totals: dict[str, int],
    phases: list[dict[str, Any]],
    timing: dict[str, float],
    errors: list[str],
    settings: dict[str, Any],
    dry_run: bool,
    limited: bool,
    processed_count: int,
    skipped_count: int,
    skipped: bool = False,
) -> ChunkGlossaryResult:
    """The run as one value; the fully-reused window sets ``skipped``."""
    return ChunkGlossaryResult(
        ok=True
        if skipped
        else bool(records) and all(record.ok for record in records) and not errors,
        chunks_path=chunks_path,
        glossary_path=glossary_path,
        output_dir=output_dir,
        language=language,
        backend=backend,
        device=device,
        chunk_count=len(records),
        ok_count=sum(1 for record in records if record.ok),
        processed_count=processed_count,
        skipped_count=skipped_count,
        entry_count=entry_count,
        entity_count=totals.get("entity_mentions", 0),
        term_count=totals.get("terms", 0),
        matched_term_count=totals.get("matched_terms", 0),
        unmatched_term_count=totals.get("unmatched_terms", 0),
        matched_chunk_count=totals.get("matched_chunks", 0),
        dry_run=dry_run,
        limited=limited,
        settings=settings,
        totals=totals,
        phases=phases,
        timing=timing,
        chunks=[record.record() for record in records],
        errors=errors,
    )


__all__ = ["build_chunk_glossary_result"]


def finish_chunk_glossary(
    setup,
    *,
    records,
    errors,
    ner_phase_seconds: float,
    match_phase_seconds: float,
    context,
    write_manifest: bool,
    dry_run: bool,
) -> ChunkGlossaryResult:
    from .artifacts import (
        _totals,
        write_chunk_glossary_manifest,
        write_chunk_glossary_trace,
    )

    totals = _totals(records)
    result = build_chunk_glossary_result(
        records=records,
        chunks_path=str(setup.source_dir),
        glossary_path=str(setup.resolved_glossary),
        output_dir=str(setup.out_dir),
        language=setup.guess.to_dict(),
        backend=setup.chosen_backend,
        device=setup.device,
        entry_count=len(setup.lexicon),
        totals=totals,
        phases=[
            {
                "name": "ner",
                "chunks": len(setup.pending),
                "entities": totals.get("entity_mentions", 0),
                "terms": totals.get("terms", 0),
                "seconds": round(ner_phase_seconds, 4),
            },
            {
                "name": "match",
                "chunks": len(records),
                "matched_terms": totals.get("matched_terms", 0),
                "unmatched_terms": totals.get("unmatched_terms", 0),
                "seconds": round(match_phase_seconds, 4),
            },
        ],
        timing={
            "run_seconds": round(context.elapsed(), 4),
            "backend_load_seconds": round(setup.backend_load_seconds, 4),
            "ner_phase_seconds": round(ner_phase_seconds, 4),
            "match_phase_seconds": round(match_phase_seconds, 4),
            "milliseconds_per_chunk": round(
                (1000.0 * ner_phase_seconds / len(records)) if records else 0.0, 3
            ),
        },
        errors=errors,
        settings=setup.settings,
        dry_run=dry_run,
        limited=setup.limited,
        processed_count=len(setup.pending),
        skipped_count=len(setup.reused_records),
    )
    if write_manifest and not dry_run:
        result.manifest_path = str(
            write_chunk_glossary_manifest(setup.out_dir, result, records)
        )
        result.trace_path = str(
            write_chunk_glossary_trace(setup.out_dir, result, records)
        )
    return cast(ChunkGlossaryResult, context.finish(result))
