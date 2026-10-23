"""Totals and artifacts for the chunk-glossary stage."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from book_translator.core.artifacts import timestamp, write_json, write_manifest, write_trace
from book_translator.core.constants import (
    CHUNK_GLOSSARY_MANIFEST_FILENAME,
    CHUNK_GLOSSARY_TRACE_FILENAME,
    DEFAULT_CHUNK_GLOSSARY_CANDIDATES,
    DEFAULT_CHUNK_GLOSSARY_MIN_TERM_CHARS,
    DEFAULT_CHUNK_GLOSSARY_SCORER,
    DEFAULT_CHUNK_GLOSSARY_THRESHOLD,
    DEFAULT_KEEP_ENTITY_TYPES,
)
from book_translator.documents.chunks import chunk_glossary_filename

from .matching import match_key
from .models import ChunkGlossary, ChunkGlossaryResult


def _totals(records: Sequence[ChunkGlossary]) -> dict[str, int]:
    """Book-level counts, derived once from the per-chunk records."""
    matched_keys: set[str] = set()
    for record in records:
        for term in record.matched_terms:
            key = match_key(str(term.get("glossary_term") or ""))
            if key:
                matched_keys.add(key)
    return {
        "chunks": len(records),
        "ok_chunks": sum(1 for record in records if record.ok),
        "failed_chunks": sum(1 for record in records if not record.ok),
        "entity_mentions": sum(record.entity_count for record in records),
        "terms": sum(record.term_count for record in records),
        "matched_terms": sum(record.matched_count for record in records),
        "unmatched_terms": sum(record.unmatched_count for record in records),
        "matched_chunks": sum(1 for record in records if record.matched_count),
        "unique_matched_glossary_terms": len(matched_keys),
    }


# --------------------------------------------------------------------------- #
# writing the artifacts
# --------------------------------------------------------------------------- #
def chunk_glossary_payload(
    record: ChunkGlossary,
    *,
    language: str = "",
    backend: str = "",
    threshold: float = DEFAULT_CHUNK_GLOSSARY_THRESHOLD,
    scorer: str = DEFAULT_CHUNK_GLOSSARY_SCORER,
    glossary_path: str = "",
    glossary_sha256: str = "",
    keep_types: Sequence[str] = DEFAULT_KEEP_ENTITY_TYPES,
    min_term_chars: int = DEFAULT_CHUNK_GLOSSARY_MIN_TERM_CHARS,
    drop_temporal_only: bool = False,
    candidates: int = DEFAULT_CHUNK_GLOSSARY_CANDIDATES,
) -> dict[str, Any]:
    """The per-chunk file: the chunk's identity, its terms and their matches.

    The settings that produced the terms are recorded with them, so a later
    ``--skip-if-present`` run can tell whether the file still answers the same
    question instead of only whether the chunk and glossary hashes match.
    """
    return {
        "stage": "chunk-glossary",
        "generated_at": timestamp(),
        "chunk": record.chunk.to_dict(),
        "language": language,
        "backend": backend,
        "threshold": threshold,
        "scorer": scorer,
        "glossary_path": glossary_path,
        "glossary_sha256": glossary_sha256,
        "keep_types": sorted(
            {str(value).strip().upper() for value in keep_types if str(value).strip()}
        ),
        "min_term_chars": int(min_term_chars),
        "drop_temporal_only": bool(drop_temporal_only),
        "candidates": int(candidates),
        "counts": {
            "entities": record.entity_count,
            "terms": record.term_count,
            "matched": record.matched_count,
            "unmatched": record.unmatched_count,
        },
        "terms": [term.to_dict() for term in record.terms],
        "matched_terms": list(record.matched_terms),
    }


def write_chunk_glossary(
    out_dir: str | Path,
    record: ChunkGlossary,
    *,
    language: str = "",
    backend: str = "",
    threshold: float = DEFAULT_CHUNK_GLOSSARY_THRESHOLD,
    scorer: str = DEFAULT_CHUNK_GLOSSARY_SCORER,
    glossary_path: str = "",
    glossary_sha256: str = "",
    keep_types: Sequence[str] = DEFAULT_KEEP_ENTITY_TYPES,
    min_term_chars: int = DEFAULT_CHUNK_GLOSSARY_MIN_TERM_CHARS,
    drop_temporal_only: bool = False,
    candidates: int = DEFAULT_CHUNK_GLOSSARY_CANDIDATES,
) -> Path:
    """Write one chunk's glossary next to the others, under the chunk's name."""
    path = Path(out_dir) / chunk_glossary_filename(record.chunk.filename)
    payload = chunk_glossary_payload(
        record,
        language=language,
        backend=backend,
        threshold=threshold,
        scorer=scorer,
        glossary_path=glossary_path,
        glossary_sha256=glossary_sha256,
        keep_types=keep_types,
        min_term_chars=min_term_chars,
        drop_temporal_only=drop_temporal_only,
        candidates=candidates,
    )
    return write_json(path, payload)


def write_chunk_glossary_manifest(
    out_dir: str | Path,
    result: ChunkGlossaryResult,
    records: Sequence[ChunkGlossary] | None = None,
) -> Path:
    """Write ``_chunk_glossary_manifest.json``: the book-level index."""
    payload = {
        "stage": "chunk-glossary",
        "ok": result.ok,
        "dry_run": result.dry_run,
        "limited": result.limited,
        "skip_if_present": bool(result.settings.get("skip_if_present")),
        "chunks_path": result.chunks_path,
        "glossary_path": result.glossary_path,
        "output_dir": result.output_dir,
        "language": result.language,
        "backend": result.backend,
        "device": result.device,
        "settings": result.settings,
        "chunk_count": result.chunk_count,
        "ok_count": result.ok_count,
        "processed_count": result.processed_count,
        "skipped_count": result.skipped_count,
        "entry_count": result.entry_count,
        "totals": result.totals,
        "phases": result.phases,
        "timing": result.timing,
        "chunks": [record.record() for record in records] if records is not None else result.chunks,
        "errors": result.errors,
    }
    return write_manifest(out_dir, CHUNK_GLOSSARY_MANIFEST_FILENAME, payload)


def write_chunk_glossary_trace(
    out_dir: str | Path,
    result: ChunkGlossaryResult,
    records: Sequence[ChunkGlossary] | None = None,
) -> Path:
    """Write the audit trail: the same run plus each chunk's terms and log."""
    payload = result.to_dict()
    if records is not None:
        payload["detail"] = [record.to_dict() for record in records]
    return write_trace(out_dir, CHUNK_GLOSSARY_TRACE_FILENAME, payload)



__all__ = [
    "_totals",
    "chunk_glossary_payload",
    "write_chunk_glossary",
    "write_chunk_glossary_manifest",
    "write_chunk_glossary_trace",
]
