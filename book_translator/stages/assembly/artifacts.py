"""Manifest and trace payloads for the assembly stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from book_translator.core.artifacts import timestamp, write_manifest, write_trace
from book_translator.core.constants import ASSEMBLY_MANIFEST_FILENAME, ASSEMBLY_TRACE_FILENAME

from .models import AssemblyResult


def assembly_manifest_payload(result: AssemblyResult) -> dict[str, Any]:
    """The book-level index body, kept separate from the write."""
    return {
        "stage": "assembly",
        "generated_at": timestamp(),
        "ok": result.ok,
        "dry_run": result.dry_run,
        "limited": result.limited,
        "first_chapters": result.first_chapters,
        "chunks_path": result.chunks_path,
        "translations_path": result.translations_path,
        "output_dir": result.output_dir,
        "chapter_count": result.chapter_count,
        "ok_count": result.ok_count,
        "assembled_count": result.assembled_count,
        "reused_count": result.reused_count,
        "incomplete_count": result.incomplete_count,
        "skipped_count": result.skipped_count,
        "settings": result.settings,
        "totals": result.totals,
        "timing": result.timing,
        "files": result.chapters,
        "errors": result.errors,
    }


def write_assembly_manifest(out_dir: str | Path, result: AssemblyResult) -> Path:
    """Write ``_translated_chapters_manifest.json``: the book-level index."""
    return write_manifest(
        out_dir, ASSEMBLY_MANIFEST_FILENAME, assembly_manifest_payload(result)
    )


def write_assembly_trace(out_dir: str | Path, result: AssemblyResult) -> Path:
    """Write ``_translated_chapters_trace.json``: the same run, for debugging."""
    return write_trace(out_dir, ASSEMBLY_TRACE_FILENAME, result.to_dict())


__all__ = [
    "assembly_manifest_payload",
    "write_assembly_manifest",
    "write_assembly_trace",
]
