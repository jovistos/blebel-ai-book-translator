"""Manifest and trace payloads for the EPUB stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from book_translator.core.artifacts import timestamp, write_manifest, write_trace
from book_translator.core.constants import EPUB_MANIFEST_FILENAME, EPUB_TRACE_FILENAME

from .result import EpubResult


def epub_manifest_payload(result: EpubResult) -> dict[str, Any]:
    """The package manifest body, kept separate from the write."""
    return {
        "stage": "epub",
        "generated_at": timestamp(),
        "ok": result.ok,
        "dry_run": result.dry_run,
        "chapters_path": result.chapters_path,
        "output_path": result.output_path,
        "bytes": result.bytes,
        "sha256": result.sha256,
        "chars": result.chars,
        "document_count": result.document_count,
        "chapter_count": result.chapter_count,
        "image_count": result.image_count,
        "toc_dropped": result.toc_dropped,
        "cover": result.cover,
        "metadata": {
            "title": result.title,
            "author": result.author,
            "language": result.language,
            "identifier": result.identifier,
            "modified": result.modified,
            "description": result.description,
            "publisher": result.publisher,
        },
        "spine": result.spine,
        "images": result.images,
        "verification": result.verification,
        "warnings": result.warnings,
        "errors": result.errors,
    }


def write_epub_manifest(output_path: Path, result: EpubResult) -> Path:
    return write_manifest(
        output_path.parent, EPUB_MANIFEST_FILENAME, epub_manifest_payload(result)
    )


def write_epub_trace(output_path: Path, result: EpubResult) -> Path:
    return write_trace(output_path.parent, EPUB_TRACE_FILENAME, result.to_dict())


__all__ = ["epub_manifest_payload", "write_epub_manifest", "write_epub_trace"]
