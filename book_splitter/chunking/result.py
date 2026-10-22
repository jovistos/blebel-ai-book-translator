"""Chunk verification and the one result builder for stage 5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..documents.chapters import Chapter
from .boundaries import PARAGRAPH_SAFE, SENTENCE_SAFE, boundary_counts
from .models import ChapterChunks, ChunkingResult, ChunkRecord
from .tokenizer import TokenizerChoice


@dataclass
class ChunkVerification:
    """The bools and counts every result path needs, computed once."""

    slices_ok: bool
    book_ok: bool
    violations: int
    boundary_counts: dict[str, int]


def verify_chunks(
    *,
    records: list[ChapterChunks],
    files: list[ChunkRecord],
    texts: dict[str, str],
    chapters: list[Chapter],
    out_dir: Path,
    settings: dict[str, Any],
    dry_run: bool,
) -> ChunkVerification:
    """Read the written files back; a write that dropped a byte must fail."""
    ordered_files = sorted(files, key=lambda entry: entry.global_index)
    slices_ok = all(record.reconstructs_byte_for_byte for record in records)
    book_ok = slices_ok
    if not dry_run:
        # Read the files back: a write that dropped a byte must fail the run, not
        # only a computation that was correct in memory.
        for entry in ordered_files:
            path = out_dir / entry.filename
            source = texts.get(entry.chapter_filename, "")
            if (
                not path.is_file()
                or path.read_text(encoding="utf-8") != source[entry.start_char : entry.end_char]
            ):
                slices_ok = False
                break
        book_reconstructed = "".join(
            (out_dir / entry.filename).read_text(encoding="utf-8")
            if (out_dir / entry.filename).is_file()
            else ""
            for entry in ordered_files
        )
        book_ok = book_ok and book_reconstructed == "".join(
            texts.get(chapter.filename, "") for chapter in chapters
        )
    violations = sum(
        1 for entry in ordered_files if int(entry.token_count) > int(settings["chunk_size"])
    )
    kinds = boundary_counts(entry.boundary_before for entry in ordered_files)
    return ChunkVerification(
        slices_ok=slices_ok,
        book_ok=book_ok,
        violations=violations,
        boundary_counts=kinds,
    )


def build_chunking_result(
    *,
    records: list[ChapterChunks],
    files: list[ChunkRecord],
    chapters_path: Path,
    out_dir: Path,
    source_file: Any,
    settings: dict[str, Any],
    tokenizer: TokenizerChoice,
    processed_chunks: int,
    errors: list[str],
    dry_run: bool,
    limited: bool,
    timing: dict[str, float],
    verification: ChunkVerification,
    skipped: bool = False,
) -> ChunkingResult:
    """The run as one value; normal, reused and empty runs share it."""
    ordered_files = sorted(files, key=lambda entry: entry.global_index)
    kinds = verification.boundary_counts
    return ChunkingResult(
        ok=bool(records)
        and all(record.ok for record in records)
        and verification.slices_ok
        and verification.book_ok
        and verification.violations == 0
        and not errors,
        chapters_path=str(chapters_path),
        output_dir=str(out_dir),
        source_file=str(source_file or ""),
        chapter_count=len(records),
        ok_count=sum(1 for record in records if record.ok),
        processed_count=sum(1 for record in records if not record.skipped),
        skipped_count=sum(1 for record in records if record.skipped),
        chunk_count=len(ordered_files),
        processed_chunk_count=processed_chunks,
        skipped=skipped,
        dry_run=dry_run,
        limited=limited,
        settings=settings,
        totals={
            "chapters": len(records),
            "chunks": len(ordered_files),
            "processed_chunks": processed_chunks,
            "reused_chunks": len(ordered_files) - processed_chunks,
            "chapter_chars": sum(record.char_count for record in records),
            "chunk_bytes": sum(int(entry.bytes) for entry in ordered_files),
            "chapter_tokens": sum(record.token_count for record in records),
            "chunk_tokens": sum(int(entry.token_count) for entry in ordered_files),
        },
        verification={
            "chunks_are_source_slices": verification.slices_ok,
            "chapters_reconstruct_byte_for_byte": all(
                record.reconstructs_byte_for_byte for record in records
            ),
            "book_reconstructs_byte_for_byte": verification.book_ok,
            "chunk_size_violations": verification.violations,
            "word_breaking_chunks": kinds.get("hard", 0),
            "paragraph_safe_chunks": sum(
                1 for entry in ordered_files if entry.boundary_before in PARAGRAPH_SAFE
            ),
            "sentence_safe_chunks": sum(
                1 for entry in ordered_files if entry.boundary_before in SENTENCE_SAFE
            ),
            "boundary_counts": kinds,
            "tokenizer_fallback": tokenizer.fallback,
            "tokenizer_note": tokenizer.note,
        },
        timing=timing,
        chapters=[record.to_dict() for record in records],
        files=[entry.to_dict() for entry in ordered_files],
        errors=errors,
    )


__all__ = ["ChunkVerification", "build_chunking_result", "verify_chunks"]
