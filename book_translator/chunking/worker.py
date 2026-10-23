"""One chapter of the chunking stage: cut it or reuse its previous files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.artifacts import sha256_text
from ..documents.chapters import Chapter
from .artifacts import chunk_filename
from .boundaries import boundary_counts
from .cut import last_preview, line_numbers, preview
from .models import ChapterChunks, ChunkRecord
from .tokenizer import TokenizerChoice


def chunk_one_chapter(
    *,
    chapter: Chapter,
    chapter_index: int,
    text: str,
    spans: list[tuple[int, int, int, str, str]],
    out_dir: Path,
    global_start: int,
    tokenizer: TokenizerChoice,
    dry_run: bool,
) -> tuple[ChapterChunks, list[ChunkRecord]]:
    """Write (or plan) one chapter's chunk files and describe them."""
    newlines = [index for index, char in enumerate(text) if char == "\n"]
    source_hash = sha256_text(text)
    record = ChapterChunks(
        chapter_id=chapter.chapter_id,
        filename=chapter.filename,
        source_path=chapter.path,
        chapter_index=chapter_index,
        title=chapter.title,
        start_line=chapter.start_line,
        end_line=chapter.end_line,
        chunk_count=len(spans),
        char_count=len(text),
        token_count=tokenizer.count_tokens(text),
        source_sha256=source_hash,
    )
    pieces: list[str] = []
    boundary_kinds: list[str] = []
    chunk_records: list[ChunkRecord] = []
    for chunk_index, (start, end, token_count, before, after) in enumerate(spans):
        piece = text[start:end]
        pieces.append(piece)
        boundary_kinds.append(before)
        filename = chunk_filename(
            global_start + chunk_index, chapter.chapter_id, chunk_index
        )
        path = out_dir / filename
        payload = piece.encode("utf-8")
        if not dry_run:
            path.write_bytes(payload)
        start_line, end_line = line_numbers(text, newlines, start, end)
        chunk_records.append(
            ChunkRecord(
                global_index=global_start + chunk_index,
                chapter_index=chapter_index,
                chunk_index=chunk_index,
                chapter_id=chapter.chapter_id,
                chapter_filename=chapter.filename,
                filename=filename,
                path=str(path),
                start_char=start,
                end_char=end,
                start_line=start_line,
                end_line=end_line,
                token_count=token_count,
                bytes=len(payload),
                sha256=sha256_text(piece),
                boundary_before=before,
                boundary_after=after,
                first_line=preview(piece),
                last_line=last_preview(piece),
            )
        )
    reconstructed = "".join(pieces)
    record.reconstructed_sha256 = sha256_text(reconstructed)
    record.reconstructs_byte_for_byte = reconstructed == text
    record.chunk_tokens = sum(span[2] for span in spans)
    record.boundary_counts = boundary_counts(boundary_kinds)
    record.log.append(
        f"{len(spans)} chunk(s) from {len(text):,} chars; "
        f"{record.token_count:,} tokens; "
        f"reconstructs={record.reconstructs_byte_for_byte}"
    )
    return record, chunk_records


def reused_chapter(
    *,
    chapter: Chapter,
    chapter_index: int,
    text: str,
    records: list[dict[str, Any]],
    out_dir: Path,
    global_start: int,
    token_count: int | None = None,
) -> ChapterChunks:
    """Describe a chapter whose chunk files are being reused unchanged."""
    source_hash = sha256_text(text)
    chunks = sorted(records, key=lambda entry: int(entry.get("chunk_index") or 0))
    reconstructed = "".join(
        (out_dir / str(entry["filename"])).read_text(encoding="utf-8")
        for entry in chunks
    )
    record = ChapterChunks(
        chapter_id=chapter.chapter_id,
        filename=chapter.filename,
        source_path=chapter.path,
        chapter_index=chapter_index,
        title=chapter.title,
        start_line=chapter.start_line,
        end_line=chapter.end_line,
        chunk_count=len(chunks),
        char_count=len(text),
        # The full-chapter count is not the sum of its chunks (each chunk is
        # measured on its own, so a boundary can add a token); prefer the count
        # the run that wrote the files recorded.
        token_count=(
            int(token_count)
            if token_count is not None
            else sum(int(entry.get("token_count") or 0) for entry in chunks)
        ),
        chunk_tokens=sum(int(entry.get("token_count") or 0) for entry in chunks),
        source_sha256=source_hash,
        reconstructed_sha256=sha256_text(reconstructed),
        reconstructs_byte_for_byte=reconstructed == text,
        first_global_index=global_start,
        last_global_index=global_start + len(chunks) - 1 if chunks else None,
        skipped=True,
        boundary_counts=boundary_counts(
            str(entry.get("boundary_before") or "") for entry in chunks
        ),
    )
    record.log.append(
        f"reused {len(chunks)} chunk(s); the files still reassemble the chapter"
    )
    return record


__all__ = ["chunk_one_chapter", "reused_chapter"]
