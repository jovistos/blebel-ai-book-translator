"""Stage 5: cut each chapter into reversible, translation-sized chunks.

The public surface is intentionally small and has two halves:

* the write side — :func:`run_chunking` (and :func:`build_chunker` /
  :func:`chunk_text` for a single string), which reads a chapters folder and
  writes ``05-chunks/`` plus ``_chunk_manifest.json``;
* the read side — :func:`load_chunk_manifest`, :func:`chunk_files`,
  :func:`reconstruct_chapter` and :func:`reconstruct_book`, which put the text
  back together without guessing a separator.

Chonkie is imported lazily, so importing :mod:`book_translator` (or running a
split-only command) does not require the chunking dependency.
"""

from .boundaries import BOUNDARY_KINDS, boundary_kind
from .runner import (
    BOUNDARY_ORDER,
    ChapterChunks,
    ChunkingResult,
    ChunkRecord,
    build_chunker,
    chapter_slug,
    chunk_filename,
    chunk_files,
    chunk_text,
    load_chunk_manifest,
    reconstruct_book,
    reconstruct_chapter,
    run_chunking,
    write_chunk_manifest,
    write_chunk_trace,
)
from .tokenizer import TokenizerChoice, resolve_tokenizer

__all__ = [
    "BOUNDARY_KINDS",
    "BOUNDARY_ORDER",
    "ChapterChunks",
    "ChunkRecord",
    "ChunkingResult",
    "TokenizerChoice",
    "boundary_kind",
    "build_chunker",
    "chapter_slug",
    "chunk_filename",
    "chunk_files",
    "chunk_text",
    "load_chunk_manifest",
    "reconstruct_book",
    "reconstruct_chapter",
    "resolve_tokenizer",
    "run_chunking",
    "write_chunk_manifest",
    "write_chunk_trace",
]
