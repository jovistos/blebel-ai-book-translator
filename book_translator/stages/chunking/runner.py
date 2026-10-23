"""Stage 5: cut every chapter into translation-sized chunks.

The stage is deliberately a different kind of cut from stage 1. Stage 1 never
reads the body and slices it by line; stage 5 reads a chapter as text and asks
chonkie's recursive chunker where to cut, so a chunk never ends in the middle of a
paragraph or a sentence when the requested length allows it. The preference order
is paragraph, sentence, line, whitespace, hard character cut.

Two properties matter more than any of that:

* **Every chunk is an exact slice of its chapter.** Concatenating a chapter's
  chunks in ``chunk_index`` order gives the chapter back byte for byte, and
  concatenating the whole folder in ``global_index`` order gives the book back.
  The run proves both and records the result in the manifest.
* **The folder is self-describing.** File names sort into reading order, and
  ``_chunk_manifest.json`` records for every chunk which chapter it belongs to,
  where in that chapter it starts and ends, its line span, its token count and
  the boundary kind on either side.

The reusable pieces live next to this module: :mod:`.cut` performs the cut,
:mod:`.worker` handles one chapter, :mod:`.result` verifies and builds the result,
and :mod:`.artifacts` owns the previous-run records and the manifest.
"""

from __future__ import annotations

import time

from book_translator.core.constants import (
    CHUNK_MANIFEST_FILENAME,
    CHUNK_TRACE_FILENAME,
    DEFAULT_CHUNK_TOKENIZER,
    DEFAULT_CHUNK_TOKENS,
)
from book_translator.core.run import RunContext
from book_translator.core.store import ArtifactStore
from book_translator.documents.chunks import chunk_files
from book_translator.documents.manifest import load_chunk_manifest, load_split_manifest

from .artifacts import (
    chapter_slug,
    chunk_filename,
    chunk_manifest_payload,
    reconstruct_book,
    reconstruct_chapter,
    write_chunk_manifest,
    write_chunk_trace,
)
from .cut import _image_safe_spans, build_chunker, chunk_text, sentence_delimiters
from .discover import discover_chunking
from .models import (
    BOUNDARY_ORDER,
    ChapterChunks,
    ChunkingResult,
    ChunkRecord,
    Clock,
    ProgressHook,
)
from .normal import run_chapters
from .result import build_chunking_result, verify_chunks
from .reuse import finish_reused_run


def run_chunking(
    chapters_path: str,
    output_dir: str | None = None,
    *,
    chunk_size: int = DEFAULT_CHUNK_TOKENS,
    tokenizer: str | None = DEFAULT_CHUNK_TOKENIZER,
    skip_if_present: bool = False,
    dry_run: bool = False,
    limit: int | None = None,
    write_manifest: bool = True,
    progress: ProgressHook | None = None,
    clock: Clock = time.perf_counter,
) -> ChunkingResult:
    """Chunk every chapter of a stage-1 (or preprocessed) folder, in order.

    ``chunk_size`` is in tokens and defaults to
    :data:`~book_translator.core.constants.DEFAULT_CHUNK_TOKENS` (1000). A chapter
    shorter than that becomes one chunk; a longer one is cut at the strongest
    boundary that keeps the chunk under the limit.

    ``skip_if_present`` reuses a chapter whose recorded chunk files still
    concatenate to the chapter's current text — and, when every chapter can be
    reused, returns before loading chonkie or a tokenizer at all. The reuse is
    refused when the previous manifest was written with a different chunk length
    or with the approximate fallback tokenizer, because then its boundaries
    cannot be trusted to respect the current limit.

    ``dry_run`` computes and verifies every cut but writes no chunk, manifest or
    trace, which is how a chunk length is priced before a long run.
    """
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be greater than 0")

    inputs = discover_chunking(
        chapters_path,
        output_dir,
        chunk_size=chunk_size,
        tokenizer=tokenizer,
        skip_if_present=skip_if_present,
        dry_run=dry_run,
        limit=limit,
    )
    context = RunContext(
        clock=clock,
        dry_run=dry_run,
        limited=inputs.limited,
        progress=progress,
    )
    if inputs.all_reusable:
        return finish_reused_run(inputs, context=context, write_manifest=write_manifest)

    chunker, choice = build_chunker(int(chunk_size), tokenizer)
    inputs.settings["tokenizer"] = choice.to_settings()
    if not dry_run:
        inputs.out_dir.mkdir(parents=True, exist_ok=True)

    chapters = inputs.chapters
    texts = inputs.texts
    settings = inputs.settings
    out_dir = inputs.out_dir
    source_dir = inputs.source_dir
    limited = inputs.limited

    records, files, errors, processed_chunks, chunk_seconds = run_chapters(
        inputs,
        chunker=chunker,
        choice=choice,
        chunk_size=int(chunk_size),
        dry_run=dry_run,
        context=context,
    )

    verification = verify_chunks(
        records=records,
        files=files,
        texts=texts,
        chapters=chapters,
        out_dir=out_dir,
        settings=settings,
        dry_run=dry_run,
    )
    result = build_chunking_result(
        records=records,
        files=files,
        chapters_path=source_dir,
        out_dir=out_dir,
        source_file=load_split_manifest(source_dir).get("source_file", ""),
        settings=settings,
        tokenizer=choice,
        processed_chunks=processed_chunks,
        errors=errors,
        dry_run=dry_run,
        limited=limited,
        timing={
            "chapter_seconds": round(
                sum(record.elapsed_seconds for record in records), 4
            ),
            "chunk_seconds": round(chunk_seconds, 4),
            "run_seconds": round(context.elapsed(), 4),
        },
        verification=verification,
    )
    store = ArtifactStore(
        out_dir,
        CHUNK_MANIFEST_FILENAME,
        CHUNK_TRACE_FILENAME,
        enabled=write_manifest and not dry_run,
    )
    return context.finish(store.write_result(result, chunk_manifest_payload(result)))


__all__ = [
    "BOUNDARY_ORDER",
    "ChapterChunks",
    "ChunkRecord",
    "ChunkingResult",
    "Clock",
    "ProgressHook",
    "build_chunker",
    "chapter_slug",
    "chunk_filename",
    "chunk_files",
    "chunk_text",
    "load_chunk_manifest",
    "reconstruct_book",
    "reconstruct_chapter",
    "run_chunking",
    "_image_safe_spans",
    "sentence_delimiters",
    "write_chunk_manifest",
    "write_chunk_trace",
]
