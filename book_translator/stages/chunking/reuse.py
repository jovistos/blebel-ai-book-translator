"""The fully-reused chunking run: no chunker, no tokenizer."""

from __future__ import annotations

from book_translator.core.constants import CHUNK_MANIFEST_FILENAME, CHUNK_TRACE_FILENAME
from book_translator.core.run import RunContext
from book_translator.core.store import ArtifactStore
from book_translator.documents.manifest import load_split_manifest

from .artifacts import (
    records_from_dicts,
)
from .boundaries import boundary_counts
from .discover import ChunkingInputs
from .models import ChapterChunks, ChunkingResult, ChunkRecord
from .result import ChunkVerification, build_chunking_result
from .tokenizer import TokenizerChoice
from .worker import reused_chapter


def finish_reused_run(
    inputs: ChunkingInputs,
    *,
    context: RunContext,
    write_manifest: bool,
) -> ChunkingResult:
    records: list[ChapterChunks] = []
    files: list[ChunkRecord] = []
    global_index = 0
    for chapter_index, chapter in enumerate(inputs.chapters):
        previous = inputs.candidates[chapter.filename]
        records.append(
            reused_chapter(
                chapter=chapter,
                chapter_index=chapter_index,
                text=inputs.texts[chapter.filename],
                records=previous,
                out_dir=inputs.out_dir,
                global_start=global_index,
                token_count=inputs.previous_token_counts.get(chapter.filename),
            )
        )
        files.extend(records_from_dicts(previous, chapter, chapter_index))
        global_index += len(previous)
    recorded = inputs.settings.get("tokenizer") or {}
    tokenizer = TokenizerChoice(
        requested=str(recorded.get("requested") or ""),
        used=str(recorded.get("used") or ""),
        tokenizer=None,
        fallback=bool(recorded.get("fallback")),
        note=str(recorded.get("note") or ""),
    )
    ordered = sorted(files, key=lambda entry: entry.global_index)
    verification = ChunkVerification(
        slices_ok=True,
        book_ok=all(record.reconstructs_byte_for_byte for record in records),
        violations=sum(
            1
            for entry in ordered
            if int(entry.token_count) > int(inputs.settings["chunk_size"])
        ),
        boundary_counts=boundary_counts(
            entry.boundary_before for entry in ordered
        ),
    )
    result = build_chunking_result(
        records=records,
        files=files,
        chapters_path=inputs.source_dir,
        out_dir=inputs.out_dir,
        source_file=load_split_manifest(inputs.source_dir).get("source_file", ""),
        settings=inputs.settings,
        tokenizer=tokenizer,
        processed_chunks=0,
        errors=[],
        dry_run=False,
        limited=inputs.limited,
        timing={
            "chapter_seconds": 0.0,
            "chunk_seconds": 0.0,
            "run_seconds": round(context.elapsed(), 4),
        },
        verification=verification,
        skipped=True,
    )
    store = ArtifactStore(
        inputs.out_dir,
        CHUNK_MANIFEST_FILENAME,
        CHUNK_TRACE_FILENAME,
        enabled=write_manifest,
    )
    store.adopt(
        manifest_path=store.existing(CHUNK_MANIFEST_FILENAME),
        trace_path=store.existing(CHUNK_TRACE_FILENAME),
    )
    return context.finish(store.attach(result))


__all__ = ["finish_reused_run"]
