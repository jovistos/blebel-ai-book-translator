"""The normal chunking pass: cut, reuse or fail one chapter at a time."""

from __future__ import annotations

from book_translator.core.run import RunContext, run_items
from book_translator.documents.chapters import Chapter

from .artifacts import records_are_ordered, records_from_dicts
from .cut import chunk_text
from .discover import ChunkingInputs
from .models import ChapterChunks, ChunkRecord
from .tokenizer import TokenizerChoice
from .worker import chunk_one_chapter, reused_chapter


def run_chapters(
    inputs: ChunkingInputs,
    *,
    chunker,
    choice: TokenizerChoice,
    chunk_size: int,
    dry_run: bool,
    context: RunContext,
) -> tuple[list[ChapterChunks], list[ChunkRecord], list[str], int, float]:
    """Visit every chapter, collecting records and files in reading order."""
    records: list[ChapterChunks] = []
    files: list[ChunkRecord] = []
    errors: list[str] = []
    global_index = 0
    processed_chunks = 0
    chunk_seconds = 0.0
    positions = {chapter.filename: index for index, chapter in enumerate(inputs.chapters)}

    def work(chapter: Chapter, position: int) -> tuple[ChapterChunks, list[ChunkRecord]]:
        nonlocal global_index, processed_chunks, chunk_seconds
        chapter_index = position - 1
        chapter_started = context.clock()
        reusable_records = (
            inputs.candidates.get(chapter.filename) if inputs.reusable else None
        )
        if reusable_records and records_are_ordered(
            reusable_records, global_index
        ):
            record = reused_chapter(
                chapter=chapter,
                chapter_index=chapter_index,
                text=inputs.texts[chapter.filename],
                records=reusable_records,
                out_dir=inputs.out_dir,
                global_start=global_index,
                token_count=inputs.previous_token_counts.get(chapter.filename),
            )
            chapter_files = records_from_dicts(
                reusable_records, chapter, chapter_index
            )
            global_index += len(reusable_records)
        else:
            try:
                spans = chunk_text(
                    inputs.texts[chapter.filename],
                    chunker=chunker,
                    tokenizer=choice,
                    chunk_size=chunk_size,
                )
                record, chapter_files = chunk_one_chapter(
                    chapter=chapter,
                    chapter_index=chapter_index,
                    text=inputs.texts[chapter.filename],
                    spans=spans,
                    out_dir=inputs.out_dir,
                    global_start=global_index,
                    tokenizer=choice,
                    dry_run=dry_run,
                )
                global_index += record.chunk_count
                processed_chunks += record.chunk_count
                chunk_seconds += context.clock() - chapter_started
            except Exception as exc:  # one bad chapter must not stop the book
                record = ChapterChunks(
                    chapter_id=chapter.chapter_id,
                    filename=chapter.filename,
                    source_path=chapter.path,
                    chapter_index=chapter_index,
                    title=chapter.title,
                    start_line=chapter.start_line,
                    end_line=chapter.end_line,
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
                chapter_files = []
                errors.append(f"{chapter.filename}: {record.error}")
        record.elapsed_seconds = context.clock() - chapter_started
        return record, chapter_files

    def failed(
        chapter: Chapter, exc: Exception
    ) -> tuple[ChapterChunks, list[ChunkRecord]]:
        record = ChapterChunks(
            chapter_id=chapter.chapter_id,
            filename=chapter.filename,
            source_path=chapter.path,
            chapter_index=positions.get(chapter.filename, 0),
            title=chapter.title,
            start_line=chapter.start_line,
            end_line=chapter.end_line,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )
        errors.append(f"{chapter.filename}: {record.error}")
        return record, []

    outcomes = run_items(
        inputs.chapters,
        work,
        on_error=failed,
        progress=lambda record, done, total: context.emit(
            record.to_dict(), done, total
        ),
    )
    for outcome in outcomes:
        if outcome.value is None:  # pragma: no cover - failed always supplies a tuple
            continue
        record, chapter_files = outcome.value
        records.append(record)
        files.extend(chapter_files)
    return (
        records,
        files,
        errors,
        processed_chunks,
        chunk_seconds,
    )


__all__ = ["run_chapters"]
