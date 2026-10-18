"""Stage 5: cut every chapter into translation-sized chunks.

The stage is deliberately a different kind of cut from stage 1. Stage 1 never
reads the body and slices it by line; stage 5 reads a chapter as text and asks
chonkie's recursive chunker where to cut, so a chunk never ends in the middle of a
paragraph or a sentence when the requested length allows it. The preference order
is:

    paragraph  ->  sentence  ->  line  ->  whitespace  ->  hard character cut

and it is encoded in chonkie ``RecursiveRules`` rather than in a hand-written
splitter, because chonkie's chunker already carries the offsets and token counts a
reversible cut needs.

Two properties matter more than any of that:

* **Every chunk is an exact slice of its chapter.** The text written to disk is
  ``chapter_text[start_char:end_char]``; nothing is trimmed, re-wrapped or
  normalised. Concatenating a chapter's chunks in ``chunk_index`` order gives the
  chapter back byte for byte, and concatenating the whole folder in
  ``global_index`` order gives the book back. The run proves both and records the
  result in the manifest.
* **The folder is self-describing.** File names sort into reading order, and
  ``_chunk_manifest.json`` records for every chunk which chapter it belongs to,
  where in that chapter it starts and ends, its line span, its token count and
  the boundary kind on either side. :func:`reconstruct_chapter` and
  :func:`reconstruct_book` are the read side of that record.

Chunk length is a token count (default :data:`DEFAULT_CHUNK_TOKENS`). When the
configured tokenizer cannot be loaded the stage falls back to a local estimate and
says so, rather than failing a run over a vocabulary download.
"""

from __future__ import annotations

import bisect
import contextlib
import hashlib
import json
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.constants import (
    CHUNK_INDEX_DIGITS,
    CHUNK_MANIFEST_FILENAME,
    CHUNK_ORDER_DIGITS,
    CHUNK_TRACE_FILENAME,
    DEFAULT_CHUNK_TOKENIZER,
    DEFAULT_CHUNK_TOKENS,
)
from ..core.errors import MissingDependencyError, ToolError
from ..documents.chapters import (
    Chapter,
    chapters_dir,
    chunks_dir,
    discover_chapters,
    load_split_manifest,
)
from ..documents.io import load_lines
from .boundaries import (
    PARAGRAPH_SAFE,
    SENTENCE_SAFE,
    boundary_counts,
    boundary_kind,
)
from .tokenizer import TokenizerChoice, resolve_tokenizer

#: The preference order, spelled out in the manifest.
BOUNDARY_ORDER: tuple[str, ...] = ("paragraph", "sentence", "line", "word", "hard")
#: Chonkie recurses on these, in order. A paragraph is kept whole when it fits.
PARAGRAPH_DELIMITERS: tuple[str, ...] = ("\r\n\r\n", "\n\n", "\r\r")
#: Sentence ends, including the common closing punctuation after the mark.
SENTENCE_END_MARKS: tuple[str, ...] = (".", "!", "?", "…")
SENTENCE_CLOSERS: tuple[str, ...] = ('"', "”", "'", "’", "»", "›", ")", "]", "}")
#: The maximum text a chunk preview (first/last line) keeps in the manifest.
PREVIEW_CHARS = 120
#: Monotonic clock used for the timings; injectable so tests can be exact.
Clock = Callable[[], float]


def sentence_delimiters() -> tuple[str, ...]:
    """Every literal delimiter chonkie should treat as a sentence end.

    Delimiters include the whitespace that follows the mark because that is where
    the next sentence starts; ``include_delim="prev"`` then puts the delimiter in
    the previous chunk, which is what keeps concatenation exact.
    """
    delimiters: list[str] = []
    for mark in SENTENCE_END_MARKS:
        for closer in ("", *SENTENCE_CLOSERS):
            delimiters.append(f"{mark}{closer} ")
            delimiters.append(f"{mark}{closer}\n")
            delimiters.append(f"{mark}{closer}\r\n")
    # De-duplicate while keeping the "plain mark" forms first.
    return tuple(dict.fromkeys(delimiters))


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #
@dataclass
class ChunkRecord:
    """One chunk file: identity, provenance and the exact span it covers."""

    global_index: int
    chapter_index: int
    chunk_index: int
    chapter_id: str
    chapter_filename: str
    filename: str
    path: str = ""
    start_char: int = 0
    end_char: int = 0
    start_line: int = 0
    end_line: int = 0
    token_count: int = 0
    bytes: int = 0
    sha256: str = ""
    boundary_before: str = ""
    boundary_after: str = ""
    first_line: str = ""
    last_line: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChapterChunks:
    """One chapter's chunking outcome, including whether it was reused."""

    chapter_id: str
    filename: str
    source_path: str
    chapter_index: int
    title: str = ""
    start_line: int | None = None
    end_line: int | None = None
    chunk_count: int = 0
    char_count: int = 0
    token_count: int = 0
    chunk_tokens: int = 0
    source_sha256: str = ""
    reconstructed_sha256: str = ""
    reconstructs_byte_for_byte: bool = False
    first_global_index: int | None = None
    last_global_index: int | None = None
    skipped: bool = False
    ok: bool = True
    error: str | None = None
    boundary_counts: dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    log: list[str] = field(default_factory=list)

    def record(self) -> dict[str, Any]:
        """The chapter entry stored in the manifest."""
        return {
            "chapter_id": self.chapter_id,
            "filename": self.filename,
            "source_path": self.source_path,
            "chapter_index": self.chapter_index,
            "title": self.title,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "chunk_count": self.chunk_count,
            "char_count": self.char_count,
            "token_count": self.token_count,
            "chunk_tokens": self.chunk_tokens,
            "source_sha256": self.source_sha256,
            "reconstructed_sha256": self.reconstructed_sha256,
            "reconstructs_byte_for_byte": self.reconstructs_byte_for_byte,
            "first_global_index": self.first_global_index,
            "last_global_index": self.last_global_index,
            "skipped": self.skipped,
            "ok": self.ok,
            "error": self.error,
            "boundary_counts": dict(self.boundary_counts),
            "elapsed_seconds": round(self.elapsed_seconds, 4),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.record(), "log": list(self.log)}


@dataclass
class ChunkingResult:
    """Outcome of a whole stage-5 run."""

    ok: bool
    chapters_path: str
    output_dir: str
    source_file: str = ""
    chapter_count: int = 0
    ok_count: int = 0
    processed_count: int = 0
    skipped_count: int = 0
    chunk_count: int = 0
    processed_chunk_count: int = 0
    skipped: bool = False
    dry_run: bool = False
    limited: bool = False
    settings: dict[str, Any] = field(default_factory=dict)
    totals: dict[str, int] = field(default_factory=dict)
    verification: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    chapters: list[dict[str, Any]] = field(default_factory=list)
    files: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    manifest_path: str | None = None
    trace_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: Called once per visited chapter as ``(chapter_record, completed, total)``.
ProgressHook = Callable[[dict[str, Any], int, int], None]


# --------------------------------------------------------------------------- #
# chonkie
# --------------------------------------------------------------------------- #
def _recursive_rules():
    """The four-level rule set: paragraph, sentence, line, whitespace.

    There is deliberately no final token-level rule. If a single word is longer
    than the requested chunk, the post-split in :func:`_enforce_limit` cuts it,
    and that cut is a source slice too; chonkie's token decode would not be.
    """
    from chonkie.types import RecursiveLevel, RecursiveRules

    return RecursiveRules(
        levels=[
            RecursiveLevel(delimiters=list(PARAGRAPH_DELIMITERS), include_delim="prev"),
            RecursiveLevel(delimiters=list(sentence_delimiters()), include_delim="prev"),
            RecursiveLevel(delimiters=["\n", "\r"], include_delim="prev"),
            RecursiveLevel(whitespace=True, include_delim="prev"),
        ]
    )


def build_chunker(chunk_size: int, tokenizer: str | None = None):
    """Build the recursive chunker and resolve the tokenizer it measures with.

    ``chonkie`` is imported here, not at module import, because the splitter, the
    preprocessor and the glossary must keep working without the chunking extra.
    """
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be greater than 0")
    try:
        from chonkie import RecursiveChunker
    except Exception as exc:  # noqa: BLE001 - an ImportError is the normal case
        raise MissingDependencyError(
            "the chunking stage needs chonkie: pip install chonkie"
        ) from exc

    choice = resolve_tokenizer(tokenizer or DEFAULT_CHUNK_TOKENIZER)
    chunker = RecursiveChunker(
        tokenizer=choice.tokenizer,
        chunk_size=int(chunk_size),
        rules=_recursive_rules(),
        min_characters_per_chunk=1,
    )
    return chunker, choice


# --------------------------------------------------------------------------- #
# the cut itself
# --------------------------------------------------------------------------- #
def _spans_from_chonkie(text: str, chunks) -> list[tuple[int, int]]:
    """Turn chonkie chunks into contiguous source spans, or explain why not."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for chunk in chunks:
        start = int(getattr(chunk, "start_index", cursor))
        end = int(getattr(chunk, "end_index", start))
        if start != cursor or not (cursor < end <= len(text)):
            raise ToolError(
                "chonkie returned an invalid chunk span "
                f"({start}:{end} for a chapter of {len(text)} characters, "
                f"expected to start at {cursor})"
            )
        spans.append((start, end))
        cursor = end
    if cursor != len(text):
        raise ToolError(
            "chonkie did not cover the whole chapter "
            f"({cursor} of {len(text)} characters); refusing to write a cut "
            "that would not reconstruct"
        )
    return spans


def _largest_prefix(
    text: str,
    start: int,
    end: int,
    count_tokens: Callable[[str], int],
    limit: int,
    *,
    window: int = 256,
) -> int:
    """The largest cut in ``(start, end]`` that keeps the prefix at or under ``limit``.

    A binary search finds the token boundary; the small backwards walk then moves
    the cut to whitespace when one is close, so an oversized paragraph is still
    not cut inside a word unless it has no whitespace at all.
    """
    low, high = start + 1, end
    best = start + 1
    while low <= high:
        middle = (low + high) // 2
        if count_tokens(text[start:middle]) <= limit:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    floor = max(start + 1, best - window)
    for cut in range(best, floor - 1, -1):
        if text[cut - 1].isspace():
            return cut
    return best


def _enforce_limit(
    text: str,
    spans: Iterable[tuple[int, int]],
    count_tokens: Callable[[str], int],
    limit: int,
) -> list[tuple[int, int]]:
    """Split any chunk chonkie left over the limit, at whitespace where possible."""
    result: list[tuple[int, int]] = []
    for start, end in spans:
        if count_tokens(text[start:end]) <= limit:
            result.append((start, end))
            continue
        cursor = start
        while cursor < end:
            if count_tokens(text[cursor:end]) <= limit:
                result.append((cursor, end))
                break
            cut = _largest_prefix(text, cursor, end, count_tokens, limit)
            if cut <= cursor:  # pragma: no cover - defensive: a single character
                cut = min(cursor + 1, end)
            result.append((cursor, cut))
            cursor = cut
    return result


def _line_numbers(text: str, newlines: list[int], start: int, end: int) -> tuple[int, int]:
    """1-based line span of ``[start, end)`` in ``text``."""
    first = bisect.bisect_left(newlines, start) + 1
    last_position = max(end - 1, start)
    last = bisect.bisect_left(newlines, last_position) + 1
    return first, last


def _preview(value: str) -> str:
    first = next((line for line in value.splitlines() if line.strip()), "")
    return first.strip()[:PREVIEW_CHARS]


def _last_preview(value: str) -> str:
    lines = [line for line in value.splitlines() if line.strip()]
    return lines[-1].strip()[:PREVIEW_CHARS] if lines else ""


def chunk_text(
    text: str,
    *,
    chunker,
    tokenizer: TokenizerChoice,
    chunk_size: int,
) -> list[tuple[int, int, int, str, str]]:
    """Cut one chapter into spans, returning ``(start, end, tokens, before, after)``.

    The token count is measured by the resolved tokenizer, not read from chonkie,
    so the number in the manifest is the number the stage actually enforced.
    """
    if not text:
        return []
    spans = _spans_from_chonkie(text, chunker.chunk(text))
    spans = _enforce_limit(text, spans, tokenizer.count_tokens, int(chunk_size))
    if "".join(text[start:end] for start, end in spans) != text:  # pragma: no cover
        raise ToolError("the computed chunk spans do not cover the chapter exactly")
    output: list[tuple[int, int, int, str, str]] = []
    for position, (start, end) in enumerate(spans):
        before = "chapter-start" if position == 0 else boundary_kind(text, start)
        after = "chapter-end" if position == len(spans) - 1 else boundary_kind(text, end)
        output.append((start, end, tokenizer.count_tokens(text[start:end]), before, after))
    return output


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
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
    :data:`~book_splitter.core.constants.DEFAULT_CHUNK_TOKENS` (1000). A chapter
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

    run_started = clock()
    all_chapters = discover_chapters(chapters_path)
    limited = limit is not None and int(limit) > 0 and int(limit) < len(all_chapters)
    chapters = all_chapters[: int(limit)] if limit is not None and int(limit) > 0 else all_chapters
    source_dir = chapters_dir(chapters_path)
    out_dir = Path(output_dir).expanduser() if output_dir else chunks_dir(source_dir)
    previous = _previous_manifest(out_dir)
    previous_files = _previous_files_by_chapter(previous)
    previous_token_counts = _previous_chapter_tokens(previous)
    reusable = bool(
        skip_if_present
        and not dry_run
        and _previous_settings_match(previous, chunk_size=chunk_size, tokenizer=tokenizer)
    )

    previous_tokenizer = (previous.get("settings") or {}).get("tokenizer") or {}
    settings = {
        "chunk_size": int(chunk_size),
        "chunker": "recursive",
        "boundary_order": list(BOUNDARY_ORDER),
        "chunk_overlap": 0,
        "tokenizer": dict(previous_tokenizer)
        if reusable and previous_tokenizer
        else {"requested": tokenizer or DEFAULT_CHUNK_TOKENIZER},
    }

    # First pass: which chapters can be reused *on content alone*? This is also
    # what lets a fully chunked book return without building a chunker.
    texts: dict[str, str] = {}
    candidates: dict[str, list[dict[str, Any]]] = {}
    for chapter in chapters:
        text = "".join(load_lines(chapter.path))
        texts[chapter.filename] = text
        if not reusable:
            continue
        records = previous_files.get(chapter.filename)
        if records and _records_match_chapter(records, text, out_dir):
            candidates[chapter.filename] = records
    simulated_index = 0
    all_reusable = True
    for chapter in chapters:
        records = candidates.get(chapter.filename)
        if not records:
            all_reusable = False
            break
        expected = simulated_index
        if not _records_are_ordered(records, expected):
            all_reusable = False
            break
        simulated_index += len(records)

    if all_reusable and chapters:
        result = _reused_result(
            chapters=chapters,
            texts=texts,
            candidates=candidates,
            out_dir=out_dir,
            chapters_path=source_dir,
            source_file=load_split_manifest(source_dir).get("source_file", ""),
            settings=settings,
            dry_run=dry_run,
            limited=limited,
            previous_token_counts=previous_token_counts,
        )
        result.manifest_path = (
            str(out_dir / CHUNK_MANIFEST_FILENAME)
            if (out_dir / CHUNK_MANIFEST_FILENAME).is_file()
            else None
        )
        result.trace_path = (
            str(out_dir / CHUNK_TRACE_FILENAME)
            if (out_dir / CHUNK_TRACE_FILENAME).is_file()
            else None
        )
        result.timing = {
            "run_seconds": round(clock() - run_started, 4),
            "chapter_seconds": 0.0,
            "chunk_seconds": 0.0,
        }
        return result

    chunker, choice = build_chunker(int(chunk_size), tokenizer)
    settings["tokenizer"] = choice.to_settings()
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    records: list[ChapterChunks] = []
    files: list[ChunkRecord] = []
    errors: list[str] = []
    global_index = 0
    processed_chunks = 0
    timing_chunk_seconds = 0.0

    for chapter_index, chapter in enumerate(chapters):
        chapter_started = clock()
        text = texts[chapter.filename]
        reusable_records = candidates.get(chapter.filename) if reusable else None
        if reusable_records and _records_are_ordered(reusable_records, global_index):
            chapter_record = _reused_chapter(
                chapter=chapter,
                chapter_index=chapter_index,
                text=text,
                records=reusable_records,
                out_dir=out_dir,
                global_start=global_index,
                token_count=previous_token_counts.get(chapter.filename),
            )
            files.extend(_records_from_dicts(reusable_records, chapter, chapter_index))
            global_index += len(reusable_records)
        else:
            try:
                spans = chunk_text(
                    text,
                    chunker=chunker,
                    tokenizer=choice,
                    chunk_size=int(chunk_size),
                )
                chapter_record, chapter_files = _chunk_one_chapter(
                    chapter=chapter,
                    chapter_index=chapter_index,
                    text=text,
                    spans=spans,
                    out_dir=out_dir,
                    global_start=global_index,
                    tokenizer=choice,
                    dry_run=dry_run,
                )
                files.extend(chapter_files)
                global_index += chapter_record.chunk_count
                processed_chunks += chapter_record.chunk_count
                timing_chunk_seconds += chapter_record.elapsed_seconds
            except Exception as exc:  # one bad chapter must not stop the book
                chapter_record = ChapterChunks(
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
                errors.append(f"{chapter.filename}: {chapter_record.error}")
        chapter_record.elapsed_seconds = clock() - chapter_started
        records.append(chapter_record)
        if progress is not None:
            with contextlib.suppress(Exception):  # a reporting hook must never break a run
                progress(chapter_record.to_dict(), len(records), len(chapters))

    result = _build_result(
        records=records,
        files=files,
        chapters=chapters,
        texts=texts,
        chapters_path=source_dir,
        out_dir=out_dir,
        source_file=load_split_manifest(source_dir).get("source_file", ""),
        settings=settings,
        tokenizer=choice,
        processed_chunks=processed_chunks,
        skipped_count=sum(1 for record in records if record.skipped),
        errors=errors,
        dry_run=dry_run,
        limited=limited,
        timing={
            "chapter_seconds": round(sum(record.elapsed_seconds for record in records), 4),
            "chunk_seconds": round(timing_chunk_seconds, 4),
            "run_seconds": round(clock() - run_started, 4),
        },
    )
    if write_manifest and not dry_run:
        result.manifest_path = str(write_chunk_manifest(out_dir, result))
        result.trace_path = str(write_chunk_trace(out_dir, result))
    return result


# --------------------------------------------------------------------------- #
# one chapter
# --------------------------------------------------------------------------- #
def _chunk_one_chapter(
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
    source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
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
        filename = chunk_filename(global_start + chunk_index, chapter.chapter_id, chunk_index)
        path = out_dir / filename
        payload = piece.encode("utf-8")
        if not dry_run:
            path.write_bytes(payload)
        start_line, end_line = _line_numbers(text, newlines, start, end)
        record_payload = ChunkRecord(
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
            sha256=hashlib.sha256(payload).hexdigest(),
            boundary_before=before,
            boundary_after=after,
            first_line=_preview(piece),
            last_line=_last_preview(piece),
        )
        chunk_records.append(record_payload)
    reconstructed = "".join(pieces)
    record.reconstructed_sha256 = hashlib.sha256(reconstructed.encode("utf-8")).hexdigest()
    record.reconstructs_byte_for_byte = reconstructed == text
    record.chunk_tokens = sum(span[2] for span in spans)
    record.boundary_counts = boundary_counts(boundary_kinds)
    record.log.append(
        f"{len(spans)} chunk(s) from {len(text):,} chars; "
        f"{record.token_count:,} tokens; "
        f"reconstructs={record.reconstructs_byte_for_byte}"
    )
    return record, chunk_records


def _reused_chapter(
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
    source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    chunks = sorted(records, key=lambda entry: int(entry.get("chunk_index") or 0))
    reconstructed = "".join(
        (out_dir / str(entry["filename"])).read_text(encoding="utf-8") for entry in chunks
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
        reconstructed_sha256=hashlib.sha256(reconstructed.encode("utf-8")).hexdigest(),
        reconstructs_byte_for_byte=reconstructed == text,
        first_global_index=global_start,
        last_global_index=global_start + len(chunks) - 1 if chunks else None,
        skipped=True,
        boundary_counts=boundary_counts(
            str(entry.get("boundary_before") or "") for entry in chunks
        ),
    )
    record.log.append(f"reused {len(chunks)} chunk(s); the files still reassemble the chapter")
    return record


def _records_from_dicts(
    records: list[dict[str, Any]], chapter: Chapter, chapter_index: int
) -> list[ChunkRecord]:
    """Rebuild typed records from a previous manifest for the result/trace."""
    rebuilt: list[ChunkRecord] = []
    for entry in sorted(records, key=lambda item: int(item.get("chunk_index") or 0)):
        rebuilt.append(
            ChunkRecord(
                global_index=int(entry.get("global_index") or 0),
                chapter_index=chapter_index,
                chunk_index=int(entry.get("chunk_index") or 0),
                chapter_id=chapter.chapter_id,
                chapter_filename=chapter.filename,
                filename=str(entry.get("filename") or ""),
                path=str(entry.get("path") or ""),
                start_char=int(entry.get("start_char") or 0),
                end_char=int(entry.get("end_char") or 0),
                start_line=int(entry.get("start_line") or 0),
                end_line=int(entry.get("end_line") or 0),
                token_count=int(entry.get("token_count") or 0),
                bytes=int(entry.get("bytes") or 0),
                sha256=str(entry.get("sha256") or ""),
                boundary_before=str(entry.get("boundary_before") or ""),
                boundary_after=str(entry.get("boundary_after") or ""),
                first_line=str(entry.get("first_line") or ""),
                last_line=str(entry.get("last_line") or ""),
            )
        )
    return rebuilt


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
def _build_result(
    *,
    records: list[ChapterChunks],
    files: list[ChunkRecord],
    chapters: list[Chapter],
    texts: dict[str, str],
    chapters_path: Path,
    out_dir: Path,
    source_file: Any,
    settings: dict[str, Any],
    tokenizer: TokenizerChoice,
    processed_chunks: int,
    skipped_count: int,
    errors: list[str],
    dry_run: bool,
    limited: bool,
    timing: dict[str, float],
) -> ChunkingResult:
    ordered_files = sorted(files, key=lambda entry: entry.global_index)
    chapters_ok = all(record.ok and record.reconstructs_byte_for_byte for record in records)
    slices_ok = all(record.reconstructs_byte_for_byte for record in records)
    book_ok = all(record.reconstructs_byte_for_byte for record in records)
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
    result = ChunkingResult(
        ok=bool(records)
        and chapters_ok
        and slices_ok
        and book_ok
        and violations == 0
        and not errors,
        chapters_path=str(chapters_path),
        output_dir=str(out_dir),
        source_file=str(source_file or ""),
        chapter_count=len(records),
        ok_count=sum(1 for record in records if record.ok),
        processed_count=sum(1 for record in records if not record.skipped),
        skipped_count=skipped_count,
        chunk_count=len(ordered_files),
        processed_chunk_count=processed_chunks,
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
            "chunks_are_source_slices": slices_ok,
            "chapters_reconstruct_byte_for_byte": all(
                record.reconstructs_byte_for_byte for record in records
            ),
            "book_reconstructs_byte_for_byte": book_ok,
            "chunk_size_violations": violations,
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
    return result


def _reused_result(
    *,
    chapters: list[Chapter],
    texts: dict[str, str],
    candidates: dict[str, list[dict[str, Any]]],
    out_dir: Path,
    chapters_path: Path,
    source_file: Any,
    settings: dict[str, Any],
    dry_run: bool,
    limited: bool,
    previous_token_counts: dict[str, int] | None = None,
) -> ChunkingResult:
    """A run where every chapter could be reused: no chunker, no tokenizer."""
    token_counts = previous_token_counts or {}
    records: list[ChapterChunks] = []
    files: list[ChunkRecord] = []
    global_index = 0
    for chapter_index, chapter in enumerate(chapters):
        previous = candidates[chapter.filename]
        record = _reused_chapter(
            chapter=chapter,
            chapter_index=chapter_index,
            text=texts[chapter.filename],
            records=previous,
            out_dir=out_dir,
            global_start=global_index,
            token_count=token_counts.get(chapter.filename),
        )
        records.append(record)
        files.extend(_records_from_dicts(previous, chapter, chapter_index))
        global_index += len(previous)
    tokenizer = TokenizerChoice(
        requested=str((settings.get("tokenizer") or {}).get("requested") or ""),
        used=str((settings.get("tokenizer") or {}).get("used") or ""),
        tokenizer=None,
        fallback=bool((settings.get("tokenizer") or {}).get("fallback")),
        note=str((settings.get("tokenizer") or {}).get("note") or ""),
    )
    ordered = sorted(files, key=lambda entry: entry.global_index)
    kinds = boundary_counts(entry.boundary_before for entry in ordered)
    violations = sum(1 for entry in ordered if int(entry.token_count) > int(settings["chunk_size"]))
    reconstructs = all(record.reconstructs_byte_for_byte for record in records)
    return ChunkingResult(
        ok=bool(records) and reconstructs and violations == 0,
        chapters_path=str(chapters_path),
        output_dir=str(out_dir),
        source_file=str(source_file or ""),
        chapter_count=len(records),
        ok_count=len(records),
        processed_count=0,
        skipped_count=len(records),
        chunk_count=len(ordered),
        processed_chunk_count=0,
        skipped=True,
        dry_run=dry_run,
        limited=limited,
        settings=settings,
        totals={
            "chapters": len(records),
            "chunks": len(ordered),
            "processed_chunks": 0,
            "reused_chunks": len(ordered),
            "chapter_chars": sum(record.char_count for record in records),
            "chunk_bytes": sum(int(entry.bytes) for entry in ordered),
            "chapter_tokens": sum(record.token_count for record in records),
            "chunk_tokens": sum(int(entry.token_count) for entry in ordered),
        },
        verification={
            "chunks_are_source_slices": True,
            "chapters_reconstruct_byte_for_byte": reconstructs,
            "book_reconstructs_byte_for_byte": reconstructs,
            "chunk_size_violations": violations,
            "word_breaking_chunks": kinds.get("hard", 0),
            "paragraph_safe_chunks": sum(
                1 for entry in ordered if entry.boundary_before in PARAGRAPH_SAFE
            ),
            "sentence_safe_chunks": sum(
                1 for entry in ordered if entry.boundary_before in SENTENCE_SAFE
            ),
            "boundary_counts": kinds,
            "tokenizer_fallback": tokenizer.fallback,
            "tokenizer_note": tokenizer.note,
        },
        chapters=[record.to_dict() for record in records],
        files=[entry.to_dict() for entry in ordered],
    )


# --------------------------------------------------------------------------- #
# reuse and prior runs
# --------------------------------------------------------------------------- #
def _previous_manifest(out_dir: Path) -> dict[str, Any]:
    path = Path(out_dir) / CHUNK_MANIFEST_FILENAME
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _previous_chapter_tokens(manifest: dict[str, Any]) -> dict[str, int]:
    """The full-chapter token count recorded by an earlier run, by filename."""
    counts: dict[str, int] = {}
    for record in manifest.get("chapters") or []:
        if not isinstance(record, dict):
            continue
        name = str(record.get("filename") or "")
        if name and record.get("token_count") is not None:
            counts[name] = int(record["token_count"])
    return counts


def _previous_files_by_chapter(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in manifest.get("files") or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("chapter_filename") or "")
        if not name:
            continue
        grouped.setdefault(name, []).append(entry)
    for entries in grouped.values():
        entries.sort(key=lambda item: int(item.get("chunk_index") or 0))
    return grouped


def _previous_settings_match(
    manifest: dict[str, Any], *, chunk_size: int, tokenizer: str | None
) -> bool:
    settings = manifest.get("settings") or {}
    if int(settings.get("chunk_size") or 0) != int(chunk_size):
        return False
    recorded = settings.get("tokenizer") or {}
    requested = str(recorded.get("requested") or "")
    if requested != (tokenizer or DEFAULT_CHUNK_TOKENIZER):
        return False
    # A previous fallback estimate cannot be trusted against a real tokenizer now.
    return not bool(recorded.get("fallback"))


def _records_are_ordered(records: list[dict[str, Any]], expected_start: int) -> bool:
    ordered = sorted(records, key=lambda item: int(item.get("chunk_index") or 0))
    if [int(item.get("chunk_index") or 0) for item in ordered] != list(range(len(ordered))):
        return False
    return all(
        int(item.get("global_index") if item.get("global_index") is not None else -1)
        == expected_start + position
        for position, item in enumerate(ordered)
    )


def _records_match_chapter(records: list[dict[str, Any]], text: str, out_dir: Path) -> bool:
    """True when the recorded files are still the chapter's exact slices.

    Both halves matter: every file must equal ``text[start_char:end_char]`` (the
    slice the manifest promises) *and* the files together must equal the chapter
    (no gap, no duplicate). A manifest whose offsets no longer describe the files
    is not reusable, even if a plain concatenation happens to look right.
    """
    ordered = sorted(records, key=lambda item: int(item.get("chunk_index") or 0))
    parts: list[str] = []
    for entry in ordered:
        path = out_dir / str(entry.get("filename") or "")
        if not path.is_file():
            return False
        try:
            piece = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return False
        start = int(entry.get("start_char") or 0)
        end = int(entry.get("end_char") or 0)
        if piece != text[start:end]:
            return False
        parts.append(piece)
    return "".join(parts) == text


# --------------------------------------------------------------------------- #
# naming
# --------------------------------------------------------------------------- #
_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
#: A chapter stem longer than this is shortened; the global index keeps the name
#: unique, so the stem is only there for a human reading the folder.
MAX_CHAPTER_SLUG_CHARS = 60


def chapter_slug(chapter_id: str) -> str:
    slug = _UNSAFE_FILENAME.sub("-", str(chapter_id or "")).strip("-._")
    if len(slug) > MAX_CHAPTER_SLUG_CHARS:
        slug = slug[:MAX_CHAPTER_SLUG_CHARS].rstrip("-._")
    return slug or "chapter"


def chunk_filename(global_index: int, chapter_id: str, chunk_index: int) -> str:
    """``00000-01-predgovor-0000.md`` — sortable, unique, chapter-tagged."""
    return (
        f"{int(global_index):0{CHUNK_ORDER_DIGITS}d}-"
        f"{chapter_slug(chapter_id)}-"
        f"{int(chunk_index):0{CHUNK_INDEX_DIGITS}d}.md"
    )


# --------------------------------------------------------------------------- #
# artifacts and reconstruction
# --------------------------------------------------------------------------- #
def write_chunk_manifest(out_dir: str | Path, result: ChunkingResult) -> Path:
    """Write ``_chunk_manifest.json``: the book's chunk index and its proof."""
    path = Path(out_dir) / CHUNK_MANIFEST_FILENAME
    payload = {
        "stage": "chunk",
        "chapters_path": result.chapters_path,
        "source_file": result.source_file,
        "ok": result.ok,
        "dry_run": result.dry_run,
        "limited": result.limited,
        "settings": result.settings,
        "chapter_count": result.chapter_count,
        "chunk_count": result.chunk_count,
        "processed_count": result.processed_count,
        "processed_chunk_count": result.processed_chunk_count,
        "skipped_count": result.skipped_count,
        "totals": result.totals,
        "verification": result.verification,
        "timing": result.timing,
        "chapters": result.chapters,
        "files": result.files,
        "errors": result.errors,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_chunk_trace(out_dir: str | Path, result: ChunkingResult) -> Path:
    """Write the audit trail: the same run, plus each chapter's own log."""
    path = Path(out_dir) / CHUNK_TRACE_FILENAME
    path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


def load_chunk_manifest(chunks_path: str | Path) -> dict[str, Any]:
    """Read ``_chunk_manifest.json`` from a chunks folder (or the file itself)."""
    path = Path(chunks_path).expanduser()
    manifest_path = (path / CHUNK_MANIFEST_FILENAME) if path.is_dir() else path
    if not manifest_path.is_file():
        return {}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def chunk_files(chunks_path: str | Path) -> list[dict[str, Any]]:
    """Every chunk record, in reading order.

    The manifest is authoritative. Without one, the folder is read by sorted file
    name — which is exactly why the global index is the first part of every name.
    """
    folder = Path(chunks_path).expanduser()
    manifest = load_chunk_manifest(folder)
    if folder.is_file():
        folder = folder.parent
    if manifest.get("files"):
        return sorted(
            (entry for entry in manifest["files"] if isinstance(entry, dict)),
            key=lambda entry: int(entry.get("global_index") or 0),
        )
    return [
        {
            "global_index": index,
            "chunk_index": index,
            "filename": path.name,
            "path": str(path),
            "chapter_filename": "",
            "chapter_id": "",
        }
        for index, path in enumerate(sorted(folder.glob("*.md")))
        if not path.name.startswith("_")
    ]


def reconstruct_chapter(chunks_path: str | Path, chapter: str) -> str:
    """Concatenate one chapter's chunks back into its exact text.

    ``chapter`` may be the chapter filename (``01-predgovor.md``) or its id
    (``01-predgovor``). The chunks are joined in ``chunk_index`` order, which the
    manifest records; no separator is inserted.
    """
    folder = Path(chunks_path).expanduser()
    manifest = load_chunk_manifest(folder)
    entries = [
        entry
        for entry in manifest.get("files") or []
        if isinstance(entry, dict)
        and str(entry.get("chapter_filename") or "") in {chapter, f"{chapter}.md"}
    ]
    if not entries:
        # Accept a chapter id that matches the manifest's chapter records.
        for record in manifest.get("chapters") or []:
            if not isinstance(record, dict):
                continue
            if (
                str(record.get("filename") or "") in {chapter, f"{chapter}.md"}
                or str(record.get("chapter_id") or "") == chapter
            ):
                if int(record.get("chunk_count") or 0) == 0:
                    return ""  # an empty chapter is a legitimate zero-chunk chapter
                name = str(record.get("filename") or "")
                entries = [
                    entry
                    for entry in manifest.get("files") or []
                    if isinstance(entry, dict) and str(entry.get("chapter_filename") or "") == name
                ]
                break
    if not entries:
        raise ToolError(f"no chunks recorded for chapter {chapter!r} in {folder}")
    entries.sort(key=lambda entry: int(entry.get("chunk_index") or 0))
    return "".join(
        (folder / str(entry.get("filename") or "")).read_text(encoding="utf-8") for entry in entries
    )


def reconstruct_book(chunks_path: str | Path) -> str:
    """Concatenate the whole folder back into the book, in global order."""
    folder = Path(chunks_path).expanduser()
    return "".join(
        (folder / str(entry.get("filename") or "")).read_text(encoding="utf-8")
        for entry in chunk_files(folder)
    )


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
    "sentence_delimiters",
    "write_chunk_manifest",
    "write_chunk_trace",
]
