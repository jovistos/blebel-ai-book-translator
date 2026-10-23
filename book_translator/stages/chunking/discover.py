"""Discovery for stage 5: paths, previous run, texts and reuse candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from book_translator.core.constants import DEFAULT_CHUNK_TOKENIZER
from book_translator.documents.chapters import Chapter, chapters_dir, chunks_dir, discover_chapters
from book_translator.documents.io import load_lines

from .artifacts import (
    previous_chapter_tokens,
    previous_files_by_chapter,
    previous_manifest,
    previous_settings_match,
    records_are_ordered,
    records_match_chapter,
)
from .models import BOUNDARY_ORDER


@dataclass
class ChunkingInputs:
    """Everything the cut needs before the first chapter is visited."""

    chapters: list[Chapter]
    source_dir: Path
    out_dir: Path
    texts: dict[str, str] = field(default_factory=dict)
    candidates: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    previous_token_counts: dict[str, int] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    reusable: bool = False
    all_reusable: bool = False
    limited: bool = False


def discover_chunking(
    chapters_path: str,
    output_dir: str | None,
    *,
    chunk_size: int,
    tokenizer: str | None,
    skip_if_present: bool,
    dry_run: bool,
    limit: int | None,
) -> ChunkingInputs:
    all_chapters = discover_chapters(chapters_path)
    limited = limit is not None and int(limit) > 0 and int(limit) < len(all_chapters)
    chapters = (
        all_chapters[: int(limit)]
        if limit is not None and int(limit) > 0
        else all_chapters
    )
    source_dir = chapters_dir(chapters_path)
    out_dir = Path(output_dir).expanduser() if output_dir else chunks_dir(source_dir)
    previous = previous_manifest(out_dir)
    previous_files = previous_files_by_chapter(previous)
    previous_token_counts = previous_chapter_tokens(previous)
    reusable = bool(
        skip_if_present
        and not dry_run
        and previous_settings_match(previous, chunk_size=chunk_size, tokenizer=tokenizer)
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
        previous_records = previous_files.get(chapter.filename)
        if previous_records and records_match_chapter(previous_records, text, out_dir):
            candidates[chapter.filename] = previous_records
    simulated_index = 0
    all_reusable = True
    for chapter in chapters:
        candidate_records = candidates.get(chapter.filename)
        if not candidate_records:
            all_reusable = False
            break
        if not records_are_ordered(candidate_records, simulated_index):
            all_reusable = False
            break
        simulated_index += len(candidate_records)

    return ChunkingInputs(
        chapters=chapters,
        source_dir=source_dir,
        out_dir=out_dir,
        texts=texts,
        candidates=candidates,
        previous_token_counts=previous_token_counts,
        settings=settings,
        reusable=reusable,
        all_reusable=all_reusable and bool(chapters),
        limited=limited,
    )


__all__ = ["ChunkingInputs", "discover_chunking"]
