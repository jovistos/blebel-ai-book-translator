"""Previous-run records, naming, manifests and reconstruction for chunking."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from book_translator.core.artifacts import read_manifest, write_manifest, write_trace
from book_translator.core.constants import (
    CHUNK_INDEX_DIGITS,
    CHUNK_MANIFEST_FILENAME,
    CHUNK_ORDER_DIGITS,
    CHUNK_TRACE_FILENAME,
    DEFAULT_CHUNK_TOKENIZER,
)
from book_translator.core.errors import ToolError
from book_translator.documents.chapters import Chapter
from book_translator.documents.chunks import chunk_files
from book_translator.documents.manifest import load_chunk_manifest

from .models import ChunkingResult, ChunkRecord


def previous_manifest(out_dir: Path) -> dict[str, Any]:
    return read_manifest(out_dir, CHUNK_MANIFEST_FILENAME)


def previous_chapter_tokens(manifest: dict[str, Any]) -> dict[str, int]:
    """The full-chapter token count recorded by an earlier run, by filename."""
    counts: dict[str, int] = {}
    for record in manifest.get("chapters") or []:
        if not isinstance(record, dict):
            continue
        name = str(record.get("filename") or "")
        if name and record.get("token_count") is not None:
            counts[name] = int(record["token_count"])
    return counts


def previous_files_by_chapter(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
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


def previous_settings_match(
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


def _global_index_or_sentinel(item: dict[str, Any]) -> int:
    """The record's ``global_index`` as an int, or ``-1`` when it has none."""
    value = item.get("global_index")
    return int(value) if value is not None else -1


def records_are_ordered(records: list[dict[str, Any]], expected_start: int) -> bool:
    ordered = sorted(records, key=lambda item: int(item.get("chunk_index") or 0))
    if [int(item.get("chunk_index") or 0) for item in ordered] != list(range(len(ordered))):
        return False
    return all(
        _global_index_or_sentinel(item) == expected_start + position
        for position, item in enumerate(ordered)
    )


def records_match_chapter(records: list[dict[str, Any]], text: str, out_dir: Path) -> bool:
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


def records_from_dicts(
    records: list[dict[str, Any]], chapter: Chapter, chapter_index: int
) -> list[ChunkRecord]:
    """Rebuild typed records from a previous manifest for the result/trace."""
    return [
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
        for entry in sorted(records, key=lambda item: int(item.get("chunk_index") or 0))
    ]


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
def chunk_manifest_payload(result: ChunkingResult) -> dict[str, Any]:
    return {
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


def write_chunk_manifest(out_dir: str | Path, result: ChunkingResult) -> Path:
    """Write ``_chunk_manifest.json``: the book's chunk index and its proof."""
    return write_manifest(out_dir, CHUNK_MANIFEST_FILENAME, chunk_manifest_payload(result))


def write_chunk_trace(out_dir: str | Path, result: ChunkingResult) -> Path:
    """Write the audit trail: the same run, plus each chapter's own log."""
    return write_trace(out_dir, CHUNK_TRACE_FILENAME, result.to_dict())


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
                    if isinstance(entry, dict)
                    and str(entry.get("chapter_filename") or "") == name
                ]
                break
    if not entries:
        raise ToolError(f"no chunks recorded for chapter {chapter!r} in {folder}")
    entries.sort(key=lambda entry: int(entry.get("chunk_index") or 0))
    return "".join(
        (folder / str(entry.get("filename") or "")).read_text(encoding="utf-8")
        for entry in entries
    )


def reconstruct_book(chunks_path: str | Path) -> str:
    """Concatenate the whole folder back into the book, in global order."""
    folder = Path(chunks_path).expanduser()
    return "".join(
        (folder / str(entry.get("filename") or "")).read_text(encoding="utf-8")
        for entry in chunk_files(folder)
    )


__all__ = [
    "chapter_slug",
    "chunk_filename",
    "chunk_manifest_payload",
    "previous_chapter_tokens",
    "previous_files_by_chapter",
    "previous_manifest",
    "previous_settings_match",
    "records_are_ordered",
    "records_from_dicts",
    "records_match_chapter",
    "reconstruct_book",
    "reconstruct_chapter",
    "write_chunk_manifest",
    "write_chunk_trace",
]
