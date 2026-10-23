"""Manifest records for a synthetic front/back matter section.

A synthetic section is a normal section from here on: it gets the same split
manifest entry a stage-1 chapter has, and the same chunk file + chunk manifest
records stage 5 writes. Keeping the two record builders here means the
orchestrator in :mod:`book_translator.matter.synthesize` only decides *what* to
create, not *how* a record looks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.artifacts import sha256_text, write_manifest, write_text
from ..core.constants import CHUNK_MANIFEST_FILENAME, MANIFEST_FILENAME
from ..documents.manifest import load_chunk_manifest
from ..postprocess.inspect import matter_kind
from .generate import BACK_MATTER_TITLE


def _sha256(text: str) -> str:
    return sha256_text(text)


def _lines(text: str) -> int:
    return len(text.splitlines()) or 1


def _tokens(text: str) -> int:
    # The matter agent only needs a stable estimate; these sections are tiny.
    return max(1, len(text) // 4)


def _preview(text: str, first: bool) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    return (lines[0] if first else lines[-1])[:120]


def chapter_entry(
    *,
    base: Path,
    filename: str,
    kind: str,
    title: str,
    text: str,
    start_line: int,
    end_line: int,
) -> dict[str, Any]:
    path = base / filename
    return {
        "filename": filename,
        "path": str(path),
        "kind": kind,
        "start_line": start_line,
        "end_line": end_line,
        "lines": _lines(text),
        "bytes": len(text.encode("utf-8")),
        "sha256": _sha256(text),
        "first_line": text.splitlines()[0] if text.splitlines() else "",
        "title": title,
        "synthetic": True,
    }


def write_chunk_record(
    *,
    chapters: Path,
    chunks: Path,
    filename: str,
    chapter_id: str,
    chapter_filename: str,
    title: str,
    text: str,
    global_index: int,
    chapter_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write one synthetic chunk file and return its file + chapter records."""
    chunk_name = f"{global_index:05d}-{chapter_id}-0000.md"
    path = chunks / chunk_name
    write_text(path, text)
    digest = _sha256(text)
    tokens = _tokens(text)
    file_record = {
        "global_index": global_index,
        "chapter_index": chapter_index,
        "chunk_index": 0,
        "chapter_id": chapter_id,
        "chapter_filename": chapter_filename,
        "filename": chunk_name,
        "path": str(path),
        "start_char": 0,
        "end_char": len(text),
        "start_line": 1,
        "end_line": _lines(text),
        "token_count": tokens,
        "bytes": len(text.encode("utf-8")),
        "sha256": digest,
        "boundary_before": "chapter-start",
        "boundary_after": "chapter-end",
        "first_line": _preview(text, True),
        "last_line": _preview(text, False),
    }
    chapter_record = {
        "chapter_id": chapter_id,
        "filename": chapter_filename,
        "source_path": str(chapters / chapter_filename),
        "chapter_index": chapter_index,
        "title": title,
        "start_line": 1,
        "end_line": _lines(text),
        "chunk_count": 1,
        "char_count": len(text),
        "token_count": tokens,
        "chunk_tokens": tokens,
        "source_sha256": digest,
        "reconstructed_sha256": digest,
        "reconstructs_byte_for_byte": True,
        "first_global_index": global_index,
        "last_global_index": global_index,
        "skipped": False,
        "ok": True,
        "error": None,
        "boundary_counts": {"chapter-start": 1, "chapter-end": 1},
        "elapsed_seconds": 0.0,
    }
    return file_record, chapter_record


def recompute_chunk_totals(manifest: dict[str, Any]) -> None:
    files = [entry for entry in manifest.get("files") or [] if isinstance(entry, dict)]
    chapters = [entry for entry in manifest.get("chapters") or [] if isinstance(entry, dict)]
    manifest["chapter_count"] = len(chapters)
    manifest["chunk_count"] = len(files)
    manifest["processed_count"] = len(chapters)
    manifest["processed_chunk_count"] = len(files)
    totals = dict(manifest.get("totals") or {})
    totals["chapters"] = len(chapters)
    totals["chunks"] = len(files)
    totals["processed_chunks"] = len(files)
    totals["chunk_bytes"] = sum(int(entry.get("bytes") or 0) for entry in files)
    totals["chunk_tokens"] = sum(int(entry.get("token_count") or 0) for entry in files)
    totals["chapter_chars"] = sum(int(entry.get("char_count") or 0) for entry in chapters)
    totals["chapter_tokens"] = sum(int(entry.get("token_count") or 0) for entry in chapters)
    manifest["totals"] = totals
    verification = dict(manifest.get("verification") or {})
    verification["synthetic_matter"] = True
    manifest["verification"] = verification


def existing_matter_entries(split: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """The existing front and back matter entries, if any."""
    front: dict[str, Any] | None = None
    back: dict[str, Any] | None = None
    files = [entry for entry in split.get("files") or [] if isinstance(entry, dict)]
    for index, entry in enumerate(files):
        name = str(entry.get("filename") or "")
        kind = str(entry.get("kind") or "").strip().lower()
        if kind in {"front-matter", "frontmatter", "front matter", "front"}:
            front = front or entry
            continue
        if kind in {"back-matter", "backmatter", "back matter", "back"}:
            back = back or entry
            continue
        guessed = matter_kind(name, index, str(entry.get("title") or ""))
        if guessed == "front":
            front = front or entry
        elif guessed == "back":
            back = back or entry
    return front, back



def update_split_manifest(
    base: Path,
    split: dict[str, Any],
    front_entry: dict[str, Any] | None,
    back_entry: dict[str, Any] | None,
) -> None:
    """Add the synthetic sections to stage 1's manifest, front first, back last."""
    files = [entry for entry in split.get("files") or [] if isinstance(entry, dict)]
    known = {str(entry.get("filename") or "") for entry in files}
    if front_entry is not None and front_entry["filename"] not in known:
        files.insert(0, front_entry)
    if back_entry is not None and back_entry["filename"] not in known:
        files.append(back_entry)
    split["files"] = files
    split["file_count"] = len(files)
    split["chapter_count"] = sum(
        1
        for entry in files
        if str(entry.get("kind") or "").lower()
        not in {"front-matter", "frontmatter", "back-matter", "backmatter"}
    )
    split["synthetic_matter"] = sorted(
        set(
            list(split.get("synthetic_matter") or [])
            + [entry["filename"] for entry in (front_entry, back_entry) if entry]
        )
    )
    write_manifest(base, MANIFEST_FILENAME, split)


def update_chunk_manifest(
    base: Path,
    chunks: Path,
    front_entry: dict[str, Any] | None,
    back_entry: dict[str, Any] | None,
) -> list[str]:
    """Give the synthetic sections chunk files + records, when stage 5 ran.

    Front matter must sort before the body, so the *recorded* indices of every
    existing chunk are shifted by one; the chunk filenames are never renamed, so
    translated chunks stay valid. Back matter goes last and takes the next
    indices.
    """
    manifest = load_chunk_manifest(chunks) if chunks.is_dir() else {}
    if not manifest:
        return []

    chunk_files = [
        entry for entry in manifest.get("files") or [] if isinstance(entry, dict)
    ]
    chapter_files = [
        entry for entry in manifest.get("chapters") or [] if isinstance(entry, dict)
    ]
    existing_chapters = {
        str(entry.get("chapter_filename") or "") for entry in chunk_files
    }
    chunk_created: list[str] = []

    if front_entry is not None and front_entry["filename"] not in existing_chapters:
        for entry in chunk_files:
            entry["global_index"] = int(entry.get("global_index") or 0) + 1
            entry["chapter_index"] = int(entry.get("chapter_index") or 0) + 1
        for entry in chapter_files:
            entry["chapter_index"] = int(entry.get("chapter_index") or 0) + 1
            for key in ("first_global_index", "last_global_index"):
                if entry.get(key) is not None:
                    entry[key] = int(entry[key]) + 1
        text = (base / front_entry["filename"]).read_text(encoding="utf-8")
        file_record, chapter_record = write_chunk_record(
            chapters=base,
            chunks=chunks,
            filename=front_entry["filename"],
            chapter_id=Path(front_entry["filename"]).stem,
            chapter_filename=front_entry["filename"],
            title=front_entry.get("title") or "Front matter",
            text=text,
            global_index=0,
            chapter_index=0,
        )
        chunk_files.insert(0, file_record)
        chapter_files.insert(0, chapter_record)
        chunk_created.append(file_record["filename"])

    if back_entry is not None and back_entry["filename"] not in existing_chapters:
        next_global = (
            max((int(entry.get("global_index") or 0) for entry in chunk_files), default=-1)
            + 1
        )
        next_chapter = (
            max(
                (int(entry.get("chapter_index") or 0) for entry in chapter_files),
                default=-1,
            )
            + 1
        )
        text = (base / back_entry["filename"]).read_text(encoding="utf-8")
        file_record, chapter_record = write_chunk_record(
            chapters=base,
            chunks=chunks,
            filename=back_entry["filename"],
            chapter_id=Path(back_entry["filename"]).stem,
            chapter_filename=back_entry["filename"],
            title=back_entry.get("title") or BACK_MATTER_TITLE,
            text=text,
            global_index=next_global,
            chapter_index=next_chapter,
        )
        chunk_files.append(file_record)
        chapter_files.append(chapter_record)
        chunk_created.append(file_record["filename"])

    manifest["files"] = chunk_files
    manifest["chapters"] = chapter_files
    recompute_chunk_totals(manifest)
    write_manifest(chunks, CHUNK_MANIFEST_FILENAME, manifest)
    return chunk_created


__all__ = [
    "chapter_entry",
    "existing_matter_entries",
    "recompute_chunk_totals",
    "update_chunk_manifest",
    "update_split_manifest",
    "write_chunk_record",
]
