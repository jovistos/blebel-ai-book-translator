"""Create front/back matter when the source book did not carry any.

A split book without front matter is legitimate — but a published translation
still needs a title page and, ideally, a back page. This module makes the two
missing sections deterministic:

* it reads the book title and author from the input filename;
* it writes ``00-front-matter.md`` and/or ``99-back-matter.md`` next to the
  split chapters;
* it appends the matching records to stage 1's manifest and stage 5's chunk
  manifest, so the matter agent, assembly and bind see them like any other
  matter section.

Nothing here calls a model. The matter agent still localizes the generated
English text; the generated section only supplies the facts it must not lose.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..chunking.runner import load_chunk_manifest
from ..core.constants import (
    CHUNK_MANIFEST_FILENAME,
    CHUNKS_DIRNAME,
    FRONT_MATTER_FILENAME,
    MANIFEST_FILENAME,
)
from ..documents.chapters import chapters_dir, load_split_manifest
from ..postprocess.inspect import matter_kind
from .metadata import BookMetadata, parse_book_metadata

#: A numbered name keeps the file in reading order and makes :func:`matter_kind`
#: classify it as back matter without relying on the manifest.
BACK_MATTER_FILENAME = "99-back-matter.md"

#: The first line the matter agent receives for a generated back page.
BACK_MATTER_TITLE = "About the Author"


class MatterSynthesisError(RuntimeError):
    """A synthetic section could not be added consistently to both manifests."""


@dataclass
class MatterSynthesis:
    """What :func:`ensure_matter_sections` created."""

    created: list[str] = field(default_factory=list)
    front_created: bool = False
    back_created: bool = False
    metadata: BookMetadata = field(default_factory=BookMetadata)
    chunk_created: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.created)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def _front_text(metadata: BookMetadata) -> str:
    title = metadata.title or metadata.stem or "Untitled"
    lines = [f"# {title}", ""]
    if metadata.author:
        lines.extend([metadata.author, ""])
    lines.append("*Translated by BleBel*")
    return "\n".join(lines).rstrip() + "\n"


def _back_text(metadata: BookMetadata) -> str:
    title = metadata.title or metadata.stem or "this book"
    lines = [f"# {BACK_MATTER_TITLE}", ""]
    if metadata.author:
        lines.append(f"{metadata.author} is the author of {title}.")
    else:
        lines.append(f"{title} was published in translation by BleBel.")
    return "\n".join(lines).rstrip() + "\n"


def _entry(
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


def _chunk_record(
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
    path.write_text(text, encoding="utf-8")
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


def _recompute_chunk_totals(manifest: dict[str, Any]) -> None:
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


def _free_name(base: Path, preferred: str, fallback: str) -> str:
    if not (base / preferred).exists():
        return preferred
    if not (base / fallback).exists():
        return fallback
    raise MatterSynthesisError(
        f"cannot create {preferred}: both {preferred} and {fallback} already exist"
    )


def _existing_entries(split: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
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


def ensure_matter_sections(
    chapters_path: str | Path,
    chunks_path: str | Path | None = None,
    *,
    source_file: str = "",
    book_title: str = "",
    book_author: str = "",
    create_front: bool = True,
    create_back: bool = True,
) -> MatterSynthesis:
    """Create the missing front/back matter chapter and chunk records.

    The split manifest and (when it exists) the chunk manifest are updated in
    place. A synthetic section is a normal section from here on: the matter
    agent localizes it, assembly joins it, postprocess styles it and bind puts it
    in the book.
    """
    base = chapters_dir(chapters_path)
    if not base.is_dir():
        raise MatterSynthesisError(f"chapters folder not found: {base}")
    split = load_split_manifest(base)
    if not split:
        raise MatterSynthesisError(
            f"no stage-1 manifest in {base}; run `split` before creating matter"
        )
    source = (
        str(source_file or "").strip()
        or str(split.get("source_path") or split.get("source_file") or "")
    )
    metadata = parse_book_metadata(source) if source else BookMetadata(
        title=str(book_title or "").strip(),
        author=str(book_author or "").strip(),
    )
    if book_title or book_author:
        metadata = BookMetadata(
            title=str(book_title or metadata.title).strip(),
            author=str(book_author or metadata.author).strip(),
            stem=metadata.stem,
        )

    existing_front, existing_back = _existing_entries(split)
    created: list[str] = []
    front_entry: dict[str, Any] | None = existing_front
    back_entry: dict[str, Any] | None = existing_back

    if create_front and front_entry is None:
        name = _free_name(base, FRONT_MATTER_FILENAME, "01-front-matter.md")
        text = _front_text(metadata)
        (base / name).write_text(text, encoding="utf-8")
        front_entry = _entry(
            base=base,
            filename=name,
            kind="front-matter",
            title=metadata.title or "Front matter",
            text=text,
            start_line=0,
            end_line=0,
        )
        created.append(name)

    if create_back and back_entry is None:
        source_lines = int(split.get("source_lines") or 0)
        name = _free_name(base, BACK_MATTER_FILENAME, "98-back-matter.md")
        text = _back_text(metadata)
        (base / name).write_text(text, encoding="utf-8")
        back_entry = _entry(
            base=base,
            filename=name,
            kind="back-matter",
            title=BACK_MATTER_TITLE,
            text=text,
            start_line=source_lines + 1 if source_lines else 10**9,
            end_line=source_lines + 1 if source_lines else 10**9,
        )
        created.append(name)

    if created:
        # Chapter records: front first, back last.
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
        (base / MANIFEST_FILENAME).write_text(
            json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    chunk_created: list[str] = []
    chunks = (
        Path(chunks_path).expanduser()
        if chunks_path is not None and str(chunks_path).strip()
        else base.parent / CHUNKS_DIRNAME
    )
    manifest = load_chunk_manifest(chunks) if chunks.is_dir() else {}
    if manifest:
        chunk_files = [
            entry for entry in manifest.get("files") or [] if isinstance(entry, dict)
        ]
        chapter_files = [
            entry for entry in manifest.get("chapters") or [] if isinstance(entry, dict)
        ]
        existing_chapters = {
            str(entry.get("chapter_filename") or "") for entry in chunk_files
        }
        # Front matter must sort before the body; shift the recorded indices,
        # never the existing files, so translated chunk names stay valid.
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
            file_record, chapter_record = _chunk_record(
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
            next_global = max((int(entry.get("global_index") or 0) for entry in chunk_files), default=-1) + 1
            next_chapter = max((int(entry.get("chapter_index") or 0) for entry in chapter_files), default=-1) + 1
            text = (base / back_entry["filename"]).read_text(encoding="utf-8")
            file_record, chapter_record = _chunk_record(
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
        _recompute_chunk_totals(manifest)
        (chunks / CHUNK_MANIFEST_FILENAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return MatterSynthesis(
        created=created,
        front_created=front_entry is not None and front_entry["filename"] in created,
        back_created=back_entry is not None and back_entry["filename"] in created,
        metadata=metadata,
        chunk_created=chunk_created,
    )


__all__ = [
    "BACK_MATTER_FILENAME",
    "BACK_MATTER_TITLE",
    "MatterSynthesis",
    "MatterSynthesisError",
    "ensure_matter_sections",
]
