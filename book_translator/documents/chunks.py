"""The read side of stage 5's chunk folder, and chunk-derived identity.

A chunk is the unit stages 5–8 pass around: a file whose name sorts into reading
order, plus the chapter, offsets and hash stage 5 recorded for it. That reading
used to live in ``chunking.runner.chunk_files`` and
``chunk_glossary.runner.discover_chunk_inputs``, so stage 8, stage 6, stage 6b
and stage 7 each imported another stage's runner to answer "what chunks are in
this folder?". The readers live here now, next to the other document helpers,
and the runners import them instead of each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.artifacts import sha256_text
from ..core.coerce import as_int as _int
from ..core.constants import CHUNKS_DIRNAME, GLOSSARY_DIRNAME, GLOSSARY_FILENAME
from ..core.errors import ToolError
from .layout import book_root
from .manifest import load_chunk_manifest


@dataclass
class ChunkInput:
    """One stage-5 chunk: where it is, what it covers and the text to tag."""

    filename: str
    path: str
    global_index: int
    chunk_index: int
    chapter_id: str = ""
    chapter_filename: str = ""
    start_char: int = 0
    end_char: int = 0
    start_line: int = 0
    end_line: int = 0
    token_count: int = 0
    sha256: str = ""
    chars: int = 0
    #: The chunk's exact text. Kept out of every serialised record.
    text: str = field(default="", repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "path": self.path,
            "global_index": self.global_index,
            "chunk_index": self.chunk_index,
            "chapter_id": self.chapter_id,
            "chapter_filename": self.chapter_filename,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "token_count": self.token_count,
            "chars": self.chars,
            "sha256": self.sha256,
        }


def resolve_chunks_dir(path: str | Path) -> Path:
    """Resolve a chunks folder, a book folder, or the manifest itself."""
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate.parent
    if candidate.name == CHUNKS_DIRNAME:
        return candidate
    nested = candidate / CHUNKS_DIRNAME
    if nested.is_dir():
        return nested
    return candidate


def resolve_glossary_path(
    glossary_path: str | Path | None,
    chunks_path: str | Path,
) -> Path:
    """Where ``glossary.json`` is: explicit, or the book folder's ``04-glossary/``."""
    if glossary_path:
        candidate = Path(glossary_path).expanduser()
        return candidate / GLOSSARY_FILENAME if candidate.is_dir() else candidate
    return book_root(chunks_path) / GLOSSARY_DIRNAME / GLOSSARY_FILENAME


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


def chunk_chapter_names(chunks_path: str | Path) -> list[str]:
    """The chapter filenames recorded by stage 5, in reading order.

    The chunk manifest's ``chapters`` list is authoritative and includes a
    chapter with zero chunks. A manifest without that list falls back to the
    unique ``chapter_filename`` values of the chunk files in global order.
    """
    manifest = load_chunk_manifest(chunks_path)
    names: list[str] = []
    seen: set[str] = set()
    for record in manifest.get("chapters") or []:
        if not isinstance(record, dict):
            continue
        name = str(record.get("filename") or "")
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    if names:
        return names

    for entry in chunk_files(chunks_path):
        name = str(entry.get("chapter_filename") or "")
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def discover_chunk_inputs(chunks_path: str | Path) -> tuple[list[ChunkInput], Path]:
    """Read every chunk file in stage-5 order, with its exact text.

    The stage-5 manifest is authoritative when it is there: it carries the global
    order, the chapter membership, the character/line spans and the chunk hash. A
    folder without one still works (the chunk files sort into reading order), but
    its files then only know their own name.
    """
    folder = resolve_chunks_dir(chunks_path)
    if not folder.is_dir():
        raise ToolError(f"chunks folder not found: {folder} — run `chunk` first")
    entries = chunk_files(folder)
    if not entries:
        raise ToolError(f"no chunk files in {folder} — run `chunk` first")

    ordered = sorted(
        entries,
        key=lambda entry: (_int(entry.get("global_index"), 0), str(entry.get("filename") or "")),
    )
    inputs: list[ChunkInput] = []
    missing: list[str] = []
    for position, entry in enumerate(ordered):
        filename = str(entry.get("filename") or Path(str(entry.get("path") or "")).name)
        if not filename:
            continue
        path = folder / filename
        if not path.is_file():
            missing.append(filename)
            continue
        text = path.read_text(encoding="utf-8")
        chapter_filename = str(entry.get("chapter_filename") or "")
        inputs.append(
            ChunkInput(
                filename=filename,
                path=str(path),
                global_index=_int(entry.get("global_index"), position),
                chunk_index=_int(entry.get("chunk_index"), 0),
                chapter_id=str(entry.get("chapter_id") or Path(chapter_filename).stem),
                chapter_filename=chapter_filename,
                start_char=_int(entry.get("start_char")),
                end_char=_int(entry.get("end_char")),
                start_line=_int(entry.get("start_line")),
                end_line=_int(entry.get("end_line")),
                token_count=_int(entry.get("token_count")),
                sha256=str(entry.get("sha256") or sha256_text(text)),
                chars=len(text),
                text=text,
            )
        )
    if missing:
        raise ToolError(
            f"{len(missing)} chunk file(s) recorded in the manifest are missing from "
            f"{folder}: {', '.join(missing[:5])}"
            + (" …" if len(missing) > 5 else "")
            + " — re-run `chunk` to rebuild them"
        )
    if not inputs:
        raise ToolError(f"no readable chunk files in {folder} — run `chunk` first")
    inputs.sort(key=lambda chunk: (chunk.global_index, chunk.filename))
    return inputs, folder


def chunk_glossary_filename(chunk_filename: str) -> str:
    """``00001-01-x-0000.md`` -> ``00001-01-x-0000.md.json``.

    Keeping the full chunk name (including its ``.md``) makes the mapping
    unambiguous and guarantees two chunks cannot collide on one output file. A
    consumer strips the final ``.json`` to find the chunk it describes.
    """
    name = Path(str(chunk_filename or "")).name or "chunk.md"
    return f"{name}.json"


def read_chunk_text(chunk: ChunkInput) -> str:
    """The chunk's text, read from disk when it was not carried on the record."""
    if chunk.text:
        return chunk.text
    try:
        return Path(chunk.path).read_text(encoding="utf-8")
    except OSError:
        return ""


def chunk_sha256(chunk: ChunkInput) -> str:
    """The chunk's recorded digest, or one computed from its text."""
    if chunk.sha256:
        return chunk.sha256
    return sha256_text(read_chunk_text(chunk))


__all__ = [
    "ChunkInput",
    "chunk_chapter_names",
    "chunk_files",
    "chunk_glossary_filename",
    "chunk_sha256",
    "discover_chunk_inputs",
    "read_chunk_text",
    "resolve_chunks_dir",
    "resolve_glossary_path",
]
