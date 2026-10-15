"""Chapter discovery: reading stage 1's output as the input of stage 2.

Stage 1 (``book-splitter``) writes ``chapters/<NN-name>.md`` plus
``_manifest.json``. Stage 2 needs exactly that: an ordered list of chapter files,
each with the source line range it came from, so a repair can always be checked
against the manifest of the stage that produced it.

Nothing here re-reads the book body; a chapter is described by its path, line
count and hashes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..core.constants import (
    CHAPTERS_DIRNAME,
    MANIFEST_FILENAME,
    PREPROCESSED_DIRNAME,
)
from ..core.errors import ToolError
from .io import load_lines


@dataclass
class Chapter:
    """One chapter file on disk, plus where it came from."""

    chapter_id: str
    path: str
    filename: str
    line_count: int
    bytes: int
    start_line: int | None = None
    end_line: int | None = None
    title: str = ""
    sha256: str = ""
    strategy: str = ""
    source_path: str = ""
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def chapters_dir(output_dir: str | Path) -> Path:
    """The folder stage 1 writes into, given either it or the book output folder."""
    out = Path(output_dir).expanduser()
    nested = out / CHAPTERS_DIRNAME
    if nested.is_dir():
        return nested
    return out


def preprocessed_dir(output_dir: str | Path) -> Path:
    """Default stage-2 output: a sibling of ``chapters/``, never over it."""
    out = Path(output_dir).expanduser()
    if out.name == CHAPTERS_DIRNAME:
        out = out.parent
    return out / PREPROCESSED_DIRNAME


def load_split_manifest(chapters_path: str | Path) -> dict:
    """Read ``_manifest.json`` from a chapters folder, or return ``{}``."""
    path = Path(chapters_path).expanduser()
    manifest_path = (path / MANIFEST_FILENAME) if path.is_dir() else path
    if not manifest_path.is_file():
        return {}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _resolve_manifest_entry(entry: dict, base: Path) -> Path:
    """Resolve a manifest filename against the manifest folder, then its parent."""
    name = str(entry.get("filename") or entry.get("file") or "")
    for candidate in (base / name, base.parent / name):
        if candidate.is_file():
            return candidate
    return base / name


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_chapters(
    chapters_path: str | Path,
    source_path: str | Path | None = None,
    verify_hashes: bool = False,
) -> list[Chapter]:
    """Ordered chapter list for a chapters folder.

    Uses ``_manifest.json`` when present (so line ranges and per-file SHA-256
    come from stage 1's own record) and falls back to globbing ``*.md`` around
    ``_``-prefixed bookkeeping files when it is missing.

    A manifest entry declaring a ``source_file`` that no longer matches the
    chapters folder is ignored for path resolution but kept in ``extras``.
    """
    base = chapters_dir(chapters_path)
    if not base.is_dir():
        raise ToolError(f"chapters folder not found: {base}")
    manifest = load_split_manifest(base)
    chapters: list[Chapter] = []
    seen: set[str] = set()

    for entry in manifest.get("files") or []:
        if not isinstance(entry, dict):
            continue
        path = _resolve_manifest_entry(entry, base)
        if not path.is_file():
            continue
        name = path.name
        if name in seen:
            continue
        seen.add(name)
        lines = load_lines(path)
        chapters.append(
            Chapter(
                chapter_id=path.stem,
                path=str(path),
                filename=name,
                line_count=len(lines),
                bytes=path.stat().st_size,
                start_line=entry.get("start_line"),
                end_line=entry.get("end_line"),
                title=str(entry.get("title") or ""),
                sha256=str(entry.get("sha256") or ""),
                strategy=str(manifest.get("strategy") or ""),
                source_path=str(source_path or entry.get("source_file") or ""),
                extras={
                    k: v
                    for k, v in entry.items()
                    if k not in {"filename", "start_line", "end_line", "title", "sha256", "bytes"}
                },
            )
        )

    if chapters:
        chapters.sort(key=lambda c: (c.start_line is None, c.start_line or 0, c.filename))
        if verify_hashes:
            for chapter in chapters:
                actual = file_sha256(chapter.path)
                if chapter.sha256 and actual != chapter.sha256:
                    raise ToolError(
                        f"chapter {chapter.filename} does not match the stage-1 manifest "
                        f"(sha256 {actual[:12]}… != {chapter.sha256[:12]}…)"
                    )
        return chapters

    for path in sorted(base.glob("*.md")):
        if path.name.startswith("_"):
            continue
        lines = load_lines(path)
        chapters.append(
            Chapter(
                chapter_id=path.stem,
                path=str(path),
                filename=path.name,
                line_count=len(lines),
                bytes=path.stat().st_size,
                source_path=str(source_path or ""),
            )
        )
    if not chapters:
        raise ToolError(f"no chapter files found in {base}")
    return chapters


def chapter_by_id(chapters: list[Chapter], chapter_id: str) -> Chapter:
    """Look a chapter up by id or filename, with a helpful error."""
    wanted = str(chapter_id or "").strip()
    for chapter in chapters:
        if wanted in {chapter.chapter_id, chapter.filename, chapter.path}:
            return chapter
    names = ", ".join(c.filename for c in chapters[:8])
    raise ToolError(f"unknown chapter {chapter_id!r}; known chapters: {names}")


def manifest_summary(chapters: list[Chapter], manifest: dict) -> dict:
    """A compact, content-free description of a chapter set."""
    return {
        "chapter_count": len(chapters),
        "strategy": manifest.get("strategy"),
        "source_file": manifest.get("source_file"),
        "total_lines": sum(c.line_count for c in chapters),
        "total_bytes": sum(c.bytes for c in chapters),
        "chapters": [
            {
                "chapter_id": c.chapter_id,
                "filename": c.filename,
                "lines": c.line_count,
                "bytes": c.bytes,
                "start_line": c.start_line,
                "end_line": c.end_line,
            }
            for c in chapters
        ],
    }
