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
    CHUNK_GLOSSARY_DIRNAME,
    CHUNKS_DIRNAME,
    GLOSSARY_DIRNAME,
    MANIFEST_FILENAME,
    PREPROCESSED_DIRNAME,
    REDUCED_DIRNAME,
    TRANSLATED_CHAPTERS_DIRNAME,
    TRANSLATION_PROMPTS_DIRNAME,
    TRANSLATIONS_DIRNAME,
    UNIQUE_DIRNAME,
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


def reduced_dir(output_dir: str | Path) -> Path:
    """Default stage-3 output: a sibling of ``chapters/``, never over it.

    Stage 3 writes the reduced chapters (entity-bearing sentences only) the same
    way stage 1 writes chapters: one file per chapter, same filenames, so the two
    folders line up one to one and can be diffed while the filters are tuned.
    """
    out = Path(output_dir).expanduser()
    if out.name == CHAPTERS_DIRNAME:
        out = out.parent
    return out / REDUCED_DIRNAME


def unique_dir(output_dir: str | Path) -> Path:
    """Default stage-3b output: the further-reduced chapters, a sibling of ``reduced/``.

    The second funnel step keeps the same filenames as ``reduced/`` and
    ``chapters/``, so the three folders line up one to one and can be diffed while
    the extra filter is tuned. The folder is derived from wherever the reduced
    chapters went — an explicit ``--reduced-out`` moves it too, so the two outputs
    stay next to each other.
    """
    out = Path(output_dir).expanduser()
    if out.name == CHAPTERS_DIRNAME:
        # Handed the chapters folder instead of the reduced one (``--out`` does
        # this), so step over to the sibling the stage would actually write.
        out = out.parent / REDUCED_DIRNAME
    return out.parent / UNIQUE_DIRNAME


def glossary_dir(output_dir: str | Path) -> Path:
    """Default stage-4 output: a sibling of ``chapters/``, never over it.

    The glossary is one book-level artifact (plus its manifest), not one file per
    chapter, so the folder is a plain home for it rather than a third parallel
    folder of chapter files.
    """
    out = Path(output_dir).expanduser()
    if out.name == CHAPTERS_DIRNAME:
        out = out.parent
    return out / GLOSSARY_DIRNAME


def chunks_dir(output_dir: str | Path) -> Path:
    """Default stage-5 output: a sibling of ``chapters/``, never over it.

    Chunks are a flat, globally ordered folder (not one sub-folder per chapter):
    the file names sort into reading order, and ``_chunk_manifest.json`` carries
    the chapter membership and the character offsets of every slice. Handed a
    ``preprocessed/`` folder it puts ``chunks/`` beside it, in the same book
    folder, which is where a translation run expects to find it.
    """
    out = Path(output_dir).expanduser()
    if out.name in {
        CHAPTERS_DIRNAME,
        PREPROCESSED_DIRNAME,
        REDUCED_DIRNAME,
        UNIQUE_DIRNAME,
        GLOSSARY_DIRNAME,
        CHUNK_GLOSSARY_DIRNAME,
        TRANSLATIONS_DIRNAME,
        TRANSLATION_PROMPTS_DIRNAME,
        TRANSLATED_CHAPTERS_DIRNAME,
    }:
        out = out.parent
    return out / CHUNKS_DIRNAME


def chunk_glossary_dir(output_dir: str | Path) -> Path:
    """Default stage-6 output: a sibling of ``chunks/``, never over it.

    One JSON file per chunk, named after the chunk (``<chunk>.json``), so a
    translation run can go from a chunk it is about to send to the exact terms it
    has to obey without reading the book-level manifest. Handed the chunks folder
    it puts ``chunk-glossary/`` beside it, in the same book folder; handed the
    book folder it does the same.
    """
    out = Path(output_dir).expanduser()
    if out.name == CHUNK_GLOSSARY_DIRNAME:
        return out
    if out.name in {
        CHAPTERS_DIRNAME,
        PREPROCESSED_DIRNAME,
        REDUCED_DIRNAME,
        UNIQUE_DIRNAME,
        GLOSSARY_DIRNAME,
        CHUNKS_DIRNAME,
        CHUNK_GLOSSARY_DIRNAME,
        TRANSLATIONS_DIRNAME,
        TRANSLATION_PROMPTS_DIRNAME,
        TRANSLATED_CHAPTERS_DIRNAME,
    }:
        out = out.parent
    return out / CHUNK_GLOSSARY_DIRNAME


def translations_dir(output_dir: str | Path) -> Path:
    """Default stage-7 output: a sibling of ``chunks/``, never over it.

    The translated chunks keep the chunk filenames, so ``chunks/`` and
    ``translations/`` line up one to one and concatenating the translations in
    filename order rebuilds the translated book. Handed the book folder it writes
    ``translations/`` in it; handed ``chunks/`` (or any other sibling artifact
    folder) it steps over to the book folder first.
    """
    out = Path(output_dir).expanduser()
    if out.name == TRANSLATIONS_DIRNAME:
        return out
    if out.name in {
        CHAPTERS_DIRNAME,
        PREPROCESSED_DIRNAME,
        REDUCED_DIRNAME,
        UNIQUE_DIRNAME,
        GLOSSARY_DIRNAME,
        CHUNKS_DIRNAME,
        CHUNK_GLOSSARY_DIRNAME,
        TRANSLATION_PROMPTS_DIRNAME,
        TRANSLATED_CHAPTERS_DIRNAME,
    }:
        out = out.parent
    return out / TRANSLATIONS_DIRNAME


def translation_prompts_dir(output_dir: str | Path) -> Path:
    """Default stage-7 debug output: the prompt/input record of every chunk.

    A sibling of ``translations/``, never inside it: the translation folder is the
    book's text and should stay clean enough to concatenate, while this folder is
    for a human debugging one call. Handed the translations folder it puts the
    debug folder beside it.
    """
    out = Path(output_dir).expanduser()
    if out.name == TRANSLATION_PROMPTS_DIRNAME:
        return out
    if out.name in {
        CHAPTERS_DIRNAME,
        PREPROCESSED_DIRNAME,
        REDUCED_DIRNAME,
        UNIQUE_DIRNAME,
        GLOSSARY_DIRNAME,
        CHUNKS_DIRNAME,
        CHUNK_GLOSSARY_DIRNAME,
        TRANSLATIONS_DIRNAME,
        TRANSLATED_CHAPTERS_DIRNAME,
    }:
        out = out.parent
    return out / TRANSLATION_PROMPTS_DIRNAME


def translated_chapters_dir(output_dir: str | Path) -> Path:
    """Default stage-8 output: the translated chapters, a sibling of ``chapters/``.

    Handed the book folder it writes ``translated-chapters/`` in it; handed any
    of the stage's sibling artifact folders (``chapters/``, ``chunks/``,
    ``translations/``…) it steps over to the book folder first. The output is a
    separate folder on purpose: ``chapters/`` is the source text and must never be
    overwritten by the translation.
    """
    out = Path(output_dir).expanduser()
    if out.name == TRANSLATED_CHAPTERS_DIRNAME:
        return out
    if out.name in {
        CHAPTERS_DIRNAME,
        PREPROCESSED_DIRNAME,
        REDUCED_DIRNAME,
        UNIQUE_DIRNAME,
        GLOSSARY_DIRNAME,
        CHUNKS_DIRNAME,
        CHUNK_GLOSSARY_DIRNAME,
        TRANSLATIONS_DIRNAME,
        TRANSLATION_PROMPTS_DIRNAME,
    }:
        out = out.parent
    return out / TRANSLATED_CHAPTERS_DIRNAME


def has_translation_files(path: str | Path) -> bool:
    """True when ``path`` holds at least one translated chunk file."""
    folder = Path(path).expanduser()
    if not folder.is_dir():
        return False
    return any(
        p.is_file() and p.suffix.lower() == ".md" and not p.name.startswith("_")
        for p in folder.glob("*.md")
    )


def has_translated_chapter_files(path: str | Path) -> bool:
    """True when ``path`` holds at least one stage-8 translated chapter file.

    Bookkeeping files (``_translated_chapters_manifest.json`` and its trace) do
    not count.
    """
    folder = Path(path).expanduser()
    if not folder.is_dir():
        return False
    return any(
        p.is_file() and p.suffix.lower() == ".md" and not p.name.startswith("_")
        for p in folder.glob("*.md")
    )


def has_chunk_files(path: str | Path) -> bool:
    """True when ``path`` holds at least one stage-5 chunk file.

    Bookkeeping files (``_chunk_manifest.json``, ``_chunk_trace.json``) do not
    count: a folder holding nothing but its own manifest is not a finished chunk
    run, and stage 6 has nothing to attach a glossary to.
    """
    folder = Path(path).expanduser()
    if not folder.is_dir():
        return False
    return any(
        p.is_file() and p.suffix.lower() == ".md" and not p.name.startswith("_")
        for p in folder.glob("*.md")
    )


def has_chapter_files(path: str | Path) -> bool:
    """True when ``path`` already holds at least one chapter file.

    Bookkeeping files (``_manifest.json``, ``_agent_trace.json``) do not count: a
    folder holding nothing but its own manifest is not a finished split.
    """
    folder = Path(path).expanduser()
    if not folder.is_dir():
        return False
    return any(p.is_file() and not p.name.startswith("_") for p in folder.glob("*.md"))


def chapter_filenames(path: str | Path) -> set[str]:
    """The ``*.md`` filenames in a folder, bookkeeping files excluded."""
    folder = Path(path).expanduser()
    if not folder.is_dir():
        return set()
    return {
        p.name for p in folder.glob("*.md") if p.is_file() and not p.name.startswith("_")
    }


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
