"""Chapter discovery: reading stage 1's output as the input of stage 2.

Stage 1 (``book-splitter``) writes ``01-chapters/<NN-name>.md`` plus
``_manifest.json``. Stage 2 needs exactly that: an ordered list of chapter files,
each with the source line range it came from, so a repair can always be checked
against the manifest of the stage that produced it.

Nothing here re-reads the book body; a chapter is described by its path, line
count and hashes.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..core.artifacts import sha256_file
from ..core.constants import (
    CHAPTERS_DIRNAME,
    TOC_FILENAME,
    TRANSLATED_BOOK_FILENAME,
    TRANSLATED_EPUB_FILENAME,
)
from ..core.errors import ToolError
from .io import load_lines
from .layout import book_root, resolve_stage_dir
from .manifest import load_split_manifest


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
    """Default stage-2 output: a sibling of ``01-chapters/``, never over it."""
    return resolve_stage_dir(output_dir, "preprocessed")


def reduced_dir(output_dir: str | Path) -> Path:
    """Default stage-3 output: a sibling of ``01-chapters/``, never over it.

    Stage 3 writes the reduced chapters (entity-bearing sentences only) the same
    way stage 1 writes chapters: one file per chapter, same filenames, so the two
    folders line up one to one and can be diffed while the filters are tuned.
    Handed a ``02-preprocessed/`` folder it still writes beside it, in the same
    book folder, so the cleaned text and its reduction stay siblings.
    """
    return resolve_stage_dir(output_dir, "reduced")


def unique_dir(output_dir: str | Path) -> Path:
    """Default stage-3b output: the further-reduced chapters, a sibling of ``03-reduced/``.

    The second funnel step keeps the same filenames as ``03-reduced/`` and
    ``01-chapters/``, so the three folders line up one to one and can be diffed
    while the extra filter is tuned. The folder is derived from wherever the
    reduced chapters went — an explicit ``--reduced-out`` moves it too, so the two
    outputs stay next to each other.
    """
    return resolve_stage_dir(output_dir, "unique")


def glossary_dir(output_dir: str | Path) -> Path:
    """Default stage-4 output: a sibling of ``01-chapters/``, never over it.

    The glossary is one book-level artifact (plus its manifest), not one file per
    chapter, so the folder is a plain home for it rather than a third parallel
    folder of chapter files. Handed the cleaned ``02-preprocessed/`` folder it
    still writes beside it, in the same book folder.
    """
    return resolve_stage_dir(output_dir, "glossary")


def chunks_dir(output_dir: str | Path) -> Path:
    """Default stage-5 output: a sibling of ``01-chapters/``, never over it.

    Chunks are a flat, globally ordered folder (not one sub-folder per chapter):
    the file names sort into reading order, and ``_chunk_manifest.json`` carries
    the chapter membership and the character offsets of every slice. Handed a
    ``02-preprocessed/`` folder it puts ``05-chunks/`` beside it, in the same book
    folder, which is where a translation run expects to find it.
    """
    return resolve_stage_dir(output_dir, "chunks")


def chunk_glossary_dir(output_dir: str | Path) -> Path:
    """Default stage-6 output: a sibling of ``05-chunks/``, never over it.

    One JSON file per chunk, named after the chunk (``<chunk>.json``), so a
    translation run can go from a chunk it is about to send to the exact terms it
    has to obey without reading the book-level manifest. Handed the chunks folder
    it puts ``06-chunk-glossary/`` beside it, in the same book folder; handed the
    book folder it does the same.
    """
    return resolve_stage_dir(output_dir, "chunk_glossary")


def translations_dir(output_dir: str | Path) -> Path:
    """Default stage-7 output: a sibling of ``05-chunks/``, never over it.

    The translated chunks keep the chunk filenames, so ``05-chunks/`` and
    ``07-translations/`` line up one to one and concatenating the translations in
    filename order rebuilds the translated book. Handed the book folder it writes
    ``07-translations/`` in it; handed ``05-chunks/`` (or any other sibling artifact
    folder) it steps over to the book folder first.
    """
    return resolve_stage_dir(output_dir, "translations")


def translation_prompts_dir(output_dir: str | Path) -> Path:
    """Default stage-7 debug output: the prompt/input record of every chunk.

    A sibling of ``07-translations/``, never inside it: the translation folder is the
    book's text and should stay clean enough to concatenate, while this folder is
    for a human debugging one call. Handed the translations folder it puts the
    debug folder beside it.
    """
    return resolve_stage_dir(output_dir, "translation_prompts")


def matter_dir(output_dir: str | Path) -> Path:
    """Default front/back matter stage output: its book-level plan and trace.

    A sibling of ``07-translations/`` (the translated chunks themselves go there, so
    stage 8 needs no new join rule). Handed any stage folder it steps over to the
    book folder first.
    """
    return resolve_stage_dir(output_dir, "matter")


def matter_prompts_dir(output_dir: str | Path) -> Path:
    """Default matter-agent debug output: the whole input of every localization."""
    return resolve_stage_dir(output_dir, "matter_prompts")


def translated_chapters_dir(output_dir: str | Path) -> Path:
    """Default stage-8 output: the translated chapters, a sibling of ``01-chapters/``.

    Handed the book folder it writes ``08-translated-chapters/`` in it; handed any
    of the stage's sibling artifact folders (``01-chapters/``, ``05-chunks/``,
    ``07-translations/``…) it steps over to the book folder first. The output is a
    separate folder on purpose: ``01-chapters/`` is the source text and must never be
    overwritten by the translation.
    """
    return resolve_stage_dir(output_dir, "translated_chapters")


def postprocessed_chapters_dir(output_dir: str | Path) -> Path:
    """Default stage-9 output: the postprocessed chapters, never over its input.

    Handed the book folder it writes ``09-postprocessed-chapters/`` in it; handed
    any sibling artifact folder (``08-translated-chapters/``, ``01-chapters/``,
    ``05-chunks/``…) it steps over to the book folder first. The translated chapter
    files are input and are never edited in place.
    """
    return resolve_stage_dir(output_dir, "postprocessed_chapters")


def is_toc_filename(name: str) -> bool:
    """True for the table-of-contents file (created or pre-existing).

    The TOC is book structure, not a chapter: the stage-9 inventory, the binder
    and the opening audit all skip it.
    """
    filename = Path(str(name or "")).name
    lower = filename.lower()
    if filename == TOC_FILENAME:
        return True
    stem = Path(lower).stem
    if stem in {"toc", "contents", "sadrzaj", "sadržaj", "table-of-contents"}:
        return True
    compact = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return compact in {"table-of-contents", "toc", "contents"} or compact.startswith(
        "table-of-contents"
    )


def translated_book_path(output_dir: str | Path) -> Path:
    """Where stage 10 writes the whole translated book as one Markdown file.

    A sibling of ``09-postprocessed-chapters/`` (or of the book folder, when that
    folder is handed in), never inside the chapter folder: the folder stays the
    concatenation source, while the single file is the reading copy.
    """
    out = Path(output_dir).expanduser()
    if out.suffix.lower() == ".md":
        return out
    return book_root(out) / TRANSLATED_BOOK_FILENAME


def translated_epub_path(output_dir: str | Path) -> Path:
    """Where stage 11 writes the book as an EPUB 3 file.

    A sibling of ``09-postprocessed-chapters/`` (or of the book folder, when that
    folder is handed in), next to stage 10's Markdown reading copy. An explicit
    ``.epub`` path is returned unchanged.
    """
    out = Path(output_dir).expanduser()
    if out.suffix.lower() == ".epub":
        return out
    return book_root(out) / TRANSLATED_EPUB_FILENAME


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


def has_postprocessed_chapter_files(path: str | Path) -> bool:
    """True when ``path`` holds at least one stage-9 postprocessed chapter.

    The table-of-contents file is book structure, not a chapter, so it is
    excluded. Bookkeeping files are excluded by their ``_`` prefix.
    """
    folder = Path(path).expanduser()
    if not folder.is_dir():
        return False
    return any(
        p.is_file()
        and p.suffix.lower() == ".md"
        and not p.name.startswith("_")
        and p.name != TOC_FILENAME
        and "table-of-contents" not in p.name.lower()
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


def _resolve_manifest_entry(entry: dict, base: Path) -> Path:
    """Resolve a manifest filename against the manifest folder, then its parent."""
    name = str(entry.get("filename") or entry.get("file") or "")
    for candidate in (base / name, base.parent / name):
        if candidate.is_file():
            return candidate
    return base / name


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's bytes."""
    return sha256_file(path)


def discover_chapters(
    chapters_path: str | Path,
    source_path: str | Path | None = None,
    verify_hashes: bool = False,
) -> list[Chapter]:
    """Ordered chapter list for a chapters folder.

    Uses ``_manifest.json`` when present (so line ranges and per-file SHA-256
    come from stage 1's own record). A cleaned ``02-preprocessed/`` folder has no
    ``_manifest.json`` but does have ``_preprocess_manifest.json`` in the same
    shape, so it reads the same way; a folder with neither falls back to globbing
    ``*.md`` around ``_``-prefixed bookkeeping files.

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
