"""Which existing folder a stage reads, and why.

Every stage command has the same three questions:

1. *Where is the book?* — a path, a bare filename in ``data/input/``, or the only
   book there.
2. *Which chapter text?* — an explicit ``--text-in``, the cleaned
   ``02-preprocessed/`` when it covers the current window, or the raw
   ``01-chapters/``.
3. *Which prior artifact?* — chunks, translated chapters or postprocessed
   chapters, with an explicit ``--*-in`` winning over the canonical layout.

Historically the pipeline answered these in ``Pipeline._cleaned_window`` and the
CLI answered them in ``_resolve_chapters_dir``/``_resolve_text_dir``, with the
two drifting. This module is the one answer. It reads :class:`StagePaths` (where
things are written) and returns paths that already exist (what a stage reads);
it never creates a folder.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from book_translator.stages.conversion.runner import classify_source, converted_markdown_path

from ..core.artifacts import sha256_file
from ..core.coerce import as_path as _path
from ..core.constants import CHAPTERS_DIRNAME, CHUNK_GLOSSARY_DIRNAME
from ..core.errors import ToolError
from ..documents.chapters import (
    chapters_dir,
    discover_chapters,
    has_chapter_files,
    has_chunk_files,
    has_postprocessed_chapter_files,
    has_translated_chapter_files,
    has_translation_files,
    postprocessed_chapters_dir,
    translated_chapters_dir,
)
from ..documents.layout import STAGE_DIRNAMES, STAGE_DIRS, book_root
from ..documents.manifest import load_split_manifest
from ..paths import resolve_book_path
from .paths import StagePaths

#: The chapter folders ``openings`` understands, most advanced first.
OPENING_STAGE_KEYS = (
    "postprocessed_chapters",
    "translated_chapters",
    "preprocessed",
    "chapters",
)


@dataclass(frozen=True)
class SourcePlan:
    """The original source and the Markdown stage 1 will read."""

    original: Path
    source: Path
    kind: str
    needs_conversion: bool
    markdown_dir: Path
    scratch_dir: Path | None = None


@dataclass(frozen=True)
class TextSource:
    """The chapter folder stages 3–5 read, plus why it was chosen."""

    path: Path
    label: str
    cleaned: bool = False


@dataclass(frozen=True)
class StageSources:
    """Every *input* folder one stage command may read.

    ``StagePaths`` says where a run writes; this value says which existing folder
    each stage reads, after the explicit flags and the fallback rules have been
    applied. Keeping them separate is what lets ``chunk --out /tmp/01-chapters``
    read the raw text while still writing to a layout rooted at ``/tmp``.
    """

    chapters: Path
    text: Path
    text_label: str
    chunk_input: Path
    chunks: Path
    chunk_glossary: Path
    glossary: Path
    translations: Path
    translated_chapters: Path
    postprocess: Path


# --------------------------------------------------------------------------- #
# the source book
# --------------------------------------------------------------------------- #
def resolve_source(
    book: str | Path | None,
    *,
    in_dir: str | Path | None = None,
    output_root: str | Path | None = None,
    dry_run: bool = False,
    scratch_dir: str | Path | None = None,
) -> SourcePlan:
    """Resolve the book argument and plan stage 1.

    A Markdown source is read directly. A non-Markdown source is converted into
    the canonical ``01-markdown/`` on a real run, and into a temporary folder on
    a dry run, so the later deterministic phases still have text to inspect
    without leaving a book artifact behind.
    """
    original = resolve_book_path(book, input_dir=in_dir).resolve()
    kind = classify_source(original)
    needs_conversion = kind != "markdown"
    if not needs_conversion:
        return SourcePlan(
            original=original,
            source=original,
            kind=kind,
            needs_conversion=False,
            markdown_dir=converted_markdown_path(original, output_root=output_root).parent,
        )

    if dry_run:
        scratch = (
            Path(scratch_dir).expanduser()
            if scratch_dir is not None
            else Path(tempfile.mkdtemp(prefix="book-translator-convert-"))
        )
        source = scratch / converted_markdown_path(original).name
        return SourcePlan(
            original=original,
            source=source,
            kind=kind,
            needs_conversion=True,
            markdown_dir=scratch,
            scratch_dir=scratch,
        )

    source = converted_markdown_path(original, output_root=output_root)
    return SourcePlan(
        original=original,
        source=source,
        kind=kind,
        needs_conversion=True,
        markdown_dir=source.parent,
    )


# --------------------------------------------------------------------------- #
# chapter text
# --------------------------------------------------------------------------- #
def resolve_chapters_input(ns: Any, paths: StagePaths) -> Path:
    """The raw chapter folder a command reads: ``--out``/``--chapters-in`` or default."""
    explicit = _path(getattr(ns, "out", None)) or _path(
        getattr(ns, "chapters_in", None)
    )
    if explicit is not None:
        return chapters_dir(explicit)
    return paths.chapters_dir


def resolve_chunk_input(ns: Any, paths: StagePaths) -> Path:
    """The chunks folder a command reads: ``--chunks-in`` or default."""
    explicit = _path(getattr(ns, "chunks_in", None))
    return explicit if explicit is not None else paths.chunks_dir


def resolve_translated_chapters_input(ns: Any, paths: StagePaths) -> Path:
    """The translated-chapters folder a command reads, with its two fallbacks."""
    explicit = _path(getattr(ns, "translated_chapters_in", None))
    if explicit is not None:
        return translated_chapters_dir(explicit)
    chunks = _path(getattr(ns, "chunks_in", None))
    if chunks is not None:
        return translated_chapters_dir(chunks)
    return paths.translated_chapters_dir


def resolve_postprocessed_input(ns: Any, paths: StagePaths) -> Path:
    """The postprocessed-chapters folder a command reads."""
    explicit = _path(getattr(ns, "postprocessed_in", None))
    return explicit if explicit is not None else paths.postprocess_dir


def resolve_openings_input(ns: Any, paths: StagePaths) -> Path:
    """The folder ``openings`` audits: explicit, or the book's most advanced one."""
    explicit = _path(getattr(ns, "chapters_in", None))
    if explicit is None:
        return most_advanced_chapters(paths)
    if explicit.is_file():
        explicit = explicit.parent
    if explicit.name in STAGE_DIRNAMES:
        return explicit
    for key in OPENING_STAGE_KEYS:
        candidate = explicit / STAGE_DIRS[key]
        if has_chapter_files(candidate):
            return candidate
    return explicit


def most_advanced_chapters(paths: StagePaths) -> Path:
    """The most advanced chapter folder that exists, in pipeline order."""
    for candidate in (
        paths.postprocess_dir,
        paths.translated_chapters_dir,
        paths.preprocessed_dir,
        paths.chapters_dir,
    ):
        if has_chapter_files(candidate):
            return candidate
    return paths.chapters_dir


def resolve_text_source(
    paths: StagePaths,
    ns: Any,
    *,
    chapter_limit: int | None = None,
    preprocess_enabled: bool = True,
) -> TextSource:
    """The chapter text stages 3–5 read: explicit, cleaned, else raw.

    ``--text-in`` always wins. Without it, the cleaned folder is used only when a
    completed stage-2 manifest proves it covers exactly the chapters in this
    window and neither the source nor the cleaned file changed since. Anything
    else falls back to the raw chapters rather than silently mixing cleaned and
    uncleaned text.
    """
    explicit = _path(getattr(ns, "text_in", None))
    if explicit is not None:
        return TextSource(chapters_dir(explicit), "explicit --text-in", cleaned=False)
    if not preprocess_enabled:
        return TextSource(paths.chapters_dir, "raw chapters (--no-preprocess)", cleaned=False)
    cleaned = cleaned_covers_window(paths, chapter_limit)
    if cleaned is not None:
        return TextSource(cleaned, "cleaned by stage 3", cleaned=True)
    if has_chapter_files(paths.preprocessed_dir):
        return TextSource(
            paths.chapters_dir,
            "cleaned text does not cover this window",
            cleaned=False,
        )
    return TextSource(paths.chapters_dir, "raw chapters", cleaned=False)


def resolve_chunk_source(
    paths: StagePaths, ns: Any, text: TextSource
) -> Path:
    """Stage 5's input: ``--chunk-in`` wins, else the resolved text folder."""
    explicit = _path(getattr(ns, "chunk_in", None))
    if explicit is not None:
        return chapters_dir(explicit)
    return text.path


def cleaned_covers_window(
    paths: StagePaths, chapter_limit: int | None
) -> Path | None:
    """``02-preprocessed/`` when it covers exactly the chapters this run reads.

    This is the old ``Pipeline._cleaned_window`` rule, now shared with every
    stage command. It returns ``None`` for a missing/partial/stale/incomplete
    cleaned folder, so a caller always falls back to the raw chapters rather than
    reading a half-cleaned book.
    """
    folder = paths.preprocessed_dir
    if not folder.is_dir() or not has_chapter_files(folder):
        return None
    manifest = load_split_manifest(folder)
    if not manifest or not manifest.get("ok") or manifest.get("dry_run"):
        return None
    entries = {
        str(entry.get("filename")): entry
        for entry in manifest.get("files") or []
        if isinstance(entry, dict) and entry.get("filename")
    }
    if not entries:
        return None
    try:
        chapters = discover_chapters(paths.chapters_dir)
    except (ToolError, OSError, ValueError):
        return None
    window = chapters[:chapter_limit] if chapter_limit else chapters
    if not window:
        return None
    for chapter in window:
        entry = entries.get(chapter.filename)
        cleaned = folder / chapter.filename
        if entry is None or not cleaned.is_file():
            return None
        recorded_lines = entry.get("source_lines")
        if recorded_lines is not None and int(recorded_lines) != chapter.line_count:
            return None
        recorded_source = str(entry.get("source_sha256") or "")
        if recorded_source and chapter.sha256 and recorded_source != chapter.sha256:
            return None
        recorded_clean = str(entry.get("sha256") or "")
        if recorded_clean and sha256_file(cleaned) != recorded_clean:
            return None
    return folder


def resolve_sources(
    ns: Any,
    paths: StagePaths,
    *,
    chapter_limit: int | None = None,
    preprocess_enabled: bool = True,
) -> StageSources:
    """Resolve every input folder one stage command may read.

    This is the bundle :func:`book_translator.pipeline.options.StageOptions.call`
    consumes, so a runner call never reaches back into ``argparse``.
    """
    text = resolve_text_source(
        paths, ns, chapter_limit=chapter_limit, preprocess_enabled=preprocess_enabled
    )
    chunks = resolve_chunk_input(ns, paths)
    return StageSources(
        chapters=resolve_chapters_input(ns, paths),
        text=text.path,
        text_label=text.label,
        chunk_input=resolve_chunk_source(paths, ns, text),
        chunks=chunks,
        chunk_glossary=_chunk_glossary_input(ns, paths),
        glossary=_path(getattr(ns, "glossary_in", None)) or paths.glossary_dir,
        translations=_path(getattr(ns, "translations_in", None))
        or paths.translations_dir,
        translated_chapters=resolve_translated_chapters_input(ns, paths),
        postprocess=resolve_postprocessed_input(ns, paths),
    )


def _chunk_glossary_input(ns: Any, paths: StagePaths) -> Path:
    explicit = _path(getattr(ns, "chunk_glossary_in", None))
    if explicit is None:
        return paths.chunk_glossary_dir
    if explicit.is_file():
        return explicit.parent
    if explicit.name == CHUNK_GLOSSARY_DIRNAME:
        return explicit
    nested = explicit / CHUNK_GLOSSARY_DIRNAME
    if nested.is_dir():
        return nested
    return explicit


# --------------------------------------------------------------------------- #
# what a phase can do with what is on disk
# --------------------------------------------------------------------------- #
def has_any_chapters(paths: StagePaths) -> bool:
    return any(
        has_chapter_files(folder)
        for folder in (
            paths.chapters_dir,
            paths.preprocessed_dir,
            paths.translated_chapters_dir,
            paths.postprocess_dir,
        )
    )


def has_any_chunks(paths: StagePaths) -> bool:
    return has_chunk_files(paths.chunks_dir)


def has_any_translations(paths: StagePaths) -> bool:
    return has_translation_files(paths.translations_dir)


def has_any_translated_chapters(paths: StagePaths) -> bool:
    return has_translated_chapter_files(paths.translated_chapters_dir)


def has_any_postprocessed_chapters(paths: StagePaths) -> bool:
    return has_postprocessed_chapter_files(paths.postprocess_dir)


def chapters_in_book_dir(book_dir: Path) -> Path:
    """Small helper for callers that only have a book folder."""
    return book_root(book_dir) / CHAPTERS_DIRNAME


def postprocessed_from_book_dir(book_dir: Path) -> Path:
    """Small helper for callers that only have a book folder."""
    return postprocessed_chapters_dir(book_dir)


__all__ = [
    "OPENING_STAGE_KEYS",
    "SourcePlan",
    "StageSources",
    "TextSource",
    "chapters_in_book_dir",
    "cleaned_covers_window",
    "has_any_chapters",
    "has_any_chunks",
    "has_any_postprocessed_chapters",
    "has_any_translated_chapters",
    "has_any_translations",
    "most_advanced_chapters",
    "postprocessed_from_book_dir",
    "resolve_chapters_input",
    "resolve_chunk_input",
    "resolve_chunk_source",
    "resolve_openings_input",
    "resolve_postprocessed_input",
    "resolve_source",
    "resolve_sources",
    "resolve_text_source",
    "resolve_translated_chapters_input",
]
