"""Reading the pipeline's stage manifests.

Every stage writes a JSON manifest next to its artifacts, and the stage after it
needs to read that manifest without importing the stage that wrote it. These
readers used to live inside the stage runners (``assembly.runner``,
``chunking.runner``, ``translation.runner``…), which made a plain
"what did the previous stage record?" question a stage→stage import. They live
here now; a stage runner that still exports one of these names re-exports it from
this module.

All readers share one contract: a missing, unreadable or non-object manifest is
``{}`` rather than an error. "Nothing was recorded" is a legitimate state for a
first run and for a resume.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.artifacts import read_manifest
from ..core.constants import (
    ASSEMBLY_MANIFEST_FILENAME,
    CHUNK_GLOSSARY_MANIFEST_FILENAME,
    CHUNK_MANIFEST_FILENAME,
    CONVERT_MANIFEST_FILENAME,
    GLOSSARY_MANIFEST_FILENAME,
    MANIFEST_FILENAME,
    MATTER_MANIFEST_FILENAME,
    POSTPROCESS_MANIFEST_FILENAME,
    PREPROCESS_MANIFEST_FILENAME,
    TRANSLATION_MANIFEST_FILENAME,
)


def load_manifest(directory_or_file: str | Path, filename: str) -> dict[str, Any]:
    """Read one named manifest from a folder (or from the file itself)."""
    return read_manifest(directory_or_file, filename)


def load_split_manifest(chapters_path: str | Path) -> dict[str, Any]:
    """Read a chapter folder's manifest, or return ``{}``.

    Stage 1 writes ``_manifest.json``; stage 2 writes
    ``_preprocess_manifest.json`` beside the cleaned chapters in the same shape
    (same filenames and order, same titles and source line ranges, plus a hash of
    the cleaned file). Reading either with one function is what lets a downstream
    stage treat ``02-preprocessed/`` exactly like ``01-chapters/``. Stage 1's
    manifest wins when both are present.
    """
    path = Path(chapters_path).expanduser()
    if not path.is_dir():
        return read_manifest(path)
    for filename in (MANIFEST_FILENAME, PREPROCESS_MANIFEST_FILENAME):
        candidate = path / filename
        if candidate.is_file():
            return read_manifest(candidate)
    return {}


def load_convert_manifest(markdown_path: str | Path) -> dict[str, Any]:
    """Read stage 0's conversion manifest from a markdown folder or file."""
    path = Path(markdown_path).expanduser()
    if path.is_file() and path.name != CONVERT_MANIFEST_FILENAME:
        path = path.parent
    return read_manifest(path, CONVERT_MANIFEST_FILENAME)


def load_glossary_manifest(glossary_path: str | Path) -> dict[str, Any]:
    """Read stage 4's glossary manifest from a folder or the file itself."""
    return read_manifest(glossary_path, GLOSSARY_MANIFEST_FILENAME)


def load_chunk_manifest(chunks_path: str | Path) -> dict[str, Any]:
    """Read ``_chunk_manifest.json`` from a chunks folder (or the file itself)."""
    return read_manifest(chunks_path, CHUNK_MANIFEST_FILENAME)


def load_chunk_glossary_manifest(chunks_path: str | Path) -> dict[str, Any]:
    """Read ``_chunk_glossary_manifest.json`` from a folder (or a file)."""
    return read_manifest(chunks_path, CHUNK_GLOSSARY_MANIFEST_FILENAME)


def load_matter_manifest(matter_path: str | Path) -> dict[str, Any]:
    """Read ``_matter_manifest.json`` from the matter stage folder."""
    return read_manifest(matter_path, MATTER_MANIFEST_FILENAME)


def load_translation_manifest(translations_path: str | Path) -> dict[str, Any]:
    """Read ``_translation_manifest.json`` from an output folder (or a file)."""
    return read_manifest(translations_path, TRANSLATION_MANIFEST_FILENAME)


def load_assembly_manifest(output_dir: str | Path) -> dict[str, Any]:
    """Read ``_translated_chapters_manifest.json`` from a folder or a file."""
    return read_manifest(output_dir, ASSEMBLY_MANIFEST_FILENAME)


def load_postprocess_manifest(output_dir: str | Path) -> dict[str, Any]:
    """Read ``_postprocess_manifest.json`` from a folder."""
    return read_manifest(output_dir, POSTPROCESS_MANIFEST_FILENAME)


__all__ = [
    "load_assembly_manifest",
    "load_chunk_glossary_manifest",
    "load_chunk_manifest",
    "load_convert_manifest",
    "load_glossary_manifest",
    "load_manifest",
    "load_matter_manifest",
    "load_postprocess_manifest",
    "load_split_manifest",
    "load_translation_manifest",
]
