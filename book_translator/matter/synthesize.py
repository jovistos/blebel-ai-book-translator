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

The work is split three ways so each part is readable on its own:

``generate``
    the English text of a synthetic section (facts only, no model);
``records``
    the split-manifest entry and the chunk file/manifest records;
``synthesize``
    this orchestrator: decide what is missing, write it, update both manifests.

Nothing here calls a model. The matter agent still localizes the generated
English text; the generated section only supplies the facts it must not lose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.artifacts import write_text
from ..core.constants import CHUNKS_DIRNAME, FRONT_MATTER_FILENAME
from ..documents.book import BookMetadata
from ..documents.chapters import chapters_dir
from ..documents.manifest import load_split_manifest
from .generate import (
    BACK_MATTER_FILENAME,
    BACK_MATTER_TITLE,
    back_matter_text,
    front_matter_text,
)
from .metadata import parse_book_metadata
from .records import (
    chapter_entry,
    existing_matter_entries,
    update_chunk_manifest,
    update_split_manifest,
)


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


def _free_name(base: Path, preferred: str, fallback: str) -> str:
    if not (base / preferred).exists():
        return preferred
    if not (base / fallback).exists():
        return fallback
    raise MatterSynthesisError(
        f"cannot create {preferred}: both {preferred} and {fallback} already exist"
    )


def _resolve_metadata(
    split: dict[str, Any],
    *,
    source_file: str,
    book_title: str,
    book_author: str,
) -> BookMetadata:
    """Title/author evidence: explicit arguments, else the input filename."""
    source = (
        str(source_file or "").strip()
        or str(split.get("source_path") or split.get("source_file") or "")
    )
    metadata = (
        parse_book_metadata(source)
        if source
        else BookMetadata(
            title=str(book_title or "").strip(),
            author=str(book_author or "").strip(),
        )
    )
    if book_title or book_author:
        metadata = BookMetadata(
            title=str(book_title or metadata.title).strip(),
            author=str(book_author or metadata.author).strip(),
            stem=metadata.stem,
        )
    return metadata


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
    metadata = _resolve_metadata(
        split,
        source_file=source_file,
        book_title=book_title,
        book_author=book_author,
    )

    existing_front, existing_back = existing_matter_entries(split)
    created: list[str] = []
    front_entry = existing_front
    back_entry = existing_back

    if create_front and front_entry is None:
        name = _free_name(base, FRONT_MATTER_FILENAME, "01-front-matter.md")
        text = front_matter_text(metadata)
        write_text(base / name, text)
        front_entry = chapter_entry(
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
        text = back_matter_text(metadata)
        write_text(base / name, text)
        back_entry = chapter_entry(
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
        update_split_manifest(base, split, front_entry, back_entry)

    chunks = (
        Path(chunks_path).expanduser()
        if chunks_path is not None and str(chunks_path).strip()
        else base.parent / CHUNKS_DIRNAME
    )
    chunk_created = update_chunk_manifest(base, chunks, front_entry, back_entry)

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
