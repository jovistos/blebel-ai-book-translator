"""Discovery for the dictionary stage: paths, previous run and pending chapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..documents.chapters import Chapter, discover_chapters, reduced_dir, unique_dir
from .artifacts import manifest_files, previous_manifest, reusable


@dataclass
class DictionaryInputs:
    """Everything the run needs before it loads a detector or a tagger."""

    chapters: list[Chapter]
    pending: list[Chapter]
    source_dir: Path
    out_dir: Path
    unique_out_dir: Path
    settings: dict[str, Any]
    previous: dict[str, Any]
    previous_files: dict[str, dict[str, Any]]
    reusable: bool
    skipped_count: int
    limited: bool


def discover_dictionary(
    chapters_path: str,
    output_dir: str | None,
    unique_output_dir: str | None,
    *,
    limit: int | None,
    min_words: int,
    require_uppercase: bool,
    keep_types: tuple[str, ...],
    drop_temporal_only: bool,
    save_reduced: bool,
    save_unique: bool,
    unique_example_limit: int,
    skip_if_present: bool,
    example_limit: int,
    dry_run: bool,
) -> DictionaryInputs:
    chapters = discover_chapters(chapters_path)
    limited = bool(limit is not None and 0 < int(limit) < len(chapters))
    if limit is not None and int(limit) > 0:
        chapters = chapters[: int(limit)]
    source_dir = Path(chapters_path).expanduser()
    out_dir = Path(output_dir).expanduser() if output_dir else reduced_dir(source_dir)
    unique_out_dir = (
        Path(unique_output_dir).expanduser()
        if unique_output_dir
        else unique_dir(out_dir)
    )

    settings = {
        "min_words": int(min_words),
        "require_uppercase": bool(require_uppercase),
        "keep_types": list(keep_types),
        "drop_temporal_only": bool(drop_temporal_only),
        "save_reduced": bool(save_reduced),
        "save_unique": bool(save_unique),
        "unique_example_limit": int(unique_example_limit),
        "skip_if_present": bool(skip_if_present),
        "example_limit": int(example_limit),
    }

    previous = previous_manifest(out_dir)
    previous_files = manifest_files(previous)
    # A rerun may only reuse a reduction made from the same chapter folder.
    # Switching between raw ``01-chapters/`` and cleaned ``02-preprocessed/``
    # keeps every filename but changes the text under it, so the old reduced
    # files are a reduction of a different chapter and must be rebuilt.
    previous_source = str(previous.get("chapters_path") or "").strip()
    same_source = not previous_source or (
        Path(previous_source).expanduser().resolve() == source_dir.resolve()
    )
    reuse_enabled = bool(skip_if_present and save_reduced and not dry_run and same_source)
    pending = (
        [
            chapter
            for chapter in chapters
            if not reusable(
                chapter,
                out_dir,
                unique_out_dir,
                previous_files,
                save_unique=save_unique,
            )
        ]
        if reuse_enabled
        else list(chapters)
    )
    return DictionaryInputs(
        chapters=chapters,
        pending=pending,
        source_dir=source_dir,
        out_dir=out_dir,
        unique_out_dir=unique_out_dir,
        settings=settings,
        previous=previous,
        previous_files=previous_files,
        reusable=reuse_enabled,
        skipped_count=len(chapters) - len(pending),
        limited=limited,
    )


__all__ = ["DictionaryInputs", "discover_dictionary"]
