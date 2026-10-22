"""One chapter of the assembly stage: read, concatenate, write, or report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.artifacts import sha256_text, write_text
from .models import AssemblyInputs, TranslatedChapter


def chunk_path(translations_path: Path, entry: dict[str, Any]) -> Path:
    return translations_path / str(entry.get("filename") or "")


def assemble_chapter(
    name: str,
    index: int,
    *,
    inputs: AssemblyInputs,
    skip_if_present: bool,
    dry_run: bool,
) -> TranslatedChapter:
    """Read, concatenate and write one chapter; every failure is a record."""
    chapter_entries = inputs.grouped.get(name) or []
    record = TranslatedChapter(
        chapter_id=Path(name).stem,
        filename=name,
        chapter_index=index,
        chunk_count=len(chapter_entries),
        translated_chunks=len(chapter_entries),
        source_chars=sum(int(entry.get("chars") or 0) for entry in chapter_entries),
    )
    missing = [
        str(entry.get("filename") or "?")
        for entry in chapter_entries
        if not chunk_path(inputs.translations_folder, entry).is_file()
    ]
    if missing:
        # A chapter is all-or-nothing: a partial prefix would look like a
        # complete chapter to the reader, so it is reported instead.
        record.skipped = True
        record.translated_chunks = len(chapter_entries) - len(missing)
        record.missing_chunks = missing
        record.note = f"{len(missing)} of {len(chapter_entries)} chunk(s) not translated"
        return record

    try:
        parts = [
            chunk_path(inputs.translations_folder, entry).read_text(encoding="utf-8")
            for entry in chapter_entries
        ]
        text = "".join(parts)
    except OSError as exc:  # a read error is one chapter's failure
        record.ok = False
        record.error = f"{type(exc).__name__}: {exc}"
        return record

    record.translation_chars = len(text)
    record.sha256 = sha256_text(text)
    target = inputs.out_dir / name
    record.output_path = str(target)
    if dry_run:
        record.reused = bool(
            skip_if_present
            and target.is_file()
            and target.read_text(encoding="utf-8") == text
        )
        return record

    try:
        if skip_if_present and target.is_file() and target.read_text(
            encoding="utf-8"
        ) == text:
            record.reused = True
        else:
            write_text(target, text)
            record.reused = False
    except OSError as exc:
        record.ok = False
        record.error = f"{type(exc).__name__}: {exc}"
    return record


def failed_chapter(
    name: str, index: int, inputs: AssemblyInputs, exc: Exception
) -> TranslatedChapter:
    """A record for an unexpected per-chapter exception."""
    entries = inputs.grouped.get(name) or []
    return TranslatedChapter(
        chapter_id=Path(name).stem,
        filename=name,
        chapter_index=index,
        chunk_count=len(entries),
        translated_chunks=0,
        source_chars=sum(int(entry.get("chars") or 0) for entry in entries),
        ok=False,
        error=f"{type(exc).__name__}: {exc}",
    )


__all__ = ["assemble_chapter", "chunk_path", "failed_chapter"]
