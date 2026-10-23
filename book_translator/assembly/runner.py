"""Stage 8: concatenate the translated chunks back into chapters.

Stage 5 cut every chapter into exact slices; stage 7 translated those slices one
call at a time. This stage puts the translated slices back together, using the
membership and order stage 5 recorded:

```
05-chunks/00001-01-predgovor-0000.md          07-translations/00001-...-0000.md
05-chunks/00002-01-predgovor-0001.md          07-translations/00002-...-0001.md
                    \\                    /
                     08-translated-chapters/01-predgovor.md
```

Two rules make the output trustworthy:

* **Group by the recorded chapter, never by filename guesses.** The chunk
  manifest says which chapter each chunk came from and its ``chunk_index``
  inside that chapter; the translation folder is joined on the chunk's own name.
  The concatenation inserts no separator, exactly like stage 5's reconstruction.
* **A chapter is written only when every one of its chunks was translated.**
  A partial translation (``translate --limit`` stopping mid-chapter) is reported
  as an incomplete chapter and skipped, not written as a plausible-looking but
  truncated chapter.

The folder is separate from ``01-chapters/`` on purpose: the source text is never
overwritten by the translation. ``--first-chapters N`` narrows the stage to a
prefix, and ``--skip-if-present`` reuses a chapter file whose assembled text is
already exactly what the current translations produce, so growing ``N`` only
assembles the new chapters.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..core.constants import (
    ASSEMBLY_MANIFEST_FILENAME,
    ASSEMBLY_TRACE_FILENAME,
)
from ..core.errors import ToolError
from ..core.run import RunContext, run_items
from ..core.store import ArtifactStore
from ..documents.chapters import translated_chapters_dir, translations_dir
from ..documents.chunks import chunk_files, resolve_chunks_dir
from ..documents.manifest import load_assembly_manifest
from ..documents.window import (
    body_chapter_names,
    chunk_chapter_names,
    is_body_chapter_name,
    normalize_limit,
)
from .artifacts import (
    assembly_manifest_payload,
    write_assembly_manifest,
    write_assembly_trace,
)
from .models import (
    AssemblyInputs,
    AssemblyResult,
    Clock,
    ProgressHook,
    TranslatedChapter,
)
from .worker import assemble_chapter, failed_chapter


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def resolve_translations_dir(
    translations_path: str | Path | None,
    chunks_path: str | Path,
) -> Path:
    """Where the translated chunks are: explicit, or the chunks folder's sibling."""
    if translations_path is not None and str(translations_path).strip():
        return Path(translations_path).expanduser()
    return translations_dir(resolve_chunks_dir(chunks_path))


def resolve_output_dir(
    output_dir: str | Path | None,
    chunks_path: str | Path,
) -> Path:
    """Where the translated chapters go: explicit, or the book's own folder."""
    if output_dir is not None and str(output_dir).strip():
        return Path(output_dir).expanduser()
    return translated_chapters_dir(resolve_chunks_dir(chunks_path))


def _chapter_records(
    chunks_path: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    """The chunk entries and the chapter names, in reading order.

    Stage 5's manifest is authoritative. A folder without one is still readable
    for chunk order, but then a chapter cannot be derived from a chunk reliably,
    so this is reported as an actionable setup error instead of guessed.
    """
    entries = chunk_files(chunks_path)
    if not entries:
        raise ToolError(f"no chunk files in {chunks_path} — run `chunk` first")
    names = chunk_chapter_names(chunks_path)
    if not names:
        raise ToolError(
            f"the chunk manifest in {chunks_path} records no chapter membership — "
            "re-run `chunk` so every chunk records the chapter it came from"
        )
    return entries, names


def _group_by_chapter(
    entries: Sequence[dict[str, Any]],
    names: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    """Chunk entries grouped by chapter, keeping only the known chapter names."""
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    for entry in entries:
        name = str(entry.get("chapter_filename") or "")
        if not name:
            raise ToolError(
                "cannot assemble chapters: a chunk does not record its chapter — "
                "re-run `chunk`"
            )
        if name in grouped:
            grouped[name].append(entry)
    for chapter_entries in grouped.values():
        chapter_entries.sort(
            key=lambda entry: (
                int(entry.get("chunk_index") or 0),
                int(entry.get("global_index") or 0),
                str(entry.get("filename") or ""),
            )
        )
    return grouped


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def _discover_assembly(
    chunks_path: str | Path,
    translations_path: str | Path | None,
    output_dir: str | Path | None,
    first_chapters: int | None,
) -> AssemblyInputs:
    """Resolve the folders, the chapter window and the chunk grouping."""
    chunk_folder = resolve_chunks_dir(chunks_path)
    translations_folder = resolve_translations_dir(translations_path, chunk_folder)
    out_dir = resolve_output_dir(output_dir, chunk_folder)
    if not translations_folder.is_dir():
        raise ToolError(
            f"translations folder not found: {translations_folder} — run `translate` first"
        )

    entries, all_names = _chapter_records(chunk_folder)
    limit = normalize_limit(first_chapters)
    body_names = body_chapter_names(all_names)
    limited = limit is not None and limit < len(body_names)
    # A window limits the *body* chapters, never the front/back matter: a book
    # built from the first N chapters still needs its title page and its back
    # page, and the matter stage localizes them for exactly this assembly.
    if limit is None:
        names = list(all_names)
    else:
        selected = set(body_names[:limit])
        names = [
            name
            for name in all_names
            if not is_body_chapter_name(name) or name in selected
        ]
    return AssemblyInputs(
        chunk_folder=chunk_folder,
        translations_folder=translations_folder,
        out_dir=out_dir,
        names=names,
        grouped=_group_by_chapter(entries, names),
        limit=limit,
        limited=limited,
    )


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def run_assembly(
    chunks_path: str | Path,
    translations_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    first_chapters: int | None = None,
    skip_if_present: bool = False,
    dry_run: bool = False,
    write_manifest: bool = True,
    progress: ProgressHook | None = None,
    clock: Clock = time.perf_counter,
) -> AssemblyResult:
    """Concatenate every translated chunk into its chapter.

    ``first_chapters`` narrows the run to the first ``N`` chapters in reading
    order. ``skip_if_present`` reuses a chapter file whose bytes already equal
    the current concatenation, which is what makes ``N=5`` followed by ``N=10``
    write only the five new chapters. ``dry_run`` reads and verifies everything
    but writes no chapter, manifest or trace.
    """
    inputs = _discover_assembly(
        chunks_path, translations_path, output_dir, first_chapters
    )
    context = RunContext(
        clock=clock,
        dry_run=dry_run,
        limited=inputs.limited,
        progress=progress,
    )
    if not dry_run:
        inputs.out_dir.mkdir(parents=True, exist_ok=True)

    def work(name: str, position: int) -> TranslatedChapter:
        return assemble_chapter(
            name,
            position - 1,
            inputs=inputs,
            skip_if_present=skip_if_present,
            dry_run=dry_run,
        )

    def failed(name: str, exc: Exception) -> TranslatedChapter:
        return failed_chapter(name, inputs.names.index(name), inputs, exc)

    outcomes = run_items(
        inputs.names,
        work,
        on_error=failed,
        progress=lambda record, done, total: context.emit(
            record.to_dict(), done, total
        ),
    )
    records = [outcome.value for outcome in outcomes if outcome.value is not None]
    result = build_assembly_result(
        records=records,
        chunk_folder=inputs.chunk_folder,
        translations_folder=inputs.translations_folder,
        out_dir=inputs.out_dir,
        limit=inputs.limit,
        limited=inputs.limited,
        dry_run=dry_run,
        skip_if_present=skip_if_present,
        write_manifest=write_manifest,
        run_seconds=context.elapsed(),
    )
    store = ArtifactStore(
        inputs.out_dir,
        ASSEMBLY_MANIFEST_FILENAME,
        ASSEMBLY_TRACE_FILENAME,
        enabled=write_manifest and not dry_run,
    )
    return context.finish(store.write_result(result, assembly_manifest_payload(result)))


def build_assembly_result(
    *,
    records: list[TranslatedChapter],
    chunk_folder: Path,
    translations_folder: Path,
    out_dir: Path,
    limit: int | None,
    limited: bool,
    dry_run: bool,
    skip_if_present: bool,
    write_manifest: bool,
    run_seconds: float,
) -> AssemblyResult:
    """The run as one value; counts are derived from the records, not counters."""
    errors = [
        f"{record.filename}: {record.error}"
        for record in records
        if record.error
    ]
    assembled_count = sum(
        1 for record in records if record.ok and not record.skipped and not record.reused
    )
    reused_count = sum(1 for record in records if record.reused)
    incomplete_count = sum(1 for record in records if record.skipped)
    totals = {
        "chapters": len(records),
        "assembled": assembled_count,
        "reused": reused_count,
        "incomplete": incomplete_count,
        "chunks": sum(record.chunk_count for record in records),
        "translated_chunks": sum(record.translated_chunks for record in records),
        "translation_chars": sum(record.translation_chars for record in records),
    }
    return AssemblyResult(
        ok=not errors,
        chunks_path=str(chunk_folder),
        translations_path=str(translations_folder),
        output_dir=str(out_dir),
        chapter_count=len(records),
        ok_count=sum(
            1 for record in records if record.ok and not record.skipped
        ),
        assembled_count=assembled_count,
        reused_count=reused_count,
        incomplete_count=incomplete_count,
        skipped_count=incomplete_count,
        dry_run=dry_run,
        limited=limited,
        first_chapters=limit,
        settings={
            "first_chapters": limit,
            "skip_if_present": bool(skip_if_present),
            "write_manifest": bool(write_manifest),
        },
        totals=totals,
        chapters=[record.to_dict() for record in records],
        errors=errors,
        timing={"run_seconds": round(run_seconds, 4)},
    )


__all__ = [
    "AssemblyInputs",
    "AssemblyResult",
    "Clock",
    "ProgressHook",
    "TranslatedChapter",
    "assembly_manifest_payload",
    "build_assembly_result",
    "load_assembly_manifest",
    "resolve_chunks_dir",
    "resolve_output_dir",
    "resolve_translations_dir",
    "run_assembly",
    "write_assembly_manifest",
    "write_assembly_trace",
]
