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

import contextlib
import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..chunking.runner import chunk_files
from ..core.constants import (
    ASSEMBLY_MANIFEST_FILENAME,
    ASSEMBLY_TRACE_FILENAME,
    CHUNKS_DIRNAME,
)
from ..core.errors import ToolError
from ..documents.chapters import translated_chapters_dir, translations_dir
from ..documents.window import (
    body_chapter_names,
    chunk_chapter_names,
    is_body_chapter_name,
    normalize_limit,
)

#: Monotonic clock used for the timings; injectable so tests can be exact.
Clock = Callable[[], float]
#: Called once per visited chapter as ``(chapter_record, completed, total)``.
ProgressHook = Callable[[dict[str, Any], int, int], None]


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #
@dataclass
class TranslatedChapter:
    """One translated chapter file: what went in, what came out and its state."""

    chapter_id: str
    filename: str
    chapter_index: int
    chunk_count: int = 0
    translated_chunks: int = 0
    source_chars: int = 0
    translation_chars: int = 0
    sha256: str = ""
    output_path: str | None = None
    reused: bool = False
    skipped: bool = False
    ok: bool = True
    error: str | None = None
    note: str | None = None
    missing_chunks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AssemblyResult:
    """Outcome of a whole stage-8 run."""

    ok: bool
    chunks_path: str
    translations_path: str
    output_dir: str
    chapter_count: int = 0
    ok_count: int = 0
    assembled_count: int = 0
    reused_count: int = 0
    incomplete_count: int = 0
    skipped_count: int = 0
    dry_run: bool = False
    limited: bool = False
    first_chapters: int | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    totals: dict[str, int] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    chapters: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    manifest_path: str | None = None
    trace_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def resolve_chunks_dir(path: str | Path) -> Path:
    """Resolve a chunks folder, the book folder that holds it, or its manifest."""
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate.parent
    if candidate.name == CHUNKS_DIRNAME:
        return candidate
    nested = candidate / CHUNKS_DIRNAME
    if nested.is_dir():
        return nested
    return candidate


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


def load_assembly_manifest(output_dir: str | Path) -> dict[str, Any]:
    """Read ``_translated_chapters_manifest.json`` from a folder or a file."""
    path = Path(output_dir).expanduser()
    manifest_path = (path / ASSEMBLY_MANIFEST_FILENAME) if path.is_dir() else path
    if not manifest_path.is_file():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def _chunk_path(translations_path: Path, entry: dict[str, Any]) -> Path:
    return translations_path / str(entry.get("filename") or "")


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
    run_started = clock()
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
    grouped = _group_by_chapter(entries, names)

    records: list[TranslatedChapter] = []
    errors: list[str] = []
    assembled_count = 0
    reused_count = 0
    incomplete_count = 0
    skipped_count = 0
    written_ok = 0

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    for index, name in enumerate(names):
        chapter_entries = grouped.get(name) or []
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
            if not _chunk_path(translations_folder, entry).is_file()
        ]
        if missing:
            # A chapter is all-or-nothing: a partial prefix would look like a
            # complete chapter to the reader, so it is reported instead.
            record.skipped = True
            record.translated_chunks = len(chapter_entries) - len(missing)
            record.missing_chunks = missing
            record.note = (
                f"{len(missing)} of {len(chapter_entries)} chunk(s) not translated"
            )
            incomplete_count += 1
            skipped_count += 1
        else:
            try:
                parts = [
                    _chunk_path(translations_folder, entry).read_text(encoding="utf-8")
                    for entry in chapter_entries
                ]
                text = "".join(parts)
            except OSError as exc:  # a read error is one chapter's failure
                record.ok = False
                record.error = f"{type(exc).__name__}: {exc}"
                errors.append(f"{name}: {record.error}")
            else:
                record.translation_chars = len(text)
                record.sha256 = _sha256_text(text)
                target = out_dir / name
                record.output_path = str(target)
                if dry_run:
                    record.reused = bool(
                        skip_if_present
                        and target.is_file()
                        and target.read_text(encoding="utf-8") == text
                    )
                    if record.reused:
                        reused_count += 1
                    else:
                        assembled_count += 1
                else:
                    try:
                        if skip_if_present and target.is_file() and target.read_text(
                            encoding="utf-8"
                        ) == text:
                            record.reused = True
                            reused_count += 1
                        else:
                            target.write_text(text, encoding="utf-8")
                            record.reused = False
                            assembled_count += 1
                    except OSError as exc:
                        record.ok = False
                        record.error = f"{type(exc).__name__}: {exc}"
                        errors.append(f"{name}: {record.error}")

        if record.ok and not record.skipped:
            written_ok += 1
        records.append(record)
        if progress is not None:
            with contextlib.suppress(Exception):  # reporting must not break a run
                progress(record.to_dict(), index + 1, len(names))

    totals = {
        "chapters": len(records),
        "assembled": assembled_count,
        "reused": reused_count,
        "incomplete": incomplete_count,
        "chunks": sum(record.chunk_count for record in records),
        "translated_chunks": sum(record.translated_chunks for record in records),
        "translation_chars": sum(record.translation_chars for record in records),
    }
    result = AssemblyResult(
        ok=not errors,
        chunks_path=str(chunk_folder),
        translations_path=str(translations_folder),
        output_dir=str(out_dir),
        chapter_count=len(records),
        ok_count=written_ok,
        assembled_count=assembled_count,
        reused_count=reused_count,
        incomplete_count=incomplete_count,
        skipped_count=skipped_count,
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
        timing={"run_seconds": round(clock() - run_started, 4)},
    )
    if write_manifest and not dry_run:
        result.manifest_path = str(write_assembly_manifest(out_dir, result))
        result.trace_path = str(write_assembly_trace(out_dir, result))
    return result


# --------------------------------------------------------------------------- #
# writing the artifacts
# --------------------------------------------------------------------------- #
def write_assembly_manifest(
    out_dir: str | Path,
    result: AssemblyResult,
) -> Path:
    """Write ``_translated_chapters_manifest.json``: the book-level index."""
    path = Path(out_dir) / ASSEMBLY_MANIFEST_FILENAME
    payload = {
        "stage": "assembly",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "ok": result.ok,
        "dry_run": result.dry_run,
        "limited": result.limited,
        "first_chapters": result.first_chapters,
        "chunks_path": result.chunks_path,
        "translations_path": result.translations_path,
        "output_dir": result.output_dir,
        "chapter_count": result.chapter_count,
        "ok_count": result.ok_count,
        "assembled_count": result.assembled_count,
        "reused_count": result.reused_count,
        "incomplete_count": result.incomplete_count,
        "skipped_count": result.skipped_count,
        "settings": result.settings,
        "totals": result.totals,
        "timing": result.timing,
        "files": result.chapters,
        "errors": result.errors,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_assembly_trace(
    out_dir: str | Path,
    result: AssemblyResult,
) -> Path:
    """Write ``_translated_chapters_trace.json``: the same run, for debugging."""
    path = Path(out_dir) / ASSEMBLY_TRACE_FILENAME
    path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


__all__ = [
    "AssemblyResult",
    "Clock",
    "ProgressHook",
    "TranslatedChapter",
    "load_assembly_manifest",
    "resolve_chunks_dir",
    "resolve_output_dir",
    "resolve_translations_dir",
    "run_assembly",
    "write_assembly_manifest",
    "write_assembly_trace",
]
