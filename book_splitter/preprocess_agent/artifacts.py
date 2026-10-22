"""Reuse, opening audit, chapter records and artifacts for preprocess."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.artifacts import read_manifest, write_manifest, write_trace
from ..core.constants import (
    PREPROCESS_MANIFEST_FILENAME,
    PREPROCESS_TRACE_FILENAME,
)
from ..documents.chapters import Chapter, file_sha256
from ..documents.io import load_lines
from ..documents.manifest import load_split_manifest
from ..openings import OpeningAuditResult, audit_openings
from .models import ChapterResult, PreprocessResult


def _previous_manifest(out_dir: Path) -> dict[str, Any]:
    return read_manifest(out_dir, PREPROCESS_MANIFEST_FILENAME)


def _previous_files(out_dir: Path) -> dict[str, dict[str, Any]]:
    manifest = _previous_manifest(out_dir)
    return {
        str(entry.get("filename")): entry
        for entry in manifest.get("files") or []
        if isinstance(entry, dict) and entry.get("filename")
    }


def _reusable_chapter(
    chapter: Chapter,
    out_dir: Path,
    previous: dict[str, dict[str, Any]],
) -> ChapterResult | None:
    """A chapter this run can reuse instead of asking the model again.

    Reuse is only safe when the source chapter and the cleaned file are both
    still the ones the previous run produced: a re-split rewrites chapter files
    under the same names, and a manual edit to ``02-preprocessed/`` must not be
    mistaken for a finished cleaning. Newer manifests record both hashes; older
    ones only carry the line counts, which is still enough to notice a re-split
    that changed the chapter's size.
    """
    entry = previous.get(chapter.filename)
    if not entry or not entry.get("verification_ok"):
        return None
    recorded_lines = entry.get("source_lines")
    if recorded_lines is not None and int(recorded_lines) != chapter.line_count:
        return None
    recorded_source = str(entry.get("source_sha256") or "")
    if recorded_source and chapter.sha256 and recorded_source != chapter.sha256:
        return None
    candidates = [
        Path(str(entry.get("path") or "")),
        Path(str(entry.get("output_path") or "")),
        out_dir / chapter.filename,
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return None
    recorded_output = str(entry.get("sha256") or "")
    if recorded_output and file_sha256(path) != recorded_output:
        return None
    return ChapterResult(
        chapter_id=chapter.chapter_id,
        filename=chapter.filename,
        source_path=chapter.path,
        output_path=str(path),
        ok=True,
        source_lines=chapter.line_count,
        output_lines=int(entry.get("lines") or 0),
        verification={"ok": True, "reused": True},
        log=["reused from the previous run (--skip-if-present)"],
    )



def _audit_openings(
    out_dir: Path,
    chapters: list[Chapter],
    results: list[ChapterResult],
    *,
    dry_run: bool,
) -> OpeningAuditResult:
    """Run the stage-2 flavour of the opening audit over the written chapters."""
    if dry_run:
        return OpeningAuditResult(
            dry_run=True,
            folder=str(out_dir),
            chapter_count=len(chapters),
            skipped="dry run — no chapter files were written",
        )
    openings = audit_openings(
        out_dir,
        [chapter.filename for chapter in chapters],
        keep_blank_slot=True,
    )
    by_name = {result.filename: result for result in results}
    for record in openings.chapters:
        chapter_result = by_name.get(str(record.get("filename")))
        applied = list(record.get("fixes") or [])
        if chapter_result is None or not applied:
            continue
        path = Path(chapter_result.output_path or out_dir / chapter_result.filename)
        if path.is_file():
            chapter_result.output_lines = len(load_lines(path))
        detail = ", ".join(
            f"line {fix.get('line_number')} ({fix.get('text')!r})" for fix in applied
        )
        chapter_result.log.append(f"openings: removed {detail}")
    return openings


def summarise_chapter(chapter: Chapter, final: dict, dry_run: bool) -> ChapterResult:
    """Turn one chapter's final graph state into a :class:`ChapterResult`."""
    applied = final.get("applied") or {}
    verification = final.get("verification") or {}
    validation = final.get("validation") or {}
    output_path = applied.get("output_path")
    if dry_run:
        ok = bool(validation.get("ok"))
    else:
        ok = bool(applied.get("ok")) and bool(verification.get("ok"))
    error = None
    if not ok:
        error = (final.get("result") or {}).get("error") or (
            "; ".join(verification.get("problems") or [])
            or "; ".join(validation.get("issues") or [])
            or "not verified"
        )
    return ChapterResult(
        chapter_id=chapter.chapter_id,
        filename=chapter.filename,
        source_path=chapter.path,
        output_path=output_path,
        ok=ok,
        source_lines=int(applied.get("source_lines") or chapter.line_count),
        output_lines=int(applied.get("output_lines") or 0),
        operations=list(final.get("operations") or []),
        text_fixes=list(validation.get("text_fixes") or []),
        removed_artefacts=list(applied.get("removed_artefacts") or []),
        verification=verification,
        attempts=int(final.get("attempts", 0)),
        error=error,
        log=list(final.get("log") or []),
    )


def chapter_records(results: list[ChapterResult], chapters: list[Chapter]) -> list[dict[str, Any]]:
    """Chapter records in the same shape stage 1 uses, so stage 3 can reuse readers.

    The record carries the stage-1 title and source range, the cleaned file's
    line/byte counts, and both hashes: ``source_sha256`` (the chapter it was
    cleaned from, so a re-split invalidates reuse) and ``sha256`` (the cleaned
    file, so a hand edit does too). ``load_split_manifest`` reads this file when a
    folder has no stage-1 ``_manifest.json``, which is how ``02-preprocessed/``
    keeps its chapter titles and order downstream.
    """
    by_id = {c.chapter_id: c for c in chapters}
    records: list[dict[str, Any]] = []
    for result in results:
        source = by_id.get(result.chapter_id)
        output = Path(result.output_path).expanduser() if result.output_path else None
        written = output is not None and output.is_file()
        records.append(
            {
                "filename": result.filename,
                "chapter_id": result.chapter_id,
                "path": result.output_path or result.source_path,
                "source_path": result.source_path,
                "title": source.title if source else "",
                "source_sha256": source.sha256 if source else "",
                "source_lines": result.source_lines,
                "lines": result.output_lines,
                "bytes": output.stat().st_size if output is not None and written else 0,
                "sha256": file_sha256(output) if output is not None and written else "",
                "start_line": source.start_line if source else None,
                "end_line": source.end_line if source else None,
                "operations": result.operations,
                "text_fixes": result.text_fixes,
                "removed_artefacts": result.removed_artefacts,
                "verification_ok": bool(result.verification.get("ok")),
            }
        )
    return records


def preprocess_manifest_payload(
    result: PreprocessResult,
    chapters: list[Chapter],
    results: list[ChapterResult] | None = None,
) -> dict[str, Any]:
    if results is None:
        results = [ChapterResult(**c) for c in result.chapters]
    stage_one = load_split_manifest(result.chapters_path)
    source_file = (
        result.corpus_scan.get("source_file")
        or stage_one.get("source_file")
        or stage_one.get("source_path")
        or ""
    )
    return {
        "stage": "preprocess",
        "chapters_path": result.chapters_path,
        "source_file": source_file,
        "source_path": source_file,
        "ok": result.ok,
        "dry_run": result.dry_run,
        "chapter_count": result.chapter_count,
        "files": chapter_records(results, chapters),
        "corpus_scan": {
            "counts": result.corpus_scan.get("counts", {}),
            "total_findings": result.corpus_scan.get("total_findings", 0),
        },
        "openings": result.openings,
        "usage": result.usage,
        "errors": result.errors,
    }


def write_preprocess_manifest(
    out_dir: Path,
    result: PreprocessResult,
    chapters: list[Chapter],
    results: list[ChapterResult] | None = None,
) -> Path:
    return write_manifest(
        out_dir,
        PREPROCESS_MANIFEST_FILENAME,
        preprocess_manifest_payload(result, chapters, results),
    )


def write_preprocess_trace(out_dir: Path, result: PreprocessResult) -> Path:
    """Write the audit trail: per chapter, what was decided and what was proven."""
    return write_trace(out_dir, PREPROCESS_TRACE_FILENAME, result.to_dict())



__all__ = [
    "_previous_files",
    "_reusable_chapter",
    "_audit_openings",
    "chapter_records",
    "preprocess_manifest_payload",
    "summarise_chapter",
    "write_preprocess_manifest",
    "write_preprocess_trace",
]
