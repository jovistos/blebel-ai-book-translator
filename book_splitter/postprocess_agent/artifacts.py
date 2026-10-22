"""Reuse, opening audit, chapter records and artifacts for postprocess."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.artifacts import timestamp, write_manifest, write_trace
from ..core.constants import (
    POSTPROCESS_MANIFEST_FILENAME,
    POSTPROCESS_TRACE_FILENAME,
)
from ..documents.chapters import file_sha256
from ..documents.io import load_lines
from ..documents.manifest import load_postprocess_manifest
from ..documents.text import norm
from ..documents.window import normalize_limit
from ..openings import OpeningAuditResult, audit_openings
from ..postprocess.inspect import matter_kind, scan_chapter_structure
from .models import ChapterPostprocessResult, PostprocessResult


def _selected_files(
    files: list[Path],
    first_chapters: Any,
    matters: dict[str, str] | None = None,
) -> list[Path]:
    """The files in the run window: all front/back matter + first N body chapters."""
    limit = normalize_limit(first_chapters)
    if limit is None:
        return files
    known = matters or {}

    def kind(path: Path) -> str:
        return known.get(path.name) or matter_kind(path.name)

    front = [path for path in files if kind(path) == "front"]
    back = [path for path in files if kind(path) == "back"]
    body = [path for path in files if kind(path) == "body"]
    selected = set(front + back + body[:limit])
    return [path for path in files if path in selected]


def _previous_chapters(output_dir: Path) -> dict[str, dict[str, Any]]:
    manifest = load_postprocess_manifest(output_dir)
    return {
        str(entry.get("filename")): entry
        for entry in manifest.get("chapters") or []
        if isinstance(entry, dict) and entry.get("filename")
    }


def _reusable(
    path: Path,
    output_dir: Path,
    previous: dict[str, dict[str, Any]],
    canonical_title: str,
) -> ChapterPostprocessResult | None:
    entry = previous.get(path.name)
    if not entry or not entry.get("verification_ok"):
        return None
    target = Path(str(entry.get("output_path") or ""))
    if not target.is_file():
        target = output_dir / path.name
    if not target.is_file():
        return None
    if str(entry.get("source_sha256") or "") != file_sha256(path):
        return None
    if str(entry.get("title") or "") != canonical_title:
        return None
    # A file written before the title-detection fix can carry its title in the
    # wrong place: a translated ``## 2`` used to be skipped as a printed page
    # number, so the title was set on a later heading and a bare number was left
    # at the top. Reuse only when the title this stage reads today is still the
    # canonical title it recorded (the opening audit can then fix any leftover
    # line around it).
    if canonical_title:
        current = scan_chapter_structure(target)
        if norm(current.title) != norm(canonical_title):
            return None
    return ChapterPostprocessResult(
        chapter_id=path.stem,
        filename=path.name,
        source_path=str(path),
        output_path=str(target),
        ok=True,
        reused=True,
        title=canonical_title,
        title_action="reused",
        source_lines=int(entry.get("source_lines") or 0),
        output_lines=int(entry.get("output_lines") or 0),
        heading_decisions=int(entry.get("heading_decisions") or 0),
        demoted_lines=list(entry.get("demoted_lines") or []),
        promoted_lines=list(entry.get("promoted_lines") or []),
        merged_subtitles=list(entry.get("merged_subtitles") or []),
        verification={"ok": True, "reused": True},
        log=["reused from the previous run (--skip-if-present)"],
    )



def _audit_openings(
    out_dir: Path,
    selected: list[Path],
    results: list[ChapterPostprocessResult],
    *,
    dry_run: bool,
) -> OpeningAuditResult:
    """Run the stage-9 flavour of the opening audit over the written chapters."""
    if dry_run:
        return OpeningAuditResult(
            dry_run=True,
            folder=str(out_dir),
            chapter_count=len(selected),
            skipped="dry run — no chapter files were written",
        )
    openings = audit_openings(
        out_dir,
        [path.name for path in selected],
        keep_blank_slot=False,
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


def summarise_chapter(
    path: Path,
    final: dict[str, Any],
    *,
    canonical_title: str,
    dry_run: bool,
) -> ChapterPostprocessResult:
    """Turn one chapter's final graph state into a :class:`ChapterPostprocessResult`."""
    applied = final.get("applied") or {}
    verification = final.get("verification") or {}
    validation = final.get("validation") or {}
    plan = final.get("chapter_plan") or {}
    title = ((plan.get("title") or {}).get("text")) or canonical_title
    action = (plan.get("title") or {}).get("action") or "keep"
    if dry_run:
        ok = bool(validation.get("ok")) and bool(verification.get("ok"))
    else:
        ok = bool(applied.get("ok")) and bool(verification.get("ok"))
    error = None
    if not ok:
        error = (
            (final.get("result") or {}).get("error")
            or "; ".join(verification.get("problems") or [])
            or "; ".join(validation.get("issues") or [])
            or "not verified"
        )
    return ChapterPostprocessResult(
        chapter_id=path.stem,
        filename=path.name,
        source_path=str(path),
        output_path=applied.get("output_path"),
        ok=ok,
        reused=False,
        title=title,
        title_action=str(action),
        source_lines=int((applied.get("applied") or {}).get("source_lines") or 0),
        output_lines=int(applied.get("output_lines") or 0),
        heading_decisions=len(plan.get("headings") or []),
        demoted_lines=list((applied.get("applied") or {}).get("demoted_lines") or []),
        promoted_lines=list((applied.get("applied") or {}).get("promoted_lines") or []),
        merged_subtitles=list(
            (applied.get("applied") or {}).get("merged_subtitles") or []
        ),
        verification=verification,
        attempts=int(final.get("attempts", 0)),
        error=error,
        log=list(final.get("log") or []),
    )


def postprocess_manifest_payload(result: PostprocessResult) -> dict[str, Any]:
    return {
        "stage": "postprocess",
        "generated_at": timestamp(),
        "ok": result.ok,
        "dry_run": result.dry_run,
        "limited": result.limited,
        "first_chapters": result.first_chapters,
        "chapters_path": result.chapters_path,
        "output_dir": result.output_dir,
        "book_plan": result.book_plan,
        "validation": result.validation,
        "toc": result.toc,
        "web": result.web,
        "totals": result.totals,
        "openings": result.openings,
        "usage": result.usage,
        "chapters": [
            {
                **chapter,
                "source_sha256": (
                    file_sha256(chapter["source_path"])
                    if Path(str(chapter.get("source_path") or "")).is_file()
                    else ""
                ),
                "verification_ok": bool((chapter.get("verification") or {}).get("ok")),
            }
            for chapter in result.chapters
        ],
        "errors": result.errors,
    }


def write_postprocess_manifest(out_dir: str | Path, result: PostprocessResult) -> Path:
    return write_manifest(
        out_dir, POSTPROCESS_MANIFEST_FILENAME, postprocess_manifest_payload(result)
    )


def write_postprocess_trace(out_dir: str | Path, result: PostprocessResult) -> Path:
    """Write ``_postprocess_trace.json``: the same run, for debugging."""
    return write_trace(out_dir, POSTPROCESS_TRACE_FILENAME, result.to_dict())



__all__ = [
    "_audit_openings",
    "_previous_chapters",
    "_reusable",
    "_selected_files",
    "postprocess_manifest_payload",
    "summarise_chapter",
    "write_postprocess_manifest",
    "write_postprocess_trace",
]
