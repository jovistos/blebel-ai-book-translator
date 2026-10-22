"""The postprocess result value and its one builder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import PostprocessResult


def build_postprocess_result(
    *,
    ok: bool,
    source_dir: Path,
    out_dir: Path,
    chapter_count: int,
    dry_run: bool,
    limited: bool,
    first_chapters: int | None,
    book_plan: dict[str, Any] | None = None,
    validation: dict[str, Any] | None = None,
    toc: dict[str, Any] | None = None,
    web: dict[str, Any] | None = None,
    totals: dict[str, int] | None = None,
    openings: dict[str, Any] | None = None,
    chapters: list[dict[str, Any]] | None = None,
    usage: dict[str, int] | None = None,
    errors: list[str] | None = None,
    skipped: bool = False,
) -> PostprocessResult:
    records = chapters or []
    return PostprocessResult(
        ok=ok,
        chapters_path=str(source_dir),
        output_dir=str(out_dir),
        chapter_count=chapter_count,
        ok_count=sum(1 for record in records if record.get("ok")),
        reused_count=sum(1 for record in records if record.get("reused")),
        skipped=skipped,
        dry_run=dry_run,
        limited=limited,
        first_chapters=first_chapters,
        book_plan=book_plan or {},
        validation=validation or {},
        toc=toc or {},
        web=web or {},
        totals=totals or {},
        openings=openings or {},
        chapters=records,
        usage=usage or {},
        errors=errors or [],
    )


__all__ = ["build_postprocess_result"]


def finish_postprocess(
    plan_state,
    *,
    results,
    openings,
    toc_summary,
    totals,
    errors,
    usage_total,
    limited: bool,
    first_chapters,
    skipped: bool,
    dry_run: bool,
    write_manifest: bool,
):
    from ..core.constants import POSTPROCESS_MANIFEST_FILENAME, POSTPROCESS_TRACE_FILENAME
    from ..core.store import ArtifactStore
    from .artifacts import postprocess_manifest_payload

    result = build_postprocess_result(
        ok=bool(results)
        and all(item.ok for item in results)
        and bool(toc_summary.get("ok"))
        and openings.ok,
        source_dir=plan_state.source_dir,
        out_dir=plan_state.out_dir,
        chapter_count=len(results),
        dry_run=dry_run,
        limited=limited,
        first_chapters=first_chapters,
        book_plan=plan_state.plan,
        validation=plan_state.validation,
        toc=toc_summary,
        web=plan_state.web,
        totals=totals,
        openings=openings.summary(),
        chapters=[item.to_dict() for item in results],
        usage=usage_total,
        errors=errors,
        skipped=skipped,
    )
    store = ArtifactStore(
        plan_state.out_dir,
        POSTPROCESS_MANIFEST_FILENAME,
        POSTPROCESS_TRACE_FILENAME,
        enabled=write_manifest and not dry_run,
    )
    return store.write_result(result, postprocess_manifest_payload(result))
