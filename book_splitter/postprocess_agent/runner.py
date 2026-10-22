"""Stage-9 entry point: postprocess translated chapters, then the TOC."""


from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.constants import (
    DEFAULT_POSTPROCESS_ATTEMPTS,
    DEFAULT_RECURSION_LIMIT,
)
from ..core.settings import LLMSettings
from ..documents.manifest import load_postprocess_manifest
from ..documents.window import normalize_limit
from ..postprocess.inspect import scan_book_structure
from ..postprocess.toc import (
    apply_toc,
    build_toc_entries,
    detect_toc,
    toc_consistency,
    verify_toc,
)
from .artifacts import (
    summarise_chapter,
    write_postprocess_manifest,
    write_postprocess_trace,
)
from .chapters import run_chapters
from .models import ChapterPostprocessResult, PostprocessResult, ProgressHook
from .plan import prepare_postprocess
from .result import finish_postprocess


def run_postprocess(
    chapters_path: str | Path,
    output_dir: str | Path | None = None,
    settings: LLMSettings | None = None,
    max_attempts: int = DEFAULT_POSTPROCESS_ATTEMPTS,
    dry_run: bool = False,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    limit: int | None = None,
    skip_if_present: bool = False,
    write_manifest: bool = True,
    require_web_search: bool = True,
    model: Any | None = None,
    book_model: Any | None = None,
    chapter_model: Any | None = None,
    progress: ProgressHook | None = None,
) -> PostprocessResult:
    """Postprocess the translated chapters into a separate folder.

    ``limit`` is the chapter window (all front/back matter + first N body chapters).
    ``skip_if_present`` reuses a chapter whose source hash and canonical title
    still match the manifest. ``dry_run`` validates every plan and computes the
    TOC without writing a chapter, manifest or trace. ``require_web_search``
    (default on) makes a published-contents lookup mandatory before the book plan;
    the lookup may fail, but it must be attempted and recorded.
    """
    prepared = prepare_postprocess(
        chapters_path,
        output_dir,
        settings,
        max_attempts=max_attempts,
        dry_run=dry_run,
        recursion_limit=recursion_limit,
        limit=limit,
        require_web_search=require_web_search,
        model=model,
        book_model=book_model,
        write_manifest=write_manifest,
    )
    if isinstance(prepared, PostprocessResult):
        return prepared
    plan_state = prepared
    source_dir = plan_state.source_dir
    out_dir = plan_state.out_dir
    selected_names = plan_state.selected_names
    book = plan_state.book
    plan = plan_state.plan
    canonical = plan_state.canonical
    errors = plan_state.errors
    usage_total = plan_state.usage_total
    settings = plan_state.settings

    results, errors, usage_total, openings, pending = run_chapters(
        plan_state,
        settings=settings,
        model=model,
        chapter_model=chapter_model,
        max_attempts=max_attempts,
        recursion_limit=recursion_limit,
        skip_if_present=skip_if_present,
        dry_run=dry_run,
        progress=progress,
    )

    toc_plan = plan.get("toc") or {}
    if dry_run:
        entries = build_toc_entries(book, plan, titles=canonical)
        entries = [
            entry for entry in entries if str(entry.get("filename")) in selected_names
        ]
    else:
        # The TOC describes the output folder as it actually stands, including
        # chapters a wider earlier run already wrote. The titles are the ones now
        # on disk, so an update can never out-vote the written chapters.
        out_book = scan_book_structure(out_dir)
        output_titles = {
            str(summary.get("filename")): str(summary.get("title") or "")
            for summary in out_book.get("chapters") or []
        }
        entries = build_toc_entries(out_book, plan, titles=output_titles)
    if dry_run:
        current = detect_toc(source_dir, book.get("chapters") or [])
        consistency = toc_consistency(current, entries)
        toc_summary = {
            "ok": True,
            "dry_run": True,
            "action": toc_plan.get("action"),
            "entry_count": len(entries),
            "location": current.get("location"),
            "would_change": toc_plan.get("action") != "keep" or not consistency.get("ok"),
            "consistency": consistency,
            "entries": entries,
        }
    elif not all(item.ok for item in results):
        # A failed chapter makes the title set incomplete; writing a TOC from it
        # would quietly omit that chapter. Leave the folder's existing TOC alone
        # and report why, instead of creating an empty one.
        toc_summary = {
            "ok": False,
            "skipped": "one or more chapter plans failed",
            "action": toc_plan.get("action"),
            "entry_count": len(entries),
            "entries": entries,
        }
        errors.append("table of contents not written: a chapter plan failed")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        applied = apply_toc(out_dir, plan, entries)
        verification = verify_toc(out_dir, plan, entries)
        toc_summary = {
            **applied,
            "ok": bool(applied.get("ok")) and bool(verification.get("ok")),
            "verification": verification,
            "entries": entries,
        }
        if not toc_summary["ok"]:
            errors.append("table of contents verification failed")

    totals = {
        "chapters": len(results),
        "ok": sum(1 for result in results if result.ok),
        "reused": sum(1 for result in results if result.reused),
        "title_set": sum(1 for result in results if result.title_action == "set"),
        "title_inserted": sum(1 for result in results if result.title_action == "insert"),
        "demoted_headings": sum(len(result.demoted_lines) for result in results),
        "promoted_headings": sum(len(result.promoted_lines) for result in results),
        "merged_subtitles": sum(len(result.merged_subtitles) for result in results),
        "toc_entries": len(entries),
        "opening_clean": openings.clean_count,
        "opening_issues": openings.issue_count,
        "opening_fixes": openings.fixed_count,
        "opening_remaining": openings.remaining_count,
    }
    return finish_postprocess(
        plan_state,
        results=results,
        openings=openings,
        toc_summary=toc_summary,
        totals=totals,
        errors=errors,
        usage_total=usage_total,
        limited=bool(normalize_limit(limit) is not None),
        first_chapters=normalize_limit(limit),
        skipped=not pending,
        dry_run=dry_run,
        write_manifest=write_manifest,
    )

__all__ = [
    "ChapterPostprocessResult",
    "PostprocessResult",
    "ProgressHook",
    "load_postprocess_manifest",
    "run_postprocess",
    "summarise_chapter",
    "write_postprocess_manifest",
    "write_postprocess_trace",
]
