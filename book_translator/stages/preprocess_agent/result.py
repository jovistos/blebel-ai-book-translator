"""The preprocess result value and its one builder."""

from __future__ import annotations

from typing import Any

from .models import ChapterResult, PreprocessResult


def build_preprocess_result(
    *,
    results: list[ChapterResult],
    chapters_path: str,
    output_dir: str,
    skipped_count: int,
    skipped: bool,
    dry_run: bool,
    parallel: bool,
    corpus_scan: dict[str, Any],
    openings: Any,
    usage: dict[str, int],
    errors: list[str],
) -> PreprocessResult:
    return PreprocessResult(
        ok=bool(results) and all(result.ok for result in results) and openings.ok,
        chapters_path=chapters_path,
        output_dir=output_dir,
        chapter_count=len(results),
        ok_count=sum(1 for result in results if result.ok),
        skipped_count=skipped_count,
        skipped=skipped,
        dry_run=dry_run,
        parallel=parallel,
        corpus_scan=corpus_scan,
        openings=openings.summary(),
        chapters=[result.to_dict() for result in results],
        usage=usage,
        errors=errors,
    )


__all__ = ["build_preprocess_result"]
