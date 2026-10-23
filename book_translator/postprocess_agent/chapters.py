"""The postprocess per-chapter loop and opening audit."""

from __future__ import annotations

import contextlib
import threading
from pathlib import Path

from .artifacts import _audit_openings, _previous_chapters, _reusable, summarise_chapter
from .graph import build_chapter_postprocess_graph
from .models import ChapterPostprocessResult, ProgressHook
from .state import ChapterPostprocessState


def run_chapters(
    plan_state,
    *,
    settings,
    model,
    chapter_model,
    max_attempts: int,
    recursion_limit: int,
    skip_if_present: bool,
    dry_run: bool,
    progress: ProgressHook | None,
):
    source_dir = plan_state.source_dir
    out_dir = plan_state.out_dir
    selected = plan_state.selected
    canonical = plan_state.canonical
    errors = list(plan_state.errors)
    usage_total = dict(plan_state.usage_total)
    previous = _previous_chapters(out_dir) if skip_if_present and not dry_run else {}
    reused: dict[str, ChapterPostprocessResult] = {}
    for path in selected:
        reusable = _reusable(path, out_dir, previous, canonical.get(path.name, ""))
        if reusable is not None:
            reused[path.name] = reusable
    pending = [path for path in selected if path.name not in reused]
    chapter_app = (
        build_chapter_postprocess_graph(
            settings,
            model=chapter_model if chapter_model is not None else model,
        )
        if pending
        else None
    )

    lock = threading.Lock()
    finished = 0

    def announce(result: ChapterPostprocessResult) -> None:
        nonlocal finished
        if progress is None:
            return
        with lock:
            finished += 1
            completed = finished
        with contextlib.suppress(Exception):
            progress(result.to_dict(), completed, len(selected))

    for path in selected:
        if path.name in reused:
            announce(reused[path.name])

    def one(path: Path) -> ChapterPostprocessResult:
        title = canonical.get(path.name, "")
        initial: ChapterPostprocessState = {
            "chapter_path": str(path),
            "chapters_path": str(source_dir),
            "output_dir": str(out_dir),
            "canonical_title": title,
            "max_attempts": max_attempts,
            "dry_run": dry_run,
            "attempts": 0,
            "usage": {},
            "log": [],
            "feedback": [],
            "messages": [],
        }
        try:
            assert chapter_app is not None
            final = chapter_app.invoke(
                initial, config={"recursion_limit": recursion_limit}
            )
        except Exception as exc:
            failed = ChapterPostprocessResult(
                chapter_id=path.stem,
                filename=path.name,
                source_path=str(path),
                title=title,
                source_lines=len(path.read_text(encoding="utf-8").splitlines()),
                error=f"{type(exc).__name__}: {exc}",
            )
            with lock:
                errors.append(f"{path.name}: {failed.error}")
            announce(failed)
            return failed

        result = summarise_chapter(path, final, canonical_title=title, dry_run=dry_run)
        with lock:
            for key, value in (final.get("usage") or {}).items():
                usage_total[key] = usage_total.get(key, 0) + int(value or 0)
            if not result.ok:
                errors.append(f"{path.name}: {result.error or 'not verified'}")
        announce(result)
        return result

    processed = [one(path) for path in pending]
    by_name = {result.filename: result for result in processed}
    results = [reused.get(path.name) or by_name[path.name] for path in selected]
    openings = _audit_openings(out_dir, selected, results, dry_run=dry_run)
    errors.extend(openings.errors)
    return results, errors, usage_total, openings, pending


__all__ = ["run_chapters"]
