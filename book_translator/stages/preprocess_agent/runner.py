"""Stage-2 entry point: preprocess every chapter of a split book.

The chapters are processed **sequentially, one at a time**, because that is what
the pipeline asks for: each chapter's decision is based on its own evidence, and
running them in order keeps the log readable and the failure mode obvious.

The loop is written so that switching to parallel execution is a one-line change
(``workers=4``) and nothing else:

* every chapter's work is a pure function of ``(chapter_path, output_dir)``;
* the graph is built once and reused, so nothing is re-imported per chapter;
* results are collected into a list and only *then* turned into the book-level
  manifest, so concurrent writers never share a file;
* the only shared mutable state is the ``errors`` list, which is appended to
  under a lock when workers are used.
"""

from __future__ import annotations

import contextlib
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from book_translator.core.constants import (
    DEFAULT_RECURSION_LIMIT,
    MAX_FEEDBACK_LOOPS,
    PREPROCESS_MANIFEST_FILENAME,
    PREPROCESS_TRACE_FILENAME,
)
from book_translator.core.settings import LLMSettings
from book_translator.core.store import ArtifactStore
from book_translator.documents.chapters import Chapter, discover_chapters, preprocessed_dir
from book_translator.stages.preprocess.corpus import scan_corpus_damage

from .artifacts import (
    _audit_openings,
    _previous_files,
    _reusable_chapter,
    chapter_records,
    preprocess_manifest_payload,
    summarise_chapter,
    write_preprocess_manifest,
    write_preprocess_trace,
)
from .graph import build_preprocess_graph
from .models import ChapterResult, PreprocessResult, ProgressHook
from .result import build_preprocess_result
from .state import PreprocessState


def run_preprocess(
    chapters_path: str,
    output_dir: str | None = None,
    settings: LLMSettings | None = None,
    max_attempts: int = MAX_FEEDBACK_LOOPS,
    dry_run: bool = False,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    workers: int = 1,
    limit: int | None = None,
    skip_if_present: bool = False,
    write_manifest: bool = True,
    model: Any | None = None,
    progress: ProgressHook | None = None,
) -> PreprocessResult:
    """Preprocess every chapter in a stage-1 output folder.

    ``workers`` > 1 switches to a thread pool. Everything else stays identical:
    the same graph, the same per-chapter contract, the same book-level manifest.

    ``progress`` is an optional reporting hook: it is called once per finished
    chapter with ``(chapter_record, completed, total)``, which is how a CLI
    narrates a run that can take minutes. It never influences the result.
    """
    chapters = discover_chapters(chapters_path)
    if limit is not None and int(limit) > 0:
        chapters = chapters[: int(limit)]
    source_dir = Path(chapters_path).expanduser()
    out_dir = Path(output_dir).expanduser() if output_dir else preprocessed_dir(source_dir)

    corpus_scan = scan_corpus_damage(chapters)
    previous = _previous_files(out_dir) if skip_if_present else {}
    reused: dict[str, ChapterResult] = {}
    for chapter in chapters:
        reusable = _reusable_chapter(chapter, out_dir, previous)
        if reusable is not None:
            reused[chapter.filename] = reusable
    pending = [chapter for chapter in chapters if chapter.filename not in reused]
    if dry_run:
        # The plan is the model's answer: validate the chapter set and the
        # reusable prefix, then stop before building the graph or a provider.
        return PreprocessResult(
            ok=True,
            chapters_path=str(source_dir),
            output_dir=str(out_dir),
            chapter_count=len(chapters),
            ok_count=len(reused),
            skipped_count=len(reused),
            skipped=not pending,
            dry_run=True,
            parallel=workers > 1,
            corpus_scan=corpus_scan,
            openings={},
            chapters=[],
            usage={},
            errors=[],
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    app = build_preprocess_graph(settings, model=model) if pending else None

    results: list[ChapterResult] = []
    errors: list[str] = []
    lock = threading.Lock()
    usage_total: dict[str, int] = {}
    finished = 0

    def announce(result: ChapterResult) -> None:
        """Hand one finished chapter to the hook, in completion order."""
        nonlocal finished
        if progress is None:
            return
        with lock:
            finished += 1
            completed = finished
        with contextlib.suppress(Exception):  # a reporting hook must not break a run
            progress(result.to_dict(), completed, len(chapters))

    for chapter in chapters:
        reusable = reused.get(chapter.filename)
        if reusable is not None:
            announce(reusable)

    def one(chapter: Chapter) -> ChapterResult:
        initial: PreprocessState = {
            "chapter_path": chapter.path,
            "output_dir": str(out_dir),
            "max_attempts": max_attempts,
            "dry_run": dry_run,
            "attempts": 0,
            "usage": {},
            "log": [],
            "feedback": [],
            "messages": [],
        }
        try:
            assert app is not None
            final = app.invoke(initial, config={"recursion_limit": recursion_limit})
        except Exception as exc:  # a broken chapter must not stop the book
            with lock:
                errors.append(f"{chapter.filename}: {type(exc).__name__}: {exc}")
            failed = ChapterResult(
                chapter_id=chapter.chapter_id,
                filename=chapter.filename,
                source_path=chapter.path,
                source_lines=chapter.line_count,
                error=f"{type(exc).__name__}: {exc}",
            )
            announce(failed)
            return failed

        result = summarise_chapter(chapter, final, dry_run=dry_run)
        with lock:
            for key, value in (final.get("usage") or {}).items():
                usage_total[key] = usage_total.get(key, 0) + int(value or 0)
            if not result.ok:
                errors.append(f"{chapter.filename}: {result.error or 'not verified'}")
        announce(result)
        return result

    processed: list[ChapterResult] = []
    if pending:
        if workers and workers > 1 and len(pending) > 1:
            with ThreadPoolExecutor(max_workers=int(workers)) as pool:
                processed = list(pool.map(one, pending))
        else:
            processed = [one(chapter) for chapter in pending]
    by_name = {result.filename: result for result in processed}
    results = [reused.get(chapter.filename) or by_name[chapter.filename] for chapter in chapters]

    # Final phase: the deterministic opening audit. It looks only at the first
    # lines of every chapter and removes the lines that are provably junk (a
    # stray copy of the chapter number, a duplicated title, a leftover page
    # number). A removed line leaves a blank slot, exactly like the stage's own
    # cleaner, so every line number in the log, the manifest and the trace keeps
    # pointing at the same text.
    openings = _audit_openings(out_dir, chapters, results, dry_run=dry_run)
    errors.extend(openings.errors)

    result = build_preprocess_result(
        results=results,
        chapters_path=str(source_dir),
        output_dir=str(out_dir),
        skipped_count=len(reused),
        skipped=not pending,
        dry_run=dry_run,
        parallel=bool(workers and workers > 1),
        corpus_scan=corpus_scan,
        openings=openings,
        usage=usage_total,
        errors=errors,
    )
    store = ArtifactStore(
        out_dir,
        PREPROCESS_MANIFEST_FILENAME,
        PREPROCESS_TRACE_FILENAME,
        enabled=write_manifest and not dry_run,
    )
    return store.write_result(
        result, preprocess_manifest_payload(result, chapters, results)
    )

__all__ = [
    "ChapterResult",
    "PreprocessResult",
    "ProgressHook",
    "chapter_records",
    "run_preprocess",
    "summarise_chapter",
    "write_preprocess_manifest",
    "write_preprocess_trace",
]
