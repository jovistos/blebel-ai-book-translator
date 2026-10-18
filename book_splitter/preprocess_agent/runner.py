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
import json
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.constants import (
    DEFAULT_RECURSION_LIMIT,
    MAX_FEEDBACK_LOOPS,
    PREPROCESS_MANIFEST_FILENAME,
    PREPROCESS_TRACE_FILENAME,
)
from ..core.settings import LLMSettings
from ..documents.chapters import Chapter, discover_chapters, preprocessed_dir
from ..preprocess.corpus import scan_corpus_damage
from .graph import build_preprocess_graph
from .state import PreprocessState


@dataclass
class ChapterResult:
    """Outcome of preprocessing one chapter."""

    chapter_id: str
    filename: str
    source_path: str
    output_path: str | None = None
    ok: bool = False
    source_lines: int = 0
    output_lines: int = 0
    operations: list[dict[str, Any]] = field(default_factory=list)
    text_fixes: list[dict[str, Any]] = field(default_factory=list)
    removed_artefacts: list[int] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)
    attempts: int = 0
    error: str | None = None
    log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: Called once per finished chapter as ``(chapter, completed, total)``, where
#: ``chapter`` is the same record (:meth:`ChapterResult.to_dict`) the manifest
#: stores. It runs in completion order, so with ``workers > 1`` it is not the
#: chapter order.
ProgressHook = Callable[[dict[str, Any], int, int], None]


@dataclass
class PreprocessResult:
    """Outcome of a whole stage-2 run."""

    ok: bool
    chapters_path: str
    output_dir: str
    chapter_count: int = 0
    ok_count: int = 0
    dry_run: bool = False
    parallel: bool = False
    corpus_scan: dict[str, Any] = field(default_factory=dict)
    chapters: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    manifest_path: str | None = None
    trace_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_preprocess(
    chapters_path: str,
    output_dir: str | None = None,
    settings: LLMSettings | None = None,
    max_attempts: int = MAX_FEEDBACK_LOOPS,
    dry_run: bool = False,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    workers: int = 1,
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
    source_dir = Path(chapters_path).expanduser()
    out_dir = Path(output_dir).expanduser() if output_dir else preprocessed_dir(source_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_scan = scan_corpus_damage(chapters)
    app = build_preprocess_graph(settings, model=model)

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

    if workers and workers > 1 and len(chapters) > 1:
        with ThreadPoolExecutor(max_workers=int(workers)) as pool:
            results = list(pool.map(one, chapters))
    else:
        results = [one(chapter) for chapter in chapters]

    result = PreprocessResult(
        ok=bool(results) and all(r.ok for r in results),
        chapters_path=str(source_dir),
        output_dir=str(out_dir),
        chapter_count=len(results),
        ok_count=sum(1 for r in results if r.ok),
        dry_run=dry_run,
        parallel=bool(workers and workers > 1),
        corpus_scan=corpus_scan,
        chapters=[r.to_dict() for r in results],
        usage=usage_total,
        errors=errors,
    )
    if write_manifest:
        result.manifest_path = str(
            write_preprocess_manifest(out_dir, result, chapters, results)
        )
        result.trace_path = str(write_preprocess_trace(out_dir, result))
    return result


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
    """Chapter records in the same shape stage 1 uses, so stage 3 can reuse readers."""
    by_id = {c.chapter_id: c for c in chapters}
    records: list[dict[str, Any]] = []
    for result in results:
        source = by_id.get(result.chapter_id)
        records.append(
            {
                "filename": result.filename,
                "chapter_id": result.chapter_id,
                "path": result.output_path or result.source_path,
                "source_path": result.source_path,
                "source_lines": result.source_lines,
                "lines": result.output_lines,
                "start_line": source.start_line if source else None,
                "end_line": source.end_line if source else None,
                "operations": result.operations,
                "text_fixes": result.text_fixes,
                "removed_artefacts": result.removed_artefacts,
                "verification_ok": bool(result.verification.get("ok")),
            }
        )
    return records


def write_preprocess_manifest(
    out_dir: Path,
    result: PreprocessResult,
    chapters: list[Chapter],
    results: list[ChapterResult] | None = None,
) -> Path:
    """Write the book-level stage-2 manifest next to the cleaned chapters.

    The record shape matches stage 1's ``_manifest.json`` (``filename``,
    ``lines``, ``start_line``, ``end_line``), so the next stage can read both
    folders with the same reader.
    """
    path = Path(out_dir) / PREPROCESS_MANIFEST_FILENAME
    if results is None:
        results = [ChapterResult(**c) for c in result.chapters]
    from ..documents.chapters import load_split_manifest

    stage_one = load_split_manifest(result.chapters_path)
    payload = {
        "stage": "preprocess",
        "chapters_path": result.chapters_path,
        "source_file": result.corpus_scan.get("source_file") or stage_one.get("source_file", ""),
        "ok": result.ok,
        "dry_run": result.dry_run,
        "chapter_count": result.chapter_count,
        "files": chapter_records(results, chapters),
        "corpus_scan": {
            "counts": result.corpus_scan.get("counts", {}),
            "total_findings": result.corpus_scan.get("total_findings", 0),
        },
        "usage": result.usage,
        "errors": result.errors,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_preprocess_trace(out_dir: Path, result: PreprocessResult) -> Path:
    """Write the audit trail: per chapter, what was decided and what was proven."""
    path = Path(out_dir) / PREPROCESS_TRACE_FILENAME
    path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


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
