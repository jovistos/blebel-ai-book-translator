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
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.artifacts import read_manifest, write_manifest, write_trace
from ..core.constants import (
    DEFAULT_RECURSION_LIMIT,
    MAX_FEEDBACK_LOOPS,
    PREPROCESS_MANIFEST_FILENAME,
    PREPROCESS_TRACE_FILENAME,
)
from ..core.settings import LLMSettings
from ..documents.chapters import (
    Chapter,
    discover_chapters,
    file_sha256,
    preprocessed_dir,
)
from ..documents.io import load_lines
from ..documents.manifest import load_split_manifest
from ..openings import OpeningAuditResult, audit_openings
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
    skipped_count: int = 0
    skipped: bool = False
    dry_run: bool = False
    parallel: bool = False
    corpus_scan: dict[str, Any] = field(default_factory=dict)
    openings: dict[str, Any] = field(default_factory=dict)
    chapters: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    manifest_path: str | None = None
    trace_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_scan = scan_corpus_damage(chapters)
    previous = _previous_files(out_dir) if skip_if_present and not dry_run else {}
    reused: dict[str, ChapterResult] = {}
    for chapter in chapters:
        reusable = _reusable_chapter(chapter, out_dir, previous)
        if reusable is not None:
            reused[chapter.filename] = reusable
    pending = [chapter for chapter in chapters if chapter.filename not in reused]
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

    result = PreprocessResult(
        ok=bool(results) and all(r.ok for r in results) and openings.ok,
        chapters_path=str(source_dir),
        output_dir=str(out_dir),
        chapter_count=len(results),
        ok_count=sum(1 for r in results if r.ok),
        skipped_count=len(reused),
        skipped=not pending,
        dry_run=dry_run,
        parallel=bool(workers and workers > 1),
        corpus_scan=corpus_scan,
        openings=openings.summary(),
        chapters=[r.to_dict() for r in results],
        usage=usage_total,
        errors=errors,
    )
    if write_manifest and not dry_run:
        result.manifest_path = str(
            write_preprocess_manifest(out_dir, result, chapters, results)
        )
        result.trace_path = str(write_preprocess_trace(out_dir, result))
    return result


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


def write_preprocess_manifest(
    out_dir: Path,
    result: PreprocessResult,
    chapters: list[Chapter],
    results: list[ChapterResult] | None = None,
) -> Path:
    """Write the book-level stage-2 manifest next to the cleaned chapters.

    The record shape matches stage 1's ``_manifest.json`` (``filename``,
    ``lines``, ``start_line``, ``end_line``, plus the title and both hashes), so
    the next stage can read either folder with the same reader. Stage 1 has used
    ``source_file`` and ``source_path`` over its lifetime, so both are carried.
    """
    if results is None:
        results = [ChapterResult(**c) for c in result.chapters]
    stage_one = load_split_manifest(result.chapters_path)
    source_file = (
        result.corpus_scan.get("source_file")
        or stage_one.get("source_file")
        or stage_one.get("source_path")
        or ""
    )
    payload = {
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
    return write_manifest(out_dir, PREPROCESS_MANIFEST_FILENAME, payload)


def write_preprocess_trace(out_dir: Path, result: PreprocessResult) -> Path:
    """Write the audit trail: per chapter, what was decided and what was proven."""
    return write_trace(out_dir, PREPROCESS_TRACE_FILENAME, result.to_dict())


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
