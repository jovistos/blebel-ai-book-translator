"""Stage-9 entry point: postprocess every translated chapter, then its TOC.

The runner mirrors stage 2's contract:

* one book-level graph first (naming policy + TOC decision);
* one chapter-level graph per chapter, in reading order, with the canonical title
  from the book plan handed to the chapter state;
* a deterministic TOC write after the chapters, generated from the final titles;
* one manifest and one trace for the whole stage.

No chapter prose is passed between graphs or stored in a result. The result
records titles, line numbers, decisions and hashes only.
"""

from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..agent.tools.postprocess import reset_chapter_scope, set_chapter_scope
from ..core.constants import (
    DEFAULT_POSTPROCESS_ATTEMPTS,
    DEFAULT_RECURSION_LIMIT,
    POSTPROCESS_MANIFEST_FILENAME,
    POSTPROCESS_TRACE_FILENAME,
    TRANSLATED_CHAPTERS_DIRNAME,
)
from ..core.errors import ToolError
from ..core.settings import LLMSettings
from ..documents.chapters import (
    file_sha256,
    postprocessed_chapters_dir,
)
from ..documents.io import load_lines
from ..documents.text import norm
from ..documents.window import normalize_limit
from ..openings import OpeningAuditResult, audit_openings
from ..postprocess.inspect import (
    chapter_files,
    matter_kind,
    scan_book_structure,
    scan_chapter_structure,
)
from ..postprocess.toc import (
    apply_toc,
    build_toc_entries,
    detect_toc,
    toc_consistency,
    verify_toc,
)
from .graph import build_chapter_postprocess_graph, build_postprocess_graph
from .state import BookPlanState, ChapterPostprocessState

#: Called once per finished chapter as ``(chapter_record, completed, total)``.
ProgressHook = Callable[[dict[str, Any], int, int], None]


@dataclass
class ChapterPostprocessResult:
    """Outcome of postprocessing one chapter."""

    chapter_id: str
    filename: str
    source_path: str
    output_path: str | None = None
    ok: bool = False
    reused: bool = False
    title: str = ""
    title_action: str = "keep"
    source_lines: int = 0
    output_lines: int = 0
    heading_decisions: int = 0
    demoted_lines: list[int] = field(default_factory=list)
    promoted_lines: list[int] = field(default_factory=list)
    merged_subtitles: list[int] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)
    attempts: int = 0
    error: str | None = None
    log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PostprocessResult:
    """Outcome of a whole stage-9 run."""

    ok: bool
    chapters_path: str
    output_dir: str
    chapter_count: int = 0
    ok_count: int = 0
    reused_count: int = 0
    skipped: bool = False
    dry_run: bool = False
    limited: bool = False
    first_chapters: int | None = None
    book_plan: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    toc: dict[str, Any] = field(default_factory=dict)
    web: dict[str, Any] = field(default_factory=dict)
    totals: dict[str, int] = field(default_factory=dict)
    openings: dict[str, Any] = field(default_factory=dict)
    chapters: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    manifest_path: str | None = None
    trace_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_postprocess_manifest(output_dir: str | Path) -> dict[str, Any]:
    """Read ``_postprocess_manifest.json`` from a folder."""
    path = Path(output_dir).expanduser() / POSTPROCESS_MANIFEST_FILENAME
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


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
    source_dir = Path(chapters_path).expanduser()
    if source_dir.is_file():
        source_dir = source_dir.parent
    source_dir = source_dir.resolve()
    nested = source_dir / TRANSLATED_CHAPTERS_DIRNAME
    if nested.is_dir():
        source_dir = nested
    files = chapter_files(source_dir)
    if not files:
        raise ToolError(
            f"no translated chapter files in {source_dir} — run `concatenate` first"
        )
    out_dir = (
        Path(output_dir).expanduser()
        if output_dir is not None
        else postprocessed_chapters_dir(source_dir)
    )
    book = scan_book_structure(source_dir)
    matters = {
        str(summary.get("filename")): str(summary.get("matter") or "body")
        for summary in book.get("chapters") or []
    }
    try:
        if out_dir.resolve() == source_dir.resolve():
            raise ToolError(
                "the postprocessed output folder must be different from the input "
                "translated-chapters folder"
            )
    except OSError:
        pass
    selected = _selected_files(files, limit, matters)
    selected_names = {path.name for path in selected}

    book_app = build_postprocess_graph(
        settings,
        model=book_model if book_model is not None else model,
    )
    book_initial: BookPlanState = {
        "chapters_path": str(source_dir),
        "max_attempts": max_attempts,
        "dry_run": dry_run,
        "require_web_search": require_web_search,
        "web_search_attempted": False,
        "web_searches": [],
        "attempts": 0,
        "usage": {},
        "log": [],
        "feedback": [],
        "messages": [],
    }
    book_scope = set_chapter_scope(source_dir)
    web: dict[str, Any] = {
        "required": require_web_search,
        "attempted": False,
        "searches": [],
    }
    try:
        book_final = book_app.invoke(
            book_initial, config={"recursion_limit": recursion_limit}
        )
    except Exception as exc:
        result = PostprocessResult(
            ok=False,
            chapters_path=str(source_dir),
            output_dir=str(out_dir),
            chapter_count=len(selected),
            web=web,
            errors=[f"book plan failed: {type(exc).__name__}: {exc}"],
        )
        if not dry_run and write_manifest:
            out_dir.mkdir(parents=True, exist_ok=True)
            result.trace_path = str(write_postprocess_trace(out_dir, result))
        return result
    finally:
        reset_chapter_scope(book_scope)
    validation = book_final.get("validation") or {}
    plan = book_final.get("book_plan") or {}
    web = {
        "required": require_web_search,
        "attempted": bool(book_final.get("web_search_attempted")),
        "searches": list(book_final.get("web_searches") or []),
    }
    if validation.get("naming"):
        # Validation normalises the plan (and may enable the optional chapter
        # names); record the effective policy rather than the submitted one.
        plan = {
            **plan,
            "naming": validation["naming"],
            "toc": validation.get("toc") or plan.get("toc") or {},
            "title_evidence": validation.get(
                "title_evidence", plan.get("title_evidence") or ""
            ),
            "title_source": validation.get(
                "title_source", plan.get("title_source") or ""
            ),
        }
    canonical = dict(validation.get("canonical_titles") or {})
    errors: list[str] = []
    usage_total: dict[str, int] = dict(book_final.get("usage") or {})

    if not validation.get("ok"):
        detail = validation.get("issues") or [(book_final.get("result") or {}).get("error")]
        errors.append(
            "book plan rejected: "
            + "; ".join(str(item) for item in detail if item)
        )
        result = PostprocessResult(
            ok=False,
            chapters_path=str(source_dir),
            output_dir=str(out_dir),
            chapter_count=len(selected),
            book_plan=plan,
            validation=validation,
            web=web,
            chapters=[],
            usage=usage_total,
            errors=errors,
        )
        if not dry_run and write_manifest:
            out_dir.mkdir(parents=True, exist_ok=True)
            result.trace_path = str(write_postprocess_trace(out_dir, result))
        return result

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

    results: list[ChapterPostprocessResult] = []
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
        scope = set_chapter_scope(source_dir)
        try:
            assert chapter_app is not None
            final = chapter_app.invoke(initial, config={"recursion_limit": recursion_limit})
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
        finally:
            reset_chapter_scope(scope)

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
    results = [
        reused.get(path.name) or by_name[path.name] for path in selected
    ]

    # Final phase: the deterministic opening audit. It reads only the first lines
    # of every chapter and removes what is provably junk there — a stray copy of
    # the chapter number under the title (`# Poglavlje 2` then `2`), a duplicated
    # title line, a leftover page number. It runs over the folder as it stands,
    # reused chapters included, and before the TOC is built.
    openings = _audit_openings(out_dir, selected, results, dry_run=dry_run)
    errors.extend(openings.errors)

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
    result = PostprocessResult(
        ok=bool(results)
        and all(item.ok for item in results)
        and bool(toc_summary.get("ok"))
        and openings.ok,
        chapters_path=str(source_dir),
        output_dir=str(out_dir),
        chapter_count=len(results),
        ok_count=sum(1 for item in results if item.ok),
        reused_count=sum(1 for item in results if item.reused),
        skipped=not pending,
        dry_run=dry_run,
        limited=bool(normalize_limit(limit) is not None),
        first_chapters=normalize_limit(limit),
        book_plan=plan,
        validation=validation,
        toc=toc_summary,
        web=web,
        totals=totals,
        openings=openings.summary(),
        chapters=[item.to_dict() for item in results],
        usage=usage_total,
        errors=errors,
    )
    if write_manifest and not dry_run:
        result.manifest_path = str(write_postprocess_manifest(out_dir, result))
        result.trace_path = str(write_postprocess_trace(out_dir, result))
    return result


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


def write_postprocess_manifest(out_dir: str | Path, result: PostprocessResult) -> Path:
    """Write ``_postprocess_manifest.json``: the book-level index."""
    folder = Path(out_dir).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / POSTPROCESS_MANIFEST_FILENAME
    payload = {
        "stage": "postprocess",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_postprocess_trace(out_dir: str | Path, result: PostprocessResult) -> Path:
    """Write ``_postprocess_trace.json``: the same run, for debugging."""
    path = Path(out_dir).expanduser() / POSTPROCESS_TRACE_FILENAME
    path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


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
