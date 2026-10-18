"""Entry points: run the agent end to end, or replay a supplied map."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..analysis.toc import detect_toc_block
from ..core.constants import (
    DEFAULT_FRONT_MATTER_LINES,
    DEFAULT_MIN_CHAPTERS,
    DEFAULT_RECURSION_LIMIT,
    FRONT_MATTER_FILENAME,
    MAX_FEEDBACK_LOOPS,
    TRACE_FILENAME,
)
from ..core.settings import LLMSettings
from ..documents.chapters import discover_chapters, has_chapter_files, load_split_manifest
from ..documents.io import load_lines
from ..slicing.slice import preview_split, slice_book
from ..slicing.validation import validate_split_map
from ..slicing.verify import verify_output
from .graph import build_graph
from .result import SplitterResult
from .state import SplitterState

#: The strategy label recorded for a run that was given its map instead of finding it.
PRESET_MAP_STRATEGY = "preset_map"
#: The strategy label recorded when existing chapter files were reused untouched.
REUSED_CHAPTERS_STRATEGY = "reused_existing_chapters"


def result_from_existing_chapters(output_dir: str | Path) -> SplitterResult:
    """Describe a chapters folder that is already there, without splitting.

    This is the ``--skip-if-present`` path: the folder holds chapter files, so
    stage 1 has nothing to do and — importantly — must not touch the existing
    provenance. No graph is built, no model is called, and ``_agent_trace.json``
    from the original split is left exactly as it was (it is only *pointed at*).
    """
    out = Path(output_dir).expanduser()
    chapters = discover_chapters(out)
    manifest = load_split_manifest(out)
    trace = out / TRACE_FILENAME
    result = SplitterResult(
        ok=True,
        source_path=str(manifest.get("source_path") or ""),
        output_dir=str(out.resolve()),
        strategy=REUSED_CHAPTERS_STRATEGY,
        chapter_count=len(chapters),
        files=list(manifest.get("files") or []),
        validation={"ok": True, "reused": True, "issues": [], "warnings": []},
        verification={"reused_existing_chapters": True},
        attempts=0,
        usage={},
        log=[
            f"skip: {len(chapters)} chapter file(s) already in {out} — the splitter "
            "was not run",
        ],
    )
    result.trace_path = str(trace) if trace.is_file() else None
    return result


def run_split(
    source_path: str,
    output_dir: str,
    settings: LLMSettings | None = None,
    front_matter_lines: int = DEFAULT_FRONT_MATTER_LINES,
    max_attempts: int = MAX_FEEDBACK_LOOPS,
    min_chapters: int = DEFAULT_MIN_CHAPTERS,
    include_front_matter: bool = True,
    dry_run: bool = False,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    skip_if_present: bool = False,
) -> SplitterResult:
    """Run the agent end to end and return a structured result.

    With ``skip_if_present`` a folder that already holds chapter files is reused
    as-is: no graph, no model, no rewrite of the existing provenance.

    Raises ``SystemExit`` when the source file does not exist, so a CLI caller
    fails with one clear line instead of a traceback.
    """
    source = str(Path(source_path).expanduser().resolve())
    if not Path(source).is_file():
        raise SystemExit(f"source file not found: {source_path}")
    out_dir = Path(output_dir).expanduser()
    if skip_if_present and not dry_run and has_chapter_files(out_dir):
        return result_from_existing_chapters(out_dir)
    if dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    app = build_graph(settings)
    initial: SplitterState = {
        "source_path": source,
        "output_dir": str(out_dir),
        "front_matter_lines": front_matter_lines,
        "max_attempts": max_attempts,
        "min_chapters": min_chapters,
        "include_front_matter": include_front_matter,
        "dry_run": dry_run,
        "attempts": 0,
        "usage": {},
        "log": [],
        "feedback": [],
        "messages": [],
    }
    final = app.invoke(initial, config={"recursion_limit": recursion_limit})
    return summarise_run(final, source, out_dir, dry_run)


def run_split_from_map(
    source_path: str,
    output_dir: str,
    split_mapping: Iterable[dict[str, Any]],
    *,
    min_chapters: int = DEFAULT_MIN_CHAPTERS,
    include_front_matter: bool = True,
    dry_run: bool = False,
    strategy: str = PRESET_MAP_STRATEGY,
    skip_if_present: bool = False,
) -> SplitterResult:
    """Slice a book from a map that was supplied instead of discovered.

    This is the ``--map`` path: no model, no API key, no network. The map is put
    through exactly the validation the graph applies to an agent's plan (so a
    ``role: "prose-chapter"`` entry still has to open a paragraph and carry its
    evidence), then sliced verbatim and verified byte for byte. A rejected map
    writes nothing but its trace.

    With ``skip_if_present`` a folder that already holds chapter files wins over
    the supplied map: nothing is re-sliced and no trace is overwritten.

    Raises ``SystemExit`` when the source file does not exist, like
    :func:`run_split`.
    """
    source = str(Path(source_path).expanduser().resolve())
    if not Path(source).is_file():
        raise SystemExit(f"source file not found: {source_path}")
    out_dir = Path(output_dir).expanduser()
    if skip_if_present and not dry_run and has_chapter_files(out_dir):
        return result_from_existing_chapters(out_dir)
    if dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    lines = load_lines(source)
    proposal = [dict(entry) for entry in split_mapping]
    toc_block = detect_toc_block(lines)
    validation = validate_split_map(
        lines, proposal, toc_block=toc_block, min_chapters=min_chapters
    )

    final: dict[str, Any] = {
        "proposed_splits": proposal,
        "proposal_strategy": strategy,
        "validation": validation,
        "toc_block": toc_block,
        "attempts": 0,
        "usage": {},
        "log": [
            f"map: {len(proposal)} split point(s) supplied; the model was not used",
            f"validate: ok={validation['ok']} chapters={validation['chapter_count']} "
            f"prose_chapters={validation.get('prose_chapter_count', 0)} "
            f"issues={len(validation['issues'])} warnings={len(validation['warnings'])}",
        ],
    }

    if validation["ok"]:
        if dry_run:
            manifest = preview_split(
                source,
                proposal,
                str(out_dir),
                include_front_matter=include_front_matter,
                front_matter_filename=FRONT_MATTER_FILENAME,
            )
        else:
            manifest = slice_book(
                source,
                proposal,
                str(out_dir),
                include_front_matter=include_front_matter,
                front_matter_filename=FRONT_MATTER_FILENAME,
                write_manifest=True,
            )
            final["verification"] = verify_output(manifest)
        final["manifest"] = manifest
        final["log"].append(
            f"slice: {'would write' if dry_run else 'wrote'} {manifest['file_count']} files "
            f"({manifest['total_lines']}/{manifest['source_lines']} lines)"
        )
    else:
        final["result"] = {
            "error": "the supplied map did not pass validation; nothing was written"
        }

    return summarise_run(final, source, out_dir, dry_run)


def summarise_run(
    final: dict, source: str, out_dir: Path, dry_run: bool
) -> SplitterResult:
    """Turn the final graph state into a :class:`SplitterResult` plus its trace."""
    manifest = final.get("manifest") or {}
    verification = final.get("verification") or {}
    validation = final.get("validation") or {}
    if dry_run:
        ok = bool(validation.get("ok")) and bool(manifest)
    else:
        ok = bool(manifest) and bool(verification.get("ok"))

    result = SplitterResult(
        ok=ok,
        source_path=source,
        output_dir=str(out_dir.resolve()),
        strategy=final.get("proposal_strategy"),
        chapter_count=manifest.get("chapter_count", 0),
        files=manifest.get("files", []),
        validation=validation,
        verification=verification,
        header_assessment=final.get("header_assessment", {}),
        attempts=int(final.get("attempts", 0)),
        usage=final.get("usage", {}),
        error=None
        if ok
        else (final.get("result") or {}).get("error")
        or ("nothing was written" if not manifest else "verification failed"),
        log=final.get("log", []),
    )
    result.trace_path = str(write_trace(out_dir, result, final))
    return result


def write_trace(out_dir: Path, result: SplitterResult, final: dict) -> Path:
    """Persist the audit trail: what the agent proposed and what was verified.

    The folder is created when missing: a run that never got as far as slicing
    has no output folder yet, and that is exactly the run whose trace — the
    proposed map, the validation issues, the tool log — explains the failure.
    """
    trace = Path(out_dir) / TRACE_FILENAME
    trace.parent.mkdir(parents=True, exist_ok=True)
    trace.write_text(
        json.dumps(
            {
                "result": result.to_dict(),
                "toc_block": final.get("toc_block"),
                "proposed_splits": final.get("proposed_splits"),
                "validation": final.get("validation"),
                "manifest": final.get("manifest"),
                "verification": final.get("verification"),
                "tool_log": final.get("log", []),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return trace
