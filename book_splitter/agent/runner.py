"""Entry point: run the agent end to end and return a structured result."""

from __future__ import annotations

import json
from pathlib import Path

from ..core.constants import (
    DEFAULT_FRONT_MATTER_LINES,
    DEFAULT_MIN_CHAPTERS,
    DEFAULT_RECURSION_LIMIT,
    MAX_FEEDBACK_LOOPS,
    TRACE_FILENAME,
)
from ..core.settings import LLMSettings
from .graph import build_graph
from .result import SplitterResult
from .state import SplitterState


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
) -> SplitterResult:
    """Run the agent end to end and return a structured result.

    Raises ``SystemExit`` when the source file does not exist, so a CLI caller
    fails with one clear line instead of a traceback.
    """
    source = str(Path(source_path).expanduser().resolve())
    if not Path(source).is_file():
        raise SystemExit(f"source file not found: {source_path}")
    out_dir = Path(output_dir).expanduser()
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
    """Persist the audit trail: what the agent proposed and what was verified."""
    trace = Path(out_dir) / TRACE_FILENAME
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
