"""Tools the agent uses to *submit* its answer."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from ...analysis.toc import detect_toc_block
from ...core.constants import DEFAULT_MIN_CHAPTERS
from ...documents.io import load_lines
from ...documents.text import excerpt, jdump, normalize_filename
from ...slicing.validation import validate_split_map
from .registry import split_stage


@split_stage
@tool
def validate_split_map_tool(file_path: str, split_mapping: list[dict]) -> str:
    """Dry-run a proposed chapter map and report problems before slicing anything.

    Call this with your candidate mapping (objects with `line_number`,
    `filename`, optional `title`) and fix every entry listed under `issues`
    before submitting the plan. Returns `{"ok", "issues", "warnings", "points"}`,
    where each point echoes the short text found at that line.
    """
    lines = load_lines(file_path)
    result = validate_split_map(
        lines,
        split_mapping,
        toc_block=detect_toc_block(lines),
        min_chapters=DEFAULT_MIN_CHAPTERS,
    )
    result["line_count"] = len(lines)
    return jdump(result)


@split_stage
@tool
def propose_split_map(
    split_mapping: list[dict], strategy: str = "unspecified", notes: str = ""
) -> str:
    """Submit the final chapter map. THIS IS THE TOOL THAT ENDS YOUR TASK.

    `split_mapping` is the complete list of split points in ascending order,
    each an object with `line_number` (1-indexed, the first line of the
    chapter), `filename` (e.g. `01-predgovor.md`) and optional `title`.
    `strategy` is one of `native_toc`, `markdown_headers`, `toc_titles`,
    `heuristics`.

    Everything the tool accepts is echoed back and then validated by the graph;
    if validation fails you will be asked to correct it, so never submit a map
    you have not dry-run with `validate_split_map_tool`.
    """
    cleaned: list[dict[str, Any]] = []
    for entry in list(split_mapping or []):
        if not isinstance(entry, dict):
            continue
        ln = entry.get("line_number", entry.get("line"))
        try:
            ln = int(ln)
        except (TypeError, ValueError):
            return jdump(
                {
                    "accepted": False,
                    "error": f"line_number must be an integer, got {entry.get('line_number')!r}",
                }
            )
        name = entry.get("filename") or entry.get("file") or entry.get("title") or f"{ln}.md"
        cleaned.append(
            {
                "line_number": ln,
                "filename": normalize_filename(str(name)),
                "title": excerpt(str(entry.get("title") or entry.get("original_header") or ""), 100),
            }
        )
    cleaned.sort(key=lambda e: e["line_number"])
    if not cleaned:
        return jdump({"accepted": False, "error": "split_mapping was empty"})
    return jdump(
        {
            "accepted": True,
            "strategy": strategy or "unknown",
            "notes": excerpt(str(notes or ""), 300),
            "count": len(cleaned),
            "split_mapping": cleaned,
        }
    )
