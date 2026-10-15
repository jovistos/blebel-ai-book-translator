"""The scout node: deterministic recon that runs before the model ever speaks."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ...analysis.headers import assess_headers
from ...analysis.toc import detect_toc_block
from ...core.constants import MAX_HEADERS_IN_REPORT
from ...documents.io import load_lines
from ..prompts import SCOUT_REPORT_TEMPLATE, SYSTEM_PROMPT
from ..state import SplitterState
from ..tools import detect_native_toc, scan_markdown_headers


def make_scout_node():
    """Build the ``scout`` node: headers, native TOC and TOC block, no LLM."""

    def scout(state: SplitterState) -> dict[str, Any]:
        source = state["source_path"]
        lines = load_lines(source)
        line_count = len(lines)

        headers_payload = scan_markdown_headers(source)
        native_toc = detect_native_toc(source)
        toc_block = detect_toc_block(lines)
        assessment = assess_headers(headers_payload["headers"], line_count)

        report = build_scout_report(
            source=source,
            line_count=line_count,
            headers_payload=headers_payload,
            toc_block=toc_block,
            assessment=assessment,
            native_toc=native_toc,
        )
        task = (
            f"Split the book at {source} into chapter files inside {state['output_dir']}.\n"
            "Follow the execution protocol. Do not read the body of the book."
        )
        return {
            "line_count": line_count,
            "native_toc": native_toc,
            "headers": headers_payload["headers"],
            "header_assessment": assessment,
            "toc_block": toc_block,
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"{task}\n\n{report}"),
            ],
            "log": [
                f"scout: {line_count} lines, {headers_payload['count']} headers, "
                f"verdict={assessment['verdict']} (score {assessment['score']}), "
                f"toc_block={toc_block.get('start_line')}-{toc_block.get('end_line')}"
            ],
        }

    return scout


def build_scout_report(
    *,
    source: str,
    line_count: int,
    headers_payload: dict[str, Any],
    toc_block: dict[str, Any],
    assessment: dict[str, Any],
    native_toc: dict[str, Any] | None = None,
) -> str:
    """Render the deterministic report that opens the conversation.

    Only line numbers and short header strings are included: this text is the
    model's entire view of the book.
    """
    shown = headers_payload["headers"][:MAX_HEADERS_IN_REPORT]
    header_list = (
        "\n".join(f"  {h['line_number']:>6}  {'#' * h['level']:<6} {h['text']}" for h in shown)
        or "  (no markdown headers found)"
    )
    if headers_payload["count"] > len(shown):
        header_list += f"\n  ... {headers_payload['count'] - len(shown)} more headers omitted"

    notes: list[str] = []
    if native_toc and native_toc.get("available"):
        notes.append(f"native TOC available with {native_toc.get('count')} entries")
    if toc_block.get("found"):
        notes.append(
            f"printed TOC block at lines {toc_block['start_line']}-{toc_block['end_line']} "
            f"(detected via {toc_block['method']}); split points inside it are invalid"
        )
    if assessment["verdict"] == "noisy":
        notes.append("header set is noisy: you must filter it before building the map")
    if assessment["verdict"] == "absent":
        notes.append("no usable headers: use the TOC / heuristics routes")

    return SCOUT_REPORT_TEMPLATE.format(
        source_path=source,
        line_count=line_count,
        toc_block=json.dumps(toc_block, ensure_ascii=False),
        assessment=json.dumps(assessment, ensure_ascii=False),
        header_count=headers_payload["count"],
        header_list=header_list,
        notes="\n".join(f"  - {n}" for n in notes) or "  - (none)",
    )
