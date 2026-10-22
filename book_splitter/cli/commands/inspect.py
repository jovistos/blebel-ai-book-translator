"""inspect command: implementation and its human summary."""

from __future__ import annotations

import argparse
from typing import Any

from book_splitter.analysis.candidates import extract_candidates
from book_splitter.analysis.headers import assess_headers
from book_splitter.analysis.toc import detect_toc_block
from book_splitter.core.constants import (
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MAX_HEADERS,
    TOC_SCAN_LINES,
)
from book_splitter.core.report import summarize_payload
from book_splitter.documents.io import load_lines, source_size_bytes
from book_splitter.paths import (
    default_input_dir,
)

from .. import parents
from ..common import (
    _ensure_markdown_source,
    _print_json,
    _resolve_book,
)


def add_parser(sub: Any) -> None:
    """Register this command's parser (the help text is snapshot-tested)."""
    p_inspect = sub.add_parser(
        "inspect",
        help="deterministic structural report (no LLM; runs stage 0 for a "
        "non-Markdown source)",
    )
    parents.add_book(p_inspect, required=False)
    p_inspect.add_argument("--max-headers", type=int, default=DEFAULT_MAX_HEADERS)
    p_inspect.add_argument("--head-lines", type=int, default=40)
    p_inspect.add_argument("--toc-scan-lines", type=int, default=TOC_SCAN_LINES)
    p_inspect.add_argument(
        "--candidates", action="store_true", help="also list heuristic candidate lines"
    )
    p_inspect.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    p_inspect.add_argument("--json", action="store_true", help="machine-readable result")
    p_inspect.add_argument("--quiet", "-q", action="store_true")
    p_inspect.set_defaults(func=cmd_inspect)


def cmd_inspect(args: argparse.Namespace) -> int:
    """Deterministic pre-flight: no LLM, no API key.

    Stage 0 runs first when the source is EPUB/PDF/TXT, so the report exists for
    every supported format; a Markdown source is read as-is.
    """
    from ...agent.tools import scan_markdown_headers

    source = _ensure_markdown_source(_resolve_book(args), output_root=getattr(args, "output_root", None))
    lines = load_lines(source)
    headers = scan_markdown_headers(str(source))["headers"]
    toc = detect_toc_block(lines, scan_lines=args.toc_scan_lines)

    report = {
        "source": str(source),
        "input_dir": str(default_input_dir()),
        "line_count": len(lines),
        "size_bytes": source_size_bytes(source),
        "header_count": len(headers),
        "header_assessment": assess_headers(headers, len(lines)),
        "headers": headers[: args.max_headers],
        "printed_toc_block": toc,
        "first_lines": [
            {"line_number": i + 1, "text": line.strip()[:120]}
            for i, line in enumerate(lines[: args.head_lines])
            if line.strip()
        ],
    }
    if args.candidates:
        report["heuristic_candidates"] = extract_candidates(
            lines, max_candidates=args.max_candidates
        )["candidates"]
    report["candidate_count"] = len(report.get("heuristic_candidates") or [])
    if args.json:
        _print_json(report)
        return 0
    summary = summarize_payload("recon", report)
    if not args.quiet:
        for label, value in summary.rows:
            print(f"{label:<16}: {value}")
        if report["first_lines"]:
            print()
            print("first lines:")
            for line in report["first_lines"]:
                print(f"  L{line['line_number']:<5} {line['text']}")
    return 0
