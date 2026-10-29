"""scan command: implementation and its human summary."""

from __future__ import annotations

import argparse
from typing import Any

from book_translator.core.report import summarize_payload
from book_translator.documents.chapters import (
    discover_chapters,
    load_split_manifest,
)
from book_translator.documents.window import (
    chapter_window_limit_for_path,
)

from .. import parents
from ..common import (
    _print_json,
    _resolve_chapters_dir,
)


def add_parser(sub: Any) -> None:
    """Register this command's parser (the help text is snapshot-tested)."""
    p_scan = sub.add_parser(
        "scan",
        help="stage 3 pre-flight: header evidence and damage profile (no LLM, no writes)",
    )
    parents.add_book(p_scan, required=False)
    p_scan.add_argument(
        "--out",
        "-o",
        default=None,
        help="the chapters folder from stage 2 (default: data/output/<book>/02-chapters)",
    )
    p_scan.add_argument(
        "--headers", action="store_true", help="print the full header record per chapter"
    )
    p_scan.add_argument(
        "--all-headers", action="store_true", help="include headers that look fine"
    )
    p_scan.add_argument("--digest", action="store_true", help="print the compact text digest")
    p_scan.add_argument("--examples", type=int, default=3)
    p_scan.add_argument(
        "--first-chapters",
        "--chapters",
        dest="first_chapters",
        type=int,
        default=None,
        help="scan only the first N body chapters (the whole book when omitted)",
    )
    p_scan.add_argument("--json", action="store_true", help="machine-readable result")
    p_scan.add_argument("--quiet", "-q", action="store_true")
    p_scan.set_defaults(func=cmd_scan)


def cmd_scan(args: argparse.Namespace) -> int:
    """Deterministic damage profile of a split book: no LLM, no writes."""
    from book_translator.stages.preprocess.corpus import scan_corpus_damage
    from book_translator.stages.preprocess.inspect import chapter_digest, inspect_chapter

    chapters_dir = _resolve_chapters_dir(args.book, args.in_dir, args.out, getattr(args, "output_root", None))
    chapters = discover_chapters(chapters_dir)
    window = chapter_window_limit_for_path(
        chapters_dir,
        getattr(args, "first_chapters", None),
        getattr(args, "limit", None),
    )
    if window is not None:
        chapters = chapters[:window]
    if args.headers:
        for chapter in chapters:
            _print_json(inspect_chapter(chapter.path, include_all=args.all_headers))
        return 0
    if args.digest:
        for chapter in chapters:
            report = inspect_chapter(chapter.path, include_all=args.all_headers)
            print(chapter_digest(report))
            print()
    report = scan_corpus_damage(chapters, max_examples_per_pattern=args.examples)
    report["chapters_path"] = str(chapters_dir)
    report["chapter_count"] = len(chapters)
    report["manifest"] = {
        "strategy": load_split_manifest(chapters_dir).get("strategy"),
        "chapter_count": len(chapters),
    }
    if args.json:
        _print_json(report)
        return 0
    summary = summarize_payload("scan", report)
    if not args.quiet:
        for label, value in summary.rows:
            print(f"{label:<16}: {value}")
        counts = report.get("counts") or {}
        if counts:
            print()
            for name, value in sorted(counts.items(), key=lambda item: -item[1]):
                print(f"  {name:<28} {value:>5}")
        if not counts:
            print("  none recognised — stage 3 has nothing to repair")
    return 0
