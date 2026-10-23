"""preprocess command: implementation and its human summary."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from book_translator.core.constants import (
    MAX_FEEDBACK_LOOPS,
)
from book_translator.core.errors import ConfigError
from book_translator.documents.window import (
    chapter_window_limit_for_path,
)

from .. import parents
from ..common import (
    _model_settings,
    _print_json,
    _print_openings_line,
    _resolve_chapters_dir,
    _window_label,
)


def add_parser(sub: Any) -> None:
    """Register this command's parser (the help text is snapshot-tested)."""
    p_pre = sub.add_parser(
        "preprocess",
        help="stage 2: clean each chapter, one chapter at a time, and write the result",
    )
    parents.add_book(p_pre, required=False)
    p_pre.add_argument(
        "--out",
        "-o",
        default=None,
        help="the chapters folder from stage 1 (default: data/output/<book>/01-chapters)",
    )
    p_pre.add_argument(
        "--preprocessed-out",
        default=None,
        help="where to write cleaned chapters (default: data/output/<book>/02-preprocessed)",
    )
    parents.add_model_flags(p_pre)
    p_pre.add_argument(
        "--max-attempts",
        type=int,
        default=MAX_FEEDBACK_LOOPS,
        help="max LLM turns per chapter before giving up on it",
    )
    p_pre.add_argument(
        "--workers",
        type=int,
        default=1,
        help="chapters processed at once (default 1: strictly one chapter at a time)",
    )
    p_pre.add_argument(
        "--no-manifest",
        action="store_true",
        help="do not write _preprocess_manifest.json",
    )
    p_pre.add_argument(
        "--first-chapters",
        "--chapters",
        dest="first_chapters",
        type=int,
        default=None,
        help="process only the first N body chapters (the whole book when omitted)",
    )
    p_pre.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process only the first N body chapters (kept as an alias of --first-chapters)",
    )
    p_pre.add_argument(
        "--skip-if-present",
        "--resume",
        dest="skip_if_present",
        action="store_true",
        help="reuse a chapter whose cleaned file a previous run verified, and start "
        "with the first chapter that is missing",
    )
    p_pre.add_argument(
        "--dry-run",
        action="store_true",
        help="validate every chapter's plan but write no files",
    )
    p_pre.set_defaults(func=cmd_preprocess)


def cmd_preprocess(args: argparse.Namespace) -> int:
    from book_translator.stages.preprocess_agent import run_preprocess

    chapters_dir = _resolve_chapters_dir(args.book, args.in_dir, args.out, getattr(args, "output_root", None))
    out_dir = args.preprocessed_out or None

    try:
        settings = _model_settings(args, "preprocess")
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    if not args.quiet:
        print(f"provider={settings.provider} model={settings.model} base_url={settings.base_url}")

    if not args.quiet:
        print(f"chapters={chapters_dir.expanduser().resolve()}")
        print(f"output={out_dir or 'data/output/<book>/02-preprocessed/'}")
    window = chapter_window_limit_for_path(
        chapters_dir, getattr(args, "first_chapters", None), args.limit
    )
    if not args.quiet:
        label = _window_label(args, window)
        if label is not None:
            print(f"window      : {label}")
    if not args.quiet and args.skip_if_present:
        print("resume      : chapters already cleaned and verified will be reused")

    result = run_preprocess(
        str(chapters_dir),
        out_dir,
        settings=settings,
        max_attempts=args.max_attempts,
        dry_run=args.dry_run,
        recursion_limit=args.recursion_limit,
        workers=args.workers,
        limit=window,
        skip_if_present=args.skip_if_present,
        write_manifest=not args.no_manifest,
    )

    if not args.quiet:
        for chapter in result.chapters:
            for line in chapter.get("log", []):
                print(f"  · {chapter['filename']} {line}")

    if args.json:
        _print_json(result.to_dict())
        return 0 if result.ok else 1

    _print_preprocess_summary(result, dry_run=args.dry_run)
    return 0 if result.ok else 1


def _print_preprocess_summary(result, dry_run: bool = False) -> None:
    print()
    print(
        f"chapters        : {result.ok_count}/{result.chapter_count} clean"
        + (f"  reused: {result.skipped_count}" if result.skipped_count else "")
    )
    print(f"llm calls       : {sum(c.get('attempts', 0) for c in result.chapters)}"
          f"  tokens: {result.usage.get('total_tokens', 0)}")
    if result.corpus_scan:
        counts = result.corpus_scan.get("counts", {})
        print("damage found    : " + ", ".join(f"{k}={v}" for k, v in counts.items() if v))
    _print_openings_line(result.openings)
    print(f"output dir      : {result.output_dir}")
    if result.chapters:
        print("files:")
        for chapter in result.chapters:
            flag = "ok " if chapter["ok"] else "FAIL"
            print(
                f"  {flag} {chapter['filename']:<52} lines "
                f"{chapter['source_lines']:>5} -> {chapter['output_lines']:<5} "
                f"ops={len(chapter['operations'])} fixes={len(chapter['text_fixes'])}"
            )
    if result.manifest_path:
        print(f"manifest        : {result.manifest_path}")
    if result.trace_path:
        print(f"trace           : {result.trace_path}")
    if not result.ok:
        print("\nFAILED chapters:", file=sys.stderr)
        for error in result.errors:
            print(f"  - {error}", file=sys.stderr)
        return
    print(
        "\nDRY RUN — every plan validated, nothing written."
        if dry_run
        else "\nOK — every chapter cleaned and verified."
    )
