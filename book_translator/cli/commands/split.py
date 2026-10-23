"""split command: implementation and its human summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from book_translator.core.constants import (
    CHAPTERS_DIRNAME,
    DEFAULT_FRONT_MATTER_LINES,
    DEFAULT_MIN_CHAPTERS,
    DEFAULT_RECURSION_LIMIT,
    MAX_FEEDBACK_LOOPS,
)
from book_translator.core.errors import ConfigError
from book_translator.core.settings import PROVIDERS
from book_translator.documents.chapters import (
    has_chapter_files,
)
from book_translator.paths import (
    default_output_dir,
    resolve_output_dir,
)

from .. import parents
from ..common import (
    _ensure_markdown_source,
    _model_settings,
    _print_json,
    _resolve_book,
)


def add_parser(sub: Any) -> None:
    """Register this command's parser (the help text is snapshot-tested)."""
    p_split = sub.add_parser("split", help="stage 1: run the agent and write chapter files")
    parents.add_book(p_split, required=True)
    p_split.add_argument(
        "--out",
        "-o",
        default=None,
        help="output folder for chapter files "
        f"(default: {default_output_dir()}/<book>/{CHAPTERS_DIRNAME})",
    )
    p_split.add_argument(
        "--map",
        default=None,
        metavar="FILE",
        help="replay a chapter map from a JSON file instead of running the agent "
        "(no API key, no network); see book_translator/slicing/mapfile.py for the format",
    )
    p_split.add_argument("--provider", choices=sorted(PROVIDERS), default=None)
    p_split.add_argument("--model", default=None)
    p_split.add_argument("--base-url", default=None)
    p_split.add_argument("--api-key", default=None)
    p_split.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="sampling temperature for stage 1 (default: the config file, else 0.0)",
    )
    parents.add_llm_flags(p_split)
    p_split.add_argument(
        "--front-matter-lines",
        type=int,
        default=DEFAULT_FRONT_MATTER_LINES,
        help="max lines the agent may read from the front of the book",
    )
    p_split.add_argument(
        "--max-attempts",
        type=int,
        default=MAX_FEEDBACK_LOOPS,
        help="max LLM turns before giving up",
    )
    p_split.add_argument(
        "--min-chapters",
        type=int,
        default=DEFAULT_MIN_CHAPTERS,
        help="reject plans with fewer split points than this",
    )
    p_split.add_argument("--recursion-limit", type=int, default=DEFAULT_RECURSION_LIMIT)
    p_split.add_argument(
        "--no-front-matter",
        action="store_true",
        help="do not write 00-front-matter.md (content before chapter 1)",
    )
    p_split.add_argument(
        "--skip-if-present",
        "--resume",
        dest="skip_if_present",
        action="store_true",
        help="reuse the chapter files already in the output folder instead of "
        "splitting again (no model, no key, and the existing trace is left alone)",
    )
    p_split.add_argument(
        "--dry-run", action="store_true", help="validate the plan but write no chapter files"
    )
    p_split.add_argument("--json", action="store_true", help="machine-readable result")
    p_split.add_argument("--quiet", "-q", action="store_true")
    p_split.set_defaults(func=cmd_split)


def cmd_split(args: argparse.Namespace) -> int:
    from ...agent import run_split, run_split_from_map
    from ...core.errors import ToolError
    from ...slicing.mapfile import load_split_map

    source = _resolve_book(args)
    out_dir = resolve_output_dir(
        source, args.out, output_root=getattr(args, "output_root", None)
    )

    if getattr(args, "skip_if_present", False) and not args.dry_run and has_chapter_files(out_dir):
        # Checked here, before any provider settings are resolved: a run that is
        # going to be skipped must not need an API key either.
        from ...agent import result_from_existing_chapters

        if not args.quiet:
            print(f"source={source}")
            print(f"output={out_dir.expanduser().resolve()}")
        return _report_split(result_from_existing_chapters(out_dir), args)

    # Stage 0: the splitter only ever reads Markdown. A TXT is copied and an
    # EPUB/text-based PDF is converted before anything tries to count lines.
    source = _ensure_markdown_source(source, output_root=getattr(args, "output_root", None))

    if getattr(args, "map", None):
        # A supplied map needs no provider, no key and no network.
        try:
            mapping, notes = load_split_map(args.map)
        except ToolError as exc:
            print(f"map error: {exc}", file=sys.stderr)
            return 2
        if not args.quiet:
            print(f"map={Path(args.map).expanduser()}")
            print(f"source={source}")
            print(f"output={out_dir.expanduser().resolve()}")
            for note in notes:
                print(f"  · {note}")
        result = run_split_from_map(
            str(source),
            str(out_dir),
            mapping,
            min_chapters=args.min_chapters,
            include_front_matter=not args.no_front_matter,
            dry_run=args.dry_run,
        )
        return _report_split(result, args)

    try:
        settings = _model_settings(args, "split")
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    if not args.quiet:
        print(f"provider={settings.provider} model={settings.model} base_url={settings.base_url}")
        print(f"source={source}")
        print(f"output={out_dir.expanduser().resolve()}")

    result = run_split(
        str(source),
        str(out_dir),
        settings=settings,
        front_matter_lines=args.front_matter_lines,
        max_attempts=args.max_attempts,
        min_chapters=args.min_chapters,
        include_front_matter=not args.no_front_matter,
        dry_run=args.dry_run,
        recursion_limit=args.recursion_limit,
    )
    return _report_split(result, args)


def _report_split(result, args: argparse.Namespace) -> int:
    """Print one split run (agent or supplied map) and return the exit code."""
    if not args.quiet:
        for line in result.log:
            print(f"  · {line}")

    if args.json:
        _print_json(result.to_dict())
        return 0 if result.ok else 1

    _print_summary(result, dry_run=args.dry_run)
    return 0 if result.ok else 1


def _print_summary(result, dry_run: bool = False) -> None:
    print()
    print(f"strategy        : {result.strategy}")
    print(f"llm calls       : {result.attempts}  tokens: {result.usage.get('total_tokens', 0)}")
    print(f"chapters        : {result.chapter_count}")
    print(f"output dir      : {result.output_dir}")
    if result.verification:
        if "reassembles_byte_for_byte" in result.verification:
            print(f"byte-for-byte   : {result.verification.get('reassembles_byte_for_byte')}")
            print(
                f"lines           : {result.verification.get('output_lines')}"
                f"/{result.verification.get('source_lines')}"
            )
        elif result.verification.get("reused_existing_chapters"):
            print("reused          : chapter files were already there; nothing was re-sliced")
    if result.files:
        print("files:")
        for f in result.files:
            start, end = f.get("start_line"), f.get("end_line")
            span = f"{start:>5}-{end:<5}" if start is not None and end is not None else " " * 11
            print(
                f"  {f.get('filename', '?'):<52} lines {span} "
                f"{int(f.get('bytes') or 0):>8} B"
            )
    if result.trace_path:
        print(f"trace           : {result.trace_path}")
    if not result.ok:
        print(f"\nFAILED: {result.error}", file=sys.stderr)
        for issue in (result.validation or {}).get("issues", []):
            print(f"  - {issue}", file=sys.stderr)
        return
    print(
        "\nDRY RUN — plan validated, nothing written."
        if dry_run
        else "\nOK — every source line is accounted for."
    )
