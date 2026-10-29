"""bind command: implementation and its human summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from book_translator.core.constants import (
    POSTPROCESSED_CHAPTERS_DIRNAME,
)
from book_translator.core.errors import ToolError
from book_translator.documents.chapters import (
    has_postprocessed_chapter_files,
    postprocessed_chapters_dir,
    translated_book_path,
)

from .. import parents
from ..common import (
    _command_paths,
    _command_text_dir,
    _print_json,
)


def add_parser(sub: Any) -> None:
    """Register this command's parser (the help text is snapshot-tested)."""
    p_bind = sub.add_parser(
        "bind",
        aliases=["full-book", "compile"],
        help="stage 11: concatenate the postprocessed chapters into one Markdown book",
    )
    parents.add_book(p_bind, required=False)
    p_bind.add_argument(
        "--out",
        "-o",
        default=None,
        help="the chapter folder (default: the cleaned data/output/<book>/03-preprocessed "
        "when stage 3 wrote it, else data/output/<book>/02-chapters)",
    )
    p_bind.add_argument(
        "--postprocessed-in",
        default=None,
        help="stage 10's output folder (default: data/output/<book>/10-postprocessed-chapters)",
    )
    p_bind.add_argument(
        "--book-out",
        default=None,
        help="where the single Markdown book goes "
        "(default: data/output/<book>/11-translated-book.md)",
    )
    p_bind.add_argument(
        "--no-manifest",
        action="store_true",
        help="do not write _bind_manifest.json / _bind_trace.json",
    )
    p_bind.add_argument(
        "--dry-run",
        action="store_true",
        help="assemble and verify the book in memory but write no file",
    )
    p_bind.add_argument("--json", action="store_true", help="machine-readable result")
    p_bind.add_argument("--quiet", "-q", action="store_true")
    p_bind.set_defaults(func=cmd_bind)


def cmd_bind(args: argparse.Namespace) -> int:
    """Stage 11: concatenate the postprocessed chapters into one Markdown book."""
    from book_translator.stages.binding import run_bind

    if args.postprocessed_in:
        source_path = Path(args.postprocessed_in).expanduser()
        if source_path.is_file():
            source_path = source_path.parent
        if (source_path / POSTPROCESSED_CHAPTERS_DIRNAME).is_dir():
            source_path = source_path / POSTPROCESSED_CHAPTERS_DIRNAME
    else:
        chapters_path = _command_text_dir(args, _command_paths(args))
        source_path = postprocessed_chapters_dir(chapters_path)
    if not has_postprocessed_chapter_files(source_path):
        print(
            f"no postprocessed chapter files in {source_path.expanduser().resolve()} — "
            "run `postprocess` first",
            file=sys.stderr,
        )
        return 2
    output_path = (
        Path(args.book_out).expanduser()
        if args.book_out
        else translated_book_path(source_path)
    )
    if not args.quiet:
        print(f"postprocessed-chapters={source_path.expanduser().resolve()}")
        print(f"book={output_path.expanduser().resolve()}")
    try:
        result = run_bind(
            str(source_path),
            str(output_path),
            dry_run=args.dry_run,
            write_manifest=not args.no_manifest,
        )
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        _print_json(result.to_dict())
        return 0 if result.ok else 1
    _print_bind_summary(result, dry_run=args.dry_run)
    return 0 if result.ok else 1


def _print_bind_summary(result, dry_run: bool = False) -> None:
    print()
    print(f"files           : {result.chapter_count}")
    print(f"table of contents: {'included' if result.toc_included else 'not found'}")
    print(f"size            : {result.chars:,} characters · {result.lines:,} lines")
    print(f"output          : {result.output_path}")
    if result.manifest_path:
        print(f"manifest        : {result.manifest_path}")
    if result.trace_path:
        print(f"trace           : {result.trace_path}")
    if not result.ok:
        print("\nFAILED:", file=sys.stderr)
        for error in result.errors:
            print(f"  - {error}", file=sys.stderr)
        return
    if dry_run:
        print("\nDRY RUN — the book was assembled and verified, nothing was written.")
        return
    print("\nOK — the postprocessed chapters were concatenated into one Markdown book.")
