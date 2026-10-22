"""openings command: implementation and its human summary."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from .. import parents
from ..common import (
    _print_json,
    _print_openings_line,
    _resolve_openings_dir,
)


def add_parser(sub: Any) -> None:
    """Register this command's parser (the help text is snapshot-tested)."""
    p_open = sub.add_parser(
        "openings",
        help="final phase of stages 2 and 9: audit every chapter's first lines and "
        "remove the safe leftovers (no LLM)",
    )
    parents.add_book(p_open, required=False)
    p_open.add_argument(
        "--chapters-in",
        default=None,
        help="the chapter folder to audit (default: the book's most advanced stage "
        "folder: 09-postprocessed-chapters/ → 08-translated-chapters/ → "
        "02-preprocessed/ → 01-chapters/)",
    )
    p_open.add_argument(
        "--keep-blank-slot",
        action="store_true",
        help="stage-2 semantics: a removed line leaves an empty line, so line numbers "
        "keep pointing at the same text (default: the line is removed and the blank "
        "around it collapsed, which is stage 9's behaviour)",
    )
    p_open.add_argument(
        "--dry-run",
        action="store_true",
        help="report every opening finding and write nothing",
    )
    p_open.add_argument("--json", action="store_true", help="machine-readable result")
    p_open.add_argument("--quiet", "-q", action="store_true")
    p_open.set_defaults(func=cmd_openings)


def cmd_openings(args: argparse.Namespace) -> int:
    """Final phase of stages 2 and 9: audit every chapter opening (no LLM)."""
    from ...openings import audit_openings

    folder = _resolve_openings_dir(args)
    if not folder.is_dir():
        print(f"error: chapter folder not found: {folder}", file=sys.stderr)
        return 2
    result = audit_openings(
        folder,
        dry_run=bool(args.dry_run),
        keep_blank_slot=bool(args.keep_blank_slot),
    )
    if args.json:
        _print_json(result.to_dict())
        return 0 if result.ok else 1
    if not args.quiet:
        print(f"chapters        : {folder.expanduser().resolve()}")
        print(
            "policy          : "
            + (
                "keep the line count (stage-2 blank slot)"
                if args.keep_blank_slot
                else "remove the line and collapse the blank (stage-9 output)"
            )
        )
    print()
    _print_openings_line(result.summary(), detail=not args.quiet)
    if not args.quiet:
        print(f"output dir      : {result.folder}")
    if not result.ok:
        print("\nFAILED openings:", file=sys.stderr)
        for error in result.errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    if result.errors:
        for error in result.errors:
            print(f"  warning         : {error}", file=sys.stderr)
    print(
        "\nDRY RUN — every opening audited, nothing was written."
        if args.dry_run
        else "\nOK — every chapter opening audited; only provable leftovers were removed."
    )
    return 0
