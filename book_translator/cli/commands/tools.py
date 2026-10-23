"""tools command: implementation and its human summary."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from ..common import (
    _print_json,
)


def add_parser(sub: Any) -> None:
    """Register this command's parser (the help text is snapshot-tested)."""
    p_tools = sub.add_parser("tools", help="list the tools, grouped by pipeline stage")
    p_tools.add_argument(
        "--stage",
        choices=["shared", "split", "preprocess", "matter", "postprocess"],
        default=None,
        help="show only one stage group (shared lists the generic tools)",
    )
    p_tools.add_argument("--json", action="store_true", help="machine-readable result")
    p_tools.add_argument("--quiet", "-q", action="store_true")
    p_tools.set_defaults(func=cmd_tools)


def cmd_tools(args: argparse.Namespace) -> int:
    from book_translator.stages.agent.tools import grouped_tools

    from ...toolkit import import_all

    import_all()

    wanted = getattr(args, "stage", None)
    groups = [
        (stage, list(items))
        for stage, items in grouped_tools()
        if not wanted or stage == wanted
    ]
    if args.json:
        _print_json(
            {
                stage: [{"name": name, "summary": summary} for name, summary in items]
                for stage, items in groups
            }
        )
        return 0
    printed = 0
    for stage, items in groups:
        if not args.quiet:
            print(f"{stage}:")
        for name, summary in items:
            print(name if args.quiet else f"    {name}")
            if not args.quiet:
                print(f"        {summary}")
            printed += 1
        if not args.quiet:
            print()
    if printed == 0:
        print(f"no tools registered for stage {wanted!r}", file=sys.stderr)
        return 1
    return 0
