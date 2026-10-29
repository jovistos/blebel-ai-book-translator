"""convert command: implementation and its human summary."""

from __future__ import annotations

import argparse
from typing import Any

from book_translator.stages.conversion import run_conversion

from .. import parents
from ..common import (
    _print_json,
    _resolve_book,
)


def add_parser(sub: Any) -> None:
    """Register this command's parser (the help text is snapshot-tested)."""
    p_convert = sub.add_parser(
        "convert",
        help="stage 1: convert an EPUB, a text-based PDF or a TXT to Markdown "
        "(docling; image-based PDFs are rejected instead of OCR'd)",
    )
    parents.add_book(p_convert, required=False)
    p_convert.add_argument(
        "--out",
        "-o",
        default=None,
        help="folder for the converted Markdown (default: data/output/<book>/01-markdown)",
    )
    p_convert.add_argument(
        "--force",
        action="store_true",
        help="convert again even when the existing Markdown still matches the source hash",
    )
    p_convert.add_argument(
        "--no-images",
        dest="images",
        action="store_false",
        help="do not extract pictures from the EPUB/PDF; the Markdown stays text-only "
        "(images are extracted by default)",
    )
    p_convert.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the source (including a PDF's text layer) and report what would "
        "happen; run no docling conversion and write nothing",
    )
    p_convert.add_argument("--json", action="store_true", help="machine-readable result")
    p_convert.add_argument("--quiet", "-q", action="store_true")
    p_convert.set_defaults(func=cmd_convert, images=True)


def cmd_convert(args: argparse.Namespace) -> int:
    """Stage 1: EPUB/PDF/TXT -> Markdown under ``01-markdown/``."""
    source = _resolve_book(args)
    result = run_conversion(
        source,
        args.out,
        output_root=getattr(args, "output_root", None),
        dry_run=args.dry_run,
        force=args.force,
        images=getattr(args, "images", True),
    )

    if args.json:
        _print_json(result.to_dict())
        return 0

    if not args.quiet:
        for line in result.log:
            print(f"  · {line}")
        print()
        print(f"source          : {result.source}")
        print(f"kind            : {result.source_kind}")
        if result.pages is not None:
            print(
                f"pdf pages       : {result.pages} "
                f"({result.text_pages}/{result.sampled_pages} sampled page(s) with text)"
            )
        print(f"method          : {result.method}")
        print(f"markdown        : {result.markdown_path}")
        if result.images or result.image_refs:
            print(f"images          : {result.images} file(s), {result.image_refs} reference(s)")
            if result.images_dir:
                print(f"images folder   : {result.images_dir}")
        if result.reused:
            print("reused          : existing Markdown still matches the source hash")
        print(
            "written         : no (dry run — nothing written)"
            if args.dry_run
            else f"written         : {'yes' if result.written else 'no'}"
        )
    return 0
