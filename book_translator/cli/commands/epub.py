"""epub command: implementation and its human summary."""

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
    chapters_dir,
    has_postprocessed_chapter_files,
    postprocessed_chapters_dir,
    translated_epub_path,
)

from .. import parents
from ..common import (
    _print_json,
    _resolve_text_dir,
)


def add_parser(sub: Any) -> None:
    """Register this command's parser (the help text is snapshot-tested)."""
    p_epub = sub.add_parser(
        "epub",
        aliases=["ebook"],
        help="stage 11: package the postprocessed chapters as an EPUB 3 book "
        "(images embedded, reader navigation instead of the printed TOC)",
    )
    parents.add_book(p_epub, required=False)
    p_epub.add_argument(
        "--out",
        "-o",
        default=None,
        help="the chapter folder (default: the cleaned data/output/<book>/02-preprocessed "
        "when stage 2 wrote it, else data/output/<book>/01-chapters)",
    )
    p_epub.add_argument(
        "--postprocessed-in",
        default=None,
        help="stage 9's output folder (default: data/output/<book>/09-postprocessed-chapters)",
    )
    p_epub.add_argument(
        "--epub-out",
        default=None,
        help="where the EPUB goes (default: data/output/<book>/11-translated-book.epub)",
    )
    p_epub.add_argument(
        "--title",
        default=None,
        help="override the book title (default: the title stage 6b recorded)",
    )
    p_epub.add_argument(
        "--author",
        default=None,
        help="override the author (default: the author stage 6b recorded)",
    )
    p_epub.add_argument(
        "--language",
        default=None,
        help="override the book language, an ISO 639 code (default: the target "
        "language stage 6b recorded)",
    )
    p_epub.add_argument(
        "--identifier",
        default=None,
        help="override the publication identifier (default: a UUID derived from the "
        "book folder, so rebuilds keep the same book identity)",
    )
    p_epub.add_argument("--description", default=None, help="book description/blurb")
    p_epub.add_argument("--publisher", default=None, help="publisher for the metadata")
    p_epub.add_argument(
        "--cover",
        default=None,
        help="cover picture — a path, or a file name inside the book's "
        "00-markdown/images/ folder",
    )
    p_epub.add_argument(
        "--no-manifest",
        action="store_true",
        help="do not write _epub_manifest.json / _epub_trace.json",
    )
    p_epub.add_argument(
        "--dry-run",
        action="store_true",
        help="build and verify the package in memory but write no file",
    )
    p_epub.add_argument("--json", action="store_true", help="machine-readable result")
    p_epub.add_argument("--quiet", "-q", action="store_true")
    p_epub.set_defaults(func=cmd_epub)


def cmd_epub(args: argparse.Namespace) -> int:
    """Stage 11: package the postprocessed chapters as an EPUB 3 book."""
    from ...epub import run_epub

    if args.postprocessed_in:
        source_path = Path(args.postprocessed_in).expanduser()
        if source_path.is_file():
            source_path = source_path.parent
        if (source_path / POSTPROCESSED_CHAPTERS_DIRNAME).is_dir():
            source_path = source_path / POSTPROCESSED_CHAPTERS_DIRNAME
    else:
        chapters_path = chapters_dir(_resolve_text_dir(
            args.book,
            args.in_dir,
            args.out,
            getattr(args, "output_root", None),
            getattr(args, "text_in", None),
        ))
        source_path = postprocessed_chapters_dir(chapters_path)
    if not has_postprocessed_chapter_files(source_path):
        print(
            f"no postprocessed chapter files in {source_path.expanduser().resolve()} — "
            "run `postprocess` first",
            file=sys.stderr,
        )
        return 2
    output_path = (
        Path(args.epub_out).expanduser()
        if args.epub_out
        else translated_epub_path(source_path)
    )
    if not args.quiet:
        print(f"postprocessed-chapters={source_path.expanduser().resolve()}")
        print(f"epub={output_path.expanduser().resolve()}")
    try:
        result = run_epub(
            str(source_path),
            str(output_path),
            title=args.title,
            author=args.author,
            language=args.language,
            identifier=args.identifier,
            description=args.description,
            publisher=args.publisher,
            cover=args.cover,
            dry_run=args.dry_run,
            write_manifest=not args.no_manifest,
        )
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        _print_json(result.to_dict())
        return 0 if result.ok else 1
    _print_epub_summary(result, dry_run=args.dry_run)
    return 0 if result.ok else 1


def _print_epub_summary(result, dry_run: bool = False) -> None:
    print()
    print(f"title           : {result.title}")
    print(f"author          : {result.author or '(none recorded)'}")
    print(f"language        : {result.language}")
    print(f"identifier      : {result.identifier}")
    print(f"documents       : {result.document_count}")
    print(f"pictures        : {result.image_count}")
    if result.cover:
        print(f"cover           : {result.cover}")
    print(
        f"size            : {result.bytes:,} bytes · "
        f"{result.chars:,} characters of Markdown"
    )
    verification = result.verification or {}
    print(
        "verification    : "
        f"mimetype {'ok' if verification.get('mimetype_first') else 'FAILED'} · "
        f"container {'ok' if verification.get('container_ok') else 'FAILED'} · "
        f"package {'ok' if verification.get('opf_ok') else 'FAILED'} · "
        f"navigation {'ok' if verification.get('nav_ok') else 'FAILED'} · "
        f"{int(verification.get('xhtml_valid') or 0)}/"
        f"{int(verification.get('documents') or 0)} XHTML"
    )
    if result.toc_dropped:
        print("table of contents: dropped (the reader's navigation is the table of contents)")
    print(f"output          : {result.output_path}")
    if result.manifest_path:
        print(f"manifest        : {result.manifest_path}")
    if result.trace_path:
        print(f"trace           : {result.trace_path}")
    for warning in result.warnings:
        print(f"warning         : {warning}", file=sys.stderr)
    if not result.ok:
        print("\nFAILED:", file=sys.stderr)
        for error in result.errors:
            print(f"  - {error}", file=sys.stderr)
        return
    if dry_run:
        print("\nDRY RUN — the package was built and verified, nothing was written.")
        return
    print("\nOK — the postprocessed chapters were packaged as an EPUB 3 book.")
