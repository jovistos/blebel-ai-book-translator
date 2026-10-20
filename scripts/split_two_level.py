#!/usr/bin/env python
"""Two-level split: book sections, and the chapter markers inside each.

Some conversions keep the part/book markers but mangle or drop most chapter
headings. A flat split then has only bad options: one giant file per book, or a
guess. This script builds a two-level map instead, deterministically:

    data/output/<book>/01-chapters/
        00-front-matter.md
        01-foreword-opening.md
        02-prologue-chapter-4.md
        03-prologue-chapter-5.md
        04-book-one-opening.md
        05-book-one-chapter-4.md
        06-book-one-chapter-5.md
        ...

Rules:

* a **book** starts at a header matching ``--book-pattern`` (the match is
  anchored, so a mangled cover-page line such as ``## PROLOGUE -LONDON`` is not
  mistaken for the real ``## PROLOGUE``);
* a **chapter** starts at a header inside that book matching
  ``--chapter-pattern``;
* a missing chapter number simply is not there: chapter 4 runs until chapter 6,
  or to the end of its book, exactly as the file has it — no renumbering, no
  invented titles, and every filename is built from the text as found;
* the lines between a book marker and that book's first chapter marker become
  ``<book>-opening.md``; a book with no chapter marker becomes one file.

The files are written by the pipeline's own slicer, so they are verbatim,
byte-for-byte reversible, and land next to a standard ``_manifest.json`` that
``python -m book_splitter scan`` can read.

    python scripts/split_two_level.py BOOK.md --dry-run
    python scripts/split_two_level.py BOOK.md
    python scripts/split_two_level.py BOOK.md --chapter-pattern '^\\d{1,3}$|^[IVXLC]{1,4}$'
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from book_splitter.agent.tools import scan_markdown_headers  # noqa: E402
from book_splitter.analysis.headers import assess_headers  # noqa: E402
from book_splitter.analysis.renumber import (  # noqa: E402
    ambiguous_markers,
    resolve_marker_names,
)
from book_splitter.core.errors import ToolError  # noqa: E402
from book_splitter.documents.io import load_lines  # noqa: E402
from book_splitter.documents.markdown import header_of  # noqa: E402
from book_splitter.documents.text import slugify  # noqa: E402
from book_splitter.paths import (  # noqa: E402
    default_input_dir,
    resolve_book_path,
    resolve_output_dir,
)
from book_splitter.slicing.slice import preview_split, slice_book  # noqa: E402
from book_splitter.slicing.verify import verify_output_dir  # noqa: E402

#: Book / part markers. Anchored on purpose: `## PROLOGUE -LONDON` (a mangled
#: cover-page line) must not shadow the real `## PROLOGUE`.
DEFAULT_BOOK_PATTERN = (
    r"^(?:FOREWORD|PREFACE|PROLOGUE|INTRODUCTION|EPILOGUE|AFTERWORD|"
    r"BOOK\s+\S+|PART\s+\S+)$"
)
#: Chapter markers: a bare number, which is how these conversions leave them.
DEFAULT_CHAPTER_PATTERN = r"^\d{1,3}$"


def parse_alias(raw: str) -> tuple[str, str]:
    """``--chapter-alias "lO=10"`` -> ``("lO", "10")``."""
    text, _, number = str(raw).partition("=")
    if not text.strip() or not number.strip():
        raise argparse.ArgumentTypeError(f"expected TEXT=NUMBER, got {raw!r}")
    return text.strip(), number.strip()


def book_sequences(
    lines: list[str], book_pattern: str, chapter_pattern: str
) -> dict[str, list[tuple[int, str]]]:
    """``{book title: [(line, marker), ...]}`` in file order."""
    book_re = re.compile(book_pattern, re.IGNORECASE)
    chapter_re = re.compile(chapter_pattern)
    headers = [
        (i + 1, header_of(line)[1].strip()) for i, line in enumerate(lines) if header_of(line)
    ]
    books = [(line, text) for line, text in headers if book_re.match(text)]
    if not books:
        raise ToolError(f"no book marker matched {book_pattern!r}")
    sequences: dict[str, list[tuple[int, str]]] = {}
    for index, (line, title) in enumerate(books):
        end = books[index + 1][0] - 1 if index + 1 < len(books) else len(lines)
        sequences[title] = [
            (ln, text) for ln, text in headers if line < ln <= end and chapter_re.match(text)
        ]
    return sequences


def build_plan(
    lines: list[str],
    book_pattern: str,
    chapter_pattern: str,
    aliases: dict[str, str] | None = None,
    book_aliases: dict[tuple[str, str], str] | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """The two-level split map, plus the headers that were skipped.

    Returns ``(mapping, skipped)`` where ``mapping`` is ready for
    :func:`book_splitter.slicing.slice.slice_book` and ``skipped`` lists
    ``(text, reason)`` for header-like lines inside a book that matched neither
    pattern — the mangled ones a human may want to look at.

    ``aliases`` maps a marker *as written* to the number it means
    (``{"lO": "10", "'3": "13"}``). It changes only the filename, and only for
    keys the caller supplied, so nothing is guessed by the script.
    """
    aliases = {key.strip(): value.strip() for key, value in (aliases or {}).items()}
    book_aliases = book_aliases or {}
    book_re = re.compile(book_pattern, re.IGNORECASE)
    headers = [
        (i + 1, header_of(line)[1].strip()) for i, line in enumerate(lines) if header_of(line)
    ]

    sequences = book_sequences(lines, book_pattern, chapter_pattern)
    books = [(line, text) for line, text in headers if book_re.match(text)]

    order: list[dict[str, Any]] = []
    for line, title in books:
        name = slugify(title, max_words=4)
        chapters = sequences[title]
        if not chapters:
            order.append({"line_number": line, "filename": f"{name}.md", "kind": "book"})
            continue
        order.append({"line_number": line, "filename": f"{name}-opening.md", "kind": "opening"})
        for ln, text in chapters:
            label = book_aliases.get((title, text)) or aliases.get(text) or text
            order.append(
                {"line_number": ln, "filename": f"{name}-chapter-{label}.md", "kind": "chapter"}
            )

    numbered = {line for item in order for line in [item["line_number"]]}
    inside = {
        (ln, text)
        for index, (line, title) in enumerate(books)
        for ln, text in headers
        if line < ln <= (books[index + 1][0] - 1 if index + 1 < len(books) else len(lines))
    }
    skipped = [
        (text, "header inside a book that is neither a book nor a chapter marker")
        for ln, text in sorted(inside)
        if ln not in numbered
    ]

    numbered_plan = [
        {**item, "filename": f"{position:02d}-{item['filename']}"}
        for position, item in enumerate(order, start=1)
    ]
    return numbered_plan, skipped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="split_two_level.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "book",
        nargs="?",
        default=None,
        help="source document — filename or path (default: the only book in data/input/)",
    )
    parser.add_argument(
        "--in-dir", default=None, help=f"folder holding the source (default: {default_input_dir()})"
    )
    parser.add_argument(
        "--out", "-o", default=None, help="output folder (default: <book>/01-chapters)"
    )
    parser.add_argument(
        "--book-pattern", default=DEFAULT_BOOK_PATTERN, help="regex for book markers"
    )
    parser.add_argument(
        "--chapter-pattern", default=DEFAULT_CHAPTER_PATTERN, help="regex for chapter markers"
    )
    parser.add_argument(
        "--chapter-alias",
        action="append",
        type=parse_alias,
        default=None,
        metavar="TEXT=NUMBER",
        help='name a mangled marker by the number it means, e.g. --chapter-alias "lO=10"',
    )
    parser.add_argument(
        "--resolve-names",
        action="store_true",
        help="let the model read mangled markers (lO -> 10, '3 -> 13) and use the corrected numbers",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="delete previously written NN-*.md files in the output folder first",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        source = resolve_book_path(args.book, input_dir=args.in_dir)
        out_dir = resolve_output_dir(source, args.out)
    except (FileNotFoundError, ToolError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    lines = load_lines(source)
    pages = scan_markdown_headers(str(source))
    assessment = assess_headers(pages["headers"], len(lines))
    aliases = dict(args.chapter_alias or [])
    book_aliases: dict[tuple[str, str], str] = {}
    try:
        sequences = book_sequences(lines, args.book_pattern, args.chapter_pattern)
        if args.resolve_names:
            ambiguous = sum(
                len(ambiguous_markers([m for _, m in marks])) for marks in sequences.values()
            )
            print(f"ambiguous       : {ambiguous} marker(s) the model must read")
            book_aliases, problems = resolve_marker_names(
                {book: [marker for _, marker in marks] for book, marks in sequences.items()}
            )
            for (book, marker), number in sorted(book_aliases.items()):
                print(f"  agent reads   : {book}: {marker!r} -> chapter {number}")
            if problems:
                print("  rejected      : " + "; ".join(problems))
            elif not book_aliases:
                print("  agent reads   : nothing to correct")
        mapping, skipped = build_plan(
            lines, args.book_pattern, args.chapter_pattern, aliases, book_aliases
        )
    except (ToolError, re.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"source          : {source}")
    print(f"headers         : {len(pages['headers'])} ({assessment['verdict']})")
    print(f"book markers    : {sum(1 for m in mapping if m['kind'] in {'book', 'opening'})}")
    print(f"chapter markers : {sum(1 for m in mapping if m['kind'] == 'chapter')}")
    print(f"output          : {out_dir}")

    if args.dry_run:
        preview = preview_split(str(source), mapping, str(out_dir))
        print(f"\nDRY RUN — {preview['file_count']} files, {preview['total_lines']} lines\n")
        for item in preview["files"]:
            print(
                f"  {item['filename']:<46} lines {item['start_line']:>5}-{item['end_line']:<5} "
                f"{item['lines']:>5} L  {item['first_line'][:40]!r}"
            )
    else:
        if args.clean:
            removed = 0
            for stale in sorted(out_dir.glob("[0-9]*.md")):
                stale.unlink()
                removed += 1
            if removed:
                print(f"cleaned         : {removed} previously written file(s)")
        manifest = slice_book(str(source), mapping, str(out_dir))
        print(
            f"\nwritten         : {manifest['file_count']} files, {manifest['total_lines']} lines\n"
        )
        for item in manifest["files"]:
            print(
                f"  {item['filename']:<46} lines {item['start_line']:>5}-{item['end_line']:<5} "
                f"{item['lines']:>5} L  {item['first_line'][:40]!r}"
            )
        report = verify_output_dir(out_dir)
        print()
        print(f"byte-for-byte   : {report.get('reassembles_byte_for_byte')}")
        print(f"lines           : {report.get('output_lines')}/{report.get('source_lines')}")
        print(f"files verified  : {report.get('files_checked')}")
        if not report.get("ok"):
            print(f"FAILED: {report.get('error') or report}", file=sys.stderr)
            return 1
        print(f"manifest        : {out_dir / '_manifest.json'}")

    if skipped:
        print(f"\nnot treated as markers ({len(skipped)}) — mangled headers to review by hand:")
        for text, _ in skipped:
            print(f"  {text[:70]!r}")
    if args.dry_run:
        print("\nDRY RUN — nothing was written.")
    else:
        print("\nOK — every source line is accounted for, verbatim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
