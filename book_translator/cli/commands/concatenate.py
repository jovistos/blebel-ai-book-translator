"""concatenate command: implementation and its human summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from book_translator.core.constants import (
    CHAPTERS_DIRNAME,
)
from book_translator.core.errors import ToolError
from book_translator.documents.chapters import (
    chapters_dir,
    chunks_dir,
    has_chunk_files,
    has_translation_files,
    translated_chapters_dir,
    translations_dir,
)
from book_translator.stages.assembly import run_assembly

from .. import parents
from ..common import (
    _print_json,
    _resolve_text_dir,
)


def add_parser(sub: Any) -> None:
    """Register this command's parser (the help text is snapshot-tested)."""
    p_concat = sub.add_parser(
        "concatenate",
        aliases=["assemble"],
        help="stage 8: concatenate the translated chunks back into one file per "
        "chapter, in a separate 08-translated-chapters/ folder",
    )
    parents.add_book(p_concat, required=False)
    p_concat.add_argument(
        "--out",
        "-o",
        default=None,
        help="the chapter folder (default: the cleaned data/output/<book>/02-preprocessed "
        "when stage 2 wrote it, else data/output/<book>/01-chapters); "
        "used to locate the chunks, translations and book folder",
    )
    p_concat.add_argument(
        "--chunks-in",
        default=None,
        help="the chunks folder from stage 5 (default: data/output/<book>/05-chunks); "
        "a book folder or _chunk_manifest.json is also accepted",
    )
    p_concat.add_argument(
        "--translations-in",
        default=None,
        help="the translated chunks from stage 7 (default: a sibling of 05-chunks/)",
    )
    p_concat.add_argument(
        "--chapters-out",
        default=None,
        help="where the translated chapter files go "
        "(default: data/output/<book>/08-translated-chapters)",
    )
    p_concat.add_argument(
        "--first-chapters",
        "--chapters",
        dest="first_chapters",
        type=int,
        default=None,
        help="assemble only the first N body chapters (the whole translated book when omitted)",
    )
    p_concat.add_argument(
        "--skip-if-present",
        "--resume",
        dest="skip_if_present",
        action="store_true",
        help="reuse a translated chapter file whose bytes already match the current "
        "chunks' translations; only new chapters are written",
    )
    p_concat.add_argument(
        "--no-manifest",
        action="store_true",
        help="do not write _translated_chapters_manifest.json / _translated_chapters_trace.json",
    )
    p_concat.add_argument(
        "--dry-run",
        action="store_true",
        help="assemble every chapter in memory but write no chapter, manifest or trace",
    )
    p_concat.add_argument("--json", action="store_true", help="machine-readable result")
    p_concat.add_argument("--quiet", "-q", action="store_true")
    p_concat.set_defaults(func=cmd_concatenate)


def cmd_concatenate(args: argparse.Namespace) -> int:
    """Stage 8: concatenate the translated chunks into one file per chapter."""
    from book_translator.stages.assembly.runner import resolve_chunks_dir

    if args.chunks_in:
        chunks_path = resolve_chunks_dir(Path(args.chunks_in).expanduser())
        chapters_path = chunks_path.parent / CHAPTERS_DIRNAME
    else:
        chapters_path = chapters_dir(_resolve_text_dir(
            args.book,
            args.in_dir,
            args.out,
            getattr(args, "output_root", None),
            getattr(args, "text_in", None),
        ))
        chunks_path = chunks_dir(chapters_path)
    if not has_chunk_files(chunks_path):
        print(
            f"no chunk files in {chunks_path.expanduser().resolve()} — run `chunk` first",
            file=sys.stderr,
        )
        return 2
    translations_path = (
        Path(args.translations_in).expanduser()
        if args.translations_in
        else translations_dir(chunks_path)
    )
    if not has_translation_files(translations_path):
        print(
            f"no translated chunks in {translations_path.expanduser().resolve()} — "
            "run `translate` first",
            file=sys.stderr,
        )
        return 2
    out_dir = (
        Path(args.chapters_out).expanduser()
        if args.chapters_out
        else translated_chapters_dir(chunks_path)
    )

    if not args.quiet:
        print(f"chapters={chapters_path.expanduser().resolve()}")
        print(f"chunks={chunks_path.expanduser().resolve()}")
        print(f"translations={translations_path.expanduser().resolve()}")
        print(f"translated-chapters={out_dir.expanduser().resolve()}")
        if args.first_chapters:
            print(f"window      : first {args.first_chapters} chapter(s)")
        if args.skip_if_present:
            print("resume      : chapter files that already match will be reused")

    try:
        result = run_assembly(
            str(chunks_path),
            str(translations_path),
            str(out_dir),
            first_chapters=args.first_chapters,
            skip_if_present=args.skip_if_present,
            dry_run=args.dry_run,
            write_manifest=not args.no_manifest,
            progress=None if args.quiet else _assembly_progress,
        )
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        _print_json(result.to_dict())
        return 0 if result.ok else 1
    _print_assembly_summary(result, dry_run=args.dry_run)
    return 0 if result.ok else 1


def _assembly_progress(chapter: dict, done: int, total: int) -> None:
    """One line per assembled (or skipped) chapter."""
    if chapter.get("skipped"):
        flag = "skip"
        tail = f"  {chapter.get('note') or 'not translated'}"
    elif chapter.get("reused"):
        flag = "reuse"
        tail = ""
    else:
        flag = "ok   "
        tail = ""
    print(
        f"  [{done}/{total}] {flag} {chapter['filename']:<44} "
        f"{int(chapter.get('chunk_count') or 0):>3} chunk(s)  "
        f"{int(chapter.get('translation_chars') or 0):>7,} chars{tail}"
    )


def _print_assembly_summary(result, dry_run: bool = False) -> None:
    print()
    print(
        f"chapters        : {result.ok_count}/{result.chapter_count} written"
        + (f"  reused: {result.reused_count}" if result.reused_count else "")
        + (f"  incomplete: {result.incomplete_count}" if result.incomplete_count else "")
    )
    if result.limited:
        print(f"window          : first {result.first_chapters} chapter(s)")
    print(
        f"chunks          : {int(result.totals.get('translated_chunks') or 0):,} translated "
        f"of {int(result.totals.get('chunks') or 0):,} in the window"
    )
    print(
        f"output          : {result.output_dir}"
        + ("  (dry run — nothing written)" if dry_run else "")
    )
    if result.manifest_path:
        print(f"manifest        : {result.manifest_path}")
    if result.trace_path:
        print(f"trace           : {result.trace_path}")
    incomplete = [chapter for chapter in result.chapters if chapter.get("skipped")]
    for chapter in incomplete:
        print(
            f"  incomplete    : {chapter['filename']} — "
            f"{chapter.get('note') or 'not translated'}"
        )
    if not result.ok:
        print("\nFAILED chapters:", file=sys.stderr)
        for error in result.errors:
            print(f"  - {error}", file=sys.stderr)
        return
    if dry_run:
        print("\nDRY RUN — every chapter was assembled in memory, nothing was written.")
        return
    if result.chapter_count == 0:
        print("\nOK — no chapters in the requested window.")
        return
    print(
        "\nOK — translated chapters written to a separate folder; "
        "the source chapters were not touched."
    )
