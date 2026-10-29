"""chunk command: implementation and its human summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from book_translator.core.constants import (
    DEFAULT_CHUNK_TOKENIZER,
    DEFAULT_CHUNK_TOKENS,
)
from book_translator.core.errors import MissingDependencyError, ToolError
from book_translator.documents.chapters import (
    chunks_dir,
    has_chapter_files,
)
from book_translator.documents.window import (
    chapter_window_limit_for_path,
)

from .. import parents
from ..common import (
    _command_paths,
    _command_text_dir,
    _print_json,
    _window_label,
)


def add_parser(sub: Any) -> None:
    """Register this command's parser (the help text is snapshot-tested)."""
    p_chunk = sub.add_parser(
        "chunk",
        help="stage 5: cut each chapter into translation-sized chunks without "
        "breaking paragraphs or sentences; every chunk is an exact source slice "
        "and the manifest records its chapter and global order",
    )
    parents.add_book(p_chunk, required=False)
    p_chunk.add_argument(
        "--out",
        "-o",
        default=None,
        help="the chapter folder (default: the cleaned data/output/<book>/03-preprocessed "
        "when stage 3 wrote it, else data/output/<book>/02-chapters); pass "
        "02-chapters to chunk the raw text instead",
    )
    p_chunk.add_argument(
        "--chunks-out",
        default=None,
        help="where to write the chunk files "
        "(default: data/output/<book>/05-chunks, a sibling of the input folder)",
    )
    p_chunk.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_TOKENS,
        help=f"maximum chunk length in tokens (default: {DEFAULT_CHUNK_TOKENS})",
    )
    p_chunk.add_argument(
        "--chunk-tokenizer",
        default=DEFAULT_CHUNK_TOKENIZER,
        help="tokenizer used to measure a chunk: a tiktoken encoding "
        f"(default: {DEFAULT_CHUNK_TOKENIZER}), 'word', 'character', 'byte', or a "
        "Hugging Face tokenizer name; falls back to a local approximation offline",
    )
    p_chunk.add_argument(
        "--skip-if-present",
        "--resume",
        dest="skip_if_present",
        action="store_true",
        help="reuse a chapter whose recorded chunk files still concatenate to its "
        "current text; a fully chunked book needs no tokenizer at all",
    )
    p_chunk.add_argument(
        "--first-chapters",
        "--chapters",
        dest="first_chapters",
        type=int,
        default=None,
        help="chunk only the first N body chapters (the whole book when omitted)",
    )
    p_chunk.add_argument(
        "--limit",
        type=int,
        default=None,
        help="chunk only the first N body chapters (kept as a unit-limit alias)",
    )
    p_chunk.add_argument(
        "--no-manifest",
        action="store_true",
        help="do not write _chunk_manifest.json / _chunk_trace.json",
    )
    p_chunk.add_argument(
        "--dry-run",
        action="store_true",
        help="compute and verify every cut but write no chunk file, manifest or trace",
    )
    p_chunk.add_argument("--json", action="store_true", help="machine-readable result")
    p_chunk.add_argument("--quiet", "-q", action="store_true")
    p_chunk.add_argument(
        "--text-in",
        default=None,
        help="chapter text folder to read (default: 03-preprocessed/ when it covers "
        "the window, else 02-chapters/); overrides --out",
    )
    p_chunk.set_defaults(func=cmd_chunk)


def cmd_chunk(args: argparse.Namespace) -> int:
    """Stage 5: cut each chapter into chunks that concatenate back exactly."""
    from book_translator.stages.chunking import run_chunking

    paths = _command_paths(args)
    limit = chapter_window_limit_for_path(
        paths.chapters_dir, getattr(args, "first_chapters", None), args.limit
    )
    chapters_path = _command_text_dir(args, paths, limit)
    if not has_chapter_files(chapters_path):
        print(
            f"no chapter files in {chapters_path.expanduser().resolve()} — run `split` first",
            file=sys.stderr,
        )
        return 2
    if int(args.chunk_size) <= 0:
        print("error: --chunk-size must be greater than 0 tokens", file=sys.stderr)
        return 2
    out_dir = (
        Path(args.chunks_out).expanduser() if args.chunks_out else chunks_dir(chapters_path)
    )

    if not args.quiet:
        print(f"chapters={chapters_path.expanduser().resolve()}")
        print(f"chunks={out_dir.expanduser().resolve()}")
        print(
            f"chunk size={args.chunk_size} tokens  "
            f"tokenizer={args.chunk_tokenizer}  overlap=0  method=recursive"
        )
        label = _window_label(args, limit)
        if label is not None:
            print(f"window      : {label}")

    try:
        result = run_chunking(
            str(chapters_path),
            str(out_dir),
            chunk_size=args.chunk_size,
            tokenizer=args.chunk_tokenizer,
            skip_if_present=args.skip_if_present,
            dry_run=args.dry_run,
            limit=limit,
            write_manifest=not args.no_manifest and not args.dry_run,
            progress=None if args.quiet else _chunk_progress,
        )
    except MissingDependencyError as exc:
        print(f"missing dependency: {exc}", file=sys.stderr)
        return 2
    except (ToolError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        _print_json(result.to_dict())
        return 0 if result.ok else 1
    _print_chunk_summary(result, dry_run=args.dry_run)
    return 0 if result.ok else 1


def _chunk_progress(chapter: dict, done: int, total: int) -> None:
    """One line per chunked (or reused) chapter, printed as it finishes."""
    flag = "reuse" if chapter.get("skipped") else "ok   "
    print(
        f"  [{done}/{total}] {flag} {chapter['filename']:<44} "
        f"{chapter.get('chunk_count', 0):>4} chunk(s)  "
        f"{int(chapter.get('token_count') or 0):>8,} tok"
    )


def _print_chunk_summary(result, dry_run: bool = False) -> None:
    settings = result.settings or {}
    tokenizer = settings.get("tokenizer") or {}
    verification = result.verification or {}
    print()
    print(
        f"chapters        : {result.ok_count}/{result.chapter_count} ok"
        + (f"  reused: {result.skipped_count}" if result.skipped_count else "")
    )
    print(
        f"chunks          : {result.chunk_count:,}  "
        f"(max {int(settings.get('chunk_size') or 0):,} tokens each, overlap 0)"
    )
    print(
        f"tokenizer       : {tokenizer.get('used') or tokenizer.get('requested') or '—'}"
        + (
            f"  (requested {tokenizer.get('requested')}; local approximation is in use)"
            if tokenizer.get("fallback")
            else ""
        )
    )
    if tokenizer.get("note"):
        print(f"  note          : {tokenizer['note']}")
    totals = result.totals or {}
    print(
        f"tokens          : {int(totals.get('chapter_tokens') or 0):,} in chapters "
        f"-> {int(totals.get('chunk_tokens') or 0):,} across chunks"
    )
    kinds = verification.get("boundary_counts") or {}
    shown = ", ".join(f"{kind}={count:,}" for kind, count in kinds.items() if count)
    if shown:
        print(f"boundaries      : {shown}")
    print(
        f"reconstruction  : chapters={verification.get('chapters_reconstruct_byte_for_byte')} "
        f"book={verification.get('book_reconstructs_byte_for_byte')} "
        f"word-breaks={verification.get('word_breaking_chunks', 0)}"
    )
    if verification.get("chunk_size_violations"):
        print(f"  warning       : {verification['chunk_size_violations']} chunk(s) exceed the limit")
    print(f"output dir      : {result.output_dir}")
    if result.manifest_path:
        print(f"manifest        : {result.manifest_path}")
    if result.trace_path:
        print(f"trace           : {result.trace_path}")
    if not result.ok:
        print("\nFAILED chapters:", file=sys.stderr)
        for error in result.errors:
            print(f"  - {error}", file=sys.stderr)
        return
    if dry_run:
        print("\nDRY RUN — every cut was verified, nothing was written.")
        return
    if result.skipped:
        print("\nOK — nothing was written: the existing chunks were reused.")
        return
    print(
        "\nOK — chunks written; concatenate them in filename order to rebuild the book, "
        "or per chapter via reconstruct_chapter()."
    )
