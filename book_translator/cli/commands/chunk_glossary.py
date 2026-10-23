"""chunk-glossary command: implementation and its human summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from book_translator.core.constants import (
    CHUNK_GLOSSARY_SCORERS,
    DEFAULT_CHUNK_GLOSSARY_CANDIDATES,
    DEFAULT_CHUNK_GLOSSARY_MIN_TERM_CHARS,
    DEFAULT_CHUNK_GLOSSARY_SCORER,
    DEFAULT_CHUNK_GLOSSARY_THRESHOLD,
    DEFAULT_GPU_INDEX,
    DEFAULT_KEEP_ENTITY_TYPES,
)
from book_translator.core.errors import MissingDependencyError, ToolError
from book_translator.core.report import human_seconds
from book_translator.documents.chapters import (
    chapters_dir,
    chunk_glossary_dir,
    chunks_dir,
    glossary_dir,
    has_chunk_files,
)
from book_translator.documents.window import (
    chapter_files_for_window,
)

from .. import parents
from ..common import (
    _parse_entity_types,
    _print_json,
    _resolve_text_dir,
)


def add_parser(sub: Any) -> None:
    """Register this command's parser (the help text is snapshot-tested)."""
    p_cg = sub.add_parser(
        "chunk-glossary",
        help="stage 6: run NER over every chunk first, then fuzzy-match each "
        "extracted term against glossary.json (rapidfuzz) and write the relevant "
        "terms per chunk, under the chunk's own name",
    )
    parents.add_book(p_cg, required=False)
    p_cg.add_argument(
        "--out",
        "-o",
        default=None,
        help="the chapter folder (default: the cleaned data/output/<book>/02-preprocessed "
        "when stage 2 wrote it, else data/output/<book>/01-chapters); "
        "the chunks and glossary folders are derived from the book folder",
    )
    p_cg.add_argument(
        "--chunks-in",
        default=None,
        help="the chunks folder from stage 5 (default: data/output/<book>/05-chunks); "
        "a book folder or the _chunk_manifest.json is also accepted",
    )
    p_cg.add_argument(
        "--glossary-in",
        default=None,
        help="the glossary folder or glossary.json from stage 4 "
        "(default: data/output/<book>/04-glossary/glossary.json)",
    )
    p_cg.add_argument(
        "--chunk-glossary-out",
        default=None,
        help="where the per-chunk JSON files land "
        "(default: data/output/<book>/06-chunk-glossary)",
    )
    p_cg.add_argument(
        "--language",
        default="auto",
        help="'auto' samples a few chunks and detects the language; otherwise an "
        "ISO code: sr, hr, sl, bg (CLASSLA) or en (stanza)",
    )
    p_cg.add_argument(
        "--gpu",
        type=int,
        default=None,
        help=f"physical GPU to pin the tagger to (default: GPU {DEFAULT_GPU_INDEX}, or "
        "$CUDA_VISIBLE_DEVICES when that is already set)",
    )
    p_cg.add_argument("--cpu", action="store_true", help="run the tagger on the CPU")
    p_cg.add_argument(
        "--keep-types",
        default=",".join(DEFAULT_KEEP_ENTITY_TYPES),
        help="entity types that count as a term "
        f"(default: {','.join(DEFAULT_KEEP_ENTITY_TYPES)})",
    )
    p_cg.add_argument(
        "--min-term-chars",
        type=int,
        default=DEFAULT_CHUNK_GLOSSARY_MIN_TERM_CHARS,
        help="drop extracted terms shorter than this "
        f"(default: {DEFAULT_CHUNK_GLOSSARY_MIN_TERM_CHARS})",
    )
    p_cg.add_argument(
        "--drop-temporal-only",
        action="store_true",
        help="do not extract dates and times (they cannot be in the glossary)",
    )
    p_cg.add_argument(
        "--threshold",
        "--match-threshold",
        dest="threshold",
        type=float,
        default=DEFAULT_CHUNK_GLOSSARY_THRESHOLD,
        help="minimum rapidfuzz score, as a percentage "
        f"(default: {DEFAULT_CHUNK_GLOSSARY_THRESHOLD}); lower is more permissive",
    )
    p_cg.add_argument(
        "--scorer",
        choices=CHUNK_GLOSSARY_SCORERS,
        default=DEFAULT_CHUNK_GLOSSARY_SCORER,
        help="rapidfuzz scorer: ratio is strict; token_sort_ratio ignores word "
        "order; wratio tolerates partial names "
        f"(default: {DEFAULT_CHUNK_GLOSSARY_SCORER})",
    )
    p_cg.add_argument(
        "--candidates",
        type=int,
        default=DEFAULT_CHUNK_GLOSSARY_CANDIDATES,
        help="glossary candidates retrieved per term; the best is the match and the "
        f"rest are recorded as alternatives (default: {DEFAULT_CHUNK_GLOSSARY_CANDIDATES})",
    )
    p_cg.add_argument(
        "--first-chapters",
        "--chapters",
        dest="first_chapters",
        type=int,
        default=None,
        help="tag and match only the chunks of the first N body chapters (the whole "
        "book when omitted)",
    )
    p_cg.add_argument(
        "--limit",
        type=int,
        default=None,
        help="tag and match only the first N chunks (a quick look before a full run)",
    )
    p_cg.add_argument(
        "--skip-if-present",
        "--resume",
        dest="skip_if_present",
        action="store_true",
        help="reuse a per-chunk glossary file whose chunk hash, threshold, scorer and "
        "glossary still match; a fully covered window needs no tagger",
    )
    p_cg.add_argument(
        "--no-manifest",
        action="store_true",
        help="do not write _chunk_glossary_manifest.json / _chunk_glossary_trace.json",
    )
    p_cg.add_argument(
        "--dry-run",
        action="store_true",
        help="run both phases but write no per-chunk file, manifest or trace",
    )
    p_cg.add_argument("--json", action="store_true", help="machine-readable result")
    p_cg.add_argument("--quiet", "-q", action="store_true")
    p_cg.add_argument(
        "--text-in",
        default=None,
        help="chapter text folder to read (default: 02-preprocessed/ when it covers "
        "the window, else 01-chapters/); overrides --out",
    )
    p_cg.set_defaults(func=cmd_chunk_glossary)


def cmd_chunk_glossary(args: argparse.Namespace) -> int:
    """Stage 6: NER over every chunk, then fuzzy-match each term to the glossary."""
    from ...chunk_glossary import run_chunk_glossary
    from ...chunk_glossary.runner import resolve_chunks_dir

    chapters_path = chapters_dir(_resolve_text_dir(
            args.book,
            args.in_dir,
            args.out,
            getattr(args, "output_root", None),
            getattr(args, "text_in", None),
        ))
    chunks_path = (
        resolve_chunks_dir(Path(args.chunks_in).expanduser())
        if args.chunks_in
        else chunks_dir(chapters_path)
    )
    if not has_chunk_files(chunks_path):
        print(
            f"no chunk files in {chunks_path.expanduser().resolve()} — run `chunk` first",
            file=sys.stderr,
        )
        return 2
    glossary_path = (
        Path(args.glossary_in).expanduser()
        if args.glossary_in
        else glossary_dir(chapters_path)
    )
    out_dir = (
        Path(args.chunk_glossary_out).expanduser()
        if args.chunk_glossary_out
        else chunk_glossary_dir(chunks_path)
    )
    keep_types = _parse_entity_types(args.keep_types)

    if not 0.0 <= float(args.threshold) <= 100.0:
        print(
            "error: --match-threshold is a percentage between 0 and 100",
            file=sys.stderr,
        )
        return 2
    if int(args.candidates) <= 0:
        print("error: --match-candidates must be greater than 0", file=sys.stderr)
        return 2
    if int(args.min_term_chars) < 0:
        print("error: --min-term-chars cannot be negative", file=sys.stderr)
        return 2

    try:
        chapter_files = chapter_files_for_window(
            chunks_path, getattr(args, "first_chapters", None)
        )
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    limit = args.limit

    if not args.quiet:
        print(f"chapters={chapters_path.expanduser().resolve()}")
        print(f"chunks={chunks_path.expanduser().resolve()}")
        print(f"glossary={glossary_path.expanduser().resolve()}")
        print(f"chunk-glossary={out_dir.expanduser().resolve()}")
        print(
            f"language={args.language}  threshold={args.threshold}  "
            f"scorer={args.scorer}  candidates={args.candidates}  "
            f"keep_types={','.join(keep_types)}  min_term_chars={args.min_term_chars}"
        )
        print("phases      : ner over every chunk, then fuzzy retrieval for every term")
        if chapter_files is not None:
            print(
                f"window      : chunks of the first {args.first_chapters} body chapter(s)"
            )
        if limit is not None:
            print(f"limit       : first {limit} chunk(s) in the window")
        if not args.drop_temporal_only:
            print("              dates/times are kept; pass --drop-temporal-only to skip them")

    try:
        result = run_chunk_glossary(
            str(chunks_path),
            str(out_dir),
            glossary_path=str(glossary_path),
            language=args.language,
            gpu=args.gpu,
            use_gpu=not args.cpu,
            keep_types=keep_types,
            min_term_chars=args.min_term_chars,
            drop_temporal_only=args.drop_temporal_only,
            threshold=args.threshold,
            scorer=args.scorer,
            candidates=args.candidates,
            dry_run=args.dry_run,
            limit=limit,
            chapter_files=chapter_files,
            skip_if_present=args.skip_if_present,
            write_manifest=not args.no_manifest and not args.dry_run,
            progress=None if args.quiet else _chunk_glossary_progress,
        )
    except MissingDependencyError as exc:
        print(f"missing dependency: {exc}", file=sys.stderr)
        return 2
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        _print_json(result.to_dict())
        return 0 if result.ok else 1
    _print_chunk_glossary_summary(result, dry_run=args.dry_run)
    return 0 if result.ok else 1


def _chunk_glossary_progress(chunk: dict, done: int, total: int) -> None:
    """One line per chunk, printed as the retrieval phase finishes it."""
    flag = "ok   " if chunk.get("ok") else "FAIL "
    print(
        f"  [{done}/{total}] {flag} {chunk['filename']:<44} "
        f"entities={chunk.get('entity_count', 0):>3}  "
        f"terms={chunk.get('term_count', 0):>3}  "
        f"matched={chunk.get('matched_count', 0):>3}  "
        f"unmatched={chunk.get('unmatched_count', 0):>3}"
    )


def _print_chunk_glossary_summary(result, dry_run: bool = False) -> None:
    language = result.language or {}
    settings = result.settings or {}
    totals = result.totals or {}
    timing = result.timing or {}
    print()
    print(
        f"language        : {language.get('language') or '—'} "
        f"({language.get('method')}, confidence {language.get('confidence', 0):.2f})"
        f"  backend: {result.backend or '—'}"
        + (f" on {result.device}" if result.device else "")
    )
    if language.get("note"):
        print(f"  note          : {language['note']}")
    print(f"chunks          : {result.ok_count}/{result.chunk_count} ok")
    print(
        f"entities        : {int(totals.get('entity_mentions') or 0):,} mention(s) "
        f"-> {int(totals.get('terms') or 0):,} distinct term(s)"
    )
    print(f"glossary        : {result.entry_count:,} entr"
          f"{'y' if result.entry_count == 1 else 'ies'} loaded")
    print(
        f"matched         : {int(totals.get('matched_terms') or 0):,} term(s) in "
        f"{int(totals.get('matched_chunks') or 0):,} chunk(s)  "
        f"(threshold {settings.get('threshold')}%, scorer {settings.get('scorer')}, "
        f"unique dictionary entries {int(totals.get('unique_matched_glossary_terms') or 0):,})"
    )
    print(f"unmatched       : {int(totals.get('unmatched_terms') or 0):,} term(s)")
    phases = result.phases or []
    for phase in phases:
        print(
            f"{phase.get('name', 'phase'):<16}: "
            f"{human_seconds(float(phase.get('seconds') or 0.0))} "
            f"for {int(phase.get('chunks') or 0):,} chunk(s)"
        )
    if timing:
        print(f"total time      : {human_seconds(float(timing.get('run_seconds') or 0.0))}")
    print(f"output dir      : {result.output_dir}")
    if result.manifest_path:
        print(f"manifest        : {result.manifest_path}")
    if result.trace_path:
        print(f"trace           : {result.trace_path}")
    if not result.ok:
        print("\nFAILED chunks:", file=sys.stderr)
        for error in result.errors:
            print(f"  - {error}", file=sys.stderr)
        return
    if dry_run:
        print("\nDRY RUN — both phases ran, nothing was written.")
        return
    if result.matched_term_count == 0:
        print(
            "\nOK — no chunk term reached the glossary; every chunk file records the "
            "unmatched terms and their closest candidate."
        )
        return
    print(
        f"\nOK — {int(totals.get('matched_terms') or 0):,} relevant term(s) saved, "
        "one file per chunk under the chunk's own name."
    )
