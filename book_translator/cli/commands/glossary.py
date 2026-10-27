"""glossary command: implementation and its human summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from book_translator.core.constants import (
    DEFAULT_GLOSSARY_ATTEMPTS,
    DEFAULT_GLOSSARY_CHUNK_CHARS,
    DEFAULT_SOURCE_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
)
from book_translator.core.errors import ConfigError, ToolError
from book_translator.core.report import human_seconds
from book_translator.core.settings import PROVIDERS
from book_translator.documents.chapters import (
    glossary_dir,
    has_chapter_files,
)
from book_translator.documents.window import (
    chapter_window_limit_for_path,
)

from .. import parents
from ..common import (
    _command_paths,
    _command_text_dir,
    _model_settings,
    _print_json,
    _window_label,
)


def add_parser(sub: Any) -> None:
    """Register this command's parser (the help text is snapshot-tested)."""
    p_gloss = sub.add_parser(
        "glossary",
        help="stage 4: walk the reduced chapters first to last and build one "
        "book-wide dictionary (term, category, locked target-language translation, "
        "context) with an LLM, feeding each chapter the dictionary so far",
    )
    parents.add_book(p_gloss, required=False)
    p_gloss.add_argument(
        "--out",
        "-o",
        default=None,
        help="the chapter folder (default: the cleaned data/output/<book>/02-preprocessed "
        "when stage 2 wrote it, else data/output/<book>/01-chapters)",
    )
    p_gloss.add_argument(
        "--sentences",
        choices=("unique", "reduced"),
        default="unique",
        help="which stage-3 folder to mine: 'unique' first mentions (default) or "
        "'reduced' every entity-bearing sentence",
    )
    p_gloss.add_argument(
        "--sentences-in",
        default=None,
        help="read the sentences from this folder instead of deriving it from --out",
    )
    p_gloss.add_argument(
        "--glossary-out",
        default=None,
        help="where to write glossary.json / glossary.md "
        "(default: data/output/<book>/04-glossary)",
    )
    p_gloss.add_argument(
        "--source-language",
        "-s",
        default=DEFAULT_SOURCE_LANGUAGE,
        help=f"the language the book is written in (default: {DEFAULT_SOURCE_LANGUAGE})",
    )
    p_gloss.add_argument(
        "--target-language",
        "-t",
        default=DEFAULT_TARGET_LANGUAGE,
        help=f"the language being translated into (default: {DEFAULT_TARGET_LANGUAGE})",
    )
    p_gloss.add_argument(
        "--book-title",
        default=None,
        help="the title the prompt uses (default: the source filename from stage 1)",
    )
    p_gloss.add_argument(
        "--include-front-matter",
        action="store_true",
        help="also mine the front matter (skipped by default)",
    )
    p_gloss.add_argument(
        "--include-back-matter",
        action="store_true",
        help="also mine the back matter (skipped by default)",
    )
    p_gloss.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_GLOSSARY_ATTEMPTS,
        help="model calls allowed per chapter before it is reported as failed "
        f"(default: {DEFAULT_GLOSSARY_ATTEMPTS})",
    )
    p_gloss.add_argument(
        "--extraction-method",
        choices=("auto", "structured", "json"),
        default="auto",
        help="'auto' asks for native structured output and falls back to JSON mode on "
        "the last attempt (default)",
    )
    p_gloss.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="response cap per model call (default: 8192); 0 leaves it to the provider",
    )
    p_gloss.add_argument(
        "--max-chapter-chars",
        type=int,
        default=DEFAULT_GLOSSARY_CHUNK_CHARS,
        help="split a chapter larger than this into parts, carrying the dictionary "
        f"across them (default: {DEFAULT_GLOSSARY_CHUNK_CHARS})",
    )
    p_gloss.add_argument(
        "--max-context-entries",
        type=int,
        default=0,
        help="cap the dictionary sent with each chapter (0 = send all; a cap trades a "
        "little consistency for input tokens on a long book)",
    )
    p_gloss.add_argument(
        "--resume",
        "--skip-if-present",
        dest="resume",
        action="store_true",
        help="reuse the glossary and the chapters already in the manifest, and start "
        "at the first chapter that is missing",
    )
    p_gloss.add_argument(
        "--first-chapters",
        "--chapters",
        dest="first_chapters",
        type=int,
        default=None,
        help="build the dictionary from only the first N body chapters (the whole book "
        "when omitted)",
    )
    p_gloss.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process only the first N body chapters (kept as a unit-limit alias)",
    )
    p_gloss.add_argument(
        "--dry-run",
        action="store_true",
        help="ask the model for every chapter but write no glossary, manifest or trace",
    )
    p_gloss.add_argument("--provider", choices=sorted(PROVIDERS), default=None)
    p_gloss.add_argument("--model", default=None)
    p_gloss.add_argument("--base-url", default=None)
    p_gloss.add_argument("--api-key", default=None)
    p_gloss.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="sampling temperature for stage 4 (default: the config file, else 0.0)",
    )
    parents.add_llm_flags(p_gloss)
    p_gloss.add_argument("--json", action="store_true", help="machine-readable result")
    p_gloss.add_argument("--quiet", "-q", action="store_true")
    p_gloss.add_argument(
        "--text-in",
        default=None,
        help="chapter text folder to read (default: 02-preprocessed/ when it covers "
        "the window, else 01-chapters/); overrides --out",
    )
    p_gloss.set_defaults(func=cmd_glossary)


def cmd_glossary(args: argparse.Namespace) -> int:
    """Stage 4: build the book-wide glossary, one chapter at a time, first to last."""
    from book_translator.stages.glossary import run_glossary
    from book_translator.stages.glossary.runner import resolve_sentences_dir

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
    try:
        sentences_path = resolve_sentences_dir(
            chapters_path, sentences_dir=args.sentences_in, sentences=args.sentences
        )
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out_dir = (
        Path(args.glossary_out).expanduser() if args.glossary_out else glossary_dir(chapters_path)
    )

    # The key is resolved before the run so a setup problem is one line, not a
    # traceback from inside a provider client. A fully resumed run needs no key —
    # it never calls the model — so a missing credential is carried, not fatal.
    settings = None
    config_error: ConfigError | None = None
    try:
        settings = _model_settings(args, "dictionary")
    except ConfigError as exc:
        config_error = exc

    if not args.quiet:
        where = (
            f"{settings.provider}/{settings.model}"
            if settings is not None
            else "no model resolved"
        )
        print(f"chapters={chapters_path.expanduser().resolve()}")
        print(f"sentences={sentences_path.expanduser().resolve()} ({args.sentences})")
        print(f"glossary={out_dir.expanduser().resolve()}")
        print(
            f"language={args.source_language} -> {args.target_language}  "
            f"model={where}  method={args.extraction_method}  attempts<={args.max_attempts}"
        )
        skipped = []
        if not args.include_front_matter:
            skipped.append("front matter")
        if not args.include_back_matter:
            skipped.append("back matter")
        if skipped:
            print(f"skip        : {', '.join(skipped)}")
        label = _window_label(args, limit)
        if label is not None:
            print(f"window      : {label}")

    try:
        result = run_glossary(
            str(chapters_path),
            str(out_dir),
            sentences_dir=str(sentences_path),
            sentences=args.sentences,
            source_language=args.source_language,
            target_language=args.target_language,
            book_title=args.book_title or "",
            settings=settings,
            max_attempts=args.max_attempts,
            method=args.extraction_method,
            max_tokens=args.max_tokens or None,
            chunk_chars=args.max_chapter_chars,
            max_context_entries=args.max_context_entries,
            include_front_matter=args.include_front_matter,
            include_back_matter=args.include_back_matter,
            skip_if_present=args.resume,
            dry_run=args.dry_run,
            limit=limit,
            progress=None if args.quiet else _glossary_progress,
        )
    except ConfigError as exc:
        print(f"error: {config_error or exc}", file=sys.stderr)
        return 2
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        _print_json(result.to_dict())
        return 0 if result.ok else 1
    _print_glossary_summary(result, dry_run=args.dry_run)
    return 0 if result.ok else 1


def _glossary_progress(chapter: dict, done: int, total: int) -> None:
    """One line per chapter, printed as its answer is folded into the dictionary."""
    tokens = chapter.get("tokens") or {}
    print(
        f"  [{done}/{total}] {chapter['filename']:<44} "
        f"{chapter['added']:>4} new  {chapter['repeats']:>3} locked  "
        f"{chapter['conflicts']:>2} conflict  "
        f"{int(tokens.get('total_tokens') or 0):>8,} tok  "
        f"{chapter['attempts']} attempt(s)"
    )


def _print_glossary_summary(result, dry_run: bool = False) -> None:
    print()
    print(
        f"glossary        : {result.source_language} -> {result.target_language}"
        f"  {result.provider or '—'}/{result.model or '—'}"
    )
    if (getattr(result, "settings", None) or {}).get("json_only"):
        print(
            "  note          : the provider rejected native structured output; "
            "the run used JSON mode"
        )
    print(
        f"chapters        : {result.ok_count}/{result.body_chapter_count} body chapter(s) "
        "processed"
        + (f"  reused: {result.reused_count}" if result.reused_count else "")
        + (f"  skipped: {result.skipped_count}" if result.skipped_count else "")
    )
    if getattr(result, "limited", False):
        print("window          : a limited prefix — the manifest is partial")
    categories = result.categories or {}
    print(
        f"entries         : {result.entry_count}"
        + (
            "  (" + ", ".join(f"{name}={count}" for name, count in categories.items()) + ")"
            if categories
            else ""
        )
    )
    if result.conflicts:
        print(f"conflicts       : {len(result.conflicts)} (the first rendering stands)")
        for conflict in result.conflicts[:3]:
            print(
                f"    - {conflict.get('term')}: locked {conflict.get('locked')!r}, "
                f"{conflict.get('chapter_filename')} proposed {conflict.get('proposed')!r}"
            )
        if len(result.conflicts) > 3:
            print(f"    … and {len(result.conflicts) - 3} more (see the manifest)")
    tokens = result.tokens or {}
    chapters_with_calls = int(tokens.get("chapters") or 0)
    print(
        f"tokens used     : {int(tokens.get('total_tokens') or 0):,} total "
        f"({int(tokens.get('input_tokens') or 0):,} in, "
        f"{int(tokens.get('output_tokens') or 0):,} out) "
        f"over {chapters_with_calls} chapter(s), {int(tokens.get('calls') or 0)} call(s)"
    )
    timing = result.timing or {}
    if timing:
        print(
            f"time            : {human_seconds(float(timing.get('run_seconds', 0.0)))}"
            f"  (model {human_seconds(float(timing.get('model_seconds', 0.0)))})"
            f"  {float(timing.get('seconds_per_chapter', 0.0)):.1f} s/chapter"
        )
    if result.resumed:
        print(
            f"resumed         : {result.reused_count} chapter(s) reused from the manifest"
        )
    print(f"output dir      : {result.output_dir}")
    if result.glossary_path:
        print(f"glossary        : {result.glossary_path}")
    if getattr(result, "csv_path", None):
        print(f"csv             : {result.csv_path}")
    if result.markdown_path:
        print(f"markdown        : {result.markdown_path}")
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
        print("\nDRY RUN — every chapter was asked, nothing was written.")
        return
    if result.entry_count == 0:
        print("\nOK — no entity was found; the glossary is empty.")
        return
    print(f"\nOK — {result.entry_count} entr{'y' if result.entry_count == 1 else 'ies'} locked in.")
