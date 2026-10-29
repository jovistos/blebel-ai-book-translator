"""dictionary command: implementation and its human summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from book_translator.core.constants import (
    DEFAULT_GPU_INDEX,
    DEFAULT_KEEP_ENTITY_TYPES,
    DEFAULT_MIN_SENTENCE_WORDS,
)
from book_translator.core.errors import MissingDependencyError, ToolError
from book_translator.core.report import human_seconds
from book_translator.documents.chapters import (
    has_chapter_files,
    reduced_dir,
    unique_dir,
)
from book_translator.documents.window import (
    chapter_window_limit_for_path,
)

from .. import parents
from ..common import (
    _command_paths,
    _command_text_dir,
    _parse_entity_types,
    _print_json,
    _window_label,
)


def add_parser(sub: Any) -> None:
    """Register this command's parser (the help text is snapshot-tested)."""
    p_dict = sub.add_parser(
        "dictionary",
        help="stage 4: reduce each chapter to its entity-bearing sentences "
        "(sentence split -> uppercase filter -> 4-word floor -> NER gate), then to "
        "the sentences that introduce a noun (first mentions only)",
    )
    parents.add_book(p_dict, required=False)
    p_dict.add_argument(
        "--out",
        "-o",
        default=None,
        help="the chapter folder (default: the cleaned data/output/<book>/03-preprocessed "
        "when stage 3 wrote it, else data/output/<book>/02-chapters); pass "
        "02-chapters to reduce the raw text instead",
    )
    p_dict.add_argument(
        "--reduced-out",
        default=None,
        help="where to write the reduced chapters (default: data/output/<book>/04-reduced)",
    )
    p_dict.add_argument(
        "--unique-out",
        default=None,
        help="where to write the first-mention chapters "
        "(default: a sibling of the reduced folder, data/output/<book>/04-reduced-unique)",
    )
    p_dict.add_argument(
        "--language",
        default="auto",
        help="'auto' samples a few chapters and detects the language; otherwise an "
        "ISO code: sr, hr, sl, bg (CLASSLA) or en (stanza)",
    )
    p_dict.add_argument(
        "--gpu",
        type=int,
        default=None,
        help=f"physical GPU to pin the tagger to (default: GPU {DEFAULT_GPU_INDEX}, or "
        "$CUDA_VISIBLE_DEVICES when that is already set)",
    )
    p_dict.add_argument("--cpu", action="store_true", help="run the tagger on the CPU")
    p_dict.add_argument(
        "--min-words",
        type=int,
        default=DEFAULT_MIN_SENTENCE_WORDS,
        help=f"drop sentences with fewer words than this (default: {DEFAULT_MIN_SENTENCE_WORDS})",
    )
    p_dict.add_argument(
        "--no-uppercase-filter",
        action="store_true",
        help="do not drop sentences whose only capital letter is the first word",
    )
    p_dict.add_argument(
        "--keep-types",
        default=",".join(DEFAULT_KEEP_ENTITY_TYPES),
        help="entity types that make a sentence survive "
        f"(default: {','.join(DEFAULT_KEEP_ENTITY_TYPES)})",
    )
    p_dict.add_argument(
        "--drop-temporal-only",
        action="store_true",
        help="also drop a sentence when every entity in it is a date or a time",
    )
    p_dict.add_argument(
        "--skip-if-present",
        "--resume",
        dest="skip_if_present",
        action="store_true",
        help="reuse a chapter whose reduced (and first-mention) file already exists; a "
        "fully reduced book returns without loading a tagger",
    )
    p_dict.add_argument(
        "--no-save-reduced",
        action="store_true",
        help="write no reduced chapters; the manifest and the first-mention chapters "
        "still land",
    )
    p_dict.add_argument(
        "--no-save-unique",
        action="store_true",
        help="write no first-mention chapters, only the reduced ones",
    )
    p_dict.add_argument(
        "--unique-only",
        action="store_true",
        help="write only the first-mention chapters (implies --no-save-reduced)",
    )
    p_dict.add_argument(
        "--first-chapters",
        "--chapters",
        dest="first_chapters",
        type=int,
        default=None,
        help="reduce only the first N body chapters of the book (the whole book when omitted)",
    )
    p_dict.add_argument(
        "--limit",
        type=int,
        default=None,
        help="reduce only the first N body chapters (kept as a unit-limit alias)",
    )
    p_dict.add_argument(
        "--examples",
        type=int,
        default=5,
        help="dropped-sentence examples kept per reason in the trace (default: 5)",
    )
    p_dict.add_argument(
        "--dry-run",
        action="store_true",
        help="reduce every chapter but write no reduced file, manifest or trace",
    )
    p_dict.add_argument("--json", action="store_true", help="machine-readable result")
    p_dict.add_argument("--quiet", "-q", action="store_true")
    p_dict.add_argument(
        "--text-in",
        default=None,
        help="chapter text folder to read (default: 03-preprocessed/ when it covers "
        "the window, else 02-chapters/); overrides --out",
    )
    p_dict.set_defaults(func=cmd_dictionary)


def cmd_dictionary(args: argparse.Namespace) -> int:
    """Stage 4: reduce each chapter to its entity-bearing sentences, then to first mentions."""
    from book_translator.stages.dictionary import run_dictionary

    # Accept either the chapters folder or the book folder that holds it: an
    # explicit ``--out`` is taken as given, while the default prefers the cleaned
    # ``03-preprocessed/`` folder only when it covers the same chapter window.
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
    out_dir = (
        Path(args.reduced_out).expanduser() if args.reduced_out else reduced_dir(chapters_path)
    )
    # `--unique-only` is the shortcut for "I only want the small list"; otherwise
    # both folders are written and each can still be switched off on its own.
    save_reduced = not args.no_save_reduced and not args.unique_only
    save_unique = not args.no_save_unique
    unique_out_dir = (
        Path(args.unique_out).expanduser() if args.unique_out else unique_dir(out_dir)
    )
    keep_types = _parse_entity_types(args.keep_types)

    if not args.quiet:
        device = "cpu" if args.cpu else f"gpu {args.gpu if args.gpu is not None else DEFAULT_GPU_INDEX}"
        print(f"chapters={chapters_path.expanduser().resolve()}")
        if save_reduced:
            print(f"reduced={out_dir.expanduser().resolve()}")
        if save_unique:
            print(f"unique={unique_out_dir.expanduser().resolve()}")
        print(f"language={args.language} device={device} keep_types={','.join(keep_types)}")
        label = _window_label(args, limit)
        if label is not None:
            print(f"window       : {label}")
        if not save_reduced and not save_unique:
            print("save         : nothing is written except the manifest")
        elif not save_reduced:
            print("save         : first mentions only (--no-save-reduced)")
        elif not save_unique:
            print("save         : reduced chapters only (--no-save-unique)")

    try:
        result = run_dictionary(
            str(chapters_path),
            str(out_dir),
            unique_output_dir=str(unique_out_dir),
            language=args.language,
            gpu=args.gpu,
            use_gpu=not args.cpu,
            min_words=args.min_words,
            require_uppercase=not args.no_uppercase_filter,
            keep_types=keep_types,
            drop_temporal_only=args.drop_temporal_only,
            skip_if_present=args.skip_if_present,
            save_reduced=save_reduced,
            save_unique=save_unique,
            dry_run=args.dry_run,
            limit=limit,
            example_limit=args.examples,
            unique_example_limit=args.examples,
            progress=None if args.quiet else _dictionary_progress,
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
    _print_dictionary_summary(result, dry_run=args.dry_run)
    return 0 if result.ok else 1


def _dictionary_progress(chapter: dict, done: int, total: int) -> None:
    """One line per reduced chapter, printed as it finishes."""
    unique = chapter.get("unique")
    tail = f" -> {chapter.get('unique_kept', 0):<5} unique" if unique is not None else ""
    print(
        f"  [{done}/{total}] {chapter['filename']:<48} "
        f"{chapter['sentences']:>5} sentences -> {chapter['kept']:<5} kept{tail}"
        f"  ner {human_seconds(float(chapter.get('ner_seconds') or 0.0)):>8}"
    )


def _print_dictionary_summary(result, dry_run: bool = False) -> None:
    language = result.language or {}
    print()
    print(
        f"language        : {language.get('language') or '—'} "
        f"({language.get('method')}, confidence {language.get('confidence', 0):.2f})"
        f"  backend: {result.backend or '—'}"
        + (f" on {result.device}" if result.device else "")
    )
    if language.get("note"):
        print(f"  note          : {language['note']}")
    totals = result.totals or {}
    unique_totals = result.unique_totals or {}
    timing = result.timing or {}
    save_unique = getattr(result, "save_unique", True)

    def _unique_line(label: str) -> str:
        dropped = unique_totals.get("dropped_duplicate_noun", 0)
        return (
            f"{label:<16}: {unique_totals.get('sentences', 0)} -> "
            f"{unique_totals.get('kept', 0)} kept "
            f"(duplicate-noun={dropped}, nouns={unique_totals.get('nouns', 0)})"
        )

    print(f"chapters        : {result.ok_count}/{result.chapter_count} ok"
          + (f"  reused: {result.skipped_count}" if result.skipped_count else ""))
    if getattr(result, "limited", False):
        print("window          : a limited prefix — the manifest is partial")
    if result.skipped:
        print(
            "nothing to do   : every reduced chapter is already present"
            + (" (both folders)" if save_unique else "")
        )
    else:
        print(
            f"sentences       : {totals.get('sentences', 0)} -> {totals.get('kept', 0)} kept"
        )
        print(
            "dropped         : "
            f"short={totals.get('dropped_short', 0)} "
            f"no-uppercase={totals.get('dropped_no_uppercase', 0)} "
            f"no-entity={totals.get('dropped_no_entity', 0)}"
            + (
                f" temporal-only={totals.get('dropped_temporal', 0)}"
                if result.settings.get("drop_temporal_only")
                else ""
            )
        )
        if save_unique:
            print(_unique_line("first mentions"))
        entities = result.entities or {}
        if entities:
            print("entities        : " + ", ".join(f"{k}={v}" for k, v in entities.items()))
        if timing:
            tagged = int(totals.get("sentences_tagged", 0))
            print(
                f"ner time        : {human_seconds(float(timing.get('ner_seconds', 0.0)))} "
                f"for {tagged:,} sentence(s)"
                + (
                    f"  ({float(timing.get('milliseconds_per_sentence', 0.0)):.1f} ms each, "
                    f"{float(timing.get('sentences_per_second', 0.0)):.0f}/s)"
                    if tagged
                    else ""
                )
            )
            if timing.get("backend_load_seconds"):
                where = result.device or ("cpu" if result.gpu is None else f"gpu {result.gpu}")
                print(
                    f"model load      : "
                    f"{human_seconds(float(timing['backend_load_seconds']))} "
                    f"({result.backend or '—'}, {where})"
                )
            print(
                f"total time      : {human_seconds(float(timing.get('run_seconds', 0.0)))}"
                f"  (chapters {human_seconds(float(timing.get('chapter_seconds', 0.0)))})"
            )
    print(f"output dir      : {result.output_dir}")
    if save_unique and getattr(result, "unique_dir", ""):
        print(f"unique dir      : {result.unique_dir}")
    if result.manifest_path:
        print(f"manifest        : {result.manifest_path}")
    if result.trace_path:
        print(f"trace           : {result.trace_path}")
    if not result.ok:
        print("\nFAILED chapters:", file=sys.stderr)
        for error in result.errors:
            print(f"  - {error}", file=sys.stderr)
        return
    if result.skipped:
        print("\nOK — nothing was written: the existing folders were reused.")
        return
    if dry_run:
        print("\nDRY RUN — every chapter was reduced, nothing was written.")
        return
    lines = []
    if result.save_reduced:
        lines.append("OK — reduced chapters written.")
    if save_unique:
        lines.append("OK — first-mention chapters written.")
    if not lines:
        lines.append("OK — the manifest was written; no chapter files.")
    print("\n" + "\n".join(lines))
