"""postprocess command: implementation and its human summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from book_translator.core.constants import (
    DEFAULT_POSTPROCESS_ATTEMPTS,
    TRANSLATED_CHAPTERS_DIRNAME,
)
from book_translator.core.errors import ConfigError, ToolError
from book_translator.documents.chapters import (
    has_translated_chapter_files,
    postprocessed_chapters_dir,
    translated_chapters_dir,
)

from .. import parents
from ..common import (
    _command_paths,
    _command_text_dir,
    _model_settings,
    _print_json,
    _print_openings_line,
    _window_label,
)


def add_parser(sub: Any) -> None:
    """Register this command's parser (the help text is snapshot-tested)."""
    p_post = sub.add_parser(
        "postprocess",
        help="stage 10: postprocess the translated chapters' headings and table of "
        "contents in a separate folder, without reading chapter prose",
    )
    parents.add_book(p_post, required=False)
    p_post.add_argument(
        "--out",
        "-o",
        default=None,
        help="the chapter folder (default: the cleaned data/output/<book>/03-preprocessed "
        "when stage 3 wrote it, else data/output/<book>/02-chapters); "
        "the translated-chapters and postprocessed-chapters folders are derived from it",
    )
    p_post.add_argument(
        "--translated-chapters-in",
        default=None,
        help="stage 9's output folder (default: data/output/<book>/09-translated-chapters)",
    )
    p_post.add_argument(
        "--chunks-in",
        default=None,
        help="alternative way to locate the book: stage 5's chunks folder; its sibling "
        "09-translated-chapters/ is used as input",
    )
    p_post.add_argument(
        "--postprocessed-out",
        default=None,
        help="where the postprocessed chapters go "
        "(default: data/output/<book>/10-postprocessed-chapters)",
    )
    parents.add_model_flags(p_post)
    p_post.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_POSTPROCESS_ATTEMPTS,
        help="model turns allowed for the book plan and for each chapter "
        f"(default: {DEFAULT_POSTPROCESS_ATTEMPTS})",
    )
    p_post.add_argument(
        "--first-chapters",
        "--chapters",
        dest="first_chapters",
        type=int,
        default=None,
        help="postprocess only front matter plus the first N body chapters "
        "(the whole translated book when omitted)",
    )
    p_post.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process only the first N body chapters (kept as a unit-limit alias)",
    )
    p_post.add_argument(
        "--skip-if-present",
        "--resume",
        dest="skip_if_present",
        action="store_true",
        help="reuse a postprocessed chapter whose source hash and canonical title still "
        "match the manifest, and start with the first chapter that is missing",
    )
    p_post.add_argument(
        "--no-manifest",
        action="store_true",
        help="do not write _postprocess_manifest.json / _postprocess_trace.json",
    )
    p_post.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the book plan and every chapter plan, and compute the TOC, "
        "but write nothing",
    )
    p_post.set_defaults(func=cmd_postprocess)


def cmd_postprocess(args: argparse.Namespace) -> int:
    """Stage 10: postprocess the translated chapters' headings and table of contents."""
    from book_translator.stages.postprocess_agent import run_postprocess
    from book_translator.stages.translation.runner import resolve_chunks_dir as _resolve_chunks

    if args.translated_chapters_in:
        source_path = Path(args.translated_chapters_in).expanduser()
        if source_path.is_file():
            source_path = source_path.parent
        if (source_path / TRANSLATED_CHAPTERS_DIRNAME).is_dir():
            source_path = source_path / TRANSLATED_CHAPTERS_DIRNAME
    elif args.chunks_in:
        source_path = translated_chapters_dir(
            _resolve_chunks(Path(args.chunks_in).expanduser())
        )
    else:
        chapters_path = _command_text_dir(args, _command_paths(args))
        source_path = translated_chapters_dir(chapters_path)
    if not has_translated_chapter_files(source_path):
        print(
            f"no translated chapter files in {source_path.expanduser().resolve()} — "
            "run `concatenate` first",
            file=sys.stderr,
        )
        return 2
    out_dir = (
        Path(args.postprocessed_out).expanduser()
        if args.postprocessed_out
        else postprocessed_chapters_dir(source_path)
    )
    if int(args.max_attempts) <= 0:
        print("error: --max-attempts must be greater than 0", file=sys.stderr)
        return 2
    # Stage 9 counts body chapters directly (front matter is always included and
    # does not consume one of the N), so its limit is the user's N, not a file count.
    limit = args.first_chapters if args.first_chapters is not None else args.limit
    if limit is not None:
        limit = int(limit)

    try:
        settings = _model_settings(args, "postprocess")
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    if not args.quiet:
        print(f"provider={settings.provider} model={settings.model} base_url={settings.base_url}")
        print(f"translated-chapters={source_path.expanduser().resolve()}")
        print(f"postprocessed-chapters={out_dir.expanduser().resolve()}")
        print(
            f"budget      : max {args.max_attempts} model turn(s) per book plan and per chapter; "
            "headings/opening only, no chapter prose"
        )
        label = _window_label(args, limit)
        if label is not None:
            print(f"window      : {label}")
        if args.skip_if_present:
            print("resume      : chapters whose source hash and title still match are reused")

    try:
        result = run_postprocess(
            str(source_path),
            str(out_dir),
            settings=settings,
            max_attempts=args.max_attempts,
            dry_run=args.dry_run,
            recursion_limit=args.recursion_limit,
            limit=limit,
            skip_if_present=args.skip_if_present,
            write_manifest=not args.no_manifest,
        )
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not args.quiet:
        for chapter in result.chapters:
            for line in chapter.get("log") or []:
                print(f"  · {chapter['filename']} {line}")

    if args.json:
        _print_json(result.to_dict())
        return 0 if result.ok else 1
    _print_postprocess_summary(result, dry_run=args.dry_run)
    return 0 if result.ok else 1


def _print_postprocess_summary(result, dry_run: bool = False) -> None:
    validation = result.validation or {}
    totals = result.totals or {}
    toc = result.toc or {}
    print()
    print(
        f"book plan       : "
        f"{(result.book_plan.get('naming') or {}).get('mode', '—')} "
        f"label={(result.book_plan.get('naming') or {}).get('label', '—')!r} "
        f"· toc {(result.book_plan.get('toc') or {}).get('action', '—')}"
    )
    print(
        f"chapters        : {result.ok_count}/{result.chapter_count} postprocessed"
        + (f"  reused: {result.reused_count}" if result.reused_count else "")
    )
    print(
        f"headings        : {int(totals.get('demoted_headings') or 0)} demoted · "
        f"{int(totals.get('promoted_headings') or 0)} promoted · "
        f"{int(totals.get('merged_subtitles') or 0)} subtitle(s) folded into a title"
    )
    print(
        f"titles          : {int(totals.get('title_set') or 0)} set · "
        f"{int(totals.get('title_inserted') or 0)} inserted"
    )
    web = result.web or {}
    print(
        f"web search      : {len(web.get('searches') or [])} lookup(s) · "
        + ("attempted" if web.get("attempted") else "NOT attempted")
        + f" · source {(result.book_plan.get('title_source') or '—')}"
    )
    evidence = str(result.book_plan.get("title_evidence") or "").strip()
    if evidence:
        print(f"title evidence  : {evidence[:200]}")
    if toc:
        print(
            f"table of contents: {toc.get('action') or '—'} "
            f"({int(totals.get('toc_entries') or 0)} entries)"
            + (f" · {toc.get('filename')}" if toc.get("filename") else "")
            + (" · dry run" if dry_run else "")
        )
    if validation.get("warnings"):
        for warning in validation["warnings"][:5]:
            print(f"  warning       : {warning}")
    print(f"output dir      : {result.output_dir}")
    _print_openings_line(result.openings)
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
        print("\nDRY RUN — every plan was validated, nothing was written.")
        return
    print(
        "\nOK — postprocessed chapters written to a separate folder; the translated "
        "chapters were not touched."
    )
