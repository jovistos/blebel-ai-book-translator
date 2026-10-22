"""matter command: implementation and its human summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from book_splitter.core.constants import (
    DEFAULT_MATTER_ATTEMPTS,
    DEFAULT_SOURCE_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
)
from book_splitter.core.errors import ConfigError, ToolError
from book_splitter.core.report import human_seconds
from book_splitter.documents.chapters import (
    chapters_dir,
    chunks_dir,
    has_chunk_files,
    matter_dir,
    matter_prompts_dir,
    translations_dir,
)

from .. import parents
from ..common import (
    _model_settings,
    _print_json,
    _resolve_chapters_dir,
)


def add_parser(sub: Any) -> None:
    """Register this command's parser (the help text is snapshot-tested)."""
    p_matter = sub.add_parser(
        "matter",
        help="stage 6b: localize the front/back matter before translation with its "
        "own LangGraph agent — no dictionary, a web-searched published title, the "
        "BleBel credit, original-edition material omitted, published formatting",
    )
    parents.add_book(p_matter, required=False)
    p_matter.add_argument(
        "--out",
        "-o",
        default=None,
        help="the chapters folder from stage 1 (default: data/output/<book>/01-chapters); "
        "the chunks, translations and matter folders are derived from the book folder",
    )
    p_matter.add_argument(
        "--chapters-in",
        default=None,
        help="the chapter folder the chunks were cut from (default: derived from the "
        "book); point it at 02-preprocessed/ when that is what was chunked",
    )
    p_matter.add_argument(
        "--chunks-in",
        default=None,
        help="the chunks folder from stage 5 (default: data/output/<book>/05-chunks); "
        "a book folder or _chunk_manifest.json is also accepted",
    )
    p_matter.add_argument(
        "--translations-out",
        default=None,
        help="where the localized matter chunks land; this is the same folder stage 7 "
        "writes, so stage 8 assembles them with the body "
        "(default: data/output/<book>/07-translations)",
    )
    p_matter.add_argument(
        "--matter-out",
        default=None,
        help="where the book plan, manifest and trace land "
        "(default: data/output/<book>/06b-matter)",
    )
    p_matter.add_argument(
        "--prompts-out",
        default=None,
        help="where the whole agent input of each matter chunk lands "
        "(default: data/output/<book>/06b-matter-prompts)",
    )
    p_matter.add_argument(
        "--source-language",
        "-s",
        default=DEFAULT_SOURCE_LANGUAGE,
        help=f"the language the book is written in (default: {DEFAULT_SOURCE_LANGUAGE})",
    )
    p_matter.add_argument(
        "--target-language",
        "-t",
        default=DEFAULT_TARGET_LANGUAGE,
        help=f"the language of the translated edition (default: {DEFAULT_TARGET_LANGUAGE})",
    )
    p_matter.add_argument(
        "--book-title",
        default=None,
        help="the source title hint (default: the front matter's title or the filename)",
    )
    parents.add_model_flags(p_matter)
    p_matter.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MATTER_ATTEMPTS,
        help="model turns allowed for the book plan and for each matter chunk "
        f"(default: {DEFAULT_MATTER_ATTEMPTS})",
    )
    p_matter.add_argument(
        "--limit",
        type=int,
        default=None,
        help="localize only the first N matter chunks (a cheap look before the run)",
    )
    p_matter.add_argument(
        "--resume",
        "--skip-if-present",
        dest="resume",
        action="store_true",
        help="reuse a matter chunk whose source hash and language pair still match the "
        "manifest, and continue with the first pending chunk",
    )
    p_matter.add_argument(
        "--no-prompts",
        action="store_true",
        help="do not write the per-chunk agent input records",
    )
    p_matter.add_argument(
        "--no-manifest",
        action="store_true",
        help="do not write _matter_manifest.json / _matter_trace.json",
    )
    p_matter.add_argument(
        "--dry-run",
        action="store_true",
        help="ask the agent for every matter chunk but write no translation, prompt, "
        "manifest or plan file",
    )
    p_matter.set_defaults(func=cmd_matter)


def cmd_matter(args: argparse.Namespace) -> int:
    """Stage 6b: localize front/back matter with its own agent, before translation."""
    from ...chunk_glossary.runner import resolve_chunks_dir
    from ...matter_agent import run_matter

    if args.chapters_in:
        chapters_path = chapters_dir(Path(args.chapters_in).expanduser())
    else:
        # The matter stage keeps stage 1's folder: it creates synthetic
        # front/back-matter files and appends them to that stage's manifest.
        chapters_path = chapters_dir(
            _resolve_chapters_dir(args.book, args.in_dir, args.out, getattr(args, "output_root", None))
        )
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
    translations_out = (
        Path(args.translations_out).expanduser()
        if args.translations_out
        else translations_dir(chunks_path)
    )
    matter_out = (
        Path(args.matter_out).expanduser()
        if args.matter_out
        else matter_dir(chunks_path)
    )
    prompts_out = (
        Path(args.prompts_out).expanduser()
        if args.prompts_out
        else matter_prompts_dir(chunks_path)
    )

    if args.limit is not None and int(args.limit) < 0:
        print("error: --limit cannot be negative", file=sys.stderr)
        return 2
    if int(args.max_attempts) <= 0:
        print("error: --max-attempts must be greater than 0", file=sys.stderr)
        return 2

    def factory():
        return _model_settings(args, "matter")

    # A fully resumed run needs no key, so a missing configuration is carried
    # until a pending chunk actually needs the model.
    settings = None
    config_error: ConfigError | None = None
    try:
        settings = factory()
    except ConfigError as exc:
        config_error = exc

    if not args.quiet:
        where = (
            f"{settings.provider}/{settings.model}"
            if settings is not None
            else "no model resolved"
        )
        print(f"chapters={chapters_path.expanduser().resolve()}")
        print(f"chunks={chunks_path.expanduser().resolve()}")
        print(f"translations={translations_out.expanduser().resolve()}")
        print(f"matter={matter_out.expanduser().resolve()}")
        if not args.no_prompts:
            print(f"prompts={prompts_out.expanduser().resolve()}")
        print(
            f"language={args.source_language} -> {args.target_language}  "
            f"model={where}  attempts<={args.max_attempts}  no dictionary"
        )
        print(
            "policy      : web-search the published title, credit BleBel, omit the "
            "original edition's imprint material, format as published front/back matter"
        )
        if args.limit:
            print(f"limit       : first {args.limit} matter chunk(s)")
        if args.resume:
            print("resume      : matter chunks already localized and unchanged are reused")

    try:
        result = run_matter(
            str(chapters_path),
            str(chunks_path),
            str(translations_out),
            prompts_dir=None if args.no_prompts else str(prompts_out),
            matter_path=str(matter_out),
            source_language=args.source_language,
            target_language=args.target_language,
            book_title=args.book_title or "",
            settings=settings,
            settings_factory=None if settings is not None else factory,
            max_attempts=args.max_attempts,
            recursion_limit=args.recursion_limit,
            dry_run=args.dry_run,
            resume=args.resume,
            limit=args.limit,
            write_manifest=not args.no_manifest,
            write_prompts=not args.no_prompts,
            progress=None if args.quiet or args.json else _matter_progress,
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
    _print_matter_summary(result, dry_run=args.dry_run)
    return 0 if result.ok else 1


def _matter_progress(chunk: dict, done: int, total: int) -> None:
    """One durable line per localized (or omitted) matter chunk."""
    if chunk.get("reused"):
        flag = "reuse"
    elif chunk.get("omitted"):
        flag = "omit "
    elif chunk.get("ok"):
        flag = "ok   "
    else:
        flag = "FAIL "
    usage = chunk.get("usage") or {}
    print(
        f"  [{done:>3}/{total}] {flag} {chunk.get('filename') or '?'!s:<46} "
        f"{chunk.get('section_kind') or '?'!s:<5} "
        f"{int(chunk.get('source_chars') or 0):>6,} -> "
        f"{int(chunk.get('translation_chars') or 0):>6,} chars · "
        f"{int(usage.get('total_tokens') or 0):>7,} tok · "
        f"{int(chunk.get('attempts') or 0)} attempt(s)"
        + (f" · {chunk.get('omit_reason')}" if chunk.get("omitted") else "")
    )


def _print_matter_summary(result, dry_run: bool = False) -> None:
    plan = result.book_plan or {}
    validation = result.validation or {}
    totals = result.totals or {}
    tokens = result.tokens or {}
    timing = result.timing or {}
    print()
    print(
        f"title           : {plan.get('translated_title') or '—'}"
        + (
            f"  (original: {plan.get('original_title')!r})"
            if plan.get("original_title")
            else ""
        )
    )
    print(
        f"credit          : {plan.get('translation_credit') or 'Translated by BleBel'}"
    )
    if plan.get("title_evidence"):
        print(f"title evidence  : {plan['title_evidence']}")
    print(
        f"sections        : {result.section_count} front/back matter file(s), "
        f"{int(totals.get('matter_chunks') or 0):,} chunk(s)"
    )
    print(
        f"chunks          : {result.ok_count}/{result.chunk_count} localized"
        + (f"  reused: {result.reused_count:,}" if result.reused_count else "")
        + (f"  omitted: {int(totals.get('omitted') or 0)}" if totals.get("omitted") else "")
    )
    omitted = plan.get("omit") or []
    if omitted:
        print(f"omitted source  : {len(omitted)} phrase(s); the first few:")
        for phrase in omitted[:5]:
            print(f"    - {phrase}")
        if len(omitted) > 5:
            print(f"    … and {len(omitted) - 5} more (see the plan)")
    if validation.get("warnings"):
        for warning in validation["warnings"][:5]:
            print(f"  warning       : {warning}")
    print(
        f"tokens used     : {int(tokens.get('total_tokens') or 0):,} total "
        f"({int(tokens.get('input_tokens') or 0):,} in, "
        f"{int(tokens.get('output_tokens') or 0):,} out) over "
        f"{int(tokens.get('llm_calls') or 0):,} call(s)"
    )
    if timing:
        print(
            f"time            : {human_seconds(float(timing.get('run_seconds', 0.0)))}"
            f"  {float(timing.get('seconds_per_chunk', 0.0)):.1f} s/chunk"
        )
    print(f"output dir      : {result.output_dir}")
    if not dry_run and result.prompts_dir:
        print(f"prompts         : {result.prompts_dir}")
    print(f"plan dir        : {result.matter_path}")
    if result.manifest_path:
        print(f"manifest        : {result.manifest_path}")
    if result.trace_path:
        print(f"trace           : {result.trace_path}")
    if result.skipped:
        print("\nOK — no front/back matter chunks to localize.")
        return
    if not result.ok:
        print("\nFAILED matter chunks:", file=sys.stderr)
        for error in result.errors:
            print(f"  - {error}", file=sys.stderr)
        return
    if dry_run:
        print("\nDRY RUN — every matter chunk was localized, nothing was written.")
        return
    print(
        "\nOK — front/back matter localized without a dictionary; the translated chunks "
        "are in 07-translations/ and stage 8 will assemble them with the body."
    )
