"""Command line interface for the two-stage book pipeline.

Documents live in dedicated folders: sources are read from ``data/input/`` and
every generated artifact is written under ``data/output/<book>/``. Both can be
overridden with ``--in-dir`` / ``--out``.

Stage 1 turns the book into chapter files; stage 2 cleans each chapter, one
chapter at a time.

Examples::

    python -m book_splitter inspect                       # report on the only book in data/input
    python -m book_splitter inspect mybook.md --candidates
    python -m book_splitter split mybook.md               # -> data/output/mybook/chapters/
    python -m book_splitter split mybook.md --out /tmp/out
    python -m book_splitter split mybook.md --dry-run
    python -m book_splitter preprocess mybook              # -> data/output/mybook/preprocessed/
    python -m book_splitter preprocess mybook --dry-run
    python -m book_splitter scan mybook                    # damage profile, no LLM, no writes
    python -m book_splitter tools --stage preprocess       # list the toolset per stage
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analysis.candidates import extract_candidates
from .analysis.headers import assess_headers
from .analysis.toc import detect_toc_block
from .core.constants import (
    DEFAULT_FRONT_MATTER_LINES,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MAX_HEADERS,
    DEFAULT_MIN_CHAPTERS,
    DEFAULT_RECURSION_LIMIT,
    MAX_FEEDBACK_LOOPS,
    TOC_SCAN_LINES,
)
from .core.errors import ConfigError
from .core.settings import PROVIDERS
from .documents.chapters import discover_chapters, load_split_manifest
from .documents.io import load_lines, source_size_bytes
from .paths import default_input_dir, default_output_dir, resolve_book_path, resolve_output_dir


def _print_json(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _resolve_book(args: argparse.Namespace) -> Path:
    return resolve_book_path(getattr(args, "book", None), input_dir=args.in_dir)


def _resolve_chapters_dir(book: str | None, in_dir: str | None, out: str | None) -> Path:
    """Where stage 1's chapters are.

    Resolution order: an explicit ``--out``, a ``--book`` / positional argument
    that is already a folder, then the default output folder of the named book
    (or of the only book in ``data/input/``).
    """
    if out:
        return Path(out).expanduser()
    if book:
        candidate = Path(book).expanduser()
        if candidate.is_dir():
            return candidate
    source = resolve_book_path(book, input_dir=in_dir)
    return resolve_output_dir(source, None)


def cmd_tools(args: argparse.Namespace) -> int:
    from .agent.tools import grouped_tools

    wanted = getattr(args, "stage", None)
    printed = 0
    for stage, items in grouped_tools():
        if wanted and stage != wanted:
            continue
        print(f"{stage}:")
        if not items:
            print("    (none)")
        for name, summary in items:
            print(f"    {name}")
            print(f"        {summary}")
        printed += len(items)
        print()
    if printed == 0:
        print(f"no tools registered for stage {wanted!r}", file=sys.stderr)
        return 1
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Deterministic pre-flight: no LLM, no API key, no writes."""
    from .agent.tools import scan_markdown_headers

    source = _resolve_book(args)
    lines = load_lines(source)
    headers = scan_markdown_headers(str(source))["headers"]
    toc = detect_toc_block(lines, scan_lines=args.toc_scan_lines)

    report = {
        "source": str(source),
        "input_dir": str(default_input_dir()),
        "line_count": len(lines),
        "size_bytes": source_size_bytes(source),
        "header_count": len(headers),
        "header_assessment": assess_headers(headers, len(lines)),
        "headers": headers[: args.max_headers],
        "printed_toc_block": toc,
        "first_lines": [
            {"line_number": i + 1, "text": line.strip()[:120]}
            for i, line in enumerate(lines[: args.head_lines])
            if line.strip()
        ],
    }
    if args.candidates:
        report["heuristic_candidates"] = extract_candidates(
            lines, max_candidates=args.max_candidates
        )["candidates"]
    _print_json(report)
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    """Deterministic damage profile of a split book: no LLM, no writes."""
    from .preprocess.corpus import scan_corpus_damage
    from .preprocess.inspect import chapter_digest, inspect_chapter

    chapters_dir = _resolve_chapters_dir(args.book, args.in_dir, args.out)
    chapters = discover_chapters(chapters_dir)
    if args.headers:
        for chapter in chapters:
            _print_json(inspect_chapter(chapter.path, include_all=args.all_headers))
        return 0
    if args.digest:
        for chapter in chapters:
            report = inspect_chapter(chapter.path, include_all=args.all_headers)
            print(chapter_digest(report))
            print()
    report = scan_corpus_damage(chapters, max_examples_per_pattern=args.examples)
    report["chapters_path"] = str(chapters_dir)
    report["manifest"] = {
        "strategy": load_split_manifest(chapters_dir).get("strategy"),
        "chapter_count": len(chapters),
    }
    _print_json(report)
    return 0


def cmd_preprocess(args: argparse.Namespace) -> int:
    from .core.settings import resolve_settings
    from .preprocess_agent import run_preprocess

    chapters_dir = _resolve_chapters_dir(args.book, args.in_dir, args.out)
    out_dir = args.preprocessed_out or None

    try:
        settings = resolve_settings(
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            temperature=args.temperature,
        )
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    if not args.quiet:
        print(f"provider={settings.provider} model={settings.model} base_url={settings.base_url}")

    if not args.quiet:
        print(f"chapters={chapters_dir.expanduser().resolve()}")
        print(f"output={out_dir or 'data/output/<book>/preprocessed/'}")

    result = run_preprocess(
        str(chapters_dir),
        out_dir,
        settings=settings,
        max_attempts=args.max_attempts,
        dry_run=args.dry_run,
        recursion_limit=args.recursion_limit,
        workers=args.workers,
        write_manifest=not args.no_manifest,
    )

    if not args.quiet:
        for chapter in result.chapters:
            for line in chapter.get("log", []):
                print(f"  · {chapter['filename']} {line}")

    if args.json:
        _print_json(result.to_dict())
        return 0 if result.ok else 1

    _print_preprocess_summary(result, dry_run=args.dry_run)
    return 0 if result.ok else 1


def _print_preprocess_summary(result, dry_run: bool = False) -> None:
    print()
    print(f"chapters        : {result.ok_count}/{result.chapter_count} clean")
    print(f"llm calls       : {sum(c.get('attempts', 0) for c in result.chapters)}"
          f"  tokens: {result.usage.get('total_tokens', 0)}")
    if result.corpus_scan:
        counts = result.corpus_scan.get("counts", {})
        print("damage found    : " + ", ".join(f"{k}={v}" for k, v in counts.items() if v))
    print(f"output dir      : {result.output_dir}")
    if result.chapters:
        print("files:")
        for chapter in result.chapters:
            flag = "ok " if chapter["ok"] else "FAIL"
            print(
                f"  {flag} {chapter['filename']:<52} lines "
                f"{chapter['source_lines']:>5} -> {chapter['output_lines']:<5} "
                f"ops={len(chapter['operations'])} fixes={len(chapter['text_fixes'])}"
            )
    if result.manifest_path:
        print(f"manifest        : {result.manifest_path}")
    if result.trace_path:
        print(f"trace           : {result.trace_path}")
    if not result.ok:
        print("\nFAILED chapters:", file=sys.stderr)
        for error in result.errors:
            print(f"  - {error}", file=sys.stderr)
        return
    print(
        "\nDRY RUN — every plan validated, nothing written."
        if dry_run
        else "\nOK — every chapter cleaned and verified."
    )


def cmd_split(args: argparse.Namespace) -> int:
    from .agent import run_split  # imported late so `inspect` needs no LLM client
    from .core.settings import resolve_settings

    source = _resolve_book(args)
    out_dir = resolve_output_dir(source, args.out)

    try:
        settings = resolve_settings(
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            temperature=args.temperature,
        )
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    if not args.quiet:
        print(f"provider={settings.provider} model={settings.model} base_url={settings.base_url}")
        print(f"source={source}")
        print(f"output={out_dir.expanduser().resolve()}")

    result = run_split(
        str(source),
        str(out_dir),
        settings=settings,
        front_matter_lines=args.front_matter_lines,
        max_attempts=args.max_attempts,
        min_chapters=args.min_chapters,
        include_front_matter=not args.no_front_matter,
        dry_run=args.dry_run,
        recursion_limit=args.recursion_limit,
    )

    if not args.quiet:
        for line in result.log:
            print(f"  · {line}")

    if args.json:
        _print_json(result.to_dict())
        return 0 if result.ok else 1

    _print_summary(result, dry_run=args.dry_run)
    return 0 if result.ok else 1


def _print_summary(result, dry_run: bool = False) -> None:
    print()
    print(f"strategy        : {result.strategy}")
    print(f"llm calls       : {result.attempts}  tokens: {result.usage.get('total_tokens', 0)}")
    print(f"chapters        : {result.chapter_count}")
    print(f"output dir      : {result.output_dir}")
    if result.verification:
        print(f"byte-for-byte   : {result.verification.get('reassembles_byte_for_byte')}")
        print(
            f"lines           : {result.verification.get('output_lines')}"
            f"/{result.verification.get('source_lines')}"
        )
    if result.files:
        print("files:")
        for f in result.files:
            print(
                f"  {f['filename']:<52} lines {f['start_line']:>5}-{f['end_line']:<5} "
                f"{f['bytes']:>8} B"
            )
    if result.trace_path:
        print(f"trace           : {result.trace_path}")
    if not result.ok:
        print(f"\nFAILED: {result.error}", file=sys.stderr)
        for issue in (result.validation or {}).get("issues", []):
            print(f"  - {issue}", file=sys.stderr)
        return
    print(
        "\nDRY RUN — plan validated, nothing written."
        if dry_run
        else "\nOK — every source line is accounted for."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="book_splitter",
        description="Two LangGraph agents that turn a Markdown book into clean chapter files: "
        "`split` cuts it into chapters without reading the body, `preprocess` repairs each "
        "chapter's headers and conversion damage one chapter at a time. Sources come from "
        "data/input/, output goes to data/output/<book>/.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_book(parser_: argparse.ArgumentParser, required: bool) -> None:
        parser_.add_argument(
            "book",
            nargs="?" if not required else None,
            help="source document — a bare filename or a path "
            "(default: the only book found in data/input/)",
        )
        parser_.add_argument(
            "--in-dir",
            default=None,
            help=f"folder holding the source documents (default: {default_input_dir()})",
        )

    def add_model_flags(parser_: argparse.ArgumentParser) -> None:
        parser_.add_argument("--provider", choices=sorted(PROVIDERS), default=None)
        parser_.add_argument("--model", default=None)
        parser_.add_argument("--base-url", default=None)
        parser_.add_argument("--api-key", default=None)
        parser_.add_argument("--temperature", type=float, default=0.0)
        parser_.add_argument("--recursion-limit", type=int, default=DEFAULT_RECURSION_LIMIT)
        parser_.add_argument("--json", action="store_true", help="machine-readable result")
        parser_.add_argument("--quiet", "-q", action="store_true")

    p_tools = sub.add_parser("tools", help="list the tools, grouped by pipeline stage")
    p_tools.add_argument(
        "--stage",
        choices=["shared", "split", "preprocess"],
        default=None,
        help="show only one stage group (shared lists the generic tools)",
    )
    p_tools.set_defaults(func=cmd_tools)

    p_inspect = sub.add_parser(
        "inspect", help="deterministic structural report (no LLM, no writes)"
    )
    add_book(p_inspect, required=False)
    p_inspect.add_argument("--max-headers", type=int, default=DEFAULT_MAX_HEADERS)
    p_inspect.add_argument("--head-lines", type=int, default=40)
    p_inspect.add_argument("--toc-scan-lines", type=int, default=TOC_SCAN_LINES)
    p_inspect.add_argument(
        "--candidates", action="store_true", help="also list heuristic candidate lines"
    )
    p_inspect.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    p_inspect.set_defaults(func=cmd_inspect)

    p_split = sub.add_parser("split", help="stage 1: run the agent and write chapter files")
    add_book(p_split, required=True)
    p_split.add_argument(
        "--out",
        "-o",
        default=None,
        help="output folder for chapter files "
        f"(default: {default_output_dir()}/<book>/chapters)",
    )
    p_split.add_argument("--provider", choices=sorted(PROVIDERS), default=None)
    p_split.add_argument("--model", default=None)
    p_split.add_argument("--base-url", default=None)
    p_split.add_argument("--api-key", default=None)
    p_split.add_argument("--temperature", type=float, default=0.0)
    p_split.add_argument(
        "--front-matter-lines",
        type=int,
        default=DEFAULT_FRONT_MATTER_LINES,
        help="max lines the agent may read from the front of the book",
    )
    p_split.add_argument(
        "--max-attempts",
        type=int,
        default=MAX_FEEDBACK_LOOPS,
        help="max LLM turns before giving up",
    )
    p_split.add_argument(
        "--min-chapters",
        type=int,
        default=DEFAULT_MIN_CHAPTERS,
        help="reject plans with fewer split points than this",
    )
    p_split.add_argument("--recursion-limit", type=int, default=DEFAULT_RECURSION_LIMIT)
    p_split.add_argument(
        "--no-front-matter",
        action="store_true",
        help="do not write 00-front-matter.md (content before chapter 1)",
    )
    p_split.add_argument(
        "--dry-run", action="store_true", help="validate the plan but write no chapter files"
    )
    p_split.add_argument("--json", action="store_true", help="machine-readable result")
    p_split.add_argument("--quiet", "-q", action="store_true")
    p_split.set_defaults(func=cmd_split)

    p_scan = sub.add_parser(
        "scan",
        help="stage 2 pre-flight: header evidence and damage profile (no LLM, no writes)",
    )
    add_book(p_scan, required=False)
    p_scan.add_argument(
        "--out",
        "-o",
        default=None,
        help="the chapters folder from stage 1 (default: data/output/<book>/chapters)",
    )
    p_scan.add_argument(
        "--headers", action="store_true", help="print the full header record per chapter"
    )
    p_scan.add_argument(
        "--all-headers", action="store_true", help="include headers that look fine"
    )
    p_scan.add_argument("--digest", action="store_true", help="print the compact text digest")
    p_scan.add_argument("--examples", type=int, default=3)
    p_scan.set_defaults(func=cmd_scan)

    p_pre = sub.add_parser(
        "preprocess",
        help="stage 2: clean each chapter, one chapter at a time, and write the result",
    )
    add_book(p_pre, required=False)
    p_pre.add_argument(
        "--out",
        "-o",
        default=None,
        help="the chapters folder from stage 1 (default: data/output/<book>/chapters)",
    )
    p_pre.add_argument(
        "--preprocessed-out",
        default=None,
        help="where to write cleaned chapters (default: data/output/<book>/preprocessed)",
    )
    add_model_flags(p_pre)
    p_pre.add_argument(
        "--max-attempts",
        type=int,
        default=MAX_FEEDBACK_LOOPS,
        help="max LLM turns per chapter before giving up on it",
    )
    p_pre.add_argument(
        "--workers",
        type=int,
        default=1,
        help="chapters processed at once (default 1: strictly one chapter at a time)",
    )
    p_pre.add_argument(
        "--no-manifest",
        action="store_true",
        help="do not write _preprocess_manifest.json",
    )
    p_pre.add_argument(
        "--dry-run",
        action="store_true",
        help="validate every chapter's plan but write no files",
    )
    p_pre.set_defaults(func=cmd_preprocess)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
