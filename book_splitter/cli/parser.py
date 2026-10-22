"""The root argument parser and dispatch for the stage commands.

Each command module owns its own parser fragment; this module only creates
the subparsers in registration order and keeps the historical entry points.
"""

from __future__ import annotations

import argparse
import sys

from book_splitter.core.errors import ConfigError, ToolError

from .commands.bind import add_parser as add_bind_parser
from .commands.chunk import add_parser as add_chunk_parser
from .commands.chunk_glossary import add_parser as add_chunk_glossary_parser
from .commands.concatenate import add_parser as add_concatenate_parser
from .commands.convert import add_parser as add_convert_parser
from .commands.dictionary import add_parser as add_dictionary_parser
from .commands.epub import add_parser as add_epub_parser
from .commands.glossary import add_parser as add_glossary_parser
from .commands.inspect import add_parser as add_inspect_parser
from .commands.matter import add_parser as add_matter_parser
from .commands.openings import add_parser as add_openings_parser
from .commands.postprocess import add_parser as add_postprocess_parser
from .commands.preprocess import add_parser as add_preprocess_parser
from .commands.scan import add_parser as add_scan_parser
from .commands.split import add_parser as add_split_parser
from .commands.tools import add_parser as add_tools_parser
from .commands.translate import add_parser as add_translate_parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="book_splitter",
        description="Stage 0 `convert` turns an EPUB, a text-based PDF or a TXT into "
        "Markdown (docling, no OCR for image-based PDFs), then four LangGraph agents, a "
        "dictionary funnel, a glossary pass, a reversible chunker, a per-chunk glossary "
        "matcher, a front/back matter localizer, a chunk translator, a heading "
        "postprocessor and a binder turn that Markdown book into a consistent translation: "
        "`split` cuts it into chapters "
        "without reading the body, "
        "`preprocess` repairs each chapter's headers and conversion damage one chapter "
        "at a time, `dictionary` reduces each chapter to the sentences that carry a "
        "named entity, `glossary` locks one rendering per term, `chunk` cuts each "
        "chapter into sentence-safe chunks that concatenate back exactly, "
        "`chunk-glossary` tags every chunk and attaches the glossary terms it mentions, "
        "`matter` localizes the front/back matter before translation (no dictionary, "
        "web-searched title, BleBel credit, original-edition material omitted), "
        "`translate` translates the body chunks in reading order with those terms, "
        "keeping each call's whole prompt for debugging, `concatenate` glues the "
        "translated chunks back into one file per chapter, `postprocess` makes the "
        "translated chapter titles, other headings and table of contents consistent "
        "without reading the prose, `openings` audits (and repairs) every chapter's "
        "first lines with no model at all, and `bind` concatenates the postprocessed "
        "chapters into one Markdown book. Sources come from data/input/, output goes to "
        "data/output/<book>/.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    add_convert_parser(sub)
    add_tools_parser(sub)
    add_inspect_parser(sub)
    add_split_parser(sub)
    add_scan_parser(sub)
    add_preprocess_parser(sub)
    add_dictionary_parser(sub)
    add_chunk_parser(sub)
    add_chunk_glossary_parser(sub)
    add_matter_parser(sub)
    add_translate_parser(sub)
    add_concatenate_parser(sub)
    add_postprocess_parser(sub)
    add_openings_parser(sub)
    add_bind_parser(sub)
    add_epub_parser(sub)
    add_glossary_parser(sub)
    p_pipeline = sub.add_parser(
        "pipeline",
        help="run the whole pipeline (same driver as scripts/run_pipeline.py)",
    )
    p_pipeline.set_defaults(func=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # One JSON object on stdout, nothing else: commands already guard their
    # human narration with ``--quiet``, so a JSON run turns that narration off
    # without changing the command's own result printing.
    if getattr(args, "json", False):
        args.quiet = True
    first_chapters = getattr(args, "first_chapters", None)
    if first_chapters is not None and int(first_chapters) < 0:
        print("error: --first-chapters cannot be negative", file=sys.stderr)
        return 2
    # Validate --llm syntax even when the command ends up needing no model (a
    # --map replay, a fully resumed run); the command's own stage is filled in
    # when the setting is actually resolved.
    try:
        from book_splitter.core.settings import parse_llm_overrides

        parse_llm_overrides(getattr(args, "llm", None), default_stage="split")
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        # ``--quiet`` silences the human narration *and* the closing summary, the
        # same contract the pipeline follows. ``--json`` still writes its one
        # object, so a quiet machine run keeps its result.
        if getattr(args, "quiet", False) and not getattr(args, "json", False):
            import contextlib
            import io

            with contextlib.redirect_stdout(io.StringIO()):
                return int(args.func(args))
        return int(args.func(args))
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ToolError as exc:
        # Stage 0's unsupported/image-based source and the other deterministic
        # stages report one actionable line instead of a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        # Includes FileNotFoundError: any disk-level problem (a read-only cache, a
        # missing folder on a computed path) is one actionable line, not a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2

