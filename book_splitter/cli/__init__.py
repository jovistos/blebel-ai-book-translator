"""The command-line interface package.

The command implementations live in :mod:`book_splitter.cli.legacy` while the
package is migrated to per-command modules; ``main`` and ``build_parser`` keep
their historical import path so ``python -m book_splitter`` and every existing
test keep working.
"""

from __future__ import annotations

import sys

from . import legacy as _legacy
from .legacy import (
    _print_glossary_summary,
    _resolve_text_dir,
    build_parser,
    cmd_bind,
    cmd_chunk,
    cmd_chunk_glossary,
    cmd_concatenate,
    cmd_convert,
    cmd_dictionary,
    cmd_epub,
    cmd_glossary,
    cmd_inspect,
    cmd_matter,
    cmd_openings,
    cmd_postprocess,
    cmd_preprocess,
    cmd_scan,
    cmd_split,
    cmd_tools,
    cmd_translate,
)


def main(argv: list[str] | None = None) -> int:
    """Dispatch ``pipeline`` to the shared driver and everything else to legacy."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "pipeline":
        from ..pipeline.parser import main as pipeline_main

        return pipeline_main(args[1:], prog="book_splitter pipeline")
    return _legacy.main(args)

__all__ = [
    "build_parser",
    "cmd_bind",
    "cmd_chunk",
    "cmd_chunk_glossary",
    "cmd_concatenate",
    "cmd_convert",
    "cmd_dictionary",
    "cmd_epub",
    "cmd_glossary",
    "cmd_inspect",
    "cmd_matter",
    "cmd_openings",
    "cmd_postprocess",
    "cmd_preprocess",
    "cmd_scan",
    "cmd_split",
    "cmd_tools",
    "cmd_translate",
    "main",
    "_print_glossary_summary",
    "_resolve_text_dir",
]
