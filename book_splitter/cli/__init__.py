"""The command-line interface package.

Each stage command now lives in its own module under
:mod:`book_splitter.cli.commands`; :mod:`book_splitter.cli.common` owns the
shared path, settings and rendering helpers, and
:mod:`book_splitter.cli.parser` builds the root parser. ``main`` and
``build_parser`` keep their historical import path so ``python -m
book_splitter`` and every existing test keep working.
"""

from __future__ import annotations

import sys

from .commands.bind import cmd_bind
from .commands.chunk import cmd_chunk
from .commands.chunk_glossary import cmd_chunk_glossary
from .commands.concatenate import cmd_concatenate
from .commands.convert import cmd_convert
from .commands.dictionary import cmd_dictionary
from .commands.epub import cmd_epub
from .commands.glossary import _print_glossary_summary, cmd_glossary
from .commands.inspect import cmd_inspect
from .commands.matter import cmd_matter
from .commands.openings import cmd_openings
from .commands.postprocess import cmd_postprocess
from .commands.preprocess import cmd_preprocess
from .commands.scan import cmd_scan
from .commands.split import cmd_split
from .commands.tools import cmd_tools
from .commands.translate import cmd_translate
from .common import _resolve_text_dir
from .parser import build_parser
from .parser import main as _command_main


def main(argv: list[str] | None = None) -> int:
    """Dispatch ``pipeline`` to the shared driver and everything else to parser."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "pipeline":
        from ..pipeline.parser import main as pipeline_main

        return pipeline_main(args[1:], prog="book_splitter pipeline")
    return _command_main(args)


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
