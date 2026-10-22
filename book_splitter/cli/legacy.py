"""Compatibility shim for the historical ``book_splitter.cli.legacy`` path.

The implementations now live in :mod:`book_splitter.cli.commands` (one
module per command) and the shared helpers in :mod:`book_splitter.cli.common`.
This module re-exports them so older imports and tests keep working.
"""

from __future__ import annotations

from .commands.bind import _print_bind_summary, cmd_bind
from .commands.chunk import _chunk_progress, _print_chunk_summary, cmd_chunk
from .commands.chunk_glossary import (
    _chunk_glossary_progress,
    _print_chunk_glossary_summary,
    cmd_chunk_glossary,
)
from .commands.concatenate import _assembly_progress, _print_assembly_summary, cmd_concatenate
from .commands.convert import cmd_convert
from .commands.dictionary import _dictionary_progress, _print_dictionary_summary, cmd_dictionary
from .commands.epub import _print_epub_summary, cmd_epub
from .commands.glossary import _glossary_progress, _print_glossary_summary, cmd_glossary
from .commands.inspect import cmd_inspect
from .commands.matter import _matter_progress, _print_matter_summary, cmd_matter
from .commands.openings import cmd_openings
from .commands.postprocess import _print_postprocess_summary, cmd_postprocess
from .commands.preprocess import _print_preprocess_summary, cmd_preprocess
from .commands.scan import cmd_scan
from .commands.split import _print_summary, _report_split, cmd_split
from .commands.tools import cmd_tools
from .commands.translate import _print_translation_summary, _TranslationBar, cmd_translate
from .common import (
    _CHAPTER_STAGE_FOLDERS,
    _command_paths,
    _command_text_dir,
    _covered_cleaned_dir,
    _ensure_markdown_source,
    _model_settings,
    _parse_entity_types,
    _print_json,
    _print_openings_line,
    _resolve_book,
    _resolve_chapters_dir,
    _resolve_openings_dir,
    _resolve_text_dir,
    _window_label,
)
from .parser import build_parser, main

__all__ = [
    "cmd_convert",
    "cmd_tools",
    "cmd_inspect",
    "cmd_scan",
    "cmd_preprocess",
    "_print_preprocess_summary",
    "cmd_dictionary",
    "_dictionary_progress",
    "_print_dictionary_summary",
    "cmd_chunk",
    "_chunk_progress",
    "_print_chunk_summary",
    "cmd_chunk_glossary",
    "_chunk_glossary_progress",
    "_print_chunk_glossary_summary",
    "cmd_matter",
    "_matter_progress",
    "_print_matter_summary",
    "_TranslationBar",
    "cmd_translate",
    "_print_translation_summary",
    "cmd_concatenate",
    "_assembly_progress",
    "_print_assembly_summary",
    "cmd_postprocess",
    "_print_postprocess_summary",
    "cmd_openings",
    "cmd_bind",
    "_print_bind_summary",
    "cmd_epub",
    "_print_epub_summary",
    "cmd_glossary",
    "_glossary_progress",
    "_print_glossary_summary",
    "cmd_split",
    "_report_split",
    "_print_summary",
    "_print_json",
    "_resolve_book",
    "_resolve_chapters_dir",
    "_resolve_text_dir",
    "_covered_cleaned_dir",
    "_command_paths",
    "_command_text_dir",
    "_ensure_markdown_source",
    "_model_settings",
    "_window_label",
    "_parse_entity_types",
    "_print_openings_line",
    "_CHAPTER_STAGE_FOLDERS",
    "_resolve_openings_dir",
    "build_parser",
    "main",
]
