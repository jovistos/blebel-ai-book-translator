"""Stage 1: split a book into chapters without reading its body.

This package exposes the Stage 1 splitter. It wraps and re-exports the implementation
in :mod:`book_translator.agent`.
"""

from __future__ import annotations

from ..agent import (
    PRESET_MAP_STRATEGY,
    REUSED_CHAPTERS_STRATEGY,
    SplitterResult,
    SplitterState,
    build_graph,
    merge_usage,
    result_from_existing_chapters,
    run_split,
    run_split_from_map,
)

__all__ = [
    "PRESET_MAP_STRATEGY",
    "REUSED_CHAPTERS_STRATEGY",
    "SplitterResult",
    "SplitterState",
    "build_graph",
    "merge_usage",
    "result_from_existing_chapters",
    "run_split",
    "run_split_from_map",
]
