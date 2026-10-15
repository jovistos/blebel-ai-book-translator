"""Deprecated import path — use :mod:`book_splitter.agent` instead."""

from __future__ import annotations

from .agent.graph import build_graph
from .agent.result import SplitterResult
from .agent.runner import run_split, write_trace
from .agent.state import SplitterState, merge_usage

__all__ = [
    "SplitterResult",
    "SplitterState",
    "build_graph",
    "merge_usage",
    "run_split",
    "write_trace",
]
