"""Stage 2: per-chapter preprocessing with a LangGraph agent.

The stage-1 package (:mod:`book_splitter.agent`) turns one long Markdown book into
one file per chapter. This package takes that folder and repairs each chapter's
headers and conversion damage, **one chapter at a time**, with a different
specialised agent that is bound to a different toolset.

``build_preprocess_graph``
    the per-chapter state machine (collect → agent → validate → apply → verify)
``run_preprocess``
    drives it over every chapter, sequentially or with a worker pool

Nothing in this package reads a chapter's prose into a prompt: the agent sees
header records, evidence flags and short damage fragments, and it *describes*
edits instead of performing them.
"""

from .graph import build_preprocess_graph
from .llm import make_preprocess_llm
from .runner import (
    ChapterResult,
    PreprocessResult,
    ProgressHook,
    chapter_records,
    run_preprocess,
    summarise_chapter,
)
from .state import PreprocessState

__all__ = [
    "ChapterResult",
    "PreprocessResult",
    "PreprocessState",
    "ProgressHook",
    "build_preprocess_graph",
    "chapter_records",
    "make_preprocess_llm",
    "run_preprocess",
    "summarise_chapter",
]
