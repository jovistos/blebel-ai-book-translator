"""The splitting agent itself: state, prompts, model, nodes, routing, graph.

``build_graph`` compiles the LangGraph state machine and ``run_split`` runs it
end to end; ``run_split_from_map`` replays a map that was supplied instead of
discovered (the ``--map`` path, no model involved). Everything else is a building
block of those.
"""

from .graph import build_graph
from .result import SplitterResult
from .runner import (
    PRESET_MAP_STRATEGY,
    REUSED_CHAPTERS_STRATEGY,
    result_from_existing_chapters,
    run_split,
    run_split_from_map,
)
from .state import SplitterState, merge_usage

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
