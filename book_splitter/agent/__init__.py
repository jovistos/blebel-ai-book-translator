"""The splitting agent itself: state, prompts, model, nodes, routing, graph.

``build_graph`` compiles the LangGraph state machine and ``run_split`` runs it
end to end; everything else is a building block of those two.
"""

from .graph import build_graph
from .result import SplitterResult
from .runner import run_split
from .state import SplitterState, merge_usage

__all__ = [
    "SplitterResult",
    "SplitterState",
    "build_graph",
    "merge_usage",
    "run_split",
]
