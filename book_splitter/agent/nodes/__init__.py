"""The splitter's own graph nodes.

Only the stations that differ from every other stage live here:

``scout``     deterministic recon (no LLM)
``validate``  deterministic scoring of the submitted map
``execute``   slice + verify (the stage's write tail)
``terminal``  the stage's give-up result, which reports the validator's issues

The shared ``agent``, ``tools`` and ``give_up`` bodies come from
:mod:`book_splitter.stage_graph.nodes`; the splitter overrides only ``give_up``.
"""

from .execute import make_slice_node, make_verify_node
from .scout import build_scout_report, make_scout_node
from .terminal import make_give_up_node
from .validate import make_validate_node

__all__ = [
    "build_scout_report",
    "make_give_up_node",
    "make_scout_node",
    "make_slice_node",
    "make_validate_node",
    "make_verify_node",
]
