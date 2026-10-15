"""Graph nodes, one module per responsibility.

``scout``     deterministic recon (no LLM)
``agent``     one LLM turn
``tools``     policy-enforcing tool executor
``validate``  deterministic scoring of the submitted map
``execute``   slice + verify
``terminal``  give-up paths
"""

from .agent import make_agent_node
from .execute import make_slice_node, make_verify_node
from .scout import build_scout_report, make_scout_node
from .terminal import make_give_up_node
from .tools import absorb_proposal, make_tool_node
from .validate import make_validate_node

__all__ = [
    "absorb_proposal",
    "build_scout_report",
    "make_agent_node",
    "make_give_up_node",
    "make_scout_node",
    "make_slice_node",
    "make_tool_node",
    "make_validate_node",
    "make_verify_node",
]
