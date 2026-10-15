"""Write policy the tool executor enforces on behalf of the graph.

Guardrail: no chapter file may be written or verified before the submitted map
has passed ``validate_split_map_tool``-equivalent validation. The model can call
``slice_book_by_line_numbers`` as often as it likes; the executor refuses and
tells it what to do instead.
"""

from __future__ import annotations

from typing import Any

from .registry import is_write_tool

BLOCKED_MESSAGE = (
    "{name} is blocked until the map passes validation. Call "
    "validate_split_map_tool, fix any issues, then propose_split_map."
)


def is_blocked(name: str, validation: dict[str, Any] | None) -> bool:
    """True when calling ``name`` now would violate the write policy."""
    return is_write_tool(name) and not bool((validation or {}).get("ok"))


def blocked_payload(name: str) -> dict[str, str]:
    """The error payload returned to the model for a blocked call."""
    return {"error": BLOCKED_MESSAGE.format(name=name)}
