"""Terminal nodes: how a run ends without a result."""

from __future__ import annotations

from typing import Any

from ..state import SplitterState


def make_give_up_node():
    """Build the ``give_up`` node used when no valid map is ever submitted."""

    def give_up(state: SplitterState) -> dict[str, Any]:
        reasons = (state.get("validation") or {}).get("issues") or [
            "the agent did not submit a valid chapter map"
        ]
        return {
            "result": {
                "ok": False,
                "error": "failed to produce a valid chapter map",
                "issues": reasons,
            },
            "log": [f"give_up: {reasons}"],
        }

    return give_up
