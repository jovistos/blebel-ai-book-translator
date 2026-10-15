"""The validate node: deterministic scoring of the submitted chapter map."""

from __future__ import annotations

from typing import Any

from ...analysis.toc import detect_toc_block
from ...core.constants import DEFAULT_MIN_CHAPTERS
from ...documents.io import load_lines
from ...slicing.validation import validate_split_map
from ..prompts import NUDGE_NO_PROPOSAL, validation_feedback
from ..state import SplitterState


def make_validate_node():
    """Build the ``validate`` node that decides slice vs. another agent turn."""

    def validate(state: SplitterState) -> dict[str, Any]:
        proposal = list(state.get("proposed_splits") or [])
        if not proposal:
            return {
                "validation": {
                    "ok": False,
                    "issues": ["no split map was submitted"],
                    "warnings": [],
                    "points": [],
                    "chapter_count": 0,
                },
                "feedback": [NUDGE_NO_PROPOSAL],
                "log": ["validate: no proposal submitted"],
            }
        lines = load_lines(state["source_path"])
        result = validate_split_map(
            lines,
            proposal,
            toc_block=state.get("toc_block") or detect_toc_block(lines),
            min_chapters=int(state.get("min_chapters", DEFAULT_MIN_CHAPTERS)),
        )
        result["strategy"] = state.get("proposal_strategy")
        feedback = [] if result["ok"] else [validation_feedback(result)]
        return {
            "validation": result,
            "feedback": feedback,
            "log": [
                f"validate: ok={result['ok']} chapters={result['chapter_count']} "
                f"issues={len(result['issues'])} warnings={len(result['warnings'])}"
            ],
        }

    return validate
