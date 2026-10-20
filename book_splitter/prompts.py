"""Deprecated import path — use :mod:`book_splitter.agent.prompts` instead."""

from __future__ import annotations

from .agent.prompts import (
    FINAL_TURN_NUDGE,
    NUDGE_NO_PROPOSAL,
    SCOUT_REPORT_TEMPLATE,
    SYSTEM_PROMPT,
    validation_feedback,
)

__all__ = [
    "FINAL_TURN_NUDGE",
    "NUDGE_NO_PROPOSAL",
    "SCOUT_REPORT_TEMPLATE",
    "SYSTEM_PROMPT",
    "validation_feedback",
]
