"""Deprecated import path — use :mod:`book_splitter.agent.prompts` instead."""

from __future__ import annotations

from .agent.prompts import (
    NUDGE_NO_PROPOSAL,
    SCOUT_REPORT_TEMPLATE,
    SYSTEM_PROMPT,
    validation_feedback,
)

__all__ = [
    "NUDGE_NO_PROPOSAL",
    "SCOUT_REPORT_TEMPLATE",
    "SYSTEM_PROMPT",
    "validation_feedback",
]
