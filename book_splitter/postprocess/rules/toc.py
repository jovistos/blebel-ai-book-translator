"""Rule group: the table-of-contents decision.

Two rules live here because they run at different points in the validation:
:func:`check_toc_plan` validates the submitted shape (and applies the
deterministic default when none was submitted); :func:`check_toc_state` compares
the chosen action with the TOC the book actually has, which needs the generated
entries and therefore runs after the canonical titles are known.
"""

from __future__ import annotations

from typing import Any

from ...core.constants import TOC_ACTIONS, TOC_STYLES
from ..titles import clean_title
from ..toc import toc_matches_entries
from .base import BookPlanDraft


def default_toc_plan(book: dict[str, Any]) -> dict[str, Any]:
    """The deterministic TOC decision for a plan that did not submit one."""
    toc = book.get("toc") or {}
    return {
        "action": "update" if toc.get("exists") else "create",
        "heading": "Sadržaj",
        "style": "numbered",
        "include_back_matter": False,
    }


def check_toc_plan(draft: BookPlanDraft) -> None:
    """Validate/normalise ``plan["toc"]``."""
    if draft.toc_raw is None:
        default = default_toc_plan(draft.book)
        draft.toc_action = default["action"]
        draft.toc_heading = default["heading"]
        draft.toc_style = default["style"]
        draft.toc_include_back_matter = default["include_back_matter"]
        draft.warn("no toc plan was submitted; the deterministic default was used")
    else:
        draft.toc_action = str(draft.toc_raw.get("action") or "").strip().lower()
        draft.toc_heading = clean_title(str(draft.toc_raw.get("heading") or "Sadržaj"))
        draft.toc_style = str(draft.toc_raw.get("style") or "numbered").strip().lower()
        draft.toc_include_back_matter = bool(draft.toc_raw.get("include_back_matter"))

    if draft.toc_action not in TOC_ACTIONS:
        draft.fail(
            f"toc.action must be one of {list(TOC_ACTIONS)}, got {draft.toc_action!r}"
        )
    if not draft.toc_heading:
        draft.fail("toc.heading cannot be empty")
    if len(draft.toc_heading) > 80:
        draft.fail("toc.heading is longer than 80 characters")
    if draft.toc_style not in TOC_STYLES:
        draft.fail(f"toc.style must be one of {list(TOC_STYLES)}, got {draft.toc_style!r}")


def check_toc_state(draft: BookPlanDraft, entries: list[dict[str, Any]]) -> None:
    """Compare the chosen action with the TOC the book actually has."""
    action = draft.toc_action
    toc = draft.book.get("toc") or {}
    if action == "create" and toc.get("exists"):
        draft.fail(
            "a table of contents already exists; use toc.action='update' (or 'keep' if it "
            "already matches the chapter titles)"
        )
    if action == "update" and not toc.get("exists"):
        draft.fail("no table of contents exists; use toc.action='create'")
    if action == "keep":
        if not toc.get("exists"):
            draft.fail("no table of contents exists; use toc.action='create'")
        elif not toc_matches_entries(toc, entries):
            draft.fail(
                "the existing table of contents does not match the chapter titles; use "
                "toc.action='update'"
            )


__all__ = ["check_toc_plan", "check_toc_state", "default_toc_plan"]
