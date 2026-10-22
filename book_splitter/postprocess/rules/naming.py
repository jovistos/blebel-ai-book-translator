"""Rule group: the book-wide naming policy.

Covers the policy's own shape (mode, label, separator, heading level), the
deterministic auto-enable of ``include_title`` when body chapters carry names,
and the precondition that there are body chapters to name at all.
"""

from __future__ import annotations

from ...core.constants import DEFAULT_CHAPTER_TITLE_LEVEL, NAMING_MODES
from ..titles import title_part_of
from .base import BookPlanDraft


def check_naming(draft: BookPlanDraft) -> None:
    """Validate and normalise ``plan["naming"]``."""
    mode = draft.mode
    # A book whose chapters carry names must not lose them just because the plan
    # asked for bare labels: the names are already in the chapter opening.
    named_chapters = [
        str(summary.get("filename"))
        for summary in draft.book.get("chapters") or []
        if summary.get("matter") == "body" and title_part_of(summary)
    ]
    if mode == "normalize" and named_chapters and not draft.include_title:
        draft.include_title = True
        draft.warn(
            f"naming.include_title was false, but {len(named_chapters)} body chapter(s) "
            "carry their own title words; it was enabled so the chapter headings and "
            "the table of contents keep those names"
        )

    if mode not in NAMING_MODES:
        draft.fail(f"naming.mode must be one of {list(NAMING_MODES)}, got {mode!r}")
    if mode == "normalize" and not draft.label:
        draft.fail("naming.label is required when naming.mode is 'normalize'")
    if len(draft.label) > 40:
        draft.fail("naming.label is longer than 40 characters")
    if "\n" in draft.separator or len(draft.separator) > 5:
        draft.fail("naming.separator must be at most 5 characters and one line")

    level = draft.naming_raw.get("level", DEFAULT_CHAPTER_TITLE_LEVEL)
    try:
        level = int(level)
    except (TypeError, ValueError):
        draft.fail("naming.level must be an integer between 1 and 3")
        level = DEFAULT_CHAPTER_TITLE_LEVEL
    if not 1 <= level <= 3:
        draft.fail("naming.level must be between 1 and 3")
    draft.level = level


def check_body_chapters(draft: BookPlanDraft) -> None:
    """A book plan needs at least one body chapter to act on."""
    if not draft.book.get("body_chapter_count"):
        draft.fail("the folder contains no body chapters to postprocess")


__all__ = ["check_body_chapters", "check_naming"]
