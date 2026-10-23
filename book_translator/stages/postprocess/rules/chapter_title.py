"""Rule group: one chapter's title decision.

The title has three actions — ``keep`` (the existing line is the title),
``set`` (replace it) and ``insert`` (there is no title line yet). The rule checks
the action, the text and the target line, and never rewrites anything: the
deterministic applier in :mod:`book_translator.stages.postprocess.plan` does that.
"""

from __future__ import annotations

from book_translator.core.constants import (
    DEFAULT_CHAPTER_TITLE_LEVEL,
    MAX_POSTPROCESS_TITLE_CHARS,
    MAX_TITLE_SCAN_LINES,
    TITLE_ACTIONS,
)
from book_translator.documents.text import norm

from ..titles import clean_title
from .base import ChapterPlanDraft


def check_chapter_title(draft: ChapterPlanDraft) -> None:
    structure = draft.structure
    title = draft.title
    action = draft.title_action
    text = draft.title_text
    title_line = structure.title_line

    level = title.get("level") or structure.title_level or DEFAULT_CHAPTER_TITLE_LEVEL
    if action not in TITLE_ACTIONS:
        draft.fail(f"title.action must be one of {list(TITLE_ACTIONS)}, got {action!r}")
    try:
        level = int(level)
    except (TypeError, ValueError):
        draft.fail("title.level must be an integer")
        level = DEFAULT_CHAPTER_TITLE_LEVEL
    if not 1 <= level <= 6:
        draft.fail("title.level must be between 1 and 6")

    if draft.canonical_title:
        expected = clean_title(draft.canonical_title)
        if text and norm(text) != norm(expected):
            draft.fail(
                f"this chapter's title is fixed by the book plan as {expected!r}; "
                f"the plan says {text!r}"
            )
        text = expected or text

    if action in {"set", "insert"} and not text:
        draft.fail("title.text cannot be empty")
    if len(text) > MAX_POSTPROCESS_TITLE_CHARS:
        draft.fail(f"title.text is longer than {MAX_POSTPROCESS_TITLE_CHARS} characters")
    if "\n" in text:
        draft.fail("title.text must be one line")

    if action == "set":
        line_number = title.get("line_number") or title_line
        if line_number is None:
            draft.fail("title.action='set' needs a line_number; this chapter has no title")
        elif line_number < 1 or line_number > len(draft.lines):
            draft.fail(f"title line {line_number} is out of range")
        elif line_number > MAX_TITLE_SCAN_LINES:
            draft.fail(
                f"title line {line_number} is outside the first {MAX_TITLE_SCAN_LINES} lines"
            )
        elif title_line is not None and line_number != title_line:
            draft.fail(
                f"line {line_number} is not the chapter title; the title candidate is "
                f"line {title_line}"
            )
        elif title_line is None:
            draft.fail(
                "this chapter has no title candidate; use action='insert' instead of "
                "promoting an arbitrary line"
            )
    elif action == "insert":
        if title_line is not None:
            draft.fail(
                f"this chapter already has a title at line {title_line}; use action='set' "
                "instead of inserting a second heading"
            )
    elif action == "keep":
        if title_line is None:
            draft.fail("this chapter has no title; use action='insert'")
        elif not structure.title_is_heading:
            draft.fail(
                "the existing title is not a Markdown heading; use action='set' to add the "
                "'#' markup"
            )
        elif draft.canonical_title and norm(
            clean_title(structure.title)
        ) != norm(clean_title(draft.canonical_title)):
            draft.fail(
                "the existing title does not match the book plan; use action='set' with the "
                "canonical title"
            )

    draft.title_text = text
    draft.title_level = level


__all__ = ["check_chapter_title"]
