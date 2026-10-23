"""Regex line search: the agent's way back when the candidate list is truncated.

``extract_heuristic_candidates`` returns *title-looking* lines, capped by count and
by the tool-result budget. In a book that lost all of its markup that list is
mostly prose noise — dialogue, short standalone sentences — so the handful of real
chapter markers can fall off the end of it. The agent then sees the first third of
the book and cannot know what it is missing.

This module gives the model one more move: hand it a pattern and get back every
line that matches, with the flags that decide whether a line may be a split
point — ``standalone``/``title_like`` for a heading or a title line, and
``paragraph_start``/``prose_start_issue`` for a chapter whose heading the
conversion dropped (see :mod:`book_translator.analysis.prose`). It stays a
*search*, never a reader: each match is clipped to a short excerpt, the number of
matches is capped, how much of each line is examined is capped, and the run has a
wall-clock budget.
"""

from __future__ import annotations

import re
import time
from typing import Any

from ..core.constants import (
    MAX_PATTERN_CHARS,
    MAX_PATTERN_LINE_CHARS,
    MAX_PATTERN_MATCHES,
    PATTERN_TIME_BUDGET_SECONDS,
)
from ..documents.markdown import is_page_number_line, is_standalone_line, looks_like_title
from ..documents.text import excerpt
from .prose import is_paragraph_start, prose_start_issue

#: Excerpt length per match — the same ceiling the other tools use.
MATCH_EXCERPT_CHARS = 90


def search_lines(
    lines: list[str],
    pattern: str,
    *,
    ignore_case: bool = False,
    max_matches: int = MAX_PATTERN_MATCHES,
    max_line_chars: int = MAX_PATTERN_LINE_CHARS,
    time_budget: float = PATTERN_TIME_BUDGET_SECONDS,
) -> dict[str, Any]:
    """Every line matching ``pattern``, clipped, flagged and capped.

    ``pattern`` is a Python regular expression searched per line, so ``^`` anchors
    it to the start of the line. Each match carries the flags that matter for
    splitting:

    * ``standalone`` (blank lines around it) and ``title_like`` (short enough to
      be a heading) — both true makes the line usable as an ordinary split point;
    * ``paragraph_start`` (nothing above it, so it can open a paragraph) and
      ``prose_start_issue`` (``None`` when the line may be used with
      ``role: "prose-chapter"``; otherwise a short code — ``blank``,
      ``mid-paragraph``, ``page-number``, ``too-short`` — saying why it may not).

    :func:`~book_translator.slicing.validation.validate_split_map` enforces exactly
    those two shapes.

    Returns ``{"error", "hint"}`` instead of raising when the pattern is empty,
    over-long or not a valid regular expression.
    """
    text = str(pattern or "").strip()
    if not text:
        return {
            "error": "the pattern is empty",
            "hint": r"pass a regular expression, for example ^Chapter \d+$",
        }
    if len(text) > MAX_PATTERN_CHARS:
        return {
            "error": f"the pattern is longer than {MAX_PATTERN_CHARS} characters",
            "hint": "search for one marker shape at a time",
        }
    try:
        compiled = re.compile(text, re.IGNORECASE if ignore_case else 0)
    except re.error as exc:
        return {
            "error": f"invalid regular expression: {exc}",
            "hint": r"escape special characters, for example ^Chapter \d+$",
        }

    limit = max(1, int(max_matches))
    window = max(1, int(max_line_chars))
    matches: list[dict[str, Any]] = []
    match_count = 0
    timed_out = False
    started = time.monotonic()

    for index, raw in enumerate(lines):
        if not compiled.search(raw[:window]):
            continue
        match_count += 1
        if len(matches) < limit:
            clean = raw.strip()
            matches.append(
                {
                    "line_number": index + 1,
                    "text": excerpt(clean, MATCH_EXCERPT_CHARS),
                    "standalone": is_standalone_line(lines, index),
                    "title_like": looks_like_title(raw),
                    "is_page_number": is_page_number_line(clean),
                    "paragraph_start": is_paragraph_start(lines, index),
                    "prose_start_issue": prose_start_issue(lines, index),
                }
            )
        if time.monotonic() - started > time_budget:
            timed_out = True
            break

    report: dict[str, Any] = {
        "pattern": text,
        "ignore_case": bool(ignore_case),
        "line_count": len(lines),
        "match_count": match_count,
        "returned": len(matches),
        "truncated": match_count > len(matches),
        "timed_out": timed_out,
        "valid_split_points": sum(
            1 for match in matches if match["standalone"] and match["title_like"]
        ),
        "usable_prose_start_points": sum(
            1 for match in matches if match["prose_start_issue"] is None
        ),
        "matches": matches,
    }
    if timed_out:
        report["hint"] = "the search hit its time budget; narrow the pattern"
    elif match_count > 10 * limit:
        report["hint"] = (
            f"{match_count} lines match; the pattern may be too broad — narrow it or raise "
            "max_matches"
        )
    elif report["truncated"]:
        report["hint"] = "only the first matches are shown; narrow the pattern or raise max_matches"
    return report


__all__ = ["MATCH_EXCERPT_CHARS", "search_lines"]
