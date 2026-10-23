"""The pattern-search tool: the model writes the regex, the tool stays bounded."""

from __future__ import annotations

from langchain_core.tools import tool

from ...analysis.search import search_lines
from ...core.constants import DEFAULT_MAX_PATTERN_MATCHES
from ...documents.io import load_lines
from ...documents.text import jdump
from .registry import split_stage


@split_stage
@tool
def find_lines_matching(
    file_path: str,
    pattern: str,
    ignore_case: bool = False,
    max_matches: int = DEFAULT_MAX_PATTERN_MATCHES,
) -> str:
    r"""Find every line of the book that matches a regular expression YOU write.

    Use this when `extract_heuristic_candidates` is truncated or too noisy and
    the book's chapter markers follow one shape: `^Chapter \d+$`, `^CHAPTER
    [IVX]+\b`, `^Part \b`, `^POGLAVLJE \d+`, `^GLAVA \d+`, `^\d+\.?$`. One call
    then returns the whole spine of the book, instead of the first few hundred
    title-looking lines.

    `pattern` is a Python regular expression matched against each line with
    `re.search`, so use `^` to anchor it to the start of a line. The search is
    bounded on every side: at most the first 2000 characters of a line are
    examined, each match is clipped to a 90-character excerpt, only `max_matches`
    (default 200) matches come back, and the scan has a short time budget. It is
    a search, never a way to read the book's prose.

    Returns `{"pattern", "match_count", "returned", "truncated", "timed_out",
    "valid_split_points", "usable_prose_start_points", "matches":
    [{"line_number", "text", "standalone", "title_like", "is_page_number",
    "paragraph_start", "prose_start_issue"}]}`.

    A match may become a split point in one of two ways:

    - `standalone` and `title_like` are both true: an ordinary heading/title
      split point;
    - `prose_start_issue` is `null`: the line opens a paragraph and may be
      proposed with `role: "prose-chapter"` (plus an `evidence` field), which is
      how you cut at a chapter whose heading the conversion dropped entirely.

    Filter on those flags instead of guessing, and read `hint` when the result
    was capped.
    """
    return jdump(
        search_lines(
            load_lines(file_path),
            pattern,
            ignore_case=ignore_case,
            max_matches=max_matches,
        )
    )


__all__ = ["find_lines_matching"]
