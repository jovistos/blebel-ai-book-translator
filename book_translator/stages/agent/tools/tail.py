"""The tail reader: is there back matter at the end of the book, and where?

Stage 1 cuts at chapter starts, so the last chapter file swallows whatever
follows the story — the author biography, the publisher's catalogue of the
author's other books, a re-printed table of contents, a teaser chapter. Those
lines are not the book, and a translation pipeline should not treat them as if
they were. This tool looks at the end of the book, reports what it finds, and
returns a ready-to-submit back-matter split point.

It is the second tool allowed to return prose (after `extract_front_matter`),
and it is bounded three times over: the window may not exceed
:data:`MAX_TAIL_CHUNK_LINES` lines, every line is clipped, and the whole window
is capped in characters. The window is built from the end backwards, so the true
tail is never the part that gets truncated.
"""

from __future__ import annotations

from langchain_core.tools import tool

from book_translator.core.constants import DEFAULT_TAIL_CHUNK_LINES
from book_translator.documents.io import load_lines
from book_translator.documents.text import jdump
from book_translator.stages.analysis.backmatter import detect_back_matter

from .registry import split_stage


@split_stage
@tool
def inspect_book_tail(
    file_path: str,
    chunk_lines: int = DEFAULT_TAIL_CHUNK_LINES,
    next_index: int = 0,
) -> str:
    """Look at the END of the book to decide whether it has back matter, and where it starts.

    Call this once your chapter spine is known, and **start with the default
    short window** (`chunk_lines=40`). The window is the last `chunk_lines` lines
    of the book — that is the tail of the last chapter — with each line clipped.

    Escalating the window (this is the point of the `chunk_lines` argument):

    1. Call with the default (or a small value) first.
    2. If the result says `escalate: true`, the window is not decisive: it either
       cuts through the block or shows no back matter while lines remain above.
       Call this tool AGAIN with `chunk_lines` set to `next_chunk_lines` from the
       result (each escalation doubles the window; the cap is 300 lines).
    3. Stop as soon as `back_matter.boundary_visible` is true, or the cap is hit.

    Reading the answer:

    * `back_matter.detected` — back matter of some kind is at the end.
    * `back_matter.boundary_visible` — its FIRST line is inside the window, so
      `back_matter.start_line` is a real boundary, not a truncation artefact.
    * `proposed_split` — when a boundary is visible, a ready map entry:
      `{"line_number", "filename", "role": "back-matter", "title"}`. Put it in
      `split_mapping` as the LAST entry of your plan. It is the one split point
      that does not have to look like a chapter title; validation accepts it only
      as the last entry, only for `role: "back-matter"`, and only when the
      trailing block is a plausible length.
    * `candidates.lines` — every evidence line with its signals, if you want to
      check the boundary yourself.

    `next_index` is the number of your last chapter plus one (`22` for a 21
    chapter book); it names the file `NN-back-matter.md`. With `next_index=0` the
    file is called `back-matter.md`. If `detected` stays false, the book has no
    back matter — do not invent a back-matter split.
    """
    lines = load_lines(file_path)
    return jdump(
        detect_back_matter(
            lines,
            chunk_lines=chunk_lines,
            next_index=next_index,
            file_label=str(file_path),
        )
    )


__all__ = ["inspect_book_tail"]
