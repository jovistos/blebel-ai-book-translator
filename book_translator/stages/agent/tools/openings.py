"""Stage-2 and stage-9 tools: the chapter-opening audit.

The deterministic final phase (``book_translator.stages.openings``) runs at the end of
both stages, but the agents need the same evidence *before* they decide: stage 2
wants to know whether the opening it is cleaning still has a page number under
the title, and stage 9's book planner wants to know which chapters have a title
that is only a number.

Two read-only tools, nothing else:

``audit_chapter_opening``
    one chapter: title, bounded opening lines, provable issues, safe fixes.
    Available to both stage-2 and stage-9 agents.
``audit_book_openings``
    the same over every chapter of a folder, cleaned chapters collapsed to a
    count. Book-level (stage 9's naming planner), because a per-chapter stage
    must not pull the whole book into its context.

Neither tool ever returns a paragraph: :data:`MAX_OPENING_AUDIT_LINES` clipped
lines per chapter, and one call expands at most
:data:`MAX_OPENING_AUDIT_CHAPTERS` chapters. The write side stays with the
stage's deterministic apply/verify pair and the final phase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from langchain_core.tools import InjectedToolArg, tool

from book_translator.core.constants import MAX_OPENING_AUDIT_CHAPTERS
from book_translator.core.errors import ToolError
from book_translator.documents.text import jdump
from book_translator.stages.openings import audit_openings, chapter_files_in, inspect_opening
from book_translator.toolkit.context import ToolContext

from .registry import postprocess_stage, preprocess_stage


@postprocess_stage
@preprocess_stage
@tool
def audit_chapter_opening(
    chapter_path: str,
    *,
    ctx: Annotated[ToolContext | None, InjectedToolArg] = None,
) -> str:
    """Audit ONE chapter's first lines: is the opening clean?

    Returns the title candidate (text, line, whether it is already a `#` heading,
    and whether it is only a number), the chapter number, at most 12 clipped
    opening lines, and every provable opening issue:

    `stray_number_after_title` / `stray_number_before_title` — a line that is
    nothing but a number next to the title (`# Poglavlje 2` then `2`); this is
    junk and the graph's final phase removes it.
    `duplicate_title` — a line that repeats the title.
    `bare_number_title` — the title is only a number (`## 2`).
    `label_only_title` — the title is only `Poglavlje` / `Chapter`.
    `missing_title` — the chapter opens on prose.
    `empty_chapter` — the file has no text.

    Pass the exact `chapter_path` from the CHAPTER REPORT. Never use this tool to
    read a chapter: it returns the opening only, capped hard.
    """
    from book_translator.toolkit.postprocess import resolve_chapter_path

    fallback = ToolContext(
        chapters_path=str(Path(chapter_path).expanduser().parent)
    )
    path = resolve_chapter_path(chapter_path, ctx or fallback)
    audit = inspect_opening(path)
    return jdump(
        {
            "chapter_path": str(path),
            "file": audit.filename,
            "title": {
                "text": audit.title,
                "line_number": audit.title_line,
                "is_heading": audit.title_is_heading,
                "source": audit.title_source,
            },
            "chapter_number": audit.chapter_number,
            "title_number": audit.title_number,
            "ok": audit.ok,
            "issues": [issue.to_dict() for issue in audit.issues],
            "safe_fixes": audit.fixes(),
            "opening": audit.lines,
            "note": "OPENING ONLY - clipped lines, capped length. Not the chapter text.",
        }
    )


@postprocess_stage
@tool
def audit_book_openings(
    chapters_path: str,
    include_clean: bool = False,
    max_chapters: int = MAX_OPENING_AUDIT_CHAPTERS,
) -> str:
    """Audit every chapter's opening in the book, in reading order.

    Returns the book-level counts (chapters, clean, issues by kind, safe fixes)
    and one compact record per chapter that has a finding: its title, the issues
    and the lines that the final phase would remove. Clean chapters are only
    counted unless `include_clean` is true.

    Use it before choosing the naming policy: a chapter whose title is only a
    number (`## 2`) needs `set` with the canonical title, not `insert` plus a
    demoted copy of the number. At most `max_chapters` chapter records are
    expanded; the counts always cover the whole book, and the opening is at most
    12 clipped lines per chapter.
    """
    folder = Path(str(chapters_path or "")).expanduser()
    if not folder.is_dir():
        raise ToolError(f"chapters folder not found: {chapters_path}")
    files = chapter_files_in(folder)
    report = audit_openings(
        folder,
        [path.name for path in files],
        dry_run=True,
        keep_clean=True,
    )
    limit = max(1, int(max_chapters))
    expanded: list[dict[str, Any]] = []
    omitted = 0
    for chapter in report.chapters:
        if not include_clean and chapter.get("ok"):
            continue
        if len(expanded) >= limit:
            omitted += 1
            continue
        expanded.append(chapter)
    return jdump(
        {
            "chapters_path": str(folder),
            "chapter_count": report.chapter_count,
            "inspected_count": report.inspected_count,
            "clean_count": report.clean_count,
            "issue_count": report.issue_count,
            "fixable_count": report.fixable_count,
            "counts": report.counts,
            "chapters": expanded,
            "chapters_omitted": omitted,
            "missing": report.missing,
            "note": "OPENINGS ONLY - the final phase removes the safe leftovers "
            "after the chapter plans are written.",
        }
    )


__all__ = ["audit_book_openings", "audit_chapter_opening"]
