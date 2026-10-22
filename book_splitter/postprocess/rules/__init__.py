"""Stage-9 validation, split into rule groups.

Each group owns one domain question and appends to a draft:

=============  ============================================================
group           question
=============  ============================================================
``evidence``    where did the title facts come from?
``naming``      what policy names the chapters (and are they nameable)?
``overrides``   do the per-chapter exceptions target real, usable titles?
``toc``         which TOC action, and does it match the book?
``titles``      are the derived canonical titles usable and unique?
``chapter_title``    is the chapter's own title decision legal?
``chapter_headings`` is every other heading decision legal?
=============  ============================================================

The call order below is the order the agent sees the messages in; it mirrors the
historical single-function validator exactly.
"""

from __future__ import annotations

from .base import BookPlanDraft, ChapterPlanDraft, Validation
from .chapter_headings import check_chapter_headings
from .chapter_title import check_chapter_title
from .evidence import check_evidence
from .naming import check_body_chapters, check_naming
from .overrides import check_overrides
from .titles import check_titles
from .toc import check_toc_plan, check_toc_state


def run_book_rules(
    draft: BookPlanDraft, *, require_web_search: bool = False
) -> BookPlanDraft:
    """Run the book-plan rules that need no derived titles."""
    check_evidence(draft, require_web_search=require_web_search)
    check_naming(draft)
    check_body_chapters(draft)
    check_overrides(draft)
    check_toc_plan(draft)
    return draft


def run_chapter_rules(draft: ChapterPlanDraft) -> ChapterPlanDraft:
    """Run one chapter's title and heading rules."""
    check_chapter_title(draft)
    check_chapter_headings(draft)
    return draft


__all__ = [
    "BookPlanDraft",
    "ChapterPlanDraft",
    "Validation",
    "check_chapter_headings",
    "check_chapter_title",
    "check_evidence",
    "check_naming",
    "check_overrides",
    "check_titles",
    "check_toc_plan",
    "check_toc_state",
    "run_book_rules",
    "run_chapter_rules",
]
