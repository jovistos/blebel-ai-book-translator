"""The stage-9 validator's contract, frozen before validation is split by rule.

The validator's return value is handed to the agent through ``jdump`` and stored
in the agent trace, so both the message text and the order in which the rules
append issues/warnings are observable artifacts. These tests pin one multi-error
book plan, one warning-only book plan and one multi-error chapter plan.
"""

from __future__ import annotations

from pathlib import Path

from book_translator.stages.postprocess.inspect import scan_chapter_structure
from book_translator.stages.postprocess.plan import validate_book_plan, validate_chapter_plan

BOOK = {
    "chapters": [
        {"filename": "00-front-matter.md", "matter": "front", "title": "Predgovor"},
        {
            "filename": "01-chapter-1.md",
            "matter": "body",
            "title": "Poglavlje 1: Dolazak",
            "chapter_number": 1,
        },
        {
            "filename": "02-chapter-2.md",
            "matter": "body",
            "title": "Poglavlje 2",
            "chapter_number": 2,
        },
    ],
    "body_chapter_count": 2,
    "toc": {"exists": False},
}


def test_book_plan_issues_are_reported_in_rule_order() -> None:
    plan = {
        "naming": {
            "mode": "bogus",
            "label": "",
            "separator": "------",
            "level": "x",
            "overrides": {"nope.md": "X"},
        },
        "toc": {"action": "bogus", "heading": "", "style": "bogus"},
    }

    result = validate_book_plan("unused", plan, book=BOOK, require_web_search=True)

    assert result["issues"] == [
        "title_evidence is required: record the URL(s) you read from the published "
        "chapter list, or the failure/no-result outcome of the mandatory web search",
        "naming.mode must be one of ['keep', 'normalize'], got 'bogus'",
        "naming.separator must be at most 5 characters and one line",
        "naming.level must be an integer between 1 and 3",
        "naming override targets an unknown chapter: 'nope.md'",
        "toc.action must be one of ['create', 'update', 'keep'], got 'bogus'",
        "toc.style must be one of ['numbered', 'bullets'], got 'bogus'",
    ]
    assert result["warnings"] == []
    assert list(result) == [
        "ok",
        "issues",
        "warnings",
        "naming",
        "toc",
        "title_evidence",
        "title_source",
        "canonical_titles",
        "toc_entries",
        "chapter_count",
        "body_chapter_count",
    ]


def test_book_plan_include_title_warning_precedes_the_label_issue() -> None:
    plan = {"naming": {"mode": "normalize", "label": ""}, "toc": {"action": "keep"}}

    result = validate_book_plan("unused", plan, book=BOOK)

    assert result["warnings"] == [
        "naming.include_title was false, but 1 body chapter(s) carry their own title "
        "words; it was enabled so the chapter headings and the table of contents keep "
        "those names"
    ]
    assert result["issues"] == [
        "naming.label is required when naming.mode is 'normalize'",
        "no table of contents exists; use toc.action='create'",
    ]


def test_chapter_plan_issues_are_reported_in_rule_order(tmp_path: Path) -> None:
    chapter = tmp_path / "01-chapter.md"
    chapter.write_text("## GLAVA 1\n\nTekst.\n\n## SEKS\n", encoding="utf-8")
    structure = scan_chapter_structure(chapter)
    plan = {
        "title": {"action": "bogus", "level": "x"},
        "headings": [
            {"line_number": 1, "decision": "demote"},
            {"line_number": 99, "decision": "keep"},
            {"line_number": 3, "decision": "keep"},
            {"line_number": 5, "decision": "bogus"},
        ],
    }

    result = validate_chapter_plan(chapter, plan, structure=structure)

    assert result["issues"] == [
        "title.action must be one of ['keep', 'set', 'insert'], got 'bogus'",
        "heading #1 (line 1): the chapter title is controlled by title.action, not headings",
        "heading #2 (line 3): not a heading or a plain ALL-CAPS heading candidate",
        "heading #3 (line 5): decision must be one of ['keep', 'demote', 'promote']",
        "heading #4 (line 99): out of range (chapter has 5 lines)",
    ]
    assert result["warnings"] == []
    assert list(result) == [
        "ok",
        "issues",
        "warnings",
        "plan",
        "canonical_title",
        "chapter",
    ]
