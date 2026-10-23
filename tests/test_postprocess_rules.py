"""Direct tests for the stage-9 validation rule groups.

The public validators are covered by ``test_postprocess.py`` and by the frozen
ordering test in ``test_postprocess_contract.py``. These tests exercise one rule
group at a time, so a future change to a single rule fails with the rule's name
instead of a wall of agent-facing text.
"""

from __future__ import annotations

from pathlib import Path

from book_translator.postprocess.inspect import scan_chapter_structure
from book_translator.postprocess.plan import normalise_chapter_plan
from book_translator.postprocess.rules import (
    BookPlanDraft,
    ChapterPlanDraft,
    check_evidence,
    check_naming,
    check_overrides,
    check_titles,
    check_toc_plan,
    check_toc_state,
)
from book_translator.postprocess.rules.chapter_headings import check_chapter_headings
from book_translator.postprocess.rules.chapter_title import check_chapter_title

BOOK = {
    "chapters": [
        {"filename": "01-a.md", "matter": "body", "title": "Poglavlje 1: Dolazak"},
        {"filename": "99-back.md", "matter": "back", "title": "Beleške"},
    ],
    "body_chapter_count": 1,
    "toc": {"exists": False},
}
STRUCTURES = {
    "01-a.md": {"filename": "01-a.md", "matter": "body"},
    "99-back.md": {"filename": "99-back.md", "matter": "back"},
}


def _book_draft(plan: dict) -> BookPlanDraft:
    return BookPlanDraft.from_plan(plan, book=BOOK, structures=STRUCTURES)


def test_naming_rule_auto_enables_include_title_and_validates_the_policy() -> None:
    draft = _book_draft(
        {"naming": {"mode": "normalize", "label": "", "separator": "long!!", "level": "x"}}
    )

    check_naming(draft)

    assert draft.include_title is True
    assert draft.level == 1
    assert draft.issues == [
        "naming.label is required when naming.mode is 'normalize'",
        "naming.separator must be at most 5 characters and one line",
        "naming.level must be an integer between 1 and 3",
    ]
    assert draft.warnings and "include_title" in draft.warnings[0]


def test_evidence_rule_warns_before_it_requires_web_search() -> None:
    draft = _book_draft({"title_source": "vibes"})

    check_evidence(draft, require_web_search=True)

    assert draft.title_source == ""
    assert draft.issues and "title_evidence is required" in draft.issues[0]
    assert draft.warnings and "title_source" in draft.warnings[0]


def test_overrides_rule_rejects_unknown_empty_and_long_targets() -> None:
    draft = _book_draft(
        {"naming": {"overrides": {"missing.md": "X", "01-a.md": ""}}}
    )
    # normalise_overrides() caps a title at the limit, so exercise the guard
    # directly rather than through a value it would have shortened.
    draft.overrides["99-back.md"] = "y" * 200

    check_overrides(draft)

    assert draft.issues[0] == "naming override targets an unknown chapter: 'missing.md'"
    assert draft.issues[1] == "naming override for '01-a.md' is empty"
    assert draft.issues[2] == "naming override for '99-back.md' is too long"


def test_toc_rule_supplies_the_default_and_rejects_a_bad_shape() -> None:
    defaulted = _book_draft({})
    check_toc_plan(defaulted)
    assert defaulted.toc_action == "create"
    assert defaulted.warnings == [
        "no toc plan was submitted; the deterministic default was used"
    ]

    broken = _book_draft({"toc": {"action": "bogus", "heading": " "}})
    check_toc_plan(broken)
    assert broken.issues == [
        "toc.action must be one of ['create', 'update', 'keep'], got 'bogus'",
        "toc.heading cannot be empty",
    ]


def test_toc_state_compares_the_action_with_the_real_book() -> None:
    draft = _book_draft({"toc": {"action": "keep"}})
    check_toc_plan(draft)

    check_toc_state(draft, [])

    assert draft.issues == ["no table of contents exists; use toc.action='create'"]


def test_titles_rule_is_silent_when_another_rule_already_failed() -> None:
    draft = _book_draft({})
    draft.fail("something else")

    check_titles(draft, {"a.md": ""})

    assert draft.issues == ["something else"]


def test_titles_rule_warns_about_duplicate_body_titles() -> None:
    draft = _book_draft({})
    draft.structures["01-a.md"] = {"filename": "01-a.md", "matter": "body"}
    draft.structures["02-b.md"] = {"filename": "02-b.md", "matter": "body"}

    check_titles(draft, {"01-a.md": "Isto", "02-b.md": "isto"})

    assert draft.issues == []
    assert draft.warnings == [
        "chapters '01-a.md' and '02-b.md' would share the title 'isto'"
    ]


def _structure(tmp_path: Path):
    chapter = tmp_path / "01-chapter.md"
    chapter.write_text("## GLAVA 1\n\nTekst.\n\n## 12\n\n## SEKS\n", encoding="utf-8")
    return scan_chapter_structure(chapter), chapter


def test_chapter_title_rule_reports_the_target_line(tmp_path: Path) -> None:
    structure, chapter = _structure(tmp_path)
    draft = ChapterPlanDraft.from_plan(
        normalise_chapter_plan(
            {"title": {"action": "set", "line_number": 99, "text": "Novo"}}
        ),
        structure=structure,
        lines=chapter.read_text(encoding="utf-8").splitlines(),
        canonical_title="",
    )

    check_chapter_title(draft)

    assert draft.issues == ["title line 99 is out of range"]
    assert draft.title_text == "Novo"


def test_chapter_headings_rule_warns_when_a_suspicious_heading_is_kept(
    tmp_path: Path,
) -> None:
    structure, chapter = _structure(tmp_path)
    draft = ChapterPlanDraft.from_plan(
        normalise_chapter_plan(
            {"title": {"action": "keep"}, "headings": [{"line_number": 5, "decision": "keep"}]}
        ),
        structure=structure,
        lines=chapter.read_text(encoding="utf-8").splitlines(),
        canonical_title="",
    )

    check_chapter_headings(draft)

    assert draft.issues == []
    assert draft.warnings == [
        "heading #1 (line 5): the inspector suggested remove but the plan keeps it"
    ]


def test_chapter_headings_rule_warns_about_unreviewed_suspicious_headings(
    tmp_path: Path,
) -> None:
    structure, chapter = _structure(tmp_path)
    draft = ChapterPlanDraft.from_plan(
        normalise_chapter_plan({"title": {"action": "keep"}, "headings": []}),
        structure=structure,
        lines=chapter.read_text(encoding="utf-8").splitlines(),
        canonical_title="",
    )

    check_chapter_headings(draft)

    assert draft.issues == []
    assert draft.warnings == [
        "1 suspicious heading(s) were not reviewed and will be kept as-is: 5"
    ]
