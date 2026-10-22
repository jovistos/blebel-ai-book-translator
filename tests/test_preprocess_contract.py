"""The stage-2 proof's external contract, frozen before the module split.

Everything a preprocessing tool returns is serialised with ``jdump`` into an
agent prompt and then stored in ``_agent_trace.json`` (hashed byte for byte by
the artifact-tree snapshot). The *key order* of these dicts is therefore part of
the artifact contract, not a formatting detail. These tests pin it together with
the provenance entry emitted for every edit kind, so the later split into
``plan`` / ``apply`` / ``verify`` can be proved behaviour-preserving.
"""

from __future__ import annotations

from pathlib import Path

from book_splitter.preprocess.apply import apply_cleaning_plan, text_of
from book_splitter.preprocess.plan import validate_cleaning_plan
from book_splitter.preprocess.verify import verify_cleaning_plan

LONG_LEFT = "x" * 200 + " i rije"

CHAPTER_LINES = [
    "## GLAVA 1",  # 1
    "",  # 2
    "## 12",  # 3  a page-number header
    "",  # 4
    "Slika 4. Blokada ljubavi.",  # 5  a standalone caption
    "",  # 6
    "TEKST KOJI JE NASLOV",  # 7  plain ALL-CAPS heading candidate
    "",  # 8
    "## 75 lič nosti. Impuls",  # 9  header with a printed page number
    "",  # 10
    "## GLAVA 2",  # 11 split heading, first half
    "",  # 12
    "## SEKS I SRCE",  # 13 split heading, second half
    "",  # 14
    LONG_LEFT,  # 15 split word, first half
    "č je bila.",  # 16 split word, second half
    "",  # 17
    "Kraj.",  # 18
]

OPERATIONS = [
    {"line_number": 3, "decision": "remove", "note": "page 12"},
    {"line_number": 5, "decision": "remove_caption", "note": "figure label"},
    {"line_number": 7, "decision": "promote", "note": "restore heading"},
    {"line_number": 9, "decision": "demote", "note": "not a heading"},
    {"line_number": 11, "decision": "merge_with_next", "note": "split heading"},
]
TEXT_FIXES = [{"line_number": 15, "fix": "split_word"}]


def _chapter(tmp_path: Path) -> Path:
    path = tmp_path / "chapter.md"
    path.write_text("".join(line + "\n" for line in CHAPTER_LINES), encoding="utf-8")
    return path


def test_a_mixed_plan_validates_and_verifies(tmp_path: Path) -> None:
    path = _chapter(tmp_path)

    validation = validate_cleaning_plan(path, OPERATIONS)
    assert validation["ok"], validation["issues"]
    assert validation["warnings"] == []

    verification = verify_cleaning_plan(path, OPERATIONS, TEXT_FIXES)
    assert verification["ok"], verification["problems"]


def test_the_provenance_of_every_edit_kind_is_stable(tmp_path: Path) -> None:
    path = _chapter(tmp_path)

    result = apply_cleaning_plan(path, OPERATIONS, TEXT_FIXES)

    assert [(entry.origin, entry.source_lines) for entry in result.provenance] == [
        ("kept", [1]),
        ("kept", [2]),
        ("removed_page_number", [3]),
        ("kept", [4]),
        ("removed_caption", [5]),
        ("kept", [6]),
        ("promoted_heading", [7]),
        ("kept", [8]),
        ("demoted_header", [9]),
        ("kept", [10]),
        ("merged_pair", [11, 12, 13]),
        ("kept", [12]),
        ("kept", [14]),
        ("joined_split_word", [15, 16]),
        ("kept", [17]),
        ("kept", [18]),
    ]
    assert result.lines[2] == ""
    assert result.lines[4] == ""
    assert result.lines[6] == "## TEKST KOJI JE NASLOV"
    assert result.lines[8] == "lič nosti. Impuls"
    assert result.lines[10] == "## GLAVA 2 SEKS I SRCE"
    assert result.lines[13].endswith("riječ je bila.")
    assert text_of(result).endswith("Kraj.\n")


def test_the_apply_summary_key_order_is_stable(tmp_path: Path) -> None:
    path = _chapter(tmp_path)
    result = apply_cleaning_plan(path, OPERATIONS, TEXT_FIXES)

    summary = result.summary()
    assert list(summary) == [
        "output_lines",
        "provenance_counts",
        "removed_artefacts",
        "removed_captions",
        "promoted_headings",
        "copied_lines",
        "joined_words",
        "merged_digits",
        "touched",
    ]
    assert summary["removed_artefacts"] == [3]
    assert summary["removed_captions"] == [5]
    assert summary["promoted_headings"] == [7]
    assert summary["merged_digits"] == {11: []}


def test_the_verify_result_key_order_is_stable(tmp_path: Path) -> None:
    path = _chapter(tmp_path)
    verification = verify_cleaning_plan(path, OPERATIONS, TEXT_FIXES)

    assert list(verification) == [
        "ok",
        "problems",
        "removed_artefacts",
        "removed_captions",
        "undeclared_removals",
        "text_fixes",
        "output_lines",
        "source_lines",
        "provenance",
    ]
    assert verification["removed_artefacts"] == [3]
    assert verification["removed_captions"] == [5]
    assert verification["text_fixes"]["checked"] == [
        {"line_number": 15, "fix": "split_word", "joined_with": 16, "word": "rije"}
    ]


def test_the_transform_facade_still_exposes_the_old_surface() -> None:
    from book_splitter.preprocess import apply, plan, transform, verify

    owners = {
        "ApplyResult": apply.ApplyResult,
        "LineEdit": apply.LineEdit,
        "apply_cleaning_plan": apply.apply_cleaning_plan,
        "looks_like_caption": plan.looks_like_caption,
        "looks_like_page_artefact": plan.looks_like_page_artefact,
        "next_content_line": plan.next_content_line,
        "strip_leading_page_number": plan.strip_leading_page_number,
        "text_of": apply.text_of,
        "validate_cleaning_plan": plan.validate_cleaning_plan,
        "validate_text_fixes": plan.validate_text_fixes,
        "verify_cleaning_plan": verify.verify_cleaning_plan,
        "verify_text_fixes": verify.verify_text_fixes,
    }
    assert transform.__all__ == list(owners)
    for name, value in owners.items():
        assert getattr(transform, name) is value
