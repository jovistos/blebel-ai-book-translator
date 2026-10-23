"""Tests for chapter-map validation."""

from __future__ import annotations

from book_translator.documents.io import load_lines
from book_translator.stages.analysis.toc import detect_toc_block
from book_translator.stages.slicing.validation import validate_split_map

from _fixtures import SAMPLE, SAMPLE_SPLITS, requires_sample

# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@requires_sample
def test_validate_accepts_the_canonical_plan(plan):
    lines = load_lines(str(SAMPLE))
    mapping = plan([(ln, f"{i:02d}-c.md") for i, ln in enumerate(SAMPLE_SPLITS, start=1)])
    result = validate_split_map(lines, mapping, toc_block=detect_toc_block(lines))
    assert result["ok"], result["issues"]
    assert result["chapter_count"] == len(SAMPLE_SPLITS)
    assert result["warnings"]  # front matter notice


@requires_sample
def test_validate_rejects_toc_entries_duplicates_and_prose_lines():
    lines = load_lines(str(SAMPLE))
    mapping = [
        {"line_number": 27, "filename": "01-toc-entry.md"},  # inside printed TOC
        {"line_number": 52, "filename": "02-prose.md"},  # first line of a paragraph
        {"line_number": 9999, "filename": "03-out-of-range.md"},
        {"line_number": 46, "filename": "04-a.md"},
        {"line_number": 46, "filename": "05-b.md"},  # duplicate line
    ]
    result = validate_split_map(lines, mapping, toc_block=detect_toc_block(lines))
    assert not result["ok"]
    joined = " | ".join(result["issues"])
    assert "inside the printed table of contents" in joined
    assert "does not look like a title" in joined
    assert "outside the file" in joined
    assert "used twice" in joined


@requires_sample
def test_validate_flags_a_missing_chapter_gap(plan):
    lines = load_lines(str(SAMPLE))
    mapping = plan([(46, "01-a.md"), (78, "02-b.md"), (600, "03-c.md")])
    result = validate_split_map(lines, mapping, max_gap_lines=100)
    assert not result["ok"] or result["warnings"]
    assert any("gap of" in w for w in result["warnings"])


def test_validate_rejects_an_empty_map(plain):
    result = validate_split_map(load_lines(str(plain)), [])
    assert not result["ok"]
    assert result["issues"] == ["split_mapping is empty"]


def test_validate_reports_non_integer_and_non_object_entries(plain):
    result = validate_split_map(
        load_lines(str(plain)),
        ["not-an-object", {"line_number": "abc", "filename": "01.md"}, {"line_number": 12}],
    )
    assert not result["ok"]
    joined = " | ".join(result["issues"])
    assert "entry is not an object" in joined
    assert "line_number is not an integer" in joined
    assert "missing filename" in joined
