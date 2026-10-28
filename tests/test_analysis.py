"""Tests for the deterministic analysis layer (no LLM, no writes)."""

from __future__ import annotations

import json

import pytest
from book_translator.documents.io import load_lines
from book_translator.stages.agent.tools import (
    extract_heuristic_candidates,
    extract_markdown_headers,
    find_lines_matching,
    locate_toc_titles_in_file,
)
from book_translator.stages.analysis.headers import assess_headers
from book_translator.stages.analysis.toc import detect_toc_block

from _fixtures import SAMPLE_SPLITS, SAMPLE_TOC_TITLES, requires_sample

# --------------------------------------------------------------------------- #
# Headers
# --------------------------------------------------------------------------- #


@requires_sample
def test_header_scan_is_noisy_but_non_empty(sample):
    lines = load_lines(str(sample))
    headers = json.loads(extract_markdown_headers.invoke({"file_path": str(sample)}))["headers"]
    assert len(lines) == 640
    assert len(headers) == 78
    assessment = assess_headers(headers, len(lines))
    # The honest verdict: headers exist, but a third of them are conversion junk.
    assert assessment["verdict"] == "noisy"
    assert assessment["chapter_like_count"] >= 15
    assert assessment["noise_ratio"] > 0.25
    assert any("noise" in r for r in assessment["reasons"])


def test_headers_absent_on_unmarkedup_book(plain):
    lines = load_lines(str(plain))
    headers = json.loads(extract_markdown_headers.invoke({"file_path": str(plain)}))["headers"]
    assert headers == []
    assert assess_headers(headers, len(lines))["verdict"] == "absent"


# --------------------------------------------------------------------------- #
# Printed table of contents
# --------------------------------------------------------------------------- #


@requires_sample
def test_detect_toc_block_sample(sample):
    block = detect_toc_block(load_lines(str(sample)))
    assert block["found"] and block["method"] == "keyword"
    assert (block["start_line"], block["end_line"]) == (24, 44)


def test_detect_toc_block_stops_at_body_repeat(plain):
    """Without `#` markup the TOC ends where the first title reappears in the body."""
    block = detect_toc_block(load_lines(str(plain)))
    assert (block["start_line"], block["end_line"]) == (5, 11)


def test_detect_toc_block_does_not_cut_on_similar_entries_or_urls():
    """TOC entries with similar text or identical URL prefixes must not trigger body repeat."""
    lines = [
        "## Sadržaj\n",
        "\n",
        "- Poglavlje 1: Prvi koraci\n",
        "- Poglavlje 2: Prvi znaci\n",
        "- [Komentari i pitanja](/tmp/docling_epub_abc/part1.html)\n",
        "- [Za dalje čitanje](/tmp/docling_epub_abc/part2.html)\n",
        "\n",
        "## Pravo Poglavlje\n",
        "Tekst poglavlja...\n",
    ]
    block = detect_toc_block(lines)
    assert block["found"] is True
    # Heading is line 1, TOC block spans lines 2 to 6 (inclusive)
    assert block["start_line"] == 2
    assert block["end_line"] == 6


# --------------------------------------------------------------------------- #
# Title location
# --------------------------------------------------------------------------- #


@requires_sample
def test_toc_locator_finds_every_chapter_and_skips_the_toc_block(sample):
    result = json.loads(
        locate_toc_titles_in_file.invoke(
            {"file_path": str(sample), "chapter_titles": SAMPLE_TOC_TITLES}
        )
    )
    assert result["found_count"] == len(SAMPLE_TOC_TITLES)
    assert result["out_of_order_pairs"] == []
    assert [m["line_number"] for m in result["matches"]] == SAMPLE_SPLITS
    # Nothing may resolve into the printed table of contents (lines 24-44).
    assert all(not 24 <= m["line_number"] <= 44 for m in result["matches"])


@requires_sample
def test_toc_locator_survives_typos(sample):
    result = json.loads(
        locate_toc_titles_in_file.invoke(
            {"file_path": str(sample), "chapter_titles": ["7. Sr čani udar", "5. Strah od ljubavi"]}
        )
    )
    got = {m["title"]: m["line_number"] for m in result["matches"]}
    assert got["7. Sr čani udar"] == 303  # body says `GLAVA 7 SR ČANI UDAR`
    assert got["5. Strah od ljubavi"] == 245  # body says `GIAVA 5 ...` (broken G)


@requires_sample
def test_locator_snaps_subtitle_to_chapter_heading(sample):
    result = json.loads(
        locate_toc_titles_in_file.invoke(
            {
                "file_path": str(sample),
                "chapter_titles": ["2. Seks i srce", "10. Zdravo srce - Osoba koja voli"],
            }
        )
    )
    got = {m["title"]: m for m in result["matches"]}
    # `## GLAVA 2` (136) then `## SEKS I SRCE` (138): the chapter starts at 136.
    assert got["2. Seks i srce"]["line_number"] == 136
    assert got["2. Seks i srce"]["snapped_from"] == 138
    assert got["10. Zdravo srce - Osoba koja voli"]["line_number"] == 434


# --------------------------------------------------------------------------- #
# Heuristic candidates
# --------------------------------------------------------------------------- #


def test_heuristic_candidates_are_short_standalone_or_keyword_lines(plain):
    result = json.loads(
        extract_heuristic_candidates.invoke({"file_path": str(plain), "max_candidates": 50})
    )
    numbers = [c["line_number"] for c in result["candidates"]]
    assert 12 in numbers and 18 in numbers  # "Predgovor", "1. Dolazak"
    assert all(len(c["text"]) <= 91 for c in result["candidates"])


# --------------------------------------------------------------------------- #
# Pattern search: the regex the agent writes itself
# --------------------------------------------------------------------------- #

#: A book that lost all its markup: chapter markers, page numbers, a short line
#: that mentions a chapter, and a long standalone paragraph.
MARKERLESS_BOOK = "\n".join(
    [
        "DON'T YOU FORGET ABOUT ME",  # 1
        "Alexandra Potter",  # 2
        "",  # 3
        "Chapter 1",  # 4
        "",  # 5
        "Two months, one week and three days since we had that conversation.",  # 6
        "",  # 7
        "19",  # 8
        "",  # 9
        "Chapter 2",  # 10
        "",  # 11
        "In actual fact it was mostly Seb telling me he loved me.",  # 12
        "",  # 13
        "The chapter of accidents, as my mother would say, was not over yet, and I had the "
        "distinct feeling it was only just beginning.",  # 14
        "",  # 15
        "Chapter 3",  # 16
        "",  # 17
        "Nothing will ever be the same again.",  # 18
        "",  # 19
        "A last line of prose that leads nowhere in particular.",  # 20
        "",  # 21
        "And one more, so a one-match cap is genuinely too small.",  # 22
        "",  # 23
    ]
)


@pytest.fixture
def markerless(tmp_path):
    path = tmp_path / "markerless.md"
    path.write_text(MARKERLESS_BOOK, encoding="utf-8")
    return path


def test_pattern_search_recovers_every_chapter_marker(markerless):
    """One regex, one call: the whole spine of a book with no markup at all."""
    result = json.loads(
        find_lines_matching.invoke({"file_path": str(markerless), "pattern": r"^Chapter \d+$"})
    )
    assert result["match_count"] == 3
    assert result["truncated"] is False and result["timed_out"] is False
    assert result["valid_split_points"] == 3
    assert [m["line_number"] for m in result["matches"]] == [4, 10, 16]
    assert all(m["standalone"] and m["title_like"] for m in result["matches"])
    assert not any(m["is_page_number"] for m in result["matches"])


def test_pattern_search_flags_prose_that_merely_mentions_a_marker(markerless):
    """A paragraph that contains the word is a match, but not a split point."""
    result = json.loads(
        find_lines_matching.invoke({"file_path": str(markerless), "pattern": "chapter"})
    )
    assert result["match_count"] == 1  # the long paragraph on line 14 only
    paragraph = result["matches"][0]
    assert paragraph["line_number"] == 14
    assert paragraph["standalone"] is True
    assert paragraph["title_like"] is False  # 128 characters: prose, not a heading
    assert result["valid_split_points"] == 0


def test_pattern_search_case_folding_is_the_agents_choice(markerless):
    sensitive = json.loads(
        find_lines_matching.invoke({"file_path": str(markerless), "pattern": "chapter"})
    )
    folded = json.loads(
        find_lines_matching.invoke(
            {"file_path": str(markerless), "pattern": "CHAPTER", "ignore_case": True}
        )
    )
    assert sensitive["match_count"] == 1
    assert folded["ignore_case"] is True
    assert folded["match_count"] == 4  # the three markers + the paragraph
    assert folded["valid_split_points"] == 3


def test_pattern_search_caps_matches_and_says_so(markerless):
    result = json.loads(
        find_lines_matching.invoke(
            {"file_path": str(markerless), "pattern": r"^Chapter \d+$", "max_matches": 2}
        )
    )
    assert result["returned"] == 2
    assert result["match_count"] == 3
    assert result["truncated"] is True
    assert "narrow the pattern" in result["hint"]


def test_pattern_search_rejects_a_bad_pattern_without_raising(markerless):
    for bad in ["", "   ", "Chapter ("]:
        result = json.loads(
            find_lines_matching.invoke({"file_path": str(markerless), "pattern": bad})
        )
        assert "error" in result and "hint" in result
        assert "matches" not in result


def test_pattern_search_is_bounded_by_the_shared_result_cap(markerless):
    # A pattern that matches every non-blank line still returns clipped excerpts.
    result = json.loads(find_lines_matching.invoke({"file_path": str(markerless), "pattern": "."}))
    assert result["match_count"] == sum(1 for line in MARKERLESS_BOOK.splitlines() if line)
    assert result["returned"] == result["match_count"]
    assert all(len(m["text"]) <= 90 for m in result["matches"])


def test_pattern_search_reports_a_too_broad_pattern(markerless):
    result = json.loads(
        find_lines_matching.invoke({"file_path": str(markerless), "pattern": ".", "max_matches": 1})
    )
    assert result["truncated"] is True
    assert "too broad" in result["hint"]
