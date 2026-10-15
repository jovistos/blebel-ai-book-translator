"""Tests for the deterministic analysis layer (no LLM, no writes)."""

from __future__ import annotations

import json

from book_splitter.agent.tools import (
    extract_heuristic_candidates,
    extract_markdown_headers,
    locate_toc_titles_in_file,
)
from book_splitter.analysis.headers import assess_headers
from book_splitter.analysis.toc import detect_toc_block
from book_splitter.documents.io import load_lines

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
