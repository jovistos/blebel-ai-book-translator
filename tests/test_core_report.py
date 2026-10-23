"""Tests for the small reporting surface shared by the CLI and the pipeline."""

from __future__ import annotations

from book_translator.core.report import (
    StageSummary,
    human_bytes,
    human_count,
    human_seconds,
    summarize_payload,
)


def test_human_seconds_covers_the_whole_range():
    assert human_seconds(0.0213) == "21 ms"
    assert human_seconds(1.5) == "1.50 s"
    assert human_seconds(389.42) == "6m 29.4s"
    assert human_seconds(3723.4) == "1h 02m 03.4s"
    assert human_seconds(-1) == "0 ms"


def test_human_bytes_and_counts():
    assert human_bytes(411144) == "401.5 KiB"
    assert human_bytes(512) == "512 B"
    assert human_bytes(2 * 1024 * 1024) == "2.0 MiB"
    assert human_count(12345) == "12,345"
    assert human_count(None) == "0"


def test_payload_summaries_for_recon_and_scan():
    recon = summarize_payload(
        "recon",
        {
            "source": "/tmp/book.md",
            "line_count": 100,
            "header_count": 8,
            "header_assessment": {"verdict": "noisy"},
            "candidate_count": 4,
            "printed_toc_block": {"found": True},
        },
    )
    assert recon.headline == "100 lines · 8 headers (noisy) · 4 candidates"
    assert ("printed TOC", "found") in recon.rows

    scan = summarize_payload(
        "scan",
        {"chapter_count": 2, "total_findings": 3, "counts": {"page_number_header": 3}},
    )
    assert scan.headline == "3 findings in 1 shape(s)"
    assert ("page_number_header", "3") in scan.rows


def test_the_generic_payload_summary_keeps_scalar_fields():
    summary = summarize_payload("custom", {"ok": True, "count": 3, "name": "x"})
    assert summary.headline == "custom"
    assert summary.rows == (("ok", "True"), ("count", "3"), ("name", "x"))


def test_stage_summary_is_a_frozen_value():
    summary = StageSummary(headline="x", rows=(("chapters", "1/1"),))
    assert summary.headline == "x"
    assert summary.rows == (("chapters", "1/1"),)
