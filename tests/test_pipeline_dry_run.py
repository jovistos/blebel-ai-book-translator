"""The dry-run contract: validate the inputs, never build a model or spend.

The fixture pipeline is built once (it exercises every stage for real), then
every model-backed runner is called with ``dry_run=True`` while
``build_chat_model`` and both settings resolvers are patched to explode. A
preview that accidentally asks for a provider fails this test with a clear
message instead of a bill.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from book_translator.stages.agent import run_split
from book_translator.stages.glossary import run_glossary
from book_translator.stages.matter_agent import run_matter
from book_translator.stages.postprocess_agent import run_postprocess
from book_translator.stages.preprocess_agent import run_preprocess
from book_translator.stages.translation import run_translation

from _pipeline_harness import run_fixture_pipeline


@pytest.fixture(scope="module")
def fixture_book(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("dry-run-fixture")
    monkeypatch = pytest.MonkeyPatch()
    try:
        book, _results = run_fixture_pipeline(root, monkeypatch)
    finally:
        monkeypatch.undo()
    return book


@pytest.fixture
def no_spend(monkeypatch):
    def explode(*_args, **_kwargs):
        raise AssertionError("a dry run tried to construct a model or resolve a provider")

    monkeypatch.setattr("book_translator.core.llm.build_chat_model", explode)
    monkeypatch.setattr("book_translator.stages.agent.graph.make_llm", explode)
    monkeypatch.setattr("book_translator.core.settings.resolve_settings", explode)
    monkeypatch.setattr(
        "book_translator.core.settings.resolve_translation_settings", explode
    )
    return explode


def test_split_dry_run_reads_the_source_but_builds_no_graph(fixture_book, no_spend):
    out = fixture_book / "dry-chapters"
    result = run_split(
        str(fixture_book / "00-markdown" / "fixture.md"),
        str(out),
        max_attempts=3,
        dry_run=True,
    )
    assert result.ok and result.verification.get("dry_run") is True
    assert result.chapter_count == 0
    assert result.attempts == 0
    assert not out.exists()


def test_preprocess_dry_run_inventories_chapters_but_builds_no_graph(
    fixture_book, no_spend
):
    out = fixture_book / "dry-preprocessed"
    result = run_preprocess(str(fixture_book / "01-chapters"), str(out), dry_run=True)
    assert result.ok and result.dry_run is True
    assert result.chapter_count > 0
    assert result.chapters == []
    assert result.corpus_scan
    assert not out.exists()


def test_glossary_dry_run_inventories_chapters_but_builds_no_extractor(
    fixture_book, no_spend
):
    out = fixture_book / "dry-glossary"
    result = run_glossary(
        str(fixture_book / "02-preprocessed"),
        str(out),
        sentences_dir=str(fixture_book / "02-preprocessed"),
        dry_run=True,
    )
    assert result.ok and result.dry_run is True
    assert result.processed_count > 0
    assert result.entry_count == 0
    assert not out.exists()


def test_matter_dry_run_inventories_sections_but_builds_no_plan(fixture_book, no_spend):
    out = fixture_book / "dry-matter-translations"
    result = run_matter(
        str(fixture_book / "01-chapters"),
        str(fixture_book / "05-chunks"),
        str(out),
        dry_run=True,
    )
    assert result.ok and result.dry_run is True
    assert result.section_count > 0
    assert result.processed_count > 0
    assert result.book_plan == {}
    assert not out.exists()


def test_translation_dry_run_inventories_chunks_but_builds_no_translator(
    fixture_book, no_spend
):
    out = fixture_book / "dry-translations"
    result = run_translation(str(fixture_book / "05-chunks"), str(out), dry_run=True)
    assert result.ok and result.dry_run is True
    assert result.chunk_count > 0
    assert result.processed_count > 0
    assert result.translator == ""
    assert not out.exists()


def test_postprocess_dry_run_inventories_chapters_but_builds_no_plan(
    fixture_book, no_spend
):
    out = fixture_book / "dry-postprocessed"
    result = run_postprocess(
        str(fixture_book / "08-translated-chapters"), str(out), dry_run=True
    )
    assert result.ok and result.dry_run is True
    assert result.chapter_count > 0
    assert result.book_plan == {}
    assert not out.exists()
