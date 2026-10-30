"""Tests for interactive checkpoints and the CLI wizard."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from book_translator.cli.commands.wizard import cmd_wizard
from book_translator.cli.parser import build_parser
from book_translator.pipeline.driver import PipelineDriver
from book_translator.pipeline.options import PipelineOptions
from book_translator.pipeline.parser import build_parser as build_pipeline_parser
from book_translator.pipeline.report import PhaseRecord


def test_pipeline_parser_accepts_interactive_flag() -> None:
    parser = build_pipeline_parser()
    args = parser.parse_args(["book.md", "--interactive"])
    assert args.interactive is True

    args_checkpoint = parser.parse_args(["book.md", "--checkpoint"])
    assert args_checkpoint.interactive is True

    options = PipelineOptions.from_namespace(args)
    assert options.interactive is True


def test_wizard_parser_registered_in_root() -> None:
    parser = build_parser()
    args = parser.parse_args(["wizard", "test_book.md", "--first-chapters", "5"])
    assert args.command == "wizard"
    assert args.book == "test_book.md"
    assert args.first_chapters == 5
    assert callable(args.func)

    args_alias = parser.parse_args(["wizard", "--chapters", "10"])
    assert args_alias.first_chapters == 10


def test_wizard_rejects_non_tty(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    parser = build_parser()
    args = parser.parse_args(["wizard"])
    ret = cmd_wizard(args)
    assert ret == 1
    captured = capsys.readouterr()
    assert "wizard requires an interactive terminal" in captured.err


def test_checkpoint_pause_continues_on_affirmative(tmp_path) -> None:
    driver = MagicMock(spec=PipelineDriver)
    driver.glossary_out = tmp_path / "04-glossary"
    driver.matter_out = tmp_path / "07-matter"
    driver.postprocess_out = tmp_path / "09-postprocess"
    driver.book_dir = tmp_path
    driver.console = MagicMock()
    driver.console.is_rich = False

    record = PhaseRecord(name="glossary", ok=True)

    with patch("builtins.input", side_effect=["c"]):
        res = PipelineDriver._checkpoint_pause(driver, "glossary", record)
        assert res is True

    with patch("builtins.input", side_effect=["a"]):
        res = PipelineDriver._checkpoint_pause(driver, "glossary", record)
        assert res is False


def test_checkpoint_pause_allows_edit(tmp_path) -> None:
    driver = MagicMock(spec=PipelineDriver)
    driver.glossary_out = tmp_path / "04-glossary"
    driver.matter_out = tmp_path / "07-matter"
    driver.postprocess_out = tmp_path / "09-postprocess"
    driver.book_dir = tmp_path
    driver.console = MagicMock()
    driver.console.is_rich = False

    record = PhaseRecord(name="glossary", ok=True)

    # First user enters 'e' (edit), then presses enter ('') to resume
    with patch("builtins.input", side_effect=["e", ""]):
        res = PipelineDriver._checkpoint_pause(driver, "glossary", record)
        assert res is True


def test_wizard_language_selection_custom(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    parser = build_parser()
    args = parser.parse_args(["wizard", "test_book.md"])

    inputs = [
        "1",     # preset: 1 (Full Pipeline)
        "1",     # provider: 1
        "de",    # source language: German
        "fr",    # destination language: French
        "",      # first chapters: full book
        "n",     # pause at checkpoints
        "n",     # skip-if-present
        "n",     # cpu
        "p",     # action: print and exit
    ]
    with patch("builtins.input", side_effect=inputs):
        ret = cmd_wizard(args)
    assert ret == 0

    captured = capsys.readouterr()
    assert "Supported Language Codes" in captured.out
    assert "de" in captured.out
    assert "fr" in captured.out
    assert "--source-language de" in captured.out
    assert "--target-language fr" in captured.out
    assert "Languages selected: de (German) → fr (French)" in captured.out


def test_wizard_language_selection_default(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    parser = build_parser()
    args = parser.parse_args(["wizard", "test_book.md"])

    inputs = [
        "1",     # preset: 1
        "1",     # provider: 1
        "",      # source language: keep default 'en'
        "",      # destination language: keep default 'sr'
        "",      # chapters
        "n",     # checkpoints
        "n",     # skip
        "n",     # cpu
        "p",     # action: print and exit
    ]
    with patch("builtins.input", side_effect=inputs):
        ret = cmd_wizard(args)
    assert ret == 0

    captured = capsys.readouterr()
    assert "Supported Language Codes" in captured.out
    assert "Languages selected: en (English) → sr (Serbian)" in captured.out
    assert "--source-language" not in captured.out
    assert "--target-language" not in captured.out


def test_wizard_language_selection_case_insensitive(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    parser = build_parser()
    args = parser.parse_args(["wizard", "test_book.md"])

    inputs = [
        "1",          # preset: 1
        "1",          # provider: 1
        "ES",         # source language uppercase
        "JA-LATN",    # destination language uppercase
        "",           # chapters
        "n",          # checkpoints
        "n",          # skip
        "n",          # cpu
        "p",          # action: print and exit
    ]
    with patch("builtins.input", side_effect=inputs):
        ret = cmd_wizard(args)
    assert ret == 0

    captured = capsys.readouterr()
    assert "--source-language es" in captured.out
    assert "--target-language ja-Latn" in captured.out
    assert "Languages selected: es (Spanish) → ja-Latn (Japanese (romanised))" in captured.out


def test_wizard_json_output(capsys: pytest.CaptureFixture) -> None:
    import json

    parser = build_parser()
    args = parser.parse_args(["wizard", "--json"])
    ret = cmd_wizard(args)
    assert ret == 0

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "languages" in data
    assert data["languages"]["en"] == "English"
    assert data["languages"]["sr"] == "Serbian"
    assert data["default_source_language"] == "en"
    assert data["default_target_language"] == "sr"


def test_checkpoint_pause_glossary_displays_csv(tmp_path, capsys: pytest.CaptureFixture) -> None:
    from book_translator.stages.glossary.artifacts import write_glossary, write_glossary_csv
    from book_translator.stages.glossary.book import GlossaryBook
    from book_translator.stages.glossary.records import GlossaryResult, GlossaryTerm

    glossary_dir = tmp_path / "04-glossary"
    glossary_dir.mkdir(parents=True)
    book = GlossaryBook()
    book.lock(GlossaryTerm(term="Gatsby", category="character", translation="Getsbi", chapter_filename="01.md"))
    result = GlossaryResult(ok=True, chapters_path="", sentences_dir="", output_dir=str(glossary_dir))
    write_glossary(glossary_dir, result, book)
    csv_file = write_glossary_csv(glossary_dir, result, book)

    driver = MagicMock(spec=PipelineDriver)
    driver.glossary_out = glossary_dir
    driver.book_dir = tmp_path
    driver.console = MagicMock()
    driver.console.is_rich = False

    record = PhaseRecord(name="glossary", ok=True)
    with patch("builtins.input", side_effect=["c"]):
        res = PipelineDriver._checkpoint_pause(driver, "glossary", record)
        assert res is True

    captured = capsys.readouterr()
    assert str(csv_file.resolve()) in captured.out
    assert "Please check out the CSV file with the dictionary entries" in captured.out
    assert "You can open and edit this CSV file to change translations" in captured.out


def test_checkpoint_pause_syncs_csv_edits(tmp_path) -> None:
    import json

    from book_translator.stages.glossary.artifacts import write_glossary, write_glossary_csv
    from book_translator.stages.glossary.book import GlossaryBook
    from book_translator.stages.glossary.records import GlossaryResult, GlossaryTerm

    glossary_dir = tmp_path / "04-glossary"
    glossary_dir.mkdir(parents=True)
    book = GlossaryBook()
    book.lock(GlossaryTerm(term="Gatsby", category="character", translation="Getsbi", chapter_filename="01.md"))
    result = GlossaryResult(ok=True, chapters_path="", sentences_dir="", output_dir=str(glossary_dir))
    write_glossary(glossary_dir, result, book)
    csv_file = write_glossary_csv(glossary_dir, result, book)

    driver = MagicMock(spec=PipelineDriver)
    driver.glossary_out = glossary_dir
    driver.book_dir = tmp_path
    driver.console = MagicMock()
    driver.console.is_rich = False

    # Simulate user editing the CSV file
    csv_file.write_text("term,category,translation,chapter_filename\nGatsby,character,Veliki Getsbi,01.md\n", encoding="utf-8")

    record = PhaseRecord(name="glossary", ok=True)
    with patch("builtins.input", side_effect=["c"]):
        res = PipelineDriver._checkpoint_pause(driver, "glossary", record)
        assert res is True

    # Check that glossary.json was updated with the edited translation
    data = json.loads((glossary_dir / "glossary.json").read_text(encoding="utf-8"))
    assert data["entries"][0]["translation"] == "Veliki Getsbi"
    driver.console.info.assert_called_with("  ✓ Applied 1 translation update(s) and 0 new term(s) from glossary.csv")


def test_chunk_glossary_load_syncs_csv(tmp_path) -> None:
    from book_translator.stages.chunk_glossary.matching import load_glossary
    from book_translator.stages.glossary.artifacts import write_glossary, write_glossary_csv
    from book_translator.stages.glossary.book import GlossaryBook
    from book_translator.stages.glossary.records import GlossaryResult, GlossaryTerm

    glossary_dir = tmp_path / "04-glossary"
    glossary_dir.mkdir(parents=True)
    book = GlossaryBook()
    book.lock(GlossaryTerm(term="Daisy", category="character", translation="Dejzi", chapter_filename="01.md"))
    result = GlossaryResult(ok=True, chapters_path="", sentences_dir="", output_dir=str(glossary_dir))
    write_glossary(glossary_dir, result, book)
    csv_file = write_glossary_csv(glossary_dir, result, book)

    # Edit CSV
    csv_file.write_text("term,category,translation,chapter_filename\nDaisy,character,Dejzi Bjukenan,01.md\n", encoding="utf-8")

    entries = load_glossary(glossary_dir / "glossary.json")
    assert len(entries) == 1
    assert entries[0].term == "Daisy"
    assert entries[0].translation == "Dejzi Bjukenan"

