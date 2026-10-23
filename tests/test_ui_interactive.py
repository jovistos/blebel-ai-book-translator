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
    driver.matter_out = tmp_path / "06b-matter"
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
    driver.matter_out = tmp_path / "06b-matter"
    driver.postprocess_out = tmp_path / "09-postprocess"
    driver.book_dir = tmp_path
    driver.console = MagicMock()
    driver.console.is_rich = False

    record = PhaseRecord(name="glossary", ok=True)

    # First user enters 'e' (edit), then presses enter ('') to resume
    with patch("builtins.input", side_effect=["e", ""]):
        res = PipelineDriver._checkpoint_pause(driver, "glossary", record)
        assert res is True
