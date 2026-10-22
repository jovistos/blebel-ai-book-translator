"""Phase 0 snapshots: every CLI ``--help`` and the public package surface.

The command-line interface and ``book_splitter.__all__`` are the two contracts a
structural refactor is most likely to break silently. These tests freeze both:
any new, removed, renamed or re-described argument, default or exported name
fails until the snapshot is deliberately regenerated with::

    UPDATE_SNAPSHOTS=1 pytest tests/test_snapshots.py

Command discovery comes from the parser itself, so adding a subcommand without
adding its help snapshot also fails (the snapshot key set is asserted).
"""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path
from typing import Any

import book_splitter
import pytest
from book_splitter.cli import build_parser

from _fixtures import ROOT
from _snapshots import assert_snapshot

#: The key used for the root parser's help in ``help.json``.
ROOT_HELP_KEY = "<root>"


def _subcommands(parser: argparse.ArgumentParser) -> list[str]:
    """Every command name the parser registers, sorted."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            choices: dict[str, Any] = action.choices
            return sorted(choices)
    raise AssertionError("the CLI has no subcommands")


def _render_help(parser: argparse.ArgumentParser, args: list[str]) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), pytest.raises(SystemExit) as exited:
        parser.parse_args(args)
    assert exited.value.code == 0
    return buffer.getvalue()


def _normalise(text: str) -> str:
    """Replace machine-specific paths and collapse argparse's line wrapping.

    The argument set, help strings and defaults are what matter; where argparse
    chooses to wrap them depends on the terminal width and the Python patch
    version, so whitespace is canonicalised rather than frozen.
    """
    for raw in {str(ROOT), str(ROOT.resolve())}:
        text = text.replace(raw, "<ROOT>")
    text = text.replace(str(Path.home()), "<HOME>")
    return " ".join(text.split())


def test_every_help_output_matches_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    # argparse wraps to the terminal width; pin it so the snapshot is portable.
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setenv("LINES", "24")

    parser = build_parser()
    commands = _subcommands(parser)

    actual = {ROOT_HELP_KEY: _normalise(_render_help(parser, ["--help"]))}
    for command in commands:
        actual[command] = _normalise(_render_help(parser, [command, "--help"]))

    assert set(actual) == {ROOT_HELP_KEY, *commands}
    assert_snapshot("help.json", actual)


def test_package_all_matches_snapshot() -> None:
    names = list(book_splitter.__all__)

    assert len(names) == len(set(names)), "book_splitter.__all__ contains duplicates"
    for name in names:
        assert hasattr(book_splitter, name), (
            f"book_splitter.__all__ names a missing attribute: {name}"
        )

    assert_snapshot("package_all.json", names)
