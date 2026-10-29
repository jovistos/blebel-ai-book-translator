"""``--skip-if-present``: reuse a stage that is already done, without redoing it.

Two places honour the flag: stage 1 (an existing chapters folder wins over a new
split) and stage 3 (an existing reduced chapter is reused). The point of the flag
is speed *and* safety — nothing already on disk is rewritten, and the original
audit trail survives.
"""

from __future__ import annotations

import json
from pathlib import Path

from book_translator import cli
from book_translator.core.constants import MANIFEST_FILENAME, TRACE_FILENAME
from book_translator.documents.chapters import has_chapter_files
from book_translator.stages.agent import runner

BOOK_TEXT = "# Book\n\n## Prva\n\ntext\n\n## Druga\n\nmore text\n"


def make_book(tmp_path: Path) -> tuple[Path, Path]:
    """A source file plus a chapters folder that looks like a finished split."""
    source = tmp_path / "book.md"
    source.write_text(BOOK_TEXT, encoding="utf-8")
    chapters = tmp_path / "02-chapters"
    chapters.mkdir()
    (chapters / "01-prva.md").write_text("## Prva\n\ntext\n", encoding="utf-8")
    (chapters / "02-druga.md").write_text("## Druga\n\nmore text\n", encoding="utf-8")
    (chapters / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "source_file": str(source),
                "strategy": "preset_map",
                "files": [
                    {"filename": "01-prva.md", "start_line": 3, "end_line": 5, "title": "Prva"},
                    {"filename": "02-druga.md", "start_line": 7, "end_line": 9, "title": "Druga"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return source, chapters


# --------------------------------------------------------------------------- #
# the runner
# --------------------------------------------------------------------------- #
def test_a_folder_with_chapters_counts_as_present():
    assert has_chapter_files(Path("/nonexistent")) is False


def test_run_split_reuses_the_existing_chapters_and_builds_no_graph(tmp_path, monkeypatch):
    source, chapters = make_book(tmp_path)
    trace = chapters / TRACE_FILENAME
    trace.write_text('{"result": "the original split"}', encoding="utf-8")

    def explode(*_, **__):  # pragma: no cover - a reused folder must not reach the agent
        raise AssertionError("the graph must not be built when the chapters are already there")

    monkeypatch.setattr(runner, "build_graph", explode)

    result = runner.run_split(str(source), str(chapters), skip_if_present=True)

    assert result.ok is True
    assert result.strategy == runner.REUSED_CHAPTERS_STRATEGY
    assert result.chapter_count == 2
    assert result.attempts == 0
    assert result.verification == {"reused_existing_chapters": True}
    assert result.trace_path == str(trace)
    # The original audit trail is pointed at, never overwritten.
    assert trace.read_text(encoding="utf-8") == '{"result": "the original split"}'


def test_a_folder_with_only_bookkeeping_files_is_not_reused(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
    assert has_chapter_files(empty) is False


def test_run_split_from_map_also_reuses(tmp_path):
    source, chapters = make_book(tmp_path)
    mapping = [{"line_number": 3, "filename": "01-prva.md"}]

    result = runner.run_split_from_map(
        str(source), str(chapters), mapping, skip_if_present=True
    )

    assert result.ok is True
    assert result.strategy == runner.REUSED_CHAPTERS_STRATEGY
    assert result.chapter_count == 2  # the folder wins over the supplied map
    assert (chapters / "01-prva.md").read_text(encoding="utf-8") == "## Prva\n\ntext\n"


# --------------------------------------------------------------------------- #
# the CLI
# --------------------------------------------------------------------------- #
def test_cli_split_skip_if_present_needs_no_provider_settings(tmp_path, monkeypatch, capsys):
    from book_translator.core import settings as settings_module

    source, chapters = make_book(tmp_path)

    def explode(*_, **__):  # pragma: no cover - a skipped split needs no model
        raise AssertionError("--skip-if-present must not resolve provider settings")

    monkeypatch.setattr(settings_module, "resolve_settings", explode)

    code = cli.main(["split", str(source), "--out", str(chapters), "--skip-if-present"])
    out = capsys.readouterr().out

    assert code == 0
    assert "chapters        : 2" in out
    assert "reused" in out
    assert "nothing was re-sliced" in out
