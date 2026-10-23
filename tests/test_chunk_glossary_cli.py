"""Offline tests for the `chunk-glossary` command: flags, wiring and the files.

The tagger is stubbed at the seam the runner builds it from, so the command's real
path runs — chunks and glossary are resolved from the book folder, both phases
execute, the per-chunk files land — without torch, a model download or a network
call. What is pinned on top of the runner's own tests is the interface: the 95 %
default, the scorer option, the folder defaults, the JSON result and the setup
errors.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from book_translator import cli
from book_translator.core.constants import (
    DEFAULT_CHUNK_GLOSSARY_MIN_TERM_CHARS,
    DEFAULT_CHUNK_GLOSSARY_SCORER,
    DEFAULT_CHUNK_GLOSSARY_THRESHOLD,
)

from test_chunk_glossary_runner import FakeBackend, _project


@pytest.fixture
def offline_tagger(monkeypatch):
    """Replace the tagger factory; no test downloads or loads a model."""
    backend = FakeBackend()
    monkeypatch.setattr(
        "book_translator.stages.chunk_glossary.runner.make_backend",
        lambda *args, **kwargs: backend,
    )
    return backend


# --------------------------------------------------------------------------- #
# the interface
# --------------------------------------------------------------------------- #
def test_the_defaults_are_95100_ratio_and_a_percentage():
    args = cli.build_parser().parse_args(["chunk-glossary"])
    assert args.func is cli.cmd_chunk_glossary
    assert args.threshold == DEFAULT_CHUNK_GLOSSARY_THRESHOLD == 95.0
    assert args.scorer == DEFAULT_CHUNK_GLOSSARY_SCORER == "ratio"
    assert args.candidates == 3
    assert args.min_term_chars == DEFAULT_CHUNK_GLOSSARY_MIN_TERM_CHARS
    assert args.language == "auto"


def test_the_threshold_and_scorer_are_options():
    args = cli.build_parser().parse_args(
        ["chunk-glossary", "--match-threshold", "90", "--scorer", "token_sort_ratio"]
    )
    assert args.threshold == 90.0
    assert args.scorer == "token_sort_ratio"


def test_an_empty_chunks_folder_is_a_setup_error(tmp_path: Path, capsys):
    chapters = tmp_path / "book" / "01-chapters"
    chapters.mkdir(parents=True)
    (chapters / "01-a.md").write_text("Džordž.\n", encoding="utf-8")
    empty = tmp_path / "book" / "05-chunks"
    empty.mkdir()
    code = cli.main(["chunk-glossary", "--out", str(chapters), "--chunks-in", str(empty)])
    assert code == 2
    assert "run `chunk` first" in capsys.readouterr().err


def test_a_missing_glossary_is_a_setup_error(tmp_path: Path, offline_tagger, capsys):
    chapters, _chunks, glossary = _project(tmp_path)
    (glossary / "glossary.json").unlink()
    code = cli.main(["chunk-glossary", "--out", str(chapters), "--language", "sr", "--cpu"])
    assert code == 2
    assert "glossary not found" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def test_the_run_writes_one_file_per_chunk(tmp_path: Path, offline_tagger, capsys):
    chapters, _chunks, _glossary = _project(tmp_path)
    code = cli.main(["chunk-glossary", "--out", str(chapters), "--language", "sr", "--cpu"])
    printed = capsys.readouterr().out

    assert code == 0
    assert "phases      : ner over every chunk, then fuzzy retrieval for every term" in printed
    assert "entities        :" in printed
    assert "matched         :" in printed
    assert "threshold 95.0%" in printed
    output = chapters.parent / "06-chunk-glossary"
    assert output.is_dir()
    files = sorted(output.glob("*.md.json"))
    assert files
    assert (output / "_chunk_glossary_manifest.json").is_file()
    assert (output / "_chunk_glossary_trace.json").is_file()
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["stage"] == "chunk-glossary"
    assert payload["chunk"]["filename"] + ".json" == files[0].name


def test_the_threshold_reaches_the_run_and_the_printed_summary(
    tmp_path: Path, offline_tagger, capsys
):
    chapters, _chunks, _glossary = _project(tmp_path)
    code = cli.main(
        [
            "chunk-glossary",
            "--out",
            str(chapters),
            "--language",
            "sr",
            "--cpu",
            "--match-threshold",
            "90",
        ]
    )
    printed = capsys.readouterr().out

    assert code == 0
    assert "threshold 90.0%" in printed


def test_json_output_is_the_machine_readable_run(tmp_path: Path, offline_tagger, capsys):
    chapters, _chunks, _glossary = _project(tmp_path)
    code = cli.main(
        [
            "chunk-glossary",
            "--out",
            str(chapters),
            "--language",
            "sr",
            "--cpu",
            "--json",
            "--quiet",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["ok"] is True
    assert [phase["name"] for phase in payload["phases"]] == ["ner", "match"]
    assert payload["output_dir"].endswith("06-chunk-glossary")
    assert payload["settings"]["threshold"] == 95.0
    assert payload["settings"]["scorer"] == "ratio"
    assert payload["chunk_count"] >= 1
    assert payload["manifest_path"].endswith("_chunk_glossary_manifest.json")


def test_dry_run_prints_the_numbers_and_writes_nothing(tmp_path: Path, offline_tagger, capsys):
    chapters, _chunks, _glossary = _project(tmp_path)
    code = cli.main(
        [
            "chunk-glossary",
            "--out",
            str(chapters),
            "--language",
            "sr",
            "--cpu",
            "--dry-run",
        ]
    )
    printed = capsys.readouterr().out

    assert code == 0
    assert "DRY RUN — both phases ran, nothing was written." in printed
    assert not (chapters.parent / "06-chunk-glossary").exists()


def test_a_non_percentage_threshold_is_rejected(tmp_path: Path, offline_tagger, capsys):
    chapters, _chunks, _glossary = _project(tmp_path)
    code = cli.main(
        [
            "chunk-glossary",
            "--out",
            str(chapters),
            "--language",
            "sr",
            "--cpu",
            "--match-threshold",
            "101",
        ]
    )
    assert code == 2
    assert "percentage between 0 and 100" in capsys.readouterr().err
