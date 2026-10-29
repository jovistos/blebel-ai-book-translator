"""The shared CLI flag contract: --json, --quiet, --output-root, --text-in."""

from __future__ import annotations

import argparse
import json

from book_translator import cli
from book_translator.cli import parents
from book_translator.pipeline import parser as pipeline_parser


def subparsers() -> dict[str, argparse.ArgumentParser]:
    parser = cli.build_parser()
    action = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    return dict(action.choices)


def option_strings(parser: argparse.ArgumentParser) -> set[str]:
    return {
        option
        for action in parser._actions
        for option in action.option_strings
    }


def test_every_real_command_accepts_json_quiet_and_output_root():
    choices = subparsers()
    for name, parser in choices.items():
        if name == "pipeline":
            # The real pipeline parser is tested separately; this is only the
            # discoverability stub used by `book_translator --help`.
            continue
        flags = option_strings(parser)
        assert "--json" in flags, f"{name} has no --json"
        assert "--quiet" in flags, f"{name} has no --quiet"
        if name != "tools":
            assert "--output-root" in flags, f"{name} has no --output-root"


def test_text_reading_commands_accept_text_in():
    choices = subparsers()
    for name in ("dictionary", "glossary", "chunk", "chunk-glossary", "translate"):
        assert "--text-in" in option_strings(choices[name]), f"{name} has no --text-in"


def test_pipeline_parser_has_the_shared_flags():
    flags = option_strings(pipeline_parser.build_parser())
    for flag in ("--json", "--quiet", "--output-root", "--text-in"):
        assert flag in flags


def test_parent_factories_expose_the_shared_contract():
    parser = argparse.ArgumentParser(
        parents=[
            parents.book_parent(required=False),
            parents.output_parent(),
            parents.dry_run_parent(),
            parents.text_parent(),
        ],
        add_help=False,
    )
    args = parser.parse_args(
        ["book.md", "--json", "--quiet", "--dry-run", "--text-in", "/tmp/text"]
    )
    assert args.book == "book.md"
    assert args.json is True
    assert args.quiet is True
    assert args.dry_run is True
    assert args.text_in == "/tmp/text"


def test_split_json_is_one_object_and_quiet_is_silent(tmp_path, capsys):
    source = tmp_path / "book.md"
    source.write_text("# A\n\n## B\n\ntext\n", encoding="utf-8")
    out = tmp_path / "out"
    code = cli.main(
        [
            "split",
            str(source),
            "--in-dir",
            str(tmp_path),
            "--output-root",
            str(out),
            "--dry-run",
            "--json",
        ]
    )
    printed = capsys.readouterr().out
    assert code == 0
    assert printed.lstrip().startswith("{")
    payload = json.loads(printed)
    assert payload["output_dir"] == str(out / "book" / "02-chapters")

    code = cli.main(
        [
            "split",
            str(source),
            "--in-dir",
            str(tmp_path),
            "--output-root",
            str(out),
            "--dry-run",
            "--quiet",
        ]
    )
    assert code == 0
    assert capsys.readouterr().out.strip() == ""


def test_chunk_text_in_wins_over_out(tmp_path, capsys):
    raw = tmp_path / "book" / "02-chapters"
    raw.mkdir(parents=True)
    (raw / "01-a.md").write_text("# A\n\nOne sentence.\n", encoding="utf-8")
    cleaned = tmp_path / "book" / "03-preprocessed"
    cleaned.mkdir(parents=True)
    (cleaned / "01-a.md").write_text("# A\n\nOne sentence.\n", encoding="utf-8")

    code = cli.main(
        [
            "chunk",
            "--out",
            str(raw),
            "--text-in",
            str(cleaned),
            "--output-root",
            str(tmp_path / "out"),
            "--dry-run",
            "--json",
            "--chunk-tokenizer",
            "word",
            "--chunk-size",
            "50",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["chapters_path"] == str(cleaned)


def test_chunk_ignores_a_partial_cleaned_folder(tmp_path, capsys):
    """A cleaned prefix must not shrink a text stage's chapter set."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "book.md").write_text("# Book\n", encoding="utf-8")

    raw = tmp_path / "out" / "book" / "02-chapters"
    raw.mkdir(parents=True)
    for name in ("01-a.md", "02-b.md", "03-c.md"):
        (raw / name).write_text(f"# {name}\n\nOne sentence.\n", encoding="utf-8")

    cleaned = tmp_path / "out" / "book" / "03-preprocessed"
    cleaned.mkdir(parents=True)
    (cleaned / "01-a.md").write_text("# 01-a.md\n\nOne sentence.\n", encoding="utf-8")
    (cleaned / "_preprocess_manifest.json").write_text(
        json.dumps(
            {
                "stage": "preprocess",
                "ok": True,
                "dry_run": False,
                "files": [{"filename": "01-a.md"}],
            }
        ),
        encoding="utf-8",
    )

    code = cli.main(
        [
            "chunk",
            "book.md",
            "--in-dir",
            str(input_dir),
            "--output-root",
            str(tmp_path / "out"),
            "--dry-run",
            "--json",
            "--chunk-tokenizer",
            "word",
            "--chunk-size",
            "50",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["chapters_path"] == str(raw)
    assert payload["chapter_count"] == 3
