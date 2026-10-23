"""Offline tests for the `glossary` command: flags, wiring and the token report.

The model is stubbed at the seam the runner builds it from, so the command's real
path is exercised — chapters are discovered, the sentences folder is resolved, the
run happens, the four artifacts are written — without a key, a network call or a
cent of spend. What is pinned on top of the runner's own tests is the interface:
the default language pair, the flags that reach the runner, and the one line the
request asked for, the total token count.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

import pytest
from book_translator import cli
from book_translator.core.errors import ExtractionError
from book_translator.core.settings import LLMSettings
from book_translator.stages.glossary import ExtractionOutcome, TokenUsage
from book_translator.stages.glossary.models import ChapterExtraction

from test_glossary_runner import FakeExtractor, entry, make_book


def settings() -> LLMSettings:
    return LLMSettings(
        provider="deepseek",
        model="deepseek-flash",
        base_url="https://api.deepseek.com/v1",
        api_key="not-needed",
    )


@pytest.fixture
def offline_model(monkeypatch):
    """Patch the two seams a real run needs: credentials and the chat client.

    The command imports ``resolve_settings`` from the settings module at call
    time, so patching it there is what keeps this test keyless and offline.
    """
    monkeypatch.setattr(
        "book_translator.core.settings.resolve_settings", lambda **kwargs: settings()
    )
    extractor = FakeExtractor(
        lambda system, user: [entry("Potter", "Poter", "character", "the narrator")],
        usage=TokenUsage(input_tokens=1_000, output_tokens=100, total_tokens=1_100, calls=1),
    )
    monkeypatch.setattr("book_translator.stages.glossary.runner.make_extractor", lambda *a, **k: extractor)
    return extractor


# --------------------------------------------------------------------------- #
# the interface
# --------------------------------------------------------------------------- #
def test_english_to_serbian_is_the_default_pair_and_the_safe_defaults_hold():
    args = cli.build_parser().parse_args(["glossary"])
    assert args.source_language == "en"
    assert args.target_language == "sr"
    assert args.sentences == "unique"
    assert args.extraction_method == "auto"
    assert args.max_attempts == 3
    assert args.include_front_matter is False
    assert args.include_back_matter is False
    assert args.resume is False
    assert args.func is cli.cmd_glossary


def test_the_language_pair_is_configurable():
    args = cli.build_parser().parse_args(
        ["glossary", "-s", "sr", "-t", "en", "--sentences", "reduced"]
    )
    assert (args.source_language, args.target_language, args.sentences) == ("sr", "en", "reduced")


def test_an_empty_chapters_folder_is_a_setup_error_not_a_traceback(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    code = cli.main(["glossary", str(empty)])
    assert code == 2
    assert "no chapter files" in capsys.readouterr().err


def test_the_run_reaches_the_model_and_writes_the_artifacts(tmp_path, offline_model, capsys):
    folder = make_book(tmp_path)
    out = tmp_path / "book" / "04-glossary"
    code = cli.main(["glossary", str(folder), "--glossary-out", str(out)])

    assert code == 0
    printed = capsys.readouterr().out
    assert "language=en -> sr" in printed
    assert "glossary        : en -> sr" in printed
    assert (out / "glossary.json").is_file()
    assert (out / "glossary.md").is_file()
    assert (out / "_glossary_manifest.json").is_file()
    assert (out / "_glossary_trace.json").is_file()
    assert offline_model.calls  # the model was really asked


def test_the_total_token_count_is_printed(tmp_path, offline_model, capsys):
    folder = make_book(tmp_path)
    cli.main(["glossary", str(folder), "--glossary-out", str(tmp_path / "book" / "04-glossary")])
    printed = capsys.readouterr().out
    assert "tokens used     : 2,200 total (2,000 in, 200 out) over 2 chapter(s), 2 call(s)" in printed


def test_json_output_is_the_machine_readable_run(tmp_path, offline_model, capsys):
    folder = make_book(tmp_path)
    code = cli.main(
        [
            "glossary",
            str(folder),
            "--glossary-out",
            str(tmp_path / "book" / "04-glossary"),
            "--json",
            "--quiet",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_language"] == "en"
    assert payload["target_language"] == "sr"
    assert payload["entry_count"] == 1
    assert payload["tokens"]["total_tokens"] == 2_200  # both body chapters
    assert payload["entries"][0]["translation"] == "Poter"
    assert payload["glossary_path"].endswith("glossary.json")


def test_quiet_drops_the_whole_transcript(tmp_path, offline_model, capsys):
    folder = make_book(tmp_path)
    out = tmp_path / "book" / "04-glossary"
    cli.main(["glossary", str(folder), "--glossary-out", str(out), "--quiet"])
    printed = capsys.readouterr().out
    # ``--quiet`` now follows the shared contract: no header, no progress and no
    # closing summary on stdout. The artifacts are still written.
    assert printed.strip() == ""
    assert (out / "glossary.json").is_file()


def test_dry_run_writes_nothing(tmp_path, offline_model, capsys):
    folder = make_book(tmp_path)
    out = tmp_path / "book" / "04-glossary"
    code = cli.main(["glossary", str(folder), "--glossary-out", str(out), "--dry-run"])
    assert code == 0
    assert "DRY RUN" in capsys.readouterr().out
    assert not out.exists()


def test_a_limit_prints_the_partial_dictionary(tmp_path, offline_model, capsys):
    folder = make_book(tmp_path)
    cli.main(
        [
            "glossary",
            str(folder),
            "--glossary-out",
            str(tmp_path / "book" / "04-glossary"),
            "--limit",
            "2",
            "--max-context-entries",
            "5",
        ]
    )
    printed = capsys.readouterr().out
    assert "1/1 body chapter(s) processed" in printed


def test_a_failed_chapter_exits_non_zero(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "book_translator.core.settings.resolve_settings", lambda **kwargs: settings()
    )

    def handler(_system: str, user: str):
        if "01-chapter-1.md" in user:
            raise ExtractionError("no valid extraction after 3 attempt(s): boom")
        return [entry("Potter", "Poter")]

    monkeypatch.setattr(
        "book_translator.stages.glossary.runner.make_extractor", lambda *a, **k: FakeExtractor(handler)
    )
    folder = make_book(tmp_path)
    code = cli.main(
        ["glossary", str(folder), "--glossary-out", str(tmp_path / "book" / "04-glossary")]
    )
    assert code == 1
    printed = capsys.readouterr()
    assert "FAILED chapters" in printed.err
    assert "01-chapter-1.md: no valid extraction after 3 attempt(s): boom" in printed.err


def test_the_summary_names_conflicts_and_resumed_chapters(capsys):
    class Result:
        source_language = "en"
        target_language = "sr"
        provider = "deepseek"
        model = "deepseek-flash"
        ok_count = 3
        body_chapter_count = 3
        reused_count = 1
        skipped_count = 2
        entry_count = 4
        categories: ClassVar[Any] = {"character": 3, "location": 1}
        conflicts: ClassVar[Any] = [
            {
                "term": "Potter",
                "locked": "Poter",
                "proposed": "Potter",
                "chapter_filename": "12-chapter-12.md",
            }
        ]
        tokens: ClassVar[Any] = {
            "input_tokens": 900,
            "output_tokens": 100,
            "total_tokens": 1_000,
            "calls": 4,
            "chapters": 3,
        }
        timing: ClassVar[Any] = {
            "run_seconds": 12.5,
            "model_seconds": 11.0,
            "seconds_per_chapter": 3.7,
            "model_calls": 4,
        }
        resumed = True
        output_dir = "/tmp/glossary"
        glossary_path = "/tmp/glossary/glossary.json"
        markdown_path = "/tmp/glossary/glossary.md"
        manifest_path = "/tmp/glossary/_glossary_manifest.json"
        trace_path = "/tmp/glossary/_glossary_trace.json"
        errors: ClassVar[Any] = []
        ok = True

    cli._print_glossary_summary(Result())
    printed = capsys.readouterr().out
    assert "conflicts       : 1 (the first rendering stands)" in printed
    assert "locked 'Poter'" in printed
    assert "tokens used     : 1,000 total (900 in, 100 out) over 3 chapter(s), 4 call(s)" in printed
    assert "resumed         : 1 chapter(s) reused from the manifest" in printed
    assert "character=3, location=1" in printed


def test_an_extraction_outcome_from_the_stub_is_enough_to_prove_the_wiring():
    """A guard on the test double itself: it answers with a validated extraction."""
    extractor = FakeExtractor(lambda s, u: [entry("X", "Iks")])
    outcome = extractor.extract(system_prompt="s", user_prompt="u")
    assert isinstance(outcome, ExtractionOutcome)
    assert isinstance(outcome.extraction, ChapterExtraction)
    assert outcome.entries[0].translation == "Iks"


def test_the_help_text_states_the_default_pair():
    parser = cli.build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if getattr(action, "choices", None) and "glossary" in action.choices
    )
    gloss = subparsers.choices["glossary"].format_help()
    assert "default: en" in gloss
    assert "default: sr" in gloss
    assert "skipped by default" in gloss


# --------------------------------------------------------------------------- #
# default input: the cleaned folder wins when stage 2 wrote it
# --------------------------------------------------------------------------- #
def _stage_folders(tmp_path):
    book = tmp_path / "book"
    raw = book / "01-chapters"
    raw.mkdir(parents=True)
    (raw / "01-a.md").write_text("# A\n\ntext\n", encoding="utf-8")
    cleaned = book / "02-preprocessed"
    cleaned.mkdir()
    (cleaned / "01-a.md").write_text("# A\n\ntext\n", encoding="utf-8")
    return book, raw, cleaned


def test_the_text_stages_default_to_the_cleaned_folder(tmp_path, monkeypatch):
    book, raw, cleaned = _stage_folders(tmp_path)
    (cleaned / "_preprocess_manifest.json").write_text(
        json.dumps(
            {
                "stage": "preprocess",
                "ok": True,
                "dry_run": False,
                "files": [{"filename": "01-a.md", "source_lines": 3}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli.common, "resolve_book_path", lambda *a, **k: book / "book.md")
    monkeypatch.setattr(cli.common, "resolve_output_dir", lambda *a, **k: raw)

    assert cli._resolve_text_dir(None, None, None) == cleaned


def test_the_cleaned_folder_is_ignored_without_a_completed_manifest(tmp_path, monkeypatch):
    book, raw, cleaned = _stage_folders(tmp_path)
    # A dry run's folder, or one whose manifest is still missing, is not a
    # trustworthy default: the raw chapters are used instead.
    (cleaned / "_preprocess_manifest.json").write_text(
        json.dumps({"stage": "preprocess", "ok": True, "dry_run": True, "files": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli.common, "resolve_book_path", lambda *a, **k: book / "book.md")
    monkeypatch.setattr(cli.common, "resolve_output_dir", lambda *a, **k: raw)

    assert cli._resolve_text_dir(None, None, None) == raw


def test_an_explicit_out_always_wins_over_the_cleaned_folder(tmp_path, monkeypatch):
    _book, raw, _cleaned = _stage_folders(tmp_path)

    assert cli._resolve_text_dir(None, None, str(raw)) == raw
