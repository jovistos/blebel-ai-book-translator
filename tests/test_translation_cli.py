"""Offline tests for the `translate` command: flags, wiring and the files.

The provider is stubbed at the seam the runner builds it from, so the command's
real path runs — chunks and chunk-glossary are resolved from the book folder, the
runner translates, the files land — without a key or a network call. What is
pinned on top of the runner's own tests is the interface: the English → Serbian
default, the folder defaults, ``--limit``, ``--resume``, the progress lines and
the token total.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from book_translator import cli
from book_translator.core.constants import (
    DEFAULT_TRANSLATION_ATTEMPTS,
    DEFAULT_TRANSLATION_MAX_TOKENS,
)
from book_translator.stages.translation import runner as translation_runner

from test_translation_runner import FakeTranslator, _files, _project


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #
def _settings(**_kwargs):
    return SimpleNamespace(
        provider="fake",
        model="fake-1",
        base_url="http://127.0.0.1:1234/v1",
        api_key="not-needed",
        temperature=0.0,
    )


@pytest.fixture
def offline(monkeypatch):
    """Replace both the settings resolution and the model behind the translator."""
    translator = FakeTranslator()
    monkeypatch.setattr(
        "book_translator.core.settings.resolve_translation_settings", _settings
    )
    monkeypatch.setattr(translation_runner, "make_translator", lambda *a, **k: translator)
    return translator


# --------------------------------------------------------------------------- #
# the interface
# --------------------------------------------------------------------------- #
def test_the_defaults_are_english_to_serbian_and_no_limit():
    args = cli.build_parser().parse_args(["translate"])
    assert args.func is cli.cmd_translate
    assert args.source_language == "en"
    assert args.target_language == "sr"
    assert args.limit is None
    assert args.glossary_scope == "chunk"
    assert args.max_attempts == DEFAULT_TRANSLATION_ATTEMPTS == 3
    assert args.max_tokens == DEFAULT_TRANSLATION_MAX_TOKENS
    assert args.resume is False


def test_the_language_pair_limit_and_resume_are_options():
    args = cli.build_parser().parse_args(
        ["translate", "-s", "sr", "-t", "en", "--limit", "2", "--resume"]
    )
    assert (args.source_language, args.target_language) == ("sr", "en")
    assert args.limit == 2
    assert args.resume is True


def test_the_translator_flag_offers_exactly_deepseek_and_lmstudio():
    parser = cli.build_parser()
    assert parser.parse_args(["translate"]).translator is None
    assert (
        parser.parse_args(["translate", "--translator", "lmstudio"]).translator
        == "lmstudio"
    )
    # ``--provider`` stays as an alias for older command lines.
    assert (
        parser.parse_args(["translate", "--provider", "deepseek"]).translator
        == "deepseek"
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["translate", "--translator", "openai"])


def test_the_lmstudio_choice_reaches_the_translation_resolver(
    tmp_path: Path, monkeypatch, capsys
):
    _book, chapters, _chunks, _chunk_glossary = _project(tmp_path)
    seen: dict = {}

    def resolve(**kwargs):
        seen.update(kwargs)
        return _settings()

    monkeypatch.setattr(
        "book_translator.core.settings.resolve_translation_settings", resolve
    )
    monkeypatch.setattr(
        translation_runner, "make_translator", lambda *a, **k: FakeTranslator()
    )

    code = cli.main(
        [
            "translate",
            "--out",
            str(chapters),
            "--limit",
            "1",
            "--translator",
            "lmstudio",
            "-q",
        ]
    )

    assert code == 0
    assert seen["provider"] == "lmstudio"


def test_an_empty_chunks_folder_is_a_setup_error(tmp_path: Path, capsys):
    chapters = tmp_path / "book" / "02-chapters"
    chapters.mkdir(parents=True)
    (chapters / "01-a.md").write_text("George.\n", encoding="utf-8")
    empty = tmp_path / "book" / "05-chunks"
    empty.mkdir()
    code = cli.main(["translate", "--out", str(chapters), "--chunks-in", str(empty)])
    assert code == 2
    assert "run `chunk` first" in capsys.readouterr().err


def test_a_missing_chunk_glossary_is_a_setup_error(tmp_path: Path, offline, capsys):
    _book, chapters, _chunks, chunk_glossary = _project(tmp_path)
    for path in chunk_glossary.iterdir():
        path.unlink()
    chunk_glossary.rmdir()
    code = cli.main(["translate", "--out", str(chapters), "--quiet"])
    assert code == 2
    assert "run `chunk-glossary` first" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def test_the_run_writes_translations_prompts_and_narrates_tokens(
    tmp_path: Path, offline, capsys
):
    book, chapters, chunks, _chunk_glossary = _project(tmp_path)

    code = cli.main(["translate", "--out", str(chapters), "--limit", "2"])
    printed = capsys.readouterr().out

    assert code == 0
    assert "language=en -> sr" in printed
    assert "chunk-glossary=" in printed
    assert "[  1/2] ok" in printed and "[  2/2] ok" in printed
    assert "tokens used     :" in printed
    assert "scope=chunk" in printed
    assert "output dir      :" in printed
    assert "prompts         :" in printed
    translations = book / "08-translations"
    prompts = book / "08-translation-prompts"
    assert _files(translations) == _files(chunks)[:2]
    assert len(_files(prompts, "*.md.json")) == 2
    assert len(_files(prompts, "*.prompt.md")) == 2
    assert (translations / "_translation_manifest.json").is_file()
    assert (translations / "_translation_trace.json").is_file()
    assert len(offline.calls) == 2


def test_the_limit_reaches_the_run(tmp_path: Path, offline, capsys):
    _book, chapters, chunks, _chunk_glossary = _project(tmp_path)
    code = cli.main(["translate", "--out", str(chapters), "--limit", "1", "--quiet"])
    assert code == 0
    assert len(offline.calls) == 1
    assert _files(_book / "08-translations") == _files(chunks)[:1]


def test_the_target_language_reaches_the_prompt(tmp_path: Path, offline, capsys):
    _book, chapters, _chunks, _chunk_glossary = _project(tmp_path)
    code = cli.main(
        ["translate", "--out", str(chapters), "--target-language", "hr", "--limit", "1", "-q"]
    )
    assert code == 0
    assert "from English into Croatian" in offline.calls[0]["system"]


def test_explicit_output_folders_are_used(tmp_path: Path, offline):
    book, chapters, _chunks, _chunk_glossary = _project(tmp_path)
    translations = tmp_path / "elsewhere" / "08-translations"
    prompts = tmp_path / "elsewhere" / "debug"
    code = cli.main(
        [
            "translate",
            "--out",
            str(chapters),
            "--limit",
            "1",
            "--translations-out",
            str(translations),
            "--prompts-out",
            str(prompts),
            "-q",
        ]
    )
    assert code == 0
    assert len(_files(translations)) == 1
    assert len(_files(prompts, "*.md.json")) == 1
    assert not (book / "08-translations").exists()


def test_no_prompts_writes_only_the_translations(tmp_path: Path, offline):
    book, chapters, _chunks, _chunk_glossary = _project(tmp_path)
    code = cli.main(
        ["translate", "--out", str(chapters), "--limit", "1", "--no-prompts", "-q"]
    )
    assert code == 0
    assert _files(book / "08-translations")
    assert not (book / "08-translation-prompts").exists()


def test_no_manifest_writes_no_bookkeeping(tmp_path: Path, offline):
    book, chapters, _chunks, _chunk_glossary = _project(tmp_path)
    code = cli.main(
        ["translate", "--out", str(chapters), "--limit", "1", "--no-manifest", "-q"]
    )
    assert code == 0
    assert not (book / "08-translations" / "_translation_manifest.json").exists()


def test_the_glossary_scope_book_sends_the_whole_dictionary(tmp_path: Path, offline):
    _book, chapters, _chunks, _chunk_glossary = _project(tmp_path)
    code = cli.main(
        [
            "translate",
            "--out",
            str(chapters),
            "--glossary-scope",
            "book",
            "--limit",
            "1",
            "-q",
        ]
    )
    assert code == 0
    assert "Mithril → Mitril" in offline.calls[0]["user"]


def test_resume_does_not_call_the_model_for_finished_chunks(tmp_path: Path, offline):
    book, chapters, _chunks, _chunk_glossary = _project(tmp_path)
    assert cli.main(["translate", "--out", str(chapters), "--limit", "2", "-q"]) == 0
    assert len(offline.calls) == 2

    code = cli.main(
        ["translate", "--out", str(chapters), "--limit", "2", "--resume", "-q"]
    )
    assert code == 0
    assert len(offline.calls) == 2  # nothing new was asked
    manifest = json.loads(
        (book / "08-translations" / "_translation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["reused_count"] == 2


def test_a_failing_chunk_exits_one_and_lists_it(tmp_path: Path, monkeypatch, capsys):
    _book, chapters, _chunks, _chunk_glossary = _project(tmp_path)

    def explode(*, source_text: str, **_kwargs):
        if "Belgrade" in source_text:
            raise RuntimeError("provider exploded")
        return "PREVEDENO"

    offline = FakeTranslator(explode)
    monkeypatch.setattr(
        "book_translator.core.settings.resolve_translation_settings", _settings
    )
    monkeypatch.setattr(translation_runner, "make_translator", lambda *a, **k: offline)

    code = cli.main(["translate", "--out", str(chapters), "-q"])
    captured = capsys.readouterr()

    assert code == 1
    assert "FAILED chunks" in captured.err
    assert "provider exploded" in captured.err


def test_dry_run_writes_nothing(tmp_path: Path, offline):
    book, chapters, _chunks, _chunk_glossary = _project(tmp_path)
    code = cli.main(
        ["translate", "--out", str(chapters), "--limit", "1", "--dry-run", "-q"]
    )
    assert code == 0
    assert len(offline.calls) == 0  # a dry run never asks the translator
    assert not (book / "08-translations").exists()
    assert not (book / "08-translation-prompts").exists()


def test_the_json_flag_returns_the_counts_and_tokens(tmp_path: Path, offline, capsys):
    _book, chapters, _chunks, _chunk_glossary = _project(tmp_path)
    code = cli.main(["translate", "--out", str(chapters), "--limit", "1", "--json", "-q"])
    out = capsys.readouterr().out
    assert code == 0
    payload = json.loads(out)
    assert payload["target_language"] == "sr"
    assert payload["chunk_count"] == 1
    assert payload["tokens"]["total_tokens"] == 150
    assert payload["output_dir"].endswith("08-translations")


def test_quiet_prints_no_progress_lines(tmp_path: Path, offline, capsys):
    _book, chapters, _chunks, _chunk_glossary = _project(tmp_path)
    code = cli.main(["translate", "--out", str(chapters), "--limit", "2", "-q"])
    out = capsys.readouterr().out
    assert code == 0
    assert "[  1/2]" not in out


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))


def test_chunks_in_accepts_the_chunk_manifest(tmp_path: Path, offline):
    book, chapters, chunks, _chunk_glossary = _project(tmp_path)
    code = cli.main(
        [
            "translate",
            "--out",
            str(chapters),
            "--chunks-in",
            str(chunks / "_chunk_manifest.json"),
            "--limit",
            "1",
            "-q",
        ]
    )
    assert code == 0
    assert len(_files(book / "08-translations")) == 1


def test_first_chapters_translates_only_the_chunks_of_that_window(
    tmp_path: Path, offline, capsys
):
    book, chapters, chunks, _chunk_glossary = _project(tmp_path)

    code = cli.main(
        ["translate", "--out", str(chapters), "--first-chapters", "2", "--quiet"]
    )

    assert code == 0
    translated = _files(book / "08-translations")
    # The fixture chunks one chapter per file, so the window is two chunks.
    assert len(translated) == 2
    assert len(_files(chunks)) == 3


def test_first_chapters_ignores_front_matter(tmp_path: Path, offline):
    from book_translator.stages.chunking import chunk_files, run_chunking

    book = tmp_path / "book"
    chapters = book / "02-chapters"
    chapters.mkdir(parents=True)
    (chapters / "00-front-matter.md").write_text("Front matter text.\n", encoding="utf-8")
    (chapters / "01-one.md").write_text("Chapter one text here.\n", encoding="utf-8")
    (chapters / "02-two.md").write_text("Chapter two text here.\n", encoding="utf-8")
    run_chunking(str(chapters), tokenizer="word", chunk_size=5)
    chunks = book / "05-chunks"
    chunk_glossary = book / "06-chunk-glossary"
    chunk_glossary.mkdir()
    for entry in chunk_files(chunks):
        payload = {"chunk": {"filename": entry["filename"]}, "matched_terms": []}
        (chunk_glossary / f"{entry['filename']}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    code = cli.main(
        ["translate", "--out", str(chapters), "--first-chapters", "1", "--quiet"]
    )

    assert code == 0
    translated = _files(book / "08-translations")
    assert len(translated) == len(
        [e for e in chunk_files(chunks) if e["chapter_filename"] == "01-one.md"]
    )
    assert not any("front-matter" in name for name in translated)
