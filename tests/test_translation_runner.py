"""Offline tests for stage 7's runner: order, glossary, debug folder, resume.

No provider and no network: the translator is a scripted double. What is pinned
is the contract the book depends on:

* every chunk is translated once, in reading order, and lands in ``translations/``
  under the chunk's own name;
* the prompt carries exactly that chunk's glossary rows, and the source chunk;
* the *whole* input is kept in ``translation-prompts/`` (structured + readable);
* ``--limit`` translates a prefix, ``--resume`` reuses finished chunks;
* a chunk that fails is reported and does not stop the book.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from book_translator.chunking import chunk_files, run_chunking
from book_translator.core.errors import ToolError
from book_translator.glossary.extract import TokenUsage
from book_translator.translation import (
    ChunkTranslationAnswer,
    TranslationOutcome,
    load_translation_manifest,
    run_translation,
)
from book_translator.translation.runner import (
    debug_filename,
    preserve_trailing_newlines,
    prompt_filename,
)

# --------------------------------------------------------------------------- #
# doubles and fixtures
# --------------------------------------------------------------------------- #
CHAPTERS = {
    "01-one.md": "Chapter One\n\nGeorge lives in London and meets Jane.\n",
    "02-two.md": "Chapter Two\n\nJane walks to Belgrade with Mithril.\n",
    "03-three.md": "Chapter Three\n\nGeorge is alone again.\n",
}
GLOSSARY = {
    "entries": [
        {"term": "George", "category": "character", "translation": "Džordž"},
        {"term": "London", "category": "location", "translation": "London"},
        {"term": "Jane", "category": "character", "translation": "Dženi"},
        {"term": "Belgrade", "category": "location", "translation": "Beograd"},
        {"term": "Mithril", "category": "artifact", "translation": "Mitril"},
    ]
}
#: Which glossary rows stage 6 would have attached to each chapter's chunk.
CHAPTER_TERMS = {
    "01-one.md": ["George", "London", "Jane"],
    "02-two.md": ["Jane", "Belgrade", "Mithril"],
    "03-three.md": [],
}
ITEM = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150, calls=1)


class FakeTranslator:
    """A scripted translator that records every prompt it was handed."""

    name = "fake/translator"

    def __init__(self, handler=None) -> None:
        self.calls: list[dict[str, str]] = []
        self.handler = handler or (
            lambda **kwargs: "PREVEDENO: " + kwargs["source_text"].strip()
        )

    def translate(self, *, system_prompt: str, user_prompt: str, source_text: str = ""):
        self.calls.append(
            {"system": system_prompt, "user": user_prompt, "source": source_text}
        )
        result = self.handler(
            system_prompt=system_prompt, user_prompt=user_prompt, source_text=source_text
        )
        if isinstance(result, TranslationOutcome):
            return result
        if isinstance(result, Exception):
            raise result
        return TranslationOutcome(
            answer=ChunkTranslationAnswer(translation=result),
            usage=ITEM,
            attempts=1,
            method="scripted",
            raw_text=f'{{"translation": "{result[:20]}"}}',
        )


def _project(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """A stage-1/4/5/6 layout: chapters, chunks, glossary and per-chunk terms."""
    book = tmp_path / "book"
    chapters = book / "01-chapters"
    chapters.mkdir(parents=True)
    for filename, text in CHAPTERS.items():
        (chapters / filename).write_text(text, encoding="utf-8")
    chunking = run_chunking(str(chapters), tokenizer="word", chunk_size=100)
    assert chunking.ok is True

    glossary = book / "04-glossary"
    glossary.mkdir()
    (glossary / "glossary.json").write_text(
        json.dumps(GLOSSARY, ensure_ascii=False), encoding="utf-8"
    )

    chunks = book / "05-chunks"
    chunk_glossary = book / "06-chunk-glossary"
    chunk_glossary.mkdir()
    by_chapter = {row["term"]: row for row in GLOSSARY["entries"]}
    for entry in chunk_files(chunks):
        name = str(entry["filename"])
        chapter = str(entry["chapter_filename"])
        rows = [
            {
                "term": term,
                "glossary_term": term,
                "category": by_chapter[term]["category"],
                "translation": by_chapter[term]["translation"],
                "mentions": 1,
                "score": 100.0,
            }
            for term in CHAPTER_TERMS.get(chapter, [])
        ]
        payload = {
            "stage": "chunk-glossary",
            "chunk": {"filename": name},
            "matched_terms": rows,
        }
        (chunk_glossary / f"{name}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
    return book, chapters, chunks, chunk_glossary


def _files(folder: Path, pattern: str = "*.md") -> list[str]:
    return [path.name for path in sorted(folder.glob(pattern)) if not path.name.startswith("_")]


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def test_every_chunk_is_translated_once_in_reading_order(tmp_path: Path):
    book, _chapters, chunks, _chunk_glossary = _project(tmp_path)
    translator = FakeTranslator()

    result = run_translation(str(chunks), translator=translator)

    assert result.ok is True
    assert result.chunk_count == len(_files(chunks))
    assert result.processed_count == result.chunk_count
    order = [chunk["filename"] for chunk in result.chunks]
    assert order == sorted(order)
    assert len(translator.calls) == result.chunk_count
    # The default folders are siblings of chunks/.
    assert Path(result.output_dir) == book / "07-translations"
    assert Path(result.prompts_dir) == book / "07-translation-prompts"
    assert _files(Path(result.output_dir)) == _files(chunks)
    for name in _files(Path(result.output_dir)):
        assert (Path(result.output_dir) / name).read_text(encoding="utf-8").startswith(
            "PREVEDENO:"
        )
    assert (book / "07-translations" / "_translation_manifest.json").is_file()
    assert (book / "07-translations" / "_translation_trace.json").is_file()


def test_the_prompt_carries_only_that_chunks_glossary_rows(tmp_path: Path):
    _book, _chapters, chunks, _chunk_glossary = _project(tmp_path)
    translator = FakeTranslator()

    run_translation(str(chunks), translator=translator)

    first, second, third = translator.calls
    assert "George → Džordž (character)" in first["user"]
    assert "London → London (location)" in first["user"]
    assert "Mithril → Mitril" not in first["user"]
    assert "Belgrade → Beograd (location)" in second["user"]
    assert "Mithril → Mitril (artifact)" in second["user"]
    assert "George → Džordž" not in second["user"]
    assert "no entity from this chunk is in the book glossary" in third["user"]
    # The source chunk is in the prompt, and the system prompt names the pair.
    assert "George lives in London" in first["user"]
    assert "from English into Serbian" in first["system"]


def test_limit_translates_only_the_first_n_chunks(tmp_path: Path):
    _book, _chapters, chunks, _chunk_glossary = _project(tmp_path)
    translator = FakeTranslator()

    result = run_translation(str(chunks), translator=translator, limit=2)

    assert result.ok is True
    assert result.limited is True
    assert result.chunk_count == 2
    assert result.processed_count == 2
    assert len(translator.calls) == 2
    assert len(_files(Path(result.output_dir))) == 2


def test_the_debug_folder_keeps_the_whole_prompt_and_the_source_chunk(tmp_path: Path):
    _book, _chapters, chunks, _chunk_glossary = _project(tmp_path)
    result = run_translation(str(chunks), translator=FakeTranslator(), limit=1)

    prompts = Path(result.prompts_dir)
    chunk_name = _files(chunks)[0]
    debug = json.loads((prompts / debug_filename(chunk_name)).read_text(encoding="utf-8"))
    readable = (prompts / prompt_filename(chunk_name)).read_text(encoding="utf-8")

    assert debug["stage"] == "translate"
    assert debug["language"] == {"source": "en", "target": "sr"}
    assert debug["chunk"]["filename"] == chunk_name
    assert debug["prompt"]["system"].startswith("You are the translator of one book.")
    assert debug["prompt"]["user"] == debug["prompt"]["messages"][1]["content"]
    assert debug["prompt"]["messages"][0]["role"] == "system"
    assert debug["prompt"]["chars"]["total"] == len(
        debug["prompt"]["system"]
    ) + len(debug["prompt"]["user"])
    assert debug["source_chunk"]["text"] == (chunks / chunk_name).read_text(encoding="utf-8")
    assert debug["glossary"]["entry_count"] == 3
    assert [row["term"] for row in debug["glossary"]["entries"]] == [
        "George",
        "London",
        "Jane",
    ]
    assert debug["answer"]["text"].startswith("PREVEDENO:")
    assert debug["answer"]["attempts"] == 1
    assert debug["ok"] is True
    assert "# System prompt" in readable
    assert "# User prompt" in readable
    assert "George → Džordž" in readable


def test_a_failed_chunk_is_reported_and_the_book_continues(tmp_path: Path):
    _book, _chapters, chunks, _chunk_glossary = _project(tmp_path)

    def explode(*, source_text: str, **_kwargs):
        if "Belgrade" in source_text:
            raise RuntimeError("provider exploded")
        return "PREVEDENO"

    translator = FakeTranslator(explode)
    result = run_translation(str(chunks), translator=translator)

    assert result.ok is False
    assert len(result.errors) == 1
    assert "provider exploded" in result.errors[0]
    assert result.ok_count == 2
    names = _files(Path(result.output_dir))
    assert len(names) == 2
    failed = next(
        chunk for chunk in result.chunks if "Belgrade" in (chunks / chunk["filename"]).read_text()
    )
    assert failed["ok"] is False
    assert failed["error"]
    # The failed chunk still left its prompt behind, which is the point of the
    # debug folder.
    debug = json.loads(
        (Path(result.prompts_dir) / debug_filename(failed["filename"])).read_text(
            encoding="utf-8"
        )
    )
    assert debug["prompt"]["system"]
    assert debug["answer"] is None
    assert debug["error"]


def test_a_missing_chunk_glossary_file_is_not_fatal(tmp_path: Path):
    _book, _chapters, chunks, chunk_glossary = _project(tmp_path)
    victim = _files(chunks)[1]
    (chunk_glossary / f"{victim}.json").unlink()

    result = run_translation(str(chunks), translator=FakeTranslator())

    assert result.ok is True
    record = next(chunk for chunk in result.chunks if chunk["filename"] == victim)
    assert record["glossary_missing"] is True
    assert result.totals["missing_glossary_chunks"] == 1
    prompts = Path(result.prompts_dir)
    debug = json.loads((prompts / debug_filename(victim)).read_text(encoding="utf-8"))
    assert debug["glossary"]["missing"] is True
    assert "no entity from this chunk is in the book glossary" in debug["prompt"]["user"]


def test_a_missing_chunk_glossary_folder_is_a_setup_error(tmp_path: Path):
    __book, _chapters, chunks, chunk_glossary = _project(tmp_path)
    for path in chunk_glossary.iterdir():
        path.unlink()
    chunk_glossary.rmdir()

    with pytest.raises(ToolError) as info:
        run_translation(str(chunks), translator=FakeTranslator())

    assert "run `chunk-glossary` first" in str(info.value)


def test_book_scope_sends_the_whole_glossary_every_time(tmp_path: Path):
    _book, _chapters, chunks, _chunk_glossary = _project(tmp_path)
    translator = FakeTranslator()

    result = run_translation(
        str(chunks), translator=translator, glossary_scope="book"
    )

    assert result.ok is True
    assert result.glossary_entries == len(GLOSSARY["entries"])
    for call in translator.calls:
        assert "Mithril → Mitril" in call["user"]
        assert "George → Džordž" in call["user"]


def test_context_chars_send_the_previous_translation_tail(tmp_path: Path):
    _book, _chapters, chunks, _chunk_glossary = _project(tmp_path)
    translator = FakeTranslator()

    run_translation(str(chunks), translator=translator, context_chars=12)

    first, second, _third = translator.calls
    assert "previous translation" not in first["user"].lower()
    assert "context only" in second["user"]
    assert "Jane." in second["user"]


def test_the_translation_keeps_the_source_chunks_trailing_newlines(tmp_path: Path):
    _book, _chapters, chunks, _chunk_glossary = _project(tmp_path)
    translator = FakeTranslator(lambda **kwargs: "PREVEDENO")  # no trailing newlines

    result = run_translation(str(chunks), translator=translator, limit=1)

    name = _files(chunks)[0]
    source = (chunks / name).read_text(encoding="utf-8")
    written = (Path(result.output_dir) / name).read_text(encoding="utf-8")
    assert written == "PREVEDENO" + source[len(source.rstrip("\n")) :]
    assert preserve_trailing_newlines("x", "a\n\n") == "x\n\n"


def test_progress_sees_every_chunk_in_reading_order(tmp_path: Path):
    _book, _chapters, chunks, _chunk_glossary = _project(tmp_path)
    seen: list[tuple[str, int, int]] = []

    run_translation(
        str(chunks),
        translator=FakeTranslator(),
        progress=lambda record, done, total: seen.append((record["filename"], done, total)),
    )

    assert [name for name, _done, _total in seen] == _files(chunks)
    assert [done for _name, done, _total in seen] == [1, 2, 3]
    assert all(total == 3 for _name, _done, total in seen)


def test_dry_run_never_calls_the_model_and_writes_nothing(tmp_path: Path):
    book, _chapters, chunks, _chunk_glossary = _project(tmp_path)
    translator = FakeTranslator()

    result = run_translation(str(chunks), translator=translator, dry_run=True)

    assert result.ok is True
    assert len(translator.calls) == 0  # a dry run never asks the translator
    assert result.dry_run is True
    assert result.chunk_count == 3  # the planned chunks are still reported
    assert result.processed_count == 3
    assert not (book / "07-translations").exists()
    assert not (book / "07-translation-prompts").exists()
    assert result.manifest_path is None
    assert result.trace_path is None


# --------------------------------------------------------------------------- #
# resume
# --------------------------------------------------------------------------- #
def test_resume_reuses_finished_chunks_and_asks_only_for_the_rest(tmp_path: Path):
    _book, _chapters, chunks, _chunk_glossary = _project(tmp_path)
    first = FakeTranslator()
    run_translation(str(chunks), translator=first, limit=2)

    second = FakeTranslator()
    result = run_translation(str(chunks), translator=second, limit=3, resume=True)

    assert result.ok is True
    assert result.resumed is True
    assert result.reused_count == 2
    assert result.processed_count == 1
    assert len(second.calls) == 1
    assert "George is alone again" in second.calls[0]["source"]
    assert result.tokens["total_tokens"] == 3 * ITEM.total_tokens


def test_a_fully_resumed_run_needs_no_translator(tmp_path: Path):
    _book, _chapters, chunks, _chunk_glossary = _project(tmp_path)
    run_translation(str(chunks), translator=FakeTranslator())

    result = run_translation(str(chunks), translator=None, resume=True)

    assert result.ok is True
    assert result.reused_count == result.chunk_count
    assert result.processed_count == 0


def test_resume_refuses_files_from_a_different_language_pair(tmp_path: Path):
    _book, _chapters, chunks, _chunk_glossary = _project(tmp_path)
    run_translation(str(chunks), translator=FakeTranslator(), target_language="sr")

    translator = FakeTranslator()
    result = run_translation(
        str(chunks), translator=translator, target_language="hr", resume=True
    )

    assert result.reused_count == 0
    assert len(translator.calls) == result.chunk_count


def test_resume_ignores_a_manifest_entry_whose_chunk_changed(tmp_path: Path):
    _book, _chapters, chunks, _chunk_glossary = _project(tmp_path)
    run_translation(str(chunks), translator=FakeTranslator(), limit=1)

    victim = _files(chunks)[0]
    (chunks / victim).write_text("Changed text for the same filename.\n", encoding="utf-8")
    translator = FakeTranslator()
    result = run_translation(str(chunks), translator=translator, limit=1, resume=True)

    assert result.reused_count == 0
    assert len(translator.calls) == 1


def test_resume_without_a_manifest_translates_everything(tmp_path: Path):
    book, _chapters, chunks, _chunk_glossary = _project(tmp_path)
    run_translation(str(chunks), translator=FakeTranslator(), limit=1)
    manifest = book / "07-translations" / "_translation_manifest.json"
    manifest.unlink()
    translator = FakeTranslator()

    result = run_translation(str(chunks), translator=translator, limit=1, resume=True)

    assert result.reused_count == 0
    assert len(translator.calls) == 1


def test_the_manifest_records_the_whole_run_and_its_tokens(tmp_path: Path):
    book, _chapters, chunks, _chunk_glossary = _project(tmp_path)
    result = run_translation(str(chunks), translator=FakeTranslator())

    manifest = load_translation_manifest(book / "07-translations")
    assert manifest["stage"] == "translate"
    assert manifest["ok"] is True
    assert manifest["ok_count"] == manifest["chunk_count"] == 3
    assert manifest["target_language"] == "sr"
    assert manifest["tokens"]["total_tokens"] == 3 * ITEM.total_tokens
    assert len(manifest["files"]) == 3
    assert all(record["translation_sha256"] for record in manifest["files"])
    assert result.settings["glossary_scope"] == "chunk"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))


def test_a_limited_resume_does_not_drop_manifest_entries_beyond_the_limit(tmp_path: Path):
    book, _chapters, chunks, _chunk_glossary = _project(tmp_path)
    run_translation(str(chunks), translator=FakeTranslator())

    result = run_translation(
        str(chunks), translator=None, limit=1, resume=True
    )

    assert result.reused_count == 1
    manifest = load_translation_manifest(book / "07-translations")
    assert len(manifest["files"]) == 3  # the two beyond the limit are still recorded
    assert {record["filename"] for record in manifest["files"]} == set(_files(chunks))


def test_a_reused_chunk_snapshot_keeps_the_records_beyond_the_window(tmp_path: Path):
    """An interruption during a limited *resume* must not drop the other chunks.

    The reused and image-only branches write a snapshot too; they need the same
    carried records as the model-call branch, or the next resume has to pay for
    the chunks that were already translated.
    """
    book, _chapters, chunks, _chunk_glossary = _project(tmp_path)
    run_translation(str(chunks), translator=FakeTranslator())

    def stop_after_first(record: dict, done: int, total: int) -> None:
        if done == 1:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_translation(
            str(chunks),
            translator=None,
            limit=1,
            resume=True,
            progress=stop_after_first,
        )

    manifest = load_translation_manifest(book / "07-translations")
    assert {record["filename"] for record in manifest["files"]} == set(_files(chunks))


def test_resume_is_invalidated_by_a_changed_context_window(tmp_path: Path):
    _book, _chapters, chunks, _chunk_glossary = _project(tmp_path)
    run_translation(str(chunks), translator=FakeTranslator(), context_chars=0)

    translator = FakeTranslator()
    result = run_translation(
        str(chunks), translator=translator, context_chars=12, resume=True
    )

    assert result.reused_count == 0
    assert len(translator.calls) == result.chunk_count


def test_resume_is_invalidated_by_a_changed_glossary(tmp_path: Path):
    __book, _chapters, chunks, chunk_glossary = _project(tmp_path)
    run_translation(str(chunks), translator=FakeTranslator())

    victim = _files(chunks)[0]
    glossary_file = chunk_glossary / f"{victim}.json"
    payload = json.loads(glossary_file.read_text(encoding="utf-8"))
    payload["matched_terms"].append(
        {"term": "New", "glossary_term": "New", "translation": "Novo", "category": "term"}
    )
    glossary_file.write_text(json.dumps(payload), encoding="utf-8")

    translator = FakeTranslator()
    result = run_translation(str(chunks), translator=translator, resume=True)

    assert result.reused_count == result.chunk_count - 1
    assert [call["source"] for call in translator.calls] == [
        (chunks / victim).read_text(encoding="utf-8")
    ]


def test_resume_is_invalidated_by_a_changed_model(tmp_path: Path):
    class AnotherTranslator(FakeTranslator):
        name = "another/model"

    _book, _chapters, chunks, _chunk_glossary = _project(tmp_path)
    run_translation(str(chunks), translator=FakeTranslator())

    translator = AnotherTranslator()
    result = run_translation(str(chunks), translator=translator, resume=True)

    assert result.reused_count == 0
    assert len(translator.calls) == result.chunk_count


def test_resume_is_invalidated_by_a_changed_method(tmp_path: Path):
    _book, _chapters, chunks, _chunk_glossary = _project(tmp_path)
    run_translation(str(chunks), translator=FakeTranslator(), method="auto")

    translator = FakeTranslator()
    result = run_translation(str(chunks), translator=translator, method="json", resume=True)

    assert result.reused_count == 0
    assert len(translator.calls) == result.chunk_count


# --------------------------------------------------------------------------- #
# the manifest survives limited and interrupted runs
# --------------------------------------------------------------------------- #
def test_a_non_resume_limited_run_keeps_the_records_of_the_other_chunks(tmp_path: Path):
    """Starting a limited run without ``--skip-if-present`` must not erase the
    record of every chunk it does not translate: their files are still valid."""
    book, _chapters, chunks, _chunk_glossary = _project(tmp_path)
    run_translation(str(chunks), translator=FakeTranslator())

    translator = FakeTranslator()
    result = run_translation(str(chunks), translator=translator, limit=1, resume=False)

    assert result.resumed is False
    assert len(translator.calls) == 1
    manifest = load_translation_manifest(book / "07-translations")
    assert {record["filename"] for record in manifest["files"]} == set(_files(chunks))

    # The two untouched chunks are still reusable, so a full resume asks nothing.
    resumed = run_translation(str(chunks), translator=None, resume=True)
    assert resumed.reused_count == len(_files(chunks))
    assert resumed.processed_count == 0


def test_an_interrupted_run_keeps_the_records_of_the_chunks_not_yet_reached(tmp_path: Path):
    """The per-chunk snapshot, not just the final write, has to carry them."""
    book, _chapters, chunks, _chunk_glossary = _project(tmp_path)
    run_translation(str(chunks), translator=FakeTranslator())

    def stop_after_first(record: dict, done: int, total: int) -> None:
        if done == 1:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_translation(
            str(chunks),
            translator=FakeTranslator(),
            resume=False,
            progress=stop_after_first,
        )

    manifest = load_translation_manifest(book / "07-translations")
    assert {record["filename"] for record in manifest["files"]} == set(_files(chunks))


def test_a_limited_run_in_a_new_language_does_not_carry_old_records(tmp_path: Path):
    """Records in the old pair are not valid in the new one, so they are dropped
    rather than mixed into the new manifest and reused by mistake."""
    book, _chapters, chunks, _chunk_glossary = _project(tmp_path)
    run_translation(str(chunks), translator=FakeTranslator(), target_language="sr")

    run_translation(
        str(chunks), translator=FakeTranslator(), target_language="hr", limit=1
    )

    manifest = load_translation_manifest(book / "07-translations")
    assert manifest["target_language"] == "hr"
    assert len(manifest["files"]) == 1


# --------------------------------------------------------------------------- #
# images: the model sees a marker, the book gets the reference back
# --------------------------------------------------------------------------- #
IMAGE = "![Image](../00-markdown/images/rabbit.png)"


def _image_project(tmp_path: Path, text: str) -> tuple[Path, Path]:
    """A book whose single chapter is ``text``, with empty glossary files."""
    book = tmp_path / "book"
    chapters = book / "01-chapters"
    chapters.mkdir(parents=True)
    (chapters / "01-image.md").write_text(text, encoding="utf-8")
    (chapters / "_manifest.json").write_text(
        json.dumps(
            {
                "source_file": "book.md",
                "strategy": "markdown_headers",
                "files": [
                    {"filename": "01-image.md", "start_line": 1, "end_line": 9, "title": "Image"}
                ],
            }
        ),
        encoding="utf-8",
    )
    chunking = run_chunking(str(chapters), tokenizer="word", chunk_size=100)
    assert chunking.ok is True

    glossary = book / "04-glossary"
    glossary.mkdir()
    (glossary / "glossary.json").write_text(
        json.dumps({"entries": []}), encoding="utf-8"
    )
    chunk_glossary = book / "06-chunk-glossary"
    chunk_glossary.mkdir()
    for entry in chunk_files(book / "05-chunks"):
        (chunk_glossary / f"{entry['filename']}.json").write_text(
            json.dumps({"stage": "chunk-glossary", "matched_terms": []}),
            encoding="utf-8",
        )
    return book, book / "05-chunks"


def test_the_model_receives_a_marker_and_the_file_gets_the_reference_back(tmp_path: Path):
    book, chunks = _image_project(
        tmp_path,
        "# One\n\nAlice looked at the picture.\n\n" + IMAGE + "\n\nThen she ran.\n",
    )
    translator = FakeTranslator()

    result = run_translation(str(chunks), translator=translator)

    assert result.ok is True
    assert len(translator.calls) == 1
    assert "[[IMG_1]]" in translator.calls[0]["source"]
    assert "../00-markdown/images/rabbit.png" not in translator.calls[0]["source"]

    translated = (book / "07-translations" / _files(chunks)[0]).read_text(encoding="utf-8")
    assert IMAGE in translated
    assert "[[IMG_" not in translated

    debug = json.loads(
        (book / "07-translation-prompts" / f"{_files(chunks)[0]}.json").read_text(
            encoding="utf-8"
        )
    )
    assert debug["images"]["count"] == 1
    assert debug["images"]["markers"]["[[IMG_1]]"] == IMAGE
    # The source chunk in the trace stays the real one, not the protected view.
    assert IMAGE in debug["source_chunk"]["text"]


def test_an_image_only_chunk_is_passed_through_without_a_model_call(tmp_path: Path):
    book, chunks = _image_project(tmp_path, IMAGE + "\n")
    translator = FakeTranslator()

    result = run_translation(str(chunks), translator=translator)

    assert result.ok is True
    assert translator.calls == []
    translated = (book / "07-translations" / _files(chunks)[0]).read_text(encoding="utf-8")
    assert translated == IMAGE + "\n"


def test_a_translation_that_drops_an_image_marker_fails_instead_of_publishing(
    tmp_path: Path,
):
    book, chunks = _image_project(
        tmp_path,
        "# One\n\nAlice looked at the picture.\n\n" + IMAGE + "\n\nThen she ran.\n",
    )
    translator = FakeTranslator(lambda **_kwargs: "PREVEDENO bez slike")

    result = run_translation(str(chunks), translator=translator)

    assert result.ok is False
    assert any("image marker" in error for error in result.errors)
    assert _files(book / "07-translations") == []
