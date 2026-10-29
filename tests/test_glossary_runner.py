"""Offline tests for the stage-4 runner: the ordered walk that builds the dictionary.

No provider, no network: the extractor is injected, so what is pinned here is the
contract the translation stage will consume. Chapters are visited body-first to
last; front matter and back matter are left out; every chapter after the first is
handed the dictionary built so far; the first rendering of a term wins and a later
disagreement is recorded instead of applied; and the token total is the sum of
every call, retries included.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, cast

import pytest
from book_translator.core.errors import ExtractionError, ToolError
from book_translator.documents.chapters import discover_chapters
from book_translator.stages.glossary import ChapterExtraction, ExtractionOutcome, TokenUsage
from book_translator.stages.glossary.runner import (
    GlossaryBook,
    GlossaryTerm,
    chapter_kind,
    chunk_sentences,
    resolve_sentences_dir,
    run_glossary,
    select_chapters,
)

# --------------------------------------------------------------------------- #
# doubles and fixtures
# --------------------------------------------------------------------------- #
ITEM = TokenUsage(input_tokens=1_000, output_tokens=100, total_tokens=1_100, calls=1)


def entry(term: str, translation: str, category: str = "character", context: str = "ctx"):
    return {
        "term": term,
        "category": category,
        "translation": translation,
        "brief_context": context,
    }


class FakeExtractor:
    """A scripted extractor that records every prompt it was handed.

    ``handler`` maps a prompt to the entries to return; returning an
    ``ExtractionOutcome`` (or raising) is how a retry, a custom usage or a failed
    chapter is scripted.
    """

    def __init__(self, handler, *, usage: TokenUsage | None = None, name: str = "fake") -> None:
        self.handler = handler
        self.usage = usage or ITEM
        self.name = name
        self.calls: list[tuple[str, str]] = []

    def extract(self, *, system_prompt: str, user_prompt: str) -> ExtractionOutcome:
        self.calls.append((system_prompt, user_prompt))
        result = self.handler(system_prompt, user_prompt)
        if isinstance(result, ExtractionOutcome):
            return result
        return ExtractionOutcome(
            extraction=ChapterExtraction.model_validate({"entries": result}),
            usage=self.usage,
            attempts=1,
            method="scripted",
            raw_text="{}",
        )

    @property
    def prompts(self) -> list[str]:
        return [user for _, user in self.calls]


def by_chapter(mapping: dict[str, list[dict]]):
    """A handler that answers according to the chapter filename in the prompt."""

    def handler(_system: str, user: str):
        for filename, entries in mapping.items():
            if f"({filename})" in user:
                return entries
        return []

    return handler


#: Two body chapters plus the two blocks the run must skip by default.
CHAPTERS = {
    "00-front-matter.md": "## Don't You Forget About Me\n\nCopyright page. Publisher: Pan Books.\n",
    "01-chapter-1.md": "Alexandra Potter lives in London.\n",
    "02-chapter-2.md": "Potter meets Charlotte in Brighton.\n",
}
KINDS = {
    "00-front-matter.md": "front-matter",
    "01-chapter-1.md": "chapter",
    "02-chapter-2.md": "chapter",
}


def make_book(
    tmp_path: Path,
    *,
    chapters: dict[str, str] | None = None,
    kinds: dict[str, str] | None = None,
    reduced: dict[str, str] | None = None,
    name: str = "book",
) -> Path:
    """A minimal stage-1 folder plus its stage-3 ``reduced-unique/`` sibling."""
    chapters = chapters or CHAPTERS
    kinds = kinds if kinds is not None else KINDS
    root = tmp_path / name
    folder = root / "02-chapters"
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, (filename, text) in enumerate(chapters.items(), start=1):
        (folder / filename).write_text(text, encoding="utf-8")
        entries.append(
            {
                "filename": filename,
                "kind": kinds.get(filename, "chapter"),
                "start_line": index,
                "end_line": index,
                "title": filename,
                "lines": len(text.splitlines()),
                "first_line": text.splitlines()[0],
            }
        )
    (folder / "_manifest.json").write_text(
        json.dumps(
            {"source_file": "Don't You Forget About Me - Alexandra Potter.md", "files": entries},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    unique = root / "04-reduced-unique"
    unique.mkdir(parents=True, exist_ok=True)
    for filename, text in (reduced if reduced is not None else chapters).items():
        (unique / filename).write_text(text, encoding="utf-8")
    return folder


def read_json(path: str | Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# which chapters, in which order
# --------------------------------------------------------------------------- #
def test_front_and_back_matter_are_skipped_but_still_reported(tmp_path):
    chapters = {
        "00-front-matter.md": "Copyright.\n",
        "01-chapter-1.md": "Alexandra Potter lives in London.\n",
        "02-back-matter.md": "About the author.\n",
    }
    kinds = {
        "00-front-matter.md": "front-matter",
        "01-chapter-1.md": "chapter",
        "02-back-matter.md": "back-matter",
    }
    folder = make_book(tmp_path, chapters=chapters, kinds=kinds)

    extractor = FakeExtractor(by_chapter({"01-chapter-1.md": [entry("Potter", "Poter")]}))
    result = run_glossary(str(folder), str(folder.parent / "04-glossary"), extractor=extractor)

    assert len(extractor.calls) == 1  # only the body chapter was mined
    # Neither skipped block was sent to the model, not even as evidence.
    assert all("Copyright" not in prompt and "About the author" not in prompt for prompt in extractor.prompts)
    assert [record["filename"] for record in result.chapters] == [
        "00-front-matter.md",
        "01-chapter-1.md",
        "02-back-matter.md",
    ]
    skipped = {record["filename"]: record["skip_reason"] for record in result.chapters}
    assert skipped == {
        "00-front-matter.md": "front matter",
        "01-chapter-1.md": None,
        "02-back-matter.md": "back matter",
    }
    assert result.skipped_count == 2
    assert result.body_chapter_count == 1
    assert result.entry_count == 1


def test_the_skip_can_be_overridden_for_either_block(tmp_path):
    folder = make_book(tmp_path)
    extractor = FakeExtractor(lambda s, u: [])
    result = run_glossary(
        str(folder),
        str(folder.parent / "04-glossary"),
        extractor=extractor,
        include_front_matter=True,
    )
    assert result.skipped_count == 0
    assert result.processed_count == 3
    assert "Copyright page" in extractor.calls[0][1]


def test_chapter_kind_comes_from_the_manifest_then_from_the_filename(tmp_path):
    folder = make_book(tmp_path)
    chapters = discover_chapters(folder)
    assert [chapter_kind(chapter) for chapter in chapters] == [
        "front-matter",
        "chapter",
        "chapter",
    ]

    # A folder with no manifest: the filename decides.
    bare = tmp_path / "bare"
    bare.mkdir()
    for name in ("00-front-matter.md", "01-a.md", "17-back-matter.md"):
        (bare / name).write_text("text\n", encoding="utf-8")
    kinds = [chapter_kind(chapter) for chapter in discover_chapters(bare)]
    assert kinds == ["front-matter", "chapter", "back-matter"]


def test_select_chapters_keeps_the_skipped_files_in_the_plan(tmp_path):
    folder = make_book(tmp_path)
    plans = select_chapters(discover_chapters(folder))
    assert [plan.will_process for plan in plans] == [False, True, True]
    assert [plan.index for plan in plans] == [1, 2, 3]
    # The body ordinal is what the prompt and the entries call the chapter.
    assert [plan.body_index for plan in plans] == [0, 1, 2]


def test_limit_stops_at_the_first_n_chapters(tmp_path):
    folder = make_book(tmp_path)
    extractor = FakeExtractor(by_chapter({"01-chapter-1.md": [entry("Potter", "Poter")]}))
    result = run_glossary(
        str(folder), str(folder.parent / "04-glossary"), extractor=extractor, limit=2
    )
    # The first two files are the front matter and chapter 1; only one is mined.
    assert len(extractor.calls) == 1
    assert result.processed_count == 1
    assert result.limited is True
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["limited"] is True
    assert manifest["chapter_count"] == len(result.chapters)


# --------------------------------------------------------------------------- #
# the dictionary grows, in order
# --------------------------------------------------------------------------- #
def test_each_chapter_is_handed_the_dictionary_the_earlier_ones_built(tmp_path):
    folder = make_book(tmp_path)
    extractor = FakeExtractor(
        by_chapter(
            {
                "01-chapter-1.md": [
                    entry("Alexandra Potter", "Aleksandra Poter", "character", "the narrator; female")
                ],
                "02-chapter-2.md": [entry("Charlotte", "Šarlot")],
            }
        )
    )
    result = run_glossary(str(folder), str(folder.parent / "04-glossary"), extractor=extractor)

    first, second = extractor.prompts
    # The book's own chapter numbering, not the file numbering (front matter is 1).
    assert "Chapter 1 (01-chapter-1.md)" in first
    assert "Chapter 2 (02-chapter-2.md)" in second
    # The first call is told the dictionary is empty.
    assert "(empty — no entity has been locked in yet)" in first
    # The second call carries the first chapter's locked rendering, verbatim —
    # and only the pair, not the context it was justified with.
    assert "Alexandra Potter → Aleksandra Poter" in second
    assert "the narrator; female" not in second
    assert "Charlotte" in second  # ...and its own evidence
    assert result.entry_count == 2
    # The system prompt is the same for both chapters: the task, not the state.
    assert extractor.calls[0][0] == extractor.calls[1][0]
    assert "being translated from English into Serbian" in extractor.calls[0][0]


def test_every_entry_keeps_the_chapter_it_came_from(tmp_path):
    folder = make_book(tmp_path)
    extractor = FakeExtractor(
        by_chapter(
            {
                "01-chapter-1.md": [entry("London", "London", "location")],
                "02-chapter-2.md": [entry("Brighton", "Brajson", "location")],
            }
        )
    )
    result = run_glossary(str(folder), str(folder.parent / "04-glossary"), extractor=extractor)
    provenance = {
        item["term"]: (item["chapter_index"], item["chapter_filename"])
        for item in result.entries
    }
    # The front matter is file 1 but not a "chapter" the model was asked about, so
    # the body chapters are numbered from 1.
    assert provenance == {
        "London": (1, "01-chapter-1.md"),
        "Brighton": (2, "02-chapter-2.md"),
    }


def test_the_first_rendering_wins_and_a_later_disagreement_is_recorded(tmp_path):
    folder = make_book(tmp_path)
    extractor = FakeExtractor(
        by_chapter(
            {
                "01-chapter-1.md": [entry("Potter", "Poter")],
                # Chapter 2 re-states a locked term, once the same way and once
                # differently: the first is a repeat, the second a conflict.
                "02-chapter-2.md": [entry("Potter", "Poter"), entry("Potter", "Potter")],
            }
        )
    )
    result = run_glossary(str(folder), str(folder.parent / "04-glossary"), extractor=extractor)

    assert result.entry_count == 1
    assert [item["translation"] for item in result.entries] == ["Poter"]
    assert result.entries[0]["also_returned_in"] == ["02-chapter-2.md"]
    assert len(result.conflicts) == 1
    conflict = result.conflicts[0]
    assert (conflict["term"], conflict["locked"], conflict["proposed"]) == (
        "Potter",
        "Poter",
        "Potter",
    )
    assert conflict["chapter_filename"] == "02-chapter-2.md"
    # The chapter record keeps the same count as the run-level list. Records are
    # one per file, so the front matter occupies index 0.
    assert result.chapters[2]["repeats"] == 1
    assert result.chapters[2]["conflicts"] == 1


def test_a_term_returned_twice_in_one_chapter_is_one_entry():
    book = GlossaryBook()
    book.merge(
        [
            ChapterExtraction.model_validate({"entries": [entry("X", "Iks")]}).entries[0],
            ChapterExtraction.model_validate({"entries": [entry("x", "IKS")]}).entries[0],
        ],
        chapter_index=1,
        chapter_filename="01.md",
    )
    assert len(book) == 1
    assert [item.translation for item in book] == ["Iks"]


# --------------------------------------------------------------------------- #
# long chapters, failed chapters, tokens
# --------------------------------------------------------------------------- #
def test_a_long_chapter_is_split_and_the_parts_share_the_dictionary(tmp_path):
    body = "\n".join(f"Marko and Ana line {index} in Novi Sad." for index in range(60))
    folder = make_book(
        tmp_path,
        chapters={"01-chapter-1.md": body + "\n"},
        kinds={"01-chapter-1.md": "chapter"},
    )
    calls: list[str] = []

    def handler(_system: str, user: str):
        calls.append(user)
        if "part 1 of" in user:
            return [entry("Marko", "Marko")]
        return [entry("Ana", "Ana")]

    extractor = FakeExtractor(handler)
    result = run_glossary(
        str(folder), str(folder.parent / "04-glossary"), extractor=extractor, chunk_chars=200
    )

    parts = result.chapters[0]["chunks"]
    assert parts > 1
    assert len(extractor.calls) == parts  # one model call per part
    assert f"part 2 of {parts}" in calls[1]
    assert "Marko → Marko" in calls[1]  # part 1's entry reaches part 2
    assert result.entry_count == 2
    assert result.chapters[0]["models_called"] == parts


def test_chunk_sentences_keeps_line_boundaries_and_never_drops_a_line():
    lines = ["a" * 10, "b" * 10, "c" * 10]
    assert chunk_sentences(lines, 25) == [["a" * 10, "b" * 10], ["c" * 10]]
    assert chunk_sentences([], 25) == [[]]
    assert chunk_sentences(lines, 0) == [lines]
    # A single line longer than the limit is its own part rather than lost.
    assert chunk_sentences(["x" * 100, "y"], 10) == [["x" * 100], ["y"]]


def test_one_failed_chapter_does_not_stop_the_book(tmp_path):
    folder = make_book(tmp_path)

    def handler(_system: str, user: str):
        if "01-chapter-1.md" in user:
            raise ExtractionError("no valid extraction after 3 attempt(s): boom")
        return [entry("Charlotte", "Šarlot")]

    extractor = FakeExtractor(handler)
    result = run_glossary(str(folder), str(folder.parent / "04-glossary"), extractor=extractor)

    assert result.ok is False
    assert result.processed_count == 2  # both body chapters were attempted
    assert result.entry_count == 1  # the failure cost the dictionary one chapter
    assert any("01-chapter-1.md" in error and "boom" in error for error in result.errors)
    assert len(extractor.calls) == 2


def test_the_request_count_is_attempts_not_chapters(tmp_path):
    """A retried chapter costs two requests; the timing must say so."""
    folder = make_book(tmp_path)
    retried = TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15, calls=1)

    def handler(system: str, user: str):
        return ExtractionOutcome(
            extraction=ChapterExtraction.model_validate({"entries": []}),
            usage=retried,
            attempts=2,
            method="json",
        )

    extractor = FakeExtractor(handler)
    result = run_glossary(str(folder), str(folder.parent / "04-glossary"), extractor=extractor)
    assert result.chapters[1]["attempts"] == 2
    assert result.timing["model_calls"] == 4  # two chapters, two attempts each
    assert result.tokens["calls"] == 2  # but only two responses were billed


def test_the_token_total_is_every_call_of_every_chapter(tmp_path):
    folder = make_book(tmp_path)
    heavy = TokenUsage(input_tokens=500, output_tokens=50, total_tokens=550, calls=2)
    extractor = FakeExtractor(lambda s, u: [entry("Potter", "Poter")], usage=heavy)
    result = run_glossary(str(folder), str(folder.parent / "04-glossary"), extractor=extractor)

    assert result.tokens == {
        "input_tokens": 1_000,
        "output_tokens": 100,
        "total_tokens": 1_100,
        "calls": 4,
        "chapters": 2,
    }
    assert result.chapters[1]["tokens"]["total_tokens"] == 550  # the first body chapter
    assert result.chapters[0]["tokens"]["total_tokens"] == 0  # the skipped front matter
    assert result.timing["model_calls"] == 2


# --------------------------------------------------------------------------- #
# artifacts
# --------------------------------------------------------------------------- #
def test_the_four_artifacts_describe_the_same_dictionary(tmp_path):
    folder = make_book(tmp_path)
    out = folder.parent / "04-glossary"
    extractor = FakeExtractor(
        by_chapter(
            {
                "01-chapter-1.md": [entry("Potter", "Poter", "character", "the narrator")],
                "02-chapter-2.md": [entry("Brighton", "Brajson", "location")],
            }
        )
    )
    result = run_glossary(str(folder), str(out), extractor=extractor)

    assert Path(result.glossary_path).name == "glossary.json"
    assert Path(result.markdown_path).name == "glossary.md"
    assert Path(result.manifest_path).name == "_glossary_manifest.json"
    assert Path(result.trace_path).name == "_glossary_trace.json"

    glossary = read_json(result.glossary_path)
    assert glossary["stage"] == "glossary"
    assert glossary["source_language"] == "en"
    assert glossary["target_language"] == "sr"
    assert glossary["entry_count"] == 2
    assert [item["term"] for item in glossary["entries"]] == ["Potter", "Brighton"]
    assert glossary["categories"] == {"character": 1, "location": 1}
    assert glossary["tokens"]["total_tokens"] == 2_200
    assert glossary["source_file"].endswith(".md")

    manifest = read_json(result.manifest_path)
    assert manifest["body_chapter_count"] == 2
    assert manifest["skipped_count"] == 1
    assert [item["filename"] for item in manifest["files"]] == list(CHAPTERS)
    assert manifest["files"][1]["added"] == 1
    assert manifest["files"][0]["skipped"] is True

    trace = read_json(result.trace_path)
    assert [item["filename"] for item in trace["detail"]] == list(CHAPTERS)
    assert trace["detail"][1]["added_terms"][0]["term"] == "Potter"
    assert trace["detail"][1]["raw_answers"] == ["{}"]

    markdown = Path(result.markdown_path).read_text(encoding="utf-8")
    assert "# Glossary" in markdown
    assert "en → sr" in markdown
    assert "| Potter | Poter | 01-chapter-1.md |" in markdown
    assert "## location" in markdown


def test_dry_run_asks_the_model_and_writes_nothing(tmp_path):
    folder = make_book(tmp_path)
    out = folder.parent / "04-glossary"
    extractor = FakeExtractor(by_chapter({"01-chapter-1.md": [entry("Potter", "Poter")]}))
    result = run_glossary(str(folder), str(out), extractor=extractor, dry_run=True)

    assert len(extractor.calls) == 0  # a dry run never asks the model
    assert result.entry_count == 0  # ...so no dictionary is built in memory
    assert result.dry_run is True
    assert result.processed_count > 0  # but the pending chapters are reported
    assert result.glossary_path is None
    assert not out.exists()


# --------------------------------------------------------------------------- #
# resuming
# --------------------------------------------------------------------------- #
def test_resume_reuses_the_dictionary_and_only_pays_for_what_is_missing(tmp_path):
    folder = make_book(tmp_path)
    out = folder.parent / "04-glossary"
    first = FakeExtractor(
        by_chapter({"01-chapter-1.md": [entry("Potter", "Poter", "character", "narrator")]})
    )
    # The first run stops after chapter 1 (the front matter is file 1, chapter 1
    # is file 2), which is what a run that died mid-book leaves behind.
    run_glossary(str(folder), str(out), extractor=first, limit=2)

    second = FakeExtractor(
        by_chapter(
            {
                "01-chapter-1.md": [entry("Potter", "Poter")],
                "02-chapter-2.md": [entry("Charlotte", "Šarlot")],
            }
        )
    )
    result = run_glossary(str(folder), str(out), extractor=second, skip_if_present=True)

    assert len(second.calls) == 1  # only chapter 2 was still pending
    assert result.resumed is True
    # The chapter already in the manifest was not asked about again.
    assert result.reused_count == 1
    assert result.processed_count == 1
    assert [item["term"] for item in result.entries] == ["Potter", "Charlotte"]
    # The first run's tokens are still part of the book's total.
    assert result.tokens["total_tokens"] == 2_200
    assert result.tokens["calls"] == 2


def test_resume_starts_over_when_the_chapter_folder_changes(tmp_path):
    """Raw and cleaned chapters share filenames but not text, so work is rebuilt."""
    folder = make_book(tmp_path)
    out = folder.parent / "04-glossary"
    first = FakeExtractor(
        by_chapter(
            {
                "01-chapter-1.md": [entry("Potter", "Poter")],
                "02-chapter-2.md": [entry("Charlotte", "Šarlot")],
            }
        )
    )
    run_glossary(str(folder), str(out), extractor=first)
    assert read_json(out / "_glossary_manifest.json")["chapters_path"] == str(folder)

    cleaned = folder.parent / "03-preprocessed"
    cleaned.mkdir()
    for path in folder.glob("*.md"):
        shutil.copy(path, cleaned / path.name)

    second = FakeExtractor(
        by_chapter(
            {
                "01-chapter-1.md": [entry("Potter", "Poter")],
                "02-chapter-2.md": [entry("Charlotte", "Šarlot")],
            }
        )
    )
    result = run_glossary(str(cleaned), str(out), extractor=second, skip_if_present=True)

    assert result.resumed is False
    assert len(second.calls) == 2  # both chapters were asked again
    assert result.reused_count == 0
    assert read_json(out / "_glossary_manifest.json")["chapters_path"] == str(cleaned)


def test_resume_refuses_to_overwrite_a_glossary_whose_file_went_missing(tmp_path):
    """A manifest says a dictionary existed; an empty file is damage, not a reset."""
    folder = make_book(tmp_path)
    out = folder.parent / "04-glossary"
    first = FakeExtractor(by_chapter({"01-chapter-1.md": [entry("Potter", "Poter")],
                                      "02-chapter-2.md": [entry("Charlotte", "Šarlot")]}))
    run_glossary(str(folder), str(out), extractor=first)
    glossary = out / "glossary.json"
    assert read_json(glossary)["entry_count"] == 2

    # Damage the dictionary but keep its manifest.
    glossary.write_text(json.dumps({"entries": []}), encoding="utf-8")

    with pytest.raises(ToolError) as excinfo:
        run_glossary(str(folder), str(out), skip_if_present=True)
    assert "refusing to overwrite" in str(excinfo.value)
    assert "records 2" in str(excinfo.value)
    # Nothing was written over the damaged file.
    assert read_json(glossary)["entries"] == []


def test_every_finished_chapter_is_persisted_immediately(tmp_path):
    """A killed run must cost the chapter in flight, not the whole dictionary."""
    folder = make_book(tmp_path)
    out = folder.parent / "04-glossary"

    def handler(_system: str, user: str):
        if "02-chapter-2.md" in user:
            raise KeyboardInterrupt  # what Ctrl-C actually does mid-call
        return [entry("Potter", "Poter")]

    with pytest.raises(KeyboardInterrupt):
        run_glossary(str(folder), str(out), extractor=FakeExtractor(handler))

    # Chapter 1 was snapshotted before chapter 2 started, manifest included.
    assert read_json(out / "glossary.json")["entry_count"] == 1
    assert read_json(out / "glossary.json")["entries"][0]["term"] == "Potter"
    assert read_json(out / "_glossary_manifest.json")["files"][1]["added"] == 1
    # ...and the manifest is what makes a resumed run start at chapter 2.
    assert (out / "glossary.md").is_file()


def test_a_run_where_every_chapter_failed_leaves_the_existing_glossary_alone(tmp_path):
    folder = make_book(tmp_path)
    out = folder.parent / "04-glossary"
    first = FakeExtractor(lambda s, u: [entry("Potter", "Poter")])
    run_glossary(str(folder), str(out), extractor=first)
    before = read_json(out / "glossary.json")

    def always_fails(_system: str, _user: str):
        raise ExtractionError("no valid extraction after 3 attempt(s): boom")

    with pytest.raises(ToolError) as excinfo:
        run_glossary(str(folder), str(out), extractor=FakeExtractor(always_fails))
    assert "every chapter failed" in str(excinfo.value)
    assert read_json(out / "glossary.json") == before  # untouched


def test_a_fresh_run_where_every_chapter_failed_writes_no_glossary(tmp_path):
    folder = make_book(tmp_path)
    out = folder.parent / "04-glossary"

    def always_fails(_system: str, _user: str):
        raise ExtractionError("boom")

    with pytest.raises(ToolError):
        run_glossary(str(folder), str(out), extractor=FakeExtractor(always_fails))
    assert not (out / "glossary.json").exists()


def test_a_fully_resumed_run_needs_no_model_at_all(tmp_path):
    folder = make_book(tmp_path)
    out = folder.parent / "04-glossary"
    extractor = FakeExtractor(lambda s, u: [entry("Potter", "Poter")])
    run_glossary(str(folder), str(out), extractor=extractor)

    result = run_glossary(str(folder), str(out), skip_if_present=True)

    assert result.ok is True
    assert result.processed_count == 0
    assert result.reused_count == 2
    assert result.entry_count == 1  # read back from glossary.json
    assert result.tokens["calls"] == 2  # the previous run's calls, carried over


def test_a_fully_resumed_run_never_resolves_provider_settings(tmp_path):
    """No pending chapter means no key, no network, no provider client."""
    folder = make_book(tmp_path)
    out = folder.parent / "04-glossary"
    run_glossary(str(folder), str(out), extractor=FakeExtractor(lambda s, u: [entry("Potter", "Poter")]))

    def must_not_resolve():
        raise AssertionError("a resumed run must not resolve provider settings")

    result = run_glossary(
        str(folder), str(out), skip_if_present=True, settings_factory=must_not_resolve
    )
    assert result.processed_count == 0
    assert result.entry_count == 1


def test_a_pending_chapter_resolves_settings_once(tmp_path, monkeypatch):
    from types import SimpleNamespace

    folder = make_book(tmp_path)
    resolved: list[int] = []

    def factory():
        resolved.append(1)
        return SimpleNamespace(
            provider="deepseek",
            model="deepseek-flash",
            base_url="https://api.deepseek.com/v1",
            api_key="not-needed",
            temperature=0.0,
            timeout=30.0,
            max_retries=1,
        )

    monkeypatch.setattr(
        "book_translator.stages.glossary.runner.make_extractor",
        lambda settings, **kwargs: FakeExtractor(lambda s, u: [entry("Potter", "Poter")]),
    )
    result = run_glossary(
        str(folder), str(folder.parent / "04-glossary"), settings_factory=factory
    )
    assert len(resolved) == 1
    assert result.provider == "deepseek"
    assert result.entry_count == 1


# --------------------------------------------------------------------------- #
# finding the evidence
# --------------------------------------------------------------------------- #
def test_the_reduced_fallback_covers_a_book_whose_unique_folder_was_never_written(tmp_path):
    folder = make_book(tmp_path)
    for child in (folder.parent / "04-reduced-unique").glob("*.md"):
        child.unlink()
    reduced = folder.parent / "04-reduced"
    reduced.mkdir()
    (reduced / "01-chapter-1.md").write_text("Alexandra Potter lives in London.\n", encoding="utf-8")

    assert resolve_sentences_dir(folder, sentences_dir=None, sentences="unique") == reduced

    extractor = FakeExtractor(lambda s, u: [entry("Potter", "Poter")])
    result = run_glossary(str(folder), str(folder.parent / "04-glossary"), extractor=extractor)
    assert result.sentences_dir == str(reduced)
    assert result.entry_count == 1


def test_sentences_reduced_reads_the_bigger_folder(tmp_path):
    folder = make_book(tmp_path)
    reduced = folder.parent / "04-reduced"
    reduced.mkdir()
    (reduced / "01-chapter-1.md").write_text("Everything with a name.\n", encoding="utf-8")
    assert resolve_sentences_dir(folder, sentences_dir=None, sentences="reduced") == reduced


def test_without_either_reduction_folder_the_run_says_so(tmp_path):
    folder = make_book(tmp_path)
    for name in ("04-reduced-unique", "04-reduced"):
        target = folder.parent / name
        if target.exists():
            for child in target.glob("*.md"):
                child.unlink()
            target.rmdir()
    with pytest.raises(ToolError) as excinfo:
        resolve_sentences_dir(folder, sentences_dir=None, sentences="unique")
    assert "run `dictionary` first" in str(excinfo.value)
    with pytest.raises(ToolError):
        resolve_sentences_dir(folder, sentences_dir=None, sentences="everything")


def test_a_missing_sentence_file_fails_only_that_chapter(tmp_path):
    folder = make_book(tmp_path)
    (folder.parent / "04-reduced-unique" / "02-chapter-2.md").unlink()
    extractor = FakeExtractor(lambda s, u: [entry("Potter", "Poter")])
    result = run_glossary(str(folder), str(folder.parent / "04-glossary"), extractor=extractor)

    assert result.ok is False
    assert len(extractor.calls) == 1
    assert any("no reduced chapter" in error for error in result.errors)


# --------------------------------------------------------------------------- #
# the dictionary object itself
# --------------------------------------------------------------------------- #
def test_a_book_can_be_rebuilt_from_its_own_entries():
    book = GlossaryBook()
    book.merge(
        ChapterExtraction.model_validate({"entries": [entry("Potter", "Poter")]}).entries,
        chapter_index=3,
        chapter_filename="03.md",
    )
    rebuilt = GlossaryBook([GlossaryTerm.from_dict(item) for item in book.to_list()])
    assert [item.term for item in rebuilt] == ["Potter"]
    assert rebuilt.categories() == {"character": 1}
    assert rebuilt.to_list()[0]["chapter_index"] == 3


def test_progress_is_reported_once_per_processed_chapter(tmp_path):
    folder = make_book(tmp_path)
    extractor = FakeExtractor(lambda s, u: [])
    seen: list[tuple[str, int, int]] = []
    run_glossary(
        str(folder),
        str(folder.parent / "04-glossary"),
        extractor=extractor,
        progress=lambda record, done, total: seen.append((record["filename"], done, total)),
    )
    assert seen == [("01-chapter-1.md", 1, 2), ("02-chapter-2.md", 2, 2)]
