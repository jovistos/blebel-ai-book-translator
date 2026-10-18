"""Stage 5: the chunks must be exact slices that concatenate back to the book.

Every test runs offline: ``tokenizer="word"`` is one of chonkie's built-in
tokenizers, so no ``tiktoken`` vocabulary and no Hugging Face download is needed.
The real chunk length default (1000 tokens) and the real tokenizer are exercised
separately by the CLI and by a smoke test that accepts either the exact tokenizer
or the documented local fallback.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from book_splitter import cli
from book_splitter.chunking import (
    load_chunk_manifest,
    reconstruct_book,
    reconstruct_chapter,
    run_chunking,
)
from book_splitter.chunking.boundaries import boundary_kind
from book_splitter.core.constants import CHUNK_MANIFEST_FILENAME, CHUNK_TRACE_FILENAME

TEXTS = {
    "01-prva.md": (
        "# Prva\n\n"
        "Prvi pasus ima tri rečenice. Druga rečenica je ovde. Treća je kratka.\n\n"
        "Drugi pasus je nov. I on ima dve rečenice.\n"
    ),
    "02-druga.md": (
        "Kratak pasus.\n\n"
        "Ovo je veoma dugačka rečenica koja treba da ostane cela kada god može. "
        "Ona ima dovoljno reči da se ne uklopi u mali chunk. "
        "Zato će je chunker prelomiti tek na granici rečenice.\n\n"
        "Kraj drugog poglavlja.\n"
    ),
}


@pytest.fixture
def chapters(tmp_path: Path) -> Path:
    """Two ordered chapter files, the shape stage 1 writes."""
    folder = tmp_path / "chapters"
    folder.mkdir()
    for name, text in TEXTS.items():
        (folder / name).write_text(text, encoding="utf-8")
    (folder / "_manifest.json").write_text(
        json.dumps(
            {
                "source_file": "book.md",
                "strategy": "markdown_headers",
                "files": [
                    {"filename": name, "start_line": 1, "end_line": 5, "title": name}
                    for name in TEXTS
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return folder


def _book_text() -> str:
    return "".join(TEXTS[name] for name in sorted(TEXTS))


# --------------------------------------------------------------------------- #
# the runner
# --------------------------------------------------------------------------- #
def test_chunks_are_exact_slices_that_reconstruct_the_book(chapters: Path):
    result = run_chunking(str(chapters), tokenizer="word", chunk_size=30)

    assert result.ok is True
    assert result.chapter_count == 2
    assert result.chunk_count == len(result.files)
    assert result.chunk_count > 2
    assert result.errors == []

    manifest = load_chunk_manifest(chapters.parent / "chunks")
    assert manifest["stage"] == "chunk"
    assert manifest["chunk_count"] == result.chunk_count
    assert manifest["verification"]["chunks_are_source_slices"] is True
    assert manifest["verification"]["chapters_reconstruct_byte_for_byte"] is True
    assert manifest["verification"]["book_reconstructs_byte_for_byte"] is True
    assert manifest["verification"]["chunk_size_violations"] == 0

    # Global order and chapter membership are explicit, not inferred from a glob.
    assert [entry["global_index"] for entry in manifest["files"]] == list(range(result.chunk_count))
    by_chapter: dict[str, list[dict]] = {}
    for entry in manifest["files"]:
        by_chapter.setdefault(entry["chapter_filename"], []).append(entry)
        assert entry["start_char"] < entry["end_char"]
        assert entry["sha256"]
        assert entry["chapter_id"]
    for entries in by_chapter.values():
        assert [entry["chunk_index"] for entry in entries] == list(range(len(entries)))
        assert len({entry["filename"] for entry in entries}) == len(entries)

    # And the actual files are the text.
    assert reconstruct_chapter(chapters.parent / "chunks", "01-prva.md") == TEXTS["01-prva.md"]
    assert reconstruct_chapter(chapters.parent / "chunks", "02-druga") == TEXTS["02-druga.md"]
    assert reconstruct_book(chapters.parent / "chunks") == _book_text()


def test_filenames_sort_into_reading_order(chapters: Path):
    result = run_chunking(str(chapters), tokenizer="word", chunk_size=20)
    folder = chapters.parent / "chunks"
    markdown = sorted(path.name for path in folder.glob("*.md"))
    assert markdown == [entry["filename"] for entry in result.files]
    assert markdown[0].startswith("00000-")
    assert markdown[-1].startswith(f"{result.chunk_count - 1:05d}-")


def test_chunk_size_is_an_option_and_is_a_hard_limit(chapters: Path):
    big = run_chunking(str(chapters), tokenizer="word", chunk_size=100)
    small = run_chunking(str(chapters), tokenizer="word", chunk_size=12)

    assert big.ok and small.ok
    assert small.chunk_count > big.chunk_count
    for entry in small.files:
        assert entry["token_count"] <= 12, entry
        assert entry["bytes"] > 0 or entry["start_char"] == entry["end_char"]
    assert small.verification["chunk_size_violations"] == 0
    assert reconstruct_book(chapters.parent / "chunks") == _book_text()


def test_paragraphs_and_sentences_are_not_broken_when_they_fit(chapters: Path):
    result = run_chunking(str(chapters), tokenizer="word", chunk_size=40)

    assert result.ok is True
    starts = {entry["boundary_before"] for entry in result.files}
    assert starts <= {"chapter-start", "paragraph"}
    ends = {entry["boundary_after"] for entry in result.files}
    assert ends <= {"paragraph", "chapter-end"}
    assert result.verification["word_breaking_chunks"] == 0
    assert result.verification["sentence_safe_chunks"] == result.chunk_count


def test_a_paragraph_that_is_too_long_is_split_without_losing_a_character(chapters: Path):
    result = run_chunking(str(chapters), tokenizer="word", chunk_size=8)

    assert result.ok is True
    # The long paragraph has to be cut below the paragraph level; the cut may be
    # at a sentence or at a word, never inside one.
    assert {"sentence", "word"} & {entry["boundary_before"] for entry in result.files}
    assert result.verification["word_breaking_chunks"] == 0
    assert reconstruct_chapter(chapters.parent / "chunks", "02-druga.md") == TEXTS["02-druga.md"]


def test_dry_run_computes_and_verifies_but_writes_nothing(chapters: Path):
    result = run_chunking(str(chapters), tokenizer="word", chunk_size=20, dry_run=True)

    assert result.ok is True
    assert result.chunk_count > 0
    assert result.dry_run is True
    assert result.manifest_path is None
    assert result.trace_path is None
    assert not (chapters.parent / "chunks").exists()


def test_skip_if_present_reuses_every_chunk_file(chapters: Path):
    first = run_chunking(str(chapters), tokenizer="word", chunk_size=20)
    assert first.ok and first.skipped is False

    folder = chapters.parent / "chunks"
    before = {path.name: path.read_bytes() for path in folder.glob("*.md")}

    second = run_chunking(str(chapters), tokenizer="word", chunk_size=20, skip_if_present=True)

    assert second.ok is True
    assert second.skipped is True
    assert second.processed_count == 0
    assert second.skipped_count == 2
    assert second.chunk_count == first.chunk_count
    assert {path.name: path.read_bytes() for path in folder.glob("*.md")} == before
    assert reconstruct_book(folder) == _book_text()


def test_a_changed_chapter_is_rechunked_on_a_reused_run(chapters: Path):
    run_chunking(str(chapters), tokenizer="word", chunk_size=20)
    changed = TEXTS["02-druga.md"] + "\nDodatni pasus posle izmene.\n"
    (chapters / "02-druga.md").write_text(changed, encoding="utf-8")

    result = run_chunking(str(chapters), tokenizer="word", chunk_size=20, skip_if_present=True)

    assert result.ok is True
    assert result.skipped_count == 1  # the first chapter was untouched
    assert result.processed_count == 1
    assert reconstruct_chapter(chapters.parent / "chunks", "02-druga.md") == changed


def test_a_different_chunk_size_forces_a_fresh_cut(chapters: Path):
    run_chunking(str(chapters), tokenizer="word", chunk_size=20)

    result = run_chunking(str(chapters), tokenizer="word", chunk_size=10, skip_if_present=True)

    assert result.ok is True
    assert result.skipped is False
    assert result.processed_count == 2
    assert all(entry["token_count"] <= 10 for entry in result.files)


def test_manifest_and_trace_are_written_next_to_the_chunks(chapters: Path):
    result = run_chunking(str(chapters), tokenizer="word", chunk_size=20)
    folder = chapters.parent / "chunks"

    assert (folder / CHUNK_MANIFEST_FILENAME).is_file()
    assert (folder / CHUNK_TRACE_FILENAME).is_file()
    trace = json.loads((folder / CHUNK_TRACE_FILENAME).read_text(encoding="utf-8"))
    assert trace["chunk_count"] == result.chunk_count
    assert trace["settings"]["chunk_size"] == 20
    assert trace["settings"]["tokenizer"]["requested"] == "word"
    assert trace["files"][0]["chapter_filename"] == "01-prva.md"


def test_no_manifest_flag_leaves_only_chunk_files(chapters: Path):
    result = run_chunking(str(chapters), tokenizer="word", chunk_size=20, write_manifest=False)
    folder = chapters.parent / "chunks"

    assert result.ok is True
    assert not (folder / CHUNK_MANIFEST_FILENAME).exists()
    assert not (folder / CHUNK_TRACE_FILENAME).exists()
    # Without a manifest the folder still reconstructs, by filename order.
    assert reconstruct_book(folder) == _book_text()


def test_an_empty_chapter_is_reported_without_breaking_the_run(tmp_path: Path):
    folder = tmp_path / "chapters"
    folder.mkdir()
    (folder / "01-empty.md").write_text("", encoding="utf-8")
    (folder / "02-text.md").write_text("Jedna rečenica. Druga.\n", encoding="utf-8")

    result = run_chunking(str(folder), tokenizer="word", chunk_size=20)

    assert result.ok is True
    assert result.chapter_count == 2
    assert result.chapters[0]["chunk_count"] == 0
    assert result.chapters[0]["reconstructs_byte_for_byte"] is True
    assert reconstruct_chapter(folder.parent / "chunks", "01-empty.md") == ""


def test_boundary_kind_names_the_strongest_boundary():
    text = "# Title\n\nPrva rečenica. Druga rečenica.\n\nNovi pasus.\n"
    assert boundary_kind(text, 0) == "chapter-start"
    assert boundary_kind(text, len("# Title\n\n")) == "paragraph"
    assert boundary_kind(text, len("# Title\n\nPrva rečenica. ")) == "sentence"
    assert boundary_kind(text, len("# Title\n\nPrva reč")) == "hard"
    assert boundary_kind(text, len(text)) == "chapter-end"


# --------------------------------------------------------------------------- #
# the CLI
# --------------------------------------------------------------------------- #
def test_cli_chunk_writes_a_manifest_and_prints_the_numbers(chapters: Path, capsys):
    code = cli.main(
        [
            "chunk",
            "--out",
            str(chapters),
            "--chunk-tokenizer",
            "word",
            "--chunk-size",
            "20",
        ]
    )
    out = capsys.readouterr().out

    assert code == 0
    assert "chunks          :" in out
    assert "reconstruction  : chapters=True book=True" in out
    assert (chapters.parent / "chunks" / CHUNK_MANIFEST_FILENAME).is_file()


def test_cli_chunk_json_is_machine_readable(chapters: Path, capsys):
    code = cli.main(
        [
            "chunk",
            "--out",
            str(chapters),
            "--chunk-tokenizer",
            "word",
            "--chunk-size",
            "20",
            "--json",
        ]
    )
    out = capsys.readouterr().out
    payload = json.loads(out[out.index("{") :])

    assert code == 0
    assert payload["ok"] is True
    assert payload["settings"]["chunk_size"] == 20
    assert payload["verification"]["book_reconstructs_byte_for_byte"] is True
    assert payload["files"][0]["chapter_filename"] == "01-prva.md"


def test_cli_chunk_rejects_a_non_positive_chunk_size(chapters: Path, capsys):
    code = cli.main(["chunk", "--out", str(chapters), "--chunk-size", "0"])
    captured = capsys.readouterr()

    assert code == 2
    assert "--chunk-size" in captured.err


def test_cli_chunk_without_chapters_explains_the_missing_stage(tmp_path: Path, capsys):
    empty = tmp_path / "chapters"
    empty.mkdir()
    code = cli.main(["chunk", "--out", str(empty)])
    captured = capsys.readouterr()

    assert code == 2
    assert "run `split` first" in captured.err
