"""Offline tests for stage 8: translated chunks -> translated chapters.

The stage is deterministic, so the tests can run it over real stage-5 chunks and
hand-written "translations". What is pinned:

* a chapter file is exactly its translated chunks concatenated in chunk order;
* ``--first-chapters N`` assembles only the first N chapters;
* ``--skip-if-present`` reuses a chapter whose bytes already match, so growing
  the window only writes the new chapters;
* a chapter with an untranslated chunk is reported incomplete and skipped, not
  written as a truncated chapter;
* the output folder is separate from the source chapters.
"""

from __future__ import annotations

from pathlib import Path

from book_translator.documents.window import (
    chunk_limit_for_chapters,
    combine_limits,
)
from book_translator.stages.assembly import run_assembly
from book_translator.stages.chunking import chunk_files, run_chunking

CHAPTERS = {
    "01-one.md": "Chapter One\n\nGeorge lives in London and meets Jane.\n",
    "02-two.md": "Chapter Two\n\nJane walks to Belgrade with Mithril.\n",
    "03-three.md": "Chapter Three\n\nGeorge is alone again, and the room is quiet.\n",
}


def _project(tmp_path: Path, *, chunk_size: int = 4) -> tuple[Path, Path, Path, Path]:
    """Chapters, chunks and one fake translation per chunk, in a book folder."""
    book = tmp_path / "book"
    chapters = book / "01-chapters"
    chapters.mkdir(parents=True)
    for filename, text in CHAPTERS.items():
        (chapters / filename).write_text(text, encoding="utf-8")
    chunking = run_chunking(str(chapters), tokenizer="word", chunk_size=chunk_size)
    assert chunking.ok is True

    chunks = book / "05-chunks"
    translations = book / "07-translations"
    translations.mkdir()
    for entry in chunk_files(chunks):
        name = str(entry["filename"])
        (translations / name).write_text(f"PREVEDENO:{name}\n", encoding="utf-8")
    return book, chapters, chunks, translations


def _chapter_files(folder: Path) -> list[str]:
    return sorted(
        path.name for path in folder.glob("*.md") if not path.name.startswith("_")
    )


def _expected_chapter(chunks: Path, translations: Path, chapter: str) -> str:
    entries = [
        entry
        for entry in chunk_files(chunks)
        if str(entry.get("chapter_filename") or "") == chapter
    ]
    entries.sort(key=lambda entry: int(entry.get("chunk_index") or 0))
    return "".join((translations / str(entry["filename"])).read_text(encoding="utf-8") for entry in entries)


def test_each_chapter_is_the_concatenation_of_its_translated_chunks(tmp_path: Path):
    book, _chapters, chunks, translations = _project(tmp_path)

    result = run_assembly(str(chunks), str(translations))

    assert result.ok is True
    assert result.chapter_count == 3
    output = Path(result.output_dir)
    assert output == book / "08-translated-chapters"
    assert _chapter_files(output) == sorted(CHAPTERS)
    for filename in CHAPTERS:
        assert (output / filename).read_text(encoding="utf-8") == _expected_chapter(
            chunks, translations, filename
        )
    # The source chapters are untouched and the output is a separate folder.
    assert (book / "01-chapters" / "01-one.md").read_text(encoding="utf-8") == CHAPTERS["01-one.md"]
    assert (output / "_translated_chapters_manifest.json").is_file()
    assert (output / "_translated_chapters_trace.json").is_file()


def test_first_chapters_assembles_only_that_prefix(tmp_path: Path):
    _book, _chapters, chunks, translations = _project(tmp_path)

    result = run_assembly(
        str(chunks), str(translations), first_chapters=2, skip_if_present=True
    )

    assert result.ok is True
    assert result.chapter_count == 2
    assert result.first_chapters == 2
    assert "03-three.md" not in _chapter_files(Path(result.output_dir))


def test_skip_if_present_reuses_old_chapters_and_writes_only_new_ones(tmp_path: Path):
    _book, _chapters, chunks, translations = _project(tmp_path)

    first = run_assembly(str(chunks), str(translations), first_chapters=1)
    assert first.assembled_count == 1
    assert first.chapters[0]["reused"] is False

    second = run_assembly(
        str(chunks), str(translations), first_chapters=3, skip_if_present=True
    )

    assert second.ok is True
    assert second.reused_count == 1
    assert second.assembled_count == 2
    assert second.chapters[0]["reused"] is True
    assert [chapter["reused"] for chapter in second.chapters] == [True, False, False]


def test_an_incomplete_chapter_is_skipped_not_truncated(tmp_path: Path):
    _book, _chapters, chunks, translations = _project(tmp_path)
    victim = next(
        entry
        for entry in chunk_files(chunks)
        if entry.get("chapter_filename") == "02-two.md"
    )
    (translations / str(victim["filename"])).unlink()

    result = run_assembly(str(chunks), str(translations))

    assert result.ok is True
    assert result.incomplete_count == 1
    output = Path(result.output_dir)
    assert "02-two.md" not in _chapter_files(output)
    assert "01-one.md" in _chapter_files(output)
    assert "03-three.md" in _chapter_files(output)


def test_dry_run_writes_no_chapter(tmp_path: Path):
    _book, _chapters, chunks, translations = _project(tmp_path)

    result = run_assembly(str(chunks), str(translations), dry_run=True)

    assert result.ok is True
    assert result.assembled_count == 3
    assert result.manifest_path is None
    assert _chapter_files(Path(result.output_dir)) == []


def test_a_chapter_window_maps_to_the_chunk_prefix(tmp_path: Path):
    _book, _chapters, chunks, _translations = _project(tmp_path)

    total = len(chunk_files(chunks))
    first_two = chunk_limit_for_chapters(chunks, 2)
    assert first_two is not None
    assert 0 < first_two < total
    assert chunk_limit_for_chapters(chunks, None) is None
    assert combine_limits(first_two, 3) == min(first_two, 3)
    assert combine_limits(None, None) is None


def test_the_concatenate_command_finds_the_book_folders(tmp_path: Path):
    from book_translator import cli

    book, chapters, _chunks, _translations = _project(tmp_path)

    code = cli.main(
        ["concatenate", "--out", str(chapters), "--first-chapters", "2", "--quiet"]
    )

    assert code == 0
    output = book / "08-translated-chapters"
    assert _chapter_files(output) == ["01-one.md", "02-two.md"]
    assert (output / "_translated_chapters_manifest.json").is_file()


def test_a_chapter_window_counts_body_chapters_not_front_matter(tmp_path: Path):
    from book_translator.documents.window import (
        body_chapter_names,
        chapter_window_limit_for_path,
        is_body_chapter_name,
    )

    chapters = tmp_path / "book" / "01-chapters"
    chapters.mkdir(parents=True)
    (chapters / "00-front-matter.md").write_text("Front matter.\n", encoding="utf-8")
    for name in ("01-one.md", "02-two.md", "03-three.md"):
        (chapters / name).write_text(f"{name} text.\n", encoding="utf-8")

    assert is_body_chapter_name("00-front-matter.md") is False
    assert is_body_chapter_name("01-one.md") is True
    assert body_chapter_names(
        ["00-front-matter.md", "01-one.md", "02-two.md", "back-matter.md"]
    ) == ["01-one.md", "02-two.md"]
    # Two body chapters need front matter + those two files.
    assert chapter_window_limit_for_path(chapters, 2) == 3
    assert chapter_window_limit_for_path(chapters, 1) == 2
    # More than the book has: the whole folder.
    assert chapter_window_limit_for_path(chapters, 99) == 4
    # The old raw limit still means exactly that many files.
    assert chapter_window_limit_for_path(chapters, None, 2) == 2


def test_assembly_first_chapters_keeps_front_and_back_matter(tmp_path: Path):
    book = tmp_path / "book"
    chapters = book / "01-chapters"
    chapters.mkdir(parents=True)
    (chapters / "00-front-matter.md").write_text("Front matter.\n", encoding="utf-8")
    for name in ("01-one.md", "02-two.md"):
        (chapters / name).write_text(f"{name} text.\n", encoding="utf-8")
    (chapters / "99-back-matter.md").write_text("Back matter.\n", encoding="utf-8")
    run_chunking(str(chapters), tokenizer="word", chunk_size=3)
    chunks = book / "05-chunks"
    translations = book / "07-translations"
    translations.mkdir()
    for entry in chunk_files(chunks):
        (translations / str(entry["filename"])).write_text(
            "T:" + str(entry["filename"]) + "\n", encoding="utf-8"
        )

    windowed = run_assembly(str(chunks), str(translations), first_chapters=1)
    assert windowed.ok is True
    assert _chapter_files(Path(windowed.output_dir)) == [
        "00-front-matter.md",
        "01-one.md",
        "99-back-matter.md",
    ]

    whole = run_assembly(
        str(chunks), str(translations), output_dir=book / "all-chapters"
    )
    assert "00-front-matter.md" in _chapter_files(Path(whole.output_dir))
    assert "99-back-matter.md" in _chapter_files(Path(whole.output_dir))
