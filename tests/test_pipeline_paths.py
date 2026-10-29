"""Tests for the one path value the pipeline and the stage CLIs share."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from book_translator.pipeline.paths import StagePaths


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path


def ns(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def test_pipeline_defaults(root: Path):
    source = root / "data" / "input" / "my-book.md"
    paths = StagePaths.for_pipeline(
        ns(out=None),
        source=source,
        input_dir=root / "data" / "input",
        output_root=root / "out",
    )
    book = root / "out" / "my-book"
    assert paths.book_dir == book
    assert paths.base == book / "02-chapters"
    assert paths.chapters_dir == book / "02-chapters"
    assert paths.markdown_dir == book / "01-markdown"
    assert paths.preprocessed_dir == book / "03-preprocessed"
    assert paths.reduced_dir == book / "04-reduced"
    assert paths.unique_dir == book / "04-reduced-unique"
    assert paths.glossary_dir == book / "04-glossary"
    assert paths.chunks_dir == book / "05-chunks"
    assert paths.chunk_glossary_dir == book / "06-chunk-glossary"
    assert paths.translations_dir == book / "08-translations"
    assert paths.translation_prompts_dir == book / "08-translation-prompts"
    assert paths.matter_dir == book / "07-matter"
    assert paths.matter_prompts_dir == book / "07-matter-prompts"
    assert paths.translated_chapters_dir == book / "09-translated-chapters"
    assert paths.postprocess_dir == book / "10-postprocessed-chapters"
    assert paths.translated_book_path == book / "11-translated-book.md"
    assert paths.epub_path == book / "12-translated-book.epub"


def test_pipeline_out_moves_the_anchor_and_its_defaults(root: Path):
    source = root / "data" / "input" / "my-book.md"
    base = root / "somewhere" / "02-chapters"
    paths = StagePaths.for_pipeline(
        ns(out=str(base)),
        source=source,
        input_dir=root / "data" / "input",
        output_root=root / "out",
    )
    assert paths.base == base
    assert paths.chapters_dir == base
    assert paths.preprocessed_dir == root / "somewhere" / "03-preprocessed"
    assert paths.chunks_dir == root / "somewhere" / "05-chunks"
    # The book folder and the Markdown folder stay under --output-root.
    assert paths.book_dir == root / "out" / "my-book"
    assert paths.markdown_dir == root / "out" / "my-book" / "01-markdown"


def test_every_explicit_out_wins(root: Path):
    source = root / "book.md"
    paths = StagePaths.for_pipeline(
        ns(
            out=None,
            preprocessed_out=root / "p",
            reduced_out=root / "r",
            unique_out=root / "u",
            glossary_out=root / "g",
            chunks_out=root / "c",
            chunk_glossary_out=root / "cg",
            translations_out=root / "t",
            translation_prompts_out=root / "tp",
            matter_out=root / "m",
            matter_prompts_out=root / "mp",
            translated_chapters_out=root / "tc",
            postprocess_out=root / "pp",
            book_out=root / "book.md",
            epub_out=root / "book.epub",
        ),
        source=source,
        output_root=root / "out",
    )
    assert paths.preprocessed_dir == root / "p"
    assert paths.reduced_dir == root / "r"
    assert paths.unique_dir == root / "u"
    assert paths.glossary_dir == root / "g"
    assert paths.chunks_dir == root / "c"
    assert paths.chunk_glossary_dir == root / "cg"
    assert paths.translations_dir == root / "t"
    assert paths.translation_prompts_dir == root / "tp"
    assert paths.matter_dir == root / "m"
    assert paths.matter_prompts_dir == root / "mp"
    assert paths.translated_chapters_dir == root / "tc"
    assert paths.postprocess_dir == root / "pp"
    assert paths.translated_book_path == root / "book.md"
    assert paths.epub_path == root / "book.epub"


def test_chunks_out_moves_the_downstream_defaults(root: Path):
    source = root / "book.md"
    paths = StagePaths.for_pipeline(
        ns(out=None, chunks_out=root / "custom-chunks"),
        source=source,
        output_root=root / "out",
    )
    assert paths.chunks_dir == root / "custom-chunks"
    assert paths.chunk_glossary_dir == root / "custom-chunks" / "06-chunk-glossary"
    assert paths.translations_dir == root / "custom-chunks" / "08-translations"
    assert paths.translated_chapters_dir == root / "custom-chunks" / "09-translated-chapters"


def test_from_base_chapters_folder(root: Path):
    base = root / "book" / "02-chapters"
    paths = StagePaths.from_base(base)
    assert paths.base == base
    assert paths.book_dir == root / "book"
    assert paths.chapters_dir == base
    assert paths.chunks_dir == root / "book" / "05-chunks"
    assert paths.postprocess_dir == root / "book" / "10-postprocessed-chapters"


def test_from_base_chunks_folder_steps_to_the_book_folder(root: Path):
    base = root / "book" / "05-chunks"
    paths = StagePaths.from_base(base)
    assert paths.book_dir == root / "book"
    assert paths.chapters_dir == root / "book" / "02-chapters"
    assert paths.chunks_dir == base


def test_from_base_postprocessed_folder(root: Path):
    base = root / "book" / "10-postprocessed-chapters"
    paths = StagePaths.from_base(base)
    assert paths.book_dir == root / "book"
    assert paths.postprocess_dir == base
    assert paths.translated_book_path == root / "book" / "11-translated-book.md"
    assert paths.epub_path == root / "book" / "12-translated-book.epub"


def test_with_source_keeps_the_layout(root: Path):
    paths = StagePaths.from_base(root / "book" / "02-chapters")
    moved = paths.with_source(root / "book" / "01-markdown" / "book.md")
    assert moved.source == root / "book" / "01-markdown" / "book.md"
    assert moved.chapters_dir == paths.chapters_dir
    assert moved.book_dir == paths.book_dir


def test_to_dict_is_json_shaped(root: Path):
    paths = StagePaths.from_base(root / "book" / "02-chapters")
    payload = paths.to_dict()
    assert payload["chapters_dir"] == str(root / "book" / "02-chapters")
    assert payload["book"] is None
    assert set(payload) >= {"book_dir", "chunks_dir", "epub_path", "output_root"}
