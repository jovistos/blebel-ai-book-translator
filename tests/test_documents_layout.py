"""Tests for the one stage-folder registry and the shared chunk/manifest readers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from book_translator.core.constants import (
    CHUNK_GLOSSARY_DIRNAME,
    CHUNKS_DIRNAME,
    GLOSSARY_DIRNAME,
    PREPROCESSED_DIRNAME,
    TRANSLATIONS_DIRNAME,
)
from book_translator.core.errors import ToolError
from book_translator.documents.chunks import (
    chunk_files,
    chunk_glossary_filename,
    discover_chunk_inputs,
    resolve_chunks_dir,
    resolve_glossary_path,
)
from book_translator.documents.layout import (
    STAGE_DIRNAMES,
    STAGE_DIRS,
    book_root,
    is_stage_dirname,
    resolve_stage_dir,
    stage_of,
)
from book_translator.documents.manifest import (
    load_chunk_glossary_manifest,
    load_chunk_manifest,
    load_split_manifest,
)


def test_the_registry_names_every_stage_folder_once():
    assert STAGE_DIRS["chunks"] == CHUNKS_DIRNAME
    assert STAGE_DIRS["translations"] == TRANSLATIONS_DIRNAME
    assert STAGE_DIRS["chunk_glossary"] == CHUNK_GLOSSARY_DIRNAME
    assert len(set(STAGE_DIRS.values())) == len(STAGE_DIRS)
    assert PREPROCESSED_DIRNAME in STAGE_DIRNAMES


def test_resolve_stage_dir_from_the_book_folder_sibling_file_and_itself(tmp_path: Path):
    book = tmp_path / "book"
    book.mkdir()
    assert resolve_stage_dir(book, "translations") == book / TRANSLATIONS_DIRNAME
    assert resolve_stage_dir(book / CHUNKS_DIRNAME, "translations") == (
        book / TRANSLATIONS_DIRNAME
    )
    assert resolve_stage_dir(book / CHUNKS_DIRNAME / "00000-a-0000.md", "glossary") == (
        book / GLOSSARY_DIRNAME
    )
    assert resolve_stage_dir(book / GLOSSARY_DIRNAME, "glossary") == book / GLOSSARY_DIRNAME


def test_book_root_and_stage_lookup(tmp_path: Path):
    book = tmp_path / "book"
    assert book_root(book / CHUNKS_DIRNAME) == book
    assert book_root(book / "book.md") == book
    assert stage_of(book / CHUNKS_DIRNAME) == "chunks"
    assert stage_of(book) is None
    assert is_stage_dirname(CHUNKS_DIRNAME) is True


def test_an_unknown_stage_is_rejected_with_the_known_names(tmp_path: Path):
    with pytest.raises(ToolError, match="known stages"):
        resolve_stage_dir(tmp_path, "not-a-stage")


def test_chunk_files_falls_back_to_the_folder_order(tmp_path: Path):
    chunks = tmp_path / CHUNKS_DIRNAME
    chunks.mkdir()
    (chunks / "00001-b-0000.md").write_text("b", encoding="utf-8")
    (chunks / "00000-a-0000.md").write_text("a", encoding="utf-8")
    (chunks / "_chunk_manifest.json").write_text("{}", encoding="utf-8")

    assert [entry["filename"] for entry in chunk_files(chunks)] == [
        "00000-a-0000.md",
        "00001-b-0000.md",
    ]
    assert resolve_chunks_dir(chunks / "_chunk_manifest.json") == chunks
    assert resolve_chunks_dir(tmp_path) == chunks


def test_discover_chunk_inputs_reads_the_manifest_and_the_exact_text(tmp_path: Path):
    chunks = tmp_path / CHUNKS_DIRNAME
    chunks.mkdir()
    (chunks / "00000-a-0000.md").write_text("hello", encoding="utf-8")
    manifest = {
        "files": [
            {
                "global_index": 0,
                "chunk_index": 0,
                "filename": "00000-a-0000.md",
                "chapter_filename": "01-a.md",
                "start_char": 0,
                "end_char": 5,
                "sha256": "ignored-and-recomputed",
            }
        ]
    }
    (chunks / "_chunk_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    inputs, folder = discover_chunk_inputs(tmp_path)
    assert folder == chunks
    assert inputs[0].text == "hello"
    assert inputs[0].chapter_id == "01-a"
    assert inputs[0].sha256


def test_chunk_glossary_filename_keeps_the_chunk_name():
    assert chunk_glossary_filename("00001-a-0000.md") == "00001-a-0000.md.json"
    assert chunk_glossary_filename("") == "chunk.md.json"


def test_resolve_glossary_path_from_a_file_folder_or_nothing(tmp_path: Path):
    book = tmp_path / "book"
    chunks = book / CHUNKS_DIRNAME
    glossary = book / GLOSSARY_DIRNAME
    chunks.mkdir(parents=True)
    glossary.mkdir()
    (glossary / "glossary.json").write_text("{}", encoding="utf-8")

    assert resolve_glossary_path(None, chunks) == glossary / "glossary.json"
    assert resolve_glossary_path(glossary, chunks) == glossary / "glossary.json"
    assert resolve_glossary_path(glossary / "glossary.json", chunks) == (
        glossary / "glossary.json"
    )


def test_manifest_readers_are_one_contract(tmp_path: Path):
    assert load_split_manifest(tmp_path) == {}
    assert load_chunk_manifest(tmp_path) == {}
    assert load_chunk_glossary_manifest(tmp_path) == {}

    (tmp_path / "_preprocess_manifest.json").write_text(
        json.dumps({"files": [{"filename": "01-a.md"}]}), encoding="utf-8"
    )
    assert load_split_manifest(tmp_path)["files"][0]["filename"] == "01-a.md"
