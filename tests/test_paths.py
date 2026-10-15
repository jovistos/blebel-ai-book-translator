"""Tests for the document layout: input folder in, output folder out."""

from __future__ import annotations

import pytest

from book_splitter.paths import (
    book_chapters_dir,
    book_output_dir,
    book_slug,
    default_input_dir,
    default_output_dir,
    resolve_book_path,
    resolve_output_dir,
    repo_root,
)

from _fixtures import ROOT

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #


def test_default_folders_are_input_and_output_under_the_repo_root():
    root = repo_root()
    assert root == ROOT
    assert default_input_dir(root) == ROOT / "data" / "input"
    assert default_output_dir(root) == ROOT / "data" / "output"


def test_per_book_output_layout():
    book = ROOT / "data" / "input" / "My Book.md"
    assert book_slug(book) == "My-Book"
    assert book_output_dir(book, root=ROOT) == ROOT / "data" / "output" / "My-Book"
    assert (
        book_chapters_dir(book, root=ROOT)
        == ROOT / "data" / "output" / "My-Book" / "chapters"
    )


def test_output_root_can_be_overridden():
    book = "book.md"
    assert book_chapters_dir(book, output_root="/tmp/out") == __import__(
        "pathlib"
    ).Path("/tmp/out/book/chapters")


# --------------------------------------------------------------------------- #
# Source resolution
# --------------------------------------------------------------------------- #


def test_resolve_book_accepts_an_explicit_path(tmp_path):
    explicit = tmp_path / "somewhere.md"
    explicit.write_text("hello\n", encoding="utf-8")
    assert resolve_book_path(explicit) == explicit.resolve()


def test_resolve_book_finds_a_bare_filename_in_the_input_folder(tmp_path):
    in_dir = tmp_path / "data" / "input"
    in_dir.mkdir(parents=True)
    book = in_dir / "novel.md"
    book.write_text("hello\n", encoding="utf-8")

    assert resolve_book_path("novel.md", input_dir=in_dir) == book.resolve()
    # A bare stem works too.
    assert resolve_book_path("novel", input_dir=in_dir) == book.resolve()


def test_resolve_book_defaults_to_the_only_document(tmp_path):
    in_dir = tmp_path / "data" / "input"
    in_dir.mkdir(parents=True)
    book = in_dir / "only.md"
    book.write_text("hello\n", encoding="utf-8")
    assert resolve_book_path(None, input_dir=in_dir) == book.resolve()


def test_resolve_book_refuses_an_ambiguous_input_folder(tmp_path):
    in_dir = tmp_path / "data" / "input"
    in_dir.mkdir(parents=True)
    for name in ("a.md", "b.md"):
        (in_dir / name).write_text("x\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="several documents"):
        resolve_book_path(None, input_dir=in_dir)


def test_resolve_book_reports_a_missing_document(tmp_path):
    in_dir = tmp_path / "data" / "input"
    in_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="book not found"):
        resolve_book_path("nope.md", input_dir=in_dir)


def test_resolve_output_dir_defaults_under_the_book_stem():
    book = ROOT / "data" / "input" / "novel.md"
    assert resolve_output_dir(book, None, root=ROOT) == (
        ROOT / "data" / "output" / "novel" / "chapters"
    )
    assert resolve_output_dir(book, "/tmp/custom", root=ROOT) == __import__("pathlib").Path(
        "/tmp/custom"
    )
