"""Tests for the one source/fallback resolver shared by CLI and pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest
from book_translator.core.artifacts import sha256_text
from book_translator.documents.manifest import MANIFEST_FILENAME
from book_translator.pipeline.paths import StagePaths
from book_translator.pipeline.sources import (
    cleaned_covers_window,
    most_advanced_chapters,
    resolve_chapters_input,
    resolve_chunk_input,
    resolve_chunk_source,
    resolve_openings_input,
    resolve_postprocessed_input,
    resolve_source,
    resolve_text_source,
    resolve_translated_chapters_input,
)


def ns(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def write_chapter(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return sha256_text(text)


def write_raw(root: Path, names: tuple[str, ...] = ("01-a.md", "02-b.md")) -> Path:
    raw = root / "book" / "02-chapters"
    files: list[dict[str, Any]] = []
    for name in names:
        digest = write_chapter(raw / name, f"# {name}\n\nText.\n")
        files.append(
            {
                "filename": name,
                "source_lines": 3,
                "start_line": len(files) * 3 + 1,
                "end_line": len(files) * 3 + 3,
                "source_sha256": digest,
                "sha256": digest,
            }
        )
    (raw / MANIFEST_FILENAME).write_text(
        json.dumps({"ok": True, "dry_run": False, "files": files}), encoding="utf-8"
    )
    return raw


def write_cleaned(root: Path, raw: Path, names: tuple[str, ...] | None = None) -> Path:
    cleaned = root / "book" / "03-preprocessed"
    raw_manifest = json.loads((raw / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    by_name = {entry["filename"]: entry for entry in raw_manifest["files"]}
    files: list[dict[str, Any]] = []
    for name in names or tuple(by_name):
        text = (raw / name).read_text(encoding="utf-8")
        digest = write_chapter(cleaned / name, text)
        entry = dict(by_name[name])
        entry["sha256"] = digest
        files.append(entry)
    cleaned.mkdir(parents=True, exist_ok=True)
    (cleaned / "_preprocess_manifest.json").write_text(
        json.dumps({"ok": True, "dry_run": False, "files": files}), encoding="utf-8"
    )
    return cleaned


@pytest.fixture
def book(tmp_path: Path) -> tuple[StagePaths, Path, Path]:
    raw = write_raw(tmp_path)
    cleaned = write_cleaned(tmp_path, raw)
    paths = StagePaths.from_base(raw)
    return paths, raw, cleaned


def test_cleaned_window_is_used_when_it_covers_every_chapter(book):
    paths, _raw, cleaned = book
    assert cleaned_covers_window(paths, None) == cleaned
    source = resolve_text_source(paths, ns(text_in=None))
    assert source.path == cleaned
    assert source.cleaned is True
    assert source.label == "cleaned by stage 3"


def test_explicit_text_in_always_wins(book):
    paths, raw, _cleaned = book
    source = resolve_text_source(paths, ns(text_in=str(raw)))
    assert source.path == raw
    assert source.label == "explicit --text-in"


def test_no_preprocess_forces_raw_text(book):
    paths, raw, _cleaned = book
    source = resolve_text_source(paths, ns(text_in=None), preprocess_enabled=False)
    assert source.path == raw
    assert source.label == "raw chapters (--no-preprocess)"


def test_a_partial_cleaned_window_falls_back_to_raw(book):
    paths, raw, _cleaned = book
    assert cleaned_covers_window(paths, 1) is not None
    # The cleaned folder has both chapters here, so remove one to make it partial
    # relative to the full raw window.
    (paths.preprocessed_dir / "02-b.md").unlink()
    assert cleaned_covers_window(paths, None) is None
    source = resolve_text_source(paths, ns(text_in=None))
    assert source.path == raw
    assert source.label == "cleaned text does not cover this window"


def test_a_stale_cleaned_file_falls_back_to_raw(book):
    paths, raw, cleaned = book
    (cleaned / "01-a.md").write_text("changed\n", encoding="utf-8")
    assert cleaned_covers_window(paths, None) is None
    assert resolve_text_source(paths, ns(text_in=None)).path == raw


def test_a_changed_source_line_count_falls_back_to_raw(book):
    paths, raw, _cleaned = book
    (raw / "01-a.md").write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
    assert cleaned_covers_window(paths, None) is None


def test_a_missing_cleaned_manifest_falls_back_to_raw(book):
    paths, raw, _cleaned = book
    (paths.preprocessed_dir / "_preprocess_manifest.json").unlink()
    assert cleaned_covers_window(paths, None) is None
    assert resolve_text_source(paths, ns(text_in=None)).path == raw


def test_chunk_in_overrides_the_text_folder(book, tmp_path):
    paths, _raw, _cleaned = book
    text = resolve_text_source(paths, ns(text_in=None))
    custom = tmp_path / "custom-chunks"
    assert resolve_chunk_source(paths, ns(chunk_in=None), text) == text.path
    assert resolve_chunk_source(paths, ns(chunk_in=str(custom)), text) == custom


def test_command_inputs_prefer_explicit_flags(book, tmp_path):
    paths, raw, _cleaned = book
    assert resolve_chapters_input(ns(out=str(raw)), paths) == raw
    custom_chunks = tmp_path / "chunks"
    assert resolve_chunk_input(ns(chunks_in=str(custom_chunks)), paths) == custom_chunks
    custom_translated = tmp_path / "book" / "09-translated-chapters"
    assert (
        resolve_translated_chapters_input(
            ns(translated_chapters_in=str(custom_translated)), paths
        )
        == custom_translated
    )
    custom_post = tmp_path / "book" / "10-postprocessed-chapters"
    assert (
        resolve_postprocessed_input(ns(postprocessed_in=str(custom_post)), paths)
        == custom_post
    )


def test_openings_uses_the_most_advanced_folder(book):
    paths, _raw, cleaned = book
    post = paths.postprocess_dir
    post.mkdir(parents=True, exist_ok=True)
    (post / "01-a.md").write_text("# A\n", encoding="utf-8")
    assert most_advanced_chapters(paths) == post
    assert resolve_openings_input(ns(chapters_in=None), paths) == post
    # An explicit book folder descends to the most advanced stage under it.
    assert resolve_openings_input(ns(chapters_in=str(paths.book_dir)), paths) == post
    # An explicit stage folder is taken as given.
    assert resolve_openings_input(ns(chapters_in=str(cleaned)), paths) == cleaned


def test_resolve_source_markdown_is_read_directly(tmp_path):
    source = tmp_path / "book.md"
    source.write_text("# Book\n", encoding="utf-8")
    plan = resolve_source(source, output_root=tmp_path / "out")
    assert plan.source == source
    assert plan.needs_conversion is False
    assert plan.scratch_dir is None


def test_resolve_source_txt_needs_conversion(tmp_path):
    source = tmp_path / "book.txt"
    source.write_text("Book\n", encoding="utf-8")
    plan = resolve_source(source, output_root=tmp_path / "out")
    assert plan.needs_conversion is True
    assert plan.source == tmp_path / "out" / "book" / "01-markdown" / "book.md"
    assert plan.scratch_dir is None


def test_resolve_source_dry_run_txt_uses_a_scratch_dir(tmp_path):
    source = tmp_path / "book.txt"
    source.write_text("Book\n", encoding="utf-8")
    scratch = tmp_path / "scratch"
    plan = resolve_source(source, output_root=tmp_path / "out", dry_run=True, scratch_dir=scratch)
    assert plan.needs_conversion is True
    assert plan.source == scratch / "book.md"
    assert plan.scratch_dir == scratch
    assert not (tmp_path / "out").exists()
