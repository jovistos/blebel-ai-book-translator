"""Tests for the one artifact writer: atomic files, manifests, hashes, clock."""

from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path

from book_splitter.core.artifacts import (
    Sha256,
    now,
    read_json,
    read_manifest,
    sha256_bytes,
    sha256_file,
    sha256_lines,
    sha256_text,
    timestamp,
    write_json,
    write_manifest,
    write_text,
    write_trace,
)


def test_write_text_replaces_the_file_and_leaves_no_temporary(tmp_path: Path):
    target = tmp_path / "nested" / "book.md"
    assert write_text(target, "first") == target
    assert target.read_text(encoding="utf-8") == "first"
    write_text(target, "second")
    assert target.read_text(encoding="utf-8") == "second"
    assert [p.name for p in target.parent.iterdir()] == ["book.md"]


def test_write_json_is_utf8_and_read_json_round_trips(tmp_path: Path):
    target = tmp_path / "glossary.json"
    write_json(target, {"entries": [{"term": "Džordž"}]})

    assert "Džordž" in target.read_text(encoding="utf-8")
    assert read_json(target) == {"entries": [{"term": "Džordž"}]}


def test_read_json_returns_the_default_for_missing_or_malformed_files(tmp_path: Path):
    assert read_json(tmp_path / "missing.json", default={}) == {}
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert read_json(broken, default={"fallback": True}) == {"fallback": True}


def test_write_manifest_and_read_manifest_agree(tmp_path: Path):
    path = write_manifest(tmp_path, "_chunk_manifest.json", {"ok": True})
    assert path == tmp_path / "_chunk_manifest.json"
    assert read_manifest(tmp_path, "_chunk_manifest.json") == {"ok": True}
    assert read_manifest(path) == {"ok": True}


def test_the_manifest_helpers_also_accept_the_file_itself(tmp_path: Path):
    path = write_manifest(tmp_path / "_named.json", "_ignored.json", {"ok": True})
    assert path == tmp_path / "_named.json"
    assert read_manifest(path) == {"ok": True}


def test_read_manifest_defaults_to_the_stage_one_name_for_a_folder(tmp_path: Path):
    (tmp_path / "_manifest.json").write_text('{"stage": 1}', encoding="utf-8")
    assert read_manifest(tmp_path) == {"stage": 1}


def test_read_manifest_is_empty_for_a_missing_or_non_object_manifest(tmp_path: Path):
    assert read_manifest(tmp_path, "_missing.json") == {}
    array = tmp_path / "_array.json"
    array.write_text("[1, 2]", encoding="utf-8")
    assert read_manifest(array) == {}


def test_write_trace_stringifies_values_json_cannot_encode(tmp_path: Path):
    path = write_trace(tmp_path, "_trace.json", {"when": now(), "path": Path("x")})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["path"] == "x"
    assert payload["when"].startswith("20")


def test_the_timestamp_is_utc_to_the_second():
    moment = now()
    assert moment.tzinfo is UTC
    assert moment.microsecond == 0
    assert timestamp().endswith("+00:00")


def test_the_sha256_family_is_the_same_function_everywhere(tmp_path: Path):
    target = tmp_path / "book.md"
    target.write_text("hello\n", encoding="utf-8")

    assert sha256_text("hello\n") == sha256_bytes(b"hello\n")
    assert sha256_file(target) == sha256_text("hello\n")
    assert sha256_lines(["one\n", "two\n", "three\n"], start=2) == sha256_text("two\nthree\n")


def test_the_incremental_hash_matches_the_one_shot_hash():
    digest = Sha256()
    digest.update("hel").update(b"lo\n")
    assert digest.hexdigest() == sha256_text("hello\n")
