"""Tests for the per-run manifest/trace store."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from book_translator.core.store import ArtifactStore


@dataclass
class Result:
    manifest_path: str | None = None
    trace_path: str | None = None


def test_the_store_writes_both_artifacts_and_attaches_their_paths(tmp_path: Path):
    store = ArtifactStore(tmp_path, "_manifest.json", "_trace.json")
    store.write({"stage": "demo", "ok": True}, {"stage": "demo", "detail": tmp_path})

    manifest = json.loads((tmp_path / "_manifest.json").read_text(encoding="utf-8"))
    trace = json.loads((tmp_path / "_trace.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "demo"
    assert trace["detail"] == str(tmp_path)

    result = store.attach(Result())
    assert result.manifest_path == str(tmp_path / "_manifest.json")
    assert result.trace_path == str(tmp_path / "_trace.json")


def test_a_disabled_store_writes_nothing_and_reports_nothing(tmp_path: Path):
    store = ArtifactStore(tmp_path, "_manifest.json", "_trace.json", enabled=False)
    store.write({"ok": True}, {"ok": True})

    assert not (tmp_path / "_manifest.json").exists()
    assert not (tmp_path / "_trace.json").exists()
    result = store.attach(Result())
    assert result.manifest_path is None
    assert result.trace_path is None


def test_the_store_can_adopt_artifacts_an_earlier_run_wrote(tmp_path: Path):
    manifest = tmp_path / "_manifest.json"
    trace = tmp_path / "_trace.json"
    manifest.write_text("{}", encoding="utf-8")
    trace.write_text("{}", encoding="utf-8")

    store = ArtifactStore(tmp_path, "_manifest.json", "_trace.json", enabled=False)
    store.adopt(manifest_path=manifest, trace_path=trace)

    result = store.attach(Result())
    assert result.manifest_path == str(manifest)
    assert result.trace_path == str(trace)


def test_a_snapshot_writes_only_the_manifest(tmp_path: Path):
    store = ArtifactStore(tmp_path, "_manifest.json", "_trace.json")
    store.snapshot({"partial": True})

    assert (tmp_path / "_manifest.json").is_file()
    assert not (tmp_path / "_trace.json").exists()
    assert store.manifest_path == str(tmp_path / "_manifest.json")
    assert store.trace_path is None


def test_a_snapshot_never_raises_on_a_write_error(tmp_path: Path, monkeypatch):
    store = ArtifactStore(tmp_path, "_manifest.json", "_trace.json")

    def explode(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("book_translator.core.store.write_manifest", explode)
    assert store.snapshot({"partial": True}) is None
    assert store.manifest_path is None


def test_existing_reports_only_a_file_that_is_there(tmp_path: Path):
    store = ArtifactStore(tmp_path, "_manifest.json", "_trace.json")
    assert store.existing("_manifest.json") is None

    (tmp_path / "_manifest.json").write_text("{}", encoding="utf-8")
    assert store.existing("_manifest.json") == tmp_path / "_manifest.json"
