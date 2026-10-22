"""Phase 0: the deterministic fixture-book contract for stages 0 through 11.

This is the safety net the structural work depends on. It runs the complete
pipeline twice on a checked-in TXT fixture, using scripted model doubles at every
model boundary, and proves:

* every stage from 0 (convert) through 11 (EPUB) reports success;
* the two runs produce the same canonical artifact tree even though they write
  to different absolute folders;
* the tree's path -> SHA-256 map matches ``tests/snapshots/artifact_tree.json``.

The tree covers **every** file the pipeline writes, including every manifest,
every ``_agent_trace.json`` and every debugging ``.prompt.md``. Volatile values
(work paths, wall-clock timestamps, measured durations) are canonicalised by
:mod:`tests._snapshots`; everything else is hashed byte for byte.

When an intended change alters the artifacts, regenerate the snapshot with::

    UPDATE_SNAPSHOTS=1 pytest tests/test_snapshots.py tests/test_pipeline_fixture.py
"""

from __future__ import annotations

from pathlib import Path

from _pipeline_harness import (
    MANIFEST_FILES,
    STAGE_NAMES,
    run_fixture_pipeline,
)
from _snapshots import artifact_tree, assert_snapshot


def _pipeline(work: Path, monkeypatch):
    book, results = run_fixture_pipeline(work, monkeypatch)
    return book, results, artifact_tree(book, work_root=work)


def test_fixture_book_runs_every_stage_and_hashes_the_whole_tree(
    tmp_path: Path, monkeypatch
) -> None:
    first_book, first_results, first_tree = _pipeline(tmp_path / "first", monkeypatch)
    _second_book, second_results, second_tree = _pipeline(
        tmp_path / "second", monkeypatch
    )

    # 1. Every stage ran and reported success. A failure names the stage.
    assert set(first_results) == set(STAGE_NAMES)
    assert set(second_results) == set(STAGE_NAMES)
    for stage in STAGE_NAMES:
        result = first_results[stage]
        assert bool(getattr(result, "ok", False)), (
            f"{stage} did not succeed: "
            f"{getattr(result, 'errors', None) or getattr(result, 'issues', None)}"
        )

    # 2. The artifact tree is a complete pipeline record, not just the chapters.
    files = first_tree["files"]
    assert files, "the fixture pipeline wrote no artifacts"
    for expected in MANIFEST_FILES:
        assert expected in files, f"manifest missing from the artifact tree: {expected}"
    assert any(name.endswith(".prompt.md") for name in files), (
        "no .prompt.md debugging artifact was written"
    )
    assert "01-chapters/_manifest.json" in files
    assert "11-translated-book.epub" in files

    # 3. Same inputs, different work folders -> identical canonical bytes.
    assert first_tree == second_tree, (
        "the fixture pipeline is not deterministic: two runs over the same fixture "
        "produced different artifact hashes"
    )

    # 4. The committed golden tree describes every file by path and hash.
    assert_snapshot("artifact_tree.json", first_tree)

    # Keep the local variable meaningful for the failure message above.
    assert first_book.is_dir()
