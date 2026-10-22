"""Helpers for the golden snapshots that guard Phase 0.

Three kinds of snapshot live in ``tests/snapshots/``:

``artifact_tree.json``
    A path -> SHA-256 map of every artifact the fixture pipeline wrote, plus a
    root hash over the map. Volatile values (absolute work paths, timestamps,
    measured durations) are canonicalised before hashing, so the map describes
    the *shape and content* of the artifacts rather than the machine that ran
    them. Every file is included, ``.prompt.md`` records and every manifest too.

``help.json``
    The exact ``--help`` output of the root parser and every subcommand.

``package_all.json``
    ``book_splitter.__all__``, in order.

Set ``UPDATE_SNAPSHOTS=1`` to rewrite the files instead of comparing. A snapshot
is data, not an assertion: the tests in ``test_snapshots.py`` and
``test_pipeline_fixture.py`` assert against it and fail with a readable diff.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

#: Where the golden files are kept.
SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"

#: ISO-8601-looking strings written by the stage manifests and prompt records.
_ISO_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?"
)

#: The run log prints tool arguments through ``short_args``, which truncates a
#: repr to a fixed width. Absolute work paths change that width, so the visible
#: slice differs between otherwise identical runs. Log arguments are narration,
#: not payload: collapse them to one stable placeholder.
_TOOL_ARGUMENTS_IN_LOG = re.compile(r"\(\s*\w+=.*?\)\s*->")

#: JSON keys whose value is a wall-clock timestamp, not content.
_TIMESTAMP_KEYS = frozenset({"generated_at", "created", "timestamp"})

#: Cross-artifact hashes whose input embeds an absolute work path. The file they
#: describe (``04-glossary/glossary.json``) is hashed in its own right, so the
#: snapshot still notices a changed glossary; only this path-dependent copy of
#: the digest is replaced.
_PATH_DEPENDENT_HASH_KEYS = frozenset({"glossary_sha256"})


def _is_timing_key(key: str) -> bool:
    """True for a measured duration or a rate derived from one."""
    lowered = key.lower()
    return (
        "seconds" in lowered
        or "milliseconds" in lowered
        or lowered.endswith("_per_second")
    )


def _canonical_text(text: str, work_root: Path) -> str:
    """Replace the machine-specific parts of a text artifact."""
    roots = {str(work_root), str(work_root.resolve())}
    for root in sorted(roots, key=len, reverse=True):
        text = text.replace(root, "<WORK>")
    text = _ISO_TIMESTAMP.sub("<TIMESTAMP>", text)
    return _TOOL_ARGUMENTS_IN_LOG.sub("(<ARGS>) ->", text)


def _canonical_value(value: Any, work_root: Path) -> Any:
    """Recursively canonicalise a JSON value."""
    if isinstance(value, dict):
        canonical: dict[str, Any] = {}
        for key, item in value.items():
            if key in _TIMESTAMP_KEYS or key.endswith("_at"):
                canonical[str(key)] = "<TIMESTAMP>"
            elif key in _PATH_DEPENDENT_HASH_KEYS:
                canonical[str(key)] = "<CONTENT_HASH>"
            elif _is_timing_key(str(key)):
                canonical[str(key)] = 0.0
            else:
                canonical[str(key)] = _canonical_value(item, work_root)
        return canonical
    if isinstance(value, list):
        return [_canonical_value(item, work_root) for item in value]
    if isinstance(value, str):
        return _canonical_text(value, work_root)
    return value


def canonical_bytes(path: Path, work_root: Path) -> bytes:
    """The bytes a snapshot hashes for one artifact.

    JSON is parsed and re-serialised with sorted keys, which both removes the
    absolute work path from every nested string and makes key order irrelevant.
    Markdown and other text get path/timestamp replacement only. Everything else
    (the EPUB is the only binary here) is hashed verbatim.
    """
    raw = path.read_bytes()
    if path.suffix == ".json":
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return raw
        canonical = _canonical_value(parsed, work_root)
        return (
            json.dumps(canonical, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw
    return _canonical_text(text, work_root).encode("utf-8")


def artifact_tree(book_output: Path, *, work_root: Path) -> dict[str, Any]:
    """Hash every file under ``book_output`` and summarize the tree.

    The returned mapping is stable across runs: files are sorted by their
    POSIX-style relative path, and the root hash is computed over that order.
    """
    files: dict[str, str] = {}
    for path in sorted(book_output.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(book_output).as_posix()
        files[relative] = hashlib.sha256(canonical_bytes(path, work_root)).hexdigest()
    digest = hashlib.sha256()
    for relative in sorted(files):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(files[relative].encode("ascii"))
        digest.update(b"\n")
    return {"root": digest.hexdigest(), "files": files}


def _serialise(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def assert_snapshot(name: str, actual: Any) -> None:
    """Compare ``actual`` with ``tests/snapshots/<name>`` (or update it)."""
    path = SNAPSHOT_DIR / name
    rendered = _serialise(actual)
    if os.environ.get("UPDATE_SNAPSHOTS", "").strip() not in {"", "0", "false", "False"}:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        return
    if not path.is_file():
        raise AssertionError(
            f"snapshot {path} is missing; run `UPDATE_SNAPSHOTS=1 pytest "
            f"tests/test_snapshots.py tests/test_pipeline_fixture.py` to create it"
        )
    expected = path.read_text(encoding="utf-8")
    if expected == rendered:
        return
    diff = difflib.unified_diff(
        expected.splitlines(),
        rendered.splitlines(),
        fromfile=str(path),
        tofile="actual",
        lineterm="",
        n=2,
    )
    shown = "\n".join(list(diff)[:200])
    raise AssertionError(
        f"snapshot mismatch: {path}\n"
        "The artifact contract changed. If the change is intended, regenerate with "
        "`UPDATE_SNAPSHOTS=1 pytest tests/test_snapshots.py tests/test_pipeline_fixture.py`.\n"
        f"{shown}"
    )


__all__ = [
    "SNAPSHOT_DIR",
    "artifact_tree",
    "assert_snapshot",
    "canonical_bytes",
]
