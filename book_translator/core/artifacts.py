"""Atomic artifact IO, hashing and the one timestamp convention.

Every stage writes JSON manifests and traces, and several of them also needed a
SHA-256 of a file or of a string. Those helpers had drifted into a dozen
modules — each with its own ``json.dumps(..., indent=2)`` and its own reading of
``datetime.now`` — so this module is the single place they come from now:

* :func:`write_json` / :func:`write_text` write through a temporary file in the
  destination directory and ``os.replace`` it into place. A killed run therefore
  leaves either the old artifact or the new one, never half a manifest.
* :func:`read_manifest` / :func:`write_manifest` / :func:`write_trace` are the
  named-artifact conveniences: a manifest is a JSON object, a trace is the same
  run with non-JSON values stringified.
* :func:`now` / :func:`timestamp` are the one clock convention: UTC, whole
  seconds, ISO-8601. A manifest written on any machine is comparable.
* :func:`sha256_text` / :func:`sha256_bytes` / :func:`sha256_file` are the one
  hash family, so a digest recorded by stage 4 can be checked by stage 7.

Nothing here decides *which* files a stage writes; that is
:mod:`book_translator.documents.layout` and the stage's own manifest writer.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: A written JSON artifact is indented so a human can diff it.
JSON_INDENT = 2
#: The stage-1 manifest name, used when ``read_manifest`` is handed a folder and
#: no name. Kept here rather than imported from constants so this module stays
#: the bottom of the artifact stack.
DEFAULT_MANIFEST_FILENAME = "_manifest.json"
#: Bytes read per iteration when hashing a file.
_HASH_CHUNK = 65_536


# --------------------------------------------------------------------------- #
# the clock
# --------------------------------------------------------------------------- #
def now() -> datetime:
    """The current UTC time, to the second — the pipeline's only clock."""
    return datetime.now(UTC).replace(microsecond=0)


def timestamp() -> str:
    """The current UTC time as ISO-8601, the form every manifest records."""
    return now().isoformat()


# --------------------------------------------------------------------------- #
# hashing
# --------------------------------------------------------------------------- #
class Sha256:
    """An incremental SHA-256, for hashing text while it is being written.

    The one-off helpers below cover a finished value; this covers the streaming
    case the slicer needs, so no module outside this one imports ``hashlib``.
    """

    def __init__(self) -> None:
        self._digest = hashlib.sha256()

    def update(self, data: str | bytes) -> Sha256:
        self._digest.update(data.encode("utf-8") if isinstance(data, str) else data)
        return self

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """SHA-256 of raw bytes, lowercase hexadecimal."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """SHA-256 of a string's UTF-8 bytes."""
    return sha256_bytes(str(text).encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(lines: list[str], start: int = 1) -> str:
    """SHA-256 of the UTF-8 bytes of lines ``start..`` (1-indexed, inclusive)."""
    digest = Sha256()
    for line in lines[start - 1 :]:
        digest.update(line)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# atomic writes
# --------------------------------------------------------------------------- #
def write_text(path: str | Path, text: str) -> Path:
    """Write ``text`` to ``path`` atomically, creating the parent folder.

    The content goes to a temporary sibling first and is renamed into place, so
    a reader never sees a partially written file and an interrupted run does not
    destroy the previous artifact.
    """
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(str(text))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise
    return target


def write_json(
    path: str | Path,
    payload: Any,
    *,
    indent: int = JSON_INDENT,
    default: Any = None,
    sort_keys: bool = False,
    trailing_newline: bool = False,
) -> Path:
    """Serialise ``payload`` as UTF-8 JSON and write it atomically."""
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=indent,
        default=default,
        sort_keys=sort_keys,
    )
    if trailing_newline:
        text += "\n"
    return write_text(path, text)


def _artifact_path(directory: str | Path, filename: str) -> Path:
    """Resolve ``directory/filename``, accepting the file itself as ``directory``."""
    target = Path(directory).expanduser()
    if target.suffix.lower() == ".json":
        return target
    return target / filename


def write_manifest(
    directory: str | Path,
    filename: str,
    payload: dict[str, Any],
) -> Path:
    """Write a stage manifest (a pretty-printed JSON object) into ``directory``.

    ``directory`` may be the folder that will hold the manifest or the manifest
    file itself; both forms are accepted so a caller never has to join paths by
    hand.
    """
    return write_json(_artifact_path(directory, filename), payload)


def write_trace(
    directory: str | Path,
    filename: str,
    payload: Any,
) -> Path:
    """Write a stage trace.

    A trace is the run's full record, so it may carry datetimes, paths and
    dataclass-ish values the manifest writer deliberately avoids; ``default=str``
    keeps it writable without teaching every stage its own serialiser.
    """
    return write_json(
        _artifact_path(directory, filename),
        payload,
        default=str,
    )


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
def read_json(path: str | Path, *, default: Any = None) -> Any:
    """Read UTF-8 JSON, returning ``default`` for a missing or malformed file."""
    target = Path(path).expanduser()
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def read_manifest(
    directory_or_file: str | Path,
    filename: str | None = None,
    *,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read a stage manifest, or return an empty object.

    ``directory_or_file`` may be the folder that holds the manifest (``filename``
    names it) or the manifest file itself. A missing, unreadable or non-object
    file is not fatal: every caller treats an absent manifest as "nothing was
    recorded yet", which is also what a resume needs.
    """
    empty: dict[str, Any] = dict(default or {})
    path = Path(directory_or_file).expanduser()
    if filename:
        manifest_path = path / filename if path.is_dir() else path
    elif path.is_dir():
        manifest_path = path / DEFAULT_MANIFEST_FILENAME
    else:
        manifest_path = path
    payload = read_json(manifest_path)
    return payload if isinstance(payload, dict) else empty


__all__ = [
    "DEFAULT_MANIFEST_FILENAME",
    "JSON_INDENT",
    "Sha256",
    "now",
    "read_json",
    "read_manifest",
    "sha256_bytes",
    "sha256_file",
    "sha256_lines",
    "sha256_text",
    "timestamp",
    "write_json",
    "write_manifest",
    "write_text",
    "write_trace",
]
