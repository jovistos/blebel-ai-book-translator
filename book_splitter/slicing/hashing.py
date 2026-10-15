"""Hashing helpers, used to prove that no byte of a book was altered."""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK = 1 << 20


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file on disk, streamed so large books stay cheap."""
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(lines: list[str], start: int = 1) -> str:
    """SHA-256 of the UTF-8 bytes of lines ``start..`` (1-indexed, inclusive)."""
    digest = hashlib.sha256()
    for line in lines[start - 1 :]:
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()
