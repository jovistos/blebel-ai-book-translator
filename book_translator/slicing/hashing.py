"""Hashing helpers, used to prove that no byte of a book was altered.

The implementation lives in :mod:`book_translator.core.artifacts`; this module is
the slicing package's historical import path and re-exports the one hash family.
"""

from __future__ import annotations

from ..core.artifacts import sha256_file, sha256_lines

__all__ = ["sha256_file", "sha256_lines"]
