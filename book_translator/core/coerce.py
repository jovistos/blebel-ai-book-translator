"""Tiny scalar coercions shared by the stage modules.

These used to be copied privately in several packages with slightly different
names (``_int``/``_as_int``/``_path``). They are deliberately dependency-free so
any layer can import them without pulling a stage runtime in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def as_int(value: Any, default: int = 0) -> int:
    """``int(value)`` when possible, else ``default``."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_path(value: Any) -> Path | None:
    """``None``/empty stays ``None``; everything else is an expanded path."""
    if value is None:
        return None
    text = str(value).strip()
    return Path(text).expanduser() if text else None


__all__ = ["as_int", "as_path"]
