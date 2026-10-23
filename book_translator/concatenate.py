"""Alias for Stage 8: concatenate translated chunks back into chapters.

Delegates to :mod:`book_translator.assembly`.
"""

from __future__ import annotations

from .assembly import (
    AssemblyResult,
    ConcatenateResult,
    run_assembly,
    run_concatenate,
)

__all__ = [
    "AssemblyResult",
    "ConcatenateResult",
    "run_assembly",
    "run_concatenate",
]
