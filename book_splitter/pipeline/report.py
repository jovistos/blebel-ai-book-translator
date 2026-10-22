"""Records the pipeline keeps about each phase for the closing report."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PhaseRecord:
    """What one phase did, for the closing summary and ``--json``."""

    name: str
    ok: bool = False
    seconds: float = 0.0
    summary: str = ""
    error: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PhaseState:
    """What the run produced, derived once for the closing "what next" hints."""

    cleaned: bool = False
    chunked: bool = False
    chunk_missing: bool = False
    chunk_glossaried: bool = False
    translated: bool = False
    mattered: bool = False
    matter_plan: dict[str, Any] = field(default_factory=dict)
    concatenated: bool = False
    postprocessed: bool = False
    bound: bool = False
    epubbed: bool = False
    glossaried: bool = False
    reduced: bool = False
    tagger_missing: bool = False


__all__ = ["PhaseRecord", "PhaseState"]
