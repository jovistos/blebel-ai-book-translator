"""Result container returned by :func:`book_splitter.agent.runner.run_split`.

A plain dataclass so callers (CLI, tests, notebooks) get a typed object instead
of a loose dictionary, while still being able to serialise it with ``to_dict``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SplitterResult:
    """Outcome of one split run: a plan, a manifest, a verification report."""

    ok: bool
    source_path: str
    output_dir: str
    strategy: str | None = None
    chapter_count: int = 0
    files: list[dict[str, Any]] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    verification: dict[str, Any] = field(default_factory=dict)
    header_assessment: dict[str, Any] = field(default_factory=dict)
    attempts: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    log: list[str] = field(default_factory=list)
    trace_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
