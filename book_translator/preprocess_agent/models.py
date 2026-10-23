"""Result types for the preprocess stage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ChapterResult:
    """Outcome of preprocessing one chapter."""

    chapter_id: str
    filename: str
    source_path: str
    output_path: str | None = None
    ok: bool = False
    source_lines: int = 0
    output_lines: int = 0
    operations: list[dict[str, Any]] = field(default_factory=list)
    text_fixes: list[dict[str, Any]] = field(default_factory=list)
    removed_artefacts: list[int] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)
    attempts: int = 0
    error: str | None = None
    log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: Called once per finished chapter as ``(chapter, completed, total)``, where
#: ``chapter`` is the same record (:meth:`ChapterResult.to_dict`) the manifest
#: stores. It runs in completion order, so with ``workers > 1`` it is not the
#: chapter order.
ProgressHook = Callable[[dict[str, Any], int, int], None]


@dataclass
class PreprocessResult:
    """Outcome of a whole stage-2 run."""

    ok: bool
    chapters_path: str
    output_dir: str
    chapter_count: int = 0
    ok_count: int = 0
    skipped_count: int = 0
    skipped: bool = False
    dry_run: bool = False
    parallel: bool = False
    corpus_scan: dict[str, Any] = field(default_factory=dict)
    openings: dict[str, Any] = field(default_factory=dict)
    chapters: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    manifest_path: str | None = None
    trace_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["ChapterResult", "PreprocessResult", "ProgressHook"]
