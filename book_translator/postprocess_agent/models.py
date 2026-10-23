"""Result types for the postprocess stage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

#: Called once per finished chapter as ``(chapter_record, completed, total)``.
ProgressHook = Callable[[dict[str, Any], int, int], None]


@dataclass
class ChapterPostprocessResult:
    """Outcome of postprocessing one chapter."""

    chapter_id: str
    filename: str
    source_path: str
    output_path: str | None = None
    ok: bool = False
    reused: bool = False
    title: str = ""
    title_action: str = "keep"
    source_lines: int = 0
    output_lines: int = 0
    heading_decisions: int = 0
    demoted_lines: list[int] = field(default_factory=list)
    promoted_lines: list[int] = field(default_factory=list)
    merged_subtitles: list[int] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)
    attempts: int = 0
    error: str | None = None
    log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PostprocessResult:
    """Outcome of a whole stage-9 run."""

    ok: bool
    chapters_path: str
    output_dir: str
    chapter_count: int = 0
    ok_count: int = 0
    reused_count: int = 0
    skipped: bool = False
    dry_run: bool = False
    limited: bool = False
    first_chapters: int | None = None
    book_plan: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    toc: dict[str, Any] = field(default_factory=dict)
    web: dict[str, Any] = field(default_factory=dict)
    totals: dict[str, int] = field(default_factory=dict)
    openings: dict[str, Any] = field(default_factory=dict)
    chapters: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    manifest_path: str | None = None
    trace_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["ChapterPostprocessResult", "PostprocessResult", "ProgressHook"]
