"""Result and record types for the assembly stage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

#: Monotonic clock used for the timings; injectable so tests can be exact.
Clock = Callable[[], float]
#: Called once per visited chapter as ``(chapter_record, completed, total)``.
ProgressHook = Callable[[dict[str, Any], int, int], None]


@dataclass
class TranslatedChapter:
    """One translated chapter file: what went in, what came out and its state."""

    chapter_id: str
    filename: str
    chapter_index: int
    chunk_count: int = 0
    translated_chunks: int = 0
    source_chars: int = 0
    translation_chars: int = 0
    sha256: str = ""
    output_path: str | None = None
    reused: bool = False
    skipped: bool = False
    ok: bool = True
    error: str | None = None
    note: str | None = None
    missing_chunks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AssemblyResult:
    """Outcome of a whole stage-8 run."""

    ok: bool
    chunks_path: str
    translations_path: str
    output_dir: str
    chapter_count: int = 0
    ok_count: int = 0
    assembled_count: int = 0
    reused_count: int = 0
    incomplete_count: int = 0
    skipped_count: int = 0
    dry_run: bool = False
    limited: bool = False
    first_chapters: int | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    totals: dict[str, int] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    chapters: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    manifest_path: str | None = None
    trace_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AssemblyInputs:
    """Everything the run needs to know before it visits the first chapter."""

    chunk_folder: Path
    translations_folder: Path
    out_dir: Path
    names: list[str]
    grouped: dict[str, list[dict[str, Any]]]
    limit: int | None
    limited: bool


__all__ = [
    "AssemblyInputs",
    "AssemblyResult",
    "Clock",
    "ProgressHook",
    "TranslatedChapter",
]
