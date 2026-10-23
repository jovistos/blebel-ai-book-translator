"""Result and record types for the chunking stage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

#: The preference order, spelled out in the manifest.
BOUNDARY_ORDER: tuple[str, ...] = ("paragraph", "sentence", "line", "word", "hard")
#: Monotonic clock used for the timings; injectable so tests can be exact.
Clock = Callable[[], float]
#: Called once per visited chapter as ``(chapter_record, completed, total)``.
ProgressHook = Callable[[dict[str, Any], int, int], None]


@dataclass
class ChunkRecord:
    """One chunk file: identity, provenance and the exact span it covers."""

    global_index: int
    chapter_index: int
    chunk_index: int
    chapter_id: str
    chapter_filename: str
    filename: str
    path: str = ""
    start_char: int = 0
    end_char: int = 0
    start_line: int = 0
    end_line: int = 0
    token_count: int = 0
    bytes: int = 0
    sha256: str = ""
    boundary_before: str = ""
    boundary_after: str = ""
    first_line: str = ""
    last_line: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChapterChunks:
    """One chapter's chunking outcome, including whether it was reused."""

    chapter_id: str
    filename: str
    source_path: str
    chapter_index: int
    title: str = ""
    start_line: int | None = None
    end_line: int | None = None
    chunk_count: int = 0
    char_count: int = 0
    token_count: int = 0
    chunk_tokens: int = 0
    source_sha256: str = ""
    reconstructed_sha256: str = ""
    reconstructs_byte_for_byte: bool = False
    first_global_index: int | None = None
    last_global_index: int | None = None
    skipped: bool = False
    ok: bool = True
    error: str | None = None
    boundary_counts: dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    log: list[str] = field(default_factory=list)

    def record(self) -> dict[str, Any]:
        """The chapter entry stored in the manifest."""
        return {
            "chapter_id": self.chapter_id,
            "filename": self.filename,
            "source_path": self.source_path,
            "chapter_index": self.chapter_index,
            "title": self.title,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "chunk_count": self.chunk_count,
            "char_count": self.char_count,
            "token_count": self.token_count,
            "chunk_tokens": self.chunk_tokens,
            "source_sha256": self.source_sha256,
            "reconstructed_sha256": self.reconstructed_sha256,
            "reconstructs_byte_for_byte": self.reconstructs_byte_for_byte,
            "first_global_index": self.first_global_index,
            "last_global_index": self.last_global_index,
            "skipped": self.skipped,
            "ok": self.ok,
            "error": self.error,
            "boundary_counts": dict(self.boundary_counts),
            "elapsed_seconds": round(self.elapsed_seconds, 4),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.record(), "log": list(self.log)}


@dataclass
class ChunkingResult:
    """Outcome of a whole stage-5 run."""

    ok: bool
    chapters_path: str
    output_dir: str
    source_file: str = ""
    chapter_count: int = 0
    ok_count: int = 0
    processed_count: int = 0
    skipped_count: int = 0
    chunk_count: int = 0
    processed_chunk_count: int = 0
    skipped: bool = False
    dry_run: bool = False
    limited: bool = False
    settings: dict[str, Any] = field(default_factory=dict)
    totals: dict[str, int] = field(default_factory=dict)
    verification: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    chapters: list[dict[str, Any]] = field(default_factory=list)
    files: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    manifest_path: str | None = None
    trace_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "BOUNDARY_ORDER",
    "ChapterChunks",
    "ChunkRecord",
    "ChunkingResult",
    "Clock",
    "ProgressHook",
]
