"""Result types for the matter stage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

#: Monotonic clock used for the timings; injectable so tests can be exact.
Clock = Callable[[], float]
#: Called once per finished matter chunk as ``(record, completed, total)``.
ProgressHook = Callable[[dict[str, Any], int, int], None]


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #
@dataclass
class MatterChunkResult:
    """One localized matter chunk: what it cost, where it went, whether it was omitted."""

    filename: str
    section_filename: str = ""
    section_kind: str = ""
    chunk_index: int = 0
    source_chars: int = 0
    translation_chars: int = 0
    source_sha256: str = ""
    omitted: bool = False
    omit_reason: str = ""
    attempts: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    output_path: str | None = None
    prompt_path: str | None = None
    debug_path: str | None = None
    ok: bool = True
    skipped: bool = False
    reused: bool = False
    error: str | None = None
    images: list[dict[str, Any]] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    #: The localized text. Kept out of the manifest (the file is the artifact).
    translation: str = field(default="", repr=False, compare=False)

    def record(self) -> dict[str, Any]:
        rec = {
            "filename": self.filename,
            "section_filename": self.section_filename,
            "section_kind": self.section_kind,
            "chunk_index": self.chunk_index,
            "source_chars": self.source_chars,
            "translation_chars": self.translation_chars,
            "source_sha256": self.source_sha256,
            "omitted": self.omitted,
            "omit_reason": self.omit_reason,
            "attempts": self.attempts,
            "usage": dict(self.usage),
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "output_path": self.output_path,
            "prompt_path": self.prompt_path,
            "debug_path": self.debug_path,
            "ok": self.ok,
            "skipped": self.skipped,
            "reused": self.reused,
            "error": self.error,
        }
        if self.images:
            rec["images"] = list(self.images)
        return rec

    def to_dict(self) -> dict[str, Any]:
        return {**self.record(), "log": list(self.log)}


@dataclass
class MatterResult:
    """Outcome of a whole front/back matter localization run."""

    ok: bool
    chapters_path: str
    chunks_path: str
    output_dir: str
    prompts_dir: str
    matter_path: str
    source_language: str = ""
    target_language: str = ""
    book_title: str = ""
    book_author: str = ""
    source_file: str = ""
    synthetic_sections: list[str] = field(default_factory=list)
    synthetic_chunks: list[str] = field(default_factory=list)
    section_count: int = 0
    chunk_count: int = 0
    ok_count: int = 0
    processed_count: int = 0
    reused_count: int = 0
    skipped_count: int = 0
    skipped: bool = False
    dry_run: bool = False
    save: bool = True
    resumed: bool = False
    limited: bool = False
    book_plan: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)
    totals: dict[str, int] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    sections: list[dict[str, Any]] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    manifest_path: str | None = None
    trace_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if not self.images:
            d.pop("images", None)
        return d


# --------------------------------------------------------------------------- #
# paths and resume
# --------------------------------------------------------------------------- #

__all__ = ["Clock", "MatterChunkResult", "MatterResult", "ProgressHook"]
