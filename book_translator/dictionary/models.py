"""Result and record types for the dictionary stage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from .unique import UniqueSelection

#: Monotonic clock used for the timings; injectable so tests can be exact.
Clock = Callable[[], float]
#: Called once per reduced chapter as ``(chapter_record, completed, total)``.
ProgressHook = Callable[[dict[str, Any], int, int], None]


@dataclass
class TextReduction:
    """What the funnel did to one chapter's text."""

    kept: list[str] = field(default_factory=list)
    #: The nouns of each kept sentence, aligned with ``kept``. This is what the
    #: first-mention step consumes, so it is recorded here rather than re-tagged.
    kept_keys: list[list[str]] = field(default_factory=list)
    source_sentences: int = 0
    dropped_short: int = 0
    dropped_no_uppercase: int = 0
    dropped_no_entity: int = 0
    dropped_temporal: int = 0
    entities: dict[str, int] = field(default_factory=dict)
    entity_mentions: list[dict[str, Any]] = field(default_factory=list)
    examples: dict[str, list[str]] = field(default_factory=dict)
    #: Sentences handed to the tagger, and the seconds it took to tag them.
    ner_calls: int = 0
    ner_seconds: float = 0.0

    @property
    def dropped_total(self) -> int:
        return (
            self.dropped_short
            + self.dropped_no_uppercase
            + self.dropped_no_entity
            + self.dropped_temporal
        )

    @property
    def noun_keys(self) -> list[str]:
        """Every distinct noun this chapter's kept sentences mention, sorted."""
        return sorted({key for keys in self.kept_keys for key in keys})

    @property
    def milliseconds_per_sentence(self) -> float:
        return (1000.0 * self.ner_seconds / self.ner_calls) if self.ner_calls else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "kept": len(self.kept),
            "dropped_total": self.dropped_total,
            "noun_keys": self.noun_keys,
            "milliseconds_per_sentence": round(self.milliseconds_per_sentence, 3),
        }


@dataclass
class ChapterReduction:
    """One chapter's reduced file, its counts and (optionally) its trace."""

    chapter_id: str
    filename: str
    source_path: str
    output_path: str | None = None
    skipped: bool = False
    ok: bool = True
    error: str | None = None
    title: str = ""
    start_line: int | None = None
    end_line: int | None = None
    reduction: TextReduction | None = None
    #: The first-mention step's own outcome, when it ran for this chapter.
    unique: UniqueSelection | None = None
    output_lines: int = 0
    output_bytes: int = 0
    sha256: str = ""
    unique_path: str | None = None
    unique_lines: int = 0
    unique_bytes: int = 0
    unique_sha256: str = ""
    #: Every noun the chapter's kept sentences mention. Recorded so a rerun with
    #: ``skip_if_present`` can rebuild the book-wide set without re-tagging.
    noun_keys: list[str] = field(default_factory=list)
    #: Wall-clock seconds spent on this chapter, and how much of it was the tagger.
    elapsed_seconds: float = 0.0
    ner_seconds: float = 0.0
    ner_calls: int = 0
    log: list[str] = field(default_factory=list)

    def record(self) -> dict[str, Any]:
        """The flat per-chapter record stored in the manifest."""
        reduction = self.reduction
        unique = self.unique
        return {
            "filename": self.filename,
            "chapter_id": self.chapter_id,
            "title": self.title,
            "path": self.output_path or self.source_path,
            "source_path": self.source_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "skipped": self.skipped,
            "ok": self.ok,
            "error": self.error,
            "sentences": reduction.source_sentences if reduction else 0,
            "kept": len(reduction.kept) if reduction else 0,
            "dropped_short": reduction.dropped_short if reduction else 0,
            "dropped_no_uppercase": reduction.dropped_no_uppercase if reduction else 0,
            "dropped_no_entity": reduction.dropped_no_entity if reduction else 0,
            "dropped_temporal": reduction.dropped_temporal if reduction else 0,
            "entities": dict(reduction.entities) if reduction else {},
            "output_lines": self.output_lines,
            "output_bytes": self.output_bytes,
            "sha256": self.sha256,
            "unique_path": self.unique_path,
            "unique_lines": self.unique_lines,
            "unique_bytes": self.unique_bytes,
            "unique_sha256": self.unique_sha256,
            "unique_kept": len(unique.kept) if unique else 0,
            "dropped_duplicate_noun": unique.dropped_duplicate if unique else 0,
            "new_nouns": unique.new_nouns if unique else 0,
            # The rerun index: every noun the chapter mentions, not just the new
            # ones, because a later chapter is tested against all of them.
            "noun_keys": list(self.noun_keys),
            "ner_calls": self.ner_calls,
            "ner_seconds": round(self.ner_seconds, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "milliseconds_per_sentence": round(
                (1000.0 * self.ner_seconds / self.ner_calls) if self.ner_calls else 0.0,
                3,
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.record(),
            "examples": dict(self.reduction.examples) if self.reduction else {},
            "entity_mentions": (
                list(self.reduction.entity_mentions) if self.reduction else []
            ),
            "unique": self.unique.to_dict() if self.unique else None,
            "log": list(self.log),
        }


@dataclass
class DictionaryResult:
    """Outcome of a whole stage-3 run."""

    ok: bool
    chapters_path: str
    output_dir: str
    #: Where the first-mention chapters went, and whether they were written.
    unique_dir: str = ""
    language: dict[str, Any] = field(default_factory=dict)
    backend: str = ""
    chapter_count: int = 0
    ok_count: int = 0
    processed_count: int = 0
    skipped_count: int = 0
    skipped: bool = False
    dry_run: bool = False
    #: True when ``limit`` kept this run to a prefix of the chapters.
    limited: bool = False
    save_reduced: bool = True
    save_unique: bool = True
    gpu: int | None = None
    #: What the tagger actually ran on: ``cuda``, ``cpu``, or ``""`` if unknown.
    device: str = ""
    settings: dict[str, Any] = field(default_factory=dict)
    totals: dict[str, int] = field(default_factory=dict)
    #: The first-mention step's own totals: nouns, kept and dropped sentences.
    unique_totals: dict[str, int] = field(default_factory=dict)
    entities: dict[str, int] = field(default_factory=dict)
    #: Seconds: model load, the tagger itself, the chapter loop and the whole run,
    #: plus the derived per-sentence rates.
    timing: dict[str, float] = field(default_factory=dict)
    chapters: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    manifest_path: str | None = None
    trace_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "ChapterReduction",
    "Clock",
    "DictionaryResult",
    "ProgressHook",
    "TextReduction",
]
