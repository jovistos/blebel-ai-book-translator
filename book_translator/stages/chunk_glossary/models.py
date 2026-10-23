"""Result and record types for the chunk-glossary stage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from book_translator.core.constants import DEFAULT_CHUNK_GLOSSARY_THRESHOLD
from book_translator.documents.chunks import ChunkInput

from .matching import MatchCandidate

#: Monotonic clock used for the timings; injectable so tests can be exact.
Clock = Callable[[], float]
#: Called once per chunk as ``(chunk_record, completed, total)``.
ProgressHook = Callable[[dict[str, Any], int, int], None]


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #
@dataclass
class EntityTerm:
    """One distinct NER term inside one chunk, with its types and mention count."""

    term: str
    types: list[str] = field(default_factory=list)
    mentions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"term": self.term, "types": list(self.types), "mentions": self.mentions}


@dataclass
class TermMatch:
    """A chunk term and the glossary candidates rapidfuzz retrieved for it.

    The threshold is stored on the result rather than applied by the matcher, so a
    per-chunk file can report the near miss (below the threshold) as well as the
    accepted match, and the same matcher can be reused at another threshold.
    """

    term: str
    types: list[str]
    mentions: int
    candidates: list[MatchCandidate] = field(default_factory=list)
    threshold: float = DEFAULT_CHUNK_GLOSSARY_THRESHOLD

    @property
    def best(self) -> MatchCandidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def matched(self) -> bool:
        return bool(self.best is not None and self.best.score >= self.threshold)

    def to_dict(self) -> dict[str, Any]:
        best = self.best
        matched = self.matched
        return {
            "term": self.term,
            "types": list(self.types),
            "mentions": self.mentions,
            "matched": matched,
            "score": round(float(best.score), 2) if best is not None else 0.0,
            "glossary": best.entry.to_dict() if matched and best is not None else None,
            "closest": best.to_dict() if best is not None and not matched else None,
            "alternatives": [candidate.to_dict() for candidate in self.candidates[1:]],
        }


@dataclass
class ChunkGlossary:
    """One chunk's outcome: its NER terms, its matches and its written file."""

    chunk: ChunkInput
    terms: list[TermMatch] = field(default_factory=list)
    matched_terms: list[dict[str, Any]] = field(default_factory=list)
    entity_count: int = 0
    ner_calls: int = 0
    ner_seconds: float = 0.0
    match_seconds: float = 0.0
    output_path: str | None = None
    ok: bool = True
    error: str | None = None
    skipped: bool = False
    log: list[str] = field(default_factory=list)

    @property
    def term_count(self) -> int:
        return len(self.terms)

    @property
    def matched_count(self) -> int:
        return len(self.matched_terms)

    @property
    def unmatched_count(self) -> int:
        return sum(1 for term in self.terms if not term.matched)

    @property
    def elapsed_seconds(self) -> float:
        return self.ner_seconds + self.match_seconds

    def record(self) -> dict[str, Any]:
        """The flat per-chunk record stored in the manifest."""
        return {
            **self.chunk.to_dict(),
            "entity_count": self.entity_count,
            "term_count": self.term_count,
            "matched_count": self.matched_count,
            "unmatched_count": self.unmatched_count,
            "matched_terms": list(self.matched_terms),
            "output_path": self.output_path,
            "ok": self.ok,
            "error": self.error,
            "skipped": self.skipped,
            "ner_calls": self.ner_calls,
            "ner_seconds": round(self.ner_seconds, 4),
            "match_seconds": round(self.match_seconds, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 4),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.record(),
            "terms": [term.to_dict() for term in self.terms],
            "log": list(self.log),
        }


@dataclass
class ChunkGlossaryResult:
    """Outcome of a whole stage-6 run."""

    ok: bool
    chunks_path: str
    glossary_path: str
    output_dir: str
    language: dict[str, Any] = field(default_factory=dict)
    backend: str = ""
    device: str = ""
    chunk_count: int = 0
    ok_count: int = 0
    processed_count: int = 0
    skipped_count: int = 0
    entry_count: int = 0
    entity_count: int = 0
    term_count: int = 0
    matched_term_count: int = 0
    unmatched_term_count: int = 0
    matched_chunk_count: int = 0
    dry_run: bool = False
    limited: bool = False
    settings: dict[str, Any] = field(default_factory=dict)
    totals: dict[str, int] = field(default_factory=dict)
    phases: list[dict[str, Any]] = field(default_factory=list)
    timing: dict[str, float] = field(default_factory=dict)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    manifest_path: str | None = None
    trace_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



__all__ = [
    "ChunkGlossary",
    "ChunkGlossaryResult",
    "Clock",
    "EntityTerm",
    "ProgressHook",
    "TermMatch",
]
