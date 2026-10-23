"""Result and record types for the translation stage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from book_translator.core.constants import DEFAULT_SOURCE_LANGUAGE, DEFAULT_TARGET_LANGUAGE
from book_translator.core.llm import TokenUsage
from book_translator.documents.chunks import ChunkInput

Clock = Callable[[], float]
ProgressHook = Callable[[dict[str, Any], int, int], None]
ChunkStartHook = Callable[[dict[str, Any]], None]


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GlossaryRow:
    """One binding rendering, as the prompt consumes it."""

    term: str
    translation: str
    category: str = "terminology"

    def to_dict(self) -> dict[str, str]:
        return {
            "term": self.term,
            "translation": self.translation,
            "category": self.category,
        }


@dataclass
class ChunkTranslation:
    """One chunk's translation: what it cost, where it went, and why it failed."""

    chunk: ChunkInput
    terms: list[GlossaryRow] = field(default_factory=list)
    glossary_missing: bool = False
    context_chars: int = 0
    source_chars: int = 0
    translation_chars: int = 0
    translation_sha256: str = ""
    #: The exact inputs that produced this translation (glossary, context,
    #: method, model). A resume only reuses the file when this still matches.
    resume_key: str = ""
    attempts: int = 0
    method: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    elapsed_seconds: float = 0.0
    output_path: str | None = None
    prompt_path: str | None = None
    debug_path: str | None = None
    ok: bool = True
    error: str | None = None
    skipped: bool = False
    reused: bool = False
    log: list[str] = field(default_factory=list)
    #: The translated text. Kept out of every serialised record (it is the file).
    translation: str = field(default="", repr=False, compare=False)
    #: The accepted answer as the model wrote it, clipped. For the trace only.
    raw_answer: str = field(default="", repr=False, compare=False)
    #: The full prompt/input payload written to the debug folder.
    debug: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def glossary_term_count(self) -> int:
        return len(self.terms)

    def record(self) -> dict[str, Any]:
        """The flat per-chunk record stored in the manifest."""
        return {
            **self.chunk.to_dict(),
            "glossary_terms": self.glossary_term_count,
            "glossary_missing": self.glossary_missing,
            "context_chars": self.context_chars,
            "source_chars": self.source_chars,
            "translation_chars": self.translation_chars,
            "translation_sha256": self.translation_sha256,
            "resume_key": self.resume_key,
            "attempts": self.attempts,
            "method": self.method,
            "tokens": self.usage.to_dict(),
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "output_path": self.output_path,
            "prompt_path": self.prompt_path,
            "debug_path": self.debug_path,
            "ok": self.ok,
            "error": self.error,
            "skipped": self.skipped,
            "reused": self.reused,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.record(),
            "terms": [term.to_dict() for term in self.terms],
            "log": list(self.log),
            "raw_answer": self.raw_answer,
        }


@dataclass
class TranslationResult:
    """Outcome of a whole stage-7 run."""

    ok: bool
    chunks_path: str
    chunk_glossary_path: str
    output_dir: str
    prompts_dir: str
    source_language: str = DEFAULT_SOURCE_LANGUAGE
    target_language: str = DEFAULT_TARGET_LANGUAGE
    book_title: str = ""
    source_file: str = ""
    translator: str = ""
    provider: str = ""
    model: str = ""
    glossary_scope: str = "chunk"
    glossary_entries: int = 0
    chunk_count: int = 0
    ok_count: int = 0
    processed_count: int = 0
    skipped_count: int = 0
    reused_count: int = 0
    dry_run: bool = False
    save: bool = True
    resumed: bool = False
    limited: bool = False
    settings: dict[str, Any] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)
    totals: dict[str, int] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    manifest_path: str | None = None
    trace_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



__all__ = ["ChunkStartHook", "ChunkTranslation", "Clock", "GlossaryRow", "ProgressHook", "TranslationResult"]
