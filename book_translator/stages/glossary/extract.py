"""The one model call of stage 4, on top of the shared structured loop.

``run_glossary`` asks one question per chapter: *which entities does this chapter
introduce, and how must each be written in the target language?* This module is
everything between that question and a validated :class:`GlossaryEntry` list:

* **Structured output, validated with Pydantic.** The provider is asked for the
  JSON object through its native structured-output mode when it has one, and
  through JSON mode otherwise; either way the answer is validated as
  :class:`~book_translator.stages.glossary.models.ChapterExtraction`. A model that wraps
  the object in prose or a code fence, or that drops a field, fails validation —
  which is a retry, not a corrupt dictionary.
* **Retries with the failure named.** Each attempt after the first carries a
  repair note quoting the previous failure, and the last attempt falls back to
  plain JSON mode. Every attempt's tokens are counted, so a chapter that needed
  three tries is *priced* as three tries.
* **Token accounting.** ``usage_metadata`` (or the older ``token_usage``) of every
  response is read and summed, because the run's whole cost story is "how many
  tokens did building this dictionary take".

The retry loop itself is :class:`~book_translator.core.llm.StructuredCaller`; this
module supplies the chapter schema and the stage-specific outcome. The token and
provider-reading primitives are re-exported here for the callers that still
import them from stage 4, but their home is now
:mod:`book_translator.core.llm`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from book_translator.core.errors import ExtractionError
from book_translator.core.llm import (
    DEFAULT_RAW_CHARS,
    METHODS,
    StructuredCaller,
    TokenUsage,
    build_chat_model,
    message_text,
    model_label,
    parse_json_object,
    structured_unsupported,
    usage_from_message,
)

from .models import ChapterExtraction
from .prompts import JSON_ONLY_NOTE, build_repair_note


@dataclass
class ExtractionOutcome:
    """One chapter's accepted answer, plus what it cost to get it."""

    extraction: ChapterExtraction
    usage: TokenUsage = field(default_factory=TokenUsage)
    #: Model calls spent on this chapter, retries included.
    attempts: int = 1
    #: ``structured``, ``json``, or the injected extractor's own name.
    method: str = "structured"
    #: The failure of the attempt *before* the accepted one, when there was one.
    error: str | None = None
    #: The accepted answer as the model wrote it, clipped. For the trace only.
    raw_text: str = ""

    @property
    def entries(self) -> list[Any]:
        return list(self.extraction.entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": len(self.extraction.entries),
            "attempts": self.attempts,
            "method": self.method,
            "error": self.error,
            "usage": self.usage.to_dict(),
        }


@runtime_checkable
class GlossaryExtractor(Protocol):
    """What ``run_glossary`` needs from a model: prompts in, entries out.

    ``run_glossary`` never builds a chat client itself, so the tests pass a
    scripted double and a real run passes
    :class:`LangChainExtractor` — the runner cannot tell them apart.
    """

    name: str

    def extract(self, *, system_prompt: str, user_prompt: str) -> ExtractionOutcome:
        """Return the validated entries of one chapter, or raise."""
        ...


class LangChainExtractor:
    """Ask a chat model for one chapter's entries, and insist on a valid answer.

    ``llm`` is any langchain-core chat model (it must implement ``invoke``; the
    native structured-output path also uses ``with_structured_output``). Keeping
    the model injectable is what lets the retry loop itself be tested offline:
    the double answers with prose on the first call and real JSON on the second,
    and the test asserts the retry happened and both calls were counted.
    """

    def __init__(
        self,
        llm: Any,
        *,
        name: str = "langchain",
        max_attempts: int = 3,
        method: str = "auto",
        raw_limit: int = DEFAULT_RAW_CHARS,
    ) -> None:
        self._caller: StructuredCaller[ChapterExtraction] = StructuredCaller(
            llm,
            ChapterExtraction,
            name=name,
            max_attempts=max_attempts,
            method=method,
            raw_limit=raw_limit,
            json_note=JSON_ONLY_NOTE,
            repair_note=build_repair_note,
            failure=lambda usage, attempts, last: ExtractionError(
                f"no valid extraction after {attempts} attempt(s): {last}"
            ),
            json_on_last_attempt=True,
        )
        self.name = self._caller.name
        self.max_attempts = self._caller.max_attempts
        self.method = self._caller.method
        self.raw_limit = self._caller.raw_limit

    @property
    def json_only(self) -> bool:
        """True once the provider proved it cannot fill a schema."""
        return self._caller.json_only

    def extract(self, *, system_prompt: str, user_prompt: str) -> ExtractionOutcome:
        """One chapter, up to ``max_attempts`` calls, all of them counted."""
        outcome = self._caller.call(system_prompt=system_prompt, user_prompt=user_prompt)
        return ExtractionOutcome(
            extraction=outcome.value,
            usage=outcome.usage,
            attempts=outcome.attempts,
            method=outcome.method,
            error=outcome.error,
            raw_text=outcome.raw_text,
        )


def make_extractor(
    settings: Any,
    *,
    max_attempts: int = 3,
    method: str = "auto",
    max_tokens: int | None = None,
) -> LangChainExtractor:
    """Build the DeepSeek (or any OpenAI-compatible) chat model behind the extractor."""
    llm = build_chat_model(settings, max_tokens=max_tokens)
    return LangChainExtractor(
        llm,
        name=model_label(settings),
        max_attempts=max_attempts,
        method=method,
    )


__all__ = [
    "DEFAULT_RAW_CHARS",
    "METHODS",
    "ExtractionOutcome",
    "GlossaryExtractor",
    "LangChainExtractor",
    "TokenUsage",
    "make_extractor",
    "message_text",
    "parse_json_object",
    "structured_unsupported",
    "usage_from_message",
]
