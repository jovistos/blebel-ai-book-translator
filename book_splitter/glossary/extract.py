"""The one model call of stage 4, and what it is allowed to cost.

``run_glossary`` asks one question per chapter: *which entities does this chapter
introduce, and how must each be written in the target language?* This module is
everything between that question and a validated :class:`GlossaryEntry` list:

* **Structured output, validated with Pydantic.** The provider is asked for the
  JSON object through its native structured-output mode when it has one, and
  through JSON mode otherwise; either way the answer is validated as
  :class:`~book_splitter.glossary.models.ChapterExtraction`. A model that wraps
  the object in prose or a code fence, or that drops a field, fails validation —
  which is a retry, not a corrupt dictionary.
* **Retries with the failure named.** Each attempt after the first carries a
  repair note quoting the previous failure, and the last attempt falls back to
  plain JSON mode. Every attempt's tokens are counted, so a chapter that needed
  three tries is *priced* as three tries.
* **Token accounting.** ``usage_metadata`` (or the older ``token_usage``) of every
  response is read and summed, because the run's whole cost story is "how many
  tokens did building this dictionary take".

Nothing here imports a provider: the extractor is handed a chat model and only
ever calls ``invoke`` on it (and ``with_structured_output``, when the model has
it). That is what lets the tests run the whole loop against a scripted double, and
what lets a local OpenAI-compatible server stand in for DeepSeek.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..core.errors import ExtractionError
from .models import ChapterExtraction
from .prompts import JSON_ONLY_NOTE, build_repair_note

#: Characters of the model's raw answer kept for the trace. Enough to see what
#: went wrong on a retry, not enough to turn a trace into a second manuscript.
DEFAULT_RAW_CHARS = 2_000
#: Extraction strategies. ``auto`` prefers native structured output and falls
#: back to JSON mode on the last attempt.
METHODS = ("auto", "structured", "json")

#: Fragments that mean "this provider has no usable structured-output mode" rather
#: than "this call went wrong". DeepSeek, for one, advertises a schema-capable
#: response format and then rejects it with a 400 ("This response_format type is
#: unavailable now"), so in ``auto`` mode the extractor stops asking for native
#: structured output for the rest of the run after the first such answer instead of
#: burning an attempt per chapter on a request that can never succeed.
UNSUPPORTED_MARKERS = (
    "response_format",
    "json_schema",
    "structured output",
    "tool_call",
    "tool calling",
    "invalid_request_error",
    "unavailable",
    "not supported",
    "unsupported",
)


def structured_unsupported(exc: BaseException) -> bool:
    """True when ``exc`` says the provider cannot do native structured output."""
    if isinstance(exc, NotImplementedError | AttributeError):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in UNSUPPORTED_MARKERS)


@dataclass
class TokenUsage:
    """Tokens spent, summed over every attempt of every chapter."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            calls=self.calls + other.calls,
        )

    def __iadd__(self, other: TokenUsage) -> TokenUsage:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.calls += other.calls
        return self

    @property
    def known(self) -> bool:
        """True when the provider reported at least one token count."""
        return bool(self.input_tokens or self.output_tokens or self.total_tokens)

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
        }


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


# --------------------------------------------------------------------------- #
# reading a provider's answer
# --------------------------------------------------------------------------- #
def _block_text(block: Any) -> str:
    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        return str(block.get("text") or block.get("content") or "")
    return str(getattr(block, "text", "") or "")


def message_text(message: Any) -> str:
    """The text of a chat response, whether it is a string or content blocks."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        return "".join(_block_text(block) for block in content)
    return str(content or "")


def usage_from_message(message: Any) -> TokenUsage:
    """Read the provider's token counts off one response.

    langchain-core puts them in ``usage_metadata``; older or less standard
    providers only fill ``response_metadata['token_usage']``. A provider that
    reports nothing still counts as one call, so the run's call count stays
    honest even when its token count is zero.
    """
    usage = TokenUsage(calls=1)
    meta = getattr(message, "usage_metadata", None)
    if isinstance(meta, dict) and meta:
        usage.input_tokens = int(meta.get("input_tokens") or 0)
        usage.output_tokens = int(meta.get("output_tokens") or 0)
        usage.total_tokens = int(meta.get("total_tokens") or 0)
    if not usage.total_tokens:
        response_meta = getattr(message, "response_metadata", None) or {}
        token_usage: Any = {}
        if isinstance(response_meta, dict):
            token_usage = response_meta.get("token_usage") or response_meta.get("usage") or {}
        if isinstance(token_usage, dict) and token_usage:
            usage.input_tokens = usage.input_tokens or int(
                token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0
            )
            usage.output_tokens = usage.output_tokens or int(
                token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0
            )
            usage.total_tokens = usage.total_tokens or int(token_usage.get("total_tokens") or 0)
    if not usage.total_tokens:
        usage.total_tokens = usage.input_tokens + usage.output_tokens
    return usage


def parse_json_object(text: str) -> Any:
    """Pull the JSON object out of an answer that may be wrapped in prose.

    Models that are asked for JSON still like to introduce it — "Here is the JSON
    you requested:" — or fence it in ```json. Both are recovered here; what is
    *not* recovered is a truncated or genuinely malformed body, which must fail so
    the caller retries instead of silently keeping half a chapter's entities.
    """
    value = str(text or "").strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[-1] if "\n" in value else ""
        if value.rstrip().endswith("```"):
            value = value.rstrip()[:-3]
    value = value.strip()
    try:
        return json.loads(value)
    except ValueError:
        pass
    start = value.find("{")
    end = value.rfind("}")
    if start != -1 and end > start:
        return json.loads(value[start : end + 1])
    raise ValueError("the answer holds no JSON object")


# --------------------------------------------------------------------------- #
# the extractor
# --------------------------------------------------------------------------- #
@dataclass
class _Attempt:
    """One model call's result: an answer, or the reason it was unusable.

    The usage is carried on both paths on purpose — a rejected answer still cost
    tokens, and the run's token report must include the calls that failed.
    """

    usage: TokenUsage = field(default_factory=TokenUsage)
    parsed: Any = None
    text: str = ""
    error: str | None = None


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
        method = str(method or "auto").lower()
        if method not in METHODS:
            raise ValueError(f"unknown extraction method {method!r}; pick one of {METHODS}")
        self._llm = llm
        self.name = str(name)
        self.max_attempts = max(1, int(max_attempts))
        self.method = method
        self.raw_limit = max(0, int(raw_limit))
        self._structured: Any = None
        self._json_llm: Any = None
        #: Set as soon as the provider proves it cannot fill a schema, so every
        #: later chapter goes straight to JSON mode.
        self.json_only = False

    # -- provider plumbing -------------------------------------------------- #
    def _structured_llm(self) -> Any:
        if self._structured is None:
            self._structured = self._llm.with_structured_output(
                ChapterExtraction, include_raw=True
            )
        return self._structured

    def _json_mode_llm(self) -> Any:
        if self._json_llm is None:
            try:
                self._json_llm = self._llm.bind(response_format={"type": "json_object"})
            except Exception:  # a provider that has no JSON mode: ask in the prompt
                self._json_llm = self._llm
        return self._json_llm

    def _messages(self, system_prompt: str, user_prompt: str) -> list[Any]:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except ImportError:  # pragma: no cover - langchain is a core dependency
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        return [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    # -- the loop ----------------------------------------------------------- #
    def _use_json(self, attempt: int) -> bool:
        if self.method == "json" or self.json_only:
            return True
        if self.method == "structured":
            return False
        # auto: native structured output first, plain JSON mode on the last try.
        return self.max_attempts > 1 and attempt == self.max_attempts

    def _call_structured(self, system_prompt: str, user_prompt: str) -> _Attempt:
        """One native structured-output call.

        A provider-level failure raises (there is no answer, so nothing to
        charge). An answer that arrived but did not fill the schema is returned
        as a failed attempt *with its usage*, because those tokens were spent —
        a retry loop that hides them under-reports the cost of the book.
        """
        result = self._structured_llm().invoke(self._messages(system_prompt, user_prompt))
        raw = result.get("raw") if isinstance(result, dict) else result
        usage = usage_from_message(raw) if raw is not None else TokenUsage(calls=1)
        text = message_text(raw) if raw is not None else ""
        if isinstance(result, dict):
            parsed = result.get("parsed")
            if parsed is None:
                error = result.get("parsing_error") or "empty answer"
                return _Attempt(usage=usage, text=text, error=f"structured output not filled: {error}")
            return _Attempt(parsed=parsed, usage=usage, text=text)
        return _Attempt(parsed=result, usage=usage, text=text)

    def _call_json(self, system_prompt: str, user_prompt: str) -> _Attempt:
        """One JSON-mode call, parsed into the same Pydantic model."""
        response = self._json_mode_llm().invoke(
            self._messages(system_prompt, user_prompt + JSON_ONLY_NOTE)
        )
        usage = usage_from_message(response)
        text = message_text(response)
        try:
            parsed = ChapterExtraction.model_validate(parse_json_object(text))
        except Exception as exc:
            return _Attempt(usage=usage, text=text, error=f"{type(exc).__name__}: {exc}")
        return _Attempt(parsed=parsed, usage=usage, text=text)

    def extract(self, *, system_prompt: str, user_prompt: str) -> ExtractionOutcome:
        """One chapter, up to ``max_attempts`` calls, all of them counted."""
        total = TokenUsage()
        last_error: str | None = None
        for attempt in range(1, self.max_attempts + 1):
            use_json = self._use_json(attempt)
            prompt = user_prompt if attempt == 1 else user_prompt + build_repair_note(last_error or "")
            try:
                result = (
                    self._call_json(system_prompt, prompt)
                    if use_json
                    else self._call_structured(system_prompt, prompt)
                )
            except Exception as exc:  # no answer at all: nothing to charge
                last_error = f"{type(exc).__name__}: {exc}"
                if not use_json and self.method == "auto" and structured_unsupported(exc):
                    # The provider cannot do this at all. Stop asking, for this
                    # chapter and for every one after it.
                    self.json_only = True
                continue
            total += result.usage
            if result.error:
                last_error = result.error
                continue
            try:
                # ``with_structured_output`` may already have validated the answer;
                # when it has not, the same Pydantic model decides — an answer with
                # an empty field is a failure here, not an entry in the dictionary.
                extraction = (
                    result.parsed
                    if isinstance(result.parsed, ChapterExtraction)
                    else ChapterExtraction.model_validate(result.parsed)
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                continue
            return ExtractionOutcome(
                extraction=extraction,
                usage=total,
                attempts=attempt,
                method="json" if use_json else "structured",
                error=last_error,
                raw_text=result.text[: self.raw_limit] if self.raw_limit else "",
            )
        raise ExtractionError(
            f"no valid extraction after {self.max_attempts} attempt(s): {last_error}"
        )


def make_extractor(
    settings: Any,
    *,
    max_attempts: int = 3,
    method: str = "auto",
    max_tokens: int | None = None,
) -> LangChainExtractor:
    """Build the DeepSeek (or any OpenAI-compatible) chat model behind the extractor.

    Imported here rather than at module import so the rest of the package — and
    the offline tests — never need a provider client to read the glossary code.
    """
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "model": settings.model,
        "base_url": settings.base_url,
        "api_key": settings.api_key,
        "temperature": getattr(settings, "temperature", 0.0),
        "timeout": getattr(settings, "timeout", 180.0),
        "max_retries": getattr(settings, "max_retries", 3),
    }
    if max_tokens:
        kwargs["max_tokens"] = int(max_tokens)
    llm = ChatOpenAI(**kwargs)
    return LangChainExtractor(
        llm,
        name=f"{getattr(settings, 'provider', 'llm')}/{getattr(settings, 'model', '')}",
        max_attempts=max_attempts,
        method=method,
    )


__all__ = [
    "DEFAULT_RAW_CHARS",
    "METHODS",
    "UNSUPPORTED_MARKERS",
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
