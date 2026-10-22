"""The one model loop every LLM stage shares.

Two stages ask a model for a Pydantic-validated answer — the glossary asks for a
chapter's entities, the translator asks for one chunk's translation — and for a
long time each carried its own copy of the same twenty-line retry loop. This
module is that loop, once:

* :func:`build_chat_model` turns an :class:`~book_splitter.core.settings.LLMSettings`
  into a chat client. Provider construction lives here, not in a stage, so a
  local OpenAI-compatible server and DeepSeek are the same two lines.
* :class:`StructuredCaller` asks for native structured output, falls back to
  JSON mode when the provider cannot fill a schema, retries with the failure
  named, and prices every attempt. It is parameterised by the Pydantic schema,
  an optional validation hook (the translator's "is this actually a
  translation?" check), and an optional recovery hook (its unescaped-quote
  salvage).
* :class:`TokenUsage`, :func:`message_text`, :func:`parse_json_object`,
  :func:`usage_from_message` and :func:`structured_unsupported` are the small
  provider-reading primitives. They used to live in stage 4 and stage 7
  imported them from there; they are now stage-neutral.

Nothing here imports a provider at module import: ``langchain_openai`` is loaded
inside :func:`build_chat_model`, so the offline tests never need a provider
client and a split-only install keeps working.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, cast

from .errors import ToolError

#: Answer strategies shared by every structured stage. ``auto`` prefers native
#: structured output; ``structured`` insists on it; ``json`` starts in JSON mode.
METHODS: tuple[str, ...] = ("auto", "structured", "json")

#: Fragments that mean "this provider has no usable structured-output mode"
#: rather than "this call went wrong". DeepSeek, for one, advertises a
#: schema-capable response format and then rejects it with a 400 ("This
#: response_format type is unavailable now"), so in ``auto`` mode the caller
#: stops asking for native structured output for the rest of the run after the
#: first such answer instead of burning an attempt per item on a request that
#: can never succeed.
UNSUPPORTED_MARKERS: tuple[str, ...] = (
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

#: Characters of the model's raw answer kept for the trace. Enough to see what
#: went wrong on a retry, not enough to turn a trace into a second manuscript.
DEFAULT_RAW_CHARS = 2_000

#: Appended to a JSON-mode prompt. The provider binding asks for an object too,
#: but a provider without that mode — and a model that simply ignores it — still
#: needs to be told, so the rule is repeated where the model will read it.
JSON_ONLY_NOTE = (
    "\n\nReturn ONLY the JSON object. No prose, no explanation, no Markdown code fence "
    "around it — the first character of your answer must be `{` and the last must be `}`."
)


# --------------------------------------------------------------------------- #
# token accounting
# --------------------------------------------------------------------------- #
@dataclass
class TokenUsage:
    """Tokens spent, summed over every attempt of every item."""

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
    the caller retries instead of silently keeping half an item.
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


def structured_unsupported(exc: BaseException) -> bool:
    """True when ``exc`` says the provider cannot do native structured output."""
    if isinstance(exc, NotImplementedError | AttributeError):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in UNSUPPORTED_MARKERS)


# --------------------------------------------------------------------------- #
# building a client
# --------------------------------------------------------------------------- #
def build_chat_model(settings: Any, *, max_tokens: int | None = None) -> Any:
    """Build the chat model behind every structured stage.

    ``settings`` is any object carrying ``model``, ``base_url``, ``api_key`` and
    optionally ``temperature``/``timeout``/``max_retries`` — in practice
    :class:`~book_splitter.core.settings.LLMSettings`. The provider is imported
    lazily so a reader of this module never needs an API key, and every stage
    builds the identical client instead of keeping its own copy of the kwargs.
    """
    from langchain_openai import ChatOpenAI

    try:
        kwargs: dict[str, Any] = {
            "model": settings.model,
            "base_url": settings.base_url,
            "api_key": settings.api_key,
            "temperature": getattr(settings, "temperature", 0.0),
            "timeout": getattr(settings, "timeout", 180.0),
            "max_retries": getattr(settings, "max_retries", 3),
        }
    except AttributeError as exc:  # a settings object missing a required field
        raise ToolError(
            "the LLM settings need model, base_url and api_key to build a chat model: "
            f"{exc}"
        ) from exc
    if max_tokens:
        kwargs["max_tokens"] = int(max_tokens)
    return ChatOpenAI(**kwargs)


def model_label(settings: Any) -> str:
    """``provider/model`` — the stable name a run records for its answers."""
    provider = str(getattr(settings, "provider", "") or "")
    model = str(getattr(settings, "model", "") or "")
    if provider and model:
        return f"{provider}/{model}"
    return provider or model or "llm"


# --------------------------------------------------------------------------- #
# the generic loop
# --------------------------------------------------------------------------- #
T = TypeVar("T")
C = TypeVar("C")

#: A hook that rejects a validated answer for a non-schema reason.
Validator = Callable[[T, Any], str | None]
#: A hook that salvages an answer whose JSON could not be parsed.
Recovery = Callable[[str], T | None]
#: A hook that normalises whatever the provider returned into the schema object.
Coercer = Callable[[Any], T]
#: A hook that builds the run-fatal error when every attempt failed.
FailureFactory = Callable[[TokenUsage, int, str | None], Exception]
#: The note appended to a retry, quoting the previous failure.
RepairNote = Callable[[str], str]


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
    recovered: bool = False


@dataclass
class StructuredOutcome(Generic[T]):
    """One item's accepted answer, plus what it cost to get it."""

    value: T
    usage: TokenUsage = field(default_factory=TokenUsage)
    #: Model calls spent on this item, retries included.
    attempts: int = 1
    #: ``structured``, ``json``, or an injected caller's own name.
    method: str = "structured"
    #: The failure of the attempt *before* the accepted one, when there was one.
    error: str | None = None
    #: True when the accepted answer was not valid JSON and had to be recovered.
    recovered: bool = False
    #: The accepted answer as the model wrote it, clipped. For the trace only.
    raw_text: str = ""


def _default_coerce(schema: type[T], parsed: Any) -> T:
    """Validate a provider value as ``schema``, accepting an instance or a mapping.

    A provider that returns an object of its own shape is passed to Pydantic as
    well: it may expose the attributes the schema needs, and a genuinely wrong
    shape fails validation, which is a retry rather than a corrupt record.
    """
    if isinstance(parsed, schema):
        return parsed
    return cast(T, schema.model_validate(parsed))  # type: ignore[attr-defined]


def _no_validation(_value: Any, _context: Any) -> str | None:
    return None


class StructuredCaller(Generic[T]):
    """Ask a chat model for one validated answer, and insist on getting it.

    The caller is stage-neutral: the schema, the prompt fragment that
    introduces the JSON object, the repair note and the optional validation /
    recovery hooks are injected, so the glossary and the translator share this
    loop without either importing the other.

    ``json_on_last_attempt`` encodes the one real difference between the two
    stages. The glossary asks first for native structured output and switches to
    plain JSON on its final attempt even when the provider has not refused;
    the translator keeps asking for the schema and flips only after a provider
    refusal, because its answer is one field and a retry is a retry of the same
    task.
    """

    def __init__(
        self,
        llm: Any,
        schema: type[T],
        *,
        name: str = "langchain",
        max_attempts: int = 3,
        method: str = "auto",
        raw_limit: int = DEFAULT_RAW_CHARS,
        json_note: str = "",
        repair_note: RepairNote | None = None,
        validator: Validator[T] | None = None,
        recovery: Recovery[T] | None = None,
        coerce: Coercer[T] | None = None,
        failure: FailureFactory | None = None,
        json_on_last_attempt: bool = True,
    ) -> None:
        method = str(method or "auto").lower()
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; pick one of {METHODS}")
        self._llm = llm
        self._schema = schema
        self.name = str(name)
        self.max_attempts = max(1, int(max_attempts))
        self.method = method
        self.raw_limit = max(0, int(raw_limit))
        self.json_note = str(json_note or "")
        self._repair_note = repair_note or (
            lambda error: f"\n\nYour previous answer was rejected: {error}\n"
            "Answer again, returning only the JSON object described above."
        )
        self._validator = validator or _no_validation
        self._recovery = recovery
        self._coerce = coerce or (lambda parsed: _default_coerce(self._schema, parsed))
        self._failure = failure or (
            lambda usage, attempts, last: ToolError(
                f"no valid answer after {attempts} attempt(s): {last}"
            )
        )
        self.json_on_last_attempt = bool(json_on_last_attempt)
        self._structured: Any = None
        self._json_llm: Any = None
        #: Set as soon as the provider proves it cannot fill a schema, so every
        #: later item goes straight to JSON mode.
        self.json_only = False

    # -- provider plumbing -------------------------------------------------- #
    def _structured_llm(self) -> Any:
        if self._structured is None:
            self._structured = self._llm.with_structured_output(
                self._schema, include_raw=True
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
        if not self.json_on_last_attempt:
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
        parsed = result.get("parsed") if isinstance(result, dict) else result
        if parsed is None:
            error = (
                (result.get("parsing_error") or "empty answer")
                if isinstance(result, dict)
                else "empty answer"
            )
            return _Attempt(
                usage=usage,
                text=text,
                error=f"structured output not filled: {error}",
            )
        try:
            return _Attempt(parsed=self._coerce(parsed), usage=usage, text=text)
        except Exception as exc:
            return _Attempt(usage=usage, text=text, error=f"{type(exc).__name__}: {exc}")

    def _call_json(self, system_prompt: str, user_prompt: str) -> _Attempt:
        """One JSON-mode call, parsed into the same Pydantic model.

        Strict JSON is tried first. A schema that offers a recovery hook gets one
        more chance when the answer is complete but malformed — the translator's
        unescaped quote — instead of a whole item's tokens spent on a retry the
        model may refuse.
        """
        response = self._json_mode_llm().invoke(
            self._messages(system_prompt, user_prompt + self.json_note)
        )
        usage = usage_from_message(response)
        text = message_text(response)
        try:
            return _Attempt(parsed=self._coerce(parse_json_object(text)), usage=usage, text=text)
        except Exception as exc:
            recovered = self._recover(text) if self._recovery is not None else None
            if recovered is None:
                return _Attempt(usage=usage, text=text, error=f"{type(exc).__name__}: {exc}")
            return _Attempt(parsed=recovered, usage=usage, text=text, recovered=True)

    def _recover(self, text: str) -> T | None:
        if self._recovery is None:
            return None
        try:
            return self._recovery(text)
        except Exception:
            return None

    def call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        validation_context: Any = None,
    ) -> StructuredOutcome[T]:
        """One item, up to ``max_attempts`` calls, all of them counted."""
        total = TokenUsage()
        last_error: str | None = None
        for attempt in range(1, self.max_attempts + 1):
            use_json = self._use_json(attempt)
            prompt = (
                user_prompt
                if attempt == 1
                else user_prompt + self._repair_note(last_error or "")
            )
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
                    # item and for every one after it.
                    self.json_only = True
                continue
            total += result.usage
            if result.error:
                last_error = result.error
                continue
            try:
                problem = self._validator(result.parsed, validation_context)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                continue
            if problem:
                last_error = problem
                continue
            return StructuredOutcome(
                value=result.parsed,
                usage=total,
                attempts=attempt,
                method="json" if use_json else "structured",
                error=last_error,
                recovered=result.recovered,
                raw_text=result.text[: self.raw_limit] if self.raw_limit else "",
            )
        raise self._failure(total, self.max_attempts, last_error)


__all__ = [
    "DEFAULT_RAW_CHARS",
    "JSON_ONLY_NOTE",
    "METHODS",
    "UNSUPPORTED_MARKERS",
    "StructuredCaller",
    "StructuredOutcome",
    "TokenUsage",
    "build_chat_model",
    "message_text",
    "model_label",
    "parse_json_object",
    "structured_unsupported",
    "usage_from_message",
]
