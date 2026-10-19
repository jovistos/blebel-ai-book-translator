"""The one model call of stage 7, and what it is allowed to cost.

``run_translation`` asks one question per chunk: *translate this chunk into the
target language, using the glossary rows it mentions.* This module is everything
between that question and a validated :class:`ChunkTranslationAnswer`:

* **Structured output, validated with Pydantic.** The provider is asked for the
  answer through its native structured-output mode when it has one, and through
  JSON mode otherwise; either way the answer is validated as
  :class:`~book_splitter.translation.models.ChunkTranslationAnswer`. A model that
  wraps the object in prose or a code fence, drops the ``translation`` field or
  returns an empty string fails validation — which is a retry, not an empty
  chunk in the book.
* **Retries with the failure named.** Each attempt after the first carries a
  repair note quoting the previous failure; ``auto`` switches to plain JSON mode
  when the provider proves it cannot fill a schema. A translation that merely
  repeats the source, or one that is a fraction of the source's length (a summary
  instead of a translation), is treated as a failure too. A complete answer that
  is only invalid JSON — a quote inside the text was left unescaped — is salvaged
  before it becomes a retry.
* **Token accounting.** ``usage_metadata`` (or the older ``token_usage``) of every
  response is read and summed, because the run's whole cost story is "how many
  tokens did translating this book take".

Nothing here imports a provider or ``langchain``: the translator is handed a chat
model and only ever calls ``invoke`` on it (and ``with_structured_output``, when
the model has it). That is what lets the tests run the whole retry loop against a
scripted double, and what lets a local OpenAI-compatible server stand in for
DeepSeek.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..core.constants import DEFAULT_TRANSLATION_RAW_CHARS
from ..core.errors import TranslationError
from ..glossary.extract import (
    TokenUsage,
    message_text,
    parse_json_object,
    structured_unsupported,
    usage_from_message,
)
from ..glossary.prompts import JSON_ONLY_NOTE
from .models import ChunkTranslationAnswer
from .prompts import build_repair_note

#: Extraction strategies. ``auto`` asks for native structured output and falls
#: back to JSON mode only when the provider proves it cannot fill a schema;
#: ``structured`` insists on native structured output; ``json`` starts in JSON
#: mode. There is deliberately no last-attempt JSON switch: this answer is a
#: single ``translation`` field, and a retry is a retry of the same task.
METHODS = ("auto", "structured", "json")

#: A whole answer wrapped in a code fence is stripped: models do it even when the
#: schema asks them not to. Only a fence that wraps the *entire* translation is
#: removed, and only for the Markdown/plain/empty language tags — a chunk that is
#: itself a code block keeps its fence.
_FENCE = re.compile(r"\A[ \t]*```([^\n`]*)\n(.*?)\n```[ \t]*\Z", re.DOTALL)
_FENCE_TAGS = {"", "json", "markdown", "md", "text", "txt", "plain", "plaintext"}

#: The one field the answer holds, and the opening quote of its value. A model in
#: JSON mode occasionally leaves a raw ``"`` inside the translation — the closing
#: quotation mark of Serbian dialogue is typed as the ASCII quote instead of
#: ``“`` — and the whole answer stops being parseable at that character.
_TRANSLATION_VALUE = re.compile(r'"translation"\s*:\s*"')

#: A quoting comma that starts another field. A quote followed by this closes the
#: ``translation`` value even though the object keeps going; a quote followed by
#: anything else is part of the text.
_NEXT_FIELD = re.compile(r'\s*,\s*"[^"\\]*"\s*:')

#: JSON's one-character escapes, for decoding a recovered value.
_JSON_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}

#: A model that ignores the "no prose" rule usually introduces its answer with
#: this, and ends it with a helpful offer. Both are removed from the *edges* of
#: the text only: the same words in the middle of a chapter are the author's.
_PREAMBLE = re.compile(
    r"\A[ \t]*(?:here(?:'s| is) (?:the|your|my) (?:complete )?translation[^\n]*|"
    r"translation[ \t]*:)[ \t]*(?:\n+|$)",
    re.IGNORECASE,
)
_POSTAMBLE = re.compile(
    r"\n+[ \t]*(?:let me know[^\n]*|i hope this helps[^\n]*|feel free to[^\n]*)[ \t]*\Z",
    re.IGNORECASE,
)

#: An answer this much smaller than its source is a summary, not a translation.
#: The ratio is deliberately loose: a terse target language can be shorter, but
#: not by five sixths.
MIN_LENGTH_RATIO = 0.15
#: Below this many source characters the identical/short checks are meaningless:
#: "42" translates to "42", and a chapter title may legitimately be one word.
MIN_CHECK_CHARS = 40
MIN_RATIO_CHECK_CHARS = 200


def clean_translation(text: Any) -> str:
    """The model's answer as publishable chunk text.

    Strips a whole-answer code fence, a leading "Here is the translation:" line
    and a trailing "let me know…" line, then removes surrounding blank lines but
    keeps the text's indentation. Trailing newlines are the runner's job (it has
    the source chunk to match), so they are not added here.
    """
    value = str(text or "").replace("\ufeff", "")
    fence = _FENCE.match(value)
    if fence and fence.group(1).strip().lower() in _FENCE_TAGS:
        value = fence.group(2)
    value = _PREAMBLE.sub("", value)
    value = _POSTAMBLE.sub("", value)
    lines = value.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def translation_problem(text: str, source_text: str = "") -> str | None:
    """Why an answer is not usable, or ``None`` when it is.

    Empty answers are always a failure. The source-relative checks only run on a
    long-enough source: an unchanged two-character chunk may be a correct
    translation (a number, a name), while an unchanged paragraph is not, and a
    five-line "translation" of a five-page chunk is a summary.
    """
    cleaned = clean_translation(text)
    if not cleaned.strip():
        return "the translation is empty"
    if not any(character.isalnum() for character in cleaned):
        return "the translation holds no words"
    source = " ".join(str(source_text or "").split())
    answer = " ".join(cleaned.split())
    if not source:
        return None
    if len(source) >= MIN_CHECK_CHARS and answer.casefold() == source.casefold():
        return "the translation is the source text unchanged"
    if len(source) >= MIN_RATIO_CHECK_CHARS and len(answer) < MIN_LENGTH_RATIO * len(source):
        return (
            f"the translation is suspiciously short ({len(answer)} characters for a "
            f"{len(source)}-character source), so text was omitted or summarised"
        )
    return None


def _decode_json_string(value: str) -> str:
    """Decode the body of a JSON string, tolerating an unescaped quote.

    ``json.loads`` cannot be used for the recovered value: the reason recovery
    runs at all is usually an unescaped ``"`` inside it. A single pass over the
    backslash escapes keeps ``\\n``, ``\\"`` and ``\\uXXXX`` honest while every
    other character — the unescaped quote included — is passed through.
    """
    out: list[str] = []
    index = 0
    length = len(value)
    while index < length:
        char = value[index]
        if char != "\\" or index + 1 >= length:
            out.append(char)
            index += 1
            continue
        escape = value[index + 1]
        if escape == "u" and index + 5 < length:
            digits = value[index + 2 : index + 6]
            try:
                code = int(digits, 16)
            except ValueError:
                pass
            else:
                # A surrogate pair is how JSON spells a character outside the
                # BMP; decode the pair rather than leaving two lone surrogates.
                if (
                    0xD800 <= code <= 0xDBFF
                    and value[index + 6 : index + 8] == "\\u"
                    and index + 11 < length
                ):
                    low_digits = value[index + 8 : index + 12]
                    try:
                        low = int(low_digits, 16)
                    except ValueError:
                        low = -1
                    if 0xDC00 <= low <= 0xDFFF:
                        out.append(chr(0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)))
                        index += 12
                        continue
                out.append(chr(code))
                index += 6
                continue
        out.append(_JSON_ESCAPES.get(escape, "\\" + escape))
        index += 2
    return "".join(out)


def recover_translation(text: str) -> str | None:
    """The translation from a JSON object whose string was left unescaped.

    The value starts at the opening quote of ``"translation"`` and ends at the
    first quote that closes the object (only whitespace before the final ``}``)
    or that is followed by another field. Every earlier quote is text the model
    forgot to escape. A missing closing brace, a quote followed by something that
    is neither, or an empty value means the shape is not the one this recovery
    understands, and ``None`` makes the caller keep the original failure (and
    retry).
    """
    value = str(text or "").replace("\ufeff", "").strip()
    match = _TRANSLATION_VALUE.search(value)
    if not match:
        return None
    start = match.end()
    brace = value.rfind("}")
    if brace < start:
        return None
    index = start
    while index < brace:
        if value[index] == "\\":
            index += 2  # an escaped character cannot close the string
            continue
        if value[index] == '"':
            tail = value[index + 1 : brace]
            if not tail.strip() or _NEXT_FIELD.match(tail):
                recovered = _decode_json_string(value[start:index])
                return recovered if recovered.strip() else None
        index += 1
    return None


@dataclass
class TranslationOutcome:
    """One chunk's accepted translation, plus what it cost to get it."""

    answer: ChunkTranslationAnswer
    usage: TokenUsage = field(default_factory=TokenUsage)
    #: Model calls spent on this chunk, retries included.
    attempts: int = 1
    #: ``structured``, ``json``, or the injected translator's own name.
    method: str = "structured"
    #: The failure of the attempt *before* the accepted one, when there was one.
    error: str | None = None
    #: True when the accepted answer was not valid JSON and had to be recovered.
    recovered: bool = False
    #: The accepted answer as the model wrote it, clipped. For the trace only.
    raw_text: str = ""

    @property
    def text(self) -> str:
        return clean_translation(self.answer.translation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chars": len(self.text),
            "attempts": self.attempts,
            "method": self.method,
            "error": self.error,
            "recovered": self.recovered,
            "usage": self.usage.to_dict(),
        }


@runtime_checkable
class ChunkTranslator(Protocol):
    """What ``run_translation`` needs from a model: prompts in, a translation out.

    ``source_text`` is passed for validation, not for translation: a double can
    ignore it, while the real translator uses it to reject an answer that merely
    echoes the source or summarises it.
    """

    name: str

    def translate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        source_text: str = "",
    ) -> TranslationOutcome:
        """Return the validated translation of one chunk, or raise."""
        ...


# --------------------------------------------------------------------------- #
# the translator
# --------------------------------------------------------------------------- #
@dataclass
class _Attempt:
    """One model call's result: a validated answer, or the reason it was unusable.

    Usage is carried on both paths on purpose — a rejected answer still cost
    tokens, and the run's token report must include the calls that failed.
    """

    usage: TokenUsage = field(default_factory=TokenUsage)
    answer: ChunkTranslationAnswer | None = None
    text: str = ""
    error: str | None = None
    #: True when the answer was recovered from malformed JSON instead of parsed.
    recovered: bool = False


class LangChainTranslator:
    """Ask a chat model to translate one chunk, and insist on a valid answer.

    ``llm`` is any langchain-core chat model (it must implement ``invoke``; the
    native structured-output path also uses ``with_structured_output``). Keeping
    the model injectable is what lets the retry loop itself be tested offline: the
    double answers with prose on the first call and valid JSON on the second, and
    the test asserts the retry happened and both calls were counted.
    """

    def __init__(
        self,
        llm: Any,
        *,
        name: str = "langchain",
        max_attempts: int = 3,
        method: str = "auto",
        raw_limit: int = DEFAULT_TRANSLATION_RAW_CHARS,
    ) -> None:
        method = str(method or "auto").lower()
        if method not in METHODS:
            raise ValueError(f"unknown translation method {method!r}; pick one of {METHODS}")
        self._llm = llm
        self.name = str(name)
        self.max_attempts = max(1, int(max_attempts))
        self.method = method
        self.raw_limit = max(0, int(raw_limit))
        self._structured: Any = None
        self._json_llm: Any = None
        #: Set as soon as the provider proves it cannot fill a schema, so every
        #: later chunk goes straight to JSON mode.
        self.json_only = False

    # -- provider plumbing -------------------------------------------------- #
    def _structured_llm(self) -> Any:
        if self._structured is None:
            self._structured = self._llm.with_structured_output(
                ChunkTranslationAnswer, include_raw=True
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
    def _use_json(self) -> bool:
        return self.method == "json" or self.json_only

    def _call_structured(self, system_prompt: str, user_prompt: str) -> _Attempt:
        """One native structured-output call.

        A provider-level failure raises (there is no answer, so nothing to
        charge). An answer that arrived but did not fill the schema is returned
        as a failed attempt *with its usage*, because those tokens were spent.
        """
        result = self._structured_llm().invoke(self._messages(system_prompt, user_prompt))
        raw = result.get("raw") if isinstance(result, dict) else result
        usage = usage_from_message(raw) if raw is not None else TokenUsage(calls=1)
        text = message_text(raw) if raw is not None else ""
        if isinstance(result, dict):
            parsed = result.get("parsed")
            if parsed is None:
                error = result.get("parsing_error") or "empty answer"
                return _Attempt(
                    usage=usage,
                    text=text,
                    error=f"structured output not filled: {error}",
                )
            try:
                answer = _as_answer(parsed)
            except Exception as exc:
                return _Attempt(usage=usage, text=text, error=f"{type(exc).__name__}: {exc}")
            return _Attempt(answer=answer, usage=usage, text=text)
        try:
            answer = _as_answer(result)
        except Exception as exc:
            return _Attempt(usage=usage, text=text, error=f"{type(exc).__name__}: {exc}")
        return _Attempt(answer=answer, usage=usage, text=text)

    def _call_json(self, system_prompt: str, user_prompt: str) -> _Attempt:
        """One JSON-mode call, parsed into the same Pydantic model.

        Strict JSON is tried first. When it fails only because the translation
        carried an unescaped quote, the answer is recovered instead of thrown
        away: the text is complete, and the alternative is a whole chunk's
        tokens spent on a retry the model may refuse.
        """
        response = self._json_mode_llm().invoke(
            self._messages(system_prompt, user_prompt + JSON_ONLY_NOTE)
        )
        usage = usage_from_message(response)
        text = message_text(response)
        try:
            parsed = ChunkTranslationAnswer.model_validate(parse_json_object(text))
        except Exception as exc:
            recovered = recover_translation(text)
            if recovered is None:
                return _Attempt(usage=usage, text=text, error=f"{type(exc).__name__}: {exc}")
            return _Attempt(
                answer=ChunkTranslationAnswer(translation=recovered),
                usage=usage,
                text=text,
                recovered=True,
            )
        return _Attempt(answer=parsed, usage=usage, text=text)

    def translate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        source_text: str = "",
    ) -> TranslationOutcome:
        """One chunk, up to ``max_attempts`` calls, all of them counted."""
        total = TokenUsage()
        last_error: str | None = None
        for attempt in range(1, self.max_attempts + 1):
            use_json = self._use_json()
            prompt = (
                user_prompt
                if attempt == 1
                else user_prompt + build_repair_note(last_error or "")
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
                    # chunk and for every one after it.
                    self.json_only = True
                continue
            total += result.usage
            if result.error or result.answer is None:
                last_error = result.error or "the answer held no translation"
                continue
            problem = translation_problem(result.answer.translation, source_text)
            if problem:
                last_error = problem
                continue
            return TranslationOutcome(
                answer=result.answer,
                usage=total,
                attempts=attempt,
                method="json" if use_json else "structured",
                error=last_error,
                recovered=result.recovered,
                raw_text=result.text[: self.raw_limit] if self.raw_limit else "",
            )
        raise _failure(total, self.max_attempts, last_error)


def _failure(usage: TokenUsage, attempts: int, last_error: str | None) -> TranslationError:
    """The run-fatal error, carrying what the failed attempts still cost."""
    error = TranslationError(
        f"no usable translation after {attempts} attempt(s): {last_error}"
    )
    error.usage = usage  # type: ignore[attr-defined]
    error.attempts = attempts  # type: ignore[attr-defined]
    return error


def _as_answer(parsed: Any) -> ChunkTranslationAnswer:
    """Normalise whatever the provider's structured mode returned."""
    if isinstance(parsed, ChunkTranslationAnswer):
        return parsed
    if isinstance(parsed, dict):
        return ChunkTranslationAnswer.model_validate(parsed)
    # A provider that returns an already-validated object of its own schema.
    translation = getattr(parsed, "translation", None)
    return ChunkTranslationAnswer(translation=translation if translation is not None else "")


def make_translator(
    settings: Any,
    *,
    max_attempts: int = 3,
    method: str = "auto",
    max_tokens: int | None = None,
) -> LangChainTranslator:
    """Build the DeepSeek (or any OpenAI-compatible) chat model behind the translator.

    Imported here rather than at module import so the rest of the package — and
    the offline tests — never need a provider client to read the translation code.
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
    return LangChainTranslator(
        llm,
        name=f"{getattr(settings, 'provider', 'llm')}/{getattr(settings, 'model', '')}",
        max_attempts=max_attempts,
        method=method,
    )


__all__ = [
    "METHODS",
    "MIN_CHECK_CHARS",
    "MIN_LENGTH_RATIO",
    "MIN_RATIO_CHECK_CHARS",
    "ChunkTranslator",
    "LangChainTranslator",
    "TranslationOutcome",
    "clean_translation",
    "make_translator",
    "recover_translation",
    "translation_problem",
]
