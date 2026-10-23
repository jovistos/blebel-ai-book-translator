"""Offline tests for the extraction loop: retries, fallbacks and token counting.

No provider, no network. The chat model is a scripted double, which is exactly
what makes the retry loop testable: the first answer can be prose, the second can
be a valid object, and the test asserts both that the retry happened and that the
run was charged for both calls. That last part is not cosmetic — the stage's whole
report is "what did this dictionary cost", so an attempt that is not counted is a
lie in the summary.
"""

from __future__ import annotations

from typing import Any

import pytest
from book_translator.core.errors import ExtractionError
from book_translator.stages.glossary.extract import (
    ExtractionOutcome,
    LangChainExtractor,
    TokenUsage,
    make_extractor,
    message_text,
    parse_json_object,
    structured_unsupported,
    usage_from_message,
)
from book_translator.stages.glossary.models import ChapterExtraction, GlossaryEntry
from book_translator.stages.glossary.prompts import JSON_ONLY_NOTE


# --------------------------------------------------------------------------- #
# doubles
# --------------------------------------------------------------------------- #
class FakeMessage:
    """Just enough of an ``AIMessage``: text plus whatever usage was scripted."""

    def __init__(self, text: str = "", usage: dict | None = None, blocks: bool = False) -> None:
        self.content = [{"type": "text", "text": text}] if blocks else text
        self.usage_metadata = usage or {}
        self.response_metadata: dict[str, Any] = {}


class ScriptedModel:
    """A chat-model double whose answers are scripted in order.

    ``answers`` are dicts: ``text``, ``usage``, and for the structured path
    ``parsed`` / ``parsing_error``. An answer that is an ``Exception`` is raised
    instead of returned, which is how a transport failure is scripted.
    """

    def __init__(self, answers: list, *, structured: bool = True, bindable: bool = True) -> None:
        self._answers = list(answers)
        self._structured = structured
        self._bindable = bindable
        self.calls: list[str] = []
        self.seen: list[list] = []

    # -- provider plumbing ----------------------------------------------- #
    def with_structured_output(self, schema, include_raw=False):
        if not self._structured:
            raise NotImplementedError("this double has no structured output")
        return _StructuredRunner(self)

    def bind(self, **kwargs):
        if not self._bindable:
            raise NotImplementedError("this double has no JSON mode")
        return self

    def invoke(self, messages):
        self.calls.append("json")
        self.seen.append(list(messages))
        answer = self._next()
        if isinstance(answer, Exception):
            raise answer
        return FakeMessage(answer.get("text", ""), answer.get("usage"), answer.get("blocks", False))

    def _next(self) -> Any:
        if not self._answers:
            raise AssertionError("the double ran out of scripted answers")
        return self._answers.pop(0)


class _StructuredRunner:
    def __init__(self, model: ScriptedModel) -> None:
        self.model = model

    def invoke(self, messages):
        self.model.calls.append("structured")
        self.model.seen.append(list(messages))
        answer = self.model._next()
        if isinstance(answer, Exception):
            raise answer
        raw = FakeMessage(answer.get("text", ""), answer.get("usage"))
        return {
            "raw": raw,
            "parsed": answer.get("parsed"),
            "parsing_error": answer.get("parsing_error"),
        }


ENTRY = {
    "term": "Potter",
    "category": "character",
    "translation": "Poter",
    "brief_context": "surname; female, singular",
}
GOOD = {"entries": [ENTRY]}
USAGE = {"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100}


def parsed():
    return ChapterExtraction.model_validate(GOOD)


def test_a_good_structured_answer_costs_one_call():
    model = ScriptedModel([{"parsed": parsed(), "text": '{"entries": [...]}', "usage": USAGE}])
    outcome = LangChainExtractor(model).extract(system_prompt="S", user_prompt="U")

    assert model.calls == ["structured"]
    assert outcome.attempts == 1
    assert outcome.method == "structured"
    assert outcome.error is None
    assert [entry.translation for entry in outcome.entries] == ["Poter"]
    assert outcome.usage.to_dict() == {**USAGE, "calls": 1}
    assert outcome.raw_text == '{"entries": [...]}'


def test_a_rejected_answer_is_retried_and_the_failure_is_named_in_the_prompt():
    model = ScriptedModel(
        [
            {"parsed": None, "parsing_error": ValueError("missing entries"), "text": "oops"},
            {"parsed": parsed(), "text": "{}", "usage": USAGE},
        ]
    )
    outcome = LangChainExtractor(model, max_attempts=2, method="structured").extract(
        system_prompt="S", user_prompt="ORIGINAL"
    )

    assert model.calls == ["structured", "structured"]
    assert outcome.attempts == 2
    assert "missing entries" in (outcome.error or "")
    # Both calls are charged, not just the successful one.
    assert outcome.usage.calls == 2
    assert outcome.usage.total_tokens == 1100
    # The retry prompt carries the original request *and* the repair note.
    retry = model.seen[1][1].content
    assert retry.startswith("ORIGINAL")
    assert "previous answer was rejected" in retry
    assert "missing entries" in retry


def test_the_last_auto_attempt_falls_back_to_json_mode():
    model = ScriptedModel(
        [
            {"parsed": None, "parsing_error": ValueError("prose"), "text": "oops", "usage": USAGE},
            {"parsed": None, "parsing_error": ValueError("still prose"), "text": "oops", "usage": USAGE},
            {"text": 'Here you go:\n```json\n{"entries": []}\n```', "usage": USAGE},
        ]
    )
    outcome = LangChainExtractor(model, max_attempts=3, method="auto").extract(
        system_prompt="S", user_prompt="U"
    )

    assert model.calls == ["structured", "structured", "json"]
    assert outcome.method == "json"
    assert outcome.attempts == 3
    assert outcome.entries == []
    # Every response is charged, including the two that were rejected.
    assert outcome.usage.calls == 3
    assert outcome.usage.total_tokens == 3300
    # JSON mode is asked for explicitly in the prompt, not only in the binding.
    assert model.seen[-1][1].content.endswith(JSON_ONLY_NOTE)


def test_a_transport_failure_has_no_answer_and_so_is_not_charged_tokens():
    model = ScriptedModel([ValueError("connection reset"), {"parsed": parsed(), "usage": USAGE}])
    outcome = LangChainExtractor(model, max_attempts=2, method="structured").extract(
        system_prompt="S", user_prompt="U"
    )
    assert outcome.attempts == 2
    assert outcome.usage.calls == 1  # only the call that actually answered
    assert outcome.usage.total_tokens == 1100
    assert "connection reset" in (outcome.error or "")


def test_json_mode_always_used_when_the_method_says_so():
    model = ScriptedModel([{"text": '{"entries": []}'}])
    outcome = LangChainExtractor(model, max_attempts=3, method="json").extract(
        system_prompt="S", user_prompt="U"
    )
    assert model.calls == ["json"]
    assert outcome.method == "json"


def test_a_model_that_cannot_bind_json_mode_is_still_asked():
    model = ScriptedModel([{"text": '{"entries": []}'}], bindable=False)
    outcome = LangChainExtractor(model, max_attempts=1, method="json").extract(
        system_prompt="S", user_prompt="U"
    )
    assert outcome.method == "json"
    assert outcome.entries == []


def test_every_attempt_failing_raises_with_the_last_reason():
    model = ScriptedModel([ValueError("boom"), ValueError("boom again")])
    with pytest.raises(ExtractionError) as excinfo:
        LangChainExtractor(model, max_attempts=2, method="structured").extract(
            system_prompt="S", user_prompt="U"
        )
    assert "2 attempt(s)" in str(excinfo.value)
    assert "boom again" in str(excinfo.value)


def test_an_answer_with_a_missing_field_is_invalid_and_retried():
    """The schema is the gate, even when the provider thought it filled it in."""
    broken = {"entries": [{**ENTRY, "translation": ""}]}
    model = ScriptedModel(
        [
            {"parsed": broken, "text": "{}"},
            {"parsed": parsed(), "text": "{}"},
        ]
    )
    outcome = LangChainExtractor(model, max_attempts=2, method="structured").extract(
        system_prompt="S", user_prompt="U"
    )
    assert outcome.attempts == 2
    assert "ValidationError" in (outcome.error or "")


def test_a_long_raw_answer_is_clipped_for_the_trace():
    model = ScriptedModel([{"parsed": parsed(), "text": "x" * 5000}])
    outcome = LangChainExtractor(model, raw_limit=100).extract(system_prompt="S", user_prompt="U")
    assert len(outcome.raw_text) == 100


def test_a_provider_without_structured_output_raises_and_can_be_retried():
    model = ScriptedModel([{"text": "{}"}], structured=False)
    with pytest.raises(ExtractionError) as excinfo:
        LangChainExtractor(model, max_attempts=1, method="structured").extract(
            system_prompt="S", user_prompt="U"
        )
    assert "NotImplementedError" in str(excinfo.value)


class UnsupportedStructuredModel(ScriptedModel):
    """A model whose schema request the *provider* refuses.

    DeepSeek behaves this way: it advertises a schema-capable response format and
    then answers with an HTTP 400, so the failure is neither a transport error nor
    a bad answer — it is "stop asking for this".
    """

    def __init__(self) -> None:
        super().__init__([{"text": '{"entries": []}'}, {"text": '{"entries": []}'}])

    def with_structured_output(self, schema, include_raw=False):
        return _RefusingRunner(self)


class _RefusingRunner:
    def __init__(self, model: ScriptedModel) -> None:
        self.model = model

    def invoke(self, messages):
        self.model.calls.append("structured")
        self.model.seen.append(list(messages))
        raise ValueError("Error code: 400 - This response_format type is unavailable now")


def test_auto_mode_stops_asking_for_structured_output_after_a_provider_refusal():
    model = UnsupportedStructuredModel()
    extractor = LangChainExtractor(model, max_attempts=2, method="auto")

    first = extractor.extract(system_prompt="S", user_prompt="U1")
    assert model.calls == ["structured", "json"]  # one refusal, then the fallback
    assert first.method == "json"
    assert extractor.json_only is True

    # The latch is what matters: chapter 2 does not pay for a doomed request.
    second = extractor.extract(system_prompt="S", user_prompt="U2")
    assert model.calls == ["structured", "json", "json"]
    assert second.method == "json"
    assert second.attempts == 1


def test_a_transient_failure_does_not_turn_structured_output_off():
    model = ScriptedModel([ValueError("connection reset"), {"parsed": parsed(), "usage": USAGE}])
    extractor = LangChainExtractor(model, max_attempts=3, method="auto")
    outcome = extractor.extract(system_prompt="S", user_prompt="U")
    assert extractor.json_only is False
    # Attempt 2 is still the native path: a dropped connection is not a refusal.
    assert model.calls == ["structured", "structured"]
    assert outcome.method == "structured"


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("This response_format type is unavailable now", True),
        ("json_schema is not supported by this model", True),
        ("tool calling is unavailable", True),
        ("connection reset by peer", False),
        ("504 Gateway Timeout", False),
    ],
)
def test_the_unsupported_provider_detector(message, expected):
    assert structured_unsupported(ValueError(message)) is expected
    assert structured_unsupported(NotImplementedError(message)) is True


# --------------------------------------------------------------------------- #
# reading provider answers
# --------------------------------------------------------------------------- #
def test_usage_reads_usage_metadata_first_and_token_usage_second():
    first = FakeMessage(usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    assert usage_from_message(first).to_dict() == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "calls": 1,
    }

    second = FakeMessage()
    second.response_metadata = {"token_usage": {"prompt_tokens": 7, "completion_tokens": 3}}
    usage = usage_from_message(second)
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (7, 3, 10)

    # A provider that reports nothing still counts as one call.
    silent = usage_from_message(FakeMessage())
    assert silent.calls == 1 and silent.total_tokens == 0
    assert silent.known is False


def test_usage_adds_up_across_calls():
    total = TokenUsage(input_tokens=1, output_tokens=2, total_tokens=3, calls=1)
    total += TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30, calls=2)
    assert total.to_dict() == {
        "input_tokens": 11,
        "output_tokens": 22,
        "total_tokens": 33,
        "calls": 3,
    }


def test_message_text_handles_strings_and_content_blocks():
    assert message_text(FakeMessage("hello")) == "hello"
    assert message_text(FakeMessage("hello", blocks=True)) == "hello"


@pytest.mark.parametrize(
    "text",
    [
        '{"entries": []}',
        '```json\n{"entries": []}\n```',
        'Here is the JSON you asked for:\n{"entries": []}\nHope it helps!',
    ],
)
def test_parse_json_object_recovers_the_object_from_prose_and_fences(text):
    assert parse_json_object(text) == {"entries": []}


def test_parse_json_object_refuses_a_truncated_answer():
    with pytest.raises(ValueError, match="no JSON object"):
        parse_json_object('{"entries": [{"term": "Pot')


def test_make_extractor_names_the_model_and_keeps_the_attempt_cap():
    class Settings:
        provider = "deepseek"
        model = "deepseek-flash"
        base_url = "https://api.deepseek.com/v1"
        api_key = "not-needed"
        temperature = 0.0
        timeout = 30.0
        max_retries = 1

    extractor = make_extractor(Settings(), max_attempts=2, method="json", max_tokens=512)
    assert extractor.name == "deepseek/deepseek-flash"
    assert extractor.max_attempts == 2
    assert extractor.method == "json"
    # The double-free path also holds for the real extractor's outcome type.
    assert ExtractionOutcome(extraction=parsed()).to_dict()["entries"] == 1


def test_an_unknown_method_is_rejected_at_construction():
    with pytest.raises(ValueError, match="unknown method"):
        LangChainExtractor(ScriptedModel([]), method="magic")


def test_a_validated_entry_is_returned_untouched():
    entry = GlossaryEntry.model_validate(ENTRY)
    model = ScriptedModel([{"parsed": ChapterExtraction(entries=[entry]), "text": "{}"}])
    outcome = LangChainExtractor(model).extract(system_prompt="S", user_prompt="U")
    assert outcome.entries[0] is entry
