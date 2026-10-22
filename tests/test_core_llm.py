"""Offline tests for the shared structured-call loop.

No provider, no network: the chat model is a scripted double, exactly as in the
stage tests. What is pinned here is the generic contract the glossary and the
translation stage both rely on — the schema, the validation hook, the recovery
hook, the JSON fallback and the token count — without either stage in the way.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from book_splitter.core.llm import (
    StructuredCaller,
    TokenUsage,
    build_chat_model,
    message_text,
    model_label,
    parse_json_object,
    structured_unsupported,
    usage_from_message,
)
from pydantic import BaseModel


class Answer(BaseModel):
    text: str


class Message:
    def __init__(self, text: str = "", usage: dict | None = None) -> None:
        self.content = text
        self.usage_metadata = usage or {}
        self.response_metadata: dict[str, Any] = {}


class ScriptedModel:
    """A chat-model double whose answers are scripted in order."""

    def __init__(self, answers: list[Any], *, structured: bool = True) -> None:
        self._answers = list(answers)
        self._structured = structured
        self.calls: list[str] = []

    def with_structured_output(self, schema, include_raw: bool = False):
        assert include_raw is True
        if not self._structured:
            raise NotImplementedError("no structured output")
        return _StructuredRunner(self)

    def bind(self, **_kwargs):
        return self

    def invoke(self, _messages):
        self.calls.append("json")
        answer = self._next()
        if isinstance(answer, Exception):
            raise answer
        return Message(answer.get("text", ""), answer.get("usage"))

    def _next(self) -> dict:
        if not self._answers:
            raise AssertionError("the double ran out of scripted answers")
        return self._answers.pop(0)


class _StructuredRunner:
    def __init__(self, model: ScriptedModel) -> None:
        self.model = model

    def invoke(self, _messages):
        self.model.calls.append("structured")
        answer = self.model._next()
        if isinstance(answer, Exception):
            raise answer
        raw = Message(answer.get("text", ""), answer.get("usage"))
        return {
            "raw": raw,
            "parsed": answer.get("parsed"),
            "parsing_error": answer.get("parsing_error"),
        }


USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


def _caller(model: ScriptedModel, **overrides) -> StructuredCaller[Answer]:
    options = {
        "max_attempts": 2,
        "method": "auto",
        "json_note": "\nJSON ONLY",
        "repair_note": lambda error: f"\nREPAIR: {error}",
        "json_on_last_attempt": False,
    }
    options.update(overrides)
    return StructuredCaller(model, Answer, **options)


def test_a_structured_answer_is_validated_and_priced():
    model = ScriptedModel([{"parsed": Answer(text="hello"), "text": "raw", "usage": USAGE}])
    outcome = _caller(model).call(system_prompt="s", user_prompt="u")

    assert model.calls == ["structured"]
    assert outcome.value.text == "hello"
    assert outcome.attempts == 1
    assert outcome.method == "structured"
    assert outcome.raw_text == "raw"
    assert outcome.usage.to_dict() == {**USAGE, "calls": 1}


def test_a_validation_hook_can_reject_an_answer_the_schema_accepted():
    model = ScriptedModel(
        [
            {"parsed": Answer(text="no"), "text": "a", "usage": USAGE},
            {"parsed": Answer(text="the whole translation"), "text": "b", "usage": USAGE},
        ]
    )
    caller = _caller(
        model,
        method="structured",
        validator=lambda answer, _context: "too short" if len(answer.text) < 5 else None,
    )
    outcome = caller.call(system_prompt="s", user_prompt="u", validation_context="source")

    assert outcome.attempts == 2
    assert outcome.value.text == "the whole translation"
    assert "too short" in (outcome.error or "")
    assert outcome.usage.calls == 2


def test_a_recovery_hook_salvages_complete_but_malformed_json():
    model = ScriptedModel([{"text": '{"text": "broken', "usage": USAGE}])
    caller = _caller(
        model,
        method="json",
        recovery=lambda _text: Answer(text="recovered"),
    )
    outcome = caller.call(system_prompt="s", user_prompt="u")

    assert outcome.recovered is True
    assert outcome.value.text == "recovered"
    assert model.calls == ["json"]


def test_without_a_recovery_hook_the_malformed_answer_is_retried():
    model = ScriptedModel(
        [
            {"text": '{"text": "broken', "usage": USAGE},
            {"text": '{"text": "fixed"}', "usage": USAGE},
        ]
    )
    outcome = _caller(model, method="json").call(system_prompt="s", user_prompt="u")
    assert outcome.attempts == 2
    assert outcome.value.text == "fixed"


def test_json_on_last_attempt_switches_only_on_the_final_try():
    model = ScriptedModel(
        [
            {"parsed": None, "parsing_error": "bad", "text": "x", "usage": USAGE},
            {"text": '{"text": "from json"}', "usage": USAGE},
        ]
    )
    outcome = _caller(model, json_on_last_attempt=True).call(
        system_prompt="s", user_prompt="u"
    )
    assert model.calls == ["structured", "json"]
    assert outcome.method == "json"
    assert outcome.value.text == "from json"


def test_a_provider_refusal_latches_json_mode_and_does_not_charge_the_call():
    model = ScriptedModel(
        [ValueError("This response_format type is unavailable now"), {"text": '{"text": "ok"}'}]
    )
    caller = _caller(model, max_attempts=2)
    outcome = caller.call(system_prompt="s", user_prompt="u")

    assert caller.json_only is True
    assert model.calls == ["structured", "json"]
    assert outcome.value.text == "ok"
    assert outcome.usage.calls == 1  # the refusal produced no answer to charge


def test_the_failure_factory_receives_the_usage_of_every_attempt():
    model = ScriptedModel([{"parsed": None, "text": "x", "usage": USAGE}] * 2)
    seen: list[TokenUsage] = []

    def failure(usage: TokenUsage, attempts: int, last: str | None) -> RuntimeError:
        seen.append(usage)
        return RuntimeError(f"{attempts}: {last}")

    caller = _caller(model, method="structured", failure=failure)
    with pytest.raises(RuntimeError, match="2: structured output not filled"):
        caller.call(system_prompt="s", user_prompt="u")
    assert seen[0].calls == 2
    assert seen[0].total_tokens == 30


def test_an_unknown_method_is_rejected_at_construction():
    with pytest.raises(ValueError, match="unknown method"):
        StructuredCaller(ScriptedModel([]), Answer, method="magic")


def test_build_chat_model_reads_one_settings_shape(monkeypatch):
    seen: dict[str, Any] = {}

    class FakeChat:
        def __init__(self, **kwargs) -> None:
            seen.update(kwargs)

    monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChat)
    settings = SimpleNamespace(
        model="m",
        base_url="http://local",
        api_key="key",
        temperature=0.1,
        timeout=9.0,
        max_retries=2,
        provider="provider",
    )
    build_chat_model(settings, max_tokens=128)

    assert seen == {
        "model": "m",
        "base_url": "http://local",
        "api_key": "key",
        "temperature": 0.1,
        "timeout": 9.0,
        "max_retries": 2,
        "max_tokens": 128,
    }
    assert model_label(settings) == "provider/m"


def test_the_provider_reading_primitives():
    assert message_text(Message("hello")) == "hello"
    assert usage_from_message(Message(usage=USAGE)).to_dict() == {**USAGE, "calls": 1}
    assert parse_json_object('Here:\n```json\n{"text": "x"}\n```') == {"text": "x"}
    assert structured_unsupported(NotImplementedError()) is True
    assert structured_unsupported(ValueError("connection reset")) is False


def test_token_usage_adds_and_reports_known():
    total = TokenUsage(input_tokens=1, output_tokens=2, total_tokens=3, calls=1)
    total += TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30, calls=2)
    assert total.to_dict() == {
        "input_tokens": 11,
        "output_tokens": 22,
        "total_tokens": 33,
        "calls": 3,
    }
    assert total.known is True
    assert TokenUsage(calls=1).known is False
