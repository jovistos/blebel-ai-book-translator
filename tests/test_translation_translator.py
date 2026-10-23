"""Offline tests for stage 7's model loop: structured output, retries, tokens.

No provider and no network: the chat model is a scripted double. What is pinned
is the contract the runner relies on — the answer is validated, a failure is
retried with the failure named, every attempt is counted, and an answer that is
empty, unchanged from the source or a summary of it never reaches the book.
"""

from __future__ import annotations

from typing import Any

import pytest
from book_translator.core.errors import TranslationError
from book_translator.stages.translation.models import ChunkTranslationAnswer
from book_translator.stages.translation.translator import (
    LangChainTranslator,
    TranslationOutcome,
    clean_translation,
    recover_translation,
    translation_problem,
)


# --------------------------------------------------------------------------- #
# doubles
# --------------------------------------------------------------------------- #
class Message:
    """The parts of a langchain message the usage reader touches."""

    def __init__(self, content: str = "", usage: dict | None = None) -> None:
        self.content = content
        self.usage_metadata = usage or {}
        self.response_metadata: dict[str, Any] = {}


class StructuredLLM:
    """A scripted ``with_structured_output`` endpoint."""

    def __init__(self, answers: list[Any], *, error: Exception | None = None) -> None:
        self.answers = list(answers)
        self.error = error
        self.calls: list[list[Any]] = []
        self.bound: dict[str, Any] | None = None
        self.schemas: list[Any] = []

    def with_structured_output(self, schema, include_raw: bool = False):
        self.schemas.append(schema)
        assert include_raw is True
        return self

    def bind(self, **kwargs):
        self.bound = kwargs
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error is not None:
            # Raise the scripted provider error once, then answer normally: this
            # is how a provider that rejects the schema and accepts JSON behaves.
            error, self.error = self.error, None
            raise error
        return self.answers.pop(0)


class JsonLLM(StructuredLLM):
    """A JSON-mode endpoint: ``invoke`` returns the raw chat message."""

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error is not None:
            error, self.error = self.error, None
            raise error
        return self.answers.pop(0)


USAGE = {"input_tokens": 100, "output_tokens": 40, "total_tokens": 140}


def answer(text: str) -> dict[str, Any]:
    return {
        "parsed": ChunkTranslationAnswer(translation=text),
        "raw": Message(text, usage=USAGE),
    }


def failure(reason: str = "translation is required") -> dict[str, Any]:
    return {"parsed": None, "parsing_error": reason, "raw": Message("", usage=USAGE)}


# --------------------------------------------------------------------------- #
# the happy path
# --------------------------------------------------------------------------- #
def test_one_structured_call_returns_the_validated_translation():
    llm = StructuredLLM([answer("Džordž je otvorio vrata.")])
    translator = LangChainTranslator(llm, max_attempts=3)

    outcome = translator.translate(
        system_prompt="system", user_prompt="user", source_text="George opened the door."
    )

    assert isinstance(outcome, TranslationOutcome)
    assert outcome.text == "Džordž je otvorio vrata."
    assert outcome.attempts == 1
    assert outcome.method == "structured"
    assert outcome.error is None
    assert outcome.usage.total_tokens == 140
    assert outcome.raw_text == "Džordž je otvorio vrata."
    assert len(llm.calls) == 1


def test_the_native_structured_schema_is_the_pydantic_model():
    llm = StructuredLLM([answer("x")])
    LangChainTranslator(llm).translate(system_prompt="s", user_prompt="u", source_text="y")
    assert llm.schemas == [ChunkTranslationAnswer]


def test_the_system_and_user_prompts_are_sent_as_two_messages():
    llm = StructuredLLM([answer("x")])
    LangChainTranslator(llm).translate(
        system_prompt="the system", user_prompt="the user", source_text="source"
    )
    messages = llm.calls[0]
    assert len(messages) == 2
    assert messages[0].content == "the system"
    assert messages[1].content == "the user"


# --------------------------------------------------------------------------- #
# retries
# --------------------------------------------------------------------------- #
def test_a_schema_failure_is_retried_with_the_reason_and_both_calls_are_priced():
    llm = StructuredLLM([failure("translation: Field required"), answer("Džordž.")])
    translator = LangChainTranslator(llm, max_attempts=3)

    outcome = translator.translate(
        system_prompt="s", user_prompt="u", source_text="George."
    )

    assert outcome.attempts == 2
    assert outcome.usage.calls == 2
    assert outcome.usage.total_tokens == 280
    assert "Field required" in (outcome.error or "")
    assert "Your previous answer was rejected" in llm.calls[1][1].content
    assert '"translation"' in llm.calls[1][1].content


def test_an_empty_translation_is_retried():
    llm = StructuredLLM([failure("Value error, the translation is empty"), answer("Džordž.")])
    outcome = LangChainTranslator(llm, max_attempts=2).translate(
        system_prompt="s", user_prompt="u", source_text="George."
    )
    assert outcome.attempts == 2
    assert "empty" in (outcome.error or "")


def test_an_unchanged_source_is_retried():
    source = "George opened the door and walked into the garden, smiling."
    llm = StructuredLLM([answer(source), answer("Džordž je otvorio vrata.")])
    outcome = LangChainTranslator(llm, max_attempts=2).translate(
        system_prompt="s", user_prompt="u", source_text=source
    )
    assert outcome.attempts == 2
    assert "unchanged" in (outcome.error or "")


def test_a_summary_is_treated_as_a_failure():
    source = "word " * 200
    llm = StructuredLLM([answer("word"), answer("reči " * 200)])
    outcome = LangChainTranslator(llm, max_attempts=2).translate(
        system_prompt="s", user_prompt="u", source_text=source
    )
    assert outcome.attempts == 2
    assert "short" in (outcome.error or "")


def test_every_attempt_failing_raises_with_the_usage_still_attached():
    llm = StructuredLLM([failure(), failure(), failure()])
    translator = LangChainTranslator(llm, max_attempts=3)

    with pytest.raises(TranslationError) as info:
        translator.translate(system_prompt="s", user_prompt="u", source_text="George.")

    assert "3 attempt" in str(info.value)
    assert info.value.usage.total_tokens == 420
    assert info.value.attempts == 3
    assert len(llm.calls) == 3


# --------------------------------------------------------------------------- #
# json mode
# --------------------------------------------------------------------------- #
def test_json_mode_parses_the_object_and_appends_the_json_only_note():
    llm = JsonLLM([Message('{"translation": "Džordž."}', usage=USAGE)])
    translator = LangChainTranslator(llm, method="json")

    outcome = translator.translate(system_prompt="s", user_prompt="u", source_text="George.")

    assert outcome.method == "json"
    assert outcome.text == "Džordž."
    assert "Return ONLY the JSON object" in llm.calls[0][1].content


def test_json_mode_retries_a_malformed_answer():
    llm = JsonLLM(
        [
            Message("Here is the translation: Džordž.", usage=USAGE),
            Message('{"translation": "Džordž."}', usage=USAGE),
        ]
    )
    outcome = LangChainTranslator(llm, method="json", max_attempts=2).translate(
        system_prompt="s", user_prompt="u", source_text="George."
    )
    assert outcome.attempts == 2
    assert "ValueError" in (outcome.error or "")


# --------------------------------------------------------------------------- #
# recovering an unescaped quote (JSON mode's recurring malformation)
# --------------------------------------------------------------------------- #
#: The exact shape that failed a real run: the model typed the closing Serbian
#: quotation mark as an ASCII ``"`` inside the JSON string.
BROKEN_ANSWER = (
    '{"translation": "„Dobro, da", kimne on, sumnjičavo gledajući po praznoj '
    'kancelariji.\\n\\n„U stvari, nisam prestala celo popodne", kažem nadmeno."}'
)


def test_json_mode_recovers_an_unescaped_quote_in_the_translation():
    llm = JsonLLM([Message(BROKEN_ANSWER, usage=USAGE)])
    translator = LangChainTranslator(llm, method="json")

    outcome = translator.translate(
        system_prompt="s", user_prompt="u", source_text="George opened the door."
    )

    assert outcome.attempts == 1
    assert outcome.usage.calls == 1
    assert outcome.recovered is True
    assert '„Dobro, da", kimne on' in outcome.text
    assert outcome.text.endswith('„U stvari, nisam prestala celo popodne", kažem nadmeno.')


def test_a_strictly_valid_answer_is_never_marked_recovered():
    llm = JsonLLM([Message('{"translation": "Džordž je otvorio vrata."}', usage=USAGE)])
    outcome = LangChainTranslator(llm, method="json").translate(
        system_prompt="s", user_prompt="u", source_text="George opened the door."
    )
    assert outcome.recovered is False


def test_recover_translation_decodes_json_escapes_around_the_raw_quote():
    recovered = recover_translation(
        '{"translation": "Prva linija.\\n\\nReče \\"zdravo\\" i ode."}'
    )
    assert recovered == 'Prva linija.\n\nReče "zdravo" i ode.'


def test_recover_translation_rejects_a_truncated_answer():
    # No closing quote and no closing brace: the model was cut off mid-chunk.
    assert recover_translation('{"translation": "Prva linija života') is None


def test_recover_translation_stops_at_a_second_field():
    # The quote after "Džordž" is followed by another key, so it closes the value.
    assert recover_translation('{"translation": "Džordž", "notes": "x"}') == "Džordž"


def test_json_mode_still_fails_and_retries_when_nothing_can_be_recovered():
    llm = JsonLLM(
        [
            Message('{"translation": "Prva linija', usage=USAGE),
            Message('{"translation": "Džordž."}', usage=USAGE),
        ]
    )
    outcome = LangChainTranslator(llm, method="json", max_attempts=2).translate(
        system_prompt="s", user_prompt="u", source_text="George."
    )
    assert outcome.attempts == 2
    assert outcome.recovered is False
    assert "the answer holds no JSON object" in (outcome.error or "")


def test_a_provider_without_structured_output_falls_back_for_the_rest_of_the_run():
    llm = StructuredLLM([], error=ValueError("This response_format type is unavailable now"))
    translator = LangChainTranslator(llm, method="auto", max_attempts=2)

    # First attempt: the provider rejects the schema; the translator switches to
    # JSON mode for the last attempt (which the same double now answers).
    llm.answers = [Message('{"translation": "Džordž."}', usage=USAGE)]

    outcome = translator.translate(system_prompt="s", user_prompt="u", source_text="George.")

    assert outcome.method == "json"
    assert translator.json_only is True


# --------------------------------------------------------------------------- #
# cleaning and validation
# --------------------------------------------------------------------------- #
def test_clean_translation_unwraps_a_whole_answer_code_fence():
    assert clean_translation("```markdown\nDžordž.\n```") == "Džordž."


def test_clean_translation_keeps_a_fence_that_is_part_of_the_text():
    text = "Text before.\n\n```python\nprint('x')\n```"
    assert clean_translation(text) == text


def test_clean_translation_drops_a_preamble_and_a_closing_offer():
    text = "Here is the translation:\n\nDžordž je ušao.\n\nLet me know if you need anything else."
    assert clean_translation(text) == "Džordž je ušao."


def test_clean_translation_removes_surrounding_blank_lines_but_keeps_indentation():
    assert clean_translation("\n\n  - jedna\n  - dve\n\n") == "  - jedna\n  - dve"


def test_translation_problem_accepts_a_short_unchanged_source():
    assert translation_problem("42", "42") is None


def test_translation_problem_rejects_an_empty_answer():
    assert "empty" in (translation_problem("   ") or "")


def test_translation_problem_rejects_an_unchanged_paragraph():
    source = "George opened the door and walked into the garden, smiling at everyone."
    assert "unchanged" in (translation_problem(source, source) or "")


def test_translation_problem_ignores_whitespace_differences_when_comparing():
    source = "word " * 30
    assert translation_problem(source.replace(" ", "\n"), source) is not None


def test_translation_problem_rejects_a_dropped_image_marker():
    source = "Alice looked. [[IMG_1]] Then she ran."

    assert translation_problem("Alisa je pogledala. [[IMG_1]] Zatim je potrčala.", source) is None

    dropped = translation_problem("Alisa je pogledala. Zatim je potrčala.", source)
    assert dropped is not None and "dropped" in dropped
    assert "[[IMG_1]]" in dropped

    doubled = translation_problem(
        "Alisa je pogledala. [[IMG_1]] [[IMG_1]] Zatim je potrčala.", source
    )
    assert doubled is not None and "repeated" in doubled

    invented = translation_problem(
        "Alisa je pogledala. [[IMG_1]] [[IMG_2]] Zatim je potrčala.", source
    )
    assert invented is not None and "invented" in invented


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
