"""The marker-naming decision: the model reads it, deterministic code checks it.

No network: a scripted model double stands in for the provider, exactly as the
graph tests do.
"""

from __future__ import annotations

import json

from book_splitter.analysis.renumber import (
    ambiguous_markers,
    build_prompt,
    parse_reply,
    resolve_marker_names,
    validate_corrections,
)
from langchain_core.messages import AIMessage

#: The real shape from `Lust for Life`: clean digits around mangled markers.
SEQUENCES = {
    "BOOK ONE": ["7", "8", "'3", "14", "15", "16", "19"],
    "BOOK FIVE": ["I", "4", "5", "7", "lO", "II", "12", "13"],
}


class _ScriptedModel:
    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[str] = []

    def invoke(self, messages, *_, **__):
        self.prompts.append(messages if isinstance(messages, str) else str(messages))
        return AIMessage(content=self.reply)


def _aliases(entries):
    return {(e["book"], e["as_written"]): e["number"] for e in entries}


def test_only_non_numeric_markers_are_ambiguous():
    assert ambiguous_markers(SEQUENCES["BOOK ONE"]) == ["'3"]
    assert ambiguous_markers(SEQUENCES["BOOK FIVE"]) == ["I", "lO", "II"]


def test_the_prompt_lists_every_section_in_file_order():
    prompt = build_prompt({"BOOK ONE": ["7", "8", "'3"], "BOOK FIVE": ["lO", "12"]})
    assert "- BOOK ONE: `7`, `8`, `'3`" in prompt
    assert "- BOOK FIVE: `lO`, `12`" in prompt
    assert "JSON only" in prompt


def test_parse_reply_tolerates_fences_and_prose():
    body = json.dumps({"chapters": [{"book": "BOOK ONE", "as_written": "'3", "number": 13}]})
    assert parse_reply(f"```json\n{body}\n```")[0]["number"] == 13
    assert parse_reply(f"Sure, here you go:\n{body}\nHope that helps!")[0]["number"] == 13
    assert parse_reply("no json at all") == []


def test_a_good_answer_is_accepted():
    entries = [
        {"book": "BOOK ONE", "as_written": "'3", "number": 13},
        {"book": "BOOK FIVE", "as_written": "I", "number": 1},
        {"book": "BOOK FIVE", "as_written": "lO", "number": 10},
        {"book": "BOOK FIVE", "as_written": "II", "number": 11},
    ]
    aliases, problems = validate_corrections(SEQUENCES, entries)
    assert problems == []
    assert aliases[("BOOK FIVE", "lO")] == "10"
    assert aliases[("BOOK ONE", "'3")] == "13"


def test_a_marker_the_model_cannot_read_is_left_alone():
    """`null` means 'not a number': the marker keeps its text, nothing is invented."""
    entries = [{"book": "BOOK FIVE", "as_written": "lO", "number": None}]
    aliases, problems = validate_corrections(SEQUENCES, entries)
    assert aliases == {} and problems == []


def test_a_non_increasing_reading_is_rejected_whole():
    entries = [
        {"book": "BOOK FIVE", "as_written": "lO", "number": 2},
        {"book": "BOOK FIVE", "as_written": "II", "number": 11},
    ]
    aliases, problems = validate_corrections(SEQUENCES, entries)
    assert aliases == {}
    assert any("not increasing" in problem for problem in problems)


def test_a_duplicate_number_is_rejected_whole():
    entries = [
        {"book": "BOOK ONE", "as_written": "'3", "number": 14},  # 14 already exists
    ]
    aliases, problems = validate_corrections(SEQUENCES, entries)
    assert aliases == {}
    assert any("duplicate" in problem for problem in problems)


def test_out_of_range_and_non_integer_answers_are_rejected():
    entries = [
        {"book": "BOOK ONE", "as_written": "'3", "number": 9999},
        {"book": "BOOK FIVE", "as_written": "lO", "number": "ten"},
    ]
    aliases, problems = validate_corrections(SEQUENCES, entries)
    assert aliases == {}
    assert any("out of range" in problem for problem in problems)
    assert any("not an integer" in problem for problem in problems)


def test_an_invented_marker_or_book_is_ignored():
    entries = [
        {"book": "BOOK NINE", "as_written": "lO", "number": 10},
        {"book": "BOOK FIVE", "as_written": "42", "number": 42},
    ]
    aliases, problems = validate_corrections(SEQUENCES, entries)
    assert aliases == {} and problems == []


def test_resolve_marker_names_uses_the_model_and_validates():
    reply = json.dumps(
        {
            "chapters": [
                {"book": "BOOK ONE", "as_written": "'3", "number": 13},
                {"book": "BOOK FIVE", "as_written": "I", "number": 1},
                {"book": "BOOK FIVE", "as_written": "lO", "number": 10},
                {"book": "BOOK FIVE", "as_written": "II", "number": 11},
            ]
        }
    )
    model = _ScriptedModel(reply)
    aliases, problems = resolve_marker_names(SEQUENCES, model=model)
    assert problems == []
    assert aliases == {
        ("BOOK ONE", "'3"): "13",
        ("BOOK FIVE", "I"): "1",
        ("BOOK FIVE", "lO"): "10",
        ("BOOK FIVE", "II"): "11",
    }
    assert "BOOK FIVE" in model.prompts[0]


def test_resolve_marker_names_skips_the_model_when_nothing_is_ambiguous():
    model = _ScriptedModel("{}")
    aliases, problems = resolve_marker_names({"BOOK ONE": ["1", "2"]}, model=model)
    assert (aliases, problems) == ({}, [])
    assert model.prompts == [], "a clean sequence must not cost a call"
