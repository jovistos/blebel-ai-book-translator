"""Offline tests for stage 7's prompts: what the model is actually told.

The prompt is the stage's product as much as the translation is — it is saved
verbatim for debugging — so what is pinned here is its *content*: the languages,
the JSON shape, the binding glossary block, the source chunk last, and the
continuity block only when there is context to send.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from book_translator.stages.translation.prompts import (
    CONTEXT_HEADER,
    GLOSSARY_HEADER,
    SOURCE_HEADER,
    TASK_HEADER,
    build_chunk_prompt,
    build_repair_note,
    build_system_prompt,
    render_glossary,
    target_style_note,
)
from book_translator.stages.translation.runner import GlossaryRow


def _prompt(**overrides) -> str:
    values: dict[str, Any] = {
        "chunk_index": 3,
        "total_chunks": 183,
        "chunk_filename": "00042-08-chapter-1-0003.md",
        "chapter_filename": "08-chapter-1.md",
        "source_text": "George opened the door and said, 'Hello.'",
        "source_language": "en",
        "target_language": "sr",
        "terms": [GlossaryRow("George", "Džordž", "character")],
        "book_title": "A Book",
    }
    values.update(overrides)
    return build_chunk_prompt(**values)


# --------------------------------------------------------------------------- #
# the system prompt
# --------------------------------------------------------------------------- #
def test_the_system_prompt_names_both_languages_and_the_book():
    prompt = build_system_prompt("en", "sr", 'The "Book"')
    assert "from English into Serbian" in prompt
    assert '"Book"' in prompt
    assert "translator of one book" in prompt


def test_the_system_prompt_asks_for_the_structured_answer():
    prompt = build_system_prompt("en", "sr")
    assert '"translation"' in prompt
    assert "JSON object" in prompt
    assert "translation` is the only field" in prompt


def test_the_system_prompt_states_the_rules_that_keep_chunks_seamless():
    prompt = build_system_prompt("en", "sr")
    assert "paragraph" in prompt
    assert "Markdown" in prompt
    assert "binding, not advisory" in prompt
    assert "may begin or end in the middle of a sentence" in prompt
    assert "Do not summarise" in prompt


def test_serbian_gets_ekavian_latin_and_case_instructions():
    prompt = build_system_prompt("en", "sr")
    assert "ekavian" in prompt
    assert "Latin alphabet" in prompt
    note = target_style_note("sr")
    assert "decline" in note


def test_an_unknown_target_language_falls_back_to_a_generic_note():
    note = target_style_note("xx")
    assert "xx" in note
    assert "natural" in note


def test_build_system_prompt_handles_no_title():
    prompt = build_system_prompt("en", "sr")
    assert "the book you are given" in prompt


# --------------------------------------------------------------------------- #
# the glossary block
# --------------------------------------------------------------------------- #
def test_render_glossary_is_one_pair_per_line():
    rows = [
        GlossaryRow("George", "Džordž", "character"),
        {"term": "London", "translation": "London", "category": "location"},
    ]
    rendered = render_glossary(rows)
    assert rendered.splitlines() == [
        "- George → Džordž (character)",
        "- London → London (location)",
    ]


def test_render_glossary_skips_incomplete_rows():
    rows = [
        GlossaryRow("", "Džordž"),
        GlossaryRow("George", ""),
        GlossaryRow("Jane", "Dženi", "character"),
    ]
    assert render_glossary(rows) == "- Jane → Dženi (character)"


def test_an_empty_glossary_says_so_instead_of_leaving_a_blank_block():
    rendered = render_glossary([], target_language="sr")
    assert "no entity from this chunk is in the book glossary" in rendered
    assert "Serbian" in rendered


@dataclass
class _Entry:
    term: str
    translation: str
    category: str = ""


def test_render_glossary_reads_dataclasses_too():
    assert render_glossary([_Entry("Mithril", "Mitril")]) == "- Mithril → Mitril"


# --------------------------------------------------------------------------- #
# the chunk prompt
# --------------------------------------------------------------------------- #
def test_the_chunk_prompt_carries_the_glossary_the_source_and_the_task():
    prompt = _prompt()
    assert GLOSSARY_HEADER in prompt
    assert "- George → Džordž (character)" in prompt
    assert SOURCE_HEADER in prompt
    assert "George opened the door" in prompt
    assert TASK_HEADER in prompt
    assert 'Chunk 3 of 183' in prompt
    assert "08-chapter-1.md" in prompt


def test_the_source_text_is_the_last_thing_before_the_task():
    """The text to act on must come after the decisions that constrain it."""
    prompt = _prompt()
    assert prompt.index(GLOSSARY_HEADER) < prompt.index(SOURCE_HEADER) < prompt.index(TASK_HEADER)


def test_no_context_block_when_context_is_disabled():
    prompt = _prompt(context_text="")
    assert CONTEXT_HEADER not in prompt
    assert "previous chunk" not in prompt.lower()


def test_context_is_sent_with_an_explicit_do_not_translate_again_note():
    prompt = _prompt(context_text="…and he closed the door.")
    assert CONTEXT_HEADER in prompt
    assert "Do not translate it again" in prompt
    assert "…and he closed the door." in prompt
    assert prompt.index(CONTEXT_HEADER) < prompt.index(SOURCE_HEADER)


def test_an_empty_glossary_keeps_the_block_and_explains_it():
    prompt = _prompt(terms=[])
    assert GLOSSARY_HEADER in prompt
    assert "no entity from this chunk is in the book glossary" in prompt


def test_a_missing_chapter_name_is_labelled():
    prompt = _prompt(chapter_filename="")
    assert "(chapter unknown)" in prompt


def test_the_prompt_asks_for_the_json_object_and_nothing_else():
    prompt = _prompt()
    assert '{"translation": "..."}' in prompt
    assert "outside the JSON object" in prompt


# --------------------------------------------------------------------------- #
# the repair note
# --------------------------------------------------------------------------- #
def test_the_repair_note_quotes_the_failure_and_repeats_the_shape():
    note = build_repair_note("structured output not filled: translation is required")
    assert "structured output not filled" in note
    assert '"translation"' in note
    assert "return only the json object" in note.lower()


def test_the_repair_note_has_a_default_reason():
    assert "could not be used" in build_repair_note("")


def test_a_very_long_failure_is_clipped():
    note = build_repair_note("x" * 2000)
    assert "x" * 401 not in note


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
