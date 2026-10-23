"""Offline tests for the glossary prompts: what the model is told, and when.

Two promises are pinned here. The first is framing: the model must know it is not
summarising a book but fixing the terminology of a book **that is being
translated**, in a named language pair, because that is what makes it choose one
rendering per entity instead of inventing a plausible one. The second is the
incremental contract: every chapter after the first must actually carry the
dictionary so far, and the chapter evidence must be the last thing in the prompt.
"""

from __future__ import annotations

from book_translator.glossary.models import GlossaryEntry
from book_translator.glossary.prompts import (
    CHAPTER_HEADER,
    DICTIONARY_HEADER,
    build_chapter_prompt,
    build_repair_note,
    build_system_prompt,
    language_name,
    render_dictionary,
    target_note,
)


def entry(term: str, translation: str, category: str = "character", context: str = ""):
    """A stored entry. ``context`` is accepted only because older callers pass it."""
    return GlossaryEntry.model_validate(
        {"term": term, "category": category, "translation": translation}
    )


def test_the_system_prompt_says_the_book_is_being_translated():
    prompt = build_system_prompt("en", "sr", "Don't You Forget About Me")
    assert "Don't You Forget About Me" in prompt
    assert "being translated from English into Serbian" in prompt
    assert "chapter by chapter" in prompt
    # The dictionary is binding, not advisory.
    assert "reuse your Serbian rendering verbatim" in prompt
    assert "never calls the same person, place, group or thing by two different names" in prompt
    # The five categories and the four fields are both spelled out.
    for category in ("character", "location", "organization", "artifact", "terminology"):
        assert f"`{category}`" in prompt
    for field in ("`term`", "`category`", "`translation`"):
        assert field in prompt
    # No fourth field: the entry is the pair and its category, nothing else.
    assert "brief_context" not in prompt
    assert "Three fields, nothing else" in prompt
    # The shape of the answer is given literally.
    assert '"entries"' in prompt


def test_an_inflected_target_gets_its_own_rules():
    serbian = build_system_prompt("en", "sr")
    assert "nominative singular" in serbian
    assert "highly inflected" in serbian
    assert "decline it through all seven cases" in serbian
    # A pair with no note still gets the generic citation-form rule.
    assert "nominative singular" not in build_system_prompt("de", "fr")
    assert "citation form" in target_note("fr")
    assert "citation form" in build_system_prompt("de", "fr")


def test_language_names_resolve_and_unknown_codes_survive():
    assert language_name("sr") == "Serbian"
    assert language_name("EN") == "English"
    assert language_name("xx") == "xx"
    assert language_name("") == "the source language"


def test_the_first_chapter_is_told_the_dictionary_is_empty():
    prompt = build_chapter_prompt(
        chapter_index=1,
        chapter_filename="01-chapter-1.md",
        sentences=["Alexandra Potter lives in London."],
        dictionary=[],
    )
    assert DICTIONARY_HEADER in prompt
    assert "(empty — no entity has been locked in yet)" in prompt
    assert CHAPTER_HEADER in prompt
    assert "Alexandra Potter lives in London." in prompt
    assert "Chapter 1 (01-chapter-1.md)" in prompt


def test_every_later_chapter_carries_the_dictionary_it_was_built_with():
    dictionary = [entry("Potter", "Poter", "character"), entry("London", "London", "location")]
    prompt = build_chapter_prompt(
        chapter_index=7,
        chapter_filename="07-chapter-7.md",
        sentences=["Phoebe met Potter in Brighton."],
        dictionary=dictionary,
    )
    for rendered in ("Potter → Poter", "London → London"):
        assert rendered in prompt
    # Only the locked pair is re-sent: every extra field would be paid for once
    # per chapter of the book.
    assert "[character]" not in prompt
    # The evidence is the dictionary, then the chapter — in that order.
    assert prompt.index(DICTIONARY_HEADER) < prompt.index(CHAPTER_HEADER)
    assert prompt.index("Potter → Poter") < prompt.index("Phoebe met Potter in Brighton.")
    # The command that makes the incremental pass safe.
    assert "not already locked in above" in prompt


def test_a_chapter_title_and_a_split_chapter_are_labelled():
    prompt = build_chapter_prompt(
        chapter_index=4,
        chapter_filename="04-glavni-lik.md",
        chapter_title="Glavni lik",
        sentences=["Marko ulazi."],
        dictionary=[],
        part=2,
        parts=3,
    )
    assert "Chapter 4 — Glavni lik (04-glavni-lik.md) (part 2 of 3)" in prompt


def test_the_rendered_dictionary_is_one_pair_per_line_and_nothing_else():
    block = render_dictionary(
        [entry("Potter", "Poter", "character"), entry("London", "London", "location")]
    )
    assert block.splitlines() == ["- Potter → Poter", "- London → London"]


def test_a_capped_dictionary_still_names_the_entries_it_hides():
    dictionary = [entry(f"T{i}", f"P{i}") for i in range(5)]
    block = render_dictionary(dictionary, max_entries=2)
    assert "T0 → P0" in block
    assert "T1 → P1" in block
    assert "3 further entries are locked in" in block
    # The hidden ones are named so the model does not re-invent them.
    for index in (2, 3, 4):
        assert f"T{index} → P{index}" in block


def test_the_repair_note_repeats_the_failure_and_the_json_only_rule():
    note = build_repair_note("ValidationError: term\nempty")
    assert "ValidationError: term empty" in note  # the failure is flattened to one line
    assert "no Markdown code fence" in note
    assert build_repair_note("") .startswith("\n\nYour previous answer was rejected")
