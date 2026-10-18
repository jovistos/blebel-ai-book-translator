"""Offline tests for the stage-4 data shape: what a model is allowed to answer.

The glossary is only as trustworthy as its validation: a field that can arrive
empty, or a category that is silently dropped, becomes a missing rendering in the
translated book. These tests pin the contract ``LangChainExtractor`` relies on —
an object-rooted answer, five known categories, non-empty text fields, and a bare
array accepted as the same thing.
"""

from __future__ import annotations

import pytest
from book_splitter.glossary.models import (
    ChapterExtraction,
    GlossaryEntry,
    TermCategory,
    entry_key,
    normalize_category,
)
from pydantic import ValidationError


def test_entry_is_the_three_fields_the_prompt_asks_for():
    entry = GlossaryEntry.model_validate(
        {
            "term": "Alexandra Potter",
            "category": "character",
            "translation": "Aleksandra Poter",
        }
    )
    assert entry.term == "Alexandra Potter"
    assert entry.category is TermCategory.CHARACTER
    assert entry.translation == "Aleksandra Poter"
    assert entry.key == "alexandra potter"
    assert entry.to_dict() == {
        "term": "Alexandra Potter",
        "category": "character",
        "translation": "Aleksandra Poter",
    }


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("character", TermCategory.CHARACTER),
        ("Characters & Aliases", TermCategory.CHARACTER),
        ("PERSON", TermCategory.CHARACTER),
        ("nickname", TermCategory.CHARACTER),
        ("Geographic Locations", TermCategory.LOCATION),
        ("places", TermCategory.LOCATION),
        ("location (city)", TermCategory.LOCATION),
        ("Organizations & Factions", TermCategory.ORGANIZATION),
        ("guild", TermCategory.ORGANIZATION),
        ("Artifacts & Objects", TermCategory.ARTIFACT),
        ("vehicle", TermCategory.ARTIFACT),
        ("Domain-Specific Terminology", TermCategory.TERMINOLOGY),
        ("measurement unit", TermCategory.TERMINOLOGY),
        ("something the model invented", TermCategory.TERMINOLOGY),
        ("", TermCategory.TERMINOLOGY),
        (None, TermCategory.TERMINOLOGY),
    ],
)
def test_a_label_the_model_invents_is_normalised_not_rejected(label, expected):
    assert normalize_category(label) is expected
    entry = GlossaryEntry.model_validate(
        {"term": "X", "category": label, "translation": "X"}
    )
    assert entry.category is expected


def test_an_entry_with_an_empty_term_or_translation_is_rejected():
    """An empty field is the failure that makes the caller retry, so it must raise."""
    with pytest.raises(ValidationError):
        GlossaryEntry.model_validate({"term": "", "category": "character", "translation": "X"})
    with pytest.raises(ValidationError):
        GlossaryEntry.model_validate({"term": "X", "category": "character", "translation": "   "})


def test_extra_fields_and_whitespace_are_tolerated():
    """Old dictionaries and chatty models both carry fields the schema dropped."""
    entry = GlossaryEntry.model_validate(
        {
            "term": "  Novi   Sad ",
            "category": "location",
            "translation": " Novi Sad ",
            "brief_context": "a context this version no longer stores",
            "confidence": 0.9,
            "notes": "the model added fields of its own",
        }
    )
    assert entry.term == "Novi Sad"
    assert entry.translation == "Novi Sad"
    assert not hasattr(entry, "brief_context")


def test_the_answer_may_be_an_object_an_array_or_a_renamed_wrapper():
    entry = {"term": "X", "category": "character", "translation": "X"}
    assert len(ChapterExtraction.model_validate({"entries": [entry]}).entries) == 1
    # A bare array: the same answer without the envelope the prompt asked for.
    assert len(ChapterExtraction.model_validate([entry]).entries) == 1
    # A wrapper the model invented instead of "entries".
    assert len(ChapterExtraction.model_validate({"glossary": [entry]}).entries) == 1


def test_an_empty_answer_is_valid_and_means_the_chapter_adds_nothing():
    for payload in ({}, {"entries": []}, {"entries": None}, []):
        extraction = ChapterExtraction.model_validate(payload)
        assert extraction.entries == []
        assert extraction.to_dict() == {"entries": []}


def test_entry_key_folds_case_and_space_but_not_inflection():
    assert entry_key("Novi  Sad") == entry_key("novi sad") == "novi sad"
    # Two spellings of one name are one entity here; two *forms* are not.
    assert entry_key("Marko") != entry_key("Marka")
