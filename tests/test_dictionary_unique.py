"""Offline tests for stage 3's second filter: first mentions only.

The step is pure bookkeeping over entity mentions the NER gate already produced —
no tagger, no model, no files. What is pinned here is the rule itself (keep a
sentence only when it introduces a noun), the noun key (case-folded, punctuation
trimmed, type ignored) and the book-wide ``seen`` set that makes it cumulative.
"""

from __future__ import annotations

from book_splitter.dictionary.ner import Entity
from book_splitter.dictionary.unique import (
    extend_seen,
    noun_key,
    select_unique_sentences,
    sentence_noun_keys,
)


# --------------------------------------------------------------------------- #
# the noun key
# --------------------------------------------------------------------------- #
def test_a_key_is_case_folded_and_trimmed():
    assert noun_key("  Marko,") == "marko"
    assert noun_key("MARKO") == "marko"
    assert noun_key("„Novi   Sad”") == "novi sad"


def test_punctuation_only_mentions_have_no_key():
    assert noun_key("") == ""
    assert noun_key(" — ") == ""
    assert noun_key("...") == ""


def test_the_type_does_not_split_a_noun():
    """``Beograd`` is one noun whether the tagger called it LOC or ORG."""
    entities = [Entity("Beograd", "LOC"), Entity("BEOGRAD", "ORG")]
    assert sentence_noun_keys(entities) == ["beograd"]


def test_a_sentence_keeps_its_nouns_in_mention_order():
    entities = [Entity("Ana", "PER"), Entity("Novi Sad", "LOC"), Entity("Ana", "PER")]
    assert sentence_noun_keys(entities) == ["ana", "novi sad"]


# --------------------------------------------------------------------------- #
# the rule
# --------------------------------------------------------------------------- #
SENTENCES = [
    "Marko je ušao u sobu.",  # marko: new -> kept
    "Marko je seo za sto.",  # marko only: seen -> dropped
    "Ana je ušla za njim.",  # ana: new -> kept
    "Marko i Ana su razgovarali.",  # both seen -> dropped
    "Novi Sad je bio daleko.",  # novi sad: new -> kept
    "Ana je volela Novi Sad.",  # both seen -> dropped
]
KEYS = [
    ["marko"],
    ["marko"],
    ["ana"],
    ["marko", "ana"],
    ["novi sad"],
    ["ana", "novi sad"],
]


def test_only_the_sentences_that_introduce_a_noun_survive():
    selection = select_unique_sentences(SENTENCES, KEYS, seen=set())

    assert selection.kept == [
        "Marko je ušao u sobu.",
        "Ana je ušla za njim.",
        "Novi Sad je bio daleko.",
    ]
    assert selection.dropped_duplicate == 3
    assert selection.new_nouns == 3
    assert selection.new_keys == ["marko", "ana", "novi sad"]
    # The index the runner stores for a rerun: nouns per kept sentence.
    assert selection.kept_keys == [["marko"], ["ana"], ["novi sad"]]
    assert selection.dropped_keys == [["marko"], ["marko", "ana"], ["ana", "novi sad"]]


def test_a_first_mention_survives_sharing_a_sentence_with_a_known_name():
    """The rule is "at least one new noun", not "no repeated noun"."""
    selection = select_unique_sentences(
        ["Marko i Ana su razgovarali."], [["marko", "ana"]], seen={"marko"}
    )
    assert selection.kept == ["Marko i Ana su razgovarali."]
    assert selection.new_keys == ["ana"]


def test_the_seen_set_is_updated_in_place_and_carries_across_calls():
    seen: set[str] = set()
    first = select_unique_sentences(["Marko je došao."], [["marko"]], seen=seen)
    second = select_unique_sentences(["Marko je otišao."], [["marko"]], seen=seen)

    assert first.kept == ["Marko je došao."]
    assert second.kept == []  # the noun is the book's, not the call's
    assert seen == {"marko"}
    assert second.seen_before == 1 and second.seen_after == 1


def test_the_counters_describe_the_book_not_the_chapter():
    seen = {"ana"}
    selection = select_unique_sentences(
        ["Marko je došao.", "Ana je otišla."], [["marko"], ["ana"]], seen=seen
    )
    assert selection.seen_before == 1
    assert selection.seen_after == 2
    assert selection.new_nouns == 1


def test_examples_are_capped_for_the_trace():
    sentences = ["Marko je došao."] * 5
    selection = select_unique_sentences(
        sentences[1:], [["marko"]] * 4, seen={"marko"}, example_limit=2
    )
    assert len(selection.examples) == 2
    assert selection.dropped == sentences[1:]


def test_an_empty_sentence_or_an_empty_key_is_never_evidence():
    selection = select_unique_sentences(["Ništa se nije desilo."], [[]], seen=set())
    assert selection.kept == []
    assert selection.dropped_duplicate == 1
    assert selection.new_nouns == 0


def test_extend_seen_normalizes_what_the_manifest_recorded():
    seen: set[str] = set()
    extend_seen(seen, ["Marko", "  Ana ", "", "marko"])
    assert seen == {"marko", "ana"}


def test_the_selection_is_serialisable_with_its_counts_and_nouns():
    """The trace keeps the nouns and the counts; the sentences live in the folders."""
    selection = select_unique_sentences(SENTENCES, KEYS, seen=set())
    payload = selection.to_dict()
    assert payload["kept"] == 3
    assert payload["dropped"] == 3
    assert payload["dropped_duplicate"] == 3
    assert payload["new_nouns"] == 3
    assert payload["seen_after"] == 3
    assert payload["new_keys"] == ["marko", "ana", "novi sad"]
    assert payload["dropped_keys"] == [["marko"], ["marko", "ana"], ["ana", "novi sad"]]
    assert payload["examples"][0] == "Marko je seo za sto."
