"""Offline tests for stage 3's sentence funnel: split -> uppercase -> 4 words.

No tagger, no model, no network: these are the deterministic steps that decide
how much text the tagger ever sees, so they are pinned character by character.
"""

from __future__ import annotations

from book_splitter.dictionary.ner import (
    Entity,
    GateDecision,
    gate_entities,
    is_temporal,
    normalize_type,
)
from book_splitter.dictionary.sentences import (
    has_later_uppercase,
    prefilter_sentences,
    split_sentences,
    word_count,
)


# --------------------------------------------------------------------------- #
# sentence splitting
# --------------------------------------------------------------------------- #
def test_paragraphs_are_split_on_blank_lines_and_soft_wraps_are_joined():
    text = "Prva rečenica je ovde.\n\nDruga rečenica\nnastavlja se u istom pasusu.\n"
    assert split_sentences(text) == [
        "Prva rečenica je ovde.",
        "Druga rečenica nastavlja se u istom pasusu.",
    ]


def test_a_terminator_ends_a_sentence_inside_a_paragraph():
    text = "Marko je došao. Ana je otišla! Da li je ostao? Nije."
    assert split_sentences(text) == [
        "Marko je došao.",
        "Ana je otišla!",
        "Da li je ostao?",
        "Nije.",
    ]


def test_headings_become_their_own_sentence_without_the_markup():
    assert split_sentences("## PRVI DEO: ISPUNJENJE LJUBAVI\n\nTekst je ovde.") == [
        "PRVI DEO: ISPUNJENJE LJUBAVI",
        "Tekst je ovde.",
    ]


def test_abbreviations_and_initials_do_not_end_a_sentence():
    text = "Dr. Jovan Jovanović je došao. J. R. R. Tolkien je pisao."
    assert split_sentences(text) == [
        "Dr. Jovan Jovanović je došao.",
        "J. R. R. Tolkien je pisao.",
    ]


def test_an_abbreviation_inside_a_sentence_is_not_a_boundary():
    text = "Doneo je knjige, olovke, itd. i otišao je kući."
    assert split_sentences(text) == ["Doneo je knjige, olovke, itd. i otišao je kući."]


def test_a_lowercase_word_after_a_dot_is_a_continuation_not_a_new_sentence():
    """`5. maja 1990.` is one sentence; the dot after the day is not a full stop."""
    assert split_sentences("Umro je 5. maja 1990. godine u Beogradu.") == [
        "Umro je 5. maja 1990. godine u Beogradu."
    ]


def test_ellipsis_inside_dialogue_stays_with_its_attribution():
    text = 'Rekao je: "Ne znam..." i otišao je. Ana je ostala.'
    assert split_sentences(text) == [
        'Rekao je: "Ne znam..." i otišao je.',
        "Ana je ostala.",
    ]


def test_a_closing_quote_is_kept_with_the_sentence_it_closes():
    text = '"Kako si?" upitao je. "Dobro sam."'
    assert split_sentences(text) == ['"Kako si?" upitao je.', '"Dobro sam."']


def test_terminators_without_a_following_sentence_are_kept():
    assert split_sentences("Sve je bilo tiho...") == ["Sve je bilo tiho..."]


def test_empty_text_has_no_sentences():
    assert split_sentences("") == []
    assert split_sentences("\n\n  \n") == []


# --------------------------------------------------------------------------- #
# the two cheap filters
# --------------------------------------------------------------------------- #
def test_uppercase_after_the_first_word_decides():
    assert has_later_uppercase("On je otišao u Novi Sad.") is True
    assert has_later_uppercase("On je otišao kući.") is False
    # The capital that opens the sentence does not count.
    assert has_later_uppercase("Marko je došao.") is False
    assert has_later_uppercase("Došao je i rekao: OK.") is True


def test_word_count_ignores_punctuation():
    assert word_count("Da, dobro je.") == 3
    assert word_count("...") == 0


def test_prefilter_drops_short_before_it_counts_and_reports_a_reason():
    sentences = [
        "On je otišao u Novi Sad.",  # kept: 5 words, capital past the first
        "On je otišao kući.",  # no capital past the first word
        "Idem kući.",  # no capital past the first word, and only 2 words
        "Ana i Marko.",  # 3 words -> too short, though `Marko` is capitalised
    ]
    result = prefilter_sentences(sentences, min_words=4, require_uppercase=True)

    assert result.kept == ["On je otišao u Novi Sad."]
    # `Idem kući.` fails both tests and is reported under the first one that saw it.
    assert result.dropped_no_uppercase == ["On je otišao kući.", "Idem kući."]
    assert result.dropped_short == ["Ana i Marko."]
    assert result.counts == {
        "input": 4,
        "kept": 1,
        "dropped_short": 1,
        "dropped_no_uppercase": 2,
    }


def test_the_uppercase_filter_can_be_switched_off():
    result = prefilter_sentences(
        ["On je otišao kući."], min_words=4, require_uppercase=False
    )
    assert result.kept == ["On je otišao kući."]


def test_a_four_word_sentence_survives_the_floor():
    result = prefilter_sentences(["Sreli smo Marka juče."], min_words=4)
    assert result.kept == ["Sreli smo Marka juče."]


# --------------------------------------------------------------------------- #
# the NER gate's decision rules (no tagger involved)
# --------------------------------------------------------------------------- #
def test_iob_prefixes_are_stripped_from_entity_types():
    assert normalize_type("B-PER") == "PER"
    assert normalize_type("i-loc") == "LOC"
    assert normalize_type("MISC") == "MISC"


def test_a_sentence_without_an_entity_is_dropped():
    decision = gate_entities([])
    assert decision == GateDecision(entities=[], keep=False, reason="no_entity")


def test_one_wanted_entity_keeps_the_whole_sentence():
    entities = [Entity("Marko", "B-PER"), Entity("Beograd", "I-LOC")]
    decision = gate_entities(entities)
    assert decision.keep is True
    assert decision.reason == "kept"
    assert decision.types == ["PER", "LOC"]


def test_unwanted_types_do_not_save_a_sentence():
    decision = gate_entities([Entity("nešto", "OTHER")], keep_types=("PER", "LOC"))
    assert decision.keep is False


def test_temporal_only_sentences_are_optional():
    dated = [Entity("1996", "MISC")]
    assert gate_entities(dated).keep is True
    dropped = gate_entities(dated, drop_temporal_only=True)
    assert dropped.keep is False
    assert dropped.reason == "temporal_only"
    # A year next to a real name does not make the sentence temporal-only.
    mixed = [Entity("1996", "MISC"), Entity("Marko", "PER")]
    assert gate_entities(mixed, drop_temporal_only=True).keep is True


def test_temporal_mentions_are_recognised_by_shape():
    assert is_temporal(Entity("1996", "MISC")) is True
    assert is_temporal(Entity("5. maja 1990.", "MISC")) is True
    assert is_temporal(Entity("decembar", "MISC")) is True
    assert is_temporal(Entity("3:30", "MISC")) is True
    assert is_temporal(Entity("Mithril", "MISC")) is False
    assert is_temporal(Entity("Novi Sad", "LOC")) is False
