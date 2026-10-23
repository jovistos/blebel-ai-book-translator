"""Stage 3: reduce each chapter to the sentences a dictionary can be mined from.

Stage 1 (:mod:`book_translator.stages.agent`) cuts a book into chapters and stage 2
(:mod:`book_translator.stages.preprocess_agent`) repairs their shapes. This package takes
stage 1's folder and turns every chapter into a **reduced chapter**: the raw
sentences that are worth mining for the proper nouns, characters and invented
terms a translation has to keep consistent.

The funnel, cheapest test first:

``split_sentences``
    deterministic sentence splitting — no model, no network;
``prefilter_sentences``
    drop sentences with no capital past the first word, then sentences under four
    words;
``gate_entities``
    keep a sentence only if the tagger found PER/LOC/ORG/MISC in it;
``select_unique_sentences``
    keep a sentence only if it introduces a noun no earlier sentence used;
``run_dictionary``
    drive that over every chapter, first to last, and write two folders:
    ``03-reduced/`` (every entity-bearing sentence) and ``03-reduced-unique/`` (the
    first mentions only, book-wide and in reading order).

The tagger is chosen by language: CLASSLA for South Slavic, stanza for English
(:mod:`book_translator.stages.dictionary.ner`). Both are **optional, lazily imported**
dependencies — see ``requirements-dictionary.txt``. Neither builds the
dictionary: CLASSLA's four traditional entity types cannot recognise a fictional
concept, so it only filters; :mod:`book_translator.stages.glossary` (stage 4) does the
mining.
"""

from .language import (
    LanguageDetector,
    LanguageGuess,
    LanguageSample,
    LinguaDetector,
    count_reflexes,
    detect_language,
    resolve_south_slavic,
    sample_chapter_text,
)
from .ner import (
    ClasslaBackend,
    Entity,
    GateDecision,
    NerBackend,
    StanzaBackend,
    backend_for_language,
    configure_gpu,
    gate_entities,
    make_backend,
)
from .runner import (
    ChapterReduction,
    DictionaryResult,
    ProgressHook,
    TextReduction,
    reduce_text,
    run_dictionary,
)
from .sentences import (
    PreFilterResult,
    has_later_uppercase,
    prefilter_sentences,
    split_sentences,
    word_count,
)
from .unique import (
    UniqueSelection,
    extend_seen,
    noun_key,
    select_unique_sentences,
    sentence_noun_keys,
)

__all__ = [
    "ChapterReduction",
    "ClasslaBackend",
    "DictionaryResult",
    "Entity",
    "GateDecision",
    "LanguageDetector",
    "LanguageGuess",
    "LanguageSample",
    "LinguaDetector",
    "NerBackend",
    "PreFilterResult",
    "ProgressHook",
    "StanzaBackend",
    "TextReduction",
    "UniqueSelection",
    "backend_for_language",
    "configure_gpu",
    "count_reflexes",
    "detect_language",
    "extend_seen",
    "gate_entities",
    "has_later_uppercase",
    "make_backend",
    "noun_key",
    "prefilter_sentences",
    "reduce_text",
    "resolve_south_slavic",
    "run_dictionary",
    "sample_chapter_text",
    "select_unique_sentences",
    "sentence_noun_keys",
    "split_sentences",
    "word_count",
]
