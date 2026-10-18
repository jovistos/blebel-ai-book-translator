"""Offline tests for stage 6's fuzzy glossary retrieval (rapidfuzz).

These pin the part that decides *which* locked entry a chunk term means: the
normalisation, the candidate ordering, the threshold semantics and the scorer
option. The matcher returns every candidate, including the near miss; the
runner, not the matcher, applies the 95 % rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from book_splitter.chunk_glossary import (
    LexiconEntry,
    RapidFuzzMatcher,
    clean_term,
    glossary_file,
    load_glossary,
    match_key,
    normalize_scorer,
    normalize_threshold,
)
from book_splitter.core.errors import ToolError


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
def test_clean_term_trims_punctuation_and_collapses_whitespace():
    assert clean_term('  "Džordž,  " ') == "Džordž"
    assert clean_term("New   York.") == "New York"
    assert clean_term("") == ""
    assert clean_term(None) == ""


def test_hyphens_and_apostrophes_survive_because_they_are_part_of_names():
    assert clean_term("Jenny Hamilton-Proctor") == "Jenny Hamilton-Proctor"
    assert clean_term("O'Brien") == "O'Brien"


def test_match_key_is_case_folded_but_keeps_diacritics():
    assert match_key("DŽORDŽ") == "džordž"
    assert match_key("  Džordž  ") == "džordž"
    assert match_key("Dzordz") != match_key("Džordž")


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def test_the_threshold_is_a_percentage_and_is_validated():
    assert normalize_threshold(95) == 95.0
    assert normalize_threshold("90.5") == 90.5
    assert normalize_threshold(0) == 0.0
    assert normalize_threshold(100) == 100.0
    with pytest.raises(ToolError):
        normalize_threshold(101)
    with pytest.raises(ToolError):
        normalize_threshold(-1)
    with pytest.raises(ToolError):
        normalize_threshold("high")


def test_the_scorer_is_validated_and_defaults_to_ratio():
    assert normalize_scorer(None) == "ratio"
    assert normalize_scorer("token_sort_ratio") == "token_sort_ratio"
    with pytest.raises(ToolError):
        normalize_scorer("levenshtein")


# --------------------------------------------------------------------------- #
# loading the glossary
# --------------------------------------------------------------------------- #
def _entry(term: str, translation: str, **extra) -> dict:
    return {"term": term, "translation": translation, **extra}


def test_load_glossary_accepts_a_folder_and_a_file(tmp_path: Path):
    folder = tmp_path / "glossary"
    folder.mkdir()
    payload = {
        "entries": [
            _entry("Džordž", "Džordž", category="character", chapter_filename="01-a.md"),
            _entry("London", "London", category="location"),
        ]
    }
    (folder / "glossary.json").write_text(json.dumps(payload), encoding="utf-8")

    from_folder = load_glossary(folder)
    from_file = load_glossary(folder / "glossary.json")

    assert glossary_file(folder) == folder / "glossary.json"
    assert [entry.term for entry in from_folder] == ["Džordž", "London"]
    assert from_folder == from_file
    assert from_folder[0].category == "character"
    assert from_folder[0].chapter_filename == "01-a.md"
    assert from_folder[0].to_dict()["translation"] == "Džordž"


def test_load_glossary_deduplicates_skips_empty_rows_and_accepts_a_bare_list(tmp_path: Path):
    path = tmp_path / "glossary.json"
    path.write_text(
        json.dumps(
            [
                _entry("Džordž", "Džordž"),
                _entry("DŽORDŽ", "Drugi prevod"),  # same key: first one wins
                _entry("", "prazno"),
                {"term": "Bez prevoda"},
                "not an entry",
            ]
        ),
        encoding="utf-8",
    )

    entries = load_glossary(path)

    assert [entry.term for entry in entries] == ["Džordž"]
    assert entries[0].translation == "Džordž"


def test_a_missing_glossary_is_one_actionable_error(tmp_path: Path):
    with pytest.raises(ToolError, match="run `glossary` first"):
        load_glossary(tmp_path / "missing" / "glossary.json")


# --------------------------------------------------------------------------- #
# the matcher
# --------------------------------------------------------------------------- #
GLOSSARY = [
    LexiconEntry("Džordž", "Džordž", "character"),
    LexiconEntry("London", "London", "location"),
    LexiconEntry("Dženi", "Dženi", "character"),
    LexiconEntry("Beograd", "Beograd", "location"),
]


def _scores(term: str, *, scorer: str = "ratio", limit: int | None = None) -> list[tuple[str, float]]:
    matcher = RapidFuzzMatcher(GLOSSARY, scorer=scorer, candidates=3)
    return [(candidate.entry.term, candidate.score) for candidate in matcher.match(term, limit=limit)]


def test_an_exact_term_scores_100_and_is_returned_first():
    scored = _scores("Džordž")
    assert scored[0] == ("Džordž", 100.0)
    assert scored[0][1] >= 95.0


def test_matching_is_case_insensitive():
    assert _scores("DŽORDŽ")[0] == ("Džordž", 100.0)


def test_a_near_miss_is_returned_but_stays_below_95():
    best, score = _scores("Džordža")[0]
    assert best == "Džordž"
    assert 90.0 <= score < 95.0


def test_the_number_of_candidates_is_configurable():
    matcher = RapidFuzzMatcher(GLOSSARY, candidates=1)
    assert len(matcher.match("Džordž")) == 1
    assert len(matcher.match("Džordž", limit=2)) == 2


def test_token_sort_ratio_ignores_word_order():
    matcher = RapidFuzzMatcher(
        [LexiconEntry("New York", "New York", "location")], scorer="token_sort_ratio"
    )
    assert matcher.match("York New")[0].score == 100.0
    assert _scores("York New", scorer="ratio")[0][1] < 100.0


def test_an_empty_glossary_or_an_empty_query_returns_no_candidates():
    assert RapidFuzzMatcher([]).match("Džordž") == []
    assert RapidFuzzMatcher(GLOSSARY).match("   ") == []
