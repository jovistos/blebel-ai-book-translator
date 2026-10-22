"""The glossary dictionary and the record shapes it stores.

``GlossaryBook`` moved out of ``runner.py`` (and then out of ``records.py``) so
the merge rules can be tested as a value. These tests cover the first-rendering
wins rule, the wire -> row conversion, and the records' single serialisation
path.
"""

from __future__ import annotations

import json

from book_splitter.glossary.book import GlossaryBook, MergeOutcome
from book_splitter.glossary.models import ChapterExtraction
from book_splitter.glossary.records import ChapterGlossary, GlossaryTerm


def _entries(*rows: dict) -> list:
    return ChapterExtraction.model_validate({"entries": list(rows)}).entries


def test_from_entry_locks_the_wire_answer_with_provenance() -> None:
    entry = _entries({"term": "Potter", "category": "person", "translation": "Poter"})[0]

    term = GlossaryTerm.from_entry(entry, chapter_index=2, chapter_filename="02.md")

    assert term.term == "Potter"
    assert term.category == "character"
    assert term.translation == "Poter"
    assert term.chapter_index == 2
    assert term.chapter_filename == "02.md"
    assert term.to_dict() == {
        "term": "Potter",
        "category": "character",
        "translation": "Poter",
        "chapter_index": 2,
        "chapter_filename": "02.md",
        "also_returned_in": [],
    }


def test_the_first_rendering_wins_and_later_drift_is_a_conflict() -> None:
    book = GlossaryBook()
    book.merge(
        _entries({"term": "Potter", "category": "character", "translation": "Poter"}),
        chapter_index=1,
        chapter_filename="01.md",
    )
    outcome = book.merge(
        _entries({"term": "potter", "category": "character", "translation": "Potter"}),
        chapter_index=2,
        chapter_filename="02.md",
    )

    assert isinstance(outcome, MergeOutcome)
    assert outcome.added == []
    assert outcome.conflicts == [
        {
            "term": "Potter",
            "chapter_index": 2,
            "chapter_filename": "02.md",
            "locked": "Poter",
            "proposed": "Potter",
        }
    ]
    assert [term.translation for term in book] == ["Poter"]


def test_a_repeat_is_deduplicated_per_file() -> None:
    book = GlossaryBook()
    book.merge(
        _entries({"term": "Potter", "category": "character", "translation": "Poter"}),
        chapter_index=1,
        chapter_filename="01.md",
    )
    outcome = book.merge(
        _entries({"term": "Potter", "category": "character", "translation": "POTER"}),
        chapter_index=3,
        chapter_filename="03.md",
    )

    assert outcome.repeats == ["Potter"]
    assert book.terms()[0].repeats == ["03.md"]


def test_a_book_can_be_rebuilt_from_its_own_rows() -> None:
    book = GlossaryBook(
        [
            GlossaryTerm(
                term="Novi Sad",
                category="location",
                translation="Novi Sad",
                chapter_index=1,
                chapter_filename="01.md",
            )
        ]
    )

    rebuilt = GlossaryBook([GlossaryTerm.from_dict(row) for row in book.to_list()])

    assert rebuilt.to_list() == book.to_list()
    assert rebuilt.categories() == {"location": 1}


def test_the_chapter_record_projects_one_declared_shape() -> None:
    record = ChapterGlossary(
        index=2,
        chapter_id="02-b",
        filename="02-b.md",
        title="Drugo",
        added=3,
        repeats=1,
        conflicts=1,
        log=["one"],
        added_terms=[{"term": "X"}],
    )

    manifest = record.record()
    trace = record.to_dict()

    assert manifest == {
        "index": 2,
        "body_index": 0,
        "filename": "02-b.md",
        "chapter_id": "02-b",
        "title": "Drugo",
        "kind": "chapter",
        "sentences_path": "",
        "output_lines": 0,
        "source_chars": 0,
        "chunks": 0,
        "models_called": 0,
        "attempts": 0,
        "method": "",
        "added": 3,
        "repeats": 1,
        "conflicts": 1,
        "skipped": False,
        "skip_reason": None,
        "reused": False,
        "ok": True,
        "error": None,
        "tokens": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        },
        "elapsed_seconds": 0.0,
    }
    assert trace == {
        **manifest,
        "log": ["one"],
        "added_terms": [{"term": "X"}],
        "conflict_detail": [],
        "raw_answers": [],
    }
    assert json.loads(json.dumps(trace)) == trace


def test_records_still_expose_the_old_book_import_path() -> None:
    from book_splitter.glossary import records

    assert records.GlossaryBook is GlossaryBook
    assert records.MergeOutcome is MergeOutcome
