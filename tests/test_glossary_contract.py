"""The glossary records' serialisation contract, frozen before the consolidation.

The glossary stage writes two shapes from one chapter record: the manifest row
(``ChapterGlossary.record``) and the trace row (``ChapterGlossary.to_dict``), and
two more records from the run and the dictionary. Their key sets and key order
are the files downstream stages read, so they are pinned here.
"""

from __future__ import annotations

import json

from book_splitter.glossary.records import (
    ChapterGlossary,
    GlossaryResult,
    GlossaryTerm,
)

MANIFEST_KEYS = [
    "index",
    "body_index",
    "filename",
    "chapter_id",
    "title",
    "kind",
    "sentences_path",
    "output_lines",
    "source_chars",
    "chunks",
    "models_called",
    "attempts",
    "method",
    "added",
    "repeats",
    "conflicts",
    "skipped",
    "skip_reason",
    "reused",
    "ok",
    "error",
    "tokens",
    "elapsed_seconds",
]
TRACE_ONLY_KEYS = ["log", "added_terms", "conflict_detail", "raw_answers"]


def test_a_dictionary_term_has_one_serialised_shape() -> None:
    term = GlossaryTerm(
        term="Potter",
        category="character",
        translation="Poter",
        chapter_index=2,
        chapter_filename="02.md",
        repeats=["03.md"],
    )

    assert term.to_dict() == {
        "term": "Potter",
        "category": "character",
        "translation": "Poter",
        "chapter_index": 2,
        "chapter_filename": "02.md",
        "also_returned_in": ["03.md"],
    }
    assert list(term.to_dict()) == [
        "term",
        "category",
        "translation",
        "chapter_index",
        "chapter_filename",
        "also_returned_in",
    ]
    assert GlossaryTerm.from_dict(term.to_dict()).to_dict() == term.to_dict()


def test_a_chapter_record_keeps_the_manifest_and_trace_shapes(tmp_path) -> None:
    record = ChapterGlossary(index=1, chapter_id="01-a", filename="01-a.md")

    assert list(record.record()) == MANIFEST_KEYS
    assert list(record.to_dict()) == [*MANIFEST_KEYS, *TRACE_ONLY_KEYS]
    assert record.record()["tokens"] == {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
    }
    assert record.record()["elapsed_seconds"] == 0.0
    assert json.loads(json.dumps(record.to_dict())) == record.to_dict()


def test_the_run_result_has_one_serialised_shape() -> None:
    result = GlossaryResult(ok=True, chapters_path="c", sentences_dir="s", output_dir="o")

    keys = list(result.to_dict())
    assert keys[:3] == ["ok", "chapters_path", "sentences_dir"]
    assert keys[-4:] == ["glossary_path", "markdown_path", "manifest_path", "trace_path"]
    assert json.loads(json.dumps(result.to_dict())) == result.to_dict()
