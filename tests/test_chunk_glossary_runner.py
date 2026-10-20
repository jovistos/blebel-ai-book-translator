"""Offline tests for stage 6's runner: two phases, in that order, chunk by chunk.

No tagger, no model download and no provider: the NER backend is a scripted
double, and the matcher is a recorder wrapped around the real rapidfuzz matcher.
What is pinned here is the contract the translation stage will consume:

* **phase 1 finishes for every chunk before phase 2 starts** — the recorder
  proves that no glossary lookup happens until the last chunk has been tagged;
* every chunk gets exactly one file, named after the chunk;
* an exact term retrieves its locked translation, a near miss stays a near miss
  at the default 95 % and becomes a match at a lower threshold;
* the manifest and the trace carry the two phases, the counts and the candidates.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from book_splitter.chunk_glossary import (
    LexiconEntry,
    RapidFuzzMatcher,
    aggregate_entities,
    collect_matched_terms,
    load_chunk_glossary_manifest,
    run_chunk_glossary,
)
from book_splitter.chunk_glossary.runner import (
    TermMatch,
    chunk_glossary_filename,
    discover_chunk_inputs,
    resolve_chunks_dir,
)
from book_splitter.chunking import run_chunking
from book_splitter.core.errors import ToolError
from book_splitter.dictionary.ner import Entity

# --------------------------------------------------------------------------- #
# doubles and fixtures
# --------------------------------------------------------------------------- #
CHAPTERS = {
    "01-a.md": "Džordž živi u Londonu.\n\nDžordža nema.\n",
    "02-b.md": "Dženi je u Beogradu.\n\nMithril je metal.\n",
}
#: The fake tagger's vocabulary. Keys are the words as they appear in the text;
#: values are the surface form and type a real tagger would return.
WORDS: dict[str, tuple[str, str]] = {
    "Džordž": ("Džordž", "PER"),
    "Džordža": ("Džordža", "PER"),
    "Londonu": ("London", "LOC"),
    "Dženi": ("Dženi", "PER"),
    "Beogradu": ("Beograd", "LOC"),
    "Mithril": ("Mithril", "MISC"),
}
GLOSSARY = {
    "entries": [
        {
            "term": "Džordž",
            "category": "character",
            "translation": "Džordž",
            "chapter_index": 1,
            "chapter_filename": "01-a.md",
        },
        {
            "term": "London",
            "category": "location",
            "translation": "London",
            "chapter_index": 1,
            "chapter_filename": "01-a.md",
        },
        {
            "term": "Dženi",
            "category": "character",
            "translation": "Dženi",
            "chapter_index": 2,
            "chapter_filename": "02-b.md",
        },
        {
            "term": "Beograd",
            "category": "location",
            "translation": "Beograd",
            "chapter_index": 2,
            "chapter_filename": "02-b.md",
        },
    ]
}


class FakeBackend:
    """A scripted NER tagger: word lookup, and a record of every text it saw."""

    name = "fake"
    language = "sr"

    def __init__(self, events: list[tuple[str, str]] | None = None) -> None:
        self.events = events if events is not None else []
        self.texts: list[str] = []

    def entities(self, text: str) -> list[Entity]:
        self.events.append(("ner", text))
        self.texts.append(text)
        found: list[Entity] = []
        for word in re.findall(r"\w+", text, re.UNICODE):
            hit = WORDS.get(word)
            if hit is not None:
                found.append(Entity(hit[0], hit[1]))
        return found


class RecordingMatcher:
    """The real matcher, with every ``match`` call appended to a shared log."""

    def __init__(self, inner: RapidFuzzMatcher, events: list[tuple[str, str]]) -> None:
        self.inner = inner
        self.events = events

    def match(self, term: str, *, limit: int | None = None):
        self.events.append(("match", term))
        return self.inner.match(term, limit=limit)


def _project(tmp_path: Path, *, chunk_size: int = 8) -> tuple[Path, Path, Path]:
    """A stage-1/5/4 layout: chapters, chunks and a glossary, in one book folder."""
    book = tmp_path / "book"
    chapters = book / "01-chapters"
    chapters.mkdir(parents=True)
    for filename, text in CHAPTERS.items():
        (chapters / filename).write_text(text, encoding="utf-8")
    chunking = run_chunking(str(chapters), tokenizer="word", chunk_size=chunk_size)
    assert chunking.ok is True
    glossary = book / "04-glossary"
    glossary.mkdir()
    (glossary / "glossary.json").write_text(
        json.dumps(GLOSSARY, ensure_ascii=False), encoding="utf-8"
    )
    return chapters, book / "05-chunks", glossary


def _lexicon() -> list[LexiconEntry]:
    return [
        LexiconEntry(
            entry["term"],
            entry["translation"],
            entry.get("category", "terminology"),
            entry.get("chapter_index", 0),
            entry.get("chapter_filename", ""),
        )
        for entry in GLOSSARY["entries"]
    ]


def _run(
    chunks: Path,
    glossary: Path,
    *,
    threshold: float = 95.0,
    backend: FakeBackend | None = None,
    events: list[tuple[str, str]] | None = None,
    language: str = "sr",
    **kwargs,
):
    log = events if events is not None else []
    backend = backend or FakeBackend(log)
    matcher = RecordingMatcher(RapidFuzzMatcher(_lexicon()), log)
    result = run_chunk_glossary(
        str(chunks),
        glossary_path=str(glossary / "glossary.json"),
        language=language,
        backend=backend,
        matcher=matcher,
        threshold=threshold,
        **kwargs,
    )
    return result, backend, log


def _payloads(result) -> list[dict]:
    folder = Path(result.output_dir)
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(folder.glob("*.md.json"))
    ]


def _all_terms(payloads: list[dict]) -> list[dict]:
    return [term for payload in payloads for term in payload["terms"]]


# --------------------------------------------------------------------------- #
# the two phases
# --------------------------------------------------------------------------- #
def test_all_ner_calls_happen_before_the_first_glossary_lookup(tmp_path: Path):
    _, chunks, glossary = _project(tmp_path)
    events: list[tuple[str, str]] = []

    result, backend, log = _run(chunks, glossary, events=events)

    assert result.ok is True
    assert result.errors == []
    assert len(backend.texts) == result.chunk_count  # one NER call per chunk
    positions = [index for index, (kind, _value) in enumerate(log) if kind == "ner"]
    matches = [index for index, (kind, _value) in enumerate(log) if kind == "match"]
    assert positions and matches
    assert max(positions) < min(matches), "a glossary lookup ran before every chunk was tagged"
    # The two phases are explicitly reported, in order.
    assert [phase["name"] for phase in result.phases] == ["ner", "match"]
    assert result.phases[0]["chunks"] == result.chunk_count


def test_an_injected_backend_supplies_its_language_when_auto(tmp_path: Path):
    _, chunks, glossary = _project(tmp_path)
    result, _, _ = _run(chunks, glossary, language="auto")

    assert result.ok is True
    assert result.language["language"] == "sr"
    assert result.language["method"] == "backend"


def test_every_chunk_gets_a_file_named_after_the_chunk(tmp_path: Path):
    _, chunks, glossary = _project(tmp_path)
    result, _, _ = _run(chunks, glossary)

    expected = {chunk_glossary_filename(path.name) for path in chunks.glob("*.md")}
    written = {path.name for path in Path(result.output_dir).glob("*.md.json")}
    assert written == expected
    assert result.chunk_count == len(expected)

    for payload in _payloads(result):
        assert payload["stage"] == "chunk-glossary"
        assert payload["chunk"]["filename"] + ".json" in written
        assert payload["counts"]["matched"] == len(payload["matched_terms"])
        assert payload["counts"]["terms"] == len(payload["terms"])


def test_an_exact_term_retrieves_the_locked_translation(tmp_path: Path):
    _, chunks, glossary = _project(tmp_path)
    result, _, _ = _run(chunks, glossary)

    payloads = _payloads(result)
    terms = {term["term"]: term for term in _all_terms(payloads)}
    assert terms["Džordž"]["matched"] is True
    assert terms["Džordž"]["score"] == 100.0
    assert terms["Džordž"]["glossary"]["translation"] == "Džordž"
    assert terms["London"]["glossary"]["translation"] == "London"

    matched = {row["translation"] for payload in payloads for row in payload["matched_terms"]}
    assert {"Džordž", "London", "Dženi", "Beograd"} <= matched


def test_a_near_miss_is_recorded_as_a_closest_candidate_not_a_match(tmp_path: Path):
    _, chunks, glossary = _project(tmp_path)
    result, _, _ = _run(chunks, glossary, threshold=95.0)

    terms = {term["term"]: term for term in _all_terms(_payloads(result))}
    assert "Džordža" in terms
    near = terms["Džordža"]
    assert near["matched"] is False
    assert near["glossary"] is None
    assert near["closest"]["term"] == "Džordž"
    assert 90.0 <= near["closest"]["score"] < 95.0


def test_the_threshold_is_an_option(tmp_path: Path):
    _, chunks, glossary = _project(tmp_path)
    strict, _, _ = _run(
        chunks, glossary, threshold=95.0, output_dir=str(tmp_path / "strict")
    )
    loose, _, _ = _run(
        chunks, glossary, threshold=90.0, output_dir=str(tmp_path / "loose")
    )

    strict_terms = {term["term"]: term for term in _all_terms(_payloads(strict))}
    loose_terms = {term["term"]: term for term in _all_terms(_payloads(loose))}

    assert strict_terms["Džordža"]["matched"] is False
    assert loose_terms["Džordža"]["matched"] is True
    assert loose_terms["Džordža"]["glossary"]["translation"] == "Džordž"
    assert loose.matched_term_count >= strict.matched_term_count


def test_an_unknown_term_is_reported_as_unmatched(tmp_path: Path):
    _, chunks, glossary = _project(tmp_path)
    result, _, _ = _run(chunks, glossary)

    mithril = next(term for term in _all_terms(_payloads(result)) if term["term"] == "Mithril")
    assert mithril["matched"] is False
    assert mithril["glossary"] is None
    assert mithril["closest"]["score"] < 95.0
    assert result.unmatched_term_count >= 1


# --------------------------------------------------------------------------- #
# artifacts
# --------------------------------------------------------------------------- #
def test_manifest_and_trace_describe_the_two_phase_run(tmp_path: Path):
    _, chunks, glossary = _project(tmp_path)
    result, _, _ = _run(chunks, glossary)

    manifest = load_chunk_glossary_manifest(result.output_dir)
    assert manifest["stage"] == "chunk-glossary"
    assert manifest["ok"] is True
    assert manifest["chunk_count"] == result.chunk_count
    assert manifest["settings"]["threshold"] == 95.0
    assert manifest["settings"]["scorer"] == "ratio"
    assert manifest["entry_count"] == len(GLOSSARY["entries"])
    assert [phase["name"] for phase in manifest["phases"]] == ["ner", "match"]
    assert len(manifest["chunks"]) == result.chunk_count
    assert all("matched_terms" in chunk for chunk in manifest["chunks"])

    trace = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
    assert len(trace["detail"]) == result.chunk_count
    assert trace["detail"][0]["terms"]


def test_dry_run_runs_both_phases_and_writes_nothing(tmp_path: Path):
    _, chunks, glossary = _project(tmp_path)
    events: list[tuple[str, str]] = []
    result, backend, _ = _run(chunks, glossary, events=events, dry_run=True)

    assert result.ok is True
    assert result.dry_run is True
    assert result.manifest_path is None
    assert result.trace_path is None
    assert not Path(result.output_dir).exists()
    assert events  # both phases ran
    assert len(backend.texts) == result.chunk_count


def test_limit_processes_only_the_first_chunks(tmp_path: Path):
    _, chunks, glossary = _project(tmp_path)
    all_result, _, _ = _run(chunks, glossary)
    limited, backend, _ = _run(chunks, glossary, limit=1)

    assert limited.ok is True
    assert limited.limited is True
    assert limited.chunk_count == 1
    assert limited.chunk_count < all_result.chunk_count
    assert len(backend.texts) == 1


def test_no_manifest_leaves_only_the_per_chunk_files(tmp_path: Path):
    _, chunks, glossary = _project(tmp_path)
    result, _, _ = _run(chunks, glossary, write_manifest=False)

    folder = Path(result.output_dir)
    assert result.manifest_path is None
    assert result.trace_path is None
    assert not (folder / "_chunk_glossary_manifest.json").exists()
    assert list(folder.glob("*.md.json"))


# --------------------------------------------------------------------------- #
# errors and helpers
# --------------------------------------------------------------------------- #
def test_a_missing_glossary_is_a_setup_error(tmp_path: Path):
    _, chunks, _ = _project(tmp_path)
    with pytest.raises(ToolError, match="glossary not found"):
        _run(chunks, tmp_path / "missing")


def test_a_missing_chunks_folder_is_a_setup_error(tmp_path: Path):
    with pytest.raises(ToolError, match="run `chunk` first"):
        run_chunk_glossary(str(tmp_path / "missing"), language="sr", backend=FakeBackend())


def test_resolve_chunks_dir_accepts_a_book_folder_or_the_manifest(tmp_path: Path):
    _, chunks, _ = _project(tmp_path)
    assert resolve_chunks_dir(chunks) == chunks
    assert resolve_chunks_dir(chunks.parent) == chunks
    assert resolve_chunks_dir(chunks / "_chunk_manifest.json") == chunks


def test_discover_chunk_inputs_reads_the_manifest_in_global_order(tmp_path: Path):
    _, chunks, _ = _project(tmp_path)
    inputs, folder = discover_chunk_inputs(chunks)

    assert folder == chunks
    assert [chunk.global_index for chunk in inputs] == list(range(len(inputs)))
    for chunk in inputs:
        assert chunk.filename
        assert chunk.text == (chunks / chunk.filename).read_text(encoding="utf-8")
        assert chunk.sha256


def test_aggregate_entities_merges_repeats_and_filters_noise():
    entities = [
        Entity("Džordž", "PER"),
        Entity("Džordž", "PER"),
        Entity("Džordž", "MISC"),
        Entity("A", "MISC"),  # too short
        Entity("1996", "MISC"),  # no letter
        Entity("London", "LOC"),
        Entity("London", "GPE"),  # not a CLASSLA type: ignored
    ]
    terms, mentions = aggregate_entities(entities, min_term_chars=2)
    by_term = {term.term: term for term in terms}

    # Every wanted-type mention counts, including the duplicate MISC one; the
    # short term, the number-only term and the unmapped GPE do not.
    assert mentions == 4
    assert by_term["Džordž"].mentions == 3
    assert by_term["Džordž"].types == ["MISC", "PER"]
    assert "A" not in by_term
    assert "1996" not in by_term
    assert by_term["London"].types == ["LOC"]


def test_aggregate_entities_accepts_a_comma_separated_keep_types_string():
    terms, mentions = aggregate_entities(
        [Entity("Džordž", "PER"), Entity("London", "LOC")], keep_types="PER"
    )
    assert [term.term for term in terms] == ["Džordž"]
    assert mentions == 1


def test_collect_matched_terms_keeps_one_row_per_glossary_entry():
    matcher = RapidFuzzMatcher([LexiconEntry("Džordž", "Džordž", "character")])
    first = TermMatch(term="Džordž", types=["PER"], mentions=2, candidates=matcher.match("Džordž"))
    second = TermMatch(
        term="Džordža", types=["PER"], mentions=1, candidates=matcher.match("Džordža"), threshold=90
    )

    rows = collect_matched_terms([first, second])

    assert len(rows) == 1
    assert rows[0]["glossary_term"] == "Džordž"
    assert rows[0]["mentions"] == 3
    assert rows[0]["score"] == 100.0


def test_skip_if_present_reuses_finished_chunks_and_only_tags_new_ones(tmp_path: Path):
    _, chunks, glossary = _project(tmp_path, chunk_size=4)
    glossary_path = glossary / "glossary.json"
    total = len(discover_chunk_inputs(chunks)[0])

    first_backend = FakeBackend()
    first = run_chunk_glossary(
        str(chunks),
        glossary_path=str(glossary_path),
        backend=first_backend,
        limit=1,
        skip_if_present=True,
    )
    assert first.ok is True
    assert first.processed_count == 1
    assert first.skipped_count == 0

    # Growing the window: the first chunk is reused, only the rest is tagged.
    second_backend = FakeBackend()
    second = run_chunk_glossary(
        str(chunks),
        glossary_path=str(glossary_path),
        backend=second_backend,
        skip_if_present=True,
    )
    assert second.ok is True
    assert second.skipped_count == 1
    assert second.processed_count == total - 1
    assert len(second_backend.texts) == total - 1
    assert first_backend.texts[0] not in second_backend.texts

    # A fully reused run needs no tagger at all.
    third_backend = FakeBackend()
    third = run_chunk_glossary(
        str(chunks),
        glossary_path=str(glossary_path),
        backend=third_backend,
        skip_if_present=True,
    )
    assert third.ok is True
    assert third.processed_count == 0
    assert third.skipped_count == total
    assert third_backend.texts == []
    assert third.totals["chunks"] == total
