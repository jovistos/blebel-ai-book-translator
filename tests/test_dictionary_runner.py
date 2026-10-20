"""Offline tests for the stage-3 runner: the funnel over a whole chapters folder.

The tagger is injected, so nothing here loads a model, downloads weights or
touches a GPU. What is pinned is the contract the rest of the pipeline depends
on: reduced files next to ``chapters/``, one raw sentence per line, a manifest in
the same shape as the other stages, language routing, and the ``skip_if_present``
short-circuit that must not even build a tagger.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest
from book_splitter.core.errors import MissingDependencyError, ToolError
from book_splitter.dictionary import language as language_module
from book_splitter.dictionary import ner as ner_module
from book_splitter.dictionary import runner as runner_module
from book_splitter.dictionary.ner import Entity, backend_for_language, configure_gpu, make_backend
from book_splitter.dictionary.runner import reduce_text, run_dictionary


# --------------------------------------------------------------------------- #
# doubles and fixtures
# --------------------------------------------------------------------------- #
class FakeNer:
    """A tagger that finds the words in its table, and records what it was asked."""

    name = "fake"

    def __init__(self, language: str = "sr", table: dict[str, str] | None = None) -> None:
        self.language = language
        self.table = (
            {"Novi Sad": "LOC", "Marko": "PER", "Beograd": "LOC", "Ana": "PER"}
            if table is None
            else table
        )
        self.calls: list[str] = []

    def entities(self, text: str) -> list[Entity]:
        self.calls.append(text)
        return [Entity(word, kind) for word, kind in self.table.items() if word in text]


class FakeDetector:
    """A detector that always answers with the same code."""

    def __init__(self, code: str = "sr", confidence: float = 0.91) -> None:
        self.code = code
        self.confidence = confidence
        self.calls: list[str] = []

    def detect(self, text: str) -> tuple[str, float]:
        self.calls.append(text)
        return self.code, self.confidence


class SteppingClock:
    """A clock that advances by a fixed step on every reading.

    With it, time spent between two adjacent readings is exactly one step, so
    the NER timings can be asserted exactly instead of ``>= 0``.
    """

    def __init__(self, step: float = 0.25) -> None:
        self.step = step
        self.now = 0.0
        self.readings = 0

    def __call__(self) -> float:
        self.readings += 1
        self.now += self.step
        return self.now


CHAPTERS = {
    "01-prva.md": (
        "## GLAVA PRVA\n\n"
        "Marko je otišao u Novi Sad. On je bio tamo. Kratko je.\n"
    ),
    # The capital has to be past the first word: `Beograd je grad.` would be
    # dropped by the no-uppercase filter, because the capital is just sentence case.
    "02-druga.md": "U Beogradu je veliki grad. Tamo sada živi Ana.\n",
}

#: Two chapters whose sentences re-use their nouns, which is what the second
#: stage-3 filter is for. Chapter 2's first sentence carries only ``Marko`` — new
#: inside that chapter, already seen across the book — so it is the sharpest test
#: that the rule is book-wide rather than per chapter.
REPEATED = {
    "01-prva.md": (
        "## GLAVA PRVA\n\n"
        "Marko je otišao u Novi Sad. Novi Sad je bio daleko.\n"
    ),
    "02-druga.md": "Tamo je Marko stigao. Ana je čekala Marka u Beogradu.\n",
}


def make_chapters(tmp_path: Path, files: dict[str, str] | None = None, name: str = "01-chapters") -> Path:
    """A minimal stage-1 folder: chapter files plus a manifest."""
    folder = tmp_path / name
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, (filename, text) in enumerate((files or CHAPTERS).items(), start=1):
        (folder / filename).write_text(text, encoding="utf-8")
        entries.append(
            {
                "filename": filename,
                "start_line": index,
                "end_line": index,
                "title": filename,
                "lines": len(text.splitlines()),
            }
        )
    (folder / "_manifest.json").write_text(
        json.dumps(
            {"source_file": "book.md", "strategy": "preset_map", "files": entries},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return folder


# --------------------------------------------------------------------------- #
# the funnel over one text
# --------------------------------------------------------------------------- #
def test_reduce_text_counts_every_reason_and_keeps_the_raw_sentences():
    backend = FakeNer()
    text = (
        "## GLAVA PRVA\n\n"
        "Marko je otišao u Novi Sad. "  # kept: PER + LOC
        "On je bio tamo. "  # no capital past the first word
        "Kratko je. "  # no capital past the first word either
        "U Beogradu je veliki grad. "  # kept: LOC
        "Sreli smo se u Subotici.\n"  # capital past the first word, but no known entity
    )
    reduction = reduce_text(text, backend=backend)

    assert reduction.kept == [
        "Marko je otišao u Novi Sad.",
        "U Beogradu je veliki grad.",
    ]
    assert reduction.source_sentences == 6
    assert reduction.dropped_short == 1  # the heading
    assert reduction.dropped_no_uppercase == 2
    assert reduction.dropped_no_entity == 1
    assert reduction.entities == {"LOC": 2, "PER": 1}
    assert [mention["text"] for mention in reduction.entity_mentions] == [
        "Novi Sad",
        "Marko",
        "Beograd",
    ]
    # Only the survivors of the cheap filters reach the tagger.
    assert backend.calls == [
        "Marko je otišao u Novi Sad.",
        "U Beogradu je veliki grad.",
        "Sreli smo se u Subotici.",
    ]


def test_reduce_text_caps_the_examples_it_keeps_for_the_trace():
    text = " ".join(f"On je bio tamo {n}." for n in range(10))
    reduction = reduce_text(text, backend=FakeNer(), example_limit=2)
    assert len(reduction.examples["no_uppercase"]) == 2


def test_reduce_text_times_only_the_tagger():
    """``ner_seconds`` counts the tagger, not the splitting or the filtering."""
    clock = SteppingClock(0.25)
    text = (
        "## GLAVA PRVA\n\n"
        "Marko je otišao u Novi Sad. "  # reaches the tagger
        "On je bio tamo. "  # filtered out before it
        "U Beogradu je veliki grad.\n"  # reaches the tagger
    )
    reduction = reduce_text(text, backend=FakeNer(), clock=clock)

    assert reduction.ner_calls == 2  # only the two survivors were tagged
    assert reduction.ner_seconds == pytest.approx(0.5)  # one step per tagger call
    assert reduction.milliseconds_per_sentence == pytest.approx(250.0)

    detail = reduction.to_dict()
    assert detail["ner_calls"] == 2
    assert detail["milliseconds_per_sentence"] == pytest.approx(250.0)


def test_the_ner_time_is_reported_per_book_and_per_chapter(tmp_path):
    chapters = make_chapters(tmp_path)
    clock = SteppingClock(0.1)
    result = run_dictionary(
        str(chapters), backend=FakeNer(), language="sr", clock=clock
    )

    # Chapter 1 sends 1 sentence to the tagger, chapter 2 sends 2.
    assert result.totals["sentences_tagged"] == 3
    assert result.timing["ner_seconds"] == pytest.approx(0.3)
    assert result.timing["milliseconds_per_sentence"] == pytest.approx(100.0)
    assert result.timing["sentences_per_second"] == pytest.approx(10.0)
    assert result.timing["run_seconds"] > 0
    assert result.timing["backend_load_seconds"] == 0.0  # an injected tagger loads nothing

    per_chapter = {c["filename"]: c for c in result.chapters}
    assert per_chapter["01-prva.md"]["ner_calls"] == 1
    assert per_chapter["02-druga.md"]["ner_calls"] == 2
    assert per_chapter["01-prva.md"]["ner_seconds"] == pytest.approx(0.1)
    assert per_chapter["01-prva.md"]["elapsed_seconds"] > 0

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["timing"]["ner_seconds"] == pytest.approx(0.3)
    assert manifest["totals"]["sentences_tagged"] == 3
    assert manifest["files"][0]["ner_seconds"] == pytest.approx(0.1)


def test_a_real_clock_measures_a_positive_ner_time(tmp_path):
    chapters = make_chapters(tmp_path)
    result = run_dictionary(str(chapters), backend=FakeNer(), language="sr")
    assert result.timing["ner_seconds"] >= 0.0
    assert result.totals["sentences_tagged"] == 3
    assert result.timing["run_seconds"] > 0.0


def test_reduce_text_can_drop_temporal_only_sentences():
    backend = FakeNer(table={"1996": "MISC", "Marko": "PER"})
    reduction = reduce_text(
        "Sreli smo se u Subotici 1996. godine. Došao je Marko 1996. godine.",
        backend=backend,
        drop_temporal_only=True,
    )
    assert reduction.kept == ["Došao je Marko 1996. godine."]
    assert reduction.dropped_temporal == 1


# --------------------------------------------------------------------------- #
# the whole stage over a folder
# --------------------------------------------------------------------------- #
def test_run_dictionary_writes_one_reduced_file_per_chapter(tmp_path):
    chapters = make_chapters(tmp_path)
    backend = FakeNer()
    seen: list[tuple[str, int, int]] = []

    result = run_dictionary(
        str(chapters),
        backend=backend,
        language="sr",
        progress=lambda chapter, done, total: seen.append((chapter["filename"], done, total)),
    )

    reduced = tmp_path / "03-reduced"
    assert result.ok is True
    assert result.chapter_count == 2
    assert result.processed_count == 2
    assert result.skipped is False
    assert result.backend == "fake"
    assert (reduced / "01-prva.md").read_text(encoding="utf-8") == (
        "Marko je otišao u Novi Sad.\n"
    )
    assert (reduced / "02-druga.md").read_text(encoding="utf-8") == (
        "U Beogradu je veliki grad.\nTamo sada živi Ana.\n"
    )
    # Chapters are processed first to last.
    assert backend.calls[0] == "Marko je otišao u Novi Sad."
    assert backend.calls[-1] == "Tamo sada živi Ana."
    # The progress hook reports completion order and the total.
    assert [done for _, done, _ in seen] == [1, 2]
    assert {total for _, _, total in seen} == {2}
    assert result.totals["kept"] == 3
    assert result.entities == {"LOC": 2, "PER": 2}


def test_the_manifest_carries_the_counts_the_next_stage_needs(tmp_path):
    chapters = make_chapters(tmp_path)
    result = run_dictionary(str(chapters), backend=FakeNer(), language="sr")

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["stage"] == "dictionary"
    assert manifest["language"]["language"] == "sr"
    assert manifest["backend"] == "fake"
    assert manifest["chapter_count"] == 2
    assert manifest["totals"]["sentences"] > manifest["totals"]["kept"]
    assert manifest["entities"] == {"LOC": 2, "PER": 2}

    first = manifest["files"][0]
    assert first["filename"] == "01-prva.md"
    assert first["kept"] == 1
    assert first["lines"] == 1  # the reader the later stages share
    assert first["source_lines"] == 3
    assert first["sha256"]
    assert first["start_line"] == 1

    trace = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
    assert trace["chapter_count"] == 2
    assert trace["detail"][0]["examples"]["short"]  # the dropped heading is on record


def test_skip_if_present_reuses_the_files_and_builds_no_tagger(tmp_path):
    chapters = make_chapters(tmp_path)
    run_dictionary(str(chapters), backend=FakeNer(), language="sr")

    second = FakeNer()
    result = run_dictionary(
        str(chapters), backend=second, language="sr", skip_if_present=True
    )

    assert result.ok is True
    assert result.skipped is True
    assert result.skipped_count == 2
    assert second.calls == []  # nothing was tagged again
    assert result.chapters[0]["skipped"] is True
    assert (tmp_path / "03-reduced" / "01-prva.md").is_file()
    # The previous run's own record is carried over instead of re-detecting.
    assert result.language["language"] == "sr"
    assert result.backend == "fake"


def test_skip_if_present_rebuilds_when_the_chapter_folder_changes(tmp_path):
    """Raw and cleaned chapters share filenames but not text, so work is redone."""
    raw = make_chapters(tmp_path)
    run_dictionary(str(raw), backend=FakeNer(), language="sr")

    cleaned = make_chapters(tmp_path, name="02-preprocessed")
    (cleaned / "01-prva.md").write_text(
        "## GLAVA PRVA\n\nMarko je otišao kući.\n", encoding="utf-8"
    )
    second = FakeNer()
    result = run_dictionary(
        str(cleaned), backend=second, language="sr", skip_if_present=True
    )

    assert result.ok is True
    assert result.skipped is False
    assert second.calls  # the reduction ran again, from the cleaned text
    assert (tmp_path / "03-reduced" / "01-prva.md").is_file()


def test_a_fully_reduced_book_needs_no_detector_either(tmp_path):
    """``--skip-if-present`` must not need lingua-py once everything is reduced."""
    chapters = make_chapters(tmp_path)
    run_dictionary(str(chapters), backend=FakeNer(), language="sr")

    class ExplodingDetector:
        def detect(self, text):  # pragma: no cover - must never be called
            raise AssertionError("the detector must not run when nothing is left to do")

    result = run_dictionary(
        str(chapters),
        language="auto",
        detector=ExplodingDetector(),
        skip_if_present=True,
    )

    assert result.ok is True
    assert result.skipped is True
    assert result.language["language"] == "sr"


def test_skip_if_present_reduces_only_the_chapters_that_are_missing(tmp_path):
    chapters = make_chapters(tmp_path)
    run_dictionary(str(chapters), backend=FakeNer(), language="sr")
    (tmp_path / "03-reduced" / "02-druga.md").unlink()

    result = run_dictionary(
        str(chapters), backend=FakeNer(), language="sr", skip_if_present=True
    )

    assert result.skipped is False
    assert result.processed_count == 1
    assert result.skipped_count == 1
    assert (tmp_path / "03-reduced" / "02-druga.md").is_file()
    assert result.totals["kept"] == 2  # both of chapter 2's sentences survive


def test_save_reduced_can_be_switched_off(tmp_path):
    chapters = make_chapters(tmp_path)
    result = run_dictionary(
        str(chapters),
        str(tmp_path / "03-reduced"),
        backend=FakeNer(),
        language="sr",
        save_reduced=False,
    )

    assert result.ok
    assert not (tmp_path / "03-reduced" / "01-prva.md").exists()
    assert Path(result.manifest_path).is_file()  # the record still lands


def test_a_dry_run_writes_nothing_but_still_counts(tmp_path):
    chapters = make_chapters(tmp_path)
    out = tmp_path / "03-reduced"
    result = run_dictionary(
        str(chapters), str(out), backend=FakeNer(), language="sr", dry_run=True
    )

    assert result.ok and result.dry_run is True
    assert not out.exists()
    assert result.manifest_path is None
    assert result.totals["kept"] == 3


def test_limit_reduces_only_the_first_chapters(tmp_path):
    chapters = make_chapters(tmp_path)
    backend = FakeNer()
    result = run_dictionary(str(chapters), backend=backend, language="sr", limit=1)

    assert result.chapter_count == 1
    assert not (tmp_path / "03-reduced" / "02-druga.md").exists()


def test_write_manifest_can_be_switched_off(tmp_path):
    chapters = make_chapters(tmp_path)
    result = run_dictionary(
        str(chapters), backend=FakeNer(), language="sr", write_manifest=False
    )
    assert result.manifest_path is None
    assert (tmp_path / "03-reduced" / "01-prva.md").is_file()


def test_an_explicit_chapters_output_folder_is_respected(tmp_path):
    chapters = make_chapters(tmp_path)
    elsewhere = tmp_path / "somewhere" / "else"
    result = run_dictionary(str(chapters), str(elsewhere), backend=FakeNer(), language="sr")
    assert Path(result.output_dir) == elsewhere
    assert (elsewhere / "01-prva.md").is_file()


# --------------------------------------------------------------------------- #
# language detection and routing
# --------------------------------------------------------------------------- #
def test_the_detected_language_routes_to_a_backend(tmp_path, monkeypatch):
    chapters = make_chapters(tmp_path)
    detector = FakeDetector("sr")
    captured: dict[str, object] = {}

    def fake_make_backend(language, **kwargs):
        captured["language"] = language
        captured.update(kwargs)
        return FakeNer(language=language)

    monkeypatch.setattr(runner_module, "make_backend", fake_make_backend)

    result = run_dictionary(str(chapters), language="auto", detector=detector)

    assert result.language["method"] == language_module.METHOD_LINGUA
    assert result.language["language"] == "sr"
    assert captured["language"] == "sr"
    assert captured["backend"] == "classla"
    assert result.backend == "fake"
    # Every chapter is sampled; the manifest is bookkeeping, not prose.
    assert len(detector.calls) == 2
    assert "_manifest" not in "".join(detector.calls)


def test_an_explicit_language_skips_detection_entirely(tmp_path, monkeypatch):
    chapters = make_chapters(tmp_path)
    detector = FakeDetector("sr")
    monkeypatch.setattr(runner_module, "make_backend", lambda language, **kw: FakeNer(language))

    result = run_dictionary(str(chapters), language="en", detector=detector)

    assert result.language == {
        "language": "en",
        "confidence": 1.0,
        "method": "explicit",
        "samples": [],
        "note": "",
        "reflex": {},
    }
    assert detector.calls == []


def test_english_routes_to_the_stanza_backend():
    assert backend_for_language("en") == "stanza"
    assert backend_for_language("sr") == "classla"
    assert backend_for_language("hr") == "classla"


# --------------------------------------------------------------------------- #
# the ekavian/ijekavian tie-breaker: lingua calls Serbian "hr" with confidence 1.0
# --------------------------------------------------------------------------- #
EKAVIAN = (
    "U vreme dok je dete spavalo, reč po reč, mesto po mesto, smeh i pesma su "
    "ispunjavali celu kuću, i svaki deo te priče je ostao u sećanju."
)
IJEKAVIAN = (
    "U vrijeme dok je dijete spavalo, riječ po riječ, mjesto po mjesto, smijeh i "
    "pjesma su ispunjavali cijelu kuću, i svaki dio te priče je ostao u sjećanju."
)


def test_the_reflex_counter_sees_the_spelling():
    from book_splitter.dictionary.language import count_reflexes

    assert count_reflexes(EKAVIAN)["ekavian"] > 5
    assert count_reflexes(EKAVIAN)["ijekavian"] == 0
    assert count_reflexes(IJEKAVIAN)["ijekavian"] > 5
    assert count_reflexes(IJEKAVIAN)["ekavian"] == 0


def test_an_ekavian_book_is_read_as_serbian_whatever_lingua_voted():
    from book_splitter.dictionary.language import resolve_south_slavic

    language, note = resolve_south_slavic("hr", EKAVIAN)
    assert language == "sr"
    assert "ekavian" in note and "hr" in note
    # Bosnian votes are questioned the same way.
    assert resolve_south_slavic("bs", EKAVIAN)[0] == "sr"


def test_an_ijekavian_book_voted_serbian_is_read_as_croatian():
    from book_splitter.dictionary.language import resolve_south_slavic

    language, note = resolve_south_slavic("sr", IJEKAVIAN)
    assert language == "hr"
    assert "ijekavian" in note


def test_the_tie_breaker_stays_out_of_ambiguous_or_thin_evidence():
    from book_splitter.dictionary.language import resolve_south_slavic

    # Too few forms to be sure.
    thin = "Bilo je lepo."
    assert resolve_south_slavic("hr", thin) == ("hr", "")
    # Both spellings present in comparable numbers: a mixed book keeps the vote.
    mixed = f"{EKAVIAN} {IJEKAVIAN}"
    assert resolve_south_slavic("hr", mixed)[0] == "hr"
    # English is never touched.
    assert resolve_south_slavic("en", EKAVIAN) == ("en", "")


def test_the_guess_reports_that_it_overruled_the_detector(tmp_path, monkeypatch):
    chapters = make_chapters(
        tmp_path,
        {"01-a.md": f"## GLAVA\n\n{EKAVIAN}\n"},
    )
    monkeypatch.setattr(runner_module, "make_backend", lambda language, **kw: FakeNer(language))

    result = run_dictionary(str(chapters), language="auto", detector=FakeDetector("hr"))

    assert result.language["language"] == "sr"
    assert result.language["method"] == language_module.METHOD_LINGUA_REFLEX
    assert "overruled" in result.language["note"]
    assert result.language["reflex"]["ekavian"] > 5
    assert result.backend == "fake"
    # The overruled vote is still on record.
    assert result.language["samples"][0]["language"] == "hr"


def test_languages_without_a_backend_are_rejected_clearly():
    with pytest.raises(ToolError, match="no NER model"):
        backend_for_language("mk")  # CLASSLA ships no NER model for Macedonian
    with pytest.raises(ToolError, match="no NER backend"):
        backend_for_language("fr")
    with pytest.raises(ToolError, match="unknown NER backend"):
        make_backend("en", backend="magic")


def test_an_undetectable_language_fails_before_any_file_is_written(tmp_path):
    chapters = make_chapters(tmp_path)
    with pytest.raises(ToolError, match="no NER backend"):
        run_dictionary(str(chapters), language="auto", detector=FakeDetector("fr"))
    assert not (tmp_path / "03-reduced").exists()


def test_a_missing_tagger_is_reported_as_a_missing_dependency():
    if importlib.util.find_spec("classla") is not None:  # pragma: no cover - env dependent
        pytest.skip("classla is installed here; the missing-dependency path cannot be hit")
    with pytest.raises(MissingDependencyError, match="pip install"):
        make_backend("sr")


def test_a_missing_language_detector_is_reported_as_a_missing_dependency():
    if importlib.util.find_spec("lingua") is not None:  # pragma: no cover - env dependent
        pytest.skip("lingua is installed here; the missing-dependency path cannot be hit")
    with pytest.raises(MissingDependencyError, match="lingua"):
        language_module.LinguaDetector()


def test_lingua_members_handles_both_enum_styles():
    """lingua 2.x (PyO3, `Language.all()`) and 1.x (a plain Python enum)."""
    import enum

    from book_splitter.dictionary.language import _lingua_members

    class Legacy(enum.Enum):
        ENGLISH = "EN"
        SERBIAN = "SR"

    class PyO3:
        class _Member:
            def __init__(self, name):
                self.iso_code_639_1 = type("Iso", (), {"name": name})

        @staticmethod
        def all():
            return [PyO3._Member("EN"), PyO3._Member("SR")]

    assert len(_lingua_members(Legacy)) == 2
    codes = [m.iso_code_639_1.name for m in _lingua_members(PyO3)]
    assert codes == ["EN", "SR"]


def test_missing_model_files_are_recognised_so_they_can_be_downloaded():
    """The first run on a new machine must download, not crash.

    The message is the one classla actually raises when its resources directory
    holds no ``resources.json`` (verified against classla 2.2.3).
    """
    from book_splitter.dictionary.ner import _looks_like_missing_models

    real = Exception(
        "Resources file not found at: /tmp/x/resources.json. Try to download the model again."
    )
    assert _looks_like_missing_models(real) is True
    assert _looks_like_missing_models(FileNotFoundError("no such file: model.pt")) is True
    # A genuinely broken install must be reported, not papered over with a download.
    assert (
        _looks_like_missing_models(Exception("No processor to load. Please check your language."))
        is False
    )


def test_stanza_ontonotes_labels_are_folded_onto_the_four_types():
    """stanza says PERSON/GPE/FAC; the gate only knows PER/LOC/ORG/MISC.

    Without this fold every person and city falls through the gate as an unknown
    type. That is exactly what happened to the first English book that ran:
    13 384 sentences in, 244 kept, and not one PER in the whole novel.
    """
    from book_splitter.dictionary.ner import (
        _STANZA_TO_CLASSLA,
        _document_entities,
        gate_entities,
    )

    class Ent:
        def __init__(self, text, type):
            self.text, self.type = text, type

    class Sent:
        def __init__(self, ents):
            self.ents = ents

    class Doc:
        sentences = [
            Sent(
                [
                    Ent("Theodorus Van Gogh", "PERSON"),
                    Ent("Breda", "FAC"),
                    Ent("London", "GPE"),
                    Ent("Goupil Galleries", "ORG"),
                    Ent("Dutch", "NORP"),
                    Ent("the Thames", "LOC"),
                    Ent("1880", "DATE"),
                ]
            )
        ]
        ents = []

    entities = _document_entities(Doc(), type_map=_STANZA_TO_CLASSLA)
    assert [(e.text, e.type) for e in entities] == [
        ("Theodorus Van Gogh", "PER"),
        ("Breda", "LOC"),
        ("London", "LOC"),
        ("Goupil Galleries", "ORG"),
        ("Dutch", "MISC"),
        ("the Thames", "LOC"),
        ("1880", "DATE"),  # unmapped: a date must not save a sentence
    ]
    decision = gate_entities(entities)
    assert decision.keep is True
    assert "PER" in decision.types


def test_the_stanza_backend_folds_its_labels_end_to_end(monkeypatch):
    """The regression, through the real backend with a fake stanza underneath."""
    import sys
    import types

    from book_splitter.dictionary.ner import StanzaBackend

    class Ent:
        def __init__(self, text, type):
            self.text, self.type = text, type

    class Doc:
        def __init__(self):
            self.sentences = [
                types.SimpleNamespace(
                    ents=[Ent("Theodorus Van Gogh", "PERSON"), Ent("Breda", "FAC")]
                )
            ]
            self.ents = []

    class FakePipeline:
        device = "cpu"

        def __call__(self, text):
            return Doc()

    class FakeStanza:
        def Pipeline(self, **kwargs):  # noqa: N802 - mirrors the library
            return FakePipeline()

    monkeypatch.setitem(sys.modules, "stanza", FakeStanza())
    backend = StanzaBackend("en", use_gpu=False)

    entities = backend.entities("Theodorus Van Gogh met his son at the Breda station.")
    assert [(e.text, e.type) for e in entities] == [
        ("Theodorus Van Gogh", "PER"),
        ("Breda", "LOC"),
    ]
    assert ner_module.gate_entities(entities).keep is True
    assert ner_module.backend_device(backend) == "cpu"


def test_the_device_can_come_from_a_device_string(monkeypatch):
    """stanza >= 1.5 reports ``device`` instead of a ``use_gpu`` flag."""
    from book_splitter.dictionary.ner import backend_device

    class Backend:
        def __init__(self, device):
            self._pipeline = type("P", (), {"device": device})()

    assert backend_device(Backend("cpu")) == "cpu"
    assert backend_device(Backend("cuda")) == "cuda"
    assert backend_device(Backend("cuda:1")) == "cuda"
    assert backend_device(Backend("")) == ""


def test_the_first_run_downloads_only_the_processors_it_will_load(monkeypatch):
    """No POS/lemma/depparse/pretrain: stage 3 loads tokenize + ner only."""
    import sys
    import types

    from book_splitter.dictionary.ner import ClasslaBackend

    calls: dict[str, object] = {}

    class FakePipeline:
        use_gpu = False

        def __call__(self, text):
            return types.SimpleNamespace(sentences=[])

    class FakeClassla:
        def __init__(self) -> None:
            self.builds = 0

        def Pipeline(self, language, **kwargs):  # noqa: N802 - mirrors the library
            self.builds += 1
            if self.builds == 1:
                raise Exception("Resources file not found at: /x/resources.json")
            calls["pipeline"] = kwargs
            return FakePipeline()

        def download(self, language, **kwargs):
            calls["download"] = {"language": language, **kwargs}

    monkeypatch.setitem(sys.modules, "classla", FakeClassla())
    backend = ClasslaBackend("sr", use_gpu=False)

    assert calls["download"] == {"language": "sr", "processors": "tokenize,ner"}
    assert calls["pipeline"]["processors"] == "tokenize,ner"
    assert calls["pipeline"]["use_gpu"] is False
    assert backend.entities("Marko je došao.") == []  # the fake document has no entities
    assert ner_module.backend_device(backend) == "cpu"


# --------------------------------------------------------------------------- #
# GPU selection
# --------------------------------------------------------------------------- #
def test_the_default_gpu_is_gpu_one(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert configure_gpu() == 1
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_an_explicit_gpu_wins(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    assert configure_gpu(0) == 0
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_an_exported_gpu_is_respected_when_none_is_asked_for(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    assert configure_gpu() == 2
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"


def test_cpu_mode_clears_the_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert configure_gpu(use_gpu=False) is None
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_run_dictionary_leaves_cuda_untouched_when_a_tagger_is_injected(tmp_path, monkeypatch):
    """An injected backend needs no device decision at all."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    chapters = make_chapters(tmp_path)
    run_dictionary(str(chapters), backend=FakeNer(), language="sr")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "7"
    assert ner_module.DEFAULT_GPU_INDEX == 1


def test_the_reported_device_is_the_one_that_actually_ran():
    """``--gpu 1`` on a box without a reachable driver is a CPU run, and says so."""
    from book_splitter.dictionary.ner import backend_device

    class Pipeline:
        def __init__(self, use_gpu):
            self.use_gpu = use_gpu

    class Backend:
        _pipeline = Pipeline(False)

    assert backend_device(Backend()) == "cpu"
    Backend._pipeline = Pipeline(True)
    assert backend_device(Backend()) == "cuda"
    # A backend that does not expose a device (an injected test double) says nothing.
    assert backend_device(FakeNer()) == ""


def test_the_cli_reports_the_device_and_not_just_the_requested_gpu(tmp_path, monkeypatch, capsys):
    from book_splitter import cli

    chapters = make_chapters(tmp_path)
    calls = _stub_cli_dictionary(monkeypatch)

    code = cli.main(["dictionary", "--out", str(chapters), "--gpu", "1"])
    out = capsys.readouterr().out

    assert code == 0
    assert calls[0]["gpu"] == 1
    assert "backend: classla on cuda" in out
    assert "model load      : 8.25 s (classla, cuda)" in out


# --------------------------------------------------------------------------- #
# the CLI
# --------------------------------------------------------------------------- #
def _stub_cli_dictionary(
    monkeypatch,
    *,
    ok: bool = True,
    skipped: bool = False,
    error: Exception | None = None,
):
    """Replace ``book_splitter.dictionary.run_dictionary`` for a CLI test."""
    import book_splitter.dictionary as dictionary_package
    from book_splitter.dictionary.runner import DictionaryResult

    calls: list[dict] = []

    def fake_run_dictionary(chapters_path, output_dir=None, **kwargs):
        calls.append({"chapters_path": chapters_path, "output_dir": output_dir, **kwargs})
        if error is not None:
            raise error
        reduced = Path(output_dir or chapters_path)
        return DictionaryResult(
            ok=ok,
            chapters_path=str(chapters_path),
            output_dir=str(output_dir or chapters_path),
            unique_dir=str(reduced.parent / "03-reduced-unique"),
            language={"language": "sr", "method": "lingua", "confidence": 0.93, "samples": []},
            backend="classla",
            chapter_count=2,
            ok_count=2 if ok else 1,
            processed_count=0 if skipped else 2,
            skipped=skipped,
            skipped_count=2 if skipped else 0,
            save_reduced=kwargs.get("save_reduced", True),
            save_unique=kwargs.get("save_unique", True),
            totals={
                "sentences": 40,
                "kept": 12,
                "sentences_tagged": 20,
                "dropped_short": 3,
                "dropped_no_uppercase": 20,
                "dropped_no_entity": 5,
            },
            unique_totals={
                "chapters": 2,
                "sentences": 12,
                "kept": 5,
                "dropped_duplicate_noun": 7,
                "new_nouns": 4,
                "nouns": 9,
                "output_lines": 5,
                "output_bytes": 120,
            },
            timing={
                "backend_load_seconds": 8.25,
                "ner_seconds": 4.0,
                "chapter_seconds": 5.5,
                "run_seconds": 14.0,
                "seconds_per_sentence": 0.2,
                "milliseconds_per_sentence": 200.0,
                "sentences_per_second": 5.0,
            },
            gpu=1,
            device="cuda",
            entities={"PER": 7, "LOC": 5},
            errors=[] if ok else ["02-a.md: the tagger crashed"],
            manifest_path=(
                str(Path(output_dir or chapters_path) / "_reduced_manifest.json") if ok else None
            ),
        )

    monkeypatch.setattr(dictionary_package, "run_dictionary", fake_run_dictionary)
    return calls


def test_cli_dictionary_reduces_a_book(tmp_path, monkeypatch, capsys):
    from book_splitter import cli

    chapters = make_chapters(tmp_path)
    calls = _stub_cli_dictionary(monkeypatch)

    code = cli.main(
        [
            "dictionary",
            "--out",
            str(chapters),
            "--language",
            "sr",
            "--keep-types",
            "loc, per",
            "--min-words",
            "5",
            "--limit",
            "1",
        ]
    )
    out = capsys.readouterr().out

    assert code == 0
    assert len(calls) == 1
    call = calls[0]
    assert call["language"] == "sr"
    assert call["keep_types"] == ("LOC", "PER")
    assert call["min_words"] == 5
    assert call["limit"] == 1
    assert call["require_uppercase"] is True
    assert call["save_reduced"] is True
    assert call["save_unique"] is True
    # Default output is a sibling of the chapters folder.
    assert Path(call["output_dir"]) == chapters.parent / "03-reduced"
    assert Path(call["unique_output_dir"]) == chapters.parent / "03-reduced-unique"
    assert "language        : sr" in out
    assert "sentences       : 40 -> 12 kept" in out
    assert "first mentions  : 12 -> 5 kept (duplicate-noun=7, nouns=9)" in out
    assert "entities        : PER=7, LOC=5" in out
    # The NER stage's own cost is reported, separately from the model load.
    assert "ner time        : 4.00 s for 20 sentence(s)" in out
    assert "200.0 ms each" in out
    assert "model load      : 8.25 s (classla, cuda)" in out
    assert "total time      : 14.00 s" in out
    assert "unique dir      :" in out and "reduced-unique" in out
    assert "OK — reduced chapters written." in out
    assert "OK — first-mention chapters written." in out


def test_cli_dictionary_can_be_pointed_at_the_preprocessed_folder(tmp_path, monkeypatch):
    from book_splitter import cli

    preprocessed = make_chapters(tmp_path, name="preprocessed")
    elsewhere = tmp_path / "out" / "03-reduced"
    calls = _stub_cli_dictionary(monkeypatch)

    code = cli.main(
        ["dictionary", "--out", str(preprocessed), "--reduced-out", str(elsewhere), "--quiet"]
    )

    assert code == 0
    assert Path(calls[0]["output_dir"]) == elsewhere
    # The first-mention folder follows the reduced one, so the two stay together.
    assert Path(calls[0]["unique_output_dir"]) == elsewhere.parent / "03-reduced-unique"


def test_cli_dictionary_accepts_the_book_folder_not_just_chapters(tmp_path, monkeypatch):
    """`--out data/output/<book>` is what people type; `scan`/`preprocess` accept it."""
    from book_splitter import cli

    make_chapters(tmp_path)  # tmp_path/chapters
    calls = _stub_cli_dictionary(monkeypatch)

    code = cli.main(["dictionary", "--out", str(tmp_path), "--quiet"])

    assert code == 0
    assert Path(calls[0]["chapters_path"]) == tmp_path / "01-chapters"
    assert Path(calls[0]["output_dir"]) == tmp_path / "03-reduced"


def test_a_tagger_that_cannot_load_is_a_clean_error(tmp_path, monkeypatch):
    """Read-only cache, no disk space, dropped download — not a traceback."""
    chapters = make_chapters(tmp_path)

    def boom(language, **kwargs):
        raise OSError("[Errno 30] Read-only file system: '/home/x/.cache/stanza'")

    monkeypatch.setattr(runner_module, "make_backend", boom)

    with pytest.raises(ToolError, match="could not load the stanza models for 'en'"):
        run_dictionary(str(chapters), language="en")
    # Nothing was created on the way out.
    assert not (tmp_path / "03-reduced").exists()


def test_the_cli_prints_one_line_for_a_tagger_that_cannot_load(tmp_path, monkeypatch, capsys):
    from book_splitter import cli

    chapters = make_chapters(tmp_path)

    def boom(language, **kwargs):
        raise OSError("Read-only file system: '/home/x/.cache/stanza'")

    monkeypatch.setattr(runner_module, "make_backend", boom)

    code = cli.main(["dictionary", "--out", str(chapters), "--language", "en"])
    err = capsys.readouterr().err

    assert code == 2
    assert err.count("\n") == 1  # one line, no traceback
    assert "could not load the stanza models" in err
    assert "Read-only file system" in err


def test_cli_dictionary_reports_a_missing_tagger_as_a_dependency_error(
    tmp_path, monkeypatch, capsys
):
    from book_splitter import cli

    chapters = make_chapters(tmp_path)
    _stub_cli_dictionary(
        monkeypatch, error=MissingDependencyError("pip install -r requirements-dictionary.txt")
    )

    code = cli.main(["dictionary", "--out", str(chapters)])
    err = capsys.readouterr().err

    assert code == 2
    assert "missing dependency" in err
    assert "requirements-dictionary.txt" in err


def test_cli_dictionary_refuses_a_folder_without_chapters(tmp_path, capsys):
    from book_splitter import cli

    empty = tmp_path / "nothing"
    empty.mkdir()

    code = cli.main(["dictionary", "--out", str(empty)])
    err = capsys.readouterr().err

    assert code == 2
    assert "no chapter files" in err
    assert "run `split` first" in err


def test_cli_dictionary_turns_off_the_uppercase_filter(tmp_path, monkeypatch):
    from book_splitter import cli

    chapters = make_chapters(tmp_path)
    calls = _stub_cli_dictionary(monkeypatch)

    code = cli.main(
        [
            "dictionary",
            "--out",
            str(chapters),
            "--no-uppercase-filter",
            "--no-save-reduced",
            "--quiet",
        ]
    )

    assert code == 0
    assert calls[0]["require_uppercase"] is False
    assert calls[0]["save_reduced"] is False
    assert calls[0]["save_unique"] is True  # only the reduced chapters were switched off


# --------------------------------------------------------------------------- #
# the first-mention folder: the same pass, a smaller output
# --------------------------------------------------------------------------- #
def test_both_folders_are_written_and_the_second_is_the_first_mentions(tmp_path):
    chapters = make_chapters(tmp_path, REPEATED)
    result = run_dictionary(str(chapters), backend=FakeNer(), language="sr")

    reduced = tmp_path / "03-reduced"
    unique = tmp_path / "03-reduced-unique"
    assert result.save_unique is True
    assert Path(result.unique_dir) == unique

    # Chapter 1 keeps two sentences, but the second says nothing new.
    assert (reduced / "01-prva.md").read_text(encoding="utf-8") == (
        "Marko je otišao u Novi Sad.\nNovi Sad je bio daleko.\n"
    )
    assert (unique / "01-prva.md").read_text(encoding="utf-8") == (
        "Marko je otišao u Novi Sad.\n"
    )
    # Chapter 2 keeps two, but `Tamo je Marko stigao.` only repeats a book-wide noun.
    assert (reduced / "02-druga.md").read_text(encoding="utf-8") == (
        "Tamo je Marko stigao.\nAna je čekala Marka u Beogradu.\n"
    )
    assert (unique / "02-druga.md").read_text(encoding="utf-8") == (
        "Ana je čekala Marka u Beogradu.\n"
    )

    assert result.totals["kept"] == 4  # the reduced stage is unchanged
    assert result.unique_totals == {
        "chapters": 2,
        "sentences": 4,
        "kept": 2,
        "dropped_duplicate_noun": 2,
        "new_nouns": 4,  # marko + novi sad, then ana + beograd
        "nouns": 4,  # marko, novi sad, ana, beograd
        "output_lines": 2,
        "output_bytes": len(
            "Marko je otišao u Novi Sad.\nAna je čekala Marka u Beogradu.\n".encode()
        ),
    }


def test_the_book_wide_rule_carries_from_chapter_to_chapter(tmp_path):
    """Chapter 2's first sentence is new *within the chapter*, seen across the book."""
    chapters = make_chapters(tmp_path, REPEATED)
    seen: list[dict] = []
    result = run_dictionary(
        str(chapters),
        backend=FakeNer(),
        language="sr",
        progress=lambda chapter, done, total: seen.append(chapter),
    )

    assert [chapter["kept"] for chapter in seen] == [2, 2]
    assert [chapter["unique_kept"] for chapter in seen] == [1, 1]
    # Chapter 1 introduces two nouns; chapter 2 adds Ana and Beograd.
    assert [chapter["unique"]["new_nouns"] for chapter in seen] == [2, 2]
    assert result.chapters[0]["noun_keys"] == ["marko", "novi sad"]
    assert result.chapters[1]["noun_keys"] == ["ana", "beograd", "marko"]


def test_the_manifest_describes_both_folders_and_carries_the_noun_index(tmp_path):
    chapters = make_chapters(tmp_path, REPEATED)
    result = run_dictionary(str(chapters), backend=FakeNer(), language="sr")

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["save_unique"] is True
    assert Path(manifest["unique_dir"]) == tmp_path / "03-reduced-unique"
    assert manifest["unique_totals"]["kept"] == 2
    assert manifest["unique_totals"]["nouns"] == 4
    assert manifest["settings"]["save_unique"] is True

    first = manifest["files"][0]
    assert first["unique_lines"] == 1
    assert first["unique_kept"] == 1
    assert first["dropped_duplicate_noun"] == 1
    assert first["new_nouns"] == 2
    assert first["noun_keys"] == ["marko", "novi sad"]
    assert first["unique_sha256"]
    assert Path(first["unique_path"]).name == "01-prva.md"

    trace = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
    selection = trace["detail"][0]["unique"]
    assert selection["kept"] == 1
    assert selection["dropped"] == 1
    assert sorted(selection["new_keys"]) == ["marko", "novi sad"]
    assert selection["dropped_keys"] == [["novi sad"]]
    assert selection["examples"] == ["Novi Sad je bio daleko."]


def test_save_unique_can_be_switched_off(tmp_path):
    chapters = make_chapters(tmp_path, REPEATED)
    result = run_dictionary(
        str(chapters), backend=FakeNer(), language="sr", save_unique=False
    )

    assert result.save_unique is False
    assert not (tmp_path / "03-reduced-unique").exists()
    assert (tmp_path / "03-reduced" / "01-prva.md").is_file()
    assert result.unique_totals["kept"] == 0
    assert result.chapters[0]["unique_kept"] == 0


def test_an_explicit_first_mention_folder_is_respected(tmp_path):
    chapters = make_chapters(tmp_path, REPEATED)
    elsewhere = tmp_path / "somewhere" / "first-mentions"
    result = run_dictionary(
        str(chapters), backend=FakeNer(), language="sr", unique_output_dir=str(elsewhere)
    )

    assert Path(result.unique_dir) == elsewhere
    assert (elsewhere / "01-prva.md").is_file()
    assert not (tmp_path / "03-reduced-unique").exists()


def test_a_dry_run_writes_neither_folder_but_counts_both(tmp_path):
    chapters = make_chapters(tmp_path, REPEATED)
    result = run_dictionary(str(chapters), backend=FakeNer(), language="sr", dry_run=True)

    assert result.dry_run is True
    assert not (tmp_path / "03-reduced").exists()
    assert not (tmp_path / "03-reduced-unique").exists()
    assert result.totals["kept"] == 4
    assert result.unique_totals["kept"] == 2


def test_a_rerun_rebuilds_the_book_wide_set_from_the_noun_index(tmp_path):
    """The chapters a rerun reuses still suppress their nouns in the ones it runs."""
    chapters = make_chapters(tmp_path, REPEATED)
    run_dictionary(str(chapters), backend=FakeNer(), language="sr")
    (tmp_path / "03-reduced" / "02-druga.md").unlink()
    (tmp_path / "03-reduced-unique" / "02-druga.md").unlink()

    backend = FakeNer()
    result = run_dictionary(
        str(chapters), backend=backend, language="sr", skip_if_present=True
    )

    assert result.skipped is False
    assert result.processed_count == 1
    assert result.skipped_count == 1
    # Chapter 1 was never re-tagged: its nouns came from the manifest.
    assert backend.calls == ["Tamo je Marko stigao.", "Ana je čekala Marka u Beogradu."]
    assert (tmp_path / "03-reduced-unique" / "02-druga.md").read_text(encoding="utf-8") == (
        "Ana je čekala Marka u Beogradu.\n"
    )
    assert result.unique_totals["nouns"] == 4


def test_a_rerun_reuses_a_chapter_only_when_its_unique_file_is_there(tmp_path):
    chapters = make_chapters(tmp_path, REPEATED)
    run_dictionary(str(chapters), backend=FakeNer(), language="sr")
    (tmp_path / "03-reduced-unique" / "02-druga.md").unlink()

    result = run_dictionary(
        str(chapters), backend=FakeNer(), language="sr", skip_if_present=True
    )

    assert result.processed_count == 1
    assert result.skipped_count == 1
    assert (tmp_path / "03-reduced-unique" / "02-druga.md").is_file()


def test_a_fully_reduced_book_with_both_folders_still_needs_no_tagger(tmp_path):
    chapters = make_chapters(tmp_path, REPEATED)
    run_dictionary(str(chapters), backend=FakeNer(), language="sr")

    second = FakeNer()
    result = run_dictionary(
        str(chapters), backend=second, language="sr", skip_if_present=True
    )

    assert result.skipped is True
    assert second.calls == []
    assert result.save_unique is True
    assert Path(result.unique_dir) == tmp_path / "03-reduced-unique"
    assert result.chapters[0]["unique_lines"] == 1
    assert result.unique_totals["kept"] == 2  # carried over from the previous manifest


# --------------------------------------------------------------------------- #
# the CLI switches for the second folder
# --------------------------------------------------------------------------- #
def test_cli_dictionary_can_write_the_first_mentions_only(tmp_path, monkeypatch, capsys):
    from book_splitter import cli

    chapters = make_chapters(tmp_path)
    calls = _stub_cli_dictionary(monkeypatch)

    code = cli.main(["dictionary", "--out", str(chapters), "--unique-only"])
    out = capsys.readouterr().out

    assert code == 0
    assert calls[0]["save_reduced"] is False
    assert calls[0]["save_unique"] is True
    assert Path(calls[0]["unique_output_dir"]) == chapters.parent / "03-reduced-unique"
    assert "save         : first mentions only (--no-save-reduced)" in out
    assert "OK — first-mention chapters written." in out
    assert "OK — reduced chapters written." not in out


def test_cli_dictionary_can_skip_the_first_mentions(tmp_path, monkeypatch, capsys):
    from book_splitter import cli

    chapters = make_chapters(tmp_path)
    calls = _stub_cli_dictionary(monkeypatch)

    code = cli.main(["dictionary", "--out", str(chapters), "--no-save-unique"])
    out = capsys.readouterr().out

    assert code == 0
    assert calls[0]["save_unique"] is False
    assert calls[0]["save_reduced"] is True
    assert "save         : reduced chapters only (--no-save-unique)" in out
    assert "first mentions  :" not in out
    assert "unique dir      :" not in out


def test_cli_dictionary_respects_an_explicit_first_mention_folder(tmp_path, monkeypatch):
    from book_splitter import cli

    chapters = make_chapters(tmp_path)
    elsewhere = tmp_path / "out" / "mentions"
    calls = _stub_cli_dictionary(monkeypatch)

    code = cli.main(
        ["dictionary", "--out", str(chapters), "--unique-out", str(elsewhere), "--quiet"]
    )

    assert code == 0
    assert Path(calls[0]["unique_output_dir"]) == elsewhere
    assert calls[0]["save_unique"] is True


def test_cli_dictionary_says_nothing_was_written_when_it_reuses(tmp_path, monkeypatch, capsys):
    from book_splitter import cli

    chapters = make_chapters(tmp_path)
    _stub_cli_dictionary(monkeypatch, skipped=True)

    code = cli.main(["dictionary", "--out", str(chapters), "--skip-if-present"])
    out = capsys.readouterr().out

    assert code == 0
    assert "nothing to do   : every reduced chapter is already present (both folders)" in out
    assert "OK — nothing was written: the existing folders were reused." in out
    assert "OK — reduced chapters written." not in out
