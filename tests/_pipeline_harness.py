"""Deterministic, offline fixture-book harness for stages 0 through 11.

The harness runs the *real* stage runners and the real on-disk contracts. Only
the model boundary is doubled:

* stages 1, 2, 6b and 9 get a scripted chat model (``AIMessage`` tool calls);
* stage 3 and 6 get a dictionary-backed NER backend;
* stage 4 gets a scripted glossary extractor;
* stage 7 gets a scripted translator.

Stage 0 copies a checked-in TXT fixture, as it would copy any plain-text source.
Stages 5, 8, 10 and 11 are fully deterministic and run untouched.

The model doubles deliberately keep their answers tiny and fixed. The point of
this harness is not to test the models; it is to produce the same complete
artifact tree on every run so the snapshot can detect a structural change.
"""

from __future__ import annotations

import datetime as datetime_module
import importlib
import os
import re
import shutil
from pathlib import Path
from typing import Any, cast

import book_translator.stages.agent.graph as agent_graph
import pytest
from book_translator import (
    run_assembly,
    run_bind,
    run_chunk_glossary,
    run_chunking,
    run_conversion,
    run_dictionary,
    run_epub,
    run_glossary,
    run_matter,
    run_postprocess,
    run_preprocess,
    run_split,
    run_translation,
)
from book_translator.stages.dictionary.ner import Entity
from book_translator.stages.glossary.extract import ExtractionOutcome, TokenUsage
from book_translator.stages.glossary.models import ChapterExtraction
from book_translator.stages.translation.models import ChunkTranslationAnswer
from book_translator.stages.translation.translator import TranslationOutcome
from langchain_core.messages import AIMessage

#: The checked-in source book. TXT exercises stage 0's copy path without docling.
FIXTURE_BOOK = Path(__file__).resolve().parent / "fixtures" / "fixture_book.txt"

#: The chapter map stage 1's scripted model submits. Line numbers are part of the
#: fixture's contract; changing the fixture therefore requires updating this map
#: (and the artifact snapshot), exactly as a real book's map would.
SPLIT_MAP: tuple[dict[str, Any], ...] = (
    {"line_number": 1, "filename": "00-title-page.md", "title": "The Hunter and the World"},
    {"line_number": 11, "filename": "01-preface.md", "title": "Preface"},
    {"line_number": 15, "filename": "02-arrival.md", "title": "1. Arrival"},
    {"line_number": 19, "filename": "03-the-storm.md", "title": "2. The Storm"},
    {"line_number": 23, "filename": "04-notes.md", "title": "Notes", "role": "back-matter"},
)

#: Stage labels in pipeline order. The test asserts every one produced a result.
STAGE_NAMES: tuple[str, ...] = (
    "1-convert",
    "2-split",
    "3-preprocess",
    "4-dictionary",
    "4-glossary",
    "5-chunk",
    "6-chunk-glossary",
    "7-matter",
    "8-translate",
    "9-assembly",
    "10-postprocess",
    "11-bind",
    "12-epub",
)

#: Every stage writes one of these. The test checks the whole set is present.
MANIFEST_FILES: tuple[str, ...] = (
    "01-markdown/_convert_manifest.json",
    "02-chapters/_manifest.json",
    "02-chapters/_agent_trace.json",
    "03-preprocessed/_preprocess_manifest.json",
    "04-reduced/_reduced_manifest.json",
    "04-glossary/_glossary_manifest.json",
    "05-chunks/_chunk_manifest.json",
    "06-chunk-glossary/_chunk_glossary_manifest.json",
    "07-matter/_matter_manifest.json",
    "08-translations/_translation_manifest.json",
    "09-translated-chapters/_translated_chapters_manifest.json",
    "10-postprocessed-chapters/_postprocess_manifest.json",
    "_bind_manifest.json",
    "_epub_manifest.json",
)

#: A fixed mtime, so stage 11's ``dcterms:modified`` and zip timestamps are stable.
FIXED_MTIME = 1_700_000_000

#: The moment every stage believes it is running at. Without this, a manifest's
#: ``generated_at`` would change the SHA-256 another stage records of that
#: manifest (stage 6 hashes ``glossary.json``), so the tree could never be stable.
FIXED_NOW = datetime_module.datetime(2024, 1, 1, 12, 0, 0)

#: Modules that write ``generated_at``/``created`` through their own imported
#: ``datetime`` symbol. They are patched one by one because that is exactly how
#: the stage code imported it.
_DATETIME_MODULES = (
    "book_translator.stages.assembly.runner",
    "book_translator.stages.binding.runner",
    "book_translator.stages.chunk_glossary.runner",
    "book_translator.stages.conversion.runner",
    "book_translator.stages.epub.metadata",
    "book_translator.stages.epub.runner",
    "book_translator.stages.glossary.runner",
    "book_translator.stages.matter_agent.runner",
    "book_translator.stages.postprocess_agent.runner",
    "book_translator.stages.translation.runner",
)


class _FixedDateTime(datetime_module.datetime):
    """``datetime`` whose ``now()`` is frozen, for deterministic manifests."""

    @classmethod
    def now(cls, tz=None):
        return cls(
            FIXED_NOW.year,
            FIXED_NOW.month,
            FIXED_NOW.day,
            FIXED_NOW.hour,
            FIXED_NOW.minute,
            FIXED_NOW.second,
            tzinfo=tz,
        )


def _freeze_stage_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every stage's wall-clock timestamp the same fixed moment."""
    for module_name in _DATETIME_MODULES:
        module = importlib.import_module(module_name)
        if hasattr(module, "datetime"):
            monkeypatch.setattr(module, "datetime", _FixedDateTime)


_USAGE = TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15, calls=1)


def _tool_call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def _prompt_text(messages: list[Any]) -> str:
    return "\n".join(str(getattr(message, "content", "")) for message in messages)


class ScriptedModel:
    """A chat-model double whose reply is a pure function of the prompt."""

    def __init__(self, responder) -> None:
        self._responder = responder
        self.calls: list[list[Any]] = []

    def invoke(self, messages, *_, **__) -> AIMessage:
        self.calls.append(list(messages))
        return cast(AIMessage, self._responder(messages, len(self.calls)))


class FixtureNer:
    """A tagger with a fixed, tiny vocabulary; no model download, no GPU."""

    name = "fixture-ner"

    def __init__(self, language: str = "en") -> None:
        self.language = language
        self.table = {
            "Milan": "PER",
            "Ana": "PER",
            "London": "LOC",
            "hunter": "MISC",
        }
        self.calls: list[str] = []

    def entities(self, text: str) -> list[Entity]:
        self.calls.append(text)
        return [Entity(term, kind) for term, kind in self.table.items() if term in text]


class FixtureExtractor:
    """A glossary extractor that answers according to the chapter in the prompt."""

    name = "fixture-glossary"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def extract(self, *, system_prompt: str, user_prompt: str) -> ExtractionOutcome:
        self.calls.append(user_prompt)
        entries = (
            [
                {"term": "Milan", "category": "character", "translation": "Milan"},
                {"term": "London", "category": "location", "translation": "London"},
                {"term": "Ana", "category": "character", "translation": "Ana"},
            ]
            if "02-arrival.md" in user_prompt
            else []
        )
        return ExtractionOutcome(
            extraction=ChapterExtraction.model_validate({"entries": entries}),
            usage=_USAGE,
            attempts=1,
            method="scripted",
            raw_text="{}",
        )


class FixtureTranslator:
    """A body translator with a fixed answer per fixture chapter."""

    name = "fixture-translator"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def translate(
        self, *, system_prompt: str, user_prompt: str, source_text: str = ""
    ) -> TranslationOutcome:
        self.calls.append(source_text)
        if "Arrival" in source_text:
            translation = (
                "# Dolazak\n\n"
                "Milan je stigao u selo u zoru. London je bio daleko, a Ana je "
                "čekala Milana kod starog mosta.\n"
            )
        elif "Storm" in source_text:
            translation = (
                "# Oluja\n\n"
                "Oluja je došla bez najave. Milan je gledao u nebo i sećao se "
                "reči lovca.\n"
            )
        else:  # pragma: no cover - the fixture has only the two body chapters
            translation = "# Prevod\n\nPrevedeni tekst.\n"
        return TranslationOutcome(
            answer=ChunkTranslationAnswer(translation=translation),
            usage=_USAGE,
            attempts=1,
            method="scripted",
            raw_text="{}",
        )


def _split_responder(_messages, _turn: int) -> AIMessage:
    return _tool_call(
        "propose_split_map",
        {"split_mapping": [dict(entry) for entry in SPLIT_MAP], "strategy": "toc_titles"},
        "split-1",
    )


def _preprocess_responder(messages, turn: int) -> AIMessage:
    match = re.search(r"Preprocess the chapter at (.+?)\.\n", _prompt_text(messages))
    if match is None:  # pragma: no cover - the node prompt is part of the contract
        raise AssertionError("preprocess prompt did not name the chapter")
    return _tool_call(
        "report_chapter_cleaning",
        {
            "chapter_path": match.group(1),
            "operations": [
                {"line_number": 1, "decision": "keep", "note": "fixture: keep"}
            ],
            "text_fixes": [],
            "notes": "fixture review",
        },
        f"preprocess-{turn}",
    )


def _matter_book_responder(_messages, turn: int) -> AIMessage:
    plan = {
        "original_title": "The Hunter and the World",
        "author": "Ivan Petrovic",
        "translated_title": "Lovac i svet",
        "title_source": "translated",
        "title_evidence": "offline fixture",
        "translation_credit": "Prevod: BleBel",
        "edition_note": "Prevod sa engleskog jezika",
        "omit": [],
        "sections": {
            "00-title-page.md": {"title": "Lovac i svet"},
            "01-preface.md": {"title": "Predgovor"},
            "04-notes.md": {"title": "Beleške"},
        },
    }
    return _tool_call("report_matter_plan", {"plan": plan}, f"matter-plan-{turn}")


def _matter_chunk_responder(messages, turn: int) -> AIMessage:
    prompt = _prompt_text(messages)
    if "00-title-page.md" in prompt:
        translation = "# Lovac i svet\n\nPrevod: BleBel\n\nIvan Petrovic\n"
    elif "01-preface.md" in prompt:
        translation = "# Predgovor\n\nOva knjiga je nastala iz razgovora o lovcu i šumi.\n"
    elif "04-notes.md" in prompt:
        translation = "# Beleške\n\nOvaj odeljak sadrži napomene o lovcu i šumi.\n"
    else:  # pragma: no cover - the fixture has only these three matter chunks
        translation = "# Deo\n\nPrevedeni tekst.\n"
    return _tool_call(
        "report_matter_localization",
        {"translation": translation, "title_used": ""},
        f"matter-chunk-{turn}",
    )


def _postprocess_book_responder(_messages, turn: int) -> AIMessage:
    # The real graph makes the published-contents lookup mandatory before it
    # accepts a book plan. Script that call too, so the fixture exercises the
    # default path rather than switching the requirement off.
    if turn == 1:
        return _tool_call(
            "web_search",
            {"query": "The Hunter and the World table of contents"},
            "postprocess-web-1",
        )
    plan = {
        "naming": {"mode": "keep", "label": "Poglavlje", "level": 1},
        "toc": {"action": "create", "heading": "Sadržaj", "style": "numbered"},
        "title_evidence": "https://example.invalid/fixture-toc",
        "title_source": "web",
    }
    return _tool_call("report_book_plan", {"plan": plan}, f"postprocess-book-{turn}")


def _fixture_web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """A fixed search result; no socket is opened."""
    results = [
        {
            "title": "Contents",
            "url": "https://example.invalid/fixture-toc",
            "snippet": "1. Arrival, 2. The Storm",
        }
    ]
    return {
        "query": query,
        "provider": "fixture",
        "result_count": len(results[:max_results]),
        "results": results[:max_results],
    }


def _postprocess_chapter_responder(messages, turn: int) -> AIMessage:
    match = re.search(r"chapter_path: (.+?\.md)", _prompt_text(messages))
    if match is None:  # pragma: no cover - the node prompt is part of the contract
        raise AssertionError("postprocess prompt did not name the chapter path")
    chapter_path = match.group(1)
    return _tool_call(
        "report_chapter_postprocess",
        {
            "chapter_path": chapter_path,
            "title": {"action": "keep", "line_number": 1},
            "headings": [],
            "notes": "fixture review",
        },
        f"postprocess-chapter-{turn}",
    )


def _require_ok(stage: str, result: Any) -> Any:
    ok = bool(getattr(result, "ok", False))
    if not ok:
        errors = getattr(result, "errors", None) or getattr(result, "issues", None)
        raise AssertionError(f"{stage} failed: {errors}")
    return result


def run_fixture_pipeline(
    work: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, Any]]:
    """Run stages 1 -> 12 into ``work`` and return the book folder and results."""
    work = work.resolve()
    _freeze_stage_clock(monkeypatch)
    input_dir = work / "input"
    input_dir.mkdir(parents=True)
    source = input_dir / "fixture.txt"
    shutil.copyfile(FIXTURE_BOOK, source)

    book = work / "out" / "fixture"
    results: dict[str, Any] = {}

    # Stage 1: TXT -> 01-markdown/<slug>.md. No docling, no OCR, no network.
    conversion = _require_ok(
        "1-convert", run_conversion(source, book / "01-markdown")
    )
    results["1-convert"] = conversion
    markdown = Path(conversion.markdown_path)

    # Stage 2: the real graph, with only the chat model doubled. The monkeypatch
    # is the same seam a provider client uses; no stage code is bypassed.
    monkeypatch.setattr(
        agent_graph, "make_llm", lambda _settings: ScriptedModel(_split_responder)
    )
    results["2-split"] = _require_ok(
        "2-split",
        run_split(str(markdown), str(book / "02-chapters"), max_attempts=3),
    )

    results["3-preprocess"] = _require_ok(
        "3-preprocess",
        run_preprocess(
            str(book / "02-chapters"),
            model=ScriptedModel(_preprocess_responder),
        ),
    )

    # Stage 4: real funnel; only the NER backend is a deterministic double.
    results["4-dictionary"] = _require_ok(
        "4-dictionary",
        run_dictionary(
            str(book / "03-preprocessed"),
            language="en",
            backend=FixtureNer("en"),
        ),
    )

    results["4-glossary"] = _require_ok(
        "4-glossary",
        run_glossary(
            str(book / "03-preprocessed"),
            str(book / "04-glossary"),
            extractor=FixtureExtractor(),
            source_language="en",
            target_language="sr",
        ),
    )

    # A large word-token budget keeps one chunk per fixture chapter, which makes
    # the downstream chunk names stable without depending on tiktoken's corpus.
    results["5-chunk"] = _require_ok(
        "5-chunk",
        run_chunking(
            str(book / "03-preprocessed"), chunk_size=200, tokenizer="word"
        ),
    )

    results["6-chunk-glossary"] = _require_ok(
        "6-chunk-glossary",
        run_chunk_glossary(
            str(book / "05-chunks"), language="en", backend=FixtureNer("en")
        ),
    )

    results["7-matter"] = _require_ok(
        "7-matter",
        run_matter(
            str(book / "02-chapters"),
            str(book / "05-chunks"),
            book_model=ScriptedModel(_matter_book_responder),
            chunk_model=ScriptedModel(_matter_chunk_responder),
            source_language="en",
            target_language="sr",
            book_title="The Hunter and the World",
            book_author="Ivan Petrovic",
        ),
    )

    results["8-translate"] = _require_ok(
        "8-translate",
        run_translation(
            str(book / "05-chunks"),
            translator=FixtureTranslator(),
            source_language="en",
            target_language="sr",
        ),
    )

    results["9-assembly"] = _require_ok(
        "9-assembly", run_assembly(str(book / "05-chunks"))
    )

    # The published-contents lookup is the only network-shaped dependency stage
    # 10 has. It stays mandatory (the default), but the search function itself is
    # a fixed in-process result, so the run is offline and deterministic.
    monkeypatch.setattr(
        "book_translator.stages.agent.tools.websearch.search_web", _fixture_web_search
    )
    results["10-postprocess"] = _require_ok(
        "10-postprocess",
        run_postprocess(
            str(book / "09-translated-chapters"),
            book_model=ScriptedModel(_postprocess_book_responder),
            chapter_model=ScriptedModel(_postprocess_chapter_responder),
        ),
    )

    results["11-bind"] = _require_ok(
        "11-bind", run_bind(str(book / "10-postprocessed-chapters"))
    )

    # Stage 12 says "the newest chapter mtime is the publication date". Pin the
    # mtimes so two runs (and two machines) package byte-identical EPUBs.
    postprocessed = book / "10-postprocessed-chapters"
    for path in sorted(postprocessed.rglob("*")):
        if path.is_file():
            os.utime(path, (FIXED_MTIME, FIXED_MTIME))
    results["12-epub"] = _require_ok("12-epub", run_epub(str(postprocessed)))

    return book, results


__all__ = [
    "FIXED_MTIME",
    "FIXTURE_BOOK",
    "MANIFEST_FILES",
    "SPLIT_MAP",
    "STAGE_NAMES",
    "run_fixture_pipeline",
]
