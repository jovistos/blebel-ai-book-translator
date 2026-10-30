"""Offline tests for the front/back matter localization stage.

No provider and no network. The tests pin the contract the pipeline depends on:

* front/back matter is found from titles even when a filename looks numbered;
* the book plan fixes the localized title, the BleBel credit and the omissions;
* the first front-matter chunk is published with the title and the credit, even
  when the agent omitted the original imprint page;
* a back-matter section gets its plan title;
* the body translation stage skips the matter chunks the new stage handled;
* the matter tools reuse the splitter's web search and the shared tools.
"""

from __future__ import annotations

from pathlib import Path

from book_translator.stages.agent.tools import SPLIT_TOOLS, stages_of, tool_names
from book_translator.stages.chunking import run_chunking
from book_translator.stages.matter.inspect import (
    discover_matter_sections,
    matter_chapter_names,
)
from book_translator.stages.matter.plan import (
    validate_matter_plan,
    validate_matter_translation,
)
from book_translator.stages.matter_agent import run_matter
from book_translator.stages.matter_agent.prompts import BOOK_SYSTEM_PROMPT
from book_translator.stages.translation import run_translation
from book_translator.stages.translation.models import ChunkTranslationAnswer
from book_translator.stages.translation.translator import TranslationOutcome
from book_translator.toolkit.matter import MATTER_TOOLS
from book_translator.toolkit.postprocess import POSTPROCESS_TOOLS
from book_translator.toolkit.preprocess import PREPROCESS_TOOLS
from langchain_core.messages import AIMessage


class ScriptedModel:
    """Minimal chat-model double: replays canned AIMessages."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.seen_messages = []

    def invoke(self, messages, *_, **__):
        self.seen_messages.append(list(messages))
        index = min(len(self.seen_messages) - 1, len(self.responses) - 1)
        return self.responses[index]


def _tool_call(name, args, call_id="call_1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


FRONT = "# The Book\n\nAlex Author\n\nISBN 978-1\n\nCopyright 1999 Original Press.\n"
PREFACE = "# PREDGOVOR\n\nOvo je predgovor o knjizi.\n"
BODY = "# Chapter One\n\nGeorge lives in London and meets Jane.\n"
BACK = "# BELESKE\n\nNapomena o izdanju.\n"


def _book(tmp_path: Path, chapters: dict[str, str] | None = None) -> tuple[Path, Path]:
    """A chapters folder, chunked with the offline word tokenizer."""
    chapters = chapters or {
        "00-front-matter.md": FRONT,
        "01-predgovor.md": PREFACE,
        "02-chapter-one.md": BODY,
        "03-beleske.md": BACK,
    }
    book = tmp_path / "book"
    folder = book / "02-chapters"
    folder.mkdir(parents=True)
    for name, text in chapters.items():
        (folder / name).write_text(text, encoding="utf-8")
    result = run_chunking(str(folder), tokenizer="word", chunk_size=40)
    assert result.ok is True
    return folder, book / "05-chunks"


def _plan(**overrides):
    plan = {
        "original_title": "The Book",
        "author": "Alex Author",
        "translated_title": "Knjiga",
        "title_source": "published",
        "title_evidence": "https://example.com/knjiga",
        "translation_credit": "Prevod: BleBel",
        "edition_note": "Prevod sa engleskog jezika",
        "omit": ["ISBN 978-1", "Copyright 1999 Original Press."],
        "sections": {
            "01-predgovor.md": {"title": "Predgovor"},
            "03-beleske.md": {"title": "Beleške"},
        },
    }
    plan.update(overrides)
    return plan


def test_matter_files_are_found_by_title_not_only_by_filename(tmp_path: Path):
    chapters, chunks = _book(tmp_path)

    sections = discover_matter_sections(chapters, chunks)

    kinds = {section.filename: section.kind for section in sections}
    assert kinds == {
        "00-front-matter.md": "front",
        "01-predgovor.md": "front",
        "03-beleske.md": "back",
    }
    assert "02-chapter-one.md" not in kinds
    assert all(section.chunk_count >= 1 for section in sections)
    names = matter_chapter_names(chapters, chunks)
    assert names["01-predgovor.md"] == "front"
    assert names["03-beleske.md"] == "back"
    assert "02-chapter-one.md" not in names


def test_the_plan_requires_a_title_and_fixes_the_blebel_credit(tmp_path: Path):
    chapters, chunks = _book(tmp_path)
    book = discover_matter_sections(chapters, chunks)

    validation = validate_matter_plan(book, _plan())

    assert validation["ok"], validation["issues"]
    assert validation["plan"]["translation_credit"] == "Prevod: BleBel"
    assert "BleBel" in validation["plan"]["translated_by"]
    assert validate_matter_plan(book, {"translated_title": ""})["ok"] is False


def test_the_first_front_chunk_is_published_with_title_and_credit(tmp_path: Path):
    chapters, chunks = _book(tmp_path)
    book = discover_matter_sections(chapters, chunks)
    front = next(section for section in book if section.kind == "front")
    validation = validate_matter_plan(book, _plan())

    result = validate_matter_translation(
        source_text=FRONT,
        section_kind="front",
        section_filename=front.filename,
        section_title=front.title,
        plan=validation["plan"],
        translation="# The Book\n\nAlex Author\n\nLicensed edition.\n",
        is_first_front_chunk=True,
    )

    assert result["ok"], result["issues"]
    assert result["text"].startswith("# Knjiga\n")
    assert "BleBel" in result["text"]
    assert "Prevod sa engleskog jezika" in result["text"]
    assert "ISBN 978-1" not in result["text"]
    assert "Copyright 1999 Original Press." not in result["text"]


def test_an_omitted_first_front_chunk_keeps_the_title_page(tmp_path: Path):
    chapters, chunks = _book(tmp_path)
    book = discover_matter_sections(chapters, chunks)
    front = next(section for section in book if section.kind == "front")
    validation = validate_matter_plan(book, _plan())

    result = validate_matter_translation(
        source_text=FRONT,
        section_kind="front",
        section_filename=front.filename,
        section_title=front.title,
        plan=validation["plan"],
        translation="",
        omitted=True,
        omit_reason="the whole original imprint page is not part of the translated edition",
        is_first_front_chunk=True,
    )

    assert result["ok"], result["issues"]
    assert result["omitted"] is True
    assert result["text"].startswith("# Knjiga\n")
    assert "BleBel" in result["text"]


def test_a_cover_picture_stays_above_the_title_page(tmp_path: Path):
    chapters, chunks = _book(tmp_path)
    book = discover_matter_sections(chapters, chunks)
    front = next(section for section in book if section.kind == "front")
    validation = validate_matter_plan(book, _plan())
    source = FRONT + "\n![Cover](cover.png)\n"

    # The model wrote the title page first and the cover under it: the cover
    # still opens the book, and the picture is never dropped.
    result = validate_matter_translation(
        source_text=source,
        section_kind="front",
        section_filename=front.filename,
        section_title=front.title,
        plan=validation["plan"],
        translation=(
            "# The Book\n\n*Prevod: BleBel*\n\n![Cover](cover.png)\n\n"
            "## Contents\n\nText.\n"
        ),
        is_first_front_chunk=True,
    )

    assert result["ok"], result["issues"]
    assert result["text"].startswith("![Cover](cover.png)\n\n# Knjiga\n")
    assert "*Prevod: BleBel*" in result["text"]

    # A model that already puts the cover first keeps it: the heading is added
    # under the picture instead of replacing it.
    result = validate_matter_translation(
        source_text=source,
        section_kind="front",
        section_filename=front.filename,
        section_title=front.title,
        plan=validation["plan"],
        translation="![Cover](cover.png)\n\n# The Book\n\n*Prevod: BleBel*\n",
        is_first_front_chunk=True,
    )

    assert result["ok"], result["issues"]
    assert result["text"].startswith("![Cover](cover.png)\n\n# Knjiga\n")


def test_inline_logos_and_images_stay_below_the_title_page(tmp_path: Path):
    """An inline imprint logo or illustration below title/author must not be lifted."""
    chapters, chunks = _book(tmp_path)
    book = discover_matter_sections(chapters, chunks)
    front = next(section for section in book if section.kind == "front")
    validation = validate_matter_plan(book, _plan())
    source = FRONT + "\n![Logo](imprint.png)\n"

    result = validate_matter_translation(
        source_text=source,
        section_kind="front",
        section_filename=front.filename,
        section_title=front.title,
        plan=validation["plan"],
        translation=(
            "# The Book\n\n*Prevod: BleBel*\n\n![Logo](imprint.png)\n\n"
            "## Contents\n\nText.\n"
        ),
        is_first_front_chunk=True,
    )

    assert result["ok"], result["issues"]
    # The title comes first, then translator credit and edition note, then the logo stays below:
    assert result["text"].startswith("# Knjiga\n\n*Prevod: BleBel*\n\n*Prevod sa engleskog jezika*\n\n![Logo](imprint.png)\n")


def test_a_back_matter_section_gets_the_plan_title(tmp_path: Path):
    chapters, chunks = _book(tmp_path)
    book = discover_matter_sections(chapters, chunks)
    back = next(section for section in book if section.kind == "back")
    validation = validate_matter_plan(book, _plan())

    result = validate_matter_translation(
        source_text=BACK,
        section_kind="back",
        section_filename=back.filename,
        section_title=back.title,
        plan=validation["plan"],
        translation="# Knjiga\n\nNeka napomena.\n",
        is_first_section_chunk=True,
    )

    assert result["ok"], result["issues"]
    assert result["text"].startswith("# Beleške\n")
    assert result["text"].count("#") == 1


def test_matter_localization_must_keep_every_image_reference(tmp_path: Path):
    chapters, chunks = _book(tmp_path)
    book = discover_matter_sections(chapters, chunks)
    back = next(section for section in book if section.kind == "back")
    validation = validate_matter_plan(book, _plan())
    image = "![Image](../01-markdown/images/portrait.png)"
    source = BACK + f"\n{image}\n"

    dropped = validate_matter_translation(
        source_text=source,
        section_kind="back",
        section_filename=back.filename,
        section_title=back.title,
        plan=validation["plan"],
        translation="# Knjiga\n\nNeka napomena.\n",
    )
    assert dropped["ok"] is False
    assert any("dropped an image" in issue for issue in dropped["issues"])

    kept = validate_matter_translation(
        source_text=source,
        section_kind="back",
        section_filename=back.filename,
        section_title=back.title,
        plan=validation["plan"],
        translation=f"# Knjiga\n\nNeka napomena.\n\n{image}\n",
    )
    assert kept["ok"], kept["issues"]
    assert image in kept["text"]


def test_run_matter_localizes_matter_chunks_and_keeps_a_plan(tmp_path: Path):
    chapters, chunks = _book(tmp_path)
    book_model = ScriptedModel([_tool_call("report_matter_plan", {"plan": _plan()})])
    chunk_model = ScriptedModel(
        [
            _tool_call(
                "report_matter_localization",
                {
                    "translation": "# Knjiga\n\nAlex Author\n\nLicensed edition.\n",
                    "notes": "localized title page",
                },
            )
        ]
    )

    result = run_matter(
        str(chapters),
        str(chunks),
        source_language="en",
        target_language="sr",
        book_model=book_model,
        chunk_model=chunk_model,
    )

    assert result.ok, result.errors
    assert result.chunk_count == 3
    assert result.ok_count == 3
    assert result.book_plan["translated_title"] == "Knjiga"
    translations = Path(result.output_dir)
    files = sorted(path.name for path in translations.glob("*.md"))
    assert len(files) == 3
    first_front = Path(result.output_dir) / files[0]
    assert first_front.read_text(encoding="utf-8").startswith("# Knjiga\n")
    assert "BleBel" in first_front.read_text(encoding="utf-8")
    assert result.manifest_path and Path(result.manifest_path).is_file()
    assert result.trace_path and Path(result.trace_path).is_file()
    assert (Path(result.matter_path) / "book-plan.json").is_file()
    # The body chapter was never part of the matter window.
    assert not (translations / "00002-02-chapter-one-0000.md").exists()


def test_run_matter_writes_a_title_page_even_when_the_first_chunk_is_omitted(tmp_path: Path):
    chapters, chunks = _book(tmp_path)
    book_model = ScriptedModel([_tool_call("report_matter_plan", {"plan": _plan()})])
    chunk_model = ScriptedModel(
        [
            _tool_call(
                "report_matter_localization",
                {
                    "translation": "",
                    "omitted": True,
                    "omit_reason": "original imprint material",
                },
            )
        ]
    )

    result = run_matter(
        str(chapters),
        str(chunks),
        source_language="en",
        target_language="sr",
        book_model=book_model,
        chunk_model=chunk_model,
    )

    assert result.ok, result.errors
    files = sorted(Path(result.output_dir).glob("*.md"))
    text = files[0].read_text(encoding="utf-8")
    assert text.startswith("# Knjiga\n")
    assert "BleBel" in text
    assert result.chunks[0]["omitted"] is True


class _Translator:
    name = "fake"

    def __init__(self):
        self.calls: list[str] = []

    def translate(self, *, system_prompt, user_prompt, source_text=""):
        self.calls.append(source_text)
        return TranslationOutcome(
            answer=ChunkTranslationAnswer(translation="PREVEDENO"),
            attempts=1,
            method="fake",
        )


def _write_chunk_glossary(chunks: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for entry in sorted(chunks.glob("*.md")):
        if entry.name.startswith("_"):
            continue
        (output / f"{entry.name}.json").write_text(
            '{"matched_terms": []}', encoding="utf-8"
        )


def test_the_body_translation_stage_skips_matter_chunks(tmp_path: Path):
    _chapters, chunks = _book(tmp_path)
    glossary = tmp_path / "06-chunk-glossary"
    _write_chunk_glossary(chunks, glossary)
    translator = _Translator()

    result = run_translation(
        str(chunks),
        str(tmp_path / "08-translations"),
        chunk_glossary_path=str(glossary),
        source_language="en",
        target_language="sr",
        translator=translator,
    )

    assert result.ok, result.errors
    assert result.chunk_count == 1  # the body chapter
    assert result.skipped_count == 3  # front matter + preface + notes
    assert len(translator.calls) == 1
    assert "George lives in London" in translator.calls[0]


def test_the_body_translation_stage_needs_no_glossary_when_only_matter_remains(tmp_path: Path):
    _chapters, chunks = _book(
        tmp_path,
        chapters={"00-front-matter.md": FRONT, "01-predgovor.md": PREFACE},
    )
    translator = _Translator()

    result = run_translation(
        str(chunks),
        str(tmp_path / "08-translations"),
        chunk_glossary_path=str(tmp_path / "missing"),
        source_language="en",
        target_language="sr",
        translator=translator,
    )

    assert result.ok is True
    assert result.chunk_count == 0
    assert result.skipped_count >= 1
    assert translator.calls == []


def test_the_matter_tools_reuse_the_splitter_web_search_and_shared_tools():
    matter_names = tool_names("matter")
    assert "web_search" in matter_names
    assert "fetch_web_page" in matter_names
    assert "read_matter_section" in matter_names
    assert "report_matter_plan" in matter_names
    assert "report_matter_localization" in matter_names
    assert stages_of("web_search") == ("split", "matter", "postprocess")
    assert "web_search" in [tool.name for tool in MATTER_TOOLS]
    assert "web_search" in [tool.name for tool in SPLIT_TOOLS]
    assert "web_search" in [tool.name for tool in POSTPROCESS_TOOLS]
    assert "web_search" not in [tool.name for tool in PREPROCESS_TOOLS]


def test_the_plan_prompt_forbids_omitting_the_authors_own_preface():
    # The Alice preface ("For this eighty-sixth thousand ...") is printing history and
    # prices written by the author himself: it must be translated, not dropped, even
    # though the plan may omit publisher imprint material of the same kind.
    assert "Never omit the author's own words" in BOOK_SYSTEM_PROMPT
    assert "eighty-sixth thousand" in BOOK_SYSTEM_PROMPT


def test_a_fully_localized_book_resumes_without_a_model(tmp_path: Path):
    chapters, chunks = _book(tmp_path)
    book_model = ScriptedModel([_tool_call("report_matter_plan", {"plan": _plan()})])
    chunk_model = ScriptedModel(
        [
            _tool_call(
                "report_matter_localization",
                {"translation": "# Knjiga\n\nAlex Author\n\nLocalized."},
            )
        ]
    )
    first = run_matter(
        str(chapters),
        str(chunks),
        source_language="en",
        target_language="sr",
        book_model=book_model,
        chunk_model=chunk_model,
    )
    assert first.ok and first.chunk_count >= 1

    second = run_matter(
        str(chapters),
        str(chunks),
        source_language="en",
        target_language="sr",
        resume=True,
    )

    assert second.ok, second.errors
    assert second.resumed is True
    assert second.reused_count == second.chunk_count
    assert second.processed_count == 0


def test_a_dry_run_does_not_synthesize_front_or_back_matter(tmp_path: Path):
    """Synthesis writes chapter and chunk records; a dry run must write none."""
    folder, chunks = _book(tmp_path, {"02-chapter-one.md": BODY})
    before = {path.name for path in folder.iterdir()}
    chunk_before = {path.name for path in chunks.iterdir()}

    result = run_matter(
        str(folder),
        str(chunks),
        source_language="en",
        target_language="sr",
        dry_run=True,
    )

    assert result.ok
    assert result.synthetic_sections == []
    assert {path.name for path in folder.iterdir()} == before
    assert {path.name for path in chunks.iterdir()} == chunk_before


def test_report_matter_localization_large_text_not_truncated():
    import json

    from book_translator.stages.matter.inspect import MatterSection
    from book_translator.toolkit.context import ToolContext
    from book_translator.toolkit.matter.tools import report_matter_localization

    source = "# NOTES\n\n" + ("This is a note entry. " * 2000)
    translation = "# NAPOMENE\n\n" + ("Ovo je unos napomene. " * 2000)
    section = MatterSection(
        filename="34-notes.md",
        path=Path("/tmp/34-notes.md"),
        kind="back",
        title="NOTES",
        chars=len(source),
        lines=100,
        chunk_filenames=["00034-34-notes-0000.md"],
    )
    ctx = ToolContext(
        chapters_path="",
        chunks_path="",
        chunk={"filename": "00034-34-notes-0000.md", "text": source},
        section=section,
        plan={"translated_title": "Naslov"},
        is_first_front_chunk=False,
        is_first_section_chunk=True,
    )
    result = report_matter_localization.invoke(
        {"translation": translation, "ctx": ctx}
    )
    assert len(result) > 24000
    assert "[tool output truncated" not in result
    payload = json.loads(result)
    assert payload["accepted"] is True
    assert len(payload["text"]) > 24000

