"""Tests for filename metadata and generated front/back matter."""

from __future__ import annotations

import json
from pathlib import Path

from book_translator.stages.chunking.runner import chunk_files
from book_translator.stages.matter.inspect import build_matter_book, discover_matter_sections
from book_translator.stages.matter.metadata import parse_book_metadata
from book_translator.stages.matter.synthesize import ensure_matter_sections


def _book(tmp_path: Path, *, with_front: bool = False) -> tuple[Path, Path]:
    book = tmp_path / "book"
    chapters = book / "02-chapters"
    chunks = book / "05-chunks"
    chapters.mkdir(parents=True)
    chunks.mkdir(parents=True)
    files = []
    if with_front:
        (chapters / "00-front-matter.md").write_text("# Front\n\nDedication.\n", encoding="utf-8")
        files.append(
            {
                "filename": "00-front-matter.md",
                "path": str(chapters / "00-front-matter.md"),
                "kind": "front-matter",
                "start_line": 1,
                "end_line": 3,
                "lines": 3,
                "bytes": 24,
                "sha256": "front",
                "first_line": "# Front",
            }
        )
    for index, name in enumerate(("01-chapter-one.md",), start=1):
        (chapters / name).write_text(f"# Chapter {index}\n\nBody.\n", encoding="utf-8")
        files.append(
            {
                "filename": name,
                "path": str(chapters / name),
                "kind": "chapter",
                "start_line": 10,
                "end_line": 12,
                "lines": 3,
                "bytes": 20,
                "sha256": name,
                "first_line": "# Chapter 1",
            }
        )
    (chapters / "_manifest.json").write_text(
        json.dumps(
            {
                "source_path": "/tmp/Tea for the Traditionally Built - Alexander McCall Smith.md",
                "source_lines": 12,
                "file_count": len(files),
                "chapter_count": 1,
                "total_lines": 12,
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    body_chunk = chunks / "00000-01-chapter-one-0000.md"
    body_chunk.write_text("# Chapter 1\n\nBody.\n", encoding="utf-8")
    (chunks / "_chunk_manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "global_index": 0,
                        "chapter_index": 0,
                        "chunk_index": 0,
                        "chapter_id": "01-chapter-one",
                        "chapter_filename": "01-chapter-one.md",
                        "filename": body_chunk.name,
                        "path": str(body_chunk),
                        "token_count": 4,
                        "bytes": 20,
                        "sha256": "body",
                    }
                ],
                "chapters": [
                    {
                        "chapter_id": "01-chapter-one",
                        "filename": "01-chapter-one.md",
                        "chapter_index": 0,
                        "chunk_count": 1,
                        "first_global_index": 0,
                        "last_global_index": 0,
                        "token_count": 4,
                    }
                ],
                "totals": {"chunks": 1},
            }
        ),
        encoding="utf-8",
    )
    return chapters, chunks


def test_filename_metadata_reads_title_and_author(tmp_path: Path):
    meta = parse_book_metadata("Tea for the Traditionally Built - Alexander McCall Smith.md")
    assert meta.title == "Tea for the Traditionally Built"
    assert meta.author == "Alexander McCall Smith"

    swapped = parse_book_metadata("Alexander McCall Smith - Tea for the Traditionally Built.md")
    assert swapped.title == "Tea for the Traditionally Built"
    assert swapped.author == "Alexander McCall Smith"

    by = parse_book_metadata("The Book by Alex Author.md")
    assert (by.title, by.author) == ("The Book", "Alex Author")


def test_missing_front_and_back_matter_are_created_with_chunks(tmp_path: Path):
    chapters, chunks = _book(tmp_path)

    result = ensure_matter_sections(chapters, chunks)

    assert result.created == ["00-front-matter.md", "99-back-matter.md"]
    assert result.front_created and result.back_created
    assert result.metadata.title == "Tea for the Traditionally Built"
    assert result.metadata.author == "Alexander McCall Smith"
    assert (chapters / "00-front-matter.md").is_file()
    assert (chapters / "99-back-matter.md").is_file()
    # The generated text is an artifact: the matter agent localises exactly these
    # facts, so a split of this module must not change one byte of them.
    assert (chapters / "00-front-matter.md").read_text(encoding="utf-8") == (
        "# Tea for the Traditionally Built\n"
        "\n"
        "Alexander McCall Smith\n"
        "\n"
        "*Translated by BleBel*\n"
    )
    assert (chapters / "99-back-matter.md").read_text(encoding="utf-8") == (
        "# About the Author\n"
        "\n"
        "Alexander McCall Smith is the author of Tea for the Traditionally Built.\n"
    )

    sections = discover_matter_sections(chapters, chunks)
    assert [(s.filename, s.kind) for s in sections] == [
        ("00-front-matter.md", "front"),
        ("99-back-matter.md", "back"),
    ]
    assert all(section.chunk_count == 1 for section in sections)

    entries = chunk_files(chunks)
    assert [entry["chapter_filename"] for entry in entries] == [
        "00-front-matter.md",
        "01-chapter-one.md",
        "99-back-matter.md",
    ]
    # Existing body chunk filenames are untouched; only the recorded index moves.
    assert entries[1]["filename"] == "00000-01-chapter-one-0000.md"
    assert entries[1]["global_index"] == 1

    book = build_matter_book(chapters, chunks)
    assert book.book_title == "Tea for the Traditionally Built"
    assert book.book_author == "Alexander McCall Smith"


def test_existing_front_matter_is_kept_and_only_back_is_created(tmp_path: Path):
    chapters, chunks = _book(tmp_path, with_front=True)

    result = ensure_matter_sections(chapters, chunks)

    assert result.created == ["99-back-matter.md"]
    assert not result.front_created
    assert result.back_created
    front = (chapters / "00-front-matter.md").read_text(encoding="utf-8")
    assert "Dedication." in front


class _ScriptedModel:
    """Replay canned tool-call messages; enough for the matter graphs."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, messages, *_, **__):
        index = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[index]


def _tool_call(name: str, args: dict):
    from langchain_core.messages import AIMessage

    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": "call_1", "type": "tool_call"}],
    )


def test_run_matter_creates_and_localizes_missing_front_and_back(tmp_path: Path):
    from book_translator.stages.matter_agent import run_matter

    chapters, chunks = _book(tmp_path)
    plan = {
        "original_title": "Tea for the Traditionally Built",
        "author": "Alexander McCall Smith",
        "translated_title": "Čaj za tradicionalno građene",
        "translation_credit": "Prevod: BleBel",
        "sections": {
            "00-front-matter.md": {"title": "Čaj za tradicionalno građene"},
            "99-back-matter.md": {"title": "O piscu"},
        },
    }
    book_model = _ScriptedModel([_tool_call("report_matter_plan", {"plan": plan})])
    chunk_model = _ScriptedModel(
        [
            _tool_call(
                "report_matter_localization",
                {
                    "translation": (
                        "# Čaj za tradicionalno građene\n\n"
                        "Aleksander Makol Smit\n\n*Prevod: BleBel*"
                    )
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
    assert result.synthetic_sections == ["00-front-matter.md", "99-back-matter.md"]
    assert result.chunk_count == 2
    assert result.book_author == "Alexander McCall Smith"
    outputs = sorted(Path(result.output_dir).glob("*.md"))
    assert [path.name for path in outputs] == [
        "00000-00-front-matter-0000.md",
        "00002-99-back-matter-0000.md",
    ]
    assert "BleBel" in outputs[0].read_text(encoding="utf-8")
    assert "O piscu" in outputs[1].read_text(encoding="utf-8")


def test_existing_matter_without_a_chunk_gets_one(tmp_path: Path):
    chapters, chunks = _book(tmp_path)
    (chapters / "99-back-matter.md").write_text(
        "# About the Author\n\nA writer.\n", encoding="utf-8"
    )
    manifest = json.loads((chapters / "_manifest.json").read_text(encoding="utf-8"))
    manifest["files"].append(
        {
            "filename": "99-back-matter.md",
            "path": str(chapters / "99-back-matter.md"),
            "kind": "back-matter",
            "start_line": 13,
            "end_line": 15,
            "lines": 3,
            "bytes": 30,
            "sha256": "back",
            "first_line": "# About the Author",
        }
    )
    (chapters / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = ensure_matter_sections(chapters, chunks)

    assert result.created == ["00-front-matter.md"]
    assert result.back_created is False
    assert any("99-back-matter" in name for name in result.chunk_created)
    assert any(
        entry["chapter_filename"] == "99-back-matter.md" for entry in chunk_files(chunks)
    )


def test_generated_section_text_has_stable_fallbacks():
    from book_translator.documents.book import BookMetadata
    from book_translator.stages.matter.generate import back_matter_text, front_matter_text

    assert front_matter_text(BookMetadata()) == "# Untitled\n\n*Translated by BleBel*\n"
    assert front_matter_text(BookMetadata(title="T", author="A")) == (
        "# T\n\nA\n\n*Translated by BleBel*\n"
    )
    assert back_matter_text(BookMetadata()) == (
        "# About the Author\n\nthis book was published in translation by BleBel.\n"
    )
    assert back_matter_text(BookMetadata(title="T", author="A")) == (
        "# About the Author\n\nA is the author of T.\n"
    )


def test_existing_matter_entries_are_found_by_kind_then_by_name():
    from book_translator.stages.matter.records import existing_matter_entries

    split = {
        "files": [
            {"filename": "00-front-matter.md", "kind": "frontmatter"},
            {"filename": "01-body.md", "kind": "chapter"},
            {"filename": "99-back-matter.md"},
        ]
    }

    front, back = existing_matter_entries(split)

    assert front is not None and front["filename"] == "00-front-matter.md"
    assert back is not None and back["filename"] == "99-back-matter.md"
