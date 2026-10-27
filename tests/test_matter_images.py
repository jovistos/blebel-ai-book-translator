"""Tests for the end-to-end multimodal image recreation phase in the matter chunks stage."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from langchain_core.messages import AIMessage

from book_translator.stages.chunking import run_chunking
from book_translator.stages.matter.images import (
    DEFAULT_MULTIMODAL_MODEL,
    build_image_recreation_prompt,
    is_translated_image_match,
    parse_image_src,
    recreate_matter_images_in_chunk,
    replace_image_src,
    resolve_matter_image_path,
    synthesize_and_translate_image,
)
from book_translator.stages.matter_agent import run_matter


class ScriptedMultimodalModel:
    """Offline double for end-to-end multimodal image-generation models."""

    def __init__(self, responses: list[Any]):
        self.responses = list(responses)
        self.seen_messages = []

    def invoke(self, messages, *_, **__):
        self.seen_messages.append(list(messages))
        idx = min(len(self.seen_messages) - 1, len(self.responses) - 1)
        resp = self.responses[idx]
        if isinstance(resp, dict):
            return AIMessage(content=json.dumps(resp))
        if isinstance(resp, list):
            return AIMessage(content=resp)
        if isinstance(resp, str):
            return AIMessage(content=resp)
        return resp


class ScriptedChatModel:
    """Minimal chat model double replaying canned tool calls."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.seen_messages = []

    def invoke(self, messages, *_, **__):
        self.seen_messages.append(list(messages))
        idx = min(len(self.seen_messages) - 1, len(self.responses) - 1)
        return self.responses[idx]


def _tool_call(name, args, call_id="call_1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def _create_sample_image(path: Path, text: str = "", color: tuple[int, int, int] = (230, 230, 230)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (300, 150), color=color)
    if text:
        draw = ImageDraw.Draw(im)
        draw.text((20, 20), text, fill=(10, 10, 10))
    im.save(path)
    return path


def _sample_image_bytes(color: tuple[int, int, int] = (40, 80, 120)) -> bytes:
    buf = io.BytesIO()
    im = Image.new("RGB", (300, 150), color=color)
    im.save(buf, format="PNG")
    return buf.getvalue()


def test_default_model_is_gemini_image_generator():
    assert DEFAULT_MULTIMODAL_MODEL == "gemini-nano-banana-2.1"


def test_parse_and_replace_image_src():
    md = "![Cover](images/cover.jpg)"
    assert parse_image_src(md) == "images/cover.jpg"
    assert replace_image_src(md, "images/cover_translated.jpg") == "![Cover](images/cover_translated.jpg)"

    md_titled = '![Cover](<images/my cover.png> "Book Title")'
    assert parse_image_src(md_titled) == "images/my cover.png"
    assert replace_image_src(md_titled, "images/trans.png") == '![Cover](images/trans.png "Book Title")'

    html = '<img src="images/seal.png" alt="Seal" />'
    assert parse_image_src(html) == "images/seal.png"
    assert replace_image_src(html, "images/seal_translated.png") == '<img src="images/seal_translated.png" alt="Seal" />'


def test_is_translated_image_match():
    assert is_translated_image_match("![Cover](images/cover.jpg)", "![Cover](images/cover.jpg)")
    assert is_translated_image_match("![Cover](images/cover.jpg)", "![Cover](images/cover_translated.jpg)")
    assert is_translated_image_match(
        "![Cover](../00-markdown/images/cover.png)",
        "![Cover](../00-markdown/images/cover_translated.png)",
    )
    assert not is_translated_image_match("![Cover](images/cover.jpg)", "![Other](images/portrait.jpg)")


def test_resolve_matter_image_path(tmp_path: Path):
    book_dir = tmp_path / "book"
    img_dir = book_dir / "00-markdown" / "images"
    sample = _create_sample_image(img_dir / "test_cover.jpg")

    chunks_dir = book_dir / "05-chunks"
    chunks_dir.mkdir(parents=True)
    chunk_file = chunks_dir / "00000-00-front-matter-0000.md"

    resolved = resolve_matter_image_path("images/test_cover.jpg", candidates=[chunk_file, book_dir])
    assert resolved is not None
    assert resolved.samefile(sample)


def test_build_image_recreation_prompt_forbids_bounding_boxes():
    prompt = build_image_recreation_prompt(
        target_language="Serbian",
        source_language="English",
        book_title="Alisa u zemlji čuda",
        book_author="Luis Kerol",
        translation_credit="Prevod: BleBel",
        matter_context="# Alisa u zemlji čuda",
    )
    assert "end-to-end from scratch" in prompt
    assert "Do NOT output bounding boxes" in prompt
    assert "Alisa u zemlji čuda" in prompt
    assert "Luis Kerol" in prompt


def test_synthesize_and_translate_image_end_to_end_inline_data(tmp_path: Path):
    img = _create_sample_image(tmp_path / "cover.png", "Alice in Wonderland")
    generated_png = _sample_image_bytes((20, 60, 40))
    b64_png = base64.b64encode(generated_png).decode("ascii")

    model = ScriptedMultimodalModel([
        [
            {"inlineData": {"mimeType": "image/png", "data": b64_png}},
            {
                "text": json.dumps(
                    {
                        "recreated": True,
                        "has_text": True,
                        "extracted_text": "Alice in Wonderland",
                        "translated_text": "Alisa u zemlji čuda",
                    }
                )
            },
        ]
    ])

    recreated, img_bytes, extracted, translated = synthesize_and_translate_image(
        img,
        model,
        target_language="Serbian",
        book_title="Alisa u zemlji čuda",
        book_author="Luis Kerol",
        matter_context="Alisa u zemlji čuda i Alisa s one strane ogledala",
    )
    assert recreated is True
    assert img_bytes == generated_png
    assert extracted == "Alice in Wonderland"
    assert translated == "Alisa u zemlji čuda"


def test_synthesize_and_translate_image_without_text(tmp_path: Path):
    img = _create_sample_image(tmp_path / "ornament.jpg")
    model = ScriptedMultimodalModel([{"recreated": False, "has_text": False}])

    recreated, img_bytes, _, _ = synthesize_and_translate_image(img, model)
    assert recreated is False
    assert img_bytes is None


def test_recreate_matter_images_in_chunk_end_to_end(tmp_path: Path):
    img_dir = tmp_path / "00-markdown" / "images"
    _create_sample_image(img_dir / "cover.jpg", "The Great Gatsby")
    _create_sample_image(img_dir / "portrait.jpg")  # no text

    chunk_text = (
        "![Cover](images/cover.jpg)\n\n"
        "# The Book\n\n"
        "![Portrait](images/portrait.jpg)\n"
    )

    generated_bytes = _sample_image_bytes((50, 90, 140))
    b64_img = base64.b64encode(generated_bytes).decode("ascii")

    model = ScriptedMultimodalModel([
        # End-to-end generated image for cover:
        {
            "recreated": True,
            "has_text": True,
            "extracted_text": "The Great Gatsby",
            "translated_text": "Veliki Getsbi",
            "image_base64": b64_img,
        },
        # Response for portrait (no text):
        {"recreated": False, "has_text": False},
    ])

    out_dir = tmp_path / "07-translations"
    out_dir.mkdir(parents=True)

    updated_text, records = recreate_matter_images_in_chunk(
        chunk_text,
        chunk_path=tmp_path / "05-chunks" / "00000-00-front-matter-0000.md",
        out_dir=out_dir,
        model=model,
        source_language="en",
        target_language="sr",
        book_title="Veliki Getsbi",
    )

    assert len(records) == 2
    assert "images/cover_translated.jpg" in updated_text
    assert "images/portrait.jpg" in updated_text  # portrait kept untouched!
    assert (out_dir / "images" / "cover_translated.jpg").read_bytes() == generated_bytes
    assert (img_dir / "cover_translated.jpg").read_bytes() == generated_bytes
    assert records[0].recreated is True
    assert records[0].has_text is True
    assert records[0].translated_text == "Veliki Getsbi"
    assert records[1].recreated is False
    assert records[1].has_text is False


def test_run_matter_end_to_end_image_generation(tmp_path: Path):
    book = tmp_path / "book"
    ch_dir = book / "01-chapters"
    ch_dir.mkdir(parents=True)
    img_dir = book / "00-markdown" / "images"
    _create_sample_image(img_dir / "title_plate.png", "A London Mystery")

    front_text = (
        "![Plate](images/title_plate.png)\n\n"
        "# A London Mystery\n\n"
        "Alex Author\n\n"
        "ISBN 978-000\n"
    )
    (ch_dir / "00-front-matter.md").write_text(front_text, encoding="utf-8")
    (ch_dir / "01-chapter-1.md").write_text("# Chapter 1\n\nIt began on a foggy night.\n", encoding="utf-8")

    run_chunking(str(ch_dir), tokenizer="word", chunk_size=40)
    chunks_dir = book / "05-chunks"

    book_model = ScriptedChatModel([
        _tool_call(
            "report_matter_plan",
            {
                "plan": {
                    "original_title": "A London Mystery",
                    "author": "Alex Author",
                    "translated_title": "Londonska misterija",
                    "title_source": "published",
                    "title_evidence": "https://example.com/misterija",
                    "translation_credit": "Prevod: BleBel",
                    "edition_note": "Prevod",
                    "omit": ["ISBN 978-000"],
                    "sections": {},
                }
            },
        )
    ])

    chunk_model = ScriptedChatModel([
        _tool_call(
            "report_matter_localization",
            {
                "translation": (
                    "![Plate](images/title_plate.png)\n\n"
                    "# Londonska misterija\n\n"
                    "Alex Author\n\n"
                    "*Prevod: BleBel*\n"
                ),
                "notes": "localized front matter",
            },
        )
    ])

    generated_plate = _sample_image_bytes((15, 25, 35))
    b64_plate = base64.b64encode(generated_plate).decode("ascii")
    image_model = ScriptedMultimodalModel([
        [
            {"inlineData": {"mimeType": "image/png", "data": b64_plate}},
            {
                "text": json.dumps(
                    {
                        "recreated": True,
                        "has_text": True,
                        "extracted_text": "A London Mystery",
                        "translated_text": "Londonska misterija",
                    }
                )
            },
        ]
    ])

    res = run_matter(
        ch_dir,
        chunks_dir,
        source_language="en",
        target_language="sr",
        book_model=book_model,
        chunk_model=chunk_model,
        image_model=image_model,
        recreate_images=True,
    )

    assert res.ok is True
    first_chunk_file = Path(res.output_dir) / "00000-00-front-matter-0000.md"
    assert first_chunk_file.is_file()
    written_text = first_chunk_file.read_text(encoding="utf-8")
    assert "images/title_plate_translated.png" in written_text
    assert (Path(res.output_dir) / "images" / "title_plate_translated.png").read_bytes() == generated_plate
    assert (img_dir / "title_plate_translated.png").read_bytes() == generated_plate
    assert len(res.images) >= 1
    assert res.images[0]["recreated"] is True
    assert "Londonska misterija" in res.images[0]["translated_text"]
