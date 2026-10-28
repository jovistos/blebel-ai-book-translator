"""Offline tests for stage 0: EPUB/PDF/TXT -> Markdown.

docling itself is never imported here: the converter is stubbed, so these tests
pin the *decisions* stage 0 makes — a TXT is copied unchanged, a PDF without a
text layer is rejected instead of OCR'd, a text-based PDF/EPUB goes to docling,
and an existing conversion is reused only while the source hash still matches.
"""

from __future__ import annotations

import base64
import json
import zipfile
from pathlib import Path

import pytest
from book_translator.core.constants import CONVERT_MANIFEST_FILENAME
from book_translator.core.errors import MissingDependencyError, ToolError
from book_translator.stages.conversion import (
    classify_source,
    converted_markdown_path,
    inspect_pdf_text_layer,
    is_markdown_source,
    run_conversion,
)
from book_translator.stages.conversion import runner as conversion_runner


def fake_docling(monkeypatch, markdown: str = "# Chapter One\n\nSome converted text.\n"):
    """Replace docling: record the calls and return fixed Markdown."""
    calls: list[tuple[str, str]] = []

    def _convert(path, kind, **_kwargs):
        calls.append((str(path), kind))
        return markdown

    monkeypatch.setattr(conversion_runner, "_convert_with_docling", _convert)
    return calls


def fake_pdf_pages(monkeypatch, texts: list[str], pages: int | None = None):
    """Replace the PDF text-layer reader with fixed page texts."""

    def _read(path, sample_pages):
        return (pages if pages is not None else len(texts)), list(texts)

    monkeypatch.setattr(conversion_runner, "_read_pdf_page_texts", _read)


# --------------------------------------------------------------------------- #
# classification and paths
# --------------------------------------------------------------------------- #
def test_classify_source_covers_the_three_supported_families(tmp_path):
    for name, expected in (
        ("book.md", "markdown"),
        ("book.markdown", "markdown"),
        ("book.txt", "text"),
        ("book.pdf", "pdf"),
        ("book.PDF", "pdf"),
        ("book.epub", "epub"),
    ):
        path = tmp_path / name
        path.write_text("x", encoding="utf-8")
        assert classify_source(path) == expected


def test_classify_source_rejects_an_unknown_extension_and_a_missing_file(tmp_path):
    other = tmp_path / "book.docx"
    other.write_text("x", encoding="utf-8")
    with pytest.raises(ToolError, match="unsupported source format"):
        classify_source(other)
    with pytest.raises(ToolError, match="not found"):
        classify_source(tmp_path / "missing.pdf")
    assert is_markdown_source("a.md") and not is_markdown_source("a.pdf")


def test_converted_markdown_path_uses_the_book_slug_and_00_markdown(tmp_path):
    assert converted_markdown_path(
        "data/input/My Book.pdf", output_root=tmp_path
    ) == tmp_path / "My-Book" / "00-markdown" / "My-Book.md"


# --------------------------------------------------------------------------- #
# TXT: rename only
# --------------------------------------------------------------------------- #
def test_a_txt_source_is_copied_byte_for_byte_and_named_md(tmp_path):
    source = tmp_path / "notes.txt"
    payload = "Chapter One\r\n\r\nSome text — ünïcode.\r\n".encode()
    source.write_bytes(payload)

    result = run_conversion(source, tmp_path / "out")

    target = tmp_path / "out" / "notes.md"
    assert result.ok and result.written and not result.reused
    assert result.source_kind == "text"
    assert result.method == "text-copy"
    assert target.read_bytes() == payload  # only the name changed
    manifest = json.loads(
        (tmp_path / "out" / CONVERT_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["source_kind"] == "text"
    assert manifest["output"] == "notes.md"


def test_a_markdown_source_is_a_no_op(tmp_path):
    source = tmp_path / "book.md"
    source.write_text("# Title\n", encoding="utf-8")

    result = run_conversion(source, tmp_path / "out")

    assert result.method == "already-markdown"
    assert result.markdown_path == str(source.resolve())
    assert not result.written
    assert not (tmp_path / "out").exists()


# --------------------------------------------------------------------------- #
# PDF: the text-layer gate
# --------------------------------------------------------------------------- #
def test_a_text_based_pdf_is_converted_with_docling(tmp_path, monkeypatch):
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-1.7 fake")
    fake_pdf_pages(monkeypatch, ["A real page of prose. " * 20] * 12, pages=320)
    calls = fake_docling(monkeypatch, "# Chapter One\n\nConverted prose from the PDF.\n")

    result = run_conversion(source, tmp_path / "out")

    assert calls == [(str(source), "pdf")]
    assert result.pages == 320
    assert result.text_pages == 12
    assert result.text_based is True
    assert result.method == "docling-pdf" and result.written
    assert (tmp_path / "out" / "book.md").read_text(encoding="utf-8").startswith("# Chapter One")
    manifest = json.loads(
        (tmp_path / "out" / CONVERT_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["pages"] == 320 and manifest["text_based"] is True
    assert manifest["method"] == "docling-pdf"


def test_an_image_based_pdf_is_rejected_and_docling_is_never_asked(tmp_path, monkeypatch):
    source = tmp_path / "scan.pdf"
    source.write_bytes(b"%PDF-1.7 fake scan")
    # Page numbers only: not a text layer worth converting.
    fake_pdf_pages(monkeypatch, ["18", "19", "", "20", ""] * 2, pages=400)

    def _must_not_run(path, kind, **_kwargs):
        raise AssertionError("docling must not run for an image-based PDF")

    monkeypatch.setattr(conversion_runner, "_convert_with_docling", _must_not_run)

    with pytest.raises(ToolError, match="no embedded text layer") as excinfo:
        run_conversion(source, tmp_path / "out")

    assert "OCR is deliberately disabled" in str(excinfo.value)
    assert not (tmp_path / "out" / "scan.md").exists()


def test_inspect_pdf_text_layer_reports_the_evidence(monkeypatch, tmp_path):
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-1.7 fake")

    fake_pdf_pages(monkeypatch, [""] * 11 + ["A page with actual words on it. " * 5], pages=200)
    report = inspect_pdf_text_layer(source)

    assert report["pages"] == 200
    assert report["sampled_pages"] == 12
    assert report["text_pages"] == 1
    assert report["text_based"] is False  # 1/12 is below the 50 % threshold
    indices = report["sample_indices"]
    assert len(indices) == 12
    assert indices[0] == 0 and indices[-1] < 200
    assert indices == sorted(set(indices))


def test_inspect_pdf_text_layer_accepts_a_short_text_pdf(monkeypatch, tmp_path):
    source = tmp_path / "short.pdf"
    source.write_bytes(b"%PDF-1.7 fake")
    fake_pdf_pages(monkeypatch, ["A single page of real prose. " * 10])

    report = inspect_pdf_text_layer(source)

    assert report["text_based"] is True
    assert report["text_pages"] >= 1


# --------------------------------------------------------------------------- #
# EPUB
# --------------------------------------------------------------------------- #
def test_an_epub_is_converted_with_docling(tmp_path, monkeypatch):
    source = tmp_path / "novel.epub"
    source.write_bytes(b"PK\x03\x04 fake epub")
    calls = fake_docling(monkeypatch, "# Chapter One\n\nThe EPUB text, long enough to keep.\n")

    result = run_conversion(source, tmp_path / "out")

    assert calls == [(str(source), "epub")]
    assert result.method == "docling-epub" and result.written
    assert (tmp_path / "out" / "novel.md").is_file()


#: A valid 1x1 PNG, so docling's image loader has real bytes to open.
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _tiny_epub(
    path: Path, *, with_image: bool = True, body_html: str | None = None
) -> Path:
    """A two-file EPUB: one chapter, optionally one picture."""
    opf = """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="bookid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Picture Book</dc:title>
    <dc:creator>A. Author</dc:creator>
    <dc:identifier id="bookid">tiny</dc:identifier>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="ch1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
    <item id="pic" href="images/rabbit.png" media-type="image/png"/>
  </manifest>
  <spine><itemref idref="ch1"/></spine>
</package>
"""
    image_tag = '<img src="images/rabbit.png" alt="Rabbit"/>' if with_image else ""
    if body_html is None:
        body_html = f"""<h1>Chapter One</h1>
<p>Alice looked at the picture and smiled.</p>
{image_tag}
<p>Then she ran away.</p>
"""
    chapter = f"""<html xmlns="http://www.w3.org/1999/xhtml"><body>
{body_html}
</body></html>
"""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        archive.writestr(
            "META-INF/container.xml",
            """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="OEBPS/content.opf"
    media-type="application/oebps-package+xml"/></rootfiles>
</container>
""",
        )
        archive.writestr("OEBPS/content.opf", opf)
        archive.writestr("OEBPS/chapter1.xhtml", chapter)
        if with_image:
            archive.writestr("OEBPS/images/rabbit.png", _PNG)
    return path


def test_a_real_epub_extracts_its_images_and_links_them(tmp_path):
    """The one integration test for stage 0's image export (tiny fixture)."""
    pytest.importorskip("docling")
    source = _tiny_epub(tmp_path / "picture.epub")
    markdown_dir = tmp_path / "book" / "00-markdown"

    result = run_conversion(source, markdown_dir)

    assert result.ok and result.written
    assert result.images == 1
    assert result.image_refs == 1
    assert result.images_dir == str(markdown_dir / "images")

    markdown = Path(result.markdown_path).read_text(encoding="utf-8")
    assert "![Image](../00-markdown/images/" in markdown
    # The reference resolves from the Markdown file *and* from a stage folder.
    reference = markdown.split("](", 1)[1].split(")", 1)[0]
    assert (markdown_dir / reference).is_file()
    (tmp_path / "book" / "01-chapters").mkdir()
    assert (tmp_path / "book" / "01-chapters" / reference).is_file()


def test_epub_presentation_tags_are_stripped_before_docling(tmp_path):
    """Small caps wrapped in spans/big must not become separate words.

    docling ends a text run at every inline-element boundary and joins the runs
    with a space, turning ``C<span>ONTENTS</span>`` into ``C ONTENTS`` and
    ``<big>S</big>TRANGE`` into ``S TRANGE``; the broken word then survives
    translation. Stage 0 converts from a copy whose presentation tags are gone,
    keeping every character of the text.
    """
    source = _tiny_epub(
        tmp_path / "spans.epub",
        body_html=(
            '<h1>C<span><span class="smallcaps">ONTENTS</span></span></h1>'
            "<p>B<span>IOGRAPHICAL</span> N<span>OTE</span></p>"
            "<p><big>S</big>TRANGE <small>C</small>ASE</p>"
        ),
    )

    flat = conversion_runner._strip_epub_presentation_tags(source, tmp_path)

    with zipfile.ZipFile(source) as original, zipfile.ZipFile(flat) as rewritten:
        assert rewritten.namelist()[0] == "mimetype"
        assert rewritten.read("mimetype") == b"application/epub+zip"
        chapter = rewritten.read("OEBPS/chapter1.xhtml").decode("utf-8")
        assert "<span" not in chapter.lower()
        assert "<big" not in chapter.lower()
        assert "<small" not in chapter.lower()
        assert "CONTENTS" in chapter
        assert "BIOGRAPHICAL NOTE" in chapter
        assert "STRANGE CASE" in chapter
        # The source is untouched and also the non-text entries are copied as-is.
        assert "<span" in original.read("OEBPS/chapter1.xhtml").decode("utf-8")
        assert rewritten.read("OEBPS/images/rabbit.png") == original.read(
            "OEBPS/images/rabbit.png"
        )


def test_a_small_caps_heading_is_converted_as_one_word(tmp_path):
    pytest.importorskip("docling")
    source = _tiny_epub(
        tmp_path / "spans.epub",
        body_html=(
            '<h1>C<span><span class="smallcaps">ONTENTS</span></span></h1>'
            "<p><big>S</big>TRANGE CASE</p>"
            "<p>Alice looked at the picture and smiled.</p>"
        ),
    )
    markdown_dir = tmp_path / "book" / "00-markdown"

    result = run_conversion(source, markdown_dir, images=False)

    markdown = Path(result.markdown_path).read_text(encoding="utf-8")
    assert "CONTENTS" in markdown
    assert "C ONTENTS" not in markdown
    assert "STRANGE CASE" in markdown
    assert "S TRANGE" not in markdown


def test_image_extraction_can_be_switched_off(tmp_path):
    pytest.importorskip("docling")
    source = _tiny_epub(tmp_path / "picture.epub")
    markdown_dir = tmp_path / "book" / "00-markdown"

    result = run_conversion(source, markdown_dir, images=False)

    assert result.ok and result.images == 0 and result.image_refs == 0
    assert not (markdown_dir / "images").exists()
    assert "![" not in Path(result.markdown_path).read_text(encoding="utf-8")


def test_docling_is_a_lazy_missing_dependency(tmp_path, monkeypatch):
    source = tmp_path / "novel.epub"
    source.write_bytes(b"PK\x03\x04 fake epub")
    # Make ``from docling... import`` fail even in an environment that has it.
    monkeypatch.setitem(__import__("sys").modules, "docling", None)
    monkeypatch.setitem(__import__("sys").modules, "docling.document_converter", None)

    with pytest.raises(MissingDependencyError, match="pip install docling"):
        conversion_runner._convert_with_docling(source, "epub")


def test_pdf_conversion_switches_docling_ocr_off(tmp_path, monkeypatch):
    """The no-OCR promise is made at docling's API, not just in the preflight."""
    import sys
    import types

    captured: dict = {}

    class PdfPipelineOptions:
        do_ocr = True
        do_table_structure = True
        generate_page_images = True
        generate_picture_images = True

    class PdfFormatOption:
        def __init__(self, pipeline_options):
            self.pipeline_options = pipeline_options

    class InputFormat:
        PDF = "pdf"

    class DocumentConverter:
        def __init__(self, format_options=None):
            captured["format_options"] = format_options

        def convert(self, path):
            captured["path"] = path

            class _Document:
                def export_to_markdown(self):
                    return "# Converted\n\n" + "A converted page of text. " * 10

            class _Result:
                document = _Document()

            return _Result()

    base_models = types.ModuleType("docling.datamodel.base_models")
    base_models.InputFormat = InputFormat
    pipeline_options = types.ModuleType("docling.datamodel.pipeline_options")
    pipeline_options.PdfPipelineOptions = PdfPipelineOptions
    datamodel = types.ModuleType("docling.datamodel")
    datamodel.base_models = base_models
    datamodel.pipeline_options = pipeline_options
    document_converter = types.ModuleType("docling.document_converter")
    document_converter.DocumentConverter = DocumentConverter
    document_converter.PdfFormatOption = PdfFormatOption
    docling = types.ModuleType("docling")
    docling.datamodel = datamodel
    docling.document_converter = document_converter
    for name, module in (
        ("docling", docling),
        ("docling.datamodel", datamodel),
        ("docling.datamodel.base_models", base_models),
        ("docling.datamodel.pipeline_options", pipeline_options),
        ("docling.document_converter", document_converter),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    markdown = conversion_runner._convert_with_docling(tmp_path / "book.pdf", "pdf")

    options = captured["format_options"]["pdf"].pipeline_options
    assert options.do_ocr is False
    assert options.do_table_structure is False
    assert captured["path"] == str(tmp_path / "book.pdf")
    assert markdown.startswith("# Converted")


# --------------------------------------------------------------------------- #
# reuse
# --------------------------------------------------------------------------- #
def test_an_existing_conversion_is_reused_until_the_source_changes(tmp_path):
    source = tmp_path / "notes.txt"
    target = tmp_path / "out" / "notes.md"
    source.write_text("first version\n", encoding="utf-8")

    first = run_conversion(source, tmp_path / "out")
    second = run_conversion(source, tmp_path / "out")

    assert first.written and not first.reused
    assert second.reused and not second.written
    assert target.read_text(encoding="utf-8") == "first version\n"

    source.write_text("second version\n", encoding="utf-8")
    third = run_conversion(source, tmp_path / "out")

    assert third.written and not third.reused
    assert target.read_text(encoding="utf-8") == "second version\n"


def test_force_converts_again_even_when_the_hash_matches(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("same\n", encoding="utf-8")
    run_conversion(source, tmp_path / "out")

    forced = run_conversion(source, tmp_path / "out", force=True)

    assert forced.written and not forced.reused


# --------------------------------------------------------------------------- #
# dry run
# --------------------------------------------------------------------------- #
def test_a_dry_run_validates_without_writing(tmp_path, monkeypatch):
    txt = tmp_path / "notes.txt"
    txt.write_text("hello\n", encoding="utf-8")
    txt_result = run_conversion(txt, tmp_path / "out", dry_run=True)
    assert txt_result.dry_run and not txt_result.written
    assert not (tmp_path / "out" / "notes.md").exists()

    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF-1.7 fake")
    fake_pdf_pages(monkeypatch, ["A text page. " * 10] * 3)

    def _must_not_run(path, kind, **_kwargs):
        raise AssertionError("a dry run must not invoke docling")

    monkeypatch.setattr(conversion_runner, "_convert_with_docling", _must_not_run)
    pdf_result = run_conversion(pdf, tmp_path / "out", dry_run=True)
    assert pdf_result.dry_run and not pdf_result.written
    assert pdf_result.method == "docling-pdf"
    assert not (tmp_path / "out" / "book.md").exists()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_convert_cli_writes_the_markdown_and_reports_json(tmp_path, capsys):
    from book_translator.cli import main

    source = tmp_path / "notes.txt"
    source.write_text("# Title\n\nBody\n", encoding="utf-8")

    code = main(["convert", str(source), "--out", str(tmp_path / "md"), "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_kind"] == "text"
    assert payload["method"] == "text-copy"
    assert (tmp_path / "md" / "notes.md").read_text(encoding="utf-8") == source.read_text(
        encoding="utf-8"
    )


def test_convert_cli_reports_an_image_based_pdf_as_an_error(tmp_path, monkeypatch, capsys):
    from book_translator.cli import main

    source = tmp_path / "scan.pdf"
    source.write_bytes(b"%PDF-1.7 fake")
    fake_pdf_pages(monkeypatch, [""] * 12, pages=100)

    code = main(["convert", str(source), "--out", str(tmp_path / "md")])

    assert code == 2
    captured = capsys.readouterr()
    assert "no embedded text layer" in captured.err
    assert "OCR is deliberately disabled" in captured.err
    assert not (tmp_path / "md").exists()


def test_split_converts_a_non_markdown_source_before_slicing(tmp_path, monkeypatch, capsys):
    """`split BOOK.txt` needs Markdown: stage 0 runs first, transparently."""
    from book_translator import cli
    from book_translator.stages.agent.result import SplitterResult

    source = tmp_path / "book.txt"
    source.write_text("# Chapter One\n\nSome text.\n", encoding="utf-8")

    real_convert = cli.common.run_conversion

    def converted_into_tmp(src, out=None, **kwargs):
        return real_convert(src, tmp_path / "md", **kwargs)

    monkeypatch.setattr(cli.common, "run_conversion", converted_into_tmp)

    seen: dict = {}

    def fake_split(source_path, output_dir, **kwargs):
        seen["source"] = source_path
        return SplitterResult(
            ok=True,
            source_path=str(source_path),
            output_dir=str(output_dir),
            strategy="markdown_headers",
            chapter_count=1,
            verification={"reassembles_byte_for_byte": True},
        )

    monkeypatch.setattr("book_translator.stages.agent.run_split", fake_split)

    code = cli.main(["split", str(source), "--out", str(tmp_path / "chapters")])

    assert code == 0
    assert Path(seen["source"]).suffix == ".md"
    assert Path(seen["source"]).read_text(encoding="utf-8") == source.read_text(encoding="utf-8")


def test_extract_epub_cover_from_opf_metadata(tmp_path: Path):
    """An EPUB whose cover is declared in OPF metadata is extracted as cover.<ext>."""
    from book_translator.stages.conversion.runner import _extract_epub_cover

    epub_file = tmp_path / "book.epub"
    cover_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIFfakecover"
    opf_content = """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Test</dc:title>
    <meta name="cover" content="cov-id"/>
  </metadata>
  <manifest>
    <item id="cov-id" href="images/real_cover.jpg" media-type="image/jpeg"/>
    <item id="body" href="text.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="body"/>
  </spine>
</package>"""

    container_xml = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""

    with zipfile.ZipFile(epub_file, "w") as z:
        z.writestr("mimetype", b"application/epub+zip")
        z.writestr("META-INF/container.xml", container_xml)
        z.writestr("OEBPS/content.opf", opf_content)
        z.writestr("OEBPS/images/real_cover.jpg", cover_bytes)

    dest_dir = tmp_path / "images"
    extracted = _extract_epub_cover(epub_file, dest_dir)
    assert extracted is not None
    assert extracted.name == "cover.jpg"
    assert extracted.read_bytes() == cover_bytes

    # If cover already exists in dest_dir, _extract_epub_cover skips
    extracted_again = _extract_epub_cover(epub_file, dest_dir)
    assert extracted_again is None
