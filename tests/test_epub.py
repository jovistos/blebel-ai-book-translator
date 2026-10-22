"""Offline tests for stage 11: the EPUB 3 package.

Every test builds a tiny book on disk — front matter, two chapters, back matter,
one picture — and then reads the zip the stage produced. Nothing here needs a
model, a network call or a reader: an EPUB is a zip, and the checks below are the
ones a reader would make (the mandatory first entry, the container, the package
document, the navigation document, well-formed XHTML, and pictures that resolve).
"""

from __future__ import annotations

import base64
import json
import posixpath
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from book_splitter import cli
from book_splitter.core.errors import MissingDependencyError, ToolError
from book_splitter.epub import render_markdown, run_epub
from book_splitter.epub.runner import _verify

#: A 1x1 transparent PNG: a real picture, small enough to keep the test readable.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
FRONT_MATTER = """# Alisa u zemlji čuda

*Prevod sa engleskog: BleBel*

![Image](../00-markdown/images/cover.png)

---

Prvi pasus predgovora.
"""
CHAPTER_ONE = """# Prvo poglavlje

Alisa je sedela pored sestre.

## Jedan podnaslov

- jedna
- dve

> Citat na početku poglavlja.

---

Drugi pasus posle prekida scene.
"""
CHAPTER_TWO = """# Drugo poglavlje

Drugo poglavlje počinje ovako.
"""
BACK_MATTER = """# O piscu

Luis Kerol je bio matematičar.
"""
TOC_PAGE = """# Sadržaj

1. Prvo poglavlje
2. Drugo poglavlje
"""
MATTER_MANIFEST = {
    "stage": "matter",
    "ok": True,
    "source_language": "en",
    "target_language": "sr",
    "book_title": "Alisa u zemlji čuda",
    "book_plan": {
        "original_title": "Alice's Adventures in Wonderland",
        "author": "Luis Kerol",
        "translated_title": "Alisa u zemlji čuda",
        "translated_by": "BleBel",
        "translation_credit": "Prevod sa engleskog: BleBel",
    },
}


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def make_book(
    root: Path,
    *,
    toc: bool = False,
    matter: dict | None = MATTER_MANIFEST,
    picture: bool = True,
) -> Path:
    """A minimal postprocessed book, and the folder stage 11 reads."""
    book = root / "Alisa-u-zemlji-cuda"
    chapters = book / "09-postprocessed-chapters"
    chapters.mkdir(parents=True)
    (book / "00-markdown" / "images").mkdir(parents=True)
    if picture:
        (book / "00-markdown" / "images" / "cover.png").write_bytes(PNG)
    (chapters / "00-front-matter.md").write_text(FRONT_MATTER, encoding="utf-8")
    (chapters / "01-prvo.md").write_text(CHAPTER_ONE, encoding="utf-8")
    (chapters / "02-drugo.md").write_text(CHAPTER_TWO, encoding="utf-8")
    (chapters / "99-back-matter.md").write_text(BACK_MATTER, encoding="utf-8")
    if toc:
        (chapters / "00-table-of-contents.md").write_text(TOC_PAGE, encoding="utf-8")
    if matter is not None:
        folder = book / "06b-matter"
        folder.mkdir()
        (folder / "_matter_manifest.json").write_text(
            json.dumps(matter, ensure_ascii=False), encoding="utf-8"
        )
    return chapters


def open_package(path: Path) -> zipfile.ZipFile:
    return zipfile.ZipFile(path)


def opf_root(archive: zipfile.ZipFile) -> ET.Element:
    return ET.fromstring(archive.read("OEBPS/content.opf"))


def _nav_list(archive: zipfile.ZipFile) -> ET.Element:
    """The top-level ``<ol>`` of the navigation document's table of contents."""
    root = ET.fromstring(archive.read("OEBPS/nav.xhtml"))
    namespace = "{http://www.w3.org/1999/xhtml}"
    ops = "{http://www.idpf.org/2007/ops}"
    toc = next(
        element
        for element in root.iter(f"{namespace}nav")
        if element.get(f"{ops}type") == "toc"
    )
    return toc.find(f"{namespace}ol")


def nav_entries(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    """The top-level entries only — one per spine document."""
    namespace = "{http://www.w3.org/1999/xhtml}"
    entries: list[tuple[str, str]] = []
    for item in _nav_list(archive).findall(f"{namespace}li"):
        link = item.find(f"{namespace}a")
        entries.append(("".join(link.itertext()).strip(), link.get("href") or ""))
    return entries


def nav_all(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    """Every entry, nested ones included, in document order."""
    namespace = "{http://www.w3.org/1999/xhtml}"
    return [
        ("".join(link.itertext()).strip(), link.get("href") or "")
        for link in _nav_list(archive).iter(f"{namespace}a")
    ]


# --------------------------------------------------------------------------- #
# the package
# --------------------------------------------------------------------------- #
def test_the_folder_becomes_an_epub3_package(tmp_path):
    chapters = make_book(tmp_path)

    result = run_epub(chapters)

    assert result.ok, result.errors
    target = Path(result.output_path)
    assert target.name == "11-translated-book.epub"
    assert target.is_file()
    assert target.parent == chapters.parent
    assert result.document_count == 4
    assert result.chars > 0

    with open_package(target) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        # The one structural rule a reader checks before anything else.
        assert names[0] == "mimetype"
        assert infos[0].compress_type == zipfile.ZIP_STORED
        assert archive.read("mimetype") == b"application/epub+zip"
        assert "META-INF/container.xml" in names
        assert "OEBPS/content.opf" in names
        assert "OEBPS/nav.xhtml" in names
        # One document per chapter, in reading order.
        assert [name for name in names if name.startswith("OEBPS/text/")] == [
            "OEBPS/text/00-front-matter.xhtml",
            "OEBPS/text/01-prvo.xhtml",
            "OEBPS/text/02-drugo.xhtml",
            "OEBPS/text/99-back-matter.xhtml",
        ]
        # Every document is well-formed XHTML with a title.
        for name in [name for name in names if name.startswith("OEBPS/text/")]:
            root = ET.fromstring(archive.read(name))
            title = root.find(".//{http://www.w3.org/1999/xhtml}title")
            assert title is not None and (title.text or "").strip()


def test_the_package_document_is_epub3_with_the_book_metadata(tmp_path):
    chapters = make_book(tmp_path)

    result = run_epub(chapters)

    with open_package(Path(result.output_path)) as archive:
        opf = opf_root(archive)
        assert opf.get("version") == "3.0"
        metadata = opf.find("{http://www.idpf.org/2007/opf}metadata")
        dc = "{http://purl.org/dc/elements/1.1/}"
        assert metadata.find(f"{dc}title").text == "Alisa u zemlji čuda"
        assert metadata.find(f"{dc}creator").text == "Luis Kerol"
        assert metadata.find(f"{dc}language").text == "sr"
        identifier = metadata.find(f"{dc}identifier")
        assert identifier.text.startswith("urn:uuid:")
        assert identifier.get("id") == opf.get("unique-identifier")
        modified = metadata.find("{http://www.idpf.org/2007/opf}meta")
        assert modified.get("property") == "dcterms:modified"
        assert result.identifier == identifier.text


def test_the_identifier_survives_a_rebuild_and_a_different_folder(tmp_path):
    first = make_book(tmp_path / "a")
    second = make_book(tmp_path / "b")

    one = run_epub(first, dry_run=True, write_manifest=False)
    two = run_epub(second, dry_run=True, write_manifest=False)

    assert one.identifier == two.identifier
    assert one.identifier.endswith("urn:uuid:" + one.identifier.split(":")[-1])
    assert one.identifier.startswith("urn:uuid:")


def test_a_rebuild_of_unchanged_chapters_is_byte_identical(tmp_path):
    chapters = make_book(tmp_path)

    one = run_epub(chapters, tmp_path / "one.epub", write_manifest=False)
    two = run_epub(chapters, tmp_path / "two.epub", write_manifest=False)

    assert one.sha256 == two.sha256
    assert Path(one.output_path).read_bytes() == Path(two.output_path).read_bytes()


def test_the_verification_reports_a_healthy_package(tmp_path):
    chapters = make_book(tmp_path)

    result = run_epub(chapters, write_manifest=False)

    verification = result.verification
    assert verification["mimetype_first"] is True
    assert verification["container_ok"] is True
    assert verification["opf_ok"] is True
    assert verification["nav_ok"] is True
    assert verification["xhtml_valid"] == 4
    assert verification["spine_count"] == 4
    assert verification["image_links_ok"] == 4


# --------------------------------------------------------------------------- #
# the table of contents
# --------------------------------------------------------------------------- #
def test_the_printed_table_of_contents_is_not_shipped(tmp_path):
    chapters = make_book(tmp_path, toc=True)

    result = run_epub(chapters, write_manifest=False)

    assert result.toc_dropped is True
    assert result.document_count == 4
    with open_package(Path(result.output_path)) as archive:
        names = archive.namelist()
        assert not any("table-of-contents" in name for name in names)
        assert "Sadržaj" not in [label for label, _target in nav_all(archive)]
        # The chapters' own titles are the navigation.
        assert [label for label, _target in nav_entries(archive)] == [
            "Alisa u zemlji čuda",
            "Prvo poglavlje",
            "Drugo poglavlje",
            "O piscu",
        ]


def test_the_navigation_nests_sections_under_their_chapter(tmp_path):
    chapters = make_book(tmp_path)

    result = run_epub(chapters, write_manifest=False)

    with open_package(Path(result.output_path)) as archive:
        assert nav_entries(archive)[1] == ("Prvo poglavlje", "text/01-prvo.xhtml")
        assert (
            "Jedan podnaslov",
            "text/01-prvo.xhtml#jedan-podnaslov",
        ) in nav_all(archive)


# --------------------------------------------------------------------------- #
# pictures
# --------------------------------------------------------------------------- #
def test_a_picture_is_embedded_once_and_its_reference_rewritten(tmp_path):
    chapters = make_book(tmp_path)
    (chapters / "02-drugo.md").write_text(
        CHAPTER_TWO + "\n![Image](../00-markdown/images/cover.png)\n", encoding="utf-8"
    )

    result = run_epub(chapters, write_manifest=False)

    assert result.ok, result.errors
    # Two references, one file in the package.
    assert result.image_count == 1
    with open_package(Path(result.output_path)) as archive:
        pictures = [n for n in archive.namelist() if n.startswith("OEBPS/images/")]
        assert pictures == ["OEBPS/images/cover.png"]
        front = archive.read("OEBPS/text/00-front-matter.xhtml").decode("utf-8")
        assert 'src="../images/cover.png"' in front
        assert "../00-markdown/images" not in front
        # A picture alone on its line is a figure, not a paragraph.
        assert "<figure>" in front
        assert result.images[0]["media_type"] == "image/png"


def test_a_missing_picture_is_reported_but_the_book_is_still_written(tmp_path):
    chapters = make_book(tmp_path, picture=False)

    result = run_epub(chapters, write_manifest=False)

    assert result.ok is False
    assert any("picture not found" in error for error in result.errors)
    assert Path(result.output_path).is_file()
    with open_package(Path(result.output_path)) as archive:
        front = archive.read("OEBPS/text/00-front-matter.xhtml").decode("utf-8")
        assert 'class="image-missing"' in front


def test_a_cover_becomes_the_first_page_of_the_book(tmp_path):
    chapters = make_book(tmp_path)

    result = run_epub(
        chapters, cover="cover.png", write_manifest=False
    )

    assert result.ok, result.errors
    assert result.cover == "images/cover.png"
    assert result.document_count == 4
    with open_package(Path(result.output_path)) as archive:
        opf = opf_root(archive)
        manifest = {
            item.get("id"): item
            for item in opf.find("{http://www.idpf.org/2007/opf}manifest")
        }
        assert manifest["cover-image"].get("properties") == "cover-image"
        spine = [
            item.get("idref")
            for item in opf.find("{http://www.idpf.org/2007/opf}spine")
        ]
        assert spine[0] == "cover"
        assert "OEBPS/text/0000-cover.xhtml" in archive.namelist()


def test_an_unknown_cover_is_an_actionable_error(tmp_path):
    chapters = make_book(tmp_path)

    with pytest.raises(ToolError):
        run_epub(chapters, cover="nope.jpg", dry_run=True, write_manifest=False)


# --------------------------------------------------------------------------- #
# metadata
# --------------------------------------------------------------------------- #
def test_explicit_metadata_wins_over_the_records(tmp_path):
    chapters = make_book(tmp_path)

    result = run_epub(
        chapters,
        title="Naslov",
        author="Autor",
        language="hr",
        identifier="urn:isbn:9780000000000",
        description="Opis",
        publisher="Izdavač",
        write_manifest=False,
    )

    assert (result.title, result.author, result.language) == ("Naslov", "Autor", "hr")
    assert result.identifier == "urn:isbn:9780000000000"
    assert result.description == "Opis"
    assert result.publisher == "Izdavač"
    with open_package(Path(result.output_path)) as archive:
        metadata = opf_root(archive).find("{http://www.idpf.org/2007/opf}metadata")
        dc = "{http://purl.org/dc/elements/1.1/}"
        assert metadata.find(f"{dc}description").text == "Opis"
        assert metadata.find(f"{dc}publisher").text == "Izdavač"


def test_a_book_without_records_still_gets_a_package(tmp_path):
    chapters = make_book(tmp_path, matter=None)

    result = run_epub(chapters, write_manifest=False)

    # The front matter's own heading and the default target language are enough.
    assert result.title == "Alisa u zemlji čuda"
    assert result.author == ""
    assert result.language == "sr"
    assert any("no recorded author" in warning for warning in result.warnings)
    assert result.ok, result.errors


# --------------------------------------------------------------------------- #
# files and records
# --------------------------------------------------------------------------- #
def test_the_manifest_records_the_files_and_the_hashes(tmp_path):
    chapters = make_book(tmp_path)

    result = run_epub(chapters)

    manifest = json.loads(
        (chapters.parent / "_epub_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["stage"] == "epub"
    assert manifest["ok"] is True
    assert len(manifest["spine"]) == 4
    assert manifest["spine"][0]["source"] == "00-front-matter.md"
    assert manifest["metadata"]["title"] == "Alisa u zemlji čuda"
    assert manifest["images"][0]["sha256"] == result.images[0]["sha256"]
    assert manifest["verification"]["nav_ok"] is True
    assert (chapters.parent / "_epub_trace.json").is_file()


def test_a_dry_run_writes_nothing(tmp_path):
    chapters = make_book(tmp_path)

    result = run_epub(chapters, dry_run=True)

    assert result.ok
    assert result.dry_run is True
    assert result.bytes > 0
    assert not (chapters.parent / "11-translated-book.epub").exists()
    assert not (chapters.parent / "_epub_manifest.json").exists()


def test_a_folder_without_postprocessed_chapters_is_an_error(tmp_path):
    empty = tmp_path / "book" / "09-postprocessed-chapters"
    empty.mkdir(parents=True)

    with pytest.raises(ToolError, match="run `postprocess` first"):
        run_epub(empty)


def test_the_stylesheet_travels_with_the_book(tmp_path):
    chapters = make_book(tmp_path)

    result = run_epub(chapters, write_manifest=False)

    with open_package(Path(result.output_path)) as archive:
        css = archive.read("OEBPS/css/style.css").decode("utf-8")
        assert "text-indent" in css
        assert "hyphens: auto" in css
        document = archive.read("OEBPS/text/01-prvo.xhtml").decode("utf-8")
        assert 'href="../css/style.css"' in document


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def test_the_renderer_keeps_the_constructs_the_books_use():
    rendered = render_markdown(
        "# Naslov\n\nPrvi *kurziv* i **podebljano** pasus.\n\n"
        "## Podnaslov\n\n- jedna\n- dve\n\n1. prvo\n2. drugo\n\n"
        "> Citat.\n\n---\n\nPoslednji pasus.\n"
    )

    assert "<h1 id=" in rendered.body
    assert "<h2 id=" in rendered.body
    assert "<em>kurziv</em>" in rendered.body
    assert "<strong>podebljano</strong>" in rendered.body
    assert "<ul>" in rendered.body and "<ol>" in rendered.body
    assert "<blockquote>" in rendered.body
    assert "<hr />" in rendered.body
    assert rendered.title == "Naslov"
    assert [heading["level"] for heading in rendered.headings] == [1, 2]
    assert rendered.headings[1]["anchor"] == "podnaslov"


def test_the_renderer_escapes_text_instead_of_passing_html_through():
    rendered = render_markdown("A & B < C > D\n\n<script>alert(1)</script>\n")

    assert "&amp;" in rendered.body
    assert "&lt;script&gt;" in rendered.body
    assert "<script>" not in rendered.body


def test_an_html_picture_tag_survives_as_a_markdown_picture():
    rendered = render_markdown('<img src="../00-markdown/images/x.png" alt="Slika">\n')

    assert '<img src="../00-markdown/images/x.png" alt="Slika" />' in rendered.body
    assert rendered.warnings == []


def test_a_missing_markdown_it_is_an_actionable_error(tmp_path, monkeypatch):
    chapters = make_book(tmp_path)
    # Hiding the module makes `from markdown_it import MarkdownIt` raise ImportError.
    monkeypatch.setitem(__import__("sys").modules, "markdown_it", None)

    with pytest.raises(MissingDependencyError, match="markdown-it-py"):
        run_epub(chapters, dry_run=True, write_manifest=False)


# --------------------------------------------------------------------------- #
# the interface
# --------------------------------------------------------------------------- #
def test_the_cli_writes_the_epub_and_reports_it(tmp_path, capsys):
    chapters = make_book(tmp_path)

    code = cli.main(
        [
            "epub",
            "--postprocessed-in",
            str(chapters),
            "--title",
            "Naslov",
            "--author",
            "Autor",
        ]
    )
    captured = capsys.readouterr()

    assert code == 0
    assert (chapters.parent / "11-translated-book.epub").is_file()
    assert "title           : Naslov" in captured.out
    assert "verification    : mimetype ok" in captured.out
    assert "OK — the postprocessed chapters were packaged" in captured.out


def test_the_cli_reports_a_missing_chapter_folder(tmp_path, capsys):
    code = cli.main(["epub", "--postprocessed-in", str(tmp_path)])

    assert code == 2
    assert "run `postprocess` first" in capsys.readouterr().err


def test_the_cli_json_result_is_machine_readable(tmp_path, capsys):
    chapters = make_book(tmp_path)

    code = cli.main(["epub", "--postprocessed-in", str(chapters), "--json", "--quiet"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["document_count"] == 4
    assert payload["title"] == "Alisa u zemlji čuda"
    assert payload["language"] == "sr"


# --------------------------------------------------------------------------- #
# the verifier itself
# --------------------------------------------------------------------------- #
def _tampered(path: Path, *, mimetype: bytes, compressed: bool = False) -> bytes:
    """A tiny hand-built package, so the verifier can be fed a broken one."""
    with zipfile.ZipFile(path, "w") as archive:
        if compressed:
            info = zipfile.ZipInfo("mimetype")
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, mimetype)
        else:
            archive.writestr("mimetype", mimetype)
        archive.writestr("META-INF/container.xml", b"<container/>")
    return path.read_bytes()


def test_the_verifier_catches_a_wrong_mimetype(tmp_path):
    data = _tampered(tmp_path / "bad.epub", mimetype=b"text/plain")

    problems, _report = _verify(data, documents=[], images=[], cover=None)

    assert any("does not say application/epub+zip" in problem for problem in problems)
    assert any("names no package document" in problem for problem in problems)


def test_the_verifier_catches_a_compressed_mimetype(tmp_path):
    data = _tampered(
        tmp_path / "bad.epub", mimetype=b"application/epub+zip", compressed=True
    )

    problems, _report = _verify(data, documents=[], images=[], cover=None)

    assert any("mimetype entry is compressed" in problem for problem in problems)


def test_the_verifier_reports_a_truncated_package_instead_of_raising(tmp_path):
    chapters = make_book(tmp_path)
    result = run_epub(chapters, write_manifest=False)
    data = Path(result.output_path).read_bytes()

    problems, _report = _verify(data[:120], documents=[], images=[], cover=None)

    assert any("not a readable zip" in problem for problem in problems)


def test_entries_use_the_newest_chapter_timestamp(tmp_path):
    chapters = make_book(tmp_path)
    newest = max(path.stat().st_mtime for path in chapters.glob("*.md"))

    result = run_epub(chapters, write_manifest=False)

    with open_package(Path(result.output_path)) as archive:
        stamps = {info.date_time for info in archive.infolist()}
    assert len(stamps) == 1
    assert posixpath.basename(result.output_path) == "11-translated-book.epub"
    # The zip stores local time, so compare through the reported ISO timestamp.
    assert result.modified.endswith("Z")
    assert newest <= Path(chapters / "01-prvo.md").stat().st_mtime + 1
