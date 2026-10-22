"""The EPUB 3 container: the zip, the package document and the navigation document.

An EPUB is an ordinary zip with a strict shape. The first entry must be a stored
(never compressed) ``mimetype`` file that says ``application/epub+zip``;
``META-INF/container.xml`` says where the package document is; the package
document (``content.opf``) lists every file, the metadata, the reading order and
the navigation document; and EPUB 3 requires that navigation document to be an
XHTML file with ``<nav epub:type="toc">``.

This module writes that shape from plain inputs — a list of rendered text
documents, a list of pictures and one stylesheet — and nothing else. It has no
opinion about Markdown (see :mod:`book_splitter.epub.render`), about where the
files come from (see :mod:`book_splitter.epub.runner`) or about which title a book
has (see :mod:`book_splitter.epub.metadata`).

The output is deterministic: zip timestamps come from the book's own newest
source file, entries are written in a fixed order, and item ids are positional.
Two runs over unchanged input produce byte-identical files.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from typing import Any
from xml.sax.saxutils import escape

from ..core.constants import (
    EPUB_MIMETYPE,
    EPUB_STYLESHEET_FILENAME,
    EPUB_TEXT_DIRNAME,
)
from .metadata import BookMetadata

#: The container's fixed entry point.
CONTAINER_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<container version="1.0" '
    'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">\n'
    "  <rootfiles>\n"
    '    <rootfile full-path="OEBPS/content.opf" '
    'media-type="application/oebps-package+xml"/>\n'
    "  </rootfiles>\n"
    "</container>\n"
)
#: Where the package document and everything it lists live.
CONTENT_DIRNAME = "OEBPS"
#: The navigation document's location inside the container.
NAV_HREF = "nav.xhtml"
#: The stylesheet's location inside the container.
STYLESHEET_HREF = f"css/{EPUB_STYLESHEET_FILENAME}"
#: Cover page document, written only when a cover picture is supplied.
COVER_HREF = f"{EPUB_TEXT_DIRNAME}/0000-cover.xhtml"

#: Suffix -> media type. Anything unknown is reported by ``media_type_for``.
MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


@dataclass
class SpineDocument:
    """One text document of the reading order."""

    href: str
    title: str
    body: str
    source: str = ""
    matter: str = "body"
    headings: list[dict[str, Any]] = field(default_factory=list)
    chars: int = 0


@dataclass
class ImageAsset:
    """One picture inside the container."""

    href: str
    media_type: str
    data: bytes
    source: str = ""
    sha256: str = ""


def media_type_for(name: str) -> str:
    """The declared media type of a picture, or ``application/octet-stream``."""
    suffix = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""
    return MEDIA_TYPES.get(suffix, "application/octet-stream")


def xhtml_document(
    *,
    title: str,
    body: str,
    language: str,
    css_href: str | None = None,
    extra_head: str = "",
) -> str:
    """Wrap a body fragment in a complete, self-contained XHTML document."""
    head = [
        "<head>",
        '  <meta charset="utf-8"/>',
        f"  <title>{_x(title)}</title>",
    ]
    if css_href:
        head.append(f'  <link rel="stylesheet" type="text/css" href="{_x(css_href)}"/>')
    if extra_head:
        head.append(f"  {extra_head}")
    head.append("</head>")
    lang = _x(language or "und")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<!DOCTYPE html>\n"
        '<html xmlns="http://www.w3.org/1999/xhtml" '
        'xmlns:epub="http://www.idpf.org/2007/ops" '
        f'xml:lang="{lang}" lang="{lang}">\n'
        + "\n".join(head)
        + "\n<body>\n"
        + body.rstrip("\n")
        + "\n</body>\n</html>\n"
    )


def build_epub(
    *,
    metadata: BookMetadata,
    documents: list[SpineDocument],
    images: list[ImageAsset],
    stylesheet: str,
    cover: ImageAsset | None = None,
) -> bytes:
    """Assemble the whole container in memory and return its bytes."""
    when = metadata.zip_datetime
    entries: list[tuple[str, bytes, bool]] = [
        ("mimetype", EPUB_MIMETYPE.encode("ascii"), True),
        ("META-INF/container.xml", CONTAINER_XML.encode("utf-8"), False),
        (f"{CONTENT_DIRNAME}/content.opf", b"", False),  # filled in below
        (f"{CONTENT_DIRNAME}/{NAV_HREF}", b"", False),
        (f"{CONTENT_DIRNAME}/{STYLESHEET_HREF}", stylesheet.encode("utf-8"), False),
    ]
    if cover is not None:
        entries.append(
            (
                f"{CONTENT_DIRNAME}/{COVER_HREF}",
                _cover_document(metadata, cover).encode("utf-8"),
                False,
            )
        )
    for document in documents:
        entries.append(
            (f"{CONTENT_DIRNAME}/{document.href}", _document_xhtml(document, metadata).encode("utf-8"), False)
        )
    for image in images:
        entries.append((f"{CONTENT_DIRNAME}/{image.href}", image.data, False))

    opf = _package_document(metadata, documents, images, cover)
    nav = _nav_document(metadata, documents, cover)
    entries[2] = (entries[2][0], opf.encode("utf-8"), False)
    entries[3] = (entries[3][0], nav.encode("utf-8"), False)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data, stored in entries:
            archive.writestr(_zip_info(name, when, stored=stored), data)
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# documents
# --------------------------------------------------------------------------- #
def _document_xhtml(document: SpineDocument, metadata: BookMetadata) -> str:
    css_href = f"../{STYLESHEET_HREF}"
    return xhtml_document(
        title=document.title or metadata.title,
        body=document.body,
        language=metadata.language,
        css_href=css_href,
    )


def _cover_document(metadata: BookMetadata, cover: ImageAsset) -> str:
    body = (
        '<div class="cover">\n'
        f'  <img src="../{_x(cover.href)}" alt="{_x(metadata.title)}"/>\n'
        "</div>"
    )
    return xhtml_document(
        title=metadata.title,
        body=body,
        language=metadata.language,
        extra_head="<style>html,body{margin:0;padding:0;height:100%}"
        ".cover{height:100%;text-align:center}"
        ".cover img{max-width:100%;max-height:100%}</style>",
    )


def _zip_info(
    name: str, when: tuple[int, int, int, int, int, int], *, stored: bool = False
) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=when)
    info.compress_type = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    info.create_system = 3  # unix, so the mode below means what it says
    info.external_attr = 0o644 << 16
    return info


# --------------------------------------------------------------------------- #
# the package document
# --------------------------------------------------------------------------- #
def _package_document(
    metadata: BookMetadata,
    documents: list[SpineDocument],
    images: list[ImageAsset],
    cover: ImageAsset | None,
) -> str:
    lang = _x(metadata.language or "und")
    metadata_lines = [
        f'    <dc:identifier id="book-id">{_x(metadata.identifier)}</dc:identifier>',
        f"    <dc:title>{_x(metadata.title)}</dc:title>",
        f"    <dc:language>{lang}</dc:language>",
    ]
    if metadata.author:
        metadata_lines.append(f"    <dc:creator>{_x(metadata.author)}</dc:creator>")
    if metadata.description:
        metadata_lines.append(
            f"    <dc:description>{_x(metadata.description)}</dc:description>"
        )
    if metadata.publisher:
        metadata_lines.append(f"    <dc:publisher>{_x(metadata.publisher)}</dc:publisher>")
    metadata_lines.append(
        f'    <meta property="dcterms:modified">{_x(metadata.modified)}</meta>'
    )

    manifest_lines = [
        f'    <item id="nav" href="{NAV_HREF}" '
        'media-type="application/xhtml+xml" properties="nav"/>',
        f'    <item id="css" href="{STYLESHEET_HREF}" media-type="text/css"/>',
    ]
    spine_lines: list[str] = []
    if cover is not None:
        manifest_lines.append(
            f'    <item id="cover-image" href="{_x(cover.href)}" '
            f'media-type="{_x(cover.media_type)}" properties="cover-image"/>'
        )
        manifest_lines.append(
            f'    <item id="cover" href="{COVER_HREF}" '
            'media-type="application/xhtml+xml"/>'
        )
        spine_lines.append('    <itemref idref="cover"/>')
    for index, document in enumerate(documents, start=1):
        manifest_lines.append(
            f'    <item id="doc-{index}" href="{_x(document.href)}" '
            'media-type="application/xhtml+xml"/>'
        )
        spine_lines.append(f'    <itemref idref="doc-{index}"/>')
    for index, image in enumerate(images, start=1):
        # The cover picture already has the ``cover-image`` manifest item above;
        # listing it again as ``img-N`` would register the same file twice and
        # leave readers with an ambiguous cover.
        if cover is not None and image.href == cover.href:
            continue
        manifest_lines.append(
            f'    <item id="img-{index}" href="{_x(image.href)}" '
            f'media-type="{_x(image.media_type)}"/>'
        )

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" '
        'unique-identifier="book-id">\n'
        '  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
        + "\n".join(metadata_lines)
        + "\n  </metadata>\n"
        "  <manifest>\n"
        + "\n".join(manifest_lines)
        + "\n  </manifest>\n"
        "  <spine>\n"
        + "\n".join(spine_lines)
        + "\n  </spine>\n"
        "</package>\n"
    )


# --------------------------------------------------------------------------- #
# the navigation document
# --------------------------------------------------------------------------- #
def _nav_document(
    metadata: BookMetadata, documents: list[SpineDocument], cover: ImageAsset | None
) -> str:
    items = "\n".join(_nav_entry(document) for document in documents)
    body = (
        '  <nav epub:type="toc" id="toc">\n'
        "    <ol>\n"
        f"{items}\n"
        "    </ol>\n"
        "  </nav>"
    )
    landmarks = _landmarks(documents, cover)
    if landmarks:
        body += (
            "\n"
            '  <nav epub:type="landmarks" hidden="hidden">\n'
            "    <ol>\n" + "\n".join(landmarks) + "\n    </ol>\n"
            "  </nav>"
        )
    return xhtml_document(
        title=metadata.title,
        body=body,
        language=metadata.language,
        css_href=STYLESHEET_HREF,
    )


def _landmarks(
    documents: list[SpineDocument], cover: ImageAsset | None
) -> list[str]:
    """The structural entry points a reader offers: cover, then the body.

    There is deliberately no ``toc`` landmark: the book ships no table-of-contents
    page (the navigation document *is* the table of contents), and pointing the
    reader at a page that does not exist is worse than omitting the entry.
    """
    entries: list[str] = []
    if cover is not None:
        entries.append(f'      <li><a epub:type="cover" href="{COVER_HREF}">Cover</a></li>')
    body_document = next(
        (document for document in documents if document.matter == "body"), None
    )
    if body_document is not None:
        entries.append(
            f'      <li><a epub:type="bodymatter" href="{_x(body_document.href)}">'
            f"{_x(body_document.title)}</a></li>"
        )
    return entries


def _nav_entry(document: SpineDocument) -> str:
    """One spine document as a navigation entry, with its own headings nested."""
    children = _heading_tree(document.headings)
    nested = _nav_list(children, document.href) if children else ""
    line = (
        f'      <li><a href="{_x(document.href)}">{_x(document.title)}</a>'
        + (f"\n{nested}\n      " if nested else "")
        + "</li>"
    )
    return line


def _heading_tree(headings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Nest a document's headings by level, dropping the document's own title.

    The spine entry already stands for the first level-1 heading, so repeating it
    one level deeper would read as a duplicate entry in every reader.
    """
    root: dict[str, Any] = {"level": 0, "children": []}
    stack = [root]
    for heading in headings:
        level = int(heading.get("level") or 0)
        text = str(heading.get("text") or "").strip()
        if level <= 1 or not text:
            continue
        node = {
            "level": level,
            "text": text,
            "anchor": str(heading.get("anchor") or ""),
            "children": [],
        }
        while len(stack) > 1 and stack[-1]["level"] >= level:
            stack.pop()
        stack[-1]["children"].append(node)
        stack.append(node)
    return root["children"]


def _nav_list(nodes: list[dict[str, Any]], document_href: str) -> str:
    lines = ["        <ol>"]
    for node in nodes:
        target = document_href
        if node["anchor"]:
            target = f"{document_href}#{node['anchor']}"
        lines.append(f'          <li><a href="{_x(target)}">{_x(node["text"])}</a>')
        children = node.get("children") or []
        if children:
            lines.append(_nav_list(children, document_href))
            lines.append("          </li>")
        else:
            lines[-1] = lines[-1] + "</li>"
    lines.append("        </ol>")
    return "\n".join(lines)


def _x(value: Any) -> str:
    """Escape text for XML content and attribute values."""
    return escape(str(value or ""), {'"': "&quot;", "'": "&apos;"})


__all__ = [
    "CONTENT_DIRNAME",
    "COVER_HREF",
    "MEDIA_TYPES",
    "NAV_HREF",
    "STYLESHEET_HREF",
    "ImageAsset",
    "SpineDocument",
    "build_epub",
    "media_type_for",
    "xhtml_document",
]
