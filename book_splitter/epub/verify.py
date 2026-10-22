"""Post-write verification of the EPUB package.

Every check returns a sentence, never an exception: the package is written even
when something is wrong, and the caller reports what is wrong with it. The
package is read once into memory (text entries only), so a truncated or corrupt
zip is a problem in the list, not a traceback in the reader's face.
"""

from __future__ import annotations

import io
import posixpath
import zipfile
from typing import Any
from xml.etree import ElementTree as ET

from ..core.constants import EPUB_MIMETYPE
from .package import COVER_HREF, ImageAsset, SpineDocument

_NS_OPF = "{http://www.idpf.org/2007/opf}"
_NS_XHTML = "{http://www.w3.org/1999/xhtml}"
_NS_OPS = "{http://www.idpf.org/2007/ops}"
_NS_CONTAINER = "{urn:oasis:names:tc:opendocument:xmlns:container}"
_NS_DC = "{http://purl.org/dc/elements/1.1/}"
#: Self-contained pictures never travel into the package.
_REMOTE_PREFIXES = ("http://", "https://", "data:", "//")


def verify(
    package: bytes,
    *,
    documents: list[SpineDocument],
    images: list[ImageAsset],
    cover: ImageAsset | None,
) -> tuple[list[str], dict[str, Any]]:
    """Re-read the built package and prove its shape."""
    problems: list[str] = []
    report: dict[str, Any] = {
        "entries": 0,
        "documents": len(documents),
        "images": len(images),
        "cover": cover.href if cover is not None else None,
        "mimetype_first": False,
        "container_ok": False,
        "opf_ok": False,
        "nav_ok": False,
        "spine_count": 0,
        "xhtml_valid": 0,
        "image_links_ok": 0,
    }
    try:
        with zipfile.ZipFile(io.BytesIO(package)) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            contents = {
                name: archive.read(name)
                for name in names
                # Text entries only: a picture book's images are not checks.
                if name == "mimetype" or name.endswith((".opf", ".xhtml", ".xml"))
            }
    except (zipfile.BadZipFile, OSError, EOFError, KeyError) as exc:
        return ([f"the package is not a readable zip file: {exc}"], report)
    report["entries"] = len(names)

    # 1. the mandatory first, uncompressed mimetype entry
    mimetype_ok = True
    if not names or names[0] != "mimetype":
        problems.append("the mimetype entry is not the first entry of the package")
        mimetype_ok = False
    else:
        if infos[0].compress_type != zipfile.ZIP_STORED:
            problems.append("the mimetype entry is compressed")
            mimetype_ok = False
        if contents.get("mimetype") != EPUB_MIMETYPE.encode("ascii"):
            problems.append("the mimetype entry does not say application/epub+zip")
            mimetype_ok = False
    report["mimetype_first"] = mimetype_ok

    # 2. the container points at a package document that exists
    opf_path = ""
    try:
        container = ET.fromstring(contents.get("META-INF/container.xml") or b"")
        rootfile = container.find(f".//{_NS_CONTAINER}rootfile")
        opf_path = (rootfile.get("full-path") if rootfile is not None else "") or ""
    except ET.ParseError:
        problems.append("META-INF/container.xml is unreadable")
    if not opf_path:
        problems.append("META-INF/container.xml names no package document")
    elif opf_path not in names:
        problems.append(f"the package document is missing: {opf_path}")
    else:
        report["container_ok"] = True

    base = posixpath.dirname(opf_path) or "."

    def in_package(href: str) -> str:
        return posixpath.join(base, href) if base != "." else href

    if opf_path and opf_path in names:
        check_opf(contents, names, opf_path, base, problems, report)

    # 3. the navigation document lists every document and links to real targets
    nav_path = in_package("nav.xhtml")
    if nav_path in names:
        check_nav(contents, names, nav_path, problems, report)
    else:
        problems.append("the navigation document is missing")

    # 4. every text document is well-formed and links to pictures that exist
    for document in documents:
        path = in_package(document.href)
        if path not in names:
            problems.append(f"spine document missing from the package: {document.href}")
            continue
        try:
            root = ET.fromstring(contents.get(path) or b"")
        except ET.ParseError as exc:
            problems.append(f"{document.href} is not well-formed XHTML: {exc}")
            continue
        report["xhtml_valid"] += 1
        title = root.find(f".//{_NS_XHTML}title")
        if title is None or not (title.text or "").strip():
            problems.append(f"{document.href} has no document title")
        missing = _missing_picture_links(names, path, root)
        if missing:
            problems.append(f"{document.href} references missing picture(s): {missing}")
        else:
            report["image_links_ok"] += 1

    # 5. the cover page, when there is one, points at the cover picture
    if cover is not None:
        if in_package(COVER_HREF) not in names:
            problems.append("the cover page is missing from the package")
        if in_package(cover.href) not in names:
            problems.append(f"the cover picture is missing: {cover.href}")

    return problems, report


def check_opf(
    contents: dict[str, bytes],
    names: list[str],
    opf_path: str,
    base: str,
    problems: list[str],
    report: dict[str, Any],
) -> None:
    before = len(problems)
    report["opf_ok"] = False
    try:
        opf = ET.fromstring(contents.get(opf_path) or b"")
    except ET.ParseError as exc:
        problems.append(f"the package document is not well-formed XML: {exc}")
        return
    if opf.get("version") != "3.0":
        problems.append(f"the package document is not EPUB 3 (version={opf.get('version')})")
    metadata = opf.find(f"{_NS_OPF}metadata")
    for tag, label in (
        (f"{_NS_DC}title", "dc:title"),
        (f"{_NS_DC}language", "dc:language"),
        (f"{_NS_DC}identifier", "dc:identifier"),
    ):
        node = metadata.find(tag) if metadata is not None else None
        if node is None or not (node.text or "").strip():
            problems.append(f"the package metadata has no {label}")
    modified = (
        metadata.find(f"{_NS_OPF}meta[@property='dcterms:modified']")
        if metadata is not None
        else None
    )
    if modified is None or not (modified.text or "").strip():
        problems.append("the package metadata has no dcterms:modified date")

    manifest = opf.find(f"{_NS_OPF}manifest")
    items = list(manifest) if manifest is not None else []
    by_id: dict[str, str] = {}
    by_href: set[str] = set()
    for item in items:
        item_id = item.get("id") or ""
        href = item.get("href") or ""
        if not item_id or not href:
            problems.append("a manifest item has no id or no href")
            continue
        if item_id in by_id:
            problems.append(f"two manifest items share the id {item_id!r}")
        if href in by_href:
            problems.append(f"two manifest items point at the same file: {href}")
        by_id[item_id] = href
        by_href.add(href)
        target = posixpath.join(base, href) if base != "." else href
        if target not in names:
            problems.append(f"the manifest lists a file the package does not contain: {href}")

    spine = opf.find(f"{_NS_OPF}spine")
    itemrefs = list(spine) if spine is not None else []
    report["spine_count"] = len(itemrefs)
    for itemref in itemrefs:
        idref = itemref.get("idref") or ""
        if idref not in by_id:
            problems.append(f"the spine points at an unknown manifest item: {idref!r}")
    nav = next((item for item in items if "nav" in (item.get("properties") or "")), None)
    if nav is None:
        problems.append('the manifest has no navigation document (properties="nav")')
    report["opf_ok"] = len(problems) == before


def check_nav(
    contents: dict[str, bytes],
    names: list[str],
    nav_path: str,
    problems: list[str],
    report: dict[str, Any],
) -> None:
    before = len(problems)
    report["nav_ok"] = False
    try:
        nav = ET.fromstring(contents.get(nav_path) or b"")
    except ET.ParseError as exc:
        problems.append(f"the navigation document is not well-formed XHTML: {exc}")
        return
    toc = next(
        (
            element
            for element in nav.iter(f"{_NS_XHTML}nav")
            if element.get(f"{_NS_OPS}type") == "toc"
        ),
        None,
    )
    if toc is None:
        problems.append('the navigation document has no <nav epub:type="toc">')
        return
    links = list(toc.iter(f"{_NS_XHTML}a"))
    if not links:
        problems.append("the navigation document has no entries")
        return
    for link in links:
        href = link.get("href") or ""
        if not href:
            problems.append("a navigation entry has no target")
            continue
        target = posixpath.normpath(
            posixpath.join(posixpath.dirname(nav_path), href.split("#", 1)[0])
        )
        if target not in names:
            problems.append(f"a navigation entry points at a missing document: {href}")
    report["nav_ok"] = len(problems) == before


def _missing_picture_links(
    names: list[str], document_path: str, root: ET.Element
) -> list[str]:
    missing: list[str] = []
    for element in root.iter(f"{_NS_XHTML}img"):
        src = element.get("src") or ""
        if not src or src.lower().startswith(_REMOTE_PREFIXES):
            continue
        target = posixpath.normpath(
            posixpath.join(posixpath.dirname(document_path), src)
        )
        if target not in names:
            missing.append(src)
    return missing


__all__ = ["check_nav", "check_opf", "verify"]
