"""Stage 11 — package the postprocessed chapters as an EPUB 3 book.

Stage 10 turns ``09-postprocessed-chapters/`` into one Markdown file: the reading
copy, and the artifact this pipeline has always ended on. This stage turns the
same folder into the book you can actually open in a reader —
``11-translated-book.epub`` next to it.

It reads the folder, not stage 10's file. One XHTML document per chapter is what
an EPUB wants: the chapter becomes a spine item that starts on a new page, the
navigation document gets one entry per chapter, and the file's own first heading
is both the entry's label and the document's ``<title>``. Re-splitting the
concatenated Markdown on ``#`` headings would have to guess those boundaries back.

Three decisions are deliberate:

**The printed table of contents is dropped.** Stage 9 may have written
``00-table-of-contents.md`` as a numbered list for the printed page. Every reader
shows its own table of contents from the navigation document, so shipping that
page too means two tables of contents, one of them dead. The file is not put in
the book; the navigation document is the table of contents.

**Pictures are embedded once.** A picture the Markdown references is copied into
the container under ``images/`` and its ``src`` is rewritten; the same picture
referenced ten times becomes one file and ten references. A reference that cannot
be resolved is reported as an error and replaced by a named placeholder, so the
book is still complete and the missing picture is visible rather than silent.

**The output is deterministic.** The package timestamp is the newest chapter
file's mtime and zip entries are written in a fixed order, so rebuilding an
unchanged book produces the same bytes (and the same SHA-256).

The stage needs ``markdown-it-py`` to render the Markdown. It is imported lazily
by :mod:`book_splitter.epub.render`, so a pipeline that never reaches stage 11
never needs it.
"""

from __future__ import annotations

import io
import posixpath
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from ..core.artifacts import sha256_bytes, timestamp, write_manifest, write_trace
from ..core.constants import (
    EPUB_IMAGES_DIRNAME,
    EPUB_MANIFEST_FILENAME,
    EPUB_MIMETYPE,
    EPUB_TEXT_DIRNAME,
    EPUB_TRACE_FILENAME,
    IMAGES_DIRNAME,
    MARKDOWN_DIRNAME,
    POSTPROCESSED_CHAPTERS_DIRNAME,
    TOC_FILENAME,
)
from ..core.errors import ToolError
from ..documents.chapters import (
    has_postprocessed_chapter_files,
    is_toc_filename,
    translated_epub_path,
)
from ..postprocess.inspect import chapter_files, scan_book_structure
from .metadata import file_timestamp, resolve_metadata
from .package import COVER_HREF, ImageAsset, SpineDocument, build_epub, media_type_for
from .render import HTML_IMG_SRC_RE, render_markdown, rewrite_images

#: The stylesheet the package carries.
STYLESHEET_PATH = Path(__file__).resolve().parent / "assets" / "style.css"

_NS_OPF = "{http://www.idpf.org/2007/opf}"
_NS_XHTML = "{http://www.w3.org/1999/xhtml}"
_NS_OPS = "{http://www.idpf.org/2007/ops}"
_NS_CONTAINER = "{urn:oasis:names:tc:opendocument:xmlns:container}"
_NS_DC = "{http://purl.org/dc/elements/1.1/}"
#: Self-contained pictures never travel into the package.
_REMOTE_PREFIXES = ("http://", "https://", "data:", "//")


@dataclass
class EpubResult:
    """What stage 11 built for one book."""

    ok: bool
    chapters_path: str
    output_path: str
    title: str = ""
    author: str = ""
    language: str = ""
    identifier: str = ""
    modified: str = ""
    description: str = ""
    publisher: str = ""
    document_count: int = 0
    chapter_count: int = 0
    image_count: int = 0
    chars: int = 0
    bytes: int = 0
    sha256: str = ""
    toc_dropped: bool = False
    cover: str | None = None
    dry_run: bool = False
    spine: list[dict[str, Any]] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    manifest_path: str | None = None
    trace_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_epub(
    postprocessed_path: str | Path,
    output_path: str | Path | None = None,
    *,
    title: str | None = None,
    author: str | None = None,
    language: str | None = None,
    identifier: str | None = None,
    description: str | None = None,
    publisher: str | None = None,
    cover: str | Path | None = None,
    dry_run: bool = False,
    write_manifest: bool = True,
) -> EpubResult:
    """Package a postprocessed chapter folder as an EPUB 3 file.

    ``output_path`` defaults to ``11-translated-book.epub`` next to the chapter
    folder. A folder handed in is understood as the postprocessed folder itself
    (or its parent book folder). A dry run builds and verifies the whole package
    in memory but writes nothing.
    """
    source = _chapters_folder(postprocessed_path)
    if not source.is_dir():
        raise ToolError(f"postprocessed chapters folder not found: {source}")
    if not has_postprocessed_chapter_files(source):
        raise ToolError(
            f"no postprocessed chapter files in {source} — run `postprocess` first"
        )
    files = chapter_files(source)
    if not files:
        raise ToolError(f"no Markdown files to package in {source}")
    target = (
        Path(output_path).expanduser()
        if output_path is not None
        else translated_epub_path(source)
    )

    structure = scan_book_structure(source)
    records = {
        str(record.get("filename") or ""): record
        for record in structure.get("chapters") or []
    }
    metadata = resolve_metadata(
        structure,
        slug=source.parent.name,
        modified=file_timestamp([path.stat().st_mtime for path in files]),
        title=title,
        author=author,
        language=language,
        identifier=identifier,
        description=description,
        publisher=publisher,
    )

    warnings: list[str] = list(metadata.warnings)
    errors: list[str] = []
    assets: dict[str, ImageAsset] = {}
    used_names: set[str] = set()
    cover_asset = (
        _cover_asset(cover, source, assets, used_names, warnings) if cover else None
    )

    documents: list[SpineDocument] = []
    chars = 0
    for path in files:
        text = path.read_text(encoding="utf-8")
        chars += len(text)
        rendered = render_markdown(text, source=path.name)
        warnings.extend(rendered.warnings)
        record = records.get(path.name) or {}
        body = _embed_images(
            rendered.body,
            base_dir=path.parent,
            assets=assets,
            used_names=used_names,
            name=path.name,
            errors=errors,
            warnings=warnings,
        )
        documents.append(
            SpineDocument(
                href=f"{EPUB_TEXT_DIRNAME}/{path.stem}.xhtml",
                title=rendered.title or str(record.get("title") or "") or path.stem,
                body=body,
                source=path.name,
                matter=str(record.get("matter") or "body"),
                headings=rendered.headings,
                chars=rendered.chars,
            )
        )

    package = build_epub(
        metadata=metadata,
        documents=documents,
        images=list(assets.values()),
        stylesheet=STYLESHEET_PATH.read_text(encoding="utf-8"),
        cover=cover_asset,
    )
    problems, verification = _verify(
        package, documents=documents, images=list(assets.values()), cover=cover_asset
    )
    errors.extend(problems)

    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(package)
        written = target.read_bytes()
        if written != package:  # pragma: no cover - a filesystem that lied
            errors.append(f"{target} was written but does not read back byte for byte")
    else:
        written = package

    result = EpubResult(
        ok=not errors,
        chapters_path=str(source),
        output_path=str(target),
        title=metadata.title,
        author=metadata.author,
        language=metadata.language,
        identifier=metadata.identifier,
        modified=metadata.modified,
        description=metadata.description,
        publisher=metadata.publisher,
        document_count=len(documents),
        chapter_count=len(documents),
        image_count=len(assets),
        chars=chars,
        bytes=len(written),
        sha256=sha256_bytes(written),
        toc_dropped=_has_toc_page(source),
        cover=cover_asset.href if cover_asset is not None else None,
        dry_run=dry_run,
        spine=[
            {
                "href": document.href,
                "source": document.source,
                "title": document.title,
                "matter": document.matter,
                "chars": document.chars,
                "headings": len(document.headings),
            }
            for document in documents
        ],
        images=[
            {
                "href": image.href,
                "media_type": image.media_type,
                "bytes": len(image.data),
                "sha256": image.sha256,
                "source": image.source,
            }
            for image in assets.values()
        ],
        verification=verification,
        warnings=warnings,
        errors=errors,
    )
    if write_manifest and not dry_run:
        result.manifest_path = str(write_epub_manifest(target, result))
        result.trace_path = str(write_epub_trace(target, result))
    return result


# --------------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------------- #
def _chapters_folder(raw: str | Path) -> Path:
    source = Path(raw).expanduser()
    if source.is_file():
        source = source.parent
    if source.name != POSTPROCESSED_CHAPTERS_DIRNAME:
        nested = source / POSTPROCESSED_CHAPTERS_DIRNAME
        if nested.is_dir():
            source = nested
    return source


def _has_toc_page(source: Path) -> bool:
    """True when the folder carried a printed table-of-contents page."""
    if (source / TOC_FILENAME).is_file():
        return True
    return any(
        path.is_file() and is_toc_filename(path.name) for path in source.glob("*.md")
    )


# --------------------------------------------------------------------------- #
# pictures
# --------------------------------------------------------------------------- #
def _embed_images(
    body: str,
    *,
    base_dir: Path,
    assets: dict[str, ImageAsset],
    used_names: set[str],
    name: str,
    errors: list[str],
    warnings: list[str],
) -> str:
    """Copy every referenced picture into the container and rewrite its ``src``."""

    def handler(src: str, tag: str) -> str:
        if not src:
            return ""
        if src.lower().startswith(_REMOTE_PREFIXES):
            warnings.append(f"{name}: remote picture left as a link: {src}")
            return tag
        path = Path(src) if Path(src).is_absolute() else base_dir / src
        key = str(path.resolve()) if path.exists() else str(path)
        asset = assets.get(key)
        if asset is None:
            if not path.is_file():
                errors.append(
                    f"{name}: picture not found: {src} "
                    "(run `convert` with images, or restore 00-markdown/images/)"
                )
                return f'<span class="image-missing">[{_escape(src)}]</span>'
            data = path.read_bytes()
            filename = _unique_name(path.name, used_names)
            asset = ImageAsset(
                href=f"{EPUB_IMAGES_DIRNAME}/{filename}",
                media_type=media_type_for(filename),
                data=data,
                source=str(path),
                sha256=sha256_bytes(data),
            )
            if asset.media_type == "application/octet-stream":
                warnings.append(f"{name}: unknown picture type, declared as bytes: {src}")
            assets[key] = asset
        return _replace_src(tag, f"../{asset.href}")

    body, _ = rewrite_images(body, handler)
    return body


def _cover_asset(
    cover: str | Path,
    source: Path,
    assets: dict[str, ImageAsset],
    used_names: set[str],
    warnings: list[str],
) -> ImageAsset:
    """Resolve ``--cover`` against the book folder and register it once."""
    path = _resolve_cover(cover, source)
    key = str(path.resolve())
    asset = assets.get(key)
    if asset is not None:
        return asset
    data = path.read_bytes()
    filename = _unique_name(path.name, used_names)
    asset = ImageAsset(
        href=f"{EPUB_IMAGES_DIRNAME}/{filename}",
        media_type=media_type_for(filename),
        data=data,
        source=str(path),
        sha256=sha256_bytes(data),
    )
    if asset.media_type == "application/octet-stream":
        warnings.append(f"cover picture type is unknown, declared as bytes: {path}")
    assets[key] = asset
    return asset


def _resolve_cover(cover: str | Path, source: Path) -> Path:
    """A cover path, or the same name inside the book's extracted pictures."""
    candidate = Path(cover).expanduser()
    if candidate.is_file():
        return candidate
    pictures = source.parent / MARKDOWN_DIRNAME / IMAGES_DIRNAME
    for option in (pictures / candidate.name, pictures / str(cover)):
        if option.is_file():
            return option
    raise ToolError(
        f"cover picture not found: {cover} (looked next to it and in {pictures})"
    )


def _unique_name(name: str, used: set[str]) -> str:
    """A safe, unique file name inside ``images/``."""
    safe = "".join(
        character
        if character.isalnum() or character in {".", "-", "_"}
        else "-"
        for character in Path(name).name
    ).strip("-") or "image"
    stem, suffix = posixpath.splitext(safe)
    candidate = safe
    bump = 2
    while candidate in used:
        candidate = f"{stem}-{bump}{suffix}"
        bump += 1
    used.add(candidate)
    return candidate


def _replace_src(tag: str, new_src: str) -> str:
    """Replace the ``src`` attribute of an ``<img>`` tag, keeping the rest."""
    return HTML_IMG_SRC_RE.sub(lambda _match: f'src="{new_src}"', tag, count=1)


def _escape(value: str) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
def _verify(
    package: bytes,
    *,
    documents: list[SpineDocument],
    images: list[ImageAsset],
    cover: ImageAsset | None,
) -> tuple[list[str], dict[str, Any]]:
    """Re-read the built package and prove its shape.

    Every check returns a sentence, never an exception: a package is written even
    when something is wrong, and the caller reports what is wrong with it. The
    package is read once into memory (text entries only), so a truncated or
    corrupt zip is a problem in the list, not a traceback in the reader's face.
    """
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
        _check_opf(contents, names, opf_path, base, problems, report)

    # 3. the navigation document lists every document and links to real targets
    nav_path = in_package("nav.xhtml")
    if nav_path in names:
        _check_nav(contents, names, nav_path, problems, report)
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


def _check_opf(
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


def _check_nav(
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


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #
def write_epub_manifest(output_path: Path, result: EpubResult) -> Path:
    payload = {
        "stage": "epub",
        "generated_at": timestamp(),
        "ok": result.ok,
        "dry_run": result.dry_run,
        "chapters_path": result.chapters_path,
        "output_path": result.output_path,
        "bytes": result.bytes,
        "sha256": result.sha256,
        "chars": result.chars,
        "document_count": result.document_count,
        "chapter_count": result.chapter_count,
        "image_count": result.image_count,
        "toc_dropped": result.toc_dropped,
        "cover": result.cover,
        "metadata": {
            "title": result.title,
            "author": result.author,
            "language": result.language,
            "identifier": result.identifier,
            "modified": result.modified,
            "description": result.description,
            "publisher": result.publisher,
        },
        "spine": result.spine,
        "images": result.images,
        "verification": result.verification,
        "warnings": result.warnings,
        "errors": result.errors,
    }
    return write_manifest(output_path.parent, EPUB_MANIFEST_FILENAME, payload)


def write_epub_trace(output_path: Path, result: EpubResult) -> Path:
    return write_trace(output_path.parent, EPUB_TRACE_FILENAME, result.to_dict())


__all__ = [
    "STYLESHEET_PATH",
    "EpubResult",
    "run_epub",
    "write_epub_manifest",
    "write_epub_trace",
]
