"""Stage 0 — turn an EPUB, a text-based PDF or a TXT into the Markdown stage 1 reads.

The splitter is deliberately line-based: it copies Markdown lines verbatim and
never re-flows prose. That contract needs one honest conversion up front instead
of a second converter hidden inside the splitter, so this module is a stage of
its own and writes to ``00-markdown/``, before ``01-chapters/``.

Three source kinds are accepted:

``.md`` / ``.markdown``
    Already Markdown. The stage is a no-op and stage 1 reads the file directly.

``.txt``
    Already plain text. The bytes are copied unchanged; only the name changes to
    ``.md``, exactly as asked.

``.epub`` / ``.pdf``
    Converted with docling. For a PDF the text layer is checked **before** the
    conversion starts: an image-based PDF is rejected with an actionable error
    rather than OCR'd. OCR would invent a different book, and this pipeline only
    translates text the source actually contains. docling is imported lazily, so
    a TXT/Markdown run needs no docling installation.

Every conversion writes ``_convert_manifest.json`` next to the Markdown, tying
the output to the source hash and the method that produced it. A second run
reuses the output while that hash still matches; ``force`` re-converts anyway.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..core.artifacts import read_json, sha256_file, timestamp, write_json, write_text
from ..core.constants import (
    CHAPTER_IMAGE_PREFIX,
    CONVERT_MANIFEST_FILENAME,
    IMAGES_DIRNAME,
    MARKDOWN_DIRNAME,
    MIN_CONVERTED_MARKDOWN_CHARS,
    PDF_MIN_TEXT_CHARS_PER_PAGE,
    PDF_MIN_TEXT_PAGE_RATIO,
    PDF_TEXT_SAMPLE_PAGES,
)
from ..core.errors import MissingDependencyError, ToolError
from ..documents.images import image_count
from ..paths import SOURCE_SUFFIXES, book_markdown_dir, book_slug

#: Formats whose content is already Markdown (stage 0 does nothing).
MARKDOWN_SUFFIXES = (".md", ".markdown")
#: Plain text: copied byte for byte into a ``.md`` file.
TEXT_SUFFIXES = (".txt",)
#: Formats docling converts.
DOCLING_SUFFIXES = (".pdf", ".epub")

_READ_CHUNK = 1024 * 1024


@dataclass
class ConversionResult:
    """What stage 0 did for one source document."""

    source: str
    source_kind: str
    markdown_path: str
    method: str
    ok: bool = True
    written: bool = False
    reused: bool = False
    dry_run: bool = False
    bytes: int = 0
    pages: int | None = None
    sampled_pages: int | None = None
    text_pages: int | None = None
    text_based: bool | None = None
    #: Files written to ``00-markdown/images/`` (0 when the source has no pictures).
    images: int = 0
    #: Image references in the converted Markdown (after duplicate files are merged).
    image_refs: int = 0
    images_dir: str | None = None
    error: str | None = None
    log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
def is_markdown_source(path: str | Path) -> bool:
    """True when the file is already Markdown and needs no conversion."""
    return Path(path).suffix.lower() in MARKDOWN_SUFFIXES


def is_supported_source(path: str | Path) -> bool:
    """True when stage 0 knows how to turn this file into Markdown."""
    return Path(path).suffix.lower() in SOURCE_SUFFIXES


def classify_source(path: str | Path) -> str:
    """``"markdown"``, ``"text"``, ``"pdf"`` or ``"epub"``, checked on disk.

    Raises :class:`ToolError` for an unsupported extension instead of letting
    docling fail with an opaque parser error.
    """
    source = Path(path).expanduser()
    if not source.is_file():
        raise ToolError(f"source document not found: {source}")

    suffix = source.suffix.lower()
    if suffix in MARKDOWN_SUFFIXES:
        return "markdown"
    if suffix in TEXT_SUFFIXES:
        return "text"
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".epub":
        return "epub"
    supported = ", ".join(SOURCE_SUFFIXES)
    raise ToolError(
        f"unsupported source format {suffix or '(no extension)'}: {source.name} — "
        f"supported formats: {supported}"
    )


def converted_markdown_path(
    book: str | Path,
    out_dir: str | Path | None = None,
    *,
    output_root: str | Path | None = None,
) -> Path:
    """Where stage 0 writes a converted source (``<slug>.md``)."""
    folder = (
        Path(out_dir).expanduser()
        if out_dir is not None and str(out_dir).strip()
        else book_markdown_dir(book, output_root=output_root)
    )
    return folder / f"{book_slug(book)}.md"


# --------------------------------------------------------------------------- #
# the stage
# --------------------------------------------------------------------------- #
def run_conversion(
    source: str | Path,
    out_dir: str | Path | None = None,
    *,
    output_root: str | Path | None = None,
    dry_run: bool = False,
    force: bool = False,
    images: bool = True,
) -> ConversionResult:
    """Convert one source document to Markdown under ``00-markdown/``.

    ``dry_run`` validates the source (including a PDF's text layer) and reports
    what would happen without running docling or writing a file. An image-based
    PDF raises :class:`ToolError` in both modes: OCR is deliberately not an
    option here.

    With ``images`` (the default) docling extracts the pictures of an EPUB/PDF
    into ``images/`` next to the Markdown and the Markdown references them. The
    links are written relative to the book folder (``../00-markdown/images/…``),
    which is correct for every stage folder one level below it. ``images=False``
    keeps the old text-only conversion.
    """
    book = Path(source).expanduser()
    kind = classify_source(book)
    target = converted_markdown_path(book, out_dir, output_root=output_root)
    image_dir = target.parent / IMAGES_DIRNAME
    # Every later stage lives one folder below the book output folder, where
    # `../00-markdown/images/` is the same path. A caller-supplied --out outside
    # the canonical layout only gets a local `images/` reference.
    canonical = target.parent.name == MARKDOWN_DIRNAME
    image_prefix = CHAPTER_IMAGE_PREFIX if canonical else f"{IMAGES_DIRNAME}/"
    result = ConversionResult(
        source=str(book.resolve()),
        source_kind=kind,
        markdown_path=str(target if kind != "markdown" else book.resolve()),
        method="already-markdown" if kind == "markdown" else "",
        dry_run=dry_run,
        bytes=book.stat().st_size,
        images_dir=str(image_dir) if kind in {"pdf", "epub"} and images else None,
    )

    if kind == "markdown":
        result.log.append("source is already Markdown — stage 1 can read it directly")
        return result

    if kind == "pdf":
        report = inspect_pdf_text_layer(book)
        result.pages = report["pages"]
        result.sampled_pages = report["sampled_pages"]
        result.text_pages = report["text_pages"]
        result.text_based = report["text_based"]
        result.log.append(
            f"PDF text layer: {report['text_pages']}/{report['sampled_pages']} sampled "
            f"page(s) carry text ({report['chars']:,} characters)"
        )
        if not report["text_based"]:
            raise ToolError(_image_based_pdf_message(book, report))

    manifest_path = target.parent / CONVERT_MANIFEST_FILENAME
    if not force and target.is_file():
        manifest = _load_manifest(manifest_path)
        if _manifest_matches(
            manifest, book, target, image_dir=image_dir if result.images_dir else None
        ):
            result.method = str(manifest.get("method") or _method_for(kind))
            result.reused = True
            result.written = False
            result.images = int(manifest.get("images") or 0)
            result.image_refs = int(manifest.get("image_refs") or 0)
            result.log.append(f"reusing the existing Markdown: {target.name}")
            return result

    if dry_run:
        result.method = _method_for(kind)
        if result.images_dir:
            result.log.append(
                "images will be extracted to "
                f"{Path(result.images_dir).relative_to(target.parent)}/"
            )
        result.log.append("dry run — the conversion was validated but nothing was written")
        return result

    target.parent.mkdir(parents=True, exist_ok=True)
    if kind == "text":
        # "Just change the file name": copy the bytes, never re-encode the text.
        target.write_bytes(book.read_bytes())
        result.method = "text-copy"
        result.written = True
        result.log.append(f"copied {book.name} to {target.name} (content unchanged)")
    else:
        markdown = _convert_with_docling(
            book,
            kind,
            image_dir=image_dir if result.images_dir else None,
            image_prefix=image_prefix,
        )
        if len(markdown.strip()) < MIN_CONVERTED_MARKDOWN_CHARS:
            raise ToolError(
                f"docling produced no usable Markdown from {book.name} — the source may be "
                "empty, encrypted or built only from images"
            )
        result.images = len([p for p in image_dir.glob("*") if p.is_file()]) if result.images_dir else 0
        result.image_refs = image_count(markdown)
        write_text(target, markdown)
        result.method = _method_for(kind)
        result.written = True
        result.bytes = len(markdown.encode("utf-8"))
        if result.images and result.images_dir:
            result.log.append(
                f"extracted {result.images} image(s) to "
                f"{Path(result.images_dir).relative_to(target.parent)}/"
            )
        result.log.append(f"converted {book.name} to {target.name} with docling")

    _write_manifest(manifest_path, book, target, result)
    return result


def _method_for(kind: str) -> str:
    return {"text": "text-copy", "pdf": "docling-pdf", "epub": "docling-epub"}.get(kind, kind)


# --------------------------------------------------------------------------- #
# PDF text-layer preflight (the no-OCR gate)
# --------------------------------------------------------------------------- #
def inspect_pdf_text_layer(
    path: str | Path,
    *,
    sample_pages: int = PDF_TEXT_SAMPLE_PAGES,
) -> dict:
    """Sample pages of a PDF and decide whether it carries an embedded text layer.

    The heavy converters are never asked to "have a go": a scan whose only text
    is a page number must be reported as image-based so the caller can skip it.
    Returns a report dict with ``text_based`` and the evidence behind it.
    """
    book = Path(path).expanduser()
    try:
        total, texts = _read_pdf_page_texts(book, sample_pages)
    except ToolError:
        raise
    except Exception as exc:  # corrupt/encrypted PDFs surface as vendor errors
        raise ToolError(f"could not read the PDF text layer of {book.name}: {exc}") from exc

    counts = [sum(1 for character in text if character.isalnum()) for text in texts]
    text_pages = sum(1 for count in counts if count >= PDF_MIN_TEXT_CHARS_PER_PAGE)
    sampled = len(counts)
    report = {
        "pages": total,
        "sampled_pages": sampled,
        "text_pages": text_pages,
        "chars": sum(counts),
        "text_based": bool(
            sampled and text_pages and (text_pages / sampled) >= PDF_MIN_TEXT_PAGE_RATIO
        ),
        "sample_indices": _sample_indices(total, sample_pages),
    }
    return report


def _image_based_pdf_message(book: Path, report: dict) -> str:
    return (
        f"{book.name} has no embedded text layer "
        f"({report['text_pages']} of {report['sampled_pages']} sampled page(s) carry text, "
        f"{report['chars']} alphanumeric characters in total). OCR is deliberately "
        "disabled for image-based PDFs: run OCR on the PDF first (or supply the "
        "EPUB/TXT edition), then re-run stage 0."
    )


def _sample_indices(total: int, limit: int) -> list[int]:
    """Evenly spread at most ``limit`` page indices across the document."""
    if total <= 0:
        return []
    if limit <= 0 or total <= limit:
        return list(range(total))
    step = total / limit
    return sorted({min(total - 1, int(index * step)) for index in range(limit)})


def _read_pdf_page_texts(path: Path, sample_pages: int) -> tuple[int, list[str]]:
    """Page count and the text of the sampled pages, using pypdfium2 or PyMuPDF.

    ``pypdfium2`` ships with docling, so the common case needs no extra install;
    PyMuPDF is the fallback for a splitter-only environment that already had it.
    """
    try:
        import pypdfium2 as pdfium  # type: ignore
    except ImportError:
        return _read_pdf_page_texts_fitz(path, sample_pages)

    document = pdfium.PdfDocument(str(path))
    try:
        total = len(document)
        texts: list[str] = []
        for index in _sample_indices(total, sample_pages):
            page = document[index]
            textpage = None
            try:
                textpage = page.get_textpage()
                texts.append(textpage.get_text_range())
            finally:
                _close(textpage)
                _close(page)
        return total, texts
    finally:
        _close(document)


def _read_pdf_page_texts_fitz(path: Path, sample_pages: int) -> tuple[int, list[str]]:
    try:
        import fitz  # type: ignore  # PyMuPDF
    except ImportError as exc:
        raise MissingDependencyError(
            "reading a PDF's text layer needs pypdfium2 (installed with docling) or "
            "PyMuPDF: pip install pypdfium2"
        ) from exc

    document = fitz.open(str(path))
    try:
        total = int(document.page_count)
        texts = [
            document.load_page(index).get_text("text")
            for index in _sample_indices(total, sample_pages)
        ]
        return total, texts
    finally:
        document.close()


def _close(obj) -> None:
    close = getattr(obj, "close", None)
    if callable(close):
        close()


# --------------------------------------------------------------------------- #
# docling
# --------------------------------------------------------------------------- #
def _convert_with_docling(
    path: Path,
    kind: str,
    *,
    image_dir: Path | None = None,
    image_prefix: str = "",
) -> str:
    """Convert a PDF/EPUB to Markdown with docling, with OCR switched off.

    For a PDF the text layer was already verified by :func:`inspect_pdf_text_layer`;
    ``do_ocr = False`` makes sure docling never silently falls back to OCR. Table
    structure is off too: a novel's tables are not the point, and the model that
    recognises them is an avoidable download.

    When ``image_dir`` is given the document's pictures are exported there and the
    Markdown references them with ``image_prefix``. EPUB images are fetched from
    the archive; PDF pictures are cropped by the same layout model that already
    locates the text.
    """
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise MissingDependencyError(
            "converting PDF/EPUB needs docling: pip install docling "
            "(TXT and Markdown sources need no dependency)"
        ) from exc

    fetch_images = image_dir is not None
    if kind == "pdf":
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        options = PdfPipelineOptions()
        options.do_ocr = False
        options.do_table_structure = False
        options.generate_page_images = False
        options.generate_picture_images = fetch_images
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )
    elif fetch_images:
        from docling.datamodel.backend_options import EpubBackendOptions
        from docling.document_converter import EpubFormatOption

        # EPUB is a zip of XHTML: the backend extracts it to a temp folder so the
        # HTML image loader may read the pictures from disk.
        converter = DocumentConverter(
            format_options={
                InputFormat.EPUB: EpubFormatOption(
                    backend_options=EpubBackendOptions(
                        # docling's default: never fetch remote EPUB resources.
                        enable_remote_fetch=False,
                        fetch_images=True,
                        enable_local_fetch=True,
                    )
                )
            }
        )
    else:
        converter = DocumentConverter()

    try:
        converted = converter.convert(str(path))
        if image_dir is None:
            return converted.document.export_to_markdown()
        from docling_core.types.doc import ImageRefMode

        image_dir.mkdir(parents=True, exist_ok=True)
        markdown = converted.document.export_to_markdown(
            image_mode=ImageRefMode.REFERENCED,
            image_dir=image_dir,
            image_uri_prefix=image_prefix,
        )
        return _dedupe_images(markdown, image_dir, image_prefix)
    except Exception as exc:
        raise ToolError(f"docling could not convert {path.name}: {exc}") from exc


def _dedupe_images(markdown: str, image_dir: Path, image_prefix: str) -> str:
    """Keep one file per distinct picture and repoint its duplicates.

    docling names each exported picture by its position as well as its content
    hash, so a section break used ten times becomes ten identical PNGs. Merging
    them keeps the output book's image folder proportional to the book, not to
    the number of times an ornament is repeated.
    """
    by_digest: dict[str, str] = {}
    for path in sorted(image_dir.iterdir()):
        if not path.is_file():
            continue
        digest = _sha256(path)
        keeper = by_digest.get(digest)
        if keeper is None:
            by_digest[digest] = path.name
            continue
        markdown = markdown.replace(f"{image_prefix}{path.name}", f"{image_prefix}{keeper}")
        path.unlink()
    return markdown


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
def _write_manifest(
    manifest_path: Path, source: Path, target: Path, result: ConversionResult
) -> None:
    payload = {
        "stage": 0,
        "source": str(source.resolve()),
        "source_kind": result.source_kind,
        "source_sha256": _sha256(source),
        "source_bytes": source.stat().st_size,
        "output": target.name,
        "output_sha256": _sha256(target),
        "method": result.method,
        "docling_version": (
            _docling_version() if result.source_kind in {"pdf", "epub"} else None
        ),
        "pages": result.pages,
        "sampled_pages": result.sampled_pages,
        "text_pages": result.text_pages,
        "text_based": result.text_based,
        "images": result.images,
        "image_refs": result.image_refs,
        "images_dir": Path(result.images_dir).name if result.images_dir else None,
        "created": timestamp(),
    }
    write_json(manifest_path, payload, trailing_newline=True)


def _load_manifest(path: Path) -> dict:
    payload = read_json(path, default={})
    return payload if isinstance(payload, dict) else {}


def _manifest_matches(
    manifest: dict,
    source: Path,
    target: Path,
    *,
    image_dir: Path | None = None,
) -> bool:
    if not manifest:
        return False
    if manifest.get("source") != str(source.resolve()):
        return False
    if manifest.get("source_sha256") != _sha256(source):
        return False
    if manifest.get("output_sha256") != _sha256(target):
        return False
    # A converted source with pictures is only reusable when the pictures are
    # still there; a deleted/partial images folder must trigger a conversion.
    if image_dir is None:
        return True
    expected = int(manifest.get("images") or 0)
    if expected <= 0:
        return True
    present = sum(1 for path in image_dir.glob("*") if path.is_file())
    return present >= expected


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _docling_version() -> str | None:
    try:
        import docling  # type: ignore
    except ImportError:
        return None
    return getattr(docling, "__version__", None)
