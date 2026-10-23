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
by :mod:`book_translator.stages.epub.render`, so a pipeline that never reaches stage 11
never needs it.
"""

from __future__ import annotations

from pathlib import Path

from book_translator.core.constants import (
    EPUB_MANIFEST_FILENAME,
    EPUB_TEXT_DIRNAME,
    EPUB_TRACE_FILENAME,
    POSTPROCESSED_CHAPTERS_DIRNAME,
    TOC_FILENAME,
)
from book_translator.core.errors import ToolError
from book_translator.core.store import ArtifactStore
from book_translator.documents.chapters import (
    has_postprocessed_chapter_files,
    is_toc_filename,
    translated_epub_path,
)
from book_translator.stages.postprocess.inspect import chapter_files, scan_book_structure

from .artifacts import epub_manifest_payload, write_epub_manifest, write_epub_trace
from .embed import cover_asset as _cover_asset
from .embed import embed_images as _embed_images
from .metadata import file_timestamp, resolve_metadata
from .package import ImageAsset, SpineDocument, build_epub
from .render import render_markdown
from .result import EpubResult, build_epub_result
from .verify import check_nav as _check_nav
from .verify import check_opf as _check_opf
from .verify import verify as _verify

#: The stylesheet the package carries.
STYLESHEET_PATH = Path(__file__).resolve().parent / "assets" / "style.css"


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

    result = build_epub_result(
        source=source,
        target=target,
        metadata=metadata,
        documents=documents,
        assets=assets,
        chars=chars,
        written=written,
        verification=verification,
        warnings=warnings,
        errors=errors,
        toc_dropped=_has_toc_page(source),
        cover=cover_asset,
        dry_run=dry_run,
    )
    store = ArtifactStore(
        target.parent,
        EPUB_MANIFEST_FILENAME,
        EPUB_TRACE_FILENAME,
        enabled=write_manifest and not dry_run,
    )
    return store.write_result(result, epub_manifest_payload(result))


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


__all__ = [
    "STYLESHEET_PATH",
    "EpubResult",
    "_check_nav",
    "_check_opf",
    "_verify",
    "build_epub_result",
    "epub_manifest_payload",
    "run_epub",
    "write_epub_manifest",
    "write_epub_trace",
]
