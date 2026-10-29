"""Deterministic full-book concatenation (stage 10).

The input is stage 9's ``09-postprocessed-chapters/`` folder. The output is one
Markdown file, ``10-translated-book.md``, in the same book folder. Nothing is
rewritten: the files are joined in reading order with one blank line between
them, and the result is verified by re-reading it.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from book_translator.core.artifacts import timestamp, write_manifest, write_text, write_trace
from book_translator.core.constants import (
    BIND_MANIFEST_FILENAME,
    BIND_TRACE_FILENAME,
    CHAPTER_IMAGE_PREFIX,
    POSTPROCESSED_CHAPTERS_DIRNAME,
    ROOT_IMAGE_PREFIX,
    TOC_FILENAME,
)
from book_translator.core.errors import ToolError
from book_translator.core.store import ArtifactStore
from book_translator.documents.chapters import (
    file_sha256,
    has_postprocessed_chapter_files,
    translated_book_path,
)
from book_translator.documents.images import image_count
from book_translator.stages.postprocess.inspect import is_toc_filename, scan_book_structure


@dataclass
class BindResult:
    """Outcome of concatenating the postprocessed chapters into one file."""

    ok: bool
    chapters_path: str
    output_path: str
    chapter_count: int = 0
    toc_included: bool = False
    chars: int = 0
    lines: int = 0
    sha256: str = ""
    #: Image references in the final file (after the book-root path rewrite).
    image_refs: int = 0
    dry_run: bool = False
    files: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    manifest_path: str | None = None
    trace_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _natural_key(name: str) -> tuple[int, str]:
    match = re.match(r"^(\d+)", name)
    return (int(match.group(1)) if match else 10**9, name.lower())


def _book_files(folder: Path) -> list[Path]:
    """Every markdown file to concatenate, including the separate TOC file."""
    files = [
        path
        for path in folder.glob("*.md")
        if path.is_file() and not path.name.startswith("_")
    ]
    return sorted(files, key=lambda path: _natural_key(path.name))


def run_bind(
    postprocessed_path: str | Path,
    output_path: str | Path | None = None,
    *,
    dry_run: bool = False,
    write_manifest: bool = True,
) -> BindResult:
    """Join every postprocessed chapter into one Markdown document.

    ``output_path`` defaults to ``10-translated-book.md`` next to the chapter folder.
    A folder handed in is understood as the postprocessed folder itself (or its
    parent book folder). A dry run builds and verifies the text but writes
    nothing.
    """
    source = Path(postprocessed_path).expanduser()
    if source.is_file():
        source = source.parent
    if source.name != POSTPROCESSED_CHAPTERS_DIRNAME:
        nested = source / POSTPROCESSED_CHAPTERS_DIRNAME
        if nested.is_dir():
            source = nested
    if not source.is_dir():
        raise ToolError(f"postprocessed chapters folder not found: {source}")
    if not has_postprocessed_chapter_files(source):
        raise ToolError(
            f"no postprocessed chapter files in {source} — run `postprocess` first"
        )
    files = _book_files(source)
    if not files:
        raise ToolError(f"no Markdown files to bind in {source}")
    target = (
        Path(output_path).expanduser()
        if output_path is not None
        else translated_book_path(source)
    )

    parts: list[str] = []
    records: list[dict[str, Any]] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        parts.append(text.rstrip("\n"))
        records.append(
            {
                "filename": path.name,
                "chars": len(text),
                "lines": len(text.splitlines()),
                "sha256": file_sha256(path),
            }
        )
    book = "\n\n".join(parts) + "\n"
    # Chapter files sit one folder below the book, hence `../01-markdown/images/`.
    # The bound book is written at the book root, where the same folder is
    # `01-markdown/images/`; rewrite once, so every picture still resolves.
    book = book.replace(CHAPTER_IMAGE_PREFIX, ROOT_IMAGE_PREFIX)
    errors: list[str] = []
    if not book.strip():
        errors.append("the assembled book is empty")
    if dry_run:
        pass
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        write_text(target, book)
    written = book if dry_run else target.read_text(encoding="utf-8")
    # Verify by re-reading the actual file (or the in-memory text on a dry run):
    # every source chunk must be present once, in order.
    cursor = 0
    for record, path in zip(records, files, strict=True):
        text = path.read_text(encoding="utf-8").rstrip("\n")
        text = text.replace(CHAPTER_IMAGE_PREFIX, ROOT_IMAGE_PREFIX)
        found = written.find(text, cursor)
        if found < 0:
            errors.append(f"{record['filename']} is missing from the assembled book")
            continue
        cursor = found + len(text)
    front_toc = bool((scan_book_structure(source).get("toc") or {}).get("exists"))
    result = build_bind_result(
        source=source,
        target=target,
        records=records,
        written=written,
        errors=errors,
        toc_included=any(is_toc_filename(path.name) for path in files)
        or (source / TOC_FILENAME).is_file()
        or front_toc,
        sha256=file_sha256(target) if target.is_file() and not dry_run else "",
        dry_run=dry_run,
    )
    store = ArtifactStore(
        target.parent,
        BIND_MANIFEST_FILENAME,
        BIND_TRACE_FILENAME,
        enabled=write_manifest and not dry_run,
    )
    return store.write_result(result, bind_manifest_payload(result))


def build_bind_result(
    *,
    source: Path,
    target: Path,
    records: list[dict[str, Any]],
    written: str,
    errors: list[str],
    toc_included: bool,
    sha256: str,
    dry_run: bool,
) -> BindResult:
    """The whole run as a result value; no file is read or written here."""
    return BindResult(
        ok=not errors,
        chapters_path=str(source),
        output_path=str(target),
        chapter_count=len(records),
        toc_included=toc_included,
        chars=len(written),
        lines=len(written.splitlines()),
        sha256=sha256,
        image_refs=image_count(written),
        dry_run=dry_run,
        files=records,
        errors=errors,
    )


def bind_manifest_payload(result: BindResult) -> dict[str, Any]:
    """The manifest body, kept separate from the write so it can be reused."""
    return {
        "stage": "bind",
        "generated_at": timestamp(),
        "ok": result.ok,
        "chapters_path": result.chapters_path,
        "output_path": result.output_path,
        "chapter_count": result.chapter_count,
        "toc_included": result.toc_included,
        "chars": result.chars,
        "lines": result.lines,
        "image_refs": result.image_refs,
        "sha256": result.sha256,
        "files": result.files,
        "errors": result.errors,
    }


def write_bind_manifest(output_path: Path, result: BindResult) -> Path:
    return write_manifest(
        output_path.parent, BIND_MANIFEST_FILENAME, bind_manifest_payload(result)
    )


def write_bind_trace(output_path: Path, result: BindResult) -> Path:
    return write_trace(output_path.parent, BIND_TRACE_FILENAME, result.to_dict())


__all__ = [
    "BindResult",
    "bind_manifest_payload",
    "build_bind_result",
    "run_bind",
    "write_bind_manifest",
    "write_bind_trace",
]
