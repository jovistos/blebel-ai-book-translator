"""The EPUB result value and its one builder."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.artifacts import sha256_bytes
from .metadata import BookMetadata
from .package import ImageAsset, SpineDocument


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


def build_epub_result(
    *,
    source: Path,
    target: Path,
    metadata: BookMetadata,
    documents: list[SpineDocument],
    assets: dict[str, ImageAsset],
    chars: int,
    written: bytes,
    verification: dict[str, Any],
    warnings: list[str],
    errors: list[str],
    toc_dropped: bool,
    cover: ImageAsset | None,
    dry_run: bool,
) -> EpubResult:
    """The package as one value; no file is read or written here."""
    return EpubResult(
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
        toc_dropped=toc_dropped,
        cover=cover.href if cover is not None else None,
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


__all__ = ["EpubResult", "build_epub_result"]
