"""Book metadata for the EPUB package (stage 11).

An EPUB states who wrote the book, what it is called and in which language it is
written, and every reader shows that. The pipeline already knows all of it: stage
6b records the original title, the author and the target language in
``06b-matter/_matter_manifest.json``, and :func:`book_translator.postprocess.inspect.book_identity`
reads them back for stage 9. This module turns that evidence into the five fields
an EPUB needs, with an explicit override winning over anything inferred.

Two of those fields are invented rather than read:

``identifier``
    A UUID derived from the book's output-folder name, so the same book is always
    the same publication across rebuilds — a random UUID would tell every reader
    that this is a new book on every run.

``modified``
    The newest chapter file's timestamp, not "now". The EPUB then says when its
    content last changed, and two runs over unchanged inputs produce the same
    bytes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from ..core.constants import DEFAULT_TARGET_LANGUAGE
from ..documents.book import BookMetadata

#: Prefix of the deterministic identifier: an EPUB identifier is a URN here.
IDENTIFIER_PREFIX = "urn:uuid:"
#: Namespace string for the UUID, so a slug can only ever mean one book.
UUID_NAMESPACE = "urn:book_translator:epub:"


def zip_timestamp(modified: str) -> tuple[int, int, int, int, int, int]:
    """``modified`` as the tuple a zip entry timestamp wants.

    The zip format cannot express a date before 1980, and a missing or malformed
    timestamp falls back to "now" rather than failing the stage; both rules are
    about the ZIP container, not about the book, which is why this lives here and
    not on :class:`~book_translator.documents.book.BookMetadata`.
    """
    try:
        moment = datetime.fromisoformat(modified.replace("Z", "+00:00"))
    except ValueError:  # pragma: no cover - defensive, the runner formats it
        moment = datetime.now(UTC)
    if moment.tzinfo is not None:
        moment = moment.astimezone(UTC)
    if moment.year < 1980:
        return (1980, 1, 1, 0, 0, 0)
    return (
        moment.year,
        moment.month,
        moment.day,
        moment.hour,
        moment.minute,
        moment.second,
    )


def resolve_metadata(
    structure: dict[str, Any] | None,
    *,
    slug: str,
    modified: str,
    title: str | None = None,
    author: str | None = None,
    language: str | None = None,
    identifier: str | None = None,
    description: str | None = None,
    publisher: str | None = None,
) -> BookMetadata:
    """Resolve the package metadata from the book's own records.

    Every argument that is passed wins over what the records say; the records
    win over the defaults. ``structure`` is stage 9's inventory
    (:func:`book_translator.postprocess.inspect.scan_book_structure`), which is
    where the matter stage's title, author and language are re-read from.
    """
    records = structure or {}
    warnings: list[str] = []

    resolved_title = _clean(title) or _clean(records.get("book_title"))
    if not resolved_title:
        resolved_title = _clean(records.get("translated_title"))
    if not resolved_title:
        resolved_title = _clean(records.get("original_title")) or slug
        warnings.append(
            "the book has no recorded title; using the folder name — pass --title"
        )

    resolved_author = _clean(author) or _clean(records.get("book_author"))
    if not resolved_author:
        warnings.append(
            "the book has no recorded author; pass --author to credit one"
        )

    resolved_language = (
        _clean(language) or _clean(records.get("target_language"))
    )
    if not resolved_language:
        resolved_language = DEFAULT_TARGET_LANGUAGE
        warnings.append(
            f"the target language is not recorded; falling back to {resolved_language!r}"
            " — pass --language"
        )

    return BookMetadata(
        title=resolved_title,
        author=resolved_author,
        language=resolved_language,
        identifier=_clean(identifier) or book_identifier(slug),
        modified=modified,
        description=_clean(description),
        publisher=_clean(publisher),
        warnings=warnings,
    )


def book_identifier(slug: str) -> str:
    """The deterministic URN this book is always published under."""
    return IDENTIFIER_PREFIX + str(uuid5(NAMESPACE_URL, UUID_NAMESPACE + str(slug)))


def file_timestamp(paths: list[Any]) -> str:
    """The newest mtime among ``paths``, as an EPUB ``dcterms:modified`` value."""
    stamps: list[float] = []
    for raw in paths:
        try:
            stamps.append(float(raw))
        except (TypeError, ValueError):
            continue
    moment = (
        datetime.fromtimestamp(max(stamps), tz=UTC) if stamps else datetime.now(UTC)
    )
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


__all__ = [
    "IDENTIFIER_PREFIX",
    "UUID_NAMESPACE",
    "BookMetadata",
    "book_identifier",
    "file_timestamp",
    "resolve_metadata",
    "zip_timestamp",
]
