"""One book record for the matter and EPUB stages.

The two stages used to own separate ``BookMetadata`` classes; a value could not
move between them without conversion, and the EPUB one could not be built until
every required field was known. These tests pin the unified shape, the two
import paths (same object) and the EPUB-only ZIP date helper that moved off the
record.
"""

from __future__ import annotations

from book_splitter.core.constants import DEFAULT_TARGET_LANGUAGE
from book_splitter.documents.book import BookMetadata as Canonical
from book_splitter.epub import metadata as epub_metadata
from book_splitter.matter import metadata as matter_metadata


def test_both_stages_use_the_one_book_record() -> None:
    assert Canonical is matter_metadata.BookMetadata is epub_metadata.BookMetadata
    assert Canonical() == matter_metadata.BookMetadata(title="", author="", stem="")


def test_the_filename_parser_fills_the_shared_record() -> None:
    source = "Tea for the Traditionally Built - Alexander McCall Smith.md"
    meta = matter_metadata.parse_book_metadata(source)

    assert isinstance(meta, Canonical)
    assert (meta.title, meta.author) == (
        "Tea for the Traditionally Built",
        "Alexander McCall Smith",
    )
    assert meta.stem == "Tea for the Traditionally Built - Alexander McCall Smith"
    assert meta.language == ""
    assert meta.warnings == []


def test_the_package_resolver_fills_the_shared_record() -> None:
    structure = {
        "book_title": "Alisa u zemlji čuda",
        "book_author": "Luis Kerol",
        "target_language": "sr",
    }

    meta = epub_metadata.resolve_metadata(
        structure, slug="alisa", modified="2020-01-02T03:04:05Z"
    )

    assert meta.title == "Alisa u zemlji čuda"
    assert meta.author == "Luis Kerol"
    assert meta.language == "sr"
    assert meta.identifier == epub_metadata.book_identifier("alisa")
    assert meta.modified == "2020-01-02T03:04:05Z"
    assert meta.warnings == []


def test_explicit_fields_win_over_the_records() -> None:
    structure = {"book_title": "Recorded", "book_author": "Recorded"}

    meta = epub_metadata.resolve_metadata(
        structure,
        slug="slug",
        modified="2020-01-02T03:04:05Z",
        title="Override",
        author="Author",
        language="en",
        identifier="urn:uuid:fixed",
        description="Desc",
        publisher="Pub",
    )

    assert (meta.title, meta.author, meta.language) == ("Override", "Author", "en")
    assert meta.identifier == "urn:uuid:fixed"
    assert (meta.description, meta.publisher) == ("Desc", "Pub")


def test_missing_evidence_is_warned_about_not_fatal() -> None:
    meta = epub_metadata.resolve_metadata({}, slug="slug", modified="2020-01-02T03:04:05Z")

    assert meta.title == "slug"
    assert meta.author == ""
    assert meta.language == DEFAULT_TARGET_LANGUAGE
    assert len(meta.warnings) == 3


def test_the_record_serialises_every_field() -> None:
    meta = Canonical(title="T", author="A", warnings=["w"])

    assert list(meta.to_dict()) == [
        "title",
        "author",
        "stem",
        "language",
        "identifier",
        "modified",
        "description",
        "publisher",
        "warnings",
    ]
    assert meta.to_dict()["warnings"] == ["w"]


def test_zip_timestamps_are_utc_and_clamped_before_1980() -> None:
    assert epub_metadata.zip_timestamp("2020-01-02T03:04:05Z") == (2020, 1, 2, 3, 4, 5)
    assert epub_metadata.zip_timestamp("2020-01-02T05:04:05+02:00") == (2020, 1, 2, 3, 4, 5)
    assert epub_metadata.zip_timestamp("1970-01-01T00:00:00Z") == (1980, 1, 1, 0, 0, 0)
    # A malformed timestamp is a packaging fallback, never a stage failure.
    assert len(epub_metadata.zip_timestamp("not-a-date")) == 6
