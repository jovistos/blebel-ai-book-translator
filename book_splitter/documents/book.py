"""The one record of a book's identity.

Two stages used to own a class with this name: the matter stage parsed a title
and an author out of the input filename, and the EPUB stage resolved the five
``dc:`` fields a package needs. They were the same idea — *what is this book
called, and who made it* — split by which fields each stage happened to read, so
a caller could not move a value from one side to the other without converting
between two dataclasses. This module is the single shape both sides use.

Field provenance:

``title`` / ``author`` / ``stem``
    evidence from the input filename (:func:`book_splitter.matter.metadata.parse_book_metadata`)
    or from the matter manifest stage 6b wrote.
``language`` / ``identifier`` / ``modified`` / ``description`` / ``publisher``
    the publication fields the EPUB package writes, resolved (and defaulted) by
    :func:`book_splitter.epub.metadata.resolve_metadata`.
``warnings``
    anything the resolver had to invent (a missing author, a language fallback),
    carried to the stage result so the run can report it.

The record is deliberately transport-neutral: it knows nothing about ZIP dates,
UUID namespaces or filenames. Those belong to the module that produces them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class BookMetadata:
    """Everything the pipeline knows about one book's identity.

    Every field has a default, because the two producers know different subsets:
    the matter stage always has a filename to parse, while a caller may construct
    an empty record and fill it before asking the EPUB stage to resolve it.
    """

    title: str = ""
    author: str = ""
    stem: str = ""
    language: str = ""
    identifier: str = ""
    modified: str = ""
    description: str = ""
    publisher: str = ""
    #: Non-fatal notes from the resolver, shown in the stage result.
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["BookMetadata"]
