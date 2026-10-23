"""The text of a synthetic front or back matter section.

Nothing here calls a model. The matter agent still localises the generated
English text; these strings only supply the facts it must not lose — the title,
the author and the translator credit.
"""

from __future__ import annotations

from book_translator.documents.book import BookMetadata

#: A numbered name keeps the file in reading order and makes ``matter_kind``
#: classify it as back matter without relying on the manifest.
BACK_MATTER_FILENAME = "99-back-matter.md"

#: The first line the matter agent receives for a generated back page.
BACK_MATTER_TITLE = "About the Author"


def front_matter_text(metadata: BookMetadata) -> str:
    title = metadata.title or metadata.stem or "Untitled"
    lines = [f"# {title}", ""]
    if metadata.author:
        lines.extend([metadata.author, ""])
    lines.append("*Translated by BleBel*")
    return "\n".join(lines).rstrip() + "\n"


def back_matter_text(metadata: BookMetadata) -> str:
    title = metadata.title or metadata.stem or "this book"
    lines = [f"# {BACK_MATTER_TITLE}", ""]
    if metadata.author:
        lines.append(f"{metadata.author} is the author of {title}.")
    else:
        lines.append(f"{title} was published in translation by BleBel.")
    return "\n".join(lines).rstrip() + "\n"

__all__ = [
    "BACK_MATTER_FILENAME",
    "BACK_MATTER_TITLE",
    "back_matter_text",
    "front_matter_text",
]
