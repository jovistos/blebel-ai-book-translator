"""Deterministic core of the front/back matter localization stage.

This package answers the questions the localization agent must not guess:

* **Which files are matter?** A file whose title is ``Predgovor``/``Uvod`` is
  front matter even when its filename is numbered, and ``Beleške``/``Pogovor``
  is back matter. :func:`discover_matter_sections` reports the sections and the
  chunks that belong to each.
* **What is a valid book plan?** The localized title, the translator credit
  (``BleBel``), the per-section titles and the list of omitted source material
  are all validated before a single call is made.
* **What if the book has no front/back matter?** :func:`ensure_matter_sections`
  reads the input filename (``Title - Author``) and creates the missing sections
  with matching chapter and chunk records, so the final book still opens and
  closes like a published edition.
* **What is a publishable translation?** :func:`finalize_translation` turns one
  chunk's localized text into its final form: it puts the localized title and
  the translator credit on the first front-matter chunk, keeps the Markdown
  spacing clean, and makes sure the omitted material is really gone.

The agent itself lives in :mod:`book_translator.matter_agent`.
"""

from __future__ import annotations

from typing import Any

from .inspect import (
    MatterBook,
    MatterSection,
    body_chapter_titles,
    discover_matter_sections,
    matter_chapter_names,
    read_section_text,
)
from .metadata import BookMetadata, parse_book_metadata
from .plan import (
    clean_markdown,
    finalize_translation,
    preserve_trailing_newlines,
    validate_matter_plan,
    validate_matter_translation,
)
from .synthesize import MatterSynthesis, ensure_matter_sections

__all__ = [
    "BookMetadata",
    "MatterBook",
    "MatterChunkResult",
    "MatterResult",
    "MatterSection",
    "MatterSynthesis",
    "body_chapter_titles",
    "build_matter_chunk_graph",
    "build_matter_graph",
    "clean_markdown",
    "discover_matter_sections",
    "ensure_matter_sections",
    "finalize_translation",
    "matter_chapter_names",
    "parse_book_metadata",
    "preserve_trailing_newlines",
    "read_section_text",
    "run_matter",
    "validate_matter_plan",
    "validate_matter_translation",
]

_AGENT_EXPORTS = {
    "MatterChunkResult",
    "MatterResult",
    "build_matter_chunk_graph",
    "build_matter_graph",
    "run_matter",
}


def __getattr__(name: str) -> Any:
    if name in _AGENT_EXPORTS:
        from .. import matter_agent
        return getattr(matter_agent, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

