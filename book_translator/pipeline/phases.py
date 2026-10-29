"""Phase metadata: the one order, the titles and the folder dependencies.

The driver owns *how* a phase runs and *what it records*; this module owns the
static facts that must not drift between the parser, the selection and the
narration:

* :data:`PHASES` — the only order a run may use;
* :data:`TITLES` — the human title printed in the phase banner;
* :data:`NEEDS_CHAPTERS` — the phases that cannot run until stage 2 has written
  a chapter folder.

Keeping them here means a new phase is added once, not in the parser, the
selection and the driver separately.
"""

from __future__ import annotations

PHASES: tuple[str, ...] = (
    "convert",
    "recon",
    "split",
    "scan",
    "preprocess",
    "dictionary",
    "chunk",
    "chunk_glossary",
    "matter",
    "translate",
    "concatenate",
    "postprocess",
    "bind",
    "epub",
)

#: Human titles used by the phase banner.
TITLES: dict[str, str] = {
    "convert": "stage 1 — convert the EPUB/PDF/TXT source to Markdown (no LLM, no OCR)",
    "recon": "what the book actually contains (no LLM, no writes)",
    "split": "stage 2 — the agent routes, the lines are copied verbatim",
    "scan": "stage 3 pre-flight — the damage profile (no LLM)",
    "preprocess": "stage 3 — clean each chapter, then prove the edit",
    "dictionary": "stage 4 — extract entities and build the book's glossary",
    "chunk": "stage 5 — cut each chapter into reversible, sentence-safe chunks (no LLM)",
    "chunk_glossary": "stage 6 — tag every chunk, then attach the glossary terms it mentions",
    "matter": "stage 7 — localize the front/back matter with its own agent (no dictionary)",
    "translate": "stage 8 — translate every body chunk, one call at a time, and keep its prompt",
    "concatenate": "stage 9 — glue the translated chunks back into chapters (no LLM)",
    "postprocess": "stage 10 — make the translated headings and table of contents consistent",
    "bind": "stage 11 — concatenate the postprocessed chapters into one Markdown book",
    "epub": "stage 12 — package the chapters as an EPUB 3 book (no LLM)",
}

#: Phases that read the chapter folder stage 2 writes.
NEEDS_CHAPTERS: frozenset[str] = frozenset(
    {
        "scan",
        "dictionary",
        "preprocess",
        "chunk",
        "chunk_glossary",
        "matter",
        "translate",
        "concatenate",
        "postprocess",
        "bind",
        "epub",
    }
)

__all__ = ["NEEDS_CHAPTERS", "PHASES", "TITLES"]
