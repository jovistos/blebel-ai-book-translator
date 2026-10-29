"""Stage 1: convert an EPUB, text-based PDF or TXT source into Markdown.

The rest of the pipeline is line-based and reads Markdown only. This package is
the one conversion step: docling for EPUB/PDF (never OCR for an image-based
PDF), a byte-for-byte copy for TXT. With images enabled (the default) the
document's pictures are exported to ``01-markdown/images/`` and referenced from
the Markdown, so illustrations survive every later stage.
See :mod:`book_translator.stages.conversion.runner`.
"""

from __future__ import annotations

from .runner import (
    DOCLING_SUFFIXES,
    MARKDOWN_SUFFIXES,
    TEXT_SUFFIXES,
    ConversionResult,
    classify_source,
    converted_markdown_path,
    inspect_pdf_text_layer,
    is_markdown_source,
    is_supported_source,
    run_conversion,
)

__all__ = [
    "DOCLING_SUFFIXES",
    "MARKDOWN_SUFFIXES",
    "TEXT_SUFFIXES",
    "ConversionResult",
    "classify_source",
    "converted_markdown_path",
    "inspect_pdf_text_layer",
    "is_markdown_source",
    "is_supported_source",
    "run_conversion",
]
