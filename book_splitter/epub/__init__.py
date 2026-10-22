"""Stage 11: package the postprocessed chapters as an EPUB 3 book.

This is the one stage that does not write Markdown. It reads
``09-postprocessed-chapters/`` — front matter, the chapters, back matter — and
writes ``11-translated-book.epub`` next to stage 10's Markdown reading copy. The
printed table of contents is deliberately not part of the book: every reader
builds its own from the navigation document.

The package is deterministic (a rebuild of unchanged chapters is byte-identical)
and it is verified after it is written: the mimetype entry, the container, the
package document, the navigation document, every spine item's XHTML and every
picture link are all re-read and checked.
"""

from .metadata import BookMetadata, book_identifier, resolve_metadata
from .package import ImageAsset, SpineDocument, build_epub, media_type_for
from .render import MARKDOWN_IT_HINT, RenderedMarkdown, render_markdown
from .runner import EpubResult, run_epub

__all__ = [
    "MARKDOWN_IT_HINT",
    "BookMetadata",
    "EpubResult",
    "ImageAsset",
    "RenderedMarkdown",
    "SpineDocument",
    "book_identifier",
    "build_epub",
    "media_type_for",
    "render_markdown",
    "resolve_metadata",
    "run_epub",
]
