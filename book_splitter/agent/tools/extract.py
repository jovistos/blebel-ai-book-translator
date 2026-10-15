"""Structural extraction tools: what the book *looks like*, not what it says."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from ...analysis.toc import detect_toc_block
from ...core.constants import (
    DEFAULT_FRONT_MATTER_LINES,
    DEFAULT_MAX_HEADERS,
    MAX_FRONT_MATTER_CHARS,
    MAX_FRONT_MATTER_LINES,
    TOC_SCAN_LINES,
)
from ...documents.io import load_lines
from ...documents.markdown import header_of
from ...documents.text import excerpt, jdump
from .registry import register_tool


def detect_native_toc(file_path: str | Path) -> dict[str, Any]:
    """Embedded bookmark outline of a PDF/EPUB, when the optional deps exist.

    Returns ``{"available": bool, "reason"?: str, "entries": [...]}``; it never
    raises, so a missing optional dependency simply means "not available".
    """
    suffix = Path(file_path).suffix.lower()
    try:
        if suffix == ".pdf":
            import fitz  # type: ignore

            doc = fitz.open(str(file_path))
            toc = doc.get_toc()
            if not toc:
                return {"available": False, "reason": "PDF has no embedded bookmarks"}
            return {
                "available": True,
                "source": "pdf-bookmarks",
                "count": len(toc),
                "entries": [
                    {"level": lvl, "title": excerpt(title, 120), "page": page}
                    for lvl, title, page in toc
                ],
            }
        if suffix == ".epub":
            import ebooklib  # type: ignore
            from bs4 import BeautifulSoup  # type: ignore
            from ebooklib import epub  # type: ignore

            book = epub.read_epub(str(file_path))
            titles: list[str] = []
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                name = item.get_name().lower()
                if "nav" in name or "toc" in name:
                    soup = BeautifulSoup(item.get_content(), "html.parser")
                    titles = [a.get_text(strip=True) for a in soup.find_all("a")]
                    break
            if not titles:
                return {"available": False, "reason": "EPUB nav document not found"}
            return {
                "available": True,
                "source": "epub-nav",
                "count": len(titles),
                "entries": [{"title": excerpt(t, 120)} for t in titles],
            }
    except ImportError as exc:  # optional dependency missing
        return {"available": False, "reason": f"optional dependency missing: {exc.name}"}
    except Exception as exc:  # pragma: no cover - defensive
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    return {
        "available": False,
        "reason": f"{suffix or 'unknown'} input has no embedded TOC metadata",
    }


def scan_markdown_headers(file_path: str, max_headers: int = DEFAULT_MAX_HEADERS) -> dict[str, Any]:
    """Every Markdown ATX header with its exact line number."""
    lines = load_lines(file_path)
    headers: list[dict[str, Any]] = []
    for idx, raw in enumerate(lines, start=1):
        head = header_of(raw)
        if head:
            headers.append({"line_number": idx, "level": head[0], "text": excerpt(head[1], 120)})
    return {
        "line_count": len(lines),
        "count": len(headers),
        "truncated": len(headers) > max_headers,
        "headers": headers[:max_headers],
    }


@register_tool
@tool
def extract_native_toc(file_path: str) -> str:
    """Extract the built-in table of contents stored as file metadata.

    Only works for PDF (PyMuPDF) or EPUB (ebooklib) sources, and returns a
    JSON object `{"available": bool, "entries": [{"title", "page"}]}`.
    Call this first: it is free, instant, and 100% accurate when it works.
    For a `.md` input it simply reports that no native TOC exists.
    """
    return jdump(detect_native_toc(file_path))


@register_tool
@tool
def extract_markdown_headers(file_path: str, max_headers: int = DEFAULT_MAX_HEADERS) -> str:
    """Scan a file for Markdown ATX headers (`#`, `##`, ...) and return their line numbers.

    Returns `{"line_count", "count", "truncated", "headers": [{"line_number",
    "level", "text"}]}`. Header text is capped at 120 characters per entry.

    Warning: in a file converted from PDF, most headers can be junk (page
    numbers, figure captions, running heads). Always judge the returned list
    before trusting it.
    """
    return jdump(scan_markdown_headers(file_path, max_headers))


@register_tool
@tool
def detect_toc_block_tool(file_path: str, scan_lines: int = TOC_SCAN_LINES) -> str:
    """Find the printed table of contents in the front matter and report its line range.

    Returns `{"found", "start_line", "end_line", "heading_line", "method",
    "entries"}` plus any dot-leader entries (`Chapter 1 ..... 15`). Use the
    line range to ignore the TOC when searching for real chapter start lines.
    """
    return jdump(detect_toc_block(load_lines(file_path), scan_lines=scan_lines))


@register_tool
@tool
def extract_front_matter(file_path: str, max_lines: int = DEFAULT_FRONT_MATTER_LINES) -> str:
    """Read only the first `max_lines` lines of the book to locate a printed TOC.

    This is the ONLY tool that returns prose, and its output is hard-capped at
    20000 characters (and 1200 lines). Use it to read the table of contents and pull out
    the chapter title strings, then hand those strings to
    `locate_toc_titles_in_file`. Never ask for more lines than you need.
    """
    lines = load_lines(file_path)
    n = max(1, min(int(max_lines), MAX_FRONT_MATTER_LINES))
    chunk = "".join(lines[:n])
    truncated = False
    if len(chunk) > MAX_FRONT_MATTER_CHARS:
        chunk = chunk[:MAX_FRONT_MATTER_CHARS]
        truncated = True
    return jdump(
        {
            "start_line": 1,
            "end_line": min(n, len(lines)),
            "truncated_at_chars": truncated,
            "text": chunk,
        }
    )
