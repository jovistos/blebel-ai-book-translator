"""Tools for looking at *lines* — generic, and used by both pipeline stages.

Three tools live here, and they are shared on purpose:

``inspect_line_window``
    a few lines of context around a line number, hard-capped in lines and in
    characters. Both stages need this ("what is on line 37?"), and both must be
    unable to turn it into "read me the chapter".
``sample_corpus_text``
    a bounded, strided sample of a chapter. Stage 2 uses it to look for
    repeating damage; stage 3 (translation) will use it for style probes.
``scan_corpus_damage``
    the counted damage profile of a whole book.

Every result marks itself as a *sample* / *window*, so the model cannot mistake
it for the document.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from ...core.constants import (
    DEFAULT_INSPECT_LINES,
    DEFAULT_MAX_HEADERS,
    DEFAULT_SAMPLE_LINES_PER_CHAPTER,
    MAX_INSPECT_CHARS,
    MAX_INSPECT_LINES,
    MAX_SAMPLE_CHARS,
)
from ...documents.chapters import discover_chapters
from ...documents.io import load_lines
from ...documents.markdown import header_of
from ...documents.text import excerpt, jdump
from ...preprocess.corpus import (
    SAMPLE_STRATEGIES,
    SAMPLE_STRIDE,
    sample_text,
    scan_corpus_damage,
)
from .registry import shared


def scan_markdown_headers(file_path: str, max_headers: int = DEFAULT_MAX_HEADERS) -> dict[str, Any]:
    """Every Markdown ATX header in a file, with its exact line number.

    The plain function behind `extract_markdown_headers`, shared by both stages
    so neither has to duplicate the scan.
    """
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


@shared
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


@shared
@tool
def inspect_line_window(
    file_path: str,
    line_number: int,
    context: int = DEFAULT_INSPECT_LINES,
) -> str:
    """Look at a *window* of a few lines around one line number.

    Use this to check what surrounds a header you are unsure about — does the
    next line continue the same heading, is the line followed by prose, is it a
    figure caption? Returns at most 6 lines and 600 characters, each clipped.
    The window is centred on `line_number`; `context` says how many lines on
    each side you want (0–3).
    """
    lines = load_lines(file_path)
    n = len(lines)
    try:
        target = int(line_number)
    except (TypeError, ValueError):
        return jdump({"error": f"line_number must be an integer, got {line_number!r}"})
    if target < 1 or target > n:
        return jdump({"error": f"line {target} is out of range (file has {n} lines)"})
    try:
        spread = int(context)
    except (TypeError, ValueError):
        spread = DEFAULT_INSPECT_LINES
    spread = max(0, min(spread, MAX_INSPECT_LINES // 2))
    start = max(1, target - spread)
    end = min(n, target + spread)
    if end - start + 1 > MAX_INSPECT_LINES:
        start = max(1, target - MAX_INSPECT_LINES + 1)
        end = min(n, start + MAX_INSPECT_LINES - 1)

    window: list[dict[str, Any]] = []
    used = 0
    for number in range(start, end + 1):
        raw = lines[number - 1].rstrip("\n")
        text = excerpt(raw, 160)
        if used + len(text) > MAX_INSPECT_CHARS:
            break
        used += len(text)
        window.append(
            {
                "line_number": number,
                "is_header": header_of(raw) is not None,
                "chars": len(raw),
                "text": text,
            }
        )
    return jdump(
        {
            "file": Path(file_path).name,
            "line_count": n,
            "requested_line": target,
            "start_line": start,
            "end_line": end,
            "note": "WINDOW ONLY - clipped lines, capped length. Not the chapter text.",
            "window": window,
        }
    )


@shared
@tool
def sample_corpus_text(
    path: str,
    max_lines: int = DEFAULT_SAMPLE_LINES_PER_CHAPTER,
    strategy: str = SAMPLE_STRIDE,
) -> str:
    """Sample a bounded number of lines from one chapter file.

    `path` is a single chapter file (see `list_chapter_files`). `strategy` is
    one of `head`, `tail`, `stride`, `longest`, `headers`. Lines are clipped to
    160 characters and the whole sample is capped at 12000 characters, so this
    can never return a whole chapter. The result is labelled SAMPLES ONLY.
    """
    if strategy not in SAMPLE_STRATEGIES:
        return jdump(
            {"error": f"unknown strategy {strategy!r}", "allowed": list(SAMPLE_STRATEGIES)}
        )
    try:
        budget = max(1, min(int(max_lines), 200))
    except (TypeError, ValueError):
        budget = DEFAULT_SAMPLE_LINES_PER_CHAPTER
    return jdump(sample_text(path, max_lines=budget, strategy=strategy, max_chars=MAX_SAMPLE_CHARS))


@shared
@tool
def scan_corpus_damage_tool(chapters_path: str, max_examples_per_pattern: int = 3) -> str:
    """Count every recognised conversion-damage pattern across a whole book.

    `chapters_path` is a chapters folder (or its `_manifest.json`). Returns
    `{"counts", "patterns", "examples", "per_chapter"}`: how often each pattern
    occurs, what the fix is, and a few short examples per pattern. This is the
    cheap way to learn what a book needs before touching any chapter.
    """
    try:
        limit = max(0, min(int(max_examples_per_pattern), 10))
    except (TypeError, ValueError):
        limit = 3
    chapters = discover_chapters(chapters_path)
    return jdump(scan_corpus_damage(chapters, max_examples_per_pattern=limit))


@shared
@tool
def list_chapter_files(chapters_path: str) -> str:
    """List the chapter files of an already-split book, in reading order.

    `chapters_path` is the output folder of the splitting stage (the one holding
    `_manifest.json`). Returns filenames, line counts, byte counts and the
    source line ranges recorded by stage 1. Call this first to learn the exact
    filenames to pass to the other tools.
    """
    chapters = discover_chapters(chapters_path)
    return jdump(
        {
            "chapters_path": str(chapters_path),
            "chapter_count": len(chapters),
            "chapters": [
                {
                    "chapter_id": c.chapter_id,
                    "filename": c.filename,
                    "path": c.path,
                    "lines": c.line_count,
                    "bytes": c.bytes,
                    "source_lines": [c.start_line, c.end_line],
                }
                for c in chapters
            ],
        }
    )


__all__ = [
    "extract_markdown_headers",
    "inspect_line_window",
    "list_chapter_files",
    "sample_corpus_text",
    "scan_corpus_damage_tool",
    "scan_markdown_headers",
]
