"""Low-level markdown line inspection.

These helpers answer "what *shape* is this line?" — header, standalone title,
prose — without ever interpreting the book's content.

The functions here are deliberately stage-agnostic: the splitter uses them to
find split points, the preprocessor uses them to decide whether a header is a
real heading or a page number. Only predicates live in this module; policy lives
in ``analysis/`` and ``preprocess/``.
"""

from __future__ import annotations

import re

from ..core.constants import MAX_PAGE_NUMBER_DIGITS, MAX_TITLE_LINE_CHARS

MD_HEADER_RE = re.compile(r"^(#{1,6})\s*(.*)$")
#: A line that is nothing but a number (``18``, ``## 101``, ``790``).
PAGE_NUMBER_ONLY_RE = re.compile(f"^#*\\s*(\\d{{1,{MAX_PAGE_NUMBER_DIGITS}}})\\s*$")
#: A number glued to the end of a word (``us12``, ``božan199``).
TRAILING_NUMBER_RE = re.compile(
    f"^(?P<head>.*?[^\\W\\d_])(?P<digits>\\d{{1,{MAX_PAGE_NUMBER_DIGITS}}})$"
)
#: A number glued to the front of a word (``12merena``).
LEADING_NUMBER_RE = re.compile(
    f"^(?P<digits>\\d{{1,{MAX_PAGE_NUMBER_DIGITS}}})(?P<tail>[^\\W\\d_].*)$"
)
#: Markdown emphasis markers, the only inline markup PDF conversions leave behind.
EMPHASIS_RE = re.compile(r"(?<!\w)(\*{1,3})(?!\s)(.+?)(?<!\s)\1(?!\w)")


def header_of(line: str) -> tuple[int, str] | None:
    """Return ``(level, title)`` when the line is a Markdown ATX header."""
    m = MD_HEADER_RE.match(line.strip())
    if not m:
        return None
    return len(m.group(1)), m.group(2).strip()


def md_level(line: str) -> int | None:
    """Header level of a line, or ``None`` when it is not a header."""
    head = header_of(line)
    return head[0] if head else None


def header_prefix(line: str) -> str:
    """The ``#`` run of a header line (``"## "``), or ``""`` for a body line."""
    m = re.match(r"^(#{1,6}\s*)", line.strip())
    return m.group(1) if m else ""


def header_body(line: str) -> str:
    """The text of a header line with its ``#`` run stripped."""
    head = header_of(line)
    return head[1] if head else line.strip()


def is_page_number_line(line: str) -> bool:
    """True when the line is only a number — ``19`` or ``## 19``.

    This is the single most common PDF-conversion artefact: the printed page
    number became a header of its own.
    """
    return bool(PAGE_NUMBER_ONLY_RE.match(line.strip()))


def is_all_caps(text: str, max_chars: int = MAX_TITLE_LINE_CHARS) -> bool:
    """True for a short, fully uppercased title (``GLAVA 1 LJUBAV JE U SRCU``).

    Letters are the only characters considered, so ``SEKS I SRCE:`` and
    ``PRVI DEO ISPUNJENJE LJUBAVI`` both qualify. This shape is the strongest
    positive evidence that a line really is a heading.
    """
    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) < 2 or len(text) > max_chars:
        return False
    return all(ch.isupper() for ch in letters)


def page_number_of(line: str) -> int | None:
    """The printed page number of an artefact line, or ``None``."""
    m = PAGE_NUMBER_ONLY_RE.match(line.strip())
    return int(m.group(1)) if m else None


def split_trailing_number(text: str) -> tuple[str, int] | None:
    """Split ``"...božan199"`` into ``("...božan", 199)``."""
    m = TRAILING_NUMBER_RE.match(text.rstrip())
    if not m:
        return None
    return m.group("head"), int(m.group("digits"))


def split_leading_number(text: str) -> tuple[int, str] | None:
    """Split ``"12merena"`` into ``(12, "merena")``."""
    m = LEADING_NUMBER_RE.match(text.lstrip())
    if not m:
        return None
    return int(m.group("digits")), m.group("tail")


def strip_emphasis(line: str) -> str:
    """Drop paired inline ``*``/``**`` emphasis markers, keep the words inside."""
    prev = None
    out = line
    while prev != out:
        prev = out
        out = EMPHASIS_RE.sub(lambda m: m.group(2), out)
    return out


def is_standalone_line(lines: list[str], idx0: int) -> bool:
    """True when the line is surrounded by blank lines (typical of a title)."""
    prev_blank = idx0 == 0 or not lines[idx0 - 1].strip()
    next_blank = idx0 == len(lines) - 1 or not lines[idx0 + 1].strip()
    return prev_blank and next_blank


def looks_like_title(line: str) -> bool:
    """A non-empty line short enough to be a heading rather than prose."""
    clean = line.strip().lstrip("#").strip()
    return bool(clean) and len(clean) <= MAX_TITLE_LINE_CHARS
