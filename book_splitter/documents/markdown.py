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
#: Dashes a converter may print around a page number (``- 18 -``, ``— 208 —``).
PAGE_NUMBER_FRAME = r"[-\u2010-\u2015\u2212]"
#: The only characters a page-number line may hold after its optional header
#: hashes and its ``\_`` escapes are removed. Whitespace is allowed anywhere,
#: because the PDF conversion often spaces the digits apart (``- 1 0 4 -``).
PAGE_NUMBER_TEXT_RE = re.compile(r"^[#\s\-–—_\d]+$")
#: The shape of a page number once its whitespace is gone: optional dashes and
#: underscores on either side of one run of digits.
PAGE_NUMBER_COMPACT_RE = re.compile(
    rf"^(?P<open>{PAGE_NUMBER_FRAME}*)(?P<under>_*)(?P<digits>\d+)"
    rf"(?P<under2>_*)(?P<close>{PAGE_NUMBER_FRAME}*)$"
)
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


def _page_number_value(line: str) -> int | None:
    """Parse a page-number line and return the printed number, or ``None``.

    The PDF converter leaves the number in several shapes, all of which have to
    survive: a bare header (``## 18``), a bare body line (``19``), the printed
    dashes (``- 2 -``, ``- 2 2 5 - -``), escaped or plain underscores that were
    emphasis markers around it (``- \\_ 1 1 \\_``), and digits spaced apart by
    the conversion (``- 1 0 4 -``, ``- 3 3``).

    A one-sided dash is accepted only when the digits themselves are spaced
    (``- 3 3``): an unspaced ``- 2`` is a Markdown list item, not a page number.
    """
    text = line.strip()
    text = re.sub(r"^#{1,6}\s*", "", text)
    # The converter escapes the underscores it keeps around the number.
    text = text.replace("\\_", "_")
    if not text or not PAGE_NUMBER_TEXT_RE.match(text):
        return None

    spaced_digits = bool(re.search(r"\d\s+\d", text))
    compact = re.sub(r"\s+", "", text)
    match = PAGE_NUMBER_COMPACT_RE.match(compact)
    if not match:
        return None
    digits = match.group("digits")
    if len(digits) > MAX_PAGE_NUMBER_DIGITS:
        return None

    has_underscore_open = bool(match.group("under"))
    has_underscore_close = bool(match.group("under2"))
    if has_underscore_open or has_underscore_close:
        # Both underscores must be there; a lone `_` is markdown emphasis.
        return int(digits) if has_underscore_open and has_underscore_close else None

    has_dash_open = bool(match.group("open"))
    has_dash_close = bool(match.group("close"))
    if has_dash_open or has_dash_close:
        if has_dash_open and has_dash_close:
            return int(digits)
        return int(digits) if spaced_digits else None
    return int(digits)


def is_page_number_line(line: str) -> bool:
    """True when the line is only a printed page number.

    Accepts ``19``, ``## 19``, ``- 19 -``, ``## — 208 —``, ``- 1 0 4 -``,
    ``- \\_ 1 1 \\_`` and one-sided spaced forms such as ``- 3 3``. A number
    inside a word or a one-sided unspaced ``- 2`` (a Markdown list item) does
    not qualify.
    """
    return _page_number_value(line) is not None


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
    """The printed page number of an artefact line, or ``None``.

    Works for every shape :func:`is_page_number_line` accepts, including the
    spaced and underscore-framed forms (``- 1 0 4 -``, ``- \\_ 1 1 \\_``).
    """
    return _page_number_value(line)


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
