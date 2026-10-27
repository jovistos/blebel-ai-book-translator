"""Deterministic inspection for stage 9 (translated-heading postprocessing).

Every function here reads a chapter only to *measure* it. What it returns is
metadata: a title candidate, a bounded opening, one record per heading, and a
short surrounding context. The chapter's prose never becomes a tool result.

The title heuristics are deliberately conservative. A title is recognised when
it is a Markdown header, a chapter-label line (``Poglavlje 4``, ``Chapter IV``),
an ALL-CAPS short line, or a short standalone line. Anything else is treated as
prose, which is exactly the case where the stage inserts a heading instead of
promoting a random sentence.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from book_translator.core.constants import (
    DEFAULT_MAX_HEADERS,
    MATTER_DIRNAME,
    MATTER_MANIFEST_FILENAME,
    MAX_POSTPROCESS_CONTEXT_LINES,
    MAX_POSTPROCESS_HEADINGS,
    MAX_POSTPROCESS_MATTER_CHARS,
    MAX_POSTPROCESS_MATTER_LINES,
    MAX_POSTPROCESS_OPENING_CHARS,
    MAX_POSTPROCESS_OPENING_LINES,
    MAX_POSTPROCESS_TITLE_CHARS,
    MAX_TITLE_SCAN_LINES,
)
from book_translator.documents.chapters import is_toc_filename
from book_translator.documents.images import has_images
from book_translator.documents.io import load_lines
from book_translator.documents.markdown import (
    header_of,
    is_all_caps,
    is_page_number_line,
    is_plain_heading_candidate,
    is_standalone_line,
    page_number_of,
)
from book_translator.documents.text import excerpt
from book_translator.documents.window import is_body_chapter_name, matter_kind_of_name
from book_translator.stages.preprocess.inspect import classify_header, inspect_chapter

#: Words that open a chapter/section label. Multilingual on purpose, like the
#: splitter's own vocabulary.
LABEL_RE = re.compile(
    r"^\s*(?:#+\s*)?(poglavlje|glava|giava|chapter|deo|dio|part|uvod|predgovor|"
    r"pogovor|prolog|epilog|bele[sš]ke|notes|acknowledg\w*|sadr[zž]aj|contents)\b",
    re.IGNORECASE,
)
#: Number words that can stand where a digit does in a chapter label. A
#: translated book writes ``GLAVA DRUGA`` as often as ``GLAVA 2``, and both are
#: bare labels whose chapter name sits on the following line.
_NUMBER_WORDS = frozenset(
    {
        # English
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen", "twenty",
        "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
        "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth",
        "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth",
        "nineteenth", "twentieth",
        # Serbian cardinals
        "jedan", "jedna", "jedno", "dva", "dve", "dvije", "tri", "četiri",
        "cetiri", "pet", "šest", "sest", "sedam", "osam", "devet", "deset",
        "jedanaest", "dvanaest", "trinaest", "četrnaest", "cetrnaest",
        "petnaest", "šesnaest", "sesnaest", "sedamnaest", "osamnaest",
        "devetnaest", "dvadeset",
        # Serbian ordinals (masc/fem/neut), the ``GLAVA DRUGA`` shape
        "prvi", "prva", "prvo", "drugi", "druga", "drugo",
        "treći", "treća", "treće", "treci", "treca", "trece",
        "četvrti", "četvrta", "četvrto", "cetvrti", "cetvrta", "cetvrto",
        "peti", "peta", "peto", "šesti", "šesta", "šesto", "sesti", "sesta",
        "sesto", "sedmi", "sedma", "sedmo", "osmi", "osma", "osmo",
        "deveti", "deveta", "deveto", "deseti", "deseta", "deseto",
        "jedanaesti", "jedanaesta", "jedanaesto", "dvanaesti", "dvanaesta",
        "dvanaesto", "trinaesti", "trinaesta", "trinaesto", "četrnaesti",
        "cetrnaesti", "četrnaesta", "cetrnaesta", "četrnaesto", "cetrnaesto",
        "petnaesti", "petnaesta", "petnaesto", "šesnaesti", "sesnaesti",
        "šesnaesta", "sesnaesta", "šesnaesto", "sesnaesto", "sedamnaesti",
        "sedamnaesta", "sedamnaesto", "osamnaesti", "osamnaesta", "osamnaesto",
        "devetnaesti", "devetnaesta", "devetnaesto", "dvadeseti", "dvadeseta",
        "dvadeseto",
    }
)
_LABEL_WORD_PATTERN = r"(?:poglavlje|glava|giava|chapter|deo|dio|part)"
_LABEL_WORD_RE = re.compile(rf"\b{_LABEL_WORD_PATTERN}\b", re.IGNORECASE)
_NUMBER_WORD_PATTERN = "|".join(
    re.escape(word) for word in sorted(_NUMBER_WORDS, key=len, reverse=True)
)
#: One leading label or number token, with the punctuation that follows it.
_CHAPTER_LABEL_TOKEN_RE = re.compile(
    rf"^\s*(?:#{{1,6}}\s*)?(?:{_LABEL_WORD_PATTERN}|\d{{1,4}}|[ivxlcdm]+|"
    rf"{_NUMBER_WORD_PATTERN})(?:\b[\s:.\-\u2013\u2014,;()]*|$)",
    re.IGNORECASE,
)
#: A leading *number* token (a digit, a roman numeral or a number word).
_CHAPTER_NUMBER_TOKEN_RE = re.compile(
    rf"^\s*(?:#{{1,6}}\s*)?(?:\d{{1,4}}|[ivxlcdm]+|{_NUMBER_WORD_PATTERN})\b",
    re.IGNORECASE,
)

#: ``Poglavlje 4`` / ``Chapter IV`` / ``GLAVA 12: Title``.
#: Titles that make a file front/back matter even when the filename does not say
#: so (``01-predgovor.md`` is a preface, not chapter 1). The naming policy must
#: not renumber these as body chapters.
FRONT_TITLE_RE = re.compile(
    r"^\s*(?:#+\s*)?(predgovor|uvod|prolog|preface|foreword|introduction|"
    r"zahvalnost|zahvalnice|acknowledg\w*)\b",
    re.IGNORECASE,
)
BACK_TITLE_RE = re.compile(
    r"^\s*(?:#+\s*)?(pogovor|epilog|bele[sš]ke|beleske|notes|dodatak|appendix|"
    r"o autoru|o autorki|about the author|index|indeks)\b",
    re.IGNORECASE,
)
LABELLED_NUMBER_RE = re.compile(
    r"^\s*(?:#+\s*)?(?:poglavlje|glava|giava|chapter|deo|dio|part)\s+"
    r"([0-9]+|[IVXLCDM]+)\b",
    re.IGNORECASE,
)
#: ``4. Title`` — a numbered chapter heading without a label word.
NUMBERED_TITLE_RE = re.compile(r"^\s*(?:#+\s*)?(\d{1,3})\s*[.)]\s+\S")
#: ``02-chapter-2.md`` / ``12-chapter-12-slug.md``.
FILENAME_CHAPTER_RE = re.compile(r"^\d+[-_ ]*chapter[-_ ]*(\d+)", re.IGNORECASE)
FILENAME_NUMBER_RE = re.compile(r"^(\d{1,3})[-_ ]")
#: A line ending in a full stop / ellipsis is prose, not a title. A question or
#: exclamation mark is allowed: ``Šta vam je na umu?`` is a real chapter title.
SENTENCE_END_RE = re.compile(r"[.…]\s*$")


def strip_chapter_label(text: str) -> str:
    """Remove leading chapter-label and number tokens from a title.

    ``Poglavlje 4: The Storm``, ``CHAPTER TWO`` and ``GLAVA DRUGA`` all lose
    their label/number part, in either order, so the chapter's *own* words are
    what remains. A title that is only a label (``GLAVA DRUGA``) returns ``""``.
    A title with no label word at all is returned unchanged, so a book title
    that happens to start with a number word (``DESET MALIH CRNACA``) keeps it.
    """
    value = re.sub(r"^#{1,6}\s*", "", (text or "").strip()).strip()
    if not _LABEL_WORD_RE.search(value):
        return value.strip(" .:-\u2013\u2014")
    original = value
    saw_number = False
    for _ in range(4):
        if _CHAPTER_NUMBER_TOKEN_RE.match(value):
            saw_number = True
        match = _CHAPTER_LABEL_TOKEN_RE.match(value)
        if match is None:
            break
        value = value[match.end() :].strip()
    value = value.strip(" .:-\u2013\u2014")
    # ``Chapter and Verse`` is a title, not a chapter label: without a number,
    # only a title that is *entirely* label words (``CHAPTER``) strips.
    if not saw_number and value:
        return original.strip(" .:-\u2013\u2014")
    return value


def is_bare_chapter_label(text: str) -> bool:
    """True when a title is nothing but a chapter word and its number.

    The subtitle detector uses this: ``GLAVA DRUGA`` is a bare label and the
    next line is the chapter name, while ``GLAVA MRAKA`` is already a name.
    """
    value = re.sub(r"^#{1,6}\s*", "", (text or "").strip()).strip()
    if not value or not _LABEL_WORD_RE.search(value):
        return False
    return not strip_chapter_label(value)


def _natural_key(name: str) -> tuple[int, str]:
    """Sort ``2-x`` before ``10-x`` while keeping non-numeric names stable."""
    m = re.match(r"^(\d+)", name)
    return (int(m.group(1)) if m else 10**9, name.lower())


def chapter_files(chapters_path: str | Path) -> list[Path]:
    """The chapter ``*.md`` files in reading order, TOC/bookkeeping excluded."""
    folder = Path(chapters_path).expanduser()
    if folder.is_file():
        folder = folder.parent
    files = [
        path
        for path in folder.glob("*.md")
        if path.is_file() and not path.name.startswith("_") and not is_toc_filename(path.name)
    ]
    return sorted(files, key=lambda path: _natural_key(path.name))


def matter_kind(filename: str, index: int = 0, title: str = "") -> str:
    """``front`` / ``body`` / ``back`` from the filename, title, then position.

    A translated chapter folder no longer carries stage 1's ``kind`` field, so a
    preface named ``01-predgovor.md`` looks like chapter 1 by filename. Its own
    title (``PREDGOVOR``) is the evidence that it is front matter, and the same
    applies to ``Pogovor``/``Beleške``/``O autoru`` at the back. The filename
    rule is shared with the chapter window (:func:`matter_kind_of_name`), so a
    body-chapter count and this classifier cannot disagree about the same file.
    """
    kind = matter_kind_of_name(filename)
    if kind != "body":
        return kind
    clean = (title or "").strip()
    if clean and FRONT_TITLE_RE.match(clean):
        return "front"
    if clean and BACK_TITLE_RE.match(clean):
        return "back"
    if not is_body_chapter_name(filename) and index == 0:
        return "front"
    return "body"


def chapter_number(filename: str, title: str = "", fallback: int | None = None) -> int | None:
    """The printed chapter number, from the filename or the title, or a fallback."""
    name = Path(str(filename or "")).name
    m = FILENAME_CHAPTER_RE.search(name)
    if m:
        return int(m.group(1))
    m = LABELLED_NUMBER_RE.match(title or "")
    if m:
        value = m.group(1)
        if value.isdigit():
            return int(value)
    m = NUMBERED_TITLE_RE.match(title or "")
    if m:
        return int(m.group(1))
    return fallback


def title_from_filename(filename: str) -> str:
    """A readable fallback title from ``NN-chapter-3-slug.md``, never invented words."""
    stem = Path(str(filename or "")).stem
    stem = re.sub(r"^\d+[-_ ]*", "", stem)
    stem = re.sub(r"^chapter[-_ ]*\d+[-_ ]*", "", stem, flags=re.IGNORECASE)
    words = [word for word in re.split(r"[-_]+", stem) if word]
    if not words:
        return ""
    text = " ".join(words).strip()
    if text.islower():
        text = text[:1].upper() + text[1:]
    return excerpt(text, MAX_POSTPROCESS_TITLE_CHARS)


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))


def _is_short_title_shape(text: str) -> bool:
    """A short, non-sentence line that could stand alone as a heading."""
    clean = (text or "").strip()
    if not clean or len(clean) > MAX_POSTPROCESS_TITLE_CHARS:
        return False
    # A picture is not a title, even when its alt text is short: a cover that
    # opens the front matter must not become the book title.
    if has_images(clean):
        return False
    if is_page_number_line(clean) or _word_count(clean) > 16:
        return False
    # A salutation ("Dragi Frenkline,") or a list fragment ends with a comma. It
    # is prose, never a heading, and must not be picked as a title to overwrite.
    if clean.endswith(","):
        return False
    if SENTENCE_END_RE.search(clean) and not clean.endswith((":", "!")):
        # A fully uppercased line keeps its title shape even with a final
        # period: dates ("8. NOVEMBAR 2000.") and "PART ONE." are headings.
        return is_all_caps(clean)
    return True


def _looks_like_title(
    text: str,
    *,
    next_blank: bool,
    is_first: bool,
    following: str = "",
) -> bool:
    """Conservative title test used for a line that has no ``#`` markup."""
    clean = (text or "").strip()
    if not _is_short_title_shape(clean):
        return False
    if LABEL_RE.match(clean):
        return True
    if is_all_caps(clean):
        return True
    # A short standalone line is a title; a short sentence followed by prose is
    # not. This is what keeps an epigraph or a one-line paragraph out.
    if (next_blank or is_first) and len(clean) <= 120 and not following:
        return True
    return bool(next_blank and len(clean) <= 120)


def _next_nonblank(lines: list[str], start: int) -> int | None:
    for index in range(start, len(lines)):
        if lines[index].strip():
            return index
    return None


def _number_heading_is_title(
    lines: list[str],
    index: int,
    chapter_number: int | None,
    first_nonblank: int,
) -> bool:
    """A first-line bare-number heading is the title, not a printed page number.

    A translated untitled chapter keeps its number but loses the label:
    ``## 2`` is what stage 8 wrote for ``CHAPTER 2``. It used to be skipped as a
    page-number heading, so the chapter was treated as title-less and a later
    section heading became the title. Accept it when it sits on the first
    non-blank line and either matches the filename's chapter number or is
    followed by prose (a page number at the top of a chapter is followed by the
    real heading).
    """
    if index != first_nonblank:
        return False
    if chapter_number is not None and page_number_of(lines[index]) == chapter_number:
        return True
    next_index = _next_nonblank(lines, index + 1)
    if next_index is None:
        return True
    following = lines[next_index].strip()
    return (
        header_of(following) is None
        and not is_page_number_line(following)
        and not LABEL_RE.match(following)
    )


def find_title_candidate(
    lines: list[str],
    *,
    chapter_number: int | None = None,
) -> dict[str, Any]:
    """Locate the chapter title in the first :data:`MAX_TITLE_SCAN_LINES` lines.

    Returns ``{"line_number", "text", "level", "is_heading", "source",
    "subtitle", "subtitle_line"}``. ``line_number`` is ``None`` when the chapter
    starts directly on prose, which is the missing-heading case the stage fixes.
    ``chapter_number`` (usually the filename's number) lets a first-line bare
    number — ``## 2``, ``2`` — be recognised as the chapter's own title instead
    of being skipped as a printed page number.
    """
    scan = lines[:MAX_TITLE_SCAN_LINES]
    first_nonblank = next((i for i, line in enumerate(scan) if line.strip()), None)
    if first_nonblank is None:
        return {
            "line_number": None,
            "text": "",
            "level": 0,
            "is_heading": False,
            "source": "missing",
            "subtitle": "",
            "subtitle_line": None,
        }

    # 1. The very first non-blank line wins when it is already a title: a plain
    #    ``Poglavlje 1`` followed later by a section header must not be mistaken
    #    for that section header.
    first_raw = scan[first_nonblank]
    if header_of(first_raw) is None:
        clean = first_raw.strip()
        next_index = _next_nonblank(lines, first_nonblank + 1)
        next_blank = next_index is not None and next_index > first_nonblank + 1
        following = ""
        if next_index is not None and next_index <= first_nonblank + 2:
            following = lines[next_index].strip()
        plain_number_title = (
            chapter_number is not None
            and page_number_of(clean) == chapter_number
            and not following.startswith("#")
            and not LABEL_RE.match(following)
        )
        if plain_number_title or _looks_like_title(
            clean,
            next_blank=next_blank,
            is_first=True,
            following=following,
        ):
            subtitle, subtitle_line = _subtitle_after(lines, first_nonblank)
            return {
                "line_number": first_nonblank + 1,
                "text": excerpt(clean, MAX_POSTPROCESS_TITLE_CHARS),
                "level": 0,
                "is_heading": False,
                "source": "plain_number" if plain_number_title else "plain",
                "subtitle": subtitle,
                "subtitle_line": subtitle_line,
            }

    # 2. A Markdown header is the strongest signal after that. The first one in
    #    the opening window is the chapter heading.
    for index in range(first_nonblank, len(scan)):
        head = header_of(scan[index])
        if not head or not head[1].strip():
            continue
        if is_page_number_line(head[1]) and not _number_heading_is_title(
            scan, index, chapter_number, first_nonblank
        ):
            continue
        title = excerpt(head[1], MAX_POSTPROCESS_TITLE_CHARS)
        subtitle, subtitle_line = _subtitle_after(lines, index)
        return {
            "line_number": index + 1,
            "text": title,
            "level": head[0],
            "is_heading": True,
            "source": "header",
            "subtitle": subtitle,
            "subtitle_line": subtitle_line,
        }

    # 3. A plain label / standalone short line that was not the first line.
    for index in range(first_nonblank + 1, len(scan)):
        raw = scan[index]
        clean = raw.strip()
        if not clean or header_of(raw):
            continue
        next_index = _next_nonblank(lines, index + 1)
        next_blank = next_index is not None and next_index > index + 1
        following = ""
        if next_index is not None and next_index <= index + 2:
            following = lines[next_index].strip()
        if _looks_like_title(
            clean,
            next_blank=next_blank,
            is_first=index == first_nonblank,
            following=following,
        ):
            subtitle, subtitle_line = _subtitle_after(lines, index)
            return {
                "line_number": index + 1,
                "text": excerpt(clean, MAX_POSTPROCESS_TITLE_CHARS),
                "level": 0,
                "is_heading": False,
                "source": "plain",
                "subtitle": subtitle,
                "subtitle_line": subtitle_line,
            }
        # The first nonblank line was not a title: everything after it is prose
        # or a late heading. Stop rather than promote a random line.
        break

    return {
        "line_number": None,
        "text": "",
        "level": 0,
        "is_heading": False,
        "source": "missing",
        "subtitle": "",
        "subtitle_line": None,
    }


def _subtitle_after(lines: list[str], title_index: int) -> tuple[str, int | None]:
    """A short second line belonging to a bare label (``Poglavlje 1`` / title)."""
    head = header_of(lines[title_index])
    title = head[1] if head else lines[title_index].strip()
    # A title that still carries words after its label/number is already the
    # full title (``Poglavlje 1: The Storm``); there is no subtitle to fold in.
    if strip_chapter_label(title):
        return "", None
    next_index = _next_nonblank(lines, title_index + 1)
    if next_index is None or next_index >= MAX_TITLE_SCAN_LINES:
        return "", None
    candidate_line = lines[next_index].strip()
    candidate_head = header_of(candidate_line)
    if candidate_head is not None:
        # ``# CHAPTER TWO`` followed by ``# WALKING IS GOOD FOR YOU``: the second
        # level-1 heading belongs to the chapter title, not to its body. A
        # deeper heading is a section of the chapter, never its subtitle.
        if candidate_head[0] != 1:
            return "", None
        candidate = candidate_head[1].strip()
    else:
        candidate = candidate_line
    if not candidate:
        return "", None
    # A following "PART ONE" style heading is structural, not this chapter's name.
    if is_bare_chapter_label(candidate):
        return "", None
    if not _is_short_title_shape(candidate) or _word_count(candidate) > 14:
        return "", None
    if SENTENCE_END_RE.search(candidate) and not candidate.endswith((":", "!")):
        return "", None
    return excerpt(candidate, 120), next_index + 1


def opening_records(
    lines: list[str],
    *,
    title_line: int | None = None,
    subtitle_line: int | None = None,
    max_lines: int = MAX_POSTPROCESS_OPENING_LINES,
    max_chars: int = MAX_POSTPROCESS_OPENING_CHARS,
) -> list[dict[str, Any]]:
    """The first few non-blank lines, clipped and labelled. Never the chapter."""
    out: list[dict[str, Any]] = []
    used = 0
    for index, raw in enumerate(lines, start=1):
        if len(out) >= max_lines:
            break
        clean = raw.strip()
        if not clean:
            continue
        text = excerpt(clean, 160)
        if used + len(text) > max_chars:
            break
        used += len(text)
        out.append(
            {
                "line_number": index,
                "text": text,
                "is_header": header_of(raw) is not None,
                "is_title": index == title_line,
                "is_subtitle": index == subtitle_line,
            }
        )
    return out


PLAIN_HEADING_EVIDENCE = (
    "standalone ALL-CAPS line with no '#' markup - a section heading whose markup was lost"
)


def plain_heading_candidates(
    lines: list[str],
    *,
    skip_lines: set[int] | None = None,
    matter: str = "body",
) -> list[dict[str, Any]]:
    """Plain ALL-CAPS lines that are section headings missing their markup.

    A translation chunk can drop the ``##`` from a real heading while keeping the
    words (``JUTRO``, ``POPOLE``, ``ČETVRTAK, 16. AVGUST 2012.``). Those lines are
    offered to the chapter agent as ``promote`` candidates: the only edit is
    putting the markup back. Front/back matter is skipped, because a dedication
    or title page is full of short ALL-CAPS names that are not headings.
    """
    if matter != "body":
        return []
    skip = skip_lines or set()
    out: list[dict[str, Any]] = []
    for index, raw in enumerate(lines):
        number = index + 1
        if number in skip:
            continue
        text = raw.strip()
        if not text or header_of(raw) is not None:
            continue
        if not is_standalone_line(lines, index) or not is_plain_heading_candidate(text):
            continue
        next_index = _next_nonblank(lines, index + 1)
        next_text = lines[next_index].strip() if next_index is not None else ""
        out.append(
            {
                "line_number": number,
                "level": 0,
                "text": excerpt(text, 110),
                "preview": excerpt(text, 110),
                "recommendation": "promote",
                "risk": "low",
                "is_all_caps": True,
                "has_structural_token": False,
                "ends_sentence": bool(SENTENCE_END_RE.search(text)),
                "next_is_header": bool(next_text and header_of(next_text) is not None),
                "next_starts_lowercase": bool(next_text[:1].islower()),
                "next_line_number": (next_index + 1) if next_index is not None else None,
                "next_text": excerpt(next_text, 110),
                "previous_is_blank": True,
                "evidence": [PLAIN_HEADING_EVIDENCE],
            }
        )
    return out


@dataclass
class ChapterStructure:
    """One chapter's title and heading metadata (never its prose)."""

    filename: str
    path: str
    line_count: int
    bytes: int
    index: int = 0
    body_index: int | None = None
    matter: str = "body"
    chapter_number: int | None = None
    title: str = ""
    title_line: int | None = None
    title_level: int = 0
    title_is_heading: bool = False
    title_source: str = "missing"
    subtitle: str = ""
    subtitle_line: int | None = None
    opening: list[dict[str, Any]] = field(default_factory=list)
    headings: list[dict[str, Any]] = field(default_factory=list)
    heading_count: int = 0
    headings_truncated: bool = False
    suspicious_headings: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_body(self) -> bool:
        return self.matter == "body"

    def summary(self) -> dict[str, Any]:
        """Compact record for the book-level inventory (no heading list)."""
        return {
            "index": self.index,
            "body_index": self.body_index,
            "filename": self.filename,
            "matter": self.matter,
            "chapter_number": self.chapter_number,
            "title": self.title,
            "title_line": self.title_line,
            "title_level": self.title_level,
            "title_is_heading": self.title_is_heading,
            "title_source": self.title_source,
            "subtitle": self.subtitle,
            "heading_count": self.heading_count,
            "headings_truncated": self.headings_truncated,
            "suspicious_heading_count": len(self.suspicious_headings),
            "line_count": self.line_count,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _compact_heading(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "line_number": record.get("line_number"),
        "level": record.get("level"),
        "text": record.get("preview") or excerpt(record.get("text", ""), 120),
        "suggested": record.get("recommendation", "keep"),
        "risk": record.get("risk", "low"),
        "is_all_caps": bool(record.get("is_all_caps")),
        "has_structural_token": bool(record.get("has_structural_token")),
        "ends_sentence": bool(record.get("ends_sentence")),
        "next_is_header": bool(record.get("next_is_header")),
        "next_starts_lowercase": bool(record.get("next_starts_lowercase")),
        "next_line_number": record.get("next_line_number"),
        "next_text": record.get("next_text", ""),
        "previous_is_blank": bool(record.get("previous_is_blank", True)),
        "evidence": list(record.get("evidence") or []),
    }


def scan_chapter_structure(
    chapter_path: str | Path,
    *,
    index: int = 0,
    body_index: int | None = None,
    max_headers: int = DEFAULT_MAX_HEADERS,
) -> ChapterStructure:
    """Read one chapter's title and heading records (metadata only)."""
    path = Path(chapter_path).expanduser()
    lines = load_lines(path)
    candidate = find_title_candidate(lines, chapter_number=chapter_number(path.name))
    title = candidate.get("text") or ""
    source = candidate.get("source") or "missing"
    if not title:
        title = title_from_filename(path.name)
        source = "filename" if title else "missing"
    matter = matter_kind(path.name, index, title)
    number = chapter_number(path.name, title, fallback=body_index)

    report = inspect_chapter(path, max_headers=min(max_headers, MAX_POSTPROCESS_HEADINGS))
    headings = [_compact_heading(record) for record in report.get("headers") or []]
    title_line = candidate.get("line_number")
    # A subtitle heading (``# CHAPTER TWO`` + ``# WALKING IS GOOD FOR YOU``) is
    # part of the chapter title, so it is never offered as a heading to demote.
    subtitle_line = candidate.get("subtitle_line")
    # A real heading whose ``##`` markup was lost is still a heading: offer it as
    # a ``promote`` candidate instead of letting it stay invisible.
    headings.extend(
        _compact_heading(record)
        for record in plain_heading_candidates(
            lines,
            skip_lines={line for line in (title_line, subtitle_line) if line},
            matter=matter,
        )
    )
    headings.sort(key=lambda item: item["line_number"] or 0)
    if subtitle_line:
        headings = [item for item in headings if item["line_number"] != subtitle_line]
    for heading in headings:
        heading["is_chapter_title"] = bool(heading["line_number"] == title_line)
    suspicious = [
        heading
        for heading in headings
        if heading["suggested"] != "keep" and heading["line_number"] != title_line
    ]

    structure = ChapterStructure(
        filename=path.name,
        path=str(path),
        line_count=len(lines),
        bytes=path.stat().st_size,
        index=index,
        body_index=body_index,
        matter=matter,
        chapter_number=number,
        title=title,
        title_line=title_line,
        title_level=int(candidate.get("level") or 0),
        title_is_heading=bool(candidate.get("is_heading")),
        title_source=source,
        subtitle=candidate.get("subtitle") or "",
        subtitle_line=candidate.get("subtitle_line"),
        opening=opening_records(
            lines,
            title_line=title_line,
            subtitle_line=candidate.get("subtitle_line"),
        ),
        headings=headings,
        heading_count=len(headings),
        headings_truncated=bool(report.get("truncated")),
        suspicious_headings=suspicious,
    )
    return structure


def _title_label(title: str) -> str:
    m = LABEL_RE.match(title or "")
    return m.group(1).lower() if m else ""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def book_identity(chapters_path: str | Path) -> dict[str, str]:
    """The original title/author and languages from the sibling matter plan.

    Stage 9 receives translated chapters, whose ``book_title`` is already in the
    target language, but a web search for the published contents needs the
    *original* title and the author. The matter stage (or its plan file) records
    both next to the chapters; this reads them when present.
    """
    folder = Path(chapters_path).expanduser()
    if folder.is_file():
        folder = folder.parent
    matter = folder.parent / MATTER_DIRNAME
    manifest = _read_json(matter / MATTER_MANIFEST_FILENAME)
    plan = manifest.get("book_plan")
    if not isinstance(plan, dict) or not plan:
        plan = _read_json(matter / "book-plan.json")
    return {
        "original_title": str(plan.get("original_title") or ""),
        "book_author": str(plan.get("author") or ""),
        "translated_title": str(plan.get("translated_title") or ""),
        "source_language": str(manifest.get("source_language") or ""),
        "target_language": str(manifest.get("target_language") or ""),
    }


def scan_book_structure(
    chapters_path: str | Path,
    *,
    include_headings: bool = False,
) -> dict[str, Any]:
    """The book-level inventory: every chapter title, front/back matter and TOC.

    ``include_headings=False`` (the default) keeps the result cheap enough to be
    the first thing an agent sees. Pass ``True`` for the full per-chapter records
    when a caller (the runner) needs them.
    """
    folder = Path(chapters_path).expanduser()
    if folder.is_file():
        folder = folder.parent
    files = chapter_files(folder)
    structures: list[ChapterStructure] = []
    for index, path in enumerate(files):
        structures.append(scan_chapter_structure(path, index=index))

    # Body numbering is assigned after classification so a front/back-matter
    # section never consumes a chapter number. `Poglavlje 1` must be the first
    # real chapter even when a preface precedes it.
    body_index = 0
    for structure in structures:
        if not structure.is_body:
            continue
        body_index += 1
        structure.body_index = body_index
        if structure.chapter_number is None:
            structure.chapter_number = body_index

    body = [structure for structure in structures if structure.is_body]
    front = next((s for s in structures if s.matter == "front"), None)
    back = next((s for s in structures if s.matter == "back"), None)
    labels = [_title_label(s.title) for s in body if _title_label(s.title)]
    label_counts: dict[str, int] = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    dominant = max(label_counts, key=lambda label: label_counts[label]) if label_counts else ""
    consistent = bool(body) and all(
        _title_label(s.title) == dominant for s in body if _title_label(s.title)
    )
    # A title that lacks the dominant label is inconsistent even when other
    # titles carry it; that is the "one chapter was never numbered" shape.
    if dominant and any(not _title_label(s.title) for s in body):
        consistent = False

    from .toc import detect_toc  # local import: toc imports chapter_files from here

    toc = detect_toc(folder, [structure.summary() for structure in structures])
    identity = book_identity(folder)
    return {
        "chapters_path": str(folder),
        "chapter_count": len(structures),
        "body_chapter_count": len(body),
        "book_title": (front.title if front else "") or (body[0].title if body else ""),
        "original_title": identity["original_title"],
        "book_author": identity["book_author"],
        "translated_title": identity["translated_title"],
        "source_language": identity["source_language"],
        "target_language": identity["target_language"],
        "front_matter": front.summary() if front else None,
        "back_matter": back.summary() if back else None,
        "chapters": [s.summary() for s in structures],
        "body_chapters": [s.summary() for s in body],
        "observed_labels": label_counts,
        "dominant_label": dominant,
        "consistent_titles": consistent,
        "toc": toc,
        "structures": [s.to_dict() for s in structures] if include_headings else None,
    }


def _block_records(
    path: Path,
    *,
    max_lines: int = MAX_POSTPROCESS_MATTER_LINES,
    max_chars: int = MAX_POSTPROCESS_MATTER_CHARS,
) -> dict[str, Any]:
    """Bounded head + tail records of one matter file, with its headings."""
    lines = load_lines(path)
    headings = [
        {"line_number": index, "level": head[0], "text": excerpt(head[1], 120)}
        for index, line in enumerate(lines, start=1)
        if (head := header_of(line))
    ]
    head_count = max(4, max_lines // 2)
    tail_count = max(4, max_lines - head_count)
    entries: list[dict[str, Any]] = []

    def add(index: int) -> None:
        raw = lines[index]
        clean = raw.strip()
        if not clean:
            return
        entries.append(
            {
                "line_number": index + 1,
                "text": excerpt(clean, 160),
                "is_header": header_of(raw) is not None,
            }
        )

    for index in range(min(head_count, len(lines))):
        add(index)
    tail_start = max(head_count, len(lines) - tail_count)
    for index in range(tail_start, len(lines)):
        add(index)

    used = 0
    trimmed: list[dict[str, Any]] = []
    for entry in entries:
        if used + len(entry["text"]) > max_chars:
            break
        used += len(entry["text"])
        trimmed.append(entry)
    return {
        "filename": path.name,
        "path": str(path),
        "line_count": len(lines),
        "heading_count": len(headings),
        "headings": headings[:30],
        "lines": trimmed,
        "truncated": len(trimmed) < len(entries),
    }


def inspect_matter_blocks(chapters_path: str | Path) -> dict[str, Any]:
    """Bounded front/back matter + the current TOC state.

    This is the only inspector that reads outside chapter openings. It returns at
    most :data:`MAX_POSTPROCESS_MATTER_LINES` clipped lines per block, never a
    whole file, and it includes the deterministic TOC detection so the agent can
    choose ``create`` / ``update`` / ``keep`` without guessing.
    """
    book = scan_book_structure(chapters_path)
    folder = Path(str(book["chapters_path"]))
    front_summaries = [
        summary
        for summary in book.get("chapters") or []
        if summary.get("matter") == "front"
    ]
    front_blocks = [
        _block_records(folder / str(front_summaries[0]["filename"]))
    ] if front_summaries else []
    back_summaries = [
        summary
        for summary in book.get("chapters") or []
        if summary.get("matter") == "back"
    ]
    back_blocks = [
        _block_records(folder / str(back_summaries[0]["filename"]))
    ] if back_summaries else []
    return {
        "chapters_path": str(folder),
        "book_title": book.get("book_title", ""),
        "chapter_count": book.get("chapter_count", 0),
        "body_chapter_count": book.get("body_chapter_count", 0),
        "observed_labels": book.get("observed_labels"),
        "dominant_label": book.get("dominant_label"),
        "consistent_titles": book.get("consistent_titles"),
        "chapter_titles": book.get("body_chapters") or [],
        "front_matter": front_blocks[0] if front_blocks else None,
        "front_matter_files": [str(item["filename"]) for item in front_summaries],
        "back_matter": back_blocks[0] if back_blocks else None,
        "back_matter_files": [str(item["filename"]) for item in back_summaries],
        "toc": book.get("toc") or {},
    }


def inspect_heading_context(
    chapter_path: str | Path,
    line_number: int,
    *,
    context_lines: int = 1,
) -> dict[str, Any]:
    """The heading at ``line_number`` plus its clipped surrounding context.

    This is what an agent calls when the header inventory is ambiguous. The
    result is capped at a few short lines and never includes a paragraph body.
    """
    path = Path(chapter_path).expanduser()
    lines = load_lines(path)
    try:
        target = int(line_number)
    except (TypeError, ValueError):
        return {"error": f"line_number must be an integer, got {line_number!r}"}
    if target < 1 or target > len(lines):
        return {"error": f"line {target} is out of range (file has {len(lines)} lines)"}
    head = header_of(lines[target - 1])
    if head is None:
        return {"error": f"line {target} is not a Markdown header", "line_number": target}
    spread = max(0, min(int(context_lines), MAX_POSTPROCESS_CONTEXT_LINES))
    before: list[dict[str, Any]] = []
    after: list[dict[str, Any]] = []
    index = target - 2
    while index >= 0 and len(before) < spread:
        if lines[index].strip():
            before.append(
                {
                    "line_number": index + 1,
                    "text": excerpt(lines[index], 140),
                    "is_header": header_of(lines[index]) is not None,
                }
            )
        index -= 1
    index = target
    while index < len(lines) and len(after) < spread:
        if lines[index].strip():
            after.append(
                {
                    "line_number": index + 1,
                    "text": excerpt(lines[index], 140),
                    "is_header": header_of(lines[index]) is not None,
                }
            )
        index += 1
    record = classify_header(lines, target)
    return {
        "file": path.name,
        "line_count": len(lines),
        "line_number": target,
        "level": head[0],
        "text": excerpt(head[1], 120),
        "suggested": record.recommendation,
        "risk": record.risk,
        "evidence": record.evidence,
        "before": before,
        "after": after,
        "note": "Context is clipped and capped; this is not the chapter text.",
    }


__all__ = [
    "ChapterStructure",
    "book_identity",
    "chapter_files",
    "chapter_number",
    "find_title_candidate",
    "inspect_heading_context",
    "inspect_matter_blocks",
    "is_toc_filename",
    "matter_kind",
    "opening_records",
    "plain_heading_candidates",
    "scan_book_structure",
    "scan_chapter_structure",
    "title_from_filename",
]
