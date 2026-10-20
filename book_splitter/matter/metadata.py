"""Book metadata carried by the input filename.

A book file is usually named after the book and its author::

    Tea for the Traditionally Built - Alexander McCall Smith.md

The matter stage needs both when the splitter found no front matter and it has
to create a title page. This parser is deliberately conservative: it only
splits on markers a human put there on purpose (`` - ``, `` — ``, `` by ``,
``_by_``) and otherwise leaves the stem alone. It never invents an author.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: Separators a filename uses between the title and the author.
_DASH_SEPARATORS = re.compile(r"\s+[-\u2013\u2014]\s+")
_SLUG_SEPARATOR = re.compile(r"-{3,}")
#: ``Title by Author`` and ``Title_by_Author``.
_BY_SEPARATOR = re.compile(r"\s+(?:by|od)\s+", re.IGNORECASE)
_UNDERSCORE_BY = re.compile(r"[_-]+(?:by|od)[_-]+", re.IGNORECASE)

#: Particles that are part of a name even though they are lowercase.
_NAME_PARTICLES = frozenset(
    {
        "al",
        "bin",
        "da",
        "das",
        "de",
        "del",
        "della",
        "den",
        "der",
        "di",
        "dos",
        "du",
        "el",
        "la",
        "le",
        "st",
        "van",
        "von",
        "y",
    }
)

#: Words that usually mean the side is a title, not a person's name.
_TITLE_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "for",
        "from",
        "in",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
        "without",
    }
)

#: A trailing edition/series phrase that is not part of the title.
_TITLE_TAIL = re.compile(
    r"\s*[:\-]\s*(?:a\s+)?(?:novel|novella|short\s+stories|stories)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BookMetadata:
    """What a filename says about a book; either field may be empty."""

    title: str = ""
    author: str = ""
    stem: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"title": self.title, "author": self.author, "stem": self.stem}


def _clean(value: str) -> str:
    text = str(value or "").replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip(" -\u2013\u2014\t")
    return text


def _name_score(text: str) -> int:
    """How much a fragment looks like a personal name rather than a title."""
    words = [word for word in re.split(r"\s+", text.strip()) if word]
    if not words or len(words) > 4:
        return 0
    score = 0
    for word in words:
        lowered = word.casefold().strip(".,")
        core = word.strip(".,'’")
        if not core:
            return 0
        if lowered in _NAME_PARTICLES:
            score += 1
            continue
        if lowered in _TITLE_WORDS:
            return 0
        if core[0].isupper():
            score += 2
        elif core.isupper() and len(core) > 1:
            score += 2
        else:
            return 0
    # Names tend to be 2–4 words; a single word is weak evidence.
    return score + (1 if 2 <= len(words) <= 4 else 0)


def _title_score(text: str) -> int:
    words = [word for word in re.split(r"\s+", text.strip()) if word]
    if not words:
        return 0
    score = 0
    for word in words:
        lowered = word.casefold().strip(".,")
        if lowered in _TITLE_WORDS:
            score += 1
        elif any(ch.islower() for ch in word.strip(".,")):
            score += 1
    return score


def _swap_guess(left: str, right: str) -> tuple[str, str]:
    """Return ``(title, author)`` for a two-part ``left - right`` stem."""
    left_name, right_name = _name_score(left), _name_score(right)
    left_title, right_title = _title_score(left), _title_score(right)
    # A clear person-name side wins, whichever side it is on.
    if right_name >= left_name + 2 and right_name >= 3:
        return left, right
    if left_name >= right_name + 2 and left_name >= 3:
        return right, left
    # Otherwise the common spelling is ``Title - Author``.
    if right_name >= 3 and left_title >= 1:
        return left, right
    if left_name >= 3 and right_title >= 1:
        return right, left
    return left, right


def parse_book_metadata(source_file: str | Path) -> BookMetadata:
    """Parse ``Title - Author`` (or ``by``/``_by_``) from a book filename.

    Only the final path component is read. A filename with no separator keeps
    its whole stem as the title and leaves the author empty, rather than
    guessing that half of a real title is a person's name.
    """
    path = Path(str(source_file or "").strip())
    stem = _clean(path.stem)
    if not stem:
        return BookMetadata()

    by_match = _BY_SEPARATOR.search(stem)
    if by_match:
        title = _clean(stem[: by_match.start()])
        author = _clean(stem[by_match.end() :])
        if title and author:
            return BookMetadata(title=_TITLE_TAIL.sub("", title), author=author, stem=stem)

    underscore_by = _UNDERSCORE_BY.search(stem)
    if underscore_by:
        title = _clean(stem[: underscore_by.start()])
        author = _clean(stem[underscore_by.end() :])
        if title and author:
            return BookMetadata(title=_TITLE_TAIL.sub("", title), author=author, stem=stem)

    slug_parts = [part for part in _SLUG_SEPARATOR.split(stem) if _clean(part)]
    if len(slug_parts) >= 2:
        title, author = _swap_guess(
            _clean(slug_parts[0].replace("-", " ")),
            _clean(slug_parts[1].replace("-", " ")),
        )
        return BookMetadata(
            title=_TITLE_TAIL.sub("", _clean(title)),
            author=_clean(author),
            stem=stem,
        )

    parts = [part for part in _DASH_SEPARATORS.split(stem) if _clean(part)]
    if len(parts) >= 2:
        title, author = _swap_guess(_clean(parts[0]), _clean(parts[1]))
        # A third part is usually a series label; leave it out of the title.
        return BookMetadata(
            title=_TITLE_TAIL.sub("", _clean(title)),
            author=_clean(author),
            stem=stem,
        )

    # ``Title_Author`` without a dash is ambiguous; replace underscores only
    # when the result still reads like a title, never split it.
    cleaned = _clean(stem.replace("_", " "))
    return BookMetadata(title=_TITLE_TAIL.sub("", cleaned), author="", stem=stem)


__all__ = ["BookMetadata", "parse_book_metadata"]
