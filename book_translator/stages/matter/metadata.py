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
from pathlib import Path

from book_translator.documents.book import BookMetadata

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

#: Common digital / Libgen / piracy release patterns.
_RELEASE_DOMAIN = re.compile(
    r"\b(?:libgen(?:\.[a-z0-9]+)?|z-?lib(?:\.org)?)\b", re.IGNORECASE
)
_RELEASE_BRACKETS = re.compile(r"^\s*\[[^\]]+\]\s*")
_YEAR_PUBLISHER = re.compile(r"\(\s*\d{4}(?:,\s*[^)]*)?\s*\)")
_TRAILING_ID = re.compile(r"[-_\s]\d{5,}\s*$")
_TRAILING_ELLIPSIS = re.compile(r"\.{2,}\s*$")


def _parse_digital_release(raw_stem: str) -> tuple[str, str] | None:
    """Extract (title, author) from Libgen, Calibre, or digital release filenames."""
    has_release_domain = bool(_RELEASE_DOMAIN.search(raw_stem))
    curly_matches = re.findall(r"\{([^}]+)\}", raw_stem)
    author = ""
    for m in curly_matches:
        cleaned_m = _clean(m)
        if not cleaned_m.isdigit() and _name_score(cleaned_m) >= 2:
            author = cleaned_m
            break

    if not author and not has_release_domain:
        return None

    title = raw_stem
    title = _RELEASE_DOMAIN.sub("", title)
    title = re.sub(r"\{[^}]+\}", "", title)
    title = _YEAR_PUBLISHER.sub("", title)
    title = _RELEASE_BRACKETS.sub("", title)
    if "---" in title:
        parts = [p for p in title.split("---") if p.strip()]
        if len(parts) >= 2:
            title = parts[-1]
    title = _TRAILING_ID.sub("", title)
    title = re.sub(r"[-_]\d{4}[-_][A-Za-z0-9-_ ]+$", "", title)
    title = _TRAILING_ID.sub("", title)
    title = _TRAILING_ELLIPSIS.sub("", title)
    title = re.sub(r"_\s*", ": ", title)

    if not author and has_release_domain:
        norm = re.sub(r"-+", " ", title)
        words = norm.split()
        for length in (4, 3, 2):
            if len(words) > length:
                candidate = " ".join(words[-length:])
                if _name_score(candidate) >= 3:
                    author = candidate
                    title = " ".join(words[:-length])
                    break

    if "-" in title and " " not in title:
        title = title.replace("-", " ")
    title = _clean(title)
    if title:
        return title, author
    return None


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
        if core[0].isupper() or (core.isupper() and len(core) > 1):
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
        if lowered in _TITLE_WORDS or any(ch.islower() for ch in word.strip(".,")):
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
    raw_stem = path.stem
    stem = _clean(raw_stem)
    if not stem:
        return BookMetadata()

    digital_meta = _parse_digital_release(raw_stem)
    if digital_meta is not None:
        title, author = digital_meta
        if title:
            return BookMetadata(
                title=_TITLE_TAIL.sub("", title),
                author=author,
                stem=stem,
            )

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
