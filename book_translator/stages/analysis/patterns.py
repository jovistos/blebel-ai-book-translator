"""Regular expressions and vocabularies used to recognise book structure.

The word lists are multilingual on purpose: the sample corpus is Serbian, but
the same PDF-conversion damage appears in English, German, Spanish, Russian,
Polish and Czech books, and the splitter should not need a per-language fork.
"""

from __future__ import annotations

import re

#: Paragraph-level strings that announce a table of contents.
TOC_WORDS: tuple[str, ...] = (
    "sadrzaj",
    "sadržaj",
    "contents",
    "table of contents",
    "toc",
    "inhalt",
    "indice",
    "índice",
    "contenido",
    "содержание",
    "оглавление",
    "kazalo",
    "obsah",
    "spis tresci",
    "spis treści",
)

#: Strings that mark a chapter/part/section opening rather than page furniture.
CHAPTER_WORDS: tuple[str, ...] = (
    "chapter",
    "part",
    "section",
    "book",
    "prologue",
    "epilogue",
    "glava",
    "poglavlje",
    "deo",
    "dio",
    "uvod",
    "predgovor",
    "pogovor",
    "zakljucak",
    "zaključak",
    "beleške",
    "beleske",
    "notes",
    "appendix",
    "dodatak",
    "sadrzaj",
    "sadržaj",
    "introduction",
    "preface",
    "foreword",
)

#: A line that starts with one of the chapter words (after markdown punctuation).
CHAPTER_RE = re.compile(
    r"^\s*[#>*\-\s]*(" + "|".join(re.escape(w) for w in CHAPTER_WORDS) + r")\b",
    re.IGNORECASE,
)
#: A line that is nothing but a page number (with optional markdown decoration).
NUMERIC_RE = re.compile(r"^[\s#*\-–—_.]*\d{1,4}[\s#*\-–—_.]*$")
#: A dot-leader TOC entry: ``Title .......... 42`` or ``Title    42``.
DOT_LEADER_RE = re.compile(r"^(.*?)(?:\s*[.\-•_]{2,}\s*|\s{3,})(\d{1,4})\s*$")
#: A numbered list heading: ``1. Title`` / ``2) Title``.
NUMBERED_RE = re.compile(r"^\d{1,2}[.)]\s+\S")
#: Dot leaders used to decide whether a block is a TOC at all.
DOT_LEADER_SCAN_RE = re.compile(r"[.\-•_]{3,}\s*\d{1,4}\s*$", re.M)

# --------------------------------------------------------------------------- #
# Back matter at the tail of the book
#
# Lower-case substrings, matched against a lower-cased line. They announce
# publisher furniture rather than prose: an author biography, the catalogue of
# the author's other books, a copyright page, a teaser excerpt, a reading-group
# guide, or the "visit / follow / friend" links of an ebook edition.
# --------------------------------------------------------------------------- #

#: ``(needle, signal name)`` — several needles may share one signal.
BACK_MATTER_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("about the author", "about-author"),
    ("a note about the author", "about-author"),
    ("about the contributors", "about-author"),
    ("about the illustrator", "about-author"),
    ("also available", "publisher-catalog"),
    ("also by", "publisher-catalog"),
    ("other books by", "publisher-catalog"),
    ("books by", "publisher-catalog"),
    ("more from", "publisher-catalog"),
    ("acknowledg", "acknowledgements"),
    ("all rights reserved", "copyright"),
    ("copyright", "copyright"),
    ("isbn", "copyright"),
    ("printed in the", "copyright"),
    ("excerpt", "excerpt"),
    ("read on for", "excerpt"),
    ("discussion questions", "reading-group"),
    ("reading group", "reading-group"),
    ("book club", "reading-group"),
    ("visit:", "publisher-links"),
    ("follow:", "publisher-links"),
    ("friend:", "publisher-links"),
    ("sign up", "publisher-links"),
    ("newsletter", "publisher-links"),
    ("www.", "publisher-links"),
    ("http://", "publisher-links"),
    ("https://", "publisher-links"),
)

#: ``In the Isabel Dalhousie Series:`` — a publisher catalogue heading.
SERIES_HEADING_RE = re.compile(r"^in the .{0,80} series:?\s*$", re.IGNORECASE)
#: A catalogue entry with an ISBN: ``Title • 978-0-307-37963-4``.
ISBN_LINE_RE = re.compile(r"(?:•|·|\*)\s*97[89][\d\-]{9,}")

#: Whole-line "about the author" headings. A heading is structurally unambiguous:
#: ``## O AUTORU`` is never chapter prose, whatever the surrounding paragraph says.
#: The list is deliberately short and multilingual — the same section has to be
#: recognised in a Serbian, German, Spanish, French or Russian edition.
ABOUT_AUTHOR_HEADINGS: tuple[str, ...] = (
    "about the author",
    "about the authors",
    "a note about the author",
    "about the writer",
    "o autoru",
    "o autorima",
    "o autorki",
    "o piscu",
    "o spisatelju",
    "o prevodiocu",
    "o prevoditelju",
    "über den autor",
    "über die autorin",
    "sobre el autor",
    "sobre la autora",
    "sobre o autor",
    "sull'autore",
    "à propos de l'auteur",
    "об авторе",
    "o avtorju",
)
ABOUT_AUTHOR_HEADING_RE = re.compile(
    r"^\s*(?:[#>*\-–—]+\s*)*("
    + "|".join(re.escape(word) for word in ABOUT_AUTHOR_HEADINGS)
    + r")\s*:?\s*$",
    re.IGNORECASE,
)

#: A sentence that reads like an author biography.
AUTHOR_BIO_RE = re.compile(
    r"\b("
    r"is the author of|is the author|was born in|is a writer|"
    r"bestselling author|award-winning author|professor emeritus|"
    r"lives in [A-Z]|"
    # Slavic editions spell the same biography shape out in their own words.
    r"autor je|autorica je|je autor|je autorica|napisao je|napisala je|"
    r"rođen je|rođena je|živi u|počasni profesor|doktorirao je|doktorirala je"
    r")\b"
)
