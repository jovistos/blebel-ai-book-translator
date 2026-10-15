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
