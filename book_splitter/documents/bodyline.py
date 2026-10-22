"""Body-line shape inspection (stage 2: preprocessing).

A PDF→Markdown conversion damages text in a handful of recognisable *shapes*.
This module recognises the shapes; it never repairs them and never judges the
book's meaning. Every predicate takes one or two lines and returns a boolean or
a small tuple, so the same shapes can be reported to an agent, asserted in a
test, or fed to a transformer.

The two repairs that are provably safe live in :mod:`book_splitter.preprocess.damage`;
what is here is only the *evidence* that they apply.
"""

from __future__ import annotations

import re
import unicodedata

from ..core.constants import (
    MAX_CAPTION_LINE_CHARS,
    MAX_PAGE_NUMBER_DIGITS,
    MAX_PLAUSIBLE_CAPTION_CHARS,
    MAX_STITCHED_WORD_CHARS,
    MIN_BROKEN_TAIL_LINE_CHARS,
)
from .markdown import is_page_number_line, split_leading_number, split_trailing_number

#: A word fragment that could still be an unfinished word.
WORD_CHARS = r"^\w+$"
#: Digits printed *inside* a word: ``us12 merena`` / ``sma182 njuje``.
#:
#: The shape is narrow on purpose. The digits must be glued to a lowercase
#: letter (so ``star 53 godine`` and ``Tiraž 1000 primeraka`` never match) and
#: the next non-space character must be lowercase — a page number interrupted a
#: word mid-flight. In particular the digits are *glued to the previous word*:
#: an inline number such as ``brzinu od 130 otkucaja`` is preceded by a space and
#: therefore never matches.
MIDWORD_NUMBER_RE = re.compile(
    f"(?<=[a-zžćčšđ])(\\d{{1,{MAX_PAGE_NUMBER_DIGITS}}})(?=\\s+[a-zžćčšđ])"
)
#: Digits glued to the *start* of a word: ``1otkucaja``, ``4su``, ``6procenata``.
#:
#: This is the other half of the same damage: the digit escaped the margin and
#: fused with the first word of the next line. Anchored at the start of the line
#: so a number that legitimately opens a paragraph keeps its meaning.
LEADING_GLUED_NUMBER_RE = re.compile(
    f"^(\\d{{1,{MAX_PAGE_NUMBER_DIGITS}}})(?=[a-zžćčšđ])"
)
#: A leading number that is part of *this* line's own prose: ``130 otkucaja``.
LEADING_INLINE_NUMBER_RE = re.compile(f"^\\d{{1,{MAX_PAGE_NUMBER_DIGITS}}}\\s")
#: A printed page number glued to the line start before a lowercase word:
#: ``30 spreč avaju`` / ``88 stvori``. Deleting it joins the split word, exactly
#: like a ``us12 merena`` page number in the middle of a line. The lowercase
#: lookahead is deliberate: ``2011 Godine ...`` is prose, not a page number.
LEADING_SPACED_NUMBER_RE = re.compile(
    f"^(\\d{{1,{MAX_PAGE_NUMBER_DIGITS}}})\\s+(?=[a-zžćčšđ])"
)
#: Spaces sitting in front of punctuation (``mišić ,`` → ``mišić,``).
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
#: Words that open a figure/table caption rather than a sentence about one.
CAPTION_MARKERS: tuple[str, ...] = (
    "slika", "slika.", "slike", "tabela", "tabla", "grafikon", "dijagram",
    "figure", "fig.", "fig", "table", "chart", "photo", "fotografija", "crtež", "crt.",
)
#: ``Slika 12a.`` / ``Tabela 3.2`` / ``Figure 7`` — a caption names an object.
CAPTION_NUMBERED_RE = re.compile(
    r"^(?:slika|slike|tabela|tabla|grafikon|dijagram|figure|fig\.?|table|chart|"
    r"photo|fotografija|crtež|crt\.?)\s*\d+[a-z]?\b",
    re.IGNORECASE,
)
#: ``Slike 12a i 12b ilustruju ...`` is prose *about* the figures, not a caption.
CAPTION_PROSE_RE = re.compile(
    r"^(?:slike|figure|tabele)\s+\d+[a-z]?\s+(?:i|ili|su|je|se|na|prikazuju|pokazuju|ilustruju)\b",
    re.IGNORECASE,
)


def is_blank(line: str) -> bool:
    """True for a whitespace-only line."""
    return not line.strip()


def is_page_artefact(line: str) -> bool:
    """True for a line that is only a page number, with or without ``#`` markup."""
    return is_page_number_line(line)


def is_caption_line(line: str) -> bool:
    """True when a line is a standalone figure/table caption.

    Deliberately narrow, because a caption line is *deleted*, never rewritten:

    * it opens with a caption marker followed by a number (``Slika 2 Dotok krvi``,
      ``Tabela 3.1``, ``Figure 7``);
    * it is short enough to be a label rather than a paragraph;
    * it holds at most one sentence terminator, so the ``.`` of ``Slika 4.`` is
      fine while ``Slike 12a i 12b ilustruju ... struktura. Slika 12a je ...``
      (two sentences, i.e. prose *about* the figures) is not.

    A sentence that merely mentions a figure (``Slike 12a i 12b ilustruju
    različstruktura lič nosti.``) therefore never qualifies.
    """
    text = line.strip().lstrip("#").strip()
    if not text or len(text) > MAX_CAPTION_LINE_CHARS:
        return False
    if not CAPTION_NUMBERED_RE.match(text) or CAPTION_PROSE_RE.match(text):
        return False
    if len(text) <= MAX_PLAUSIBLE_CAPTION_CHARS:
        return True
    # The ``.`` of ``Slika 4.`` counts as a terminator too, so allow two of them.
    return text.count(".") <= 2


def starts_with_glued_number(line: str) -> bool:
    """True when the line opens with digits fused to a lowercase word."""
    return bool(LEADING_GLUED_NUMBER_RE.match(line))


def starts_with_spaced_number(line: str) -> bool:
    """True when the line opens with a printed page number, then text."""
    return bool(LEADING_SPACED_NUMBER_RE.match(line))


def starts_with_inline_number(line: str) -> bool:
    """True when the line opens with a number that is followed by a separator.

    ``130 otkucaja`` is the author's own number; ``1otkucaja`` is a page number
    that escaped the margin and swallowed the start of the word.
    """
    return bool(LEADING_INLINE_NUMBER_RE.match(line))


def leading_glued_number(line: str) -> tuple[int, int] | None:
    """The ``(offset, digits)`` of a leading glued number, or ``None``."""
    m = LEADING_GLUED_NUMBER_RE.match(line)
    if not m:
        return None
    return m.start(1), int(m.group(1))


def has_midword_number(line: str, extra_chars: frozenset[int] = frozenset()) -> list[dict]:
    """Every ``<letters><digits><whitespace><letters>`` run, with offsets.

    ``extra_chars`` lists byte offsets of digits already known to be legitimate
    (a table row, a footnote), which are skipped.
    """
    hits: list[dict] = []
    for m in MIDWORD_NUMBER_RE.finditer(line):
        if m.start(1) in extra_chars:
            continue
        hits.append(
            {
                "offset": m.start(1),
                "digits": m.group(1),
                "position": "midword",
                "before": line[max(0, m.start(1) - 12) : m.start(1)],
                "after": line[m.end(1) : m.end(1) + 12],
            }
        )
    lead = LEADING_GLUED_NUMBER_RE.match(line)
    if lead and lead.start(1) not in extra_chars:
        hits.append(
            {
                "offset": lead.start(1),
                "digits": lead.group(1),
                "position": "leading",
                "before": "",
                "after": line[lead.end(1) : lead.end(1) + 12],
            }
        )
    spaced = LEADING_SPACED_NUMBER_RE.match(line)
    if spaced and spaced.start(1) not in extra_chars:
        hits.append(
            {
                "offset": spaced.start(1),
                "digits": spaced.group(1),
                "position": "leading_spaced",
                "before": "",
                "after": line[spaced.end() : spaced.end() + 12],
            }
        )
    return hits


def ends_with_broken_word(line: str) -> bool:
    """True when the line is long prose whose last word looks cut in half.

    Heuristic, on purpose: the line must be long enough to be body text, and its
    final token must be a short lowercase fragment. ``... problema i što dublje
    pronikne u njegove uzroke. To zahteva pažljivu psi`` qualifies; ``... see
    figure 2`` does not.
    """
    text = line.rstrip()
    if len(text) < MIN_BROKEN_TAIL_LINE_CHARS:
        return False
    if is_page_number_line(text):
        return False
    tail = trailing_token(text)
    if not tail or len(tail) < 2 or len(tail) > MAX_STITCHED_WORD_CHARS:
        return False
    if re.match(WORD_CHARS, tail) is None:
        return False
    return tail[-1].islower() and not tail[-1].isdigit()


def starts_with_lowercase(line: str) -> bool:
    """True when the line opens with a lowercase word — a continuation, not a title."""
    text = line.lstrip()
    return bool(text) and text[0].islower()


def trailing_token(line: str) -> str:
    """The last whitespace-separated token of a stripped line."""
    parts = line.rstrip().split()
    return parts[-1] if parts else ""


def leading_token(line: str) -> str:
    """The first whitespace-separated token of a stripped line."""
    parts = line.lstrip().split()
    return parts[0] if parts else ""


def leading_fragment(line: str) -> str:
    """The first *word* fragment of a line, with a leading page number removed.

    ``"199stvena"`` → ``"stvena"``, ``"1otkucaja"`` → ``"otkucaja"``,
    ``"182 ljutnje"`` → ``"ljutnje"``, ``"stvena"`` → ``"stvena"``. This is the
    fragment a repair glues to the previous line, so the page number that caused
    the break never survives the repair.
    """
    for token in line.lstrip().split():
        if token.isdigit():
            continue  # the printed page number itself
        glued = LEADING_GLUED_NUMBER_RE.match(token)
        if glued:
            return token[glued.end(1) :]
        return token
    return ""


def joined_word(left: str, right: str) -> str:
    """The word formed by gluing two fragments back together, page number dropped."""
    return f"{trailing_token(left)}{leading_fragment(right)}"


def is_word_continuation(left: str, right: str) -> bool:
    """True when ``right`` looks like the rest of ``left``'s last word.

    This is the shape produced when a page number was printed *inside* a word:
    ``... pažljivu psi`` + ``182 ljutnje ...``. A line that opens with a page
    number glued to a word (``1otkucaja``) counts too, because the number is the
    first thing on the new page and the word continues around it.
    """
    if not ends_with_broken_word(left):
        return False
    if is_page_number_line(right) or is_blank(right):
        return False
    nxt = right.lstrip()
    if not nxt:
        return False
    if not nxt[0].islower() and not starts_with_glued_number(nxt) and not starts_with_spaced_number(nxt):
        return False
    word = joined_word(left, nxt)
    if len(word) > MAX_STITCHED_WORD_CHARS:
        return False
    # Trailing punctuation is fine: ``božan`` + ``stvena.`` is one word followed
    # by a full stop.
    return bool(re.match(r"^\w+[.,;:!?…»\")']*$", word))


def script_of(text: str) -> str:
    """Coarse script label used to avoid gluing scripts that cannot join."""
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            for token in ("LATIN", "CYRILLIC", "GREEK", "ARABIC", "HEBREW"):
                if token in name:
                    return token.lower()
    return "unknown"


def same_script(left: str, right: str) -> bool:
    """True when both fragments use the same script (a cheap sanity check)."""
    a, b = script_of(left), script_of(right)
    return a == b or "unknown" in (a, b)


def numeric_tokens(line: str) -> list[str]:
    """Every standalone number in a line, in order."""
    return re.findall(rf"\d{{1,{MAX_PAGE_NUMBER_DIGITS}}}", line)


def count_space_before_punct(line: str) -> int:
    """How many punctuation marks have stray whitespace in front of them."""
    return len(SPACE_BEFORE_PUNCT_RE.findall(line))


def ends_sentence(line: str) -> bool:
    """True when the line's last visible character closes a sentence."""
    text = line.rstrip()
    return bool(text) and text[-1] in ".!?…\"')»"


def describes_split_number(left: str, right: str) -> dict | None:
    """Explain a ``<word fragment>`` + ``<digits> <word fragment>`` pair.

    Returns ``{"word", "digits", "left_fragment", "right_fragment"}`` when the
    pair has the page-number-inside-a-word shape, else ``None``.
    """
    left_split = split_trailing_number(trailing_token(left))
    if left_split is None:
        return None
    head, digits = left_split
    nxt = right.lstrip()
    leading = leading_token(nxt)
    if not leading:
        return None
    if not same_script(head, leading):
        return None
    return {
        "word": f"{head}{leading}",
        "digits": digits,
        "left_fragment": head,
        "right_fragment": leading,
    }


def leading_number_of(line: str) -> tuple[int, str] | None:
    """The ``(digits, rest)`` of a line that opens with a number glued to a word."""
    return split_leading_number(line.strip())


__all__ = [
    "count_space_before_punct",
    "describes_split_number",
    "ends_sentence",
    "ends_with_broken_word",
    "has_midword_number",
    "is_blank",
    "CAPTION_MARKERS",
    "is_caption_line",
    "is_page_artefact",
    "is_word_continuation",
    "joined_word",
    "leading_fragment",
    "leading_glued_number",
    "leading_number_of",
    "leading_token",
    "numeric_tokens",
    "same_script",
    "script_of",
    "starts_with_glued_number",
    "starts_with_inline_number",
    "starts_with_lowercase",
    "starts_with_spaced_number",
    "trailing_token",
]
