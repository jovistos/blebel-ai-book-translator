"""What kind of boundary a chunk cut lands on.

Chonkie's recursive chunker prefers paragraphs, then sentences, then lines, then
words. This module names the result so the manifest can say, for every chunk,
whether the cut before it preserved a paragraph, only a sentence, or — when the
chunk length was too small for the content — a line or a word.

Nothing here guesses from the chunk text: a boundary is classified from the
*position* in the original chapter text, which is also the position the chunk was
sliced from. That keeps the report honest about what actually happened.
"""

from __future__ import annotations

import re

#: Characters that end a sentence in the languages this pipeline handles.
SENTENCE_END = frozenset(".!?…")
#: Closing punctuation that may follow the sentence-ending mark: ``."``, ``.”``.
CLOSERS = frozenset("\"'”’»›)]}")
#: A blank line, optionally carrying indentation, immediately before the cut.
_PARAGRAPH_TAIL = re.compile(r"\n[ \t]*\n[ \t]*$")

#: Boundary kinds, from strongest (a paragraph is whole) to weakest (a word is cut).
BOUNDARY_KINDS: tuple[str, ...] = (
    "chapter-start",
    "chapter-end",
    "paragraph",
    "sentence",
    "line",
    "word",
    "hard",
)

#: A cut that keeps the paragraph it precedes intact.
PARAGRAPH_SAFE: frozenset[str] = frozenset({"chapter-start", "paragraph"})
#: A cut that keeps at least the sentence it precedes intact.
SENTENCE_SAFE: frozenset[str] = frozenset({"chapter-start", "paragraph", "sentence"})
#: A cut inside a word. The chunker only does this for content with no whitespace
#: wide enough for the requested chunk length.
WORD_BREAKING: frozenset[str] = frozenset({"hard"})


def _ends_sentence(value: str) -> bool:
    """True when ``value`` ends with sentence punctuation, allowing closers."""
    trimmed = value.rstrip()
    for _ in range(4):
        if trimmed and trimmed[-1] in CLOSERS:
            trimmed = trimmed[:-1].rstrip()
            continue
        break
    return bool(trimmed) and trimmed[-1] in SENTENCE_END


def boundary_kind(text: str, position: int) -> str:
    """Classify the cut at ``position`` in ``text``.

    ``position == 0`` is the start of a chapter and ``position == len(text)`` its
    end. Everything else is inspected from the text *before* the cut, because a
    boundary belongs to the chunk that ends there and the chunk that starts there
    in equal measure.
    """
    if position <= 0:
        return "chapter-start"
    if position >= len(text):
        return "chapter-end"
    before = text[:position]
    if _PARAGRAPH_TAIL.search(before[-256:]):
        return "paragraph"
    if _ends_sentence(before):
        return "sentence"
    if before[-1] in "\r\n":
        return "line"
    if before[-1].isspace():
        return "word"
    return "hard"


def boundary_counts(kinds) -> dict[str, int]:
    """Count boundary kinds, with every known kind present (zero when unused)."""
    counts = dict.fromkeys(BOUNDARY_KINDS, 0)
    for kind in kinds:
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def is_paragraph_safe(kind: str) -> bool:
    return kind in PARAGRAPH_SAFE


def is_sentence_safe(kind: str) -> bool:
    return kind in SENTENCE_SAFE


__all__ = [
    "BOUNDARY_KINDS",
    "CLOSERS",
    "PARAGRAPH_SAFE",
    "SENTENCE_END",
    "SENTENCE_SAFE",
    "WORD_BREAKING",
    "boundary_counts",
    "boundary_kind",
    "is_paragraph_safe",
    "is_sentence_safe",
]
