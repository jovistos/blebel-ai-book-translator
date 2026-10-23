"""Sentence splitting and the two cheap filters that run before the tagger.

Stage 3 is a funnel, cheapest test first:

1. **split** the chapter into sentences (pure Python, no model);
2. drop sentences **without an uppercase letter past the first word** — in a
   Latin-script language the first capital is just sentence case, so a sentence
   whose only capital is its first letter carries no proper noun at all;
3. drop sentences **shorter than four words** — too little context to be
   dictionary evidence;
4. *then* hand the survivors to the tagger (:mod:`book_translator.stages.dictionary.ner`),
   which is the only expensive step.

Steps 1–3 are deterministic and dependency-free on purpose: they cut the text the
tagger has to see, and they are the part that can be tuned and unit-tested
without a GPU.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from book_translator.documents.images import strip_images

#: Word tokens: letters, digits and underscores, Unicode-aware, so ``želja`` and
#: ``Mithril`` are single words in every language this pipeline handles.
_WORD = re.compile(r"\w+", re.UNICODE)
#: A sentence terminator plus any closing quotes/brackets stuck to it.
_TERMINATOR = re.compile(r"[.!?…]+[»“”‘’\"'’)\]]*")
#: A Markdown heading marker at the start of a block, stripped before splitting.
_HEADING_MARKER = re.compile(r"^#{1,6}\s+")
#: One dropped-sentence example kept per reason per chapter (for the trace).
EXAMPLE_CHARS = 160

#: Words whose trailing dot does not end a sentence. Titles, Latin abbreviations
#: and the Serbian/Croatian/Bulgarian shorthand a converted book is full of.
_ABBREVIATIONS = frozenset(
    {
        # titles and ranks
        "g", "gđa", "gđica", "gdin", "dr", "dr.", "prof", "mr", "ing", "ing.arch",
        "mr.sc", "doc", "sv", "svet", "vlč", "vlč.",
        "mr.", "mrs", "ms", "jr", "sr", "st", "rev", "hon", "gen", "col", "sgt",
        "capt", "lt", "pres", "gov", "sen", "rep",
        # Latin and academic
        "i.e", "e.g", "etc", "vs", "cf", "al", "ibid", "ca", "approx", "resp",
        "no", "nos", "fig", "figs", "vol", "vols", "ed", "eds", "pp", "ch", "sec",
        # Serbian/Croatian/Bulgarian shorthand
        "itd", "itd.", "tj", "tj.", "npr", "npr.", "br", "br.", "str", "str.",
        "god", "god.", "op", "op.", "cit", "cit.", "ur", "ur.", "izd", "izd.",
        "čl", "čl.", "st.", "mj", "mj.", "tzv", "tzv.", "usp", "usp.", "usporedi",
        "drž", "drž.", "sl", "sl.", "dod", "dod.", "nap", "nap.", "v", "v.",
        "sv.", "svet.", "knj", "knj.", "pogl", "pogl.", "pril", "pril.",
        # months, weekdays (English and Serbian, both dotted in some styles)
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
        "nov", "dec", "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri",
        "sat", "sun", "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "avg.",
        "sept.", "okt.", "nov.", "dec.",
        # units and business
        "inc", "ltd", "co", "corp", "dept", "est", "min", "max", "hr", "hrs",
        "kg", "km", "cm", "mm", "secs", "sq", "tel", "fax", "art", "art.",
    }
)


def split_sentences(text: str) -> list[str]:
    """Split chapter prose into sentences, deterministically.

    Paragraphs are split on blank lines; single newlines inside a paragraph are
    treated as soft wraps (PDF conversions hard-wrap prose) and joined. A
    terminator only ends a sentence when it is followed by whitespace or the end
    of the text, and a single dot after a known abbreviation (or after a
    one-letter initial, ``J. R. R. Tolkien``) does not end one at all.
    """
    sentences: list[str] = []
    for block in _blocks(text):
        sentences.extend(_split_block(block))
    return [s for s in sentences if s]


def _blocks(text: str) -> list[str]:
    """Paragraph blocks: blank-line separated, Markdown heading markers stripped."""
    blocks: list[str] = []
    current: list[str] = []
    for line in str(text).splitlines():
        if line.strip():
            current.append(line)
            continue
        if current:
            blocks.append("\n".join(current))
            current = []
    if current:
        blocks.append("\n".join(current))
    return blocks


def _split_block(block: str) -> list[str]:
    """One paragraph (or heading line) turned into its sentences."""
    text = " ".join(block.split())
    if not text:
        return []
    # Pictures are not vocabulary evidence. Remove their Markdown/HTML before
    # the sentence rules see a path such as `../00-markdown/images/rabbit.png`.
    text = strip_images(text)
    if not text:
        return []
    text = _HEADING_MARKER.sub("", text)
    if not text:
        return []

    out: list[str] = []
    start = 0
    for match in _TERMINATOR.finditer(text):
        end = match.end()
        # A dot inside a number or an ellipsis mid-word is not a boundary: only a
        # terminator followed by whitespace (or the end of the block) ends one.
        if end < len(text) and not text[end].isspace():
            continue
        if match.group(0) == "." and _is_abbreviation(text, match.start()):
            continue
        # A new sentence starts with a capital, a digit, a quote or a dash. A
        # lowercase word after the dot is a continuation — `5. maja 1990.` — and
        # so is the attribution after a closing quote (`"Ne znam…" rekao je`).
        if _continues_in_lowercase(text, end):
            continue
        sentence = text[start:end].strip()
        if sentence:
            out.append(sentence)
        start = end
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def _continues_in_lowercase(text: str, end: int) -> bool:
    """True when the first character after a terminator is a lowercase letter."""
    for char in text[end:]:
        if char.isspace():
            continue
        return char.isalpha() and char.islower()
    return False


def _is_abbreviation(text: str, dot_index: int) -> bool:
    """True when the dot at ``dot_index`` closes an abbreviation or an initial."""
    prefix = text[:dot_index].rstrip()
    word = ""
    for char in reversed(prefix):
        if char.isalnum():
            word = char + word
        else:
            break
    if not word:
        return False
    if len(word) == 1 and word.isalpha():
        return True
    return word.lower() in _ABBREVIATIONS


def word_count(sentence: str) -> int:
    """Number of word tokens in a sentence."""
    return len(_WORD.findall(sentence))


def has_later_uppercase(sentence: str) -> bool:
    """True when some word after the first one contains a capital letter.

    ``"On je otišao u Novi Sad."`` keeps its capital in ``Novi``/``Sad``;
    ``"On je otišao kući."`` has none past the first word and is dropped.
    """
    words = _WORD.findall(sentence)
    return any(char.isupper() for word in words[1:] for char in word)


def is_short(sentence: str, min_words: int = 4) -> bool:
    """True when the sentence has fewer than ``min_words`` words."""
    return word_count(sentence) < int(min_words)


@dataclass
class PreFilterResult:
    """What the dependency-free filters kept and why they dropped the rest."""

    kept: list[str] = field(default_factory=list)
    dropped_short: list[str] = field(default_factory=list)
    dropped_no_uppercase: list[str] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        return {
            "input": len(self.kept) + len(self.dropped_short) + len(self.dropped_no_uppercase),
            "kept": len(self.kept),
            "dropped_short": len(self.dropped_short),
            "dropped_no_uppercase": len(self.dropped_no_uppercase),
        }


def prefilter_sentences(
    sentences: list[str],
    *,
    min_words: int = 4,
    require_uppercase: bool = True,
) -> PreFilterResult:
    """Apply the no-uppercase test, then the minimum-length test.

    The order matters for the trace only — a sentence can fail both, and it is
    reported under the first reason that rejected it. ``require_uppercase=False``
    turns the first filter off, which is what ``--no-uppercase-filter`` does.
    """
    result = PreFilterResult()
    for sentence in sentences:
        if require_uppercase and not has_later_uppercase(sentence):
            result.dropped_no_uppercase.append(sentence)
            continue
        if is_short(sentence, min_words):
            result.dropped_short.append(sentence)
            continue
        result.kept.append(sentence)
    return result


__all__ = [
    "EXAMPLE_CHARS",
    "PreFilterResult",
    "has_later_uppercase",
    "is_short",
    "prefilter_sentences",
    "split_sentences",
    "word_count",
]
