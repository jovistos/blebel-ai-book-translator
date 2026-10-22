"""The prompts that build the glossary, and the renderers they need.

Two prompts, one book:

``build_system_prompt``
    sent with every chapter and identical for all of them. It states the one
    thing that makes this stage work: the model is not summarising a text, it is
    fixing the terminology of a **book that is being translated** from one
    language into another, and whatever it writes here will be reused verbatim by
    the translator that renders the book itself.
``build_chapter_prompt``
    sent once per chapter (or once per part of a long one). It carries the
    dictionary built from every earlier chapter and then the sentence evidence
    from this one, so a new name can be made compatible with the names already
    locked in.

The split is deliberate: the system prompt is constant, so a provider that
caches a prefix pays for it once, and the growing dictionary never invalidates
the part of the request that explains the task.

``render_dictionary`` is the other half of the "compatible names" promise — it is
what the next chapter's call actually sees of the work so far, and it is
deliberately minimal: one entry per line, the source term and the locked
rendering, nothing else. The category and the context are for the translator that
renders the book (they are in ``glossary.json`` and ``glossary.md``), and since
this block is re-sent with every chapter, every field kept here is paid for once
per chapter.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from ..core.llm import JSON_ONLY_NOTE

#: ISO code -> the name the prompt uses for it. Codes outside this table are
#: passed through unchanged, so an unusual pair still produces a usable prompt.
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "sr": "Serbian",
    "hr": "Croatian",
    "bs": "Bosnian",
    "sl": "Slovenian",
    "mk": "Macedonian",
    "bg": "Bulgarian",
    "ru": "Russian",
    "uk": "Ukrainian",
    "pl": "Polish",
    "cs": "Czech",
    "sk": "Slovak",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "sv": "Swedish",
    "no": "Norwegian",
    "da": "Danish",
    "fi": "Finnish",
    "el": "Greek",
    "tr": "Turkish",
    "hu": "Hungarian",
    "ro": "Romanian",
    "ar": "Arabic",
    "he": "Hebrew",
    "fa": "Persian",
    "hi": "Hindi",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ja-Latn": "Japanese (romanised)",
}

#: How a target language wants its names, beyond the generic rules. This is the
#: part of the prompt that keeps a *locked* rendering usable inside a sentence:
#: an inflected language needs the citation form plus the gender, because the
#: translator will have to decline the name in cases the glossary never saw.
TARGET_NOTES: dict[str, str] = {
    "sr": (
        "Serbian is a highly inflected language. Give every name in its citation form "
        "(nominative singular; for a plural-only name, the nominative plural), so the "
        "translator can decline it through all seven cases. Transcribe foreign names the "
        "way Serbian does (`Potter` -> `Poter`, `Phoebe` -> `Fibi`) and use the established "
        "Serbian exonym for a real place when one exists (`London` -> `London`, `Greece` -> "
        "`Grčka`); invented names are transliterated, not translated."
    ),
    "hr": (
        "Croatian is a highly inflected language. Give every name in its citation form "
        "(nominative singular), so the translator can decline it. Foreign names are "
        "transcribed the way Croatian does; real places use the established Croatian exonym."
    ),
    "ru": (
        "Russian is a highly inflected language. Give every name in its citation form "
        "(nominative singular), so the translator can decline it. Transcribe foreign names "
        "into Cyrillic the way Russian does and use the established Russian exonym for real "
        "places."
    ),
}
DEFAULT_TARGET_NOTE = (
    "Give every name in its citation form (the form a dictionary would list it under), so "
    "the translator can fit it into ordinary sentences."
)

#: Appended to the user prompt of the last attempt, when the provider is asked for
#: plain JSON mode instead of native structured output. The constant itself lives
#: in :mod:`book_splitter.core.llm`, next to the loop that appends it.

#: Headers of the two blocks of a chapter prompt. Kept as constants so the tests
#: (and the trace) can name the same strings the model sees.
DICTIONARY_HEADER = "## Dictionary built from the previous chapters"
CHAPTER_HEADER = "## Sentences from the current chapter"


def language_name(code: str) -> str:
    """``"sr"`` -> ``"Serbian"``; an unknown code is returned unchanged."""
    value = str(code or "").strip()
    if not value:
        return "the source language"
    return LANGUAGE_NAMES.get(value) or LANGUAGE_NAMES.get(value.lower()) or value


def target_note(target_language: str) -> str:
    """Language-specific rendering rules for the target, or the generic note."""
    code = str(target_language or "").strip().lower()
    return TARGET_NOTES.get(code, DEFAULT_TARGET_NOTE)


def build_system_prompt(
    source_language: str,
    target_language: str,
    book_title: str = "",
) -> str:
    """The constant half of the request: what this model is and what it must do.

    It says, in the model's own words, that a whole book is being translated and
    that its answers are the terminology the translator is not allowed to
    second-guess. That framing is the point of the prompt: a model that thinks it
    is writing a summary of names invents plausible renderings, while a model that
    knows it is pinning down a book's terminology for a translator produces one
    rendering per entity and says why.
    """
    source = language_name(source_language)
    target = language_name(target_language)
    title = (book_title or "").strip()
    book = f'"{title}"' if title else "the book you are given"
    return f"""You are the terminology curator for the full translation of one book.

{book} is being translated from {source} into {target}, chapter by chapter, by a \
translation model that cannot see the whole book at once. Your job is to decide — once, \
now — how each entity in it must be written in {target}, so that the finished translation \
never calls the same person, place, group or thing by two different names.

What you write is not a suggestion. It is the dictionary the translator is bound to: it \
will reuse your {target} rendering verbatim in every chapter, and it will not invent its \
own. Two chapters translated weeks apart must read as one book, so your rendering must be \
the one a {target} reader would expect to meet on every page.

For each chapter you receive two things: the sentences a filter kept from that chapter — \
the ones that carry a name, a place, a title or an invented term — and the dictionary the \
earlier chapters have already produced (empty for the first chapter). Read the sentences, \
find every entity in them, and lock in its {target} rendering.

Return a single JSON object with exactly this shape:

{{
  "entries": [
    {{
      "term": "...",
      "category": "...",
      "translation": "..."
    }}
  ]
}}

Fields:

- `term` — the entity exactly as this book names it, in {source}, in its canonical form \
(drop a leading article; if {source} inflects the name, use the form the book would list it \
under). One entity per entry: do not bundle "Marko and Ana" into one term, and do not use a \
grammatical form found in one sentence as the term.
- `category` — exactly one of: `character`, `location`, `organization`, `artifact`, \
`terminology`.
    - `character`: people and named animals, including nicknames, surnames-only references, \
honorifics and titles used as names ("the Doctor", "Lord Peter").
    - `location`: cities, countries, regions, realms, rivers, seas, streets and named \
buildings or rooms.
    - `organization`: guilds, companies, armies, military units, clubs, institutions, \
schools, churches, dynasties and families treated as a body.
    - `artifact`: unique objects, vehicles, weapons, devices, documents, and named \
technologies or substances.
    - `terminology`: recurring domain-specific terms that need one fixed translation — \
measurement units, historical roles, procedural or technical concepts, invented jargon.
- `translation` — the locked {target} rendering, in its citation form. Transcribe a foreign \
name the way {target} does; use the established {target} name for a real place or a real \
historical figure when one exists; translate a descriptive name (a guild, a title, a \
concept) into {target} rather than leaving it in {source}. Never leave this field in \
{source} unless the name is genuinely written that way in {target} too.
{target_note(target_language)}

Three fields, nothing else: the answer is a dictionary, not a commentary. Do not add a \
description, a translation note, an explanation, or any field the shape above does not show.

Extract entities, not vocabulary. Skip ordinary common nouns, verbs and adjectives, \
generic descriptions, numbers, dates and measurements that are not recurring terms of art, \
single letters, page numbers, and Markdown headings or fragments of them. If a name appears \
only inside a quoted title of another work, it is not an entity of this book — skip it. \
Four names in one sentence are four entries; a sentence with no entity contributes nothing.

Consistency rules, in order of importance:

1. If an entity is already in the dictionary you are given, do NOT return it again. It is \
already locked, and the earlier rendering wins.
2. If the current chapter reveals an entity that is clearly the same as one already listed \
under another form, do not create a second entry: reuse the existing `term`.
3. If a chapter reveals something genuinely new about an entry already locked (a full name \
behind a nickname, a real-world counterpart), still do not return the entry — the dictionary \
keeps what was first decided.
4. Never return two entries with the same `term`, and never return an entry whose \
`translation` is empty or identical to a placeholder such as "TBD" or "N/A".

Return only new entities. An empty `"entries": []` is a correct and expected answer for a \
chapter that introduces nothing new."""


def _field(entry: Any, name: str) -> str:
    """Read one field from a Pydantic entry, a dataclass or a plain dict."""
    value = entry.get(name) if isinstance(entry, dict) else getattr(entry, name, None)
    if value is None:
        return ""
    # An enum (``TermCategory``) stringifies as ``TermCategory.CHARACTER``.
    return " ".join(str(getattr(value, "value", value)).split())


def render_dictionary(entries: Iterable[Any], *, max_entries: int = 0) -> str:
    """The "dictionary so far" block: what the next chapter's call is told.

    **One line per locked pair, and nothing else.** The category and the context
    are written for the translator that renders the book, not for this stage: this
    block is re-sent with every chapter, so every field in it is paid for once per
    chapter of the book. What a later chapter actually needs is the pair itself —
    "this term is already taken, and this is the rendering it owns" — and that is
    exactly what a line carries. The context still goes into ``glossary.json`` and
    ``glossary.md``, where it is read by the stage that needs it.

    ``max_entries`` (0 = every entry) is the other half of the same cost control:
    when it is capped, the entries that do not fit are still named, in a one-line
    "already locked" list, so the model does not re-invent them.
    """
    items = list(entries)
    if not items:
        return "(empty — no entity has been locked in yet)"
    shown = items if max_entries <= 0 else items[: int(max_entries)]
    hidden = items[len(shown) :]
    lines = [f"- {_field(entry, 'term')} → {_field(entry, 'translation')}" for entry in shown]
    if hidden:
        already = "; ".join(
            f"{_field(entry, 'term')} → {_field(entry, 'translation')}" for entry in hidden
        )
        lines.append(
            f"({len(hidden)} further entr{'y' if len(hidden) == 1 else 'ies'} are locked in "
            f"and must not be returned again: {already})"
        )
    return "\n".join(lines)


def build_chapter_prompt(
    *,
    chapter_index: int,
    chapter_filename: str,
    chapter_title: str = "",
    sentences: Sequence[str],
    dictionary: Iterable[Any] = (),
    max_dictionary_entries: int = 0,
    part: int = 1,
    parts: int = 1,
) -> str:
    """The variable half: the dictionary so far, then this chapter's evidence.

    The order matters. The dictionary comes first because the task the model is
    given — "return only what is new" — is only answerable after it has read what
    is already locked in; the sentences come second so the last thing in the
    prompt is the text the answer must be about.
    """
    dictionary_block = render_dictionary(
        dictionary,
        max_entries=max_dictionary_entries,
    )
    title = f" — {chapter_title.strip()}" if chapter_title.strip() else ""
    part_note = f" (part {part} of {parts})" if parts > 1 else ""
    evidence = "\n".join(line.rstrip() for line in sentences if line.strip())
    return f"""{DICTIONARY_HEADER}

{dictionary_block}

{CHAPTER_HEADER}

Chapter {chapter_index}{title} ({chapter_filename}){part_note}. These are the sentences \
that carry a name, place, title or invented term, one per line, exactly as the book writes \
them:

{evidence}

Extract every entity these sentences name that is not already locked in above, each with \
the translation you are locking in for it. Return only the JSON object: \
{{"entries": [...]}}. An empty list is the right answer when nothing here is new."""


def build_repair_note(error: str) -> str:
    """The instruction added to a retry after an unusable answer.

    It names the failure so the model can correct it, and it repeats the one rule
    that produces most of the retries: the answer is the JSON object and nothing
    else.
    """
    reason = " ".join(str(error or "").split())[:400] or "the answer could not be parsed"
    return (
        f"\n\nYour previous answer was rejected: {reason}\n"
        "Answer again, and this time return only the JSON object described above — no "
        "prose, no explanation, no Markdown code fence, every field present and non-empty."
    )


__all__ = [
    "CHAPTER_HEADER",
    "DEFAULT_TARGET_NOTE",
    "DICTIONARY_HEADER",
    "JSON_ONLY_NOTE",
    "LANGUAGE_NAMES",
    "TARGET_NOTES",
    "build_chapter_prompt",
    "build_repair_note",
    "build_system_prompt",
    "language_name",
    "render_dictionary",
    "target_note",
]
