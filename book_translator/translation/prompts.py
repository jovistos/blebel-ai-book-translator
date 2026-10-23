"""The prompts that translate one chunk, and the renderers they need.

Stage 6 ends with one JSON file per chunk that lists the glossary rows the chunk
mentions. Stage 7 is the call that file exists for: one chunk in, one translated
chunk out, with the locked renderings in front of the model.

Two prompts, one per chunk:

``build_system_prompt``
    constant for the whole book. It states what the model is (the translator of
    one book, not of isolated paragraphs), what it must preserve (paragraphs,
    Markdown, quoted speech, numbers), that the glossary is binding, and that the
    answer is one JSON object with a single ``translation`` field.
``build_chunk_prompt``
    the variable half: the glossary rows first (what is already decided), then
    the optional tail of the previous translation (continuity only), then the
    source chunk itself (the evidence), and finally the one-line task.

The order is the same contract as stage 4's: the decisions the model must obey
come first, so it can read them before the text, and the text to act on comes
last. The system prompt is constant so a provider that caches a prefix pays for
it once per book.

The prompts are also saved verbatim under ``07-translation-prompts/`` (see
:mod:`book_translator.translation.runner`), so what the model actually saw can be
read without rebuilding a prompt from the glossary and the chunk.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..glossary.prompts import language_name

#: Block headers. Kept as constants so the tests, the per-chunk debug file and
#: the prompt renderer cannot drift apart.
GLOSSARY_HEADER = "## Glossary for this chunk — binding"
CONTEXT_HEADER = "## The end of the previous translation — context only"
SOURCE_HEADER = "## Source text"
TASK_HEADER = "## Your task"

#: How a target language wants the *prose*, beyond the generic rules. The script
#: note matters: the project's Serbian glossary is written in Latin letters, and
#: a translation that switches to Cyrillic would no longer match the names in it.
#: A language outside this table gets :data:`DEFAULT_STYLE_NOTE`.
TARGET_STYLE_NOTES: dict[str, str] = {
    "sr": (
        "Write standard Serbian in the ekavian dialect, in the same script as the "
        "glossary renderings (the project's Serbian glossary uses the Latin alphabet). "
        "Serbian is inflected: decline a locked name through the cases the sentence "
        "needs, but never change its stem. Use the quotation marks and dialogue dashes "
        "of published Serbian prose, and keep every quoted passage quoted."
    ),
    "hr": (
        "Write standard Croatian, in the same script as the glossary renderings. "
        "Croatian is inflected: decline a locked name through the cases the sentence "
        "needs, but never change its stem. Keep every quoted passage quoted."
    ),
    "bs": (
        "Write standard Bosnian, in the same script as the glossary renderings. "
        "Decline a locked name through the cases the sentence needs, but never change "
        "its stem. Keep every quoted passage quoted."
    ),
    "ru": (
        "Write standard Russian in Cyrillic. Russian is inflected: decline a locked "
        "name through the cases the sentence needs, but never change its stem. Keep "
        "every quoted passage quoted."
    ),
    "bg": (
        "Write standard Bulgarian in Cyrillic. Use the definite article the sentence "
        "needs, but never change the stem of a locked name. Keep every quoted passage "
        "quoted."
    ),
    "mk": (
        "Write standard Macedonian in Cyrillic. Decline a locked name as the sentence "
        "requires, but never change its stem. Keep every quoted passage quoted."
    ),
    "en": (
        "Write natural English. Keep every quoted passage quoted and preserve the "
        "source's contractions and register."
    ),
}
DEFAULT_STYLE_NOTE = (
    "Write natural {target} prose. Follow the target language's grammar and "
    "punctuation conventions, and keep every quoted passage quoted."
)


def target_style_note(target_language: str) -> str:
    """Language-specific prose rules for the target, or the generic note."""
    code = str(target_language or "").strip().lower()
    note = TARGET_STYLE_NOTES.get(code)
    if note:
        return note
    return DEFAULT_STYLE_NOTE.format(target=language_name(target_language))


def _field(entry: Any, name: str) -> str:
    """Read one field from a dataclass, a Pydantic entry or a plain dict."""
    value = entry.get(name) if isinstance(entry, dict) else getattr(entry, name, None)
    if value is None:
        return ""
    return " ".join(str(getattr(value, "value", value)).split())


def render_glossary(entries: Iterable[Any], *, target_language: str = "") -> str:
    """The binding glossary block: one ``term → rendering`` line per entry.

    Deliberately terse. Only the pair is sent because two rows already carry the
    information a translator needs ("this is taken, and this is what it owns"),
    and the block is paid for once per chunk of the book. The category is kept
    when it is there because it is read as part of the line's shape (a person
    versus a place occasionally changes the case a rendering takes), and it costs
    one word. Entries without a term or a rendering are skipped: they cannot bind
    anything.
    """
    lines: list[str] = []
    for entry in entries:
        term = _field(entry, "term")
        translation = _field(entry, "translation")
        if not term or not translation:
            continue
        category = _field(entry, "category")
        lines.append(f"- {term} → {translation}" + (f" ({category})" if category else ""))
    if not lines:
        target = language_name(target_language)
        return (
            "(no entity from this chunk is in the book glossary — render names the way "
            f"{target} does, and stay consistent with the glossary's style)"
        )
    return "\n".join(lines)


def build_system_prompt(
    source_language: str,
    target_language: str,
    book_title: str = "",
) -> str:
    """The constant half of the request: the translator's standing instructions.

    The framing matters. A model told "translate this text" produces a good
    isolated paragraph; a model told "you are translating one chunk of a book that
    will be joined back together" preserves the paragraph boundaries, keeps an
    unfinished opening sentence unfinished, and obeys the glossary instead of
    improving it. That framing is the point of this prompt.
    """
    source = language_name(source_language)
    target = language_name(target_language)
    title = (book_title or "").strip()
    book = f'"{title}"' if title else "the book you are given"
    return f"""You are the translator of one book.

{book} is being translated from {source} into {target}. You receive it one chunk at a \
time, in reading order, and every translated chunk is joined to the next to form one \
continuous {target} book. The reader will never see the chunk boundaries, so the seams \
must disappear: a chunk must read as if it were translated as part of the whole chapter.

What you receive is one chunk: an exact slice of the book's text. It may begin or end in \
the middle of a sentence or a paragraph, because it was cut for the translation model's \
context window, not by the author. Translate the words that are there. Do not complete an \
unfinished sentence, do not repeat what came before the chunk, and do not add an opening \
or a closing line that the author did not write.

Rules, in order of importance:

1. Translate all of it. Omit no sentence, clause, paragraph, epigraph, dedication or \
dialogue line. Do not summarise, abridge or expand. Translating a 900-word chunk with \
600 words is a failure even when the 600 words read well.
2. Preserve the paragraph structure. A new paragraph in the source is a new paragraph in \
the translation, and a paragraph that continues is not split. Keep blank lines between \
paragraphs and the source's whitespace-only layout.
3. Preserve the Markdown exactly as it is: heading markers (`#`), bold and italic markers \
(`**`, `*`, `_`), blockquotes (`>`), list markers, tables, horizontal rules, links, inline \
code and HTML. Translate the words inside those markers, never the markers themselves. Do \
not add a heading, a title or a horizontal rule the source does not have.
3b. The book has illustrations. Each one appears in the source as a marker such as \
`[[IMG_1]]` or `[[IMG_2]]`. Copy every marker exactly once, unchanged, in the same place it \
had in the source — on its own line when it was on its own line, and never inside a word. A \
marker is not prose: do not translate it, delete it, duplicate it, move it to another \
paragraph, or invent a new one.
4. The glossary you are given is binding, not advisory. Every rendering in it was fixed \
for this book before the translation began. Use each rendering exactly as written. When \
{target} grammar requires a case ending, decline the name but do not change its stem; \
never replace a locked name with a synonym, a translation or a different spelling.
5. Names, places and terms that are not in the glossary are still part of the same book: \
render them the way {target} normally does (transcribe a foreign name, use the established \
{target} exonym for a real place) and match the style of the glossary. If the same name \
recurs in one chunk, render it the same way every time.
6. Dialogue and quotation: keep every quoted passage quoted, keep it attributed the same \
way, and use the quotation-mark and dialogue-dash conventions of published {target} prose. \
Do not turn direct speech into reported speech or the reverse.
7. Keep numbers, dates, times and measurements as the source writes them. Translate the \
words around them (month names, units that have a normal {target} equivalent) without \
changing the figures.
8. Translate idioms, imagery and wordplay into {target} rather than word for word. The \
result must read as prose written in {target}, while saying what the {source} says.
9. Do not leave a sentence in {source} unless it is a proper name, a quoted foreign phrase \
or a term that {target} also uses unchanged.
{target_style_note(target_language)}

Return a single JSON object with exactly this shape:

{{
  "translation": "<the complete translated chunk>"
}}

`translation` is the only field, and it holds the whole translated chunk with its \
paragraphs and Markdown preserved. Escape quotes and newlines as JSON requires. The text \
of the translation must not contain a raw ASCII `"`: escape it as `\\"`, or write the \
quotation marks the target language itself prints (for Serbian, `„…“`). A raw `"` ends the \
JSON string early and the answer is rejected. No prose outside the object, no explanation, \
no translator's note, no Markdown code fence around the JSON, and no summary of what you did."""


def build_chunk_prompt(
    *,
    chunk_index: int,
    total_chunks: int,
    chunk_filename: str,
    chapter_filename: str,
    source_text: str,
    source_language: str,
    target_language: str,
    terms: Iterable[Any] = (),
    context_text: str = "",
    book_title: str = "",
) -> str:
    """The variable half: glossary, optional context, this chunk, and the task.

    ``chunk_index`` is 1-based and counts the chunks this run will translate, so
    "chunk 4 of 183" matches the progress bar. The chapter and the chunk's own
    filename are named too: when a translation reads badly, the first question is
    always "which part of which chapter is this".
    """
    source = language_name(source_language)
    target = language_name(target_language)
    title = (book_title or "").strip()
    glossary_block = render_glossary(terms, target_language=target_language)
    chapter = chapter_filename.strip() or "(chapter unknown)"
    context_block = ""
    if context_text.strip():
        context_block = f"""
{CONTEXT_HEADER}

The last part of the previous chunk's translation. It is here only to show you the \
voice, the names and the flow you are continuing. Do not translate it again, do not \
repeat it in your answer, and do not let it become part of the translation:

{context_text.strip()}

"""
    book_line = f'"{title}"' if title else "the book"
    return f"""## The book

{book_line} — {source} into {target}.

## This chunk

Chunk {chunk_index} of {total_chunks} · chapter {chapter} · file {chunk_filename}.

{GLOSSARY_HEADER}

The renderings below are fixed for this book. Use each one exactly as written; when \
{target} grammar needs a case ending, decline the name without changing its stem.

{glossary_block}
{context_block}
{SOURCE_HEADER} ({source}) — translate this

{source_text}

{TASK_HEADER}

Translate the source text above into {target}. Return only the JSON object with the \
complete translated chunk in its `translation` field:

{{"translation": "..."}}

Do not summarise, do not omit a paragraph, do not add a sentence, copy every `[[IMG_n]]` \
marker exactly once in its original place, and do not write anything outside the JSON object."""


def build_repair_note(error: str) -> str:
    """The instruction added to a retry after an unusable answer.

    It names the failure so the model can correct it, and repeats the two rules
    that produce almost every retry: the answer is a JSON object, and it must hold
    the whole translation rather than a summary of it.
    """
    reason = " ".join(str(error or "").split())[:400] or "the answer could not be used"
    return (
        f"\n\nYour previous answer was rejected: {reason}\n"
        "Translate the same chunk again. This time return only the JSON object described "
        'above — `{"translation": "..."}` — with the complete translation of every '
        "paragraph, the Markdown preserved, and no prose outside the object."
    )


__all__ = [
    "CONTEXT_HEADER",
    "DEFAULT_STYLE_NOTE",
    "GLOSSARY_HEADER",
    "SOURCE_HEADER",
    "TARGET_STYLE_NOTES",
    "TASK_HEADER",
    "build_chunk_prompt",
    "build_repair_note",
    "build_system_prompt",
    "render_glossary",
    "target_style_note",
]
