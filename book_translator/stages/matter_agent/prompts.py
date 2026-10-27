"""Prompts and deterministic reports for the front/back matter stage.

The prompt is deliberately *not* stage 7's. A chapter is translated literally and
the glossary is binding; front/back matter is **localized** for a new edition.
The model is told:

* this is the front/back matter of a book that is itself a translation;
* the title must be the appropriate target-language title, ideally the one a
  publisher already used (that is what `web_search` is for);
* the localized edition must say it was translated by **BleBel**;
* material that belongs only to the original edition (imprint, ISBN, CIP,
  printing history, original publisher's permissions) must be omitted, and every
  omission listed;
* the result must be formatted as published front/back matter, not as a raw
  translation dump;
* no dictionary is sent, and none should be requested.

The reports are rendered next to the prompts so what the model may see can be
read in one place.
"""

from __future__ import annotations

import json
from typing import Any

from book_translator.stages.glossary.prompts import language_name
from book_translator.stages.matter.inspect import MatterBook, MatterSection

BOOK_SYSTEM_PROMPT = """You are the localization editor for the front and back matter of a translated book.

The book you receive is being published **in another language**. Its front matter and back
matter are not ordinary chapters: they are the parts that introduce the translated edition,
name its title, credit its translator, and carry material that must be re-published (or
deliberately not re-published) in the new edition. You plan that localization once for the
whole book, before any chunk is localized.

## What you must decide
1. **The title of the translated edition.** It must sound like a published title in the
   target language. Before inventing a rendering, search the web for an existing published
   translation of this book. Use `web_search` with the original title, the author and the
   target language (and/or the target country); read the most promising result with
   `fetch_web_page`. If a published translation exists, reuse its exact title, keep the
   publisher's capitalization and punctuation, and record the URL in `title_evidence`.
   Only when no published translation can be found, translate the title appropriately.
   Never claim a title is published without evidence. If `web_search` is unavailable,
   rate-limited, or returns nothing usable, **stop searching after one attempt** and
   translate the title yourself (`title_source: "local"`); submit the plan anyway.
2. **The translator credit.** The front matter of the translated book must say, in the
   target language, that this edition was translated by **BleBel**. Supply the localized
   sentence as `translation_credit`; the name `BleBel` must be in it.
3. **What belongs in the translated edition.** Translate/localize the author's preface,
   introduction, dedication, epigraph, acknowledgements, notes, afterword, appendix and
   "about the author". Omit material that exists only because of the original printing:
   original publisher imprints, ISBN/barcode lines, CIP/legal-deposit records, printing
   history, original copyright and permissions pages, "printed in ..." lines, prices, and
   references to the original print edition. List every omitted source phrase exactly in
   `omit`; the graph checks that the phrase is gone. When in doubt about a legal notice,
   omit it and list it — a translated edition has its own legal page.
   **Never omit the author's own words, whatever they are about.** A paragraph, sentence
   or note written by the author is the book, not apparatus: even when it mentions print
   runs ("for this eighty-sixth thousand"), electrotypes, prices or the original
   publisher, it is part of the author's preface/dedication and must be translated, not
   dropped. Only material written by the publisher (imprint page, ISBN, copyright and
   permissions pages, cover/design credits) may be omitted. Inside the author's own
   preface, drop nothing but bare cross-reference pointers to the original pagination
   (for example "(see p. 59)"). When an author's paragraph and publisher apparatus share a
   passage, keep the author's paragraph whole.
4. **The shape of the matter.** Say how the first front-matter file should open (localized
   title, credit, source-language note) and give each section its localized title in
   `sections`. Keep the book's own structure: do not merge a preface into the title page,
   do not invent a table of contents, and do not add chapters.

## Hard rules
- Read `list_matter_sections` first, then `read_matter_section` for the front matter and
  every back-matter section you need to plan. You may read the body chapter *titles* with
  `list_book_chapters`; never read body prose.
- This stage has no dictionary and no glossary. Do not ask for one.
- Do not translate the whole book here. You produce a plan, not a manuscript.
- The target language is the reader's language. Everything you write into the plan is in
  the target language except `original_title`, `title_evidence` and the omitted source
  phrases, which quote the source.
- Do not invent a translator, a publisher, an ISBN or a copyright holder. The only credit
  you add is `BleBel`.

## Protocol
1. `list_matter_sections` — see every front/back file and the body chapter titles.
2. `read_matter_section` — read the title page/front matter first; read the back matter
   when it exists. Use `web_search`/`fetch_web_page` once to look for a published
   translation; if it fails, do not retry — plan from the local title/author hints.
3. `validate_matter_plan_tool` — dry-run your plan; fix every issue and warning that
   matters.
4. `report_matter_plan` — submit the complete plan. That ends this phase.

Keep your reasoning short. When you call a tool, say in one sentence why. Do not paste
tool output back into the conversation."""


CHUNK_SYSTEM_PROMPT = """You are localizing ONE chunk of the front or back matter of a translated book.

The book plan in your task is binding. It fixes the translated title, the translator
credit (`BleBel`), the per-section titles, and the exact source phrases that must be
omitted. You receive one exact chunk of a front/back matter file and return its complete
localization in Markdown.

## What "localization" means here
- Translate everything in the chunk except the phrases the plan says to omit. This is not
  a literal body-chapter translation: it must read like the front/back matter of a
  published book in the target language.
- If this chunk is the **first front-matter chunk**, the translated edition must open with
  the localized title and state that it was translated by **BleBel**. The graph adds a
  title and credit if you omit them, so write them deliberately instead of relying on the
  fallback. When the chunk carries a **cover picture**, the picture comes first: put the
  title page directly after the cover, never above it.
- Format it beautifully with Markdown: one `#` title, a short italic credit line, blank
  lines between blocks, `##` section headings for a contents/preface/notes block, lists
  for an epigraph or a contents list, and a `---` rule only when it separates the title
  page from the rest. Never wrap the whole answer in a code fence.
- Keep the author's name, the dedication, the preface's meaning and every note. Remove
  only the original-edition material listed in the plan, and never invent a replacement
  legal notice, publisher or ISBN.
- **Keep every picture.** The chunk may contain Markdown image references
  (`![…](…)`) or raw `<img …>` tags. Copy each one into the localized text exactly as it
  is, at the same point in the layout. Never translate, remove, reorder or rewrite an image
  reference, and never invent a new one.
- Do not summarize, abridge or expand. Translate the complete chunk, then localize its
  layout.
- The target language's typography is expected (quotation marks, dashes, capitalization).
  Names from the body chapters are not in a glossary here; use the forms the plan and the
  surrounding front matter establish.

## Protocol
1. The current chunk's exact text is in the task (and `read_matter_chunk` returns it
   again); `read_matter_section` can show the whole section or the title page for context;
   `web_search`/`fetch_web_page` can confirm a published title. Use them when the chunk
   alone is ambiguous.
2. `validate_matter_localization` — dry-run the complete translation. Fix every issue.
3. `report_matter_localization` — submit it. This is the only way to finish the chunk.
   Set `omitted: true` with a reason when the whole chunk is original-edition material
   the plan says to drop. On the first front-matter chunk, never omit the title/credit
   page: localize the title page and drop only the material inside it.

Keep your reasoning short. Never answer with prose outside the report tool."""

NUDGE_BOOK_NO_PLAN = """You have not submitted the localization plan yet.
List the matter sections, read the front matter, search for an existing published
translation of the title, decide the localized title and the BleBel credit, list the
original-edition phrases to omit, dry-run the plan with `validate_matter_plan_tool`, and
finish by calling `report_matter_plan`. Do not stop before it succeeds."""

NUDGE_CHUNK_NO_TRANSLATION = """You have not submitted this chunk's localization yet.
Read the exact chunk, localize all of it (except the plan's omitted phrases), format it as
published front/back matter, dry-run it with `validate_matter_localization`, and finish by
calling `report_matter_localization`. Do not stop before it succeeds."""


def _section_line(section: MatterSection) -> str:
    return (
        f"{section.filename} [{section.kind}] title={section.title!r} "
        f"chars={section.chars} chunks={section.chunk_count}"
    )


def build_book_report(book: MatterBook) -> str:
    """The compact inventory the plan agent reads first (no section prose)."""
    target = language_name(book.target_language)
    source = language_name(book.source_language)
    lines = [
        "MATTER INVENTORY (deterministic; read the text with read_matter_section)",
        f"chapters_path: {book.chapters_path}",
        f"chunks_path: {book.chunks_path}",
        f"source_file: {book.source_file or '(unknown)'}",
        f"book_title_hint: {book.book_title or '(unknown)'}",
        f"book_author_hint: {book.book_author or '(unknown)'}",
        f"languages: {source} -> {target} (source=\"{book.source_language}\", "
        f"target=\"{book.target_language}\")",
        f"matter_sections: {len(book.sections)}  matter_chunks: {book.chunk_count}",
        "",
        "front/back matter files (reading order):",
    ]
    if book.sections:
        lines.extend(f"  {_section_line(section)}" for section in book.sections)
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append(f"body chapters: {len(book.body_chapters)}")
    lines.extend(
        f"  {chapter.get('filename') or ''!s:<40} "
        f"{chapter.get('title')!r} (index {chapter.get('chapter_index')})"
        for chapter in book.body_chapters[:200]
    )
    if len(book.body_chapters) > 200:
        lines.append(f"  … {len(book.body_chapters) - 200} more")
    lines.extend(
        [
            "",
            "The body chapter titles above are the only body content you may use "
            "(for a localized contents block).",
        ]
    )
    return "\n".join(lines)


def _plan_json(plan: dict[str, Any]) -> str:
    return json.dumps(plan or {}, ensure_ascii=False, indent=2, default=str)


def build_chunk_report(
    *,
    section: MatterSection,
    chunk: Any,
    plan: dict[str, Any],
    source_language: str,
    target_language: str,
    is_first_front_chunk: bool,
    is_first_section_chunk: bool,
) -> str:
    """The per-chunk task: binding plan + the exact source chunk."""
    filename = getattr(chunk, "filename", "")
    chapter = getattr(chunk, "chapter_filename", section.filename)
    index = getattr(chunk, "chunk_index", 0)
    text = getattr(chunk, "text", "") or ""
    source = language_name(source_language)
    target = language_name(target_language)
    flags = []
    if is_first_front_chunk:
        flags.append(
            "THIS IS THE FIRST FRONT-MATTER CHUNK: open the translated edition here "
            "with the localized title and the BleBel credit."
        )
    if is_first_section_chunk and section.kind == "back":
        flags.append("THIS IS THE FIRST CHUNK OF THIS BACK-MATTER SECTION.")
    flag_block = "\n".join(f"- {flag}" for flag in flags) or "- (not a first chunk)"
    return f"""## The translated book

- source language: {source}
- target language: {target}
- matter file: {section.filename} ({section.kind} matter)
- current section title: {section.title!r}
- chunk: {filename} ({chapter}, chunk index {index})

## The binding localization plan

{_plan_json(plan)}

## This chunk's role

{flag_block}

## Source text ({source}) — localize this

{text}

## Your task

Localize the source text above into {target} as the {section.kind} matter of the
translated edition. Read the plan, drop only what it says to drop, format the result as
published Markdown, and finish with `report_matter_localization`:

{{"translation": "<the complete localized chunk>", "omitted": false, "omit_reason": ""}}

Do not summarize, do not omit a paragraph the plan did not name, do not add a sentence
the original does not support, and do not write anything outside the tool call."""


def book_validation_feedback(validation: dict[str, Any]) -> str:
    issues = "\n".join(f"- {issue}" for issue in validation.get("issues") or []) or "- (none)"
    warnings = "\n".join(
        f"- {warning}" for warning in validation.get("warnings") or []
    ) or "- (none)"
    return f"""Your localization plan was rejected. Fix these and report again:

ISSUES (must all be fixed):
{issues}

WARNINGS (read them):
{warnings}

Dry-run the corrected plan with `validate_matter_plan_tool`, then call
`report_matter_plan`. The plan must contain a target-language `translated_title` and a
`translation_credit` naming BleBel, and every omitted phrase must be an exact source
phrase."""


def chunk_validation_feedback(validation: dict[str, Any]) -> str:
    issues = "\n".join(f"- {issue}" for issue in validation.get("issues") or []) or "- (none)"
    warnings = "\n".join(
        f"- {warning}" for warning in validation.get("warnings") or []
    ) or "- (none)"
    return f"""Your localization was rejected. Fix these and report again:

ISSUES (must all be fixed):
{issues}

WARNINGS (read them):
{warnings}

Dry-run the corrected text with `validate_matter_localization`, then call
`report_matter_localization`. The translation must be the complete chunk, not a summary,
and it must not contain a phrase the plan told you to omit. On the first front-matter
chunk it must carry the localized title and the BleBel credit."""


__all__ = [
    "BOOK_SYSTEM_PROMPT",
    "CHUNK_SYSTEM_PROMPT",
    "NUDGE_BOOK_NO_PLAN",
    "NUDGE_CHUNK_NO_TRANSLATION",
    "book_validation_feedback",
    "build_book_report",
    "build_chunk_report",
    "chunk_validation_feedback",
]
