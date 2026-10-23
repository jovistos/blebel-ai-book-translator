"""Tools for the front/back matter localization agent.

This is the one stage that *must* read prose: its job is to translate a title
page, a preface, a copyright page and a notes section. The tools are therefore
different from the metadata-only tools of stages 1, 2 and 9:

**book-plan family** (one plan per book)
    ``list_matter_sections``, ``read_matter_section``, ``list_book_chapters``,
    ``web_search`` (reused from the splitter), ``fetch_web_page``,
    ``validate_matter_plan_tool`` and ``report_matter_plan``.

**per-chunk family** (one localization per matter chunk)
    ``read_matter_chunk``, ``read_matter_section`` (for the surrounding title
    page and table of contents), ``web_search`` / ``fetch_web_page`` (for an
    existing published title), ``validate_matter_localization`` and
    ``report_matter_localization``.

The shared tools of the other agents are bound as well where they help:
``list_chapter_files``, ``extract_markdown_headers`` and ``inspect_line_window``
answer structural questions without inventing a new inspector. The write side is
not a model tool: the graph writes the accepted translation after validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from langchain_core.tools import InjectedToolArg, tool

from book_translator.stages.agent.tools.registry import matter_stage
from book_translator.stages.matter.inspect import (
    MatterBook,
    MatterSection,
    body_chapter_titles,
    read_section_text,
)
from book_translator.stages.matter.plan import (
    validate_matter_plan as validate_matter_plan_fn,
)
from book_translator.stages.matter.plan import (
    validate_matter_translation as validate_matter_translation_fn,
)

from ...core.constants import (
    DEFAULT_MATTER_CREDIT,
    MATTER_TRANSLATOR_NAME,
    MAX_MATTER_READ_CHARS,
)
from ...core.errors import ToolError
from ...documents.text import jdump
from ..context import ToolContext

#: A tool call carries only what the model chose. The book, section and chunk the
#: graph is working on arrive explicitly as :class:`ToolContext`, so a call is
#: resolved against the run instead of against a process-global scope.


def _book(ctx: ToolContext) -> MatterBook:
    book = ctx.book
    if not isinstance(book, MatterBook):
        raise ToolError("the matter tools have no book inventory")
    return book


def _section(ctx: ToolContext, filename: str = "") -> MatterSection:
    section = ctx.section
    wanted = Path(str(filename or "")).name
    if wanted and wanted != getattr(section, "filename", ""):
        found = _book(ctx).section(wanted)
        if found is None:
            raise ToolError(
                f"unknown front/back matter file {wanted!r}; known: "
                + ", ".join(section.filename for section in _book(ctx).sections)
            )
        return found
    if isinstance(section, MatterSection):
        return section
    if wanted:
        found = _book(ctx).section(wanted)
        if found is not None:
            return found
    raise ToolError("no current matter section; pass the filename from the plan report")


def _chunk_text(ctx: ToolContext, filename: str = "") -> dict[str, Any]:
    """The current chunk's exact text, or another matter chunk by filename."""
    chunk = ctx.chunk
    wanted = Path(str(filename or "")).name
    if wanted and wanted != getattr(chunk, "filename", ""):
        book = _book(ctx)
        folder = Path(book.chunks_path)
        path = folder / wanted
        if not path.is_file():
            raise ToolError(f"matter chunk not found: {wanted} (folder: {folder})")
        return {
            "filename": wanted,
            "path": str(path),
            "chars": len(path.read_text(encoding="utf-8")),
            "text": path.read_text(encoding="utf-8"),
        }
    if chunk is None:
        return {}
    if isinstance(chunk, dict):
        return {
            "filename": chunk.get("filename"),
            "path": chunk.get("path"),
            "chars": len(str(chunk.get("text") or "")),
            "text": str(chunk.get("text") or ""),
        }
    return {
        "filename": getattr(chunk, "filename", ""),
        "path": getattr(chunk, "path", ""),
        "chars": len(getattr(chunk, "text", "") or ""),
        "text": getattr(chunk, "text", "") or "",
        "chapter_filename": getattr(chunk, "chapter_filename", ""),
        "chunk_index": getattr(chunk, "chunk_index", 0),
    }


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
@matter_stage
@tool
def list_matter_sections(*, ctx: Annotated[ToolContext, InjectedToolArg]) -> str:
    """List the book's front and back matter files, in reading order.

    Call this first. It returns every matter file with its `kind` (`front` or
    `back`), title, line/character count and the number of chunks stage 5 cut
    from it, plus the body chapter titles a localized table of contents may
    need. It never returns prose; use `read_matter_section` for that.
    """
    book = _book(ctx)
    return jdump(
        {
            "chapters_path": book.chapters_path,
            "chunks_path": book.chunks_path,
            "book_title": book.book_title,
            "source_file": book.source_file,
            "source_language": book.source_language,
            "target_language": book.target_language,
            "sections": [section.to_dict() for section in book.sections],
            "body_chapters": list(book.body_chapters),
        }
    )


@matter_stage
@tool
def read_matter_section(
    filename: str,
    max_chars: int = MAX_MATTER_READ_CHARS,
    *,
    ctx: Annotated[ToolContext, InjectedToolArg],
) -> str:
    """Read one front/back matter file as text.

    `filename` is a name from `list_matter_sections` (e.g. `00-front-matter.md`
    or `01-predgovor.md`). The whole section is returned up to a hard cap; the
    result says `truncated: true` when the cap was hit. Unlike the splitter and
    postprocessor, this stage is allowed to read the text: it has to localize it.
    """
    section = _section(ctx, filename)
    report = read_section_text(section, max_chars=max_chars)
    report["note"] = (
        "The text of one front/back matter section. Translate/localize it; never "
        "let text from the body chapters leak into it."
    )
    return jdump(report)


@matter_stage
@tool
def read_matter_chunk(
    filename: str = "", *, ctx: Annotated[ToolContext, InjectedToolArg]
) -> str:
    """Read one matter chunk's exact source text (and its neighbours by name).

    Called with no `filename`, it returns the chunk this graph is localizing.
    Pass a filename from `list_matter_sections`/the prompt to read an adjacent
    chunk for continuity. The result includes the chunk's chapter and index.
    """
    report = _chunk_text(ctx, filename)
    if not report:
        raise ToolError("this graph has no current chunk")
    report["note"] = (
        "Exact source text of one matter chunk. Do not summarize it; localize all "
        "of it, except text the book plan says to omit."
    )
    return jdump(report)


@matter_stage
@tool
def list_book_chapters(*, ctx: Annotated[ToolContext, InjectedToolArg]) -> str:
    """The body chapters' titles, for localizing a printed table of contents.

    Returns `[{filename, title, chapter_index, start_line}]` in reading order.
    Only titles and positions: never chapter prose.
    """
    book = _book(ctx)
    titles = body_chapter_titles(book.chapters_path) or list(book.body_chapters)
    return jdump({"chapters_path": book.chapters_path, "chapters": titles})


# --------------------------------------------------------------------------- #
# plan family
# --------------------------------------------------------------------------- #
@matter_stage
@tool
def validate_matter_plan_tool(
    plan: dict[str, Any], *, ctx: Annotated[ToolContext, InjectedToolArg]
) -> str:
    """Dry-run the book-level localization plan and return every issue.

    Use it before `report_matter_plan`. The plan has these fields:

    * `original_title`, `author` — read from the front matter; strings.
    * `translated_title` — **required**; the exact title of the translated
      edition. Reuse a published rendering found with `web_search`, otherwise
      translate the title appropriately.
    * `title_source` — `published` / `web` / `translated` / `working`.
    * `title_evidence` — the URL or citation that proves a published title.
    * `translation_credit` — a sentence in the target language naming `BleBel`
      (e.g. `Prevod: BleBel`). Required in effect: the front matter must say the
      book is translated by BleBel.
    * `edition_note` — optional localized sentence such as the source language.
    * `omit` — exact source phrases that must **not** appear in the localized
      book (original imprint, ISBN, CIP, printing history, original publisher's
      copyright/permissions, references to the print edition).
    * `sections` — optional per-file overrides:
      `{"01-predgovor.md": {"title": "Predgovor", "omit": [...]}}`.

    Returns `{"ok", "issues", "warnings", "plan"}`. Fix every issue.
    """
    book = _book(ctx)
    validation = validate_matter_plan_fn(
        book, plan, target_language=book.target_language
    )
    return jdump(validation)


@matter_stage
@tool
def report_matter_plan(
    plan: dict[str, Any], *, ctx: Annotated[ToolContext, InjectedToolArg]
) -> str:
    """Submit the book-level localization plan. This ends the planning phase.

    The plan is validated first. If it is accepted, the graph carries it to every
    matter chunk; if not, the issues are returned and you must correct and call
    this tool again. Do not write a plan as prose in the conversation.
    """
    book = _book(ctx)
    validation = validate_matter_plan_fn(
        book, plan, target_language=book.target_language
    )
    if not validation.get("ok"):
        return jdump({"accepted": False, "validation": validation})
    return jdump(
        {
            "accepted": True,
            "plan": validation["plan"],
            "validation": validation,
            "note": "The plan is now binding for every front/back matter chunk.",
        }
    )


# --------------------------------------------------------------------------- #
# per-chunk translation family
# --------------------------------------------------------------------------- #
@matter_stage
@tool
def validate_matter_localization(
    translation: str = "",
    omitted: bool = False,
    omit_reason: str = "",
    *,
    ctx: Annotated[ToolContext, InjectedToolArg],
) -> str:
    """Dry-run one matter chunk's localization before submitting it.

    `translation` is the complete localized chunk in Markdown. If the book plan
    says this chunk is original-edition material that should not be in the
    translated book, set `omitted: true` and explain why in `omit_reason`
    instead of inventing text. The first front-matter chunk is special: the
    graph adds the localized title and the `BleBel` credit if you did not, and
    then checks they are present.

    Returns `{"ok", "issues", "warnings", "text"}` where `text` is exactly what
    would be written.
    """
    section = _section(ctx)
    chunk = ctx.chunk
    if chunk is None:
        raise ToolError("this graph has no current chunk")
    source = (
        str(chunk.get("text") or "")
        if isinstance(chunk, dict)
        else str(getattr(chunk, "text", "") or "")
    )
    validation = validate_matter_translation_fn(
        source_text=source,
        section_kind=section.kind,
        section_filename=section.filename,
        section_title=section.title,
        plan=dict(ctx.plan or {}),
        translation=translation,
        omitted=bool(omitted),
        omit_reason=omit_reason,
        is_first_front_chunk=bool(ctx.is_first_front_chunk),
        is_first_section_chunk=bool(ctx.is_first_section_chunk),
    )
    return jdump(validation)


@matter_stage
@tool
def report_matter_localization(
    translation: str = "",
    omitted: bool = False,
    omit_reason: str = "",
    notes: str = "",
    title_used: str = "",
    *,
    ctx: Annotated[ToolContext, InjectedToolArg],
) -> str:
    """Submit one front/back matter chunk's finished localization.

    This is the only way to finish a chunk; do not stop with the translation in
    the conversation. `translation` is the **complete** Markdown localization of
    the source chunk read by `read_matter_chunk`, not a summary. Set
    `omitted: true` with a reason when the plan says the original material does
    not belong in the translated edition.

    Returns `{"accepted": true, "text": "...", ...}` on success, or
    `{"accepted": false, "validation": {"issues": [...]}}`. On rejection, fix the
    named issues and call this tool again.
    """
    section = _section(ctx)
    chunk = ctx.chunk
    if chunk is None:
        raise ToolError("this graph has no current chunk")
    source = (
        str(chunk.get("text") or "")
        if isinstance(chunk, dict)
        else str(getattr(chunk, "text", "") or "")
    )
    validation = validate_matter_translation_fn(
        source_text=source,
        section_kind=section.kind,
        section_filename=section.filename,
        section_title=section.title,
        plan=dict(ctx.plan or {}),
        translation=translation,
        omitted=bool(omitted),
        omit_reason=omit_reason,
        is_first_front_chunk=bool(ctx.is_first_front_chunk),
        is_first_section_chunk=bool(ctx.is_first_section_chunk),
    )
    if not validation.get("ok"):
        return jdump({"accepted": False, "validation": validation})
    return jdump(
        {
            "accepted": True,
            "filename": getattr(chunk, "filename", chunk.get("filename") if isinstance(chunk, dict) else ""),
            "section": section.filename,
            "kind": section.kind,
            "omitted": bool(omitted),
            "omit_reason": omit_reason,
            "notes": notes,
            "title_used": title_used
            or str((ctx.plan or {}).get("translated_title") or ""),
            "text": validation["text"],
            "warnings": validation.get("warnings") or [],
        }
    )


#: Tools the book-plan node may call.
BOOK_TOOL_NAMES: tuple[str, ...] = (
    "list_matter_sections",
    "read_matter_section",
    "list_book_chapters",
    "extract_markdown_headers",
    "list_chapter_files",
    "web_search",
    "fetch_web_page",
    "validate_matter_plan_tool",
    "report_matter_plan",
)

#: Tools one matter-chunk graph may call. It reuses the splitter's web tools and
#: the shared structural tools instead of duplicating them.
CHUNK_TOOL_NAMES: tuple[str, ...] = (
    "list_matter_sections",
    "read_matter_section",
    "read_matter_chunk",
    "list_book_chapters",
    "list_chapter_files",
    "extract_markdown_headers",
    "inspect_line_window",
    "web_search",
    "fetch_web_page",
    "validate_matter_localization",
    "report_matter_localization",
)


__all__ = [
    "BOOK_TOOL_NAMES",
    "CHUNK_TOOL_NAMES",
    "MATTER_TRANSLATOR_NAME",
    "DEFAULT_MATTER_CREDIT",
    "list_book_chapters",
    "list_matter_sections",
    "read_matter_chunk",
    "read_matter_section",
    "report_matter_localization",
    "report_matter_plan",
    "validate_matter_localization",
    "validate_matter_plan_tool",
]
