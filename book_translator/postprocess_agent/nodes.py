"""The stage-9 graphs' own nodes.

Book graph: ``survey`` -> (shared agent/tools) -> ``book_validate``.
Chapter graph: ``collect`` -> (shared agent/tools) -> ``chapter_validate`` ->
``apply`` -> ``verify``.

The shared ``agent``, ``tools`` and ``give_up`` bodies come from
:mod:`book_translator.stage_graph.nodes`. What is left here is what differs: the
two deterministic recon nodes, the two validators, and the write tail.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..documents.chapters import postprocessed_chapters_dir
from ..postprocess.inspect import scan_book_structure, scan_chapter_structure
from ..postprocess.plan import (
    apply_chapter_plan,
    validate_book_plan,
    validate_chapter_plan,
    verify_chapter_plan,
)
from ..toolkit.context import ToolContext
from ..toolkit.postprocess import apply_chapter_postprocess
from .prompts import (
    BOOK_SYSTEM_PROMPT,
    CHAPTER_SYSTEM_PROMPT,
    book_validation_feedback,
    build_book_report,
    build_chapter_report,
    chapter_validation_feedback,
)
from .state import BookPlanState, ChapterPostprocessState


# --------------------------------------------------------------------------- #
# book graph
# --------------------------------------------------------------------------- #
def make_survey_node():
    """Build ``survey``: the deterministic book inventory, no LLM."""

    def survey(state: BookPlanState) -> dict[str, Any]:
        book = scan_book_structure(state["chapters_path"])
        report = build_book_report(book)
        task = (
            f"Plan the heading postprocessing for the translated book at "
            f"{state['chapters_path']}.\n"
            "Read the whole chapter-title list first, search the web for the published "
            "chapter list (mandatory before the plan), then choose one naming policy and "
            "a table-of-contents action. Do not read chapter prose."
        )
        return {
            "book": book,
            "attempts": 0,
            "messages": [
                SystemMessage(content=BOOK_SYSTEM_PROMPT),
                HumanMessage(content=f"{task}\n\n{report}"),
            ],
            "log": [
                f"survey: {book.get('chapter_count', 0)} files, "
                f"{book.get('body_chapter_count', 0)} body chapters, "
                f"toc={'yes' if (book.get('toc') or {}).get('exists') else 'no'}",
            ],
        }

    return survey


def make_book_validate_node():
    """Build ``book_validate``: deterministic validation of the naming/TOC plan."""

    def book_validate(state: BookPlanState) -> dict[str, Any]:
        plan = dict(state.get("book_plan") or {})
        validation = validate_book_plan(
            state["chapters_path"],
            plan,
            book=state.get("book") or None,
            require_web_search=bool(state.get("require_web_search", True)),
        )
        feedback: list[str] = []
        if not validation.get("ok"):
            feedback.append(book_validation_feedback(validation))
        return {
            "validation": validation,
            "feedback": feedback,
            "log": [
                f"book_validate: ok={validation.get('ok')} "
                f"issues={len(validation.get('issues') or [])} "
                f"warnings={len(validation.get('warnings') or [])}"
            ],
        }

    return book_validate


# --------------------------------------------------------------------------- #
# chapter graph
# --------------------------------------------------------------------------- #
def make_collect_node():
    """Build ``collect``: the chapter's title/heading metadata."""

    def collect(state: ChapterPostprocessState) -> dict[str, Any]:
        structure = scan_chapter_structure(state["chapter_path"])
        canonical = str(state.get("canonical_title") or structure.title)
        report = build_chapter_report(structure, canonical)
        task = (
            f"Postprocess the headings of {structure.filename}.\n"
            f"chapter_path: {structure.path}\n"
            f"The book plan fixes its chapter title as {canonical!r}. "
            "Set/insert/keep that title, then decide keep/demote/promote for the other "
            "headings (a plain ALL-CAPS section heading is promoted). "
            "Pass the exact chapter_path above to every chapter tool; never a bare filename. "
            "Never read the chapter's prose."
        )
        return {
            "line_count": structure.line_count,
            "structure": structure.to_dict(),
            "attempts": 0,
            "messages": [
                SystemMessage(content=CHAPTER_SYSTEM_PROMPT),
                HumanMessage(content=f"{task}\n\n{report}"),
            ],
            "log": [
                f"collect: {structure.heading_count} headings, "
                f"{len(structure.suspicious_headings)} suspicious, "
                f"title={'missing' if structure.title_line is None else structure.title_line}"
            ],
        }

    return collect


def make_chapter_validate_node():
    """Build ``chapter_validate``: check the chapter plan against the real file."""

    def chapter_validate(state: ChapterPostprocessState) -> dict[str, Any]:
        validation = validate_chapter_plan(
            state["chapter_path"],
            dict(state.get("chapter_plan") or {}),
            canonical_title=str(state.get("canonical_title") or ""),
        )
        feedback: list[str] = []
        if not validation.get("ok"):
            feedback.append(chapter_validation_feedback(validation))
        return {
            "validation": validation,
            "feedback": feedback,
            "log": [
                f"chapter_validate: ok={validation.get('ok')} "
                f"issues={len(validation.get('issues') or [])} "
                f"warnings={len(validation.get('warnings') or [])}"
            ],
        }

    return chapter_validate


def make_apply_node():
    """Build ``apply``: write the chapter, or validate it in a dry run."""

    def apply(state: ChapterPostprocessState) -> dict[str, Any]:
        chapter = state["chapter_path"]
        plan = dict(state.get("chapter_plan") or {})
        canonical = str(state.get("canonical_title") or "")
        if state.get("dry_run"):
            applied = apply_chapter_plan(chapter, plan, canonical_title=canonical)
            verification = verify_chapter_plan(
                chapter, plan, canonical_title=canonical
            )
            payload: dict[str, Any] = {
                "ok": bool(verification.get("ok")),
                "dry_run": True,
                "output_path": None,
                "output_lines": len(applied.lines),
                "applied": applied.summary(),
                "verification": verification,
            }
            return {
                "applied": payload,
                "log": ["apply: DRY RUN, nothing written"],
            }

        out_dir = state.get("output_dir") or str(postprocessed_chapters_dir(chapter))
        try:
            raw = apply_chapter_postprocess.invoke(
                {
                    "chapter_path": chapter,
                    "plan": plan,
                    "canonical_title": canonical,
                    "output_dir": out_dir,
                    "ctx": ToolContext.from_state(state),
                }
            )
            payload = json.loads(raw)
        except Exception as exc:
            payload = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "verification": {"ok": False, "problems": [str(exc)]},
            }
        return {
            "applied": payload,
            "log": [
                f"apply: {'wrote' if payload.get('ok') else 'FAILED'} "
                f"{Path(str(payload.get('output_path') or '')).name or 'chapter'}"
            ],
        }

    return apply


def make_verify_node():
    """Build ``verify``: prove the written chapter only changed declared lines."""

    def verify(state: ChapterPostprocessState) -> dict[str, Any]:
        applied = state.get("applied") or {}
        verification = dict(applied.get("verification") or {})
        verification["ok"] = bool(applied.get("ok")) and bool(verification.get("ok"))
        return {
            "verification": verification,
            "log": [f"verify: ok={verification.get('ok')}"],
        }

    return verify


__all__ = [
    "make_apply_node",
    "make_book_validate_node",
    "make_chapter_validate_node",
    "make_collect_node",
    "make_survey_node",
    "make_verify_node",
]
