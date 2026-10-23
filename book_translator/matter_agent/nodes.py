"""The matter graphs' own nodes.

Book graph: ``survey`` -> (shared agent/tools) -> ``book_validate``.

Chunk graph: ``collect`` -> (shared agent/tools) -> ``chunk_validate`` ->
``apply`` -> ``verify``.

The shared ``agent``, ``tools`` and ``give_up`` bodies come from
:mod:`book_translator.stage_graph.nodes`. The apply node is the only writer, and it
writes exactly the text the validator approved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..core.artifacts import write_text
from ..matter.plan import (
    preserve_trailing_newlines,
)
from ..matter.plan import (
    validate_matter_plan as validate_matter_plan_fn,
)
from ..matter.plan import (
    validate_matter_translation as validate_matter_translation_fn,
)
from .prompts import (
    BOOK_SYSTEM_PROMPT,
    CHUNK_SYSTEM_PROMPT,
    book_validation_feedback,
    build_book_report,
    build_chunk_report,
    chunk_validation_feedback,
)
from .state import MatterChunkState, MatterPlanState


# --------------------------------------------------------------------------- #
# book graph
# --------------------------------------------------------------------------- #
def make_survey_node():
    """Build ``survey``: deterministic matter inventory, then the planning task."""

    def survey(state: MatterPlanState) -> dict[str, Any]:
        book = state.get("book")
        if book is None:
            raise ValueError("the matter graph needs the book inventory in its state")
        report = build_book_report(book)
        task = (
            f"Plan the localization of the front/back matter of {book.book_title!r}"
            + (f" by {book.book_author}" if book.book_author else "")
            + f" ({book.source_language} -> {book.target_language}).\n"
            "Read the matter files, search for an existing published translation of the "
            "title, and submit one binding plan. The front matter must present the book "
            "as a translation and credit BleBel."
        )
        return {
            "attempts": 0,
            "messages": [
                SystemMessage(content=BOOK_SYSTEM_PROMPT),
                HumanMessage(content=f"{task}\n\n{report}"),
            ],
            "log": [
                f"survey: {len(book.sections)} matter section(s), "
                f"{book.chunk_count} matter chunk(s), {len(book.body_chapters)} body chapter(s)",
            ],
        }

    return survey


def make_book_validate_node():
    """Build ``book_validate``: deterministic plan validation."""

    def book_validate(state: MatterPlanState) -> dict[str, Any]:
        book = state.get("book")
        plan = dict(state.get("matter_plan") or {})
        validation = validate_matter_plan_fn(
            book,
            plan,
            target_language=str(getattr(book, "target_language", "") or ""),
        )
        feedback: list[str] = []
        if not validation.get("ok"):
            feedback.append(book_validation_feedback(validation))
        return {
            "validation": validation,
            "matter_plan": validation.get("plan") or plan,
            "feedback": feedback,
            "log": [
                f"book_validate: ok={validation.get('ok')} "
                f"issues={len(validation.get('issues') or [])} "
                f"warnings={len(validation.get('warnings') or [])}"
            ],
        }

    return book_validate


# --------------------------------------------------------------------------- #
# chunk graph
# --------------------------------------------------------------------------- #
def make_collect_node():
    """Build ``collect``: the section metadata and the exact chunk text."""

    def collect(state: MatterChunkState) -> dict[str, Any]:
        section = state.get("section")
        chunk = state.get("chunk")
        if section is None or chunk is None:
            raise ValueError("the matter chunk graph needs a section and a chunk")
        report = build_chunk_report(
            section=section,
            chunk=chunk,
            plan=dict(state.get("plan") or {}),
            source_language=str(state.get("source_language") or ""),
            target_language=str(state.get("target_language") or ""),
            is_first_front_chunk=bool(state.get("is_first_front_chunk")),
            is_first_section_chunk=bool(state.get("is_first_section_chunk")),
        )
        task = (
            f"Localize {getattr(chunk, 'filename', '')} from "
            f"{section.filename} ({section.kind} matter).\n"
            "Return the complete localized chunk through report_matter_localization."
        )
        return {
            "attempts": 0,
            "messages": [
                SystemMessage(content=CHUNK_SYSTEM_PROMPT),
                HumanMessage(content=f"{task}\n\n{report}"),
            ],
            "log": [
                f"collect: {getattr(chunk, 'filename', '')} "
                f"({len(getattr(chunk, 'text', '') or '')} chars), "
                f"first_front={bool(state.get('is_first_front_chunk'))}"
            ],
        }

    return collect


def make_chunk_validate_node():
    """Build ``chunk_validate``: check the localization against the real chunk."""

    def chunk_validate(state: MatterChunkState) -> dict[str, Any]:
        section = state.get("section")
        chunk = state.get("chunk")
        plan = dict(state.get("plan") or {})
        submitted = dict(state.get("chunk_plan") or {})
        validation = validate_matter_translation_fn(
            source_text=str(getattr(chunk, "text", "") or ""),
            section_kind=str(getattr(section, "kind", "front")),
            section_filename=str(getattr(section, "filename", "")),
            section_title=str(getattr(section, "title", "")),
            plan=plan,
            translation=str(submitted.get("translation") or submitted.get("text") or ""),
            omitted=bool(submitted.get("omitted")),
            omit_reason=str(submitted.get("omit_reason") or ""),
            is_first_front_chunk=bool(state.get("is_first_front_chunk")),
            is_first_section_chunk=bool(state.get("is_first_section_chunk")),
        )
        feedback: list[str] = []
        if not validation.get("ok"):
            feedback.append(chunk_validation_feedback(validation))
        return {
            "validation": validation,
            "translation": str(validation.get("text") or ""),
            "feedback": feedback,
            "log": [
                f"chunk_validate: ok={validation.get('ok')} "
                f"omitted={bool(submitted.get('omitted'))} "
                f"chars={len(validation.get('text') or '')}"
            ],
        }

    return chunk_validate


def make_apply_node():
    """Build ``apply``: write the validated localization, or nothing on a dry run."""

    def apply(state: MatterChunkState) -> dict[str, Any]:
        chunk = state.get("chunk")
        text = str(state.get("translation") or "")
        submitted = dict(state.get("chunk_plan") or {})
        omitted = bool(submitted.get("omitted"))
        source = str(getattr(chunk, "text", "") or "")
        # An omitted chunk is empty — except the first front-matter chunk, where
        # the validator has already put the localized title and the BleBel credit
        # into `translation`. Never throw that title page away because the source
        # imprint inside it was omitted.
        final = preserve_trailing_newlines(text, source) if text else ""

        if state.get("dry_run") or state.get("save") is False:
            return {
                "applied": {
                    "ok": True,
                    "dry_run": True,
                    "output_path": None,
                    "output_chars": len(final),
                    "omitted": omitted,
                },
                "log": ["apply: DRY RUN, nothing written"],
            }
        out_dir = Path(str(state.get("output_dir") or "")).expanduser()
        target = out_dir / str(getattr(chunk, "filename", "chunk.md"))
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            write_text(target, final)
            payload = {
                "ok": True,
                "dry_run": False,
                "output_path": str(target),
                "output_chars": len(final),
                "omitted": omitted,
            }
        except OSError as exc:
            payload = {
                "ok": False,
                "dry_run": False,
                "output_path": str(target),
                "error": f"{type(exc).__name__}: {exc}",
            }
        return {
            "applied": payload,
            "log": [
                f"apply: {'wrote' if payload.get('ok') else 'FAILED'} {target.name} "
                f"({len(final)} chars)"
            ],
        }

    return apply


def make_verify_node():
    """Build ``verify``: prove the file holds exactly the validated text."""

    def verify(state: MatterChunkState) -> dict[str, Any]:
        applied = state.get("applied") or {}
        expected = str(state.get("translation") or "")
        source = str(getattr(state.get("chunk"), "text", "") or "")
        expected = preserve_trailing_newlines(expected, source) if expected else ""
        problems: list[str] = []
        if not applied.get("ok"):
            problems.append(str(applied.get("error") or "the write failed"))
        if not state.get("dry_run") and state.get("save") is not False:
            path = Path(str(applied.get("output_path") or ""))
            if not path.is_file():
                problems.append("the output file does not exist")
            else:
                try:
                    actual = path.read_text(encoding="utf-8")
                except OSError as exc:
                    problems.append(f"the output file cannot be read: {exc}")
                else:
                    if actual != expected:
                        problems.append(
                            f"the output file differs from the validated text "
                            f"({len(actual)} vs {len(expected)} characters)"
                        )
        verification = {
            "ok": not problems,
            "dry_run": bool(state.get("dry_run")),
            "problems": problems,
            "chars": len(expected),
        }
        return {
            "verification": verification,
            "log": [f"verify: ok={verification['ok']}"],
        }

    return verify


__all__ = [
    "make_apply_node",
    "make_book_validate_node",
    "make_chunk_validate_node",
    "make_collect_node",
    "make_survey_node",
    "make_verify_node",
]
