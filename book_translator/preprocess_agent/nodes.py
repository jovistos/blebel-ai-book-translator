"""The stage-2 graph's own nodes: recon, validation and the write tail.

The shared ``agent`` and ``tools`` bodies, and the ``give_up`` result, come from
:mod:`book_translator.stage_graph.nodes`. What is left here is what actually
differs from every other stage:

``collect``
    deterministic recon: the chapter's headers, their evidence and the damage
    report. No LLM, no writes, no prose.
``validate``
    deterministically check the plan against the real chapter.
``apply``
    write the cleaned chapter (only reachable after validation passed).
``verify``
    prove the output only differs in the declared ways.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..core.artifacts import write_text
from ..documents.chapters import preprocessed_dir
from ..preprocess.apply import apply_cleaning_plan, text_of
from ..preprocess.corpus import DAMAGE_PATTERNS
from ..preprocess.damage import damage_digest, scan_chapter_damage
from ..preprocess.inspect import chapter_digest, inspect_chapter
from ..preprocess.plan import validate_cleaning_plan, validate_text_fixes
from ..preprocess.verify import verify_cleaning_plan
from .prompts import SYSTEM_PROMPT, build_report
from .state import PreprocessState


def make_collect_node():
    """Build the ``collect`` node: header + damage recon for one chapter."""

    def collect(state: PreprocessState) -> dict[str, Any]:
        chapter = state["chapter_path"]
        header_report = inspect_chapter(chapter)
        damage_report = scan_chapter_damage(chapter)
        for name in DAMAGE_PATTERNS:
            damage_report["by_kind"].setdefault(name, 0)

        report = build_report(
            filename=header_report["chapter"]["filename"],
            line_count=header_report["chapter"]["line_count"],
            header_report=header_report,
            damage_report=damage_report,
            header_digest_text=chapter_digest(header_report),
            damage_digest_text=damage_digest(damage_report),
        )
        task = (
            f"Preprocess the chapter at {chapter}.\n"
            "Follow the protocol: collect the headers, decide on each one, validate, then report. "
            "Never read the chapter's prose."
        )
        return {
            "line_count": header_report["chapter"]["line_count"],
            "header_report": header_report,
            "damage_report": damage_report,
            "attempts": 0,
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"{task}\n\n{report}"),
            ],
            "log": [f"collect: {header_report['header_count']} headers, {damage_report['finding_count']} damage findings"],
        }

    return collect


def make_validate_node():
    """Build the ``validate`` node: check the plan and its text fixes."""

    def validate(state: PreprocessState) -> dict[str, Any]:
        chapter = state["chapter_path"]
        operations = list(state.get("operations") or [])
        text_fixes = list(state.get("text_fixes") or [])

        result = validate_cleaning_plan(chapter, operations)
        accepted_fixes, fix_issues, fix_warnings = validate_text_fixes(chapter, text_fixes)
        result["text_fixes"] = accepted_fixes
        result["issues"] = list(result.get("issues") or []) + fix_issues
        result["warnings"] = list(result.get("warnings") or []) + fix_warnings
        result["ok"] = bool(result.get("ok")) and not fix_issues and bool(operations)
        if not operations:
            result["issues"].append(
                "the plan is empty: submit at least one operation (every chapter has at "
                "least one header to consider)"
            )
            result["ok"] = False
        feedback: list[str] = []
        if not result["ok"]:
            feedback.append(
                "Your cleaning plan for this chapter was rejected. Fix these and report again:\n"
                + "\n".join(f"- {issue}" for issue in result["issues"])
            )
        return {
            "validation": result,
            "feedback": feedback,
            "log": [
                f"validate: ok={result['ok']} issues={len(result['issues'])} "
                f"warnings={len(result['warnings'])}"
            ],
        }

    return validate


def make_apply_node():
    """Build the ``apply`` node: write the cleaned chapter (deterministic)."""

    def apply(state: PreprocessState) -> dict[str, Any]:
        chapter = state["chapter_path"]
        operations = list(state.get("operations") or [])
        text_fixes = list(state.get("validation", {}).get("text_fixes") or [])
        if state.get("dry_run"):
            result = apply_cleaning_plan(chapter, operations, text_fixes)
            return {
                "applied": {
                    "ok": True,
                    "dry_run": True,
                    "output_lines": len(result.lines),
                    "provenance": result.summary(),
                },
                "log": ["apply: DRY RUN, nothing written"],
            }

        out_dir = state.get("output_dir") or str(preprocessed_dir(chapter))
        from pathlib import Path

        target_dir = Path(out_dir).expanduser()
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / Path(chapter).name

        result = apply_cleaning_plan(chapter, operations, text_fixes)
        write_text(target, text_of(result))
        verification = verify_cleaning_plan(chapter, operations, text_fixes)
        payload = {
            "ok": bool(verification.get("ok")),
            "chapter_path": chapter,
            "output_path": str(target),
            "output_lines": len(result.lines),
            "source_lines": verification.get("source_lines"),
            "removed_artefacts": result.removed_artefacts,
            "provenance": result.summary(),
            "verification": verification,
        }
        return {
            "applied": payload,
            "log": [f"apply: wrote {target.name} ({payload['output_lines']} lines)"],
        }

    return apply


def make_verify_node():
    """Build the ``verify`` node: re-check the written file's provenance."""

    def verify(state: PreprocessState) -> dict[str, Any]:
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
    "make_collect_node",
    "make_validate_node",
    "make_verify_node",
]
