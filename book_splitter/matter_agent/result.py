"""The matter result value and its one builder."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .models import MatterChunkResult, MatterResult

if TYPE_CHECKING:
    from .plan import MatterPlan


def build_matter_result(
    plan_state: MatterPlan,
    *,
    records: list[MatterChunkResult],
    errors: list[str],
    usage_tokens: dict[str, int],
    run_seconds: float,
) -> MatterResult:
    book = plan_state.book
    sections = plan_state.sections
    pending = plan_state.pending
    totals = {
        "sections": len(sections),
        "matter_chunks": len(pending),
        "ok": sum(1 for record in records if record.ok),
        "omitted": sum(1 for record in records if record.omitted),
        "reused": sum(1 for record in records if record.reused),
        "localized_chars": sum(record.translation_chars for record in records),
    }
    return MatterResult(
        ok=bool(records) and all(record.ok for record in records) and not errors,
        chapters_path=str(book.chapters_path),
        chunks_path=str(plan_state.chunks),
        output_dir=str(plan_state.out_dir),
        prompts_dir=str(plan_state.prompt_dir) if plan_state.write_prompts else "",
        matter_path=str(plan_state.stage_dir),
        source_language=plan_state.source_lang,
        target_language=plan_state.target_lang,
        book_title=book.book_title,
        book_author=book.book_author,
        source_file=book.source_file,
        synthetic_sections=list(plan_state.synthesis.created)
        if plan_state.synthesis
        else [],
        synthetic_chunks=list(plan_state.synthesis.chunk_created)
        if plan_state.synthesis
        else [],
        section_count=len(sections),
        chunk_count=len(records),
        ok_count=sum(1 for record in records if record.ok),
        processed_count=len(pending),
        reused_count=sum(1 for record in records if record.reused),
        skipped_count=0,
        skipped=not records,
        dry_run=plan_state.dry_run,
        save=plan_state.save,
        resumed=plan_state.resume_enabled,
        limited=plan_state.limited,
        book_plan=plan_state.plan,
        validation=plan_state.validation,
        settings={
            "max_attempts": max(1, int(plan_state.max_attempts)),
            "recursion_limit": int(plan_state.recursion_limit),
            "limit": int(plan_state.limit) if plan_state.limit is not None else None,
            "chapter_files": list(plan_state.chapter_files)
            if plan_state.chapter_files is not None
            else None,
            "resume": bool(plan_state.resume),
            "write_prompts": bool(plan_state.write_prompts),
        },
        tokens={**usage_tokens, "chunks": len(pending)},
        totals=totals,
        timing={
            "run_seconds": round(run_seconds, 4),
            "seconds_per_chunk": round(run_seconds / len(records), 4)
            if records
            else 0.0,
        },
        sections=[section.to_dict() for section in sections],
        chunks=[record.record() for record in records],
        errors=errors,
    )


def finish_matter(
    plan_state: MatterPlan,
    *,
    records: list[MatterChunkResult],
    errors: list[str],
    usage_tokens: dict[str, int],
    run_seconds: float,
    write_manifest: bool,
) -> MatterResult:
    from ..core.constants import MATTER_MANIFEST_FILENAME, MATTER_TRACE_FILENAME
    from ..core.store import ArtifactStore
    from .artifacts import matter_manifest_payload

    result = build_matter_result(
        plan_state,
        records=records,
        errors=errors,
        usage_tokens=usage_tokens,
        run_seconds=run_seconds,
    )
    store = ArtifactStore(
        plan_state.stage_dir,
        MATTER_MANIFEST_FILENAME,
        MATTER_TRACE_FILENAME,
        enabled=write_manifest and not plan_state.dry_run and plan_state.save,
    )
    return store.write_result(result, matter_manifest_payload(result))


__all__ = ["build_matter_result", "finish_matter"]
