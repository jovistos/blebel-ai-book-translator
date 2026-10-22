"""The matter result value and its one builder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import MatterChunkResult, MatterResult


def build_matter_result(
    *,
    book,
    chunks: Path,
    out_dir: Path,
    prompt_dir: Path,
    stage_dir: Path,
    source_lang: str,
    target_lang: str,
    synthesis: Any,
    sections: list[Any],
    records: list[MatterChunkResult],
    errors: list[str],
    pending: list[Any],
    plan: dict[str, Any],
    validation: dict[str, Any],
    settings: Any,
    usage_tokens: dict[str, int],
    limited: bool,
    dry_run: bool,
    save: bool,
    resume_enabled: bool,
    max_attempts: int,
    recursion_limit: int,
    limit: int | None,
    chapter_files: Any,
    resume: bool,
    write_prompts: bool,
    run_seconds: float,
) -> MatterResult:
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
        chunks_path=str(chunks),
        output_dir=str(out_dir),
        prompts_dir=str(prompt_dir) if write_prompts else "",
        matter_path=str(stage_dir),
        source_language=source_lang,
        target_language=target_lang,
        book_title=book.book_title,
        book_author=book.book_author,
        source_file=book.source_file,
        synthetic_sections=list(synthesis.created) if synthesis else [],
        synthetic_chunks=list(synthesis.chunk_created) if synthesis else [],
        section_count=len(sections),
        chunk_count=len(records),
        ok_count=sum(1 for record in records if record.ok),
        processed_count=len(pending),
        reused_count=sum(1 for record in records if record.reused),
        skipped_count=0,
        skipped=not records,
        dry_run=dry_run,
        save=save,
        resumed=resume_enabled,
        limited=limited,
        book_plan=plan,
        validation=validation,
        settings={
            "max_attempts": max(1, int(max_attempts)),
            "recursion_limit": int(recursion_limit),
            "limit": int(limit) if limit is not None else None,
            "chapter_files": list(chapter_files) if chapter_files is not None else None,
            "resume": bool(resume),
            "write_prompts": bool(write_prompts),
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


__all__ = ["build_matter_result"]


def finish_matter(
    plan_state,
    *,
    records,
    errors,
    usage_tokens,
    run_seconds: float,
    write_manifest: bool,
) -> MatterResult:
    from ..core.constants import MATTER_MANIFEST_FILENAME, MATTER_TRACE_FILENAME
    from ..core.store import ArtifactStore
    from .artifacts import matter_manifest_payload

    result = build_matter_result(
        book=plan_state.book,
        chunks=plan_state.chunks,
        out_dir=plan_state.out_dir,
        prompt_dir=plan_state.prompt_dir,
        stage_dir=plan_state.stage_dir,
        source_lang=plan_state.source_lang,
        target_lang=plan_state.target_lang,
        synthesis=plan_state.synthesis,
        sections=plan_state.sections,
        records=records,
        errors=errors,
        pending=plan_state.pending,
        plan=plan_state.plan,
        validation=plan_state.validation,
        settings=plan_state.settings,
        usage_tokens=usage_tokens,
        limited=plan_state.limited,
        dry_run=plan_state.dry_run,
        save=plan_state.save,
        resume_enabled=plan_state.resume_enabled,
        max_attempts=plan_state.max_attempts,
        recursion_limit=plan_state.recursion_limit,
        limit=plan_state.limit,
        chapter_files=plan_state.chapter_files,
        resume=plan_state.resume,
        write_prompts=plan_state.write_prompts,
        run_seconds=run_seconds,
    )
    store = ArtifactStore(
        plan_state.stage_dir,
        MATTER_MANIFEST_FILENAME,
        MATTER_TRACE_FILENAME,
        enabled=write_manifest and not plan_state.dry_run and plan_state.save,
    )
    return store.write_result(result, matter_manifest_payload(result))
