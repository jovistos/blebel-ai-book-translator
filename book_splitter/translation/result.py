"""The translation result value and its one builder."""

from __future__ import annotations

from .loop import LoopOutcome
from .plan import TranslationPlan
from .records import TranslationResult
from .support import _timing, _totals


def build_translation_result(
    *,
    plan: TranslationPlan,
    run_seconds: float,
    outcome: LoopOutcome | None = None,
    empty: bool = False,
) -> TranslationResult:
    if empty:
        return TranslationResult(
            ok=True,
            chunks_path=str(plan.source_dir),
            chunk_glossary_path=str(plan.resolved_chunk_glossary),
            output_dir=str(plan.out_dir),
            prompts_dir=str(plan.prompt_dir),
            source_language=plan.source_language,
            target_language=plan.target_language,
            book_title=plan.title,
            source_file=plan.source_file,
            glossary_scope=plan.scope,
            chunk_count=0,
            ok_count=0,
            processed_count=0,
            skipped_count=plan.matter_skipped,
            dry_run=plan.dry_run,
            save=plan.save,
            resumed=False,
            limited=plan.limited,
            settings={
                "glossary_scope": plan.scope,
                "include_matter": plan.include_matter,
                "matter_skipped": plan.matter_skipped,
            },
            tokens={"calls": 0, "chunks": 0},
            totals={
                "translated_chunks": 0,
                "matter_skipped": plan.matter_skipped,
            },
            timing={"run_seconds": round(run_seconds, 4)},
            chunks=[],
            errors=[],
        )

    assert outcome is not None
    records = outcome.records
    return TranslationResult(
        ok=bool(records) and all(record.ok for record in records) and not outcome.errors,
        chunks_path=str(plan.source_dir),
        chunk_glossary_path=str(plan.resolved_chunk_glossary),
        output_dir=str(plan.out_dir),
        prompts_dir=str(plan.prompt_dir) if plan.write_prompts else "",
        source_language=plan.source_language,
        target_language=plan.target_language,
        book_title=plan.title,
        source_file=plan.source_file,
        translator=plan.translator_name,
        provider=str(getattr(plan.settings, "provider", "") or ""),
        model=str(getattr(plan.settings, "model", "") or ""),
        glossary_scope=plan.scope,
        glossary_entries=len(plan.book_rows) if plan.scope == "book" else 0,
        chunk_count=len(records),
        ok_count=sum(1 for record in records if record.ok),
        processed_count=len(plan.pending),
        skipped_count=plan.matter_skipped,
        reused_count=sum(1 for record in records if record.reused),
        dry_run=plan.dry_run,
        save=plan.save,
        resumed=plan.resumed,
        limited=plan.limited,
        settings=outcome.settings_payload,
        tokens={**outcome.totals.to_dict(), "chunks": len(plan.pending)},
        totals=_totals(records),
        timing=_timing(records, run_seconds=run_seconds),
        chunks=[record.record() for record in records],
        errors=outcome.errors,
    )


__all__ = ["build_translation_result"]
