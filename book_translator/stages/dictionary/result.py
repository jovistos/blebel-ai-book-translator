"""The dictionary result value and its one builder."""

from __future__ import annotations

from typing import Any

from book_translator.core.store import ArtifactStore

from .models import ChapterReduction, DictionaryResult


def build_dictionary_result(
    *,
    records: list[ChapterReduction],
    chapters_path: str,
    output_dir: str,
    unique_dir: str,
    language: dict[str, Any],
    backend: str,
    gpu: int | None,
    device: str,
    settings: dict[str, Any],
    totals: dict[str, int],
    unique_totals: dict[str, int],
    entities: dict[str, int],
    timing: dict[str, float],
    errors: list[str],
    dry_run: bool,
    limited: bool,
    save_reduced: bool,
    save_unique: bool,
    skipped: bool = False,
) -> DictionaryResult:
    """The run as one value; the fully-reused run sets ``skipped``."""
    return DictionaryResult(
        ok=True if skipped else bool(records) and all(record.ok for record in records),
        chapters_path=chapters_path,
        output_dir=output_dir,
        unique_dir=unique_dir,
        language=language,
        backend=backend,
        chapter_count=len(records),
        ok_count=sum(1 for record in records if record.ok),
        processed_count=sum(1 for record in records if not record.skipped),
        skipped_count=sum(1 for record in records if record.skipped),
        skipped=skipped,
        dry_run=dry_run,
        limited=limited,
        save_reduced=save_reduced,
        save_unique=save_unique,
        gpu=gpu,
        device=device,
        settings=settings,
        totals=totals,
        unique_totals=unique_totals,
        entities=entities,
        timing=timing,
        chapters=[record.record() for record in records],
        errors=errors,
    )


__all__ = ["build_dictionary_result"]


def write_dictionary_result(
    store: ArtifactStore, result: DictionaryResult, chapters: Any, records: Any
) -> DictionaryResult:
    if not store.enabled:
        return result
    from .artifacts import reduced_manifest_payload

    path = store.write_manifest(reduced_manifest_payload(result, chapters, records))
    if path is not None:
        result.manifest_path = str(path)
    trace = result.to_dict()
    trace["detail"] = [record.to_dict() for record in records]
    store.write_trace(trace)
    return store.attach(result)
