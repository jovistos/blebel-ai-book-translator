"""Previous-run records, totals and manifests for the dictionary stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from book_translator.core.artifacts import read_manifest, write_manifest, write_trace
from book_translator.core.constants import REDUCED_MANIFEST_FILENAME, REDUCED_TRACE_FILENAME
from book_translator.documents.chapters import Chapter, file_sha256
from book_translator.documents.io import load_lines
from book_translator.documents.manifest import load_split_manifest

from .models import ChapterReduction, DictionaryResult


def previous_manifest(out_dir: Path) -> dict[str, Any]:
    """The manifest of an earlier stage-3 run in this folder, or ``{}``."""
    return read_manifest(out_dir, REDUCED_MANIFEST_FILENAME)


def manifest_files(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The previous manifest's per-chapter entries, keyed by filename."""
    entries: dict[str, dict[str, Any]] = {}
    for entry in manifest.get("files") or []:
        if isinstance(entry, dict) and entry.get("filename"):
            entries[str(entry["filename"])] = entry
    return entries


def reusable(
    chapter: Chapter,
    out_dir: Path,
    unique_out_dir: Path,
    previous_files: dict[str, dict[str, Any]],
    *,
    save_unique: bool,
) -> bool:
    """True when a rerun may keep this chapter's outputs instead of redoing them.

    The reduced file is enough when the first-mention step is off. With it on, the
    unique file must exist *and* the previous manifest must carry the chapter's
    noun index: without that index a later chapter could not tell which nouns were
    already met, so the chapter has to be tagged again rather than silently skew
    the book-wide set.
    """
    if not (out_dir / chapter.filename).is_file():
        return False
    if not save_unique:
        return True
    if not (unique_out_dir / chapter.filename).is_file():
        return False
    entry = previous_files.get(chapter.filename)
    return entry is not None and isinstance(entry.get("noun_keys"), list)


def skipped_record(
    chapter: Chapter,
    out_dir: Path,
    unique_out_dir: Path | None = None,
    previous: dict[str, Any] | None = None,
) -> ChapterReduction:
    """A chapter that was reused instead of reduced, described from what is on disk.

    ``previous`` is the chapter's own entry in the manifest of the run that wrote
    these files. Its ``noun_keys`` matter: they are how a resumed run rebuilds the
    book-wide noun set without tagging the chapter again.
    """
    path = Path(out_dir) / chapter.filename
    record = ChapterReduction(
        chapter_id=chapter.chapter_id,
        filename=chapter.filename,
        source_path=chapter.path,
        title=chapter.title,
        start_line=chapter.start_line,
        end_line=chapter.end_line,
        skipped=True,
    )
    if path.is_file():
        record.output_path = str(path)
        record.output_lines = len(load_lines(path)) if path.stat().st_size else 0
        record.output_bytes = path.stat().st_size
        record.sha256 = file_sha256(path)
    if unique_out_dir is not None:
        unique_path = Path(unique_out_dir) / chapter.filename
        if unique_path.is_file():
            record.unique_path = str(unique_path)
            record.unique_lines = (
                len(load_lines(unique_path)) if unique_path.stat().st_size else 0
            )
            record.unique_bytes = unique_path.stat().st_size
            record.unique_sha256 = file_sha256(unique_path)
    if previous:
        keys = previous.get("noun_keys")
        record.noun_keys = [str(key) for key in keys] if isinstance(keys, list) else []
    record.log.append("reused: the reduced chapter already exists")
    return record


def totals(records: list[ChapterReduction]) -> dict[str, int]:
    reductions = [r.reduction for r in records if r.reduction is not None]
    return {
        "chapters_processed": len(reductions),
        "sentences": sum(r.source_sentences for r in reductions),
        "kept": sum(len(r.kept) for r in reductions),
        "dropped_short": sum(r.dropped_short for r in reductions),
        "dropped_no_uppercase": sum(r.dropped_no_uppercase for r in reductions),
        "dropped_no_entity": sum(r.dropped_no_entity for r in reductions),
        "dropped_temporal": sum(r.dropped_temporal for r in reductions),
        # How many sentences actually reached the tagger (the NER stage's input).
        "sentences_tagged": sum(r.ner_calls for r in records),
        "output_lines": sum(r.output_lines for r in records),
        "output_bytes": sum(r.output_bytes for r in records),
    }


def unique_totals(records: list[ChapterReduction]) -> dict[str, int]:
    """The first-mention step's own totals, over the chapters visited this run."""
    selections = [r.unique for r in records if r.unique is not None]
    if not selections:
        # The step was switched off (or every chapter was reused): report zeros
        # rather than a "sentences in, nothing out" shape that reads like a bug.
        return {
            "chapters": 0,
            "sentences": 0,
            "kept": 0,
            "dropped_duplicate_noun": 0,
            "new_nouns": 0,
            "nouns": 0,
            "output_lines": 0,
            "output_bytes": 0,
        }
    return {
        "chapters": len(selections),
        "sentences": sum(
            len(r.reduction.kept) for r in records if r.reduction is not None
        ),
        "kept": sum(len(s.kept) for s in selections),
        "dropped_duplicate_noun": sum(s.dropped_duplicate for s in selections),
        "new_nouns": sum(s.new_nouns for s in selections),
        "nouns": len({key for record in records for key in record.noun_keys}),
        "output_lines": sum(r.unique_lines for r in records),
        "output_bytes": sum(r.unique_bytes for r in records),
    }


def timing(
    records: list[ChapterReduction],
    *,
    backend_load_seconds: float,
    run_seconds: float,
) -> dict[str, float]:
    """The run's timings, with the NER stage broken out."""
    ner_seconds = sum(r.ner_seconds for r in records)
    chapter_seconds = sum(r.elapsed_seconds for r in records)
    tagged = sum(r.ner_calls for r in records)
    return {
        "backend_load_seconds": round(float(backend_load_seconds), 4),
        "ner_seconds": round(float(ner_seconds), 4),
        "chapter_seconds": round(float(chapter_seconds), 4),
        "run_seconds": round(float(run_seconds), 4),
        "seconds_per_sentence": round(ner_seconds / tagged, 6) if tagged else 0.0,
        "milliseconds_per_sentence": (
            round(1000.0 * ner_seconds / tagged, 3) if tagged else 0.0
        ),
        "sentences_per_second": round(tagged / ner_seconds, 3) if ner_seconds > 0 else 0.0,
    }


def entity_totals(records: list[ChapterReduction]) -> dict[str, int]:
    totals_by_type: dict[str, int] = {}
    for record in records:
        if record.reduction is None:
            continue
        for entity_type, count in record.reduction.entities.items():
            totals_by_type[entity_type] = totals_by_type.get(entity_type, 0) + int(count)
    return dict(sorted(totals_by_type.items()))


def reduced_manifest_payload(
    result: DictionaryResult,
    chapters: list[Chapter],
    records: list[ChapterReduction] | None = None,
) -> dict[str, Any]:
    if records is None:
        records = [
            ChapterReduction(
                chapter_id=str(c.get("chapter_id") or c.get("filename") or ""),
                filename=str(c.get("filename") or ""),
                source_path=str(c.get("source_path") or ""),
            )
            for c in result.chapters
        ]
    stage_one = load_split_manifest(result.chapters_path)
    return {
        "stage": "dictionary",
        "chapters_path": result.chapters_path,
        "source_file": stage_one.get("source_file", ""),
        "ok": result.ok,
        "dry_run": result.dry_run,
        "limited": result.limited,
        "language": result.language,
        "backend": result.backend,
        "gpu": result.gpu,
        "device": result.device,
        "settings": result.settings,
        "chapter_count": result.chapter_count,
        "processed_count": result.processed_count,
        "skipped_count": result.skipped_count,
        "save_reduced": result.save_reduced,
        "save_unique": result.save_unique,
        "unique_dir": result.unique_dir,
        "totals": result.totals,
        "unique_totals": result.unique_totals,
        "timing": result.timing,
        "entities": result.entities,
        "files": [
            {
                **record.record(),
                "lines": record.output_lines,
                "source_lines": _source_lines(chapters, record),
            }
            for record in records
        ],
        "errors": result.errors,
    }


def write_reduced_manifest(
    out_dir: Path,
    result: DictionaryResult,
    chapters: list[Chapter],
    records: list[ChapterReduction] | None = None,
) -> Path:
    """Write the book-level stage-3 manifest next to the reduced chapters."""
    return write_manifest(
        out_dir, REDUCED_MANIFEST_FILENAME, reduced_manifest_payload(result, chapters, records)
    )


def _source_lines(chapters: list[Chapter], record: ChapterReduction) -> int:
    for chapter in chapters:
        if chapter.filename == record.filename:
            return chapter.line_count
    return 0


def write_reduced_trace(
    out_dir: Path,
    result: DictionaryResult,
    records: list[ChapterReduction] | None = None,
) -> Path:
    """Write the audit trail: per chapter, what was dropped and why."""
    payload = result.to_dict()
    if records is not None:
        payload["detail"] = [record.to_dict() for record in records]
    return write_trace(out_dir, REDUCED_TRACE_FILENAME, payload)


__all__ = [
    "entity_totals",
    "manifest_files",
    "previous_manifest",
    "reduced_manifest_payload",
    "reusable",
    "skipped_record",
    "timing",
    "totals",
    "unique_totals",
    "write_reduced_manifest",
    "write_reduced_trace",
]
