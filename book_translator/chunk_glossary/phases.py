"""The two strict phases: NER over every chunk, then retrieval over every term."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from ..core.run import RunContext
from ..documents.chunks import ChunkInput
from ..documents.images import strip_images
from .artifacts import write_chunk_glossary
from .matching import TermMatcher
from .models import ChunkGlossary, EntityTerm
from .terms import aggregate_entities, collect_matched_terms, match_term


def run_phases(
    *,
    inputs: Sequence[ChunkInput],
    pending: Sequence[ChunkInput],
    reused_records: dict[str, ChunkGlossary],
    backend,
    matcher: TermMatcher,
    out_dir: Path,
    dry_run: bool,
    threshold: float,
    scorer: str,
    resolved_glossary: Path,
    glossary_sha: str,
    wanted_types: Sequence[str],
    minimum_chars: int,
    drop_temporal_only: bool,
    candidate_count: int,
    language: str,
    chosen_backend: str,
    context: RunContext,
    clock: Callable[[], float],
) -> tuple[list[ChunkGlossary], list[str], float, float]:
    records: list[ChunkGlossary] = []
    pending_records: dict[str, ChunkGlossary] = {}
    entity_terms: dict[str, list[EntityTerm]] = {}
    errors: list[str] = []

    # -- phase 1: NER over every chunk that still needs it ----------------- #
    ner_phase_started = clock()
    for chunk in pending:
        record = ChunkGlossary(chunk=chunk)
        tag_started = clock()
        try:
            ner_text = strip_images(chunk.text)
            if ner_text.strip():
                # Image paths are not named entities; the tagger only sees prose.
                entities = backend.entities(ner_text)
                record.ner_calls = 1
                terms, mentions = aggregate_entities(
                    entities,
                    keep_types=wanted_types,
                    min_term_chars=minimum_chars,
                    drop_temporal_only=drop_temporal_only,
                )
            else:
                terms, mentions = [], 0
            entity_terms[chunk.filename] = terms
            record.entity_count = mentions
            record.log.append(
                f"ner: {mentions} mention(s) -> {len(terms)} distinct term(s)"
            )
        except Exception as exc:  # one bad chunk must not stop the book
            record.ok = False
            record.error = f"{type(exc).__name__}: {exc}"
            entity_terms[chunk.filename] = []
            errors.append(f"{chunk.filename}: {record.error}")
            record.log.append(f"ner failed: {record.error}")
        record.ner_seconds = clock() - tag_started
        pending_records[chunk.filename] = record
    ner_phase_seconds = clock() - ner_phase_started

    # -- phase 2: retrieval for every term collected above ----------------- #
    match_phase_started = clock()
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    for completed, chunk in enumerate(inputs, start=1):
        record = reused_records.get(chunk.filename) or pending_records[chunk.filename]
        if record.skipped:
            # A resumed chunk already has its file; do not re-tag or re-match it.
            records.append(record)
            context.emit(record.to_dict(), completed, len(inputs))
            continue
        if record.ok:
            match_started = clock()
            try:
                terms = entity_terms.get(record.chunk.filename, [])
                record.terms = [
                    match_term(term, matcher, threshold=threshold) for term in terms
                ]
                record.matched_terms = collect_matched_terms(record.terms)
                record.entity_count = sum(term.mentions for term in terms)
                if not dry_run:
                    path = write_chunk_glossary(
                        out_dir,
                        record,
                        language=language,
                        backend=chosen_backend,
                        threshold=threshold,
                        scorer=scorer,
                        glossary_path=str(resolved_glossary),
                        glossary_sha256=glossary_sha,
                        keep_types=wanted_types,
                        min_term_chars=minimum_chars,
                        drop_temporal_only=bool(drop_temporal_only),
                        candidates=candidate_count,
                    )
                    record.output_path = str(path)
                record.log.append(
                    f"match: {record.term_count} term(s) -> "
                    f"{record.matched_count} matched, {record.unmatched_count} unmatched"
                )
            except Exception as exc:  # one bad chunk must not stop the book
                record.ok = False
                record.error = f"{type(exc).__name__}: {exc}"
                errors.append(f"{record.chunk.filename}: {record.error}")
                record.log.append(f"match failed: {record.error}")
            record.match_seconds = clock() - match_started
        records.append(record)
        context.emit(record.to_dict(), completed, len(inputs))
    match_phase_seconds = clock() - match_phase_started
    return records, errors, ner_phase_seconds, match_phase_seconds


__all__ = ["run_phases"]
