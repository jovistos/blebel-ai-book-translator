"""NER aggregation, retrieval and resume checks for the chunk-glossary stage."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from ..core.artifacts import read_json, sha256_file
from ..core.coerce import as_int as _int
from ..core.constants import (
    DEFAULT_CHUNK_GLOSSARY_MIN_TERM_CHARS,
    DEFAULT_CHUNK_GLOSSARY_THRESHOLD,
    DEFAULT_KEEP_ENTITY_TYPES,
)
from ..core.errors import ToolError
from ..dictionary.ner import Entity, is_temporal, normalize_type
from ..documents.chapters import Chapter
from ..documents.chunks import ChunkInput, chunk_glossary_filename
from .matching import (
    LexiconEntry,
    MatchCandidate,
    TermMatcher,
    clean_term,
    match_key,
    normalize_scorer,
)
from .models import ChunkGlossary, EntityTerm, TermMatch


def _glossary_sha256(path: str | Path) -> str:
    """SHA-256 of the glossary file, so a resume notices a changed dictionary."""
    try:
        return sha256_file(path)
    except OSError:
        return ""


def _candidate_from_payload(item: Any) -> MatchCandidate | None:
    """Rebuild one candidate from a per-chunk file's ``glossary``/``closest``."""
    if not isinstance(item, dict):
        return None
    term = clean_term(item.get("term"))
    if not term:
        return None
    try:
        score = float(item.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    try:
        chapter_index = int(item.get("chapter_index") or 0)
    except (TypeError, ValueError):
        chapter_index = 0
    return MatchCandidate(
        entry=LexiconEntry(
            term=term,
            translation=str(item.get("translation") or ""),
            category=str(item.get("category") or "terminology"),
            chapter_index=chapter_index,
            chapter_filename=str(item.get("chapter_filename") or ""),
        ),
        score=score,
    )


def _terms_from_payload(items: Any, threshold: float) -> list[TermMatch]:
    """Rebuild the ``TermMatch`` list a per-chunk file recorded.

    A resumed run must be able to report the same totals as the original one
    without re-running NER, so the per-chunk JSON is enough: each term carries
    its match (or nearest miss) and the alternatives, and the file records the
    counts the result sums.
    """
    terms: list[TermMatch] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        candidates: list[MatchCandidate] = []
        for key in ("glossary", "closest"):
            candidate = _candidate_from_payload(item.get(key))
            if candidate is not None:
                candidates.append(candidate)
        for alternative in item.get("alternatives") or []:
            candidate = _candidate_from_payload(alternative)
            if candidate is not None:
                candidates.append(candidate)
        terms.append(
            TermMatch(
                term=clean_term(item.get("term")),
                types=[str(value) for value in item.get("types") or []],
                mentions=_int(item.get("mentions")),
                candidates=candidates,
                threshold=threshold,
            )
        )
    return terms


def _reusable_chunk_glossary(
    chunk: ChunkInput,
    out_dir: Path,
    *,
    glossary_sha256: str,
    threshold: float,
    scorer: str,
    keep_types: Sequence[str],
    min_term_chars: int,
    drop_temporal_only: bool,
    candidates: int,
) -> ChunkGlossary | None:
    """A per-chunk file that is still valid for ``chunk``, ready to be reused.

    Every setting that changes the file's content is part of the check, not only
    the chunk and glossary hashes: narrowing ``keep_types`` (or changing the
    minimum term length, the temporal rule or the number of recorded candidates)
    produces different terms for the same chunk, so the old file is not a valid
    answer for the new run.
    """
    path = out_dir / chunk_glossary_filename(chunk.filename)
    if not path.is_file():
        return None
    payload = read_json(path)
    if not isinstance(payload, dict):
        return None

    recorded = payload.get("chunk") or {}
    if isinstance(recorded, dict):
        recorded_sha = str(recorded.get("sha256") or "")
        if recorded_sha and recorded_sha != chunk.sha256:
            return None

    try:
        recorded_threshold = payload.get("threshold")
        if recorded_threshold is not None and abs(float(recorded_threshold) - threshold) > 1e-9:
            return None
    except (TypeError, ValueError):
        return None
    recorded_scorer = payload.get("scorer")
    if recorded_scorer:
        try:
            if normalize_scorer(str(recorded_scorer)) != scorer:
                return None
        except ToolError:
            return None

    recorded_glossary = str(payload.get("glossary_sha256") or "")
    if recorded_glossary and glossary_sha256 and recorded_glossary != glossary_sha256:
        return None

    recorded_keep = payload.get("keep_types")
    if isinstance(recorded_keep, (list, tuple)):
        recorded_types = {
            str(value).strip().upper()
            for value in recorded_keep
            if str(value).strip()
        }
        current_types = {str(value).strip().upper() for value in keep_types}
        if recorded_types != current_types:
            return None
    recorded_min = payload.get("min_term_chars")
    if recorded_min is not None and _int(recorded_min, -1) != int(min_term_chars):
        return None
    if "drop_temporal_only" in payload and bool(payload.get("drop_temporal_only")) != bool(
        drop_temporal_only
    ):
        return None
    recorded_candidates = payload.get("candidates")
    if recorded_candidates is not None and _int(recorded_candidates, -1) != int(candidates):
        return None

    terms = _terms_from_payload(payload.get("terms"), threshold)
    matched = payload.get("matched_terms")
    if not isinstance(matched, list):
        matched = []
    counts = payload.get("counts") or {}
    entity_count = _int(counts.get("entities")) if isinstance(counts, dict) else 0
    if not entity_count:
        entity_count = sum(term.mentions for term in terms)
    return ChunkGlossary(
        chunk=chunk,
        terms=terms,
        matched_terms=[item for item in matched if isinstance(item, dict)],
        entity_count=entity_count,
        output_path=str(path),
        skipped=True,
    )


def _proxy_chapters(chunks: Sequence[ChunkInput]) -> list[Chapter]:
    """Present chunks to stage 3's language detector as chapter-shaped files.

    The detector only needs a ``path`` and a ``filename``: it samples a window
    from a few files. Reusing the exact detector is what keeps stage 6's language
    routing (and the Serbian/Croatian reflex tie-breaker) identical to stage 3's.
    """
    proxies: list[Chapter] = []
    for chunk in chunks:
        text = chunk.text
        line_count = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
        proxies.append(
            Chapter(
                chapter_id=Path(chunk.filename).stem,
                path=chunk.path,
                filename=chunk.filename,
                line_count=line_count,
                bytes=len(text.encode("utf-8")),
            )
        )
    return proxies


# --------------------------------------------------------------------------- #
# phase 1: NER over every chunk
# --------------------------------------------------------------------------- #
def aggregate_entities(
    entities: Iterable[Entity],
    *,
    keep_types: Sequence[str] = DEFAULT_KEEP_ENTITY_TYPES,
    min_term_chars: int = DEFAULT_CHUNK_GLOSSARY_MIN_TERM_CHARS,
    drop_temporal_only: bool = False,
) -> tuple[list[EntityTerm], int]:
    """Collapse one chunk's mentions into distinct terms.

    Returns ``(terms, mention_count)``. A mention counts only when its type is
    wanted and its surface form survives the noise filters (a length floor, at
    least one letter, and optionally "not just a date or a time"). Mentions of the
    same term — same case-folded text — are merged: the chunk needs *Džordž* once,
    with a mention count, not once per occurrence.
    """
    if isinstance(keep_types, str):
        keep_types = tuple(part.strip() for part in keep_types.split(",") if part.strip())
    wanted = {str(value).strip().upper() for value in keep_types}
    minimum = max(0, _int(min_term_chars, DEFAULT_CHUNK_GLOSSARY_MIN_TERM_CHARS))
    terms: dict[str, EntityTerm] = {}
    mentions = 0
    for entity in entities:
        entity_type = normalize_type(entity.type)
        if entity_type not in wanted:
            continue
        if drop_temporal_only and is_temporal(entity):
            continue
        term = clean_term(entity.text)
        if len(term) < minimum or not any(character.isalpha() for character in term):
            continue
        key = match_key(term)
        if not key:
            continue
        mentions += 1
        current = terms.get(key)
        if current is None:
            terms[key] = EntityTerm(term=term, types=[entity_type], mentions=1)
            continue
        current.mentions += 1
        if entity_type not in current.types:
            current.types.append(entity_type)
    for entity_term in terms.values():
        entity_term.types.sort()
    return list(terms.values()), mentions


# --------------------------------------------------------------------------- #
# phase 2: fuzzy retrieval
# --------------------------------------------------------------------------- #
def match_term(
    term: EntityTerm,
    matcher: TermMatcher,
    *,
    threshold: float = DEFAULT_CHUNK_GLOSSARY_THRESHOLD,
) -> TermMatch:
    """Retrieve the glossary candidates for one NER term."""
    candidates = matcher.match(term.term)
    return TermMatch(
        term=term.term,
        types=list(term.types),
        mentions=term.mentions,
        candidates=list(candidates),
        threshold=threshold,
    )


def collect_matched_terms(terms: Iterable[TermMatch]) -> list[dict[str, Any]]:
    """The chunk's accepted matches, one row per glossary entry.

    Two extracted terms can resolve to the same dictionary entry (``Džordž`` and
    ``Džordža`` both name the character). The chunk glossary is keyed on the
    *glossary* entry, so they are merged into one row and the mentions are summed;
    a repeated entry in a translation prompt is paid for, and says nothing new.
    """
    merged: dict[str, dict[str, Any]] = {}
    for term in terms:
        if not term.matched:
            continue
        best = term.best
        if best is None:  # pragma: no cover - matched implies a best candidate
            continue
        entry = best.entry
        key = entry.key
        row = {
            "term": term.term,
            "types": list(term.types),
            "mentions": term.mentions,
            "score": round(float(best.score), 2),
            "glossary_term": entry.term,
            "category": entry.category,
            "translation": entry.translation,
            "glossary_chapter": entry.chapter_filename,
            "glossary_chapter_index": entry.chapter_index,
        }
        existing = merged.get(key)
        if existing is None:
            merged[key] = row
            continue
        existing["mentions"] += term.mentions
        existing["types"] = sorted(set(existing["types"]) | set(term.types))
        if row["score"] > existing["score"]:
            existing["score"] = row["score"]
            existing["term"] = row["term"]
    return list(merged.values())


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #


__all__ = [
    "_proxy_chapters",
    "_reusable_chunk_glossary",
    "_terms_from_payload",
    "aggregate_entities",
    "collect_matched_terms",
    "match_term",
]
