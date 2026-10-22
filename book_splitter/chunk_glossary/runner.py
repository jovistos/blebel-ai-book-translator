"""Stage 6: attach the book glossary to every chunk, in two strict phases.

Stage 5 produced a flat folder of translation-sized chunks; stage 4 produced one
book-wide dictionary. This stage is the join between them, and the whole reason it
exists is that a translator sending chunk 137 must know which of the book's locked
renderings that chunk actually mentions.

The run is deliberately **two phases**, in this order:

```
phase 1 — NER over every chunk                         (one complete pass)
    05-chunks/00000-…  -> tagger -> ["Džordž"(PER), "London"(LOC), …]
    05-chunks/00001-…  -> tagger -> ["Dženi"(PER), "Beograd"(LOC), …]
    …
    no glossary is read yet; the phase is complete before phase 2 starts

phase 2 — fuzzy retrieval for every collected term      (one complete pass)
    "Džordž" -> glossary entry "Džordž"      100.0  -> character / Džordž
    "Dženi"  -> glossary entry "Dženi"       100.0  -> character / Dženi
    "Mithril"-> best 71.4, below threshold          -> unmatched
    …
    06-chunk-glossary/<chunk>.json  (the chunk's own name, one file per chunk)
```

That ordering is the contract, not an implementation detail: the NER pass must
finish for **all** chunks before the first lookup, so the stage has one complete
picture of the book's chunk-level entities before it decides what any of them
means. The tests pin it by recording the order of the two injected doubles.

The tagger is stage 3's: the same :mod:`book_splitter.dictionary.ner` backends,
chosen by language, used here to *find* terms rather than to gate sentences. The
retrieval is ``rapidfuzz`` and nothing else: stage 4 already locked each
rendering, so a chunk term only needs the entry that names it. The score is a
percentage on rapidfuzz's 0..100 scale and is a command-line option (95 by
default).

The output folder mirrors the chunk folder: ``05-chunks/00001-x-0000.md`` gets
``06-chunk-glossary/00001-x-0000.md.json``. The file carries the chunk's identity
(global order, chapter, character and line spans), every NER term with its match
or near miss, and a ``matched_terms`` list a translation prompt can consume
directly.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.artifacts import (
    read_json,
    sha256_file,
    timestamp,
    write_json,
    write_manifest,
    write_trace,
)
from ..core.constants import (
    CHUNK_GLOSSARY_MANIFEST_FILENAME,
    CHUNK_GLOSSARY_TRACE_FILENAME,
    DEFAULT_CHUNK_GLOSSARY_CANDIDATES,
    DEFAULT_CHUNK_GLOSSARY_MIN_TERM_CHARS,
    DEFAULT_CHUNK_GLOSSARY_SCORER,
    DEFAULT_CHUNK_GLOSSARY_THRESHOLD,
    DEFAULT_KEEP_ENTITY_TYPES,
)
from ..core.errors import MissingDependencyError, ToolError
from ..core.run import RunContext
from ..dictionary.language import LanguageDetector, LanguageGuess, detect_language
from ..dictionary.ner import (
    Entity,
    NerBackend,
    backend_device,
    configure_gpu,
    is_temporal,
    make_backend,
    normalize_type,
)
from ..documents.chapters import Chapter, chunk_glossary_dir
from ..documents.chunks import (
    ChunkInput,
    chunk_glossary_filename,
    discover_chunk_inputs,
    resolve_chunks_dir,
    resolve_glossary_path,
)
from ..documents.images import strip_images
from ..documents.manifest import load_chunk_glossary_manifest
from .matching import (
    LexiconEntry,
    MatchCandidate,
    RapidFuzzMatcher,
    TermMatcher,
    clean_term,
    load_glossary,
    match_key,
    normalize_scorer,
    normalize_threshold,
)

#: Monotonic clock used for the timings; injectable so tests can be exact.
Clock = Callable[[], float]
#: Called once per chunk as ``(chunk_record, completed, total)``.
ProgressHook = Callable[[dict[str, Any], int, int], None]


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #
@dataclass
class EntityTerm:
    """One distinct NER term inside one chunk, with its types and mention count."""

    term: str
    types: list[str] = field(default_factory=list)
    mentions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"term": self.term, "types": list(self.types), "mentions": self.mentions}


@dataclass
class TermMatch:
    """A chunk term and the glossary candidates rapidfuzz retrieved for it.

    The threshold is stored on the result rather than applied by the matcher, so a
    per-chunk file can report the near miss (below the threshold) as well as the
    accepted match, and the same matcher can be reused at another threshold.
    """

    term: str
    types: list[str]
    mentions: int
    candidates: list[MatchCandidate] = field(default_factory=list)
    threshold: float = DEFAULT_CHUNK_GLOSSARY_THRESHOLD

    @property
    def best(self) -> MatchCandidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def matched(self) -> bool:
        return bool(self.best is not None and self.best.score >= self.threshold)

    def to_dict(self) -> dict[str, Any]:
        best = self.best
        matched = self.matched
        return {
            "term": self.term,
            "types": list(self.types),
            "mentions": self.mentions,
            "matched": matched,
            "score": round(float(best.score), 2) if best is not None else 0.0,
            "glossary": best.entry.to_dict() if matched and best is not None else None,
            "closest": best.to_dict() if best is not None and not matched else None,
            "alternatives": [candidate.to_dict() for candidate in self.candidates[1:]],
        }


@dataclass
class ChunkGlossary:
    """One chunk's outcome: its NER terms, its matches and its written file."""

    chunk: ChunkInput
    terms: list[TermMatch] = field(default_factory=list)
    matched_terms: list[dict[str, Any]] = field(default_factory=list)
    entity_count: int = 0
    ner_calls: int = 0
    ner_seconds: float = 0.0
    match_seconds: float = 0.0
    output_path: str | None = None
    ok: bool = True
    error: str | None = None
    skipped: bool = False
    log: list[str] = field(default_factory=list)

    @property
    def term_count(self) -> int:
        return len(self.terms)

    @property
    def matched_count(self) -> int:
        return len(self.matched_terms)

    @property
    def unmatched_count(self) -> int:
        return sum(1 for term in self.terms if not term.matched)

    @property
    def elapsed_seconds(self) -> float:
        return self.ner_seconds + self.match_seconds

    def record(self) -> dict[str, Any]:
        """The flat per-chunk record stored in the manifest."""
        return {
            **self.chunk.to_dict(),
            "entity_count": self.entity_count,
            "term_count": self.term_count,
            "matched_count": self.matched_count,
            "unmatched_count": self.unmatched_count,
            "matched_terms": list(self.matched_terms),
            "output_path": self.output_path,
            "ok": self.ok,
            "error": self.error,
            "skipped": self.skipped,
            "ner_calls": self.ner_calls,
            "ner_seconds": round(self.ner_seconds, 4),
            "match_seconds": round(self.match_seconds, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 4),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.record(),
            "terms": [term.to_dict() for term in self.terms],
            "log": list(self.log),
        }


@dataclass
class ChunkGlossaryResult:
    """Outcome of a whole stage-6 run."""

    ok: bool
    chunks_path: str
    glossary_path: str
    output_dir: str
    language: dict[str, Any] = field(default_factory=dict)
    backend: str = ""
    device: str = ""
    chunk_count: int = 0
    ok_count: int = 0
    processed_count: int = 0
    skipped_count: int = 0
    entry_count: int = 0
    entity_count: int = 0
    term_count: int = 0
    matched_term_count: int = 0
    unmatched_term_count: int = 0
    matched_chunk_count: int = 0
    dry_run: bool = False
    limited: bool = False
    settings: dict[str, Any] = field(default_factory=dict)
    totals: dict[str, int] = field(default_factory=dict)
    phases: list[dict[str, Any]] = field(default_factory=list)
    timing: dict[str, float] = field(default_factory=dict)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    manifest_path: str | None = None
    trace_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def _glossary_sha256(path: str | Path) -> str:
    """SHA-256 of the glossary file, so a resume notices a changed dictionary."""
    try:
        return sha256_file(path)
    except OSError:
        return ""


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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
def run_chunk_glossary(
    chunks_path: str,
    output_dir: str | None = None,
    *,
    glossary_path: str | Path | None = None,
    language: str = "auto",
    detector: LanguageDetector | None = None,
    backend: NerBackend | None = None,
    backend_name: str | None = None,
    matcher: TermMatcher | None = None,
    gpu: int | None = None,
    use_gpu: bool = True,
    auto_download: bool = True,
    keep_types: Sequence[str] = DEFAULT_KEEP_ENTITY_TYPES,
    min_term_chars: int = DEFAULT_CHUNK_GLOSSARY_MIN_TERM_CHARS,
    drop_temporal_only: bool = False,
    threshold: float = DEFAULT_CHUNK_GLOSSARY_THRESHOLD,
    scorer: str = DEFAULT_CHUNK_GLOSSARY_SCORER,
    candidates: int = DEFAULT_CHUNK_GLOSSARY_CANDIDATES,
    dry_run: bool = False,
    limit: int | None = None,
    chapter_files: Sequence[str] | None = None,
    skip_if_present: bool = False,
    write_manifest: bool = True,
    progress: ProgressHook | None = None,
    clock: Clock = time.perf_counter,
) -> ChunkGlossaryResult:
    """Build the per-chunk glossary in two passes: all NER, then all retrieval.

    ``backend`` and ``matcher`` are injectable, which is how the offline tests run
    the whole stage without torch, a model download, or a real glossary; a real
    run builds the tagger from ``language`` (detected from the chunks when
    ``language="auto"``) and the rapidfuzz matcher from ``glossary_path``.

    ``dry_run`` still runs both phases — the NER output *is* the artifact, so there
    is nothing to validate without tagging — but writes no per-chunk file,
    manifest or trace. ``chapter_files`` restricts the run to chunks belonging to
    those chapter filenames (the chapter-aligned window); ``limit`` then caps the
    filtered list, which is the cheap way to inspect a book before a full run.
    ``skip_if_present`` reuses a per-chunk file whose recorded chunk hash,
    threshold, scorer and glossary hash still match, and returns before loading a
    tagger when every chunk is covered; growing a chapter window therefore only
    tags the new chunks.
    """
    threshold = normalize_threshold(threshold)
    scorer = normalize_scorer(scorer)
    candidate_count = max(1, _int(candidates, DEFAULT_CHUNK_GLOSSARY_CANDIDATES))
    minimum_chars = max(0, _int(min_term_chars, DEFAULT_CHUNK_GLOSSARY_MIN_TERM_CHARS))
    if isinstance(keep_types, str):
        keep_types = tuple(part.strip() for part in keep_types.split(",") if part.strip())
    wanted_types = tuple(str(value).strip().upper() for value in keep_types if str(value).strip())
    if not wanted_types:
        wanted_types = tuple(DEFAULT_KEEP_ENTITY_TYPES)

    all_inputs, source_dir = discover_chunk_inputs(chunks_path)
    if chapter_files is not None:
        allowed = {str(name) for name in chapter_files}
        all_inputs = [
            chunk for chunk in all_inputs if chunk.chapter_filename in allowed
        ]
        if not all_inputs:
            raise ToolError(
                "no chunks belong to the requested chapter window "
                f"({', '.join(sorted(allowed)[:5])}…)"
            )
    limited = limit is not None and 0 < _int(limit) < len(all_inputs)
    inputs = all_inputs[: _int(limit)] if limit is not None and _int(limit) > 0 else all_inputs
    context = RunContext(clock=clock, dry_run=dry_run, limited=limited, progress=progress)

    out_dir = (
        Path(output_dir).expanduser()
        if output_dir
        else chunk_glossary_dir(source_dir)
    )
    resolved_glossary = resolve_glossary_path(glossary_path, source_dir)
    lexicon = load_glossary(resolved_glossary)
    glossary_sha = _glossary_sha256(resolved_glossary)
    reuse_enabled = bool(skip_if_present and not dry_run)
    reused_records: dict[str, ChunkGlossary] = {}
    if reuse_enabled:
        for chunk in inputs:
            reusable = _reusable_chunk_glossary(
                chunk,
                out_dir,
                glossary_sha256=glossary_sha,
                threshold=threshold,
                scorer=scorer,
                keep_types=wanted_types,
                min_term_chars=minimum_chars,
                drop_temporal_only=bool(drop_temporal_only),
                candidates=candidate_count,
            )
            if reusable is not None:
                reused_records[chunk.filename] = reusable
    pending = [chunk for chunk in inputs if chunk.filename not in reused_records]

    if not pending:
        # Nothing left to tag: report the files already on disk instead of loading
        # a tagger, so a finished (or widened) window needs no model at all.
        previous = load_chunk_glossary_manifest(out_dir)
        reused_result_records = [reused_records[chunk.filename] for chunk in inputs]
        totals = _totals(reused_result_records)
        settings = {
            "language": previous.get("language") or {},
            "backend": str(previous.get("backend") or ""),
            "device": str(previous.get("device") or ""),
            "threshold": threshold,
            "scorer": scorer,
            "candidates": candidate_count,
            "keep_types": list(wanted_types),
            "min_term_chars": minimum_chars,
            "drop_temporal_only": bool(drop_temporal_only),
            "limit": int(limit) if limit is not None else None,
            "chapter_files": list(chapter_files) if chapter_files is not None else None,
            "skip_if_present": bool(skip_if_present),
        }
        result = ChunkGlossaryResult(
            ok=True,
            chunks_path=str(source_dir),
            glossary_path=str(resolved_glossary),
            output_dir=str(out_dir),
            language=dict(previous.get("language") or {}),
            backend=str(previous.get("backend") or ""),
            device=str(previous.get("device") or ""),
            chunk_count=len(reused_result_records),
            ok_count=len(reused_result_records),
            processed_count=0,
            skipped_count=len(reused_result_records),
            entry_count=len(lexicon),
            entity_count=totals.get("entity_mentions", 0),
            term_count=totals.get("terms", 0),
            matched_term_count=totals.get("matched_terms", 0),
            unmatched_term_count=totals.get("unmatched_terms", 0),
            matched_chunk_count=totals.get("matched_chunks", 0),
            dry_run=dry_run,
            limited=limited,
            settings=settings,
            totals=totals,
            phases=[
                {"name": "ner", "chunks": 0, "entities": 0, "terms": 0, "seconds": 0.0},
                {
                    "name": "match",
                    "chunks": 0,
                    "matched_terms": 0,
                    "unmatched_terms": 0,
                    "seconds": 0.0,
                },
            ],
            timing={
                "run_seconds": round(context.elapsed(), 4),
                "backend_load_seconds": 0.0,
                "ner_phase_seconds": 0.0,
                "match_phase_seconds": 0.0,
                "milliseconds_per_chunk": 0.0,
            },
            chunks=[record.record() for record in reused_result_records],
            errors=[],
        )
        if write_manifest and not dry_run:
            result.manifest_path = str(
                write_chunk_glossary_manifest(out_dir, result, reused_result_records)
            )
            result.trace_path = str(
                write_chunk_glossary_trace(out_dir, result, reused_result_records)
            )
        return result

    # Built after the resume check on purpose: a fully reused window must not
    # require rapidfuzz or a tagger, only the glossary file it already used.
    # Built before the NER phase on purpose: a missing rapidfuzz or an unreadable
    # glossary is a setup problem, and discovering it after tagging the whole
    # book would waste the expensive half of the run. It is *called* only in
    # phase 2, after every chunk has been tagged.
    matcher = matcher or RapidFuzzMatcher(
        lexicon, scorer=scorer, candidates=candidate_count
    )
    matcher_name = type(matcher).__name__

    # An injected backend already names the language it speaks; asking lingua-py
    # again would only add a dependency to a run that does not need one.
    injected_language = str(getattr(backend, "language", "") or "") if backend is not None else ""
    if injected_language and str(language or "").strip().lower() in {"", "auto"}:
        guess = LanguageGuess(
            language=injected_language,
            confidence=1.0,
            method="backend",
        )
    else:
        guess = detect_language(_proxy_chapters(inputs), language=language, detector=detector)

    backend_load_started = clock()
    resolved_gpu: int | None = None
    if backend is None:
        resolved_gpu = configure_gpu(gpu, use_gpu=use_gpu)
        try:
            backend = make_backend(
                guess.language,
                use_gpu=use_gpu,
                auto_download=auto_download,
                backend=backend_name,
            )
        except MissingDependencyError:
            raise
        except Exception as exc:
            raise ToolError(
                f"could not load the {backend_name or guess.language!r} NER models: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    backend_load_seconds = clock() - backend_load_started
    chosen_backend = backend.name
    device = backend_device(backend)

    settings = {
        "language": guess.language,
        "language_method": guess.method,
        "backend": chosen_backend,
        "device": device,
        "gpu": resolved_gpu,
        "threshold": threshold,
        "scorer": scorer,
        "candidates": candidate_count,
        "matcher": matcher_name,
        "keep_types": list(wanted_types),
        "min_term_chars": minimum_chars,
        "drop_temporal_only": bool(drop_temporal_only),
        "limit": int(limit) if limit is not None else None,
        "chapter_files": list(chapter_files) if chapter_files is not None else None,
        "skip_if_present": bool(skip_if_present),
    }

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
                        language=guess.language,
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

    totals = _totals(records)
    result = ChunkGlossaryResult(
        ok=bool(records) and all(record.ok for record in records) and not errors,
        chunks_path=str(source_dir),
        glossary_path=str(resolved_glossary),
        output_dir=str(out_dir),
        language=guess.to_dict(),
        backend=chosen_backend,
        device=device,
        chunk_count=len(records),
        ok_count=sum(1 for record in records if record.ok),
        processed_count=len(pending),
        skipped_count=len(reused_records),
        entry_count=len(lexicon),
        entity_count=totals.get("entity_mentions", 0),
        term_count=totals.get("terms", 0),
        matched_term_count=totals.get("matched_terms", 0),
        unmatched_term_count=totals.get("unmatched_terms", 0),
        matched_chunk_count=totals.get("matched_chunks", 0),
        dry_run=dry_run,
        limited=limited,
        settings=settings,
        totals=totals,
        phases=[
            {
                "name": "ner",
                "chunks": len(pending),
                "entities": totals.get("entity_mentions", 0),
                "terms": totals.get("terms", 0),
                "seconds": round(ner_phase_seconds, 4),
            },
            {
                "name": "match",
                "chunks": len(records),
                "matched_terms": totals.get("matched_terms", 0),
                "unmatched_terms": totals.get("unmatched_terms", 0),
                "seconds": round(match_phase_seconds, 4),
            },
        ],
        timing={
            "run_seconds": round(context.elapsed(), 4),
            "backend_load_seconds": round(backend_load_seconds, 4),
            "ner_phase_seconds": round(ner_phase_seconds, 4),
            "match_phase_seconds": round(match_phase_seconds, 4),
            "milliseconds_per_chunk": round(
                (1000.0 * ner_phase_seconds / len(records)) if records else 0.0, 3
            ),
        },
        chunks=[record.record() for record in records],
        errors=errors,
    )
    if write_manifest and not dry_run:
        result.manifest_path = str(write_chunk_glossary_manifest(out_dir, result, records))
        result.trace_path = str(write_chunk_glossary_trace(out_dir, result, records))
    return context.finish(result)


def _totals(records: Sequence[ChunkGlossary]) -> dict[str, int]:
    """Book-level counts, derived once from the per-chunk records."""
    matched_keys: set[str] = set()
    for record in records:
        for term in record.matched_terms:
            key = match_key(str(term.get("glossary_term") or ""))
            if key:
                matched_keys.add(key)
    return {
        "chunks": len(records),
        "ok_chunks": sum(1 for record in records if record.ok),
        "failed_chunks": sum(1 for record in records if not record.ok),
        "entity_mentions": sum(record.entity_count for record in records),
        "terms": sum(record.term_count for record in records),
        "matched_terms": sum(record.matched_count for record in records),
        "unmatched_terms": sum(record.unmatched_count for record in records),
        "matched_chunks": sum(1 for record in records if record.matched_count),
        "unique_matched_glossary_terms": len(matched_keys),
    }


# --------------------------------------------------------------------------- #
# writing the artifacts
# --------------------------------------------------------------------------- #
def chunk_glossary_payload(
    record: ChunkGlossary,
    *,
    language: str = "",
    backend: str = "",
    threshold: float = DEFAULT_CHUNK_GLOSSARY_THRESHOLD,
    scorer: str = DEFAULT_CHUNK_GLOSSARY_SCORER,
    glossary_path: str = "",
    glossary_sha256: str = "",
    keep_types: Sequence[str] = DEFAULT_KEEP_ENTITY_TYPES,
    min_term_chars: int = DEFAULT_CHUNK_GLOSSARY_MIN_TERM_CHARS,
    drop_temporal_only: bool = False,
    candidates: int = DEFAULT_CHUNK_GLOSSARY_CANDIDATES,
) -> dict[str, Any]:
    """The per-chunk file: the chunk's identity, its terms and their matches.

    The settings that produced the terms are recorded with them, so a later
    ``--skip-if-present`` run can tell whether the file still answers the same
    question instead of only whether the chunk and glossary hashes match.
    """
    return {
        "stage": "chunk-glossary",
        "generated_at": timestamp(),
        "chunk": record.chunk.to_dict(),
        "language": language,
        "backend": backend,
        "threshold": threshold,
        "scorer": scorer,
        "glossary_path": glossary_path,
        "glossary_sha256": glossary_sha256,
        "keep_types": sorted(
            {str(value).strip().upper() for value in keep_types if str(value).strip()}
        ),
        "min_term_chars": int(min_term_chars),
        "drop_temporal_only": bool(drop_temporal_only),
        "candidates": int(candidates),
        "counts": {
            "entities": record.entity_count,
            "terms": record.term_count,
            "matched": record.matched_count,
            "unmatched": record.unmatched_count,
        },
        "terms": [term.to_dict() for term in record.terms],
        "matched_terms": list(record.matched_terms),
    }


def write_chunk_glossary(
    out_dir: str | Path,
    record: ChunkGlossary,
    *,
    language: str = "",
    backend: str = "",
    threshold: float = DEFAULT_CHUNK_GLOSSARY_THRESHOLD,
    scorer: str = DEFAULT_CHUNK_GLOSSARY_SCORER,
    glossary_path: str = "",
    glossary_sha256: str = "",
    keep_types: Sequence[str] = DEFAULT_KEEP_ENTITY_TYPES,
    min_term_chars: int = DEFAULT_CHUNK_GLOSSARY_MIN_TERM_CHARS,
    drop_temporal_only: bool = False,
    candidates: int = DEFAULT_CHUNK_GLOSSARY_CANDIDATES,
) -> Path:
    """Write one chunk's glossary next to the others, under the chunk's name."""
    path = Path(out_dir) / chunk_glossary_filename(record.chunk.filename)
    payload = chunk_glossary_payload(
        record,
        language=language,
        backend=backend,
        threshold=threshold,
        scorer=scorer,
        glossary_path=glossary_path,
        glossary_sha256=glossary_sha256,
        keep_types=keep_types,
        min_term_chars=min_term_chars,
        drop_temporal_only=drop_temporal_only,
        candidates=candidates,
    )
    return write_json(path, payload)


def write_chunk_glossary_manifest(
    out_dir: str | Path,
    result: ChunkGlossaryResult,
    records: Sequence[ChunkGlossary] | None = None,
) -> Path:
    """Write ``_chunk_glossary_manifest.json``: the book-level index."""
    payload = {
        "stage": "chunk-glossary",
        "ok": result.ok,
        "dry_run": result.dry_run,
        "limited": result.limited,
        "skip_if_present": bool(result.settings.get("skip_if_present")),
        "chunks_path": result.chunks_path,
        "glossary_path": result.glossary_path,
        "output_dir": result.output_dir,
        "language": result.language,
        "backend": result.backend,
        "device": result.device,
        "settings": result.settings,
        "chunk_count": result.chunk_count,
        "ok_count": result.ok_count,
        "processed_count": result.processed_count,
        "skipped_count": result.skipped_count,
        "entry_count": result.entry_count,
        "totals": result.totals,
        "phases": result.phases,
        "timing": result.timing,
        "chunks": [record.record() for record in records] if records is not None else result.chunks,
        "errors": result.errors,
    }
    return write_manifest(out_dir, CHUNK_GLOSSARY_MANIFEST_FILENAME, payload)


def write_chunk_glossary_trace(
    out_dir: str | Path,
    result: ChunkGlossaryResult,
    records: Sequence[ChunkGlossary] | None = None,
) -> Path:
    """Write the audit trail: the same run plus each chunk's terms and log."""
    payload = result.to_dict()
    if records is not None:
        payload["detail"] = [record.to_dict() for record in records]
    return write_trace(out_dir, CHUNK_GLOSSARY_TRACE_FILENAME, payload)


__all__ = [
    "ChunkGlossary",
    "ChunkGlossaryResult",
    "ChunkInput",
    "Clock",
    "EntityTerm",
    "ProgressHook",
    "TermMatch",
    "aggregate_entities",
    "chunk_glossary_filename",
    "chunk_glossary_payload",
    "collect_matched_terms",
    "discover_chunk_inputs",
    "load_chunk_glossary_manifest",
    "match_term",
    "resolve_chunks_dir",
    "resolve_glossary_path",
    "run_chunk_glossary",
    "write_chunk_glossary",
    "write_chunk_glossary_manifest",
    "write_chunk_glossary_trace",
]
