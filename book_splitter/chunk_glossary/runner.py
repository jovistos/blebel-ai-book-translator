"""Stage 6: attach the book glossary to every chunk, in two strict phases.

Stage 5 produced a flat folder of translation-sized chunks; stage 4 produced one
book-wide dictionary. This stage is the join between them, and the whole reason it
exists is that a translator sending chunk 137 must know which of the book's locked
renderings that chunk actually mentions.

The run is deliberately **two phases**, in this order:

```
phase 1 — NER over every chunk                         (one complete pass)
    chunks/00000-…  -> tagger -> ["Džordž"(PER), "London"(LOC), …]
    chunks/00001-…  -> tagger -> ["Dženi"(PER), "Beograd"(LOC), …]
    …
    no glossary is read yet; the phase is complete before phase 2 starts

phase 2 — fuzzy retrieval for every collected term      (one complete pass)
    "Džordž" -> glossary entry "Džordž"      100.0  -> character / Džordž
    "Dženi"  -> glossary entry "Dženi"       100.0  -> character / Dženi
    "Mithril"-> best 71.4, below threshold          -> unmatched
    …
    chunk-glossary/<chunk>.json  (the chunk's own name, one file per chunk)
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

The output folder mirrors the chunk folder: ``chunks/00001-x-0000.md`` gets
``chunk-glossary/00001-x-0000.md.json``. The file carries the chunk's identity
(global order, chapter, character and line spans), every NER term with its match
or near miss, and a ``matched_terms`` list a translation prompt can consume
directly.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..chunking.runner import chunk_files
from ..core.constants import (
    CHAPTERS_DIRNAME,
    CHUNK_GLOSSARY_DIRNAME,
    CHUNK_GLOSSARY_MANIFEST_FILENAME,
    CHUNK_GLOSSARY_TRACE_FILENAME,
    CHUNKS_DIRNAME,
    DEFAULT_CHUNK_GLOSSARY_CANDIDATES,
    DEFAULT_CHUNK_GLOSSARY_MIN_TERM_CHARS,
    DEFAULT_CHUNK_GLOSSARY_SCORER,
    DEFAULT_CHUNK_GLOSSARY_THRESHOLD,
    DEFAULT_KEEP_ENTITY_TYPES,
    GLOSSARY_DIRNAME,
    GLOSSARY_FILENAME,
    PREPROCESSED_DIRNAME,
    REDUCED_DIRNAME,
    UNIQUE_DIRNAME,
)
from ..core.errors import MissingDependencyError, ToolError
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
from .matching import (
    MatchCandidate,
    RapidFuzzMatcher,
    TermMatcher,
    clean_term,
    glossary_file,
    load_glossary,
    match_key,
    normalize_scorer,
    normalize_threshold,
)

#: Folder names that identify the book root when a stage is handed one of its own
#: output folders instead of the book folder itself.
_BOOK_FOLDER_NAMES = frozenset(
    {
        CHAPTERS_DIRNAME,
        PREPROCESSED_DIRNAME,
        REDUCED_DIRNAME,
        UNIQUE_DIRNAME,
        GLOSSARY_DIRNAME,
        CHUNKS_DIRNAME,
        CHUNK_GLOSSARY_DIRNAME,
    }
)

#: Monotonic clock used for the timings; injectable so tests can be exact.
Clock = Callable[[], float]
#: Called once per chunk as ``(chunk_record, completed, total)``.
ProgressHook = Callable[[dict[str, Any], int, int], None]


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #
@dataclass
class ChunkInput:
    """One stage-5 chunk: where it is, what it covers and the text to tag."""

    filename: str
    path: str
    global_index: int
    chunk_index: int
    chapter_id: str = ""
    chapter_filename: str = ""
    start_char: int = 0
    end_char: int = 0
    start_line: int = 0
    end_line: int = 0
    token_count: int = 0
    sha256: str = ""
    chars: int = 0
    #: The chunk's exact text. Kept out of every serialised record.
    text: str = field(default="", repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "path": self.path,
            "global_index": self.global_index,
            "chunk_index": self.chunk_index,
            "chapter_id": self.chapter_id,
            "chapter_filename": self.chapter_filename,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "token_count": self.token_count,
            "chars": self.chars,
            "sha256": self.sha256,
        }


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
def resolve_chunks_dir(path: str | Path) -> Path:
    """Resolve a chunks folder, a book folder, or the manifest itself."""
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate.parent
    if candidate.name == CHUNKS_DIRNAME:
        return candidate
    nested = candidate / CHUNKS_DIRNAME
    if nested.is_dir():
        return nested
    return candidate


def _book_root(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_file():
        candidate = candidate.parent
    return candidate.parent if candidate.name in _BOOK_FOLDER_NAMES else candidate


def resolve_glossary_path(
    glossary_path: str | Path | None,
    chunks_path: str | Path,
) -> Path:
    """Where ``glossary.json`` is: explicit, or the book folder's ``glossary/``."""
    if glossary_path:
        return glossary_file(glossary_path)
    return _book_root(chunks_path) / GLOSSARY_DIRNAME / GLOSSARY_FILENAME


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def discover_chunk_inputs(chunks_path: str | Path) -> tuple[list[ChunkInput], Path]:
    """Read every chunk file in stage-5 order, with its exact text.

    The stage-5 manifest is authoritative when it is there: it carries the global
    order, the chapter membership, the character/line spans and the chunk hash. A
    folder without one still works (the chunk files sort into reading order), but
    its files then only know their own name.
    """
    folder = resolve_chunks_dir(chunks_path)
    if not folder.is_dir():
        raise ToolError(f"chunks folder not found: {folder} — run `chunk` first")
    entries = chunk_files(folder)
    if not entries:
        raise ToolError(f"no chunk files in {folder} — run `chunk` first")

    ordered = sorted(
        entries,
        key=lambda entry: (_int(entry.get("global_index"), 0), str(entry.get("filename") or "")),
    )
    inputs: list[ChunkInput] = []
    missing: list[str] = []
    for position, entry in enumerate(ordered):
        filename = str(entry.get("filename") or Path(str(entry.get("path") or "")).name)
        if not filename:
            continue
        path = folder / filename
        if not path.is_file():
            missing.append(filename)
            continue
        text = path.read_text(encoding="utf-8")
        payload = text.encode("utf-8")
        chapter_filename = str(entry.get("chapter_filename") or "")
        inputs.append(
            ChunkInput(
                filename=filename,
                path=str(path),
                global_index=_int(entry.get("global_index"), position),
                chunk_index=_int(entry.get("chunk_index"), 0),
                chapter_id=str(entry.get("chapter_id") or Path(chapter_filename).stem),
                chapter_filename=chapter_filename,
                start_char=_int(entry.get("start_char")),
                end_char=_int(entry.get("end_char")),
                start_line=_int(entry.get("start_line")),
                end_line=_int(entry.get("end_line")),
                token_count=_int(entry.get("token_count")),
                sha256=str(entry.get("sha256") or hashlib.sha256(payload).hexdigest()),
                chars=len(text),
                text=text,
            )
        )
    if missing:
        raise ToolError(
            f"{len(missing)} chunk file(s) recorded in the manifest are missing from "
            f"{folder}: {', '.join(missing[:5])}"
            + (" …" if len(missing) > 5 else "")
            + " — re-run `chunk` to rebuild them"
        )
    if not inputs:
        raise ToolError(f"no readable chunk files in {folder} — run `chunk` first")
    inputs.sort(key=lambda chunk: (chunk.global_index, chunk.filename))
    return inputs, folder


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
    for term in terms.values():
        term.types.sort()
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
    manifest or trace. ``limit`` tags and matches only the first N chunks, which
    is the cheap way to inspect a book before a full run.
    """
    run_started = clock()
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
    limited = limit is not None and 0 < _int(limit) < len(all_inputs)
    inputs = all_inputs[: _int(limit)] if limit is not None and _int(limit) > 0 else all_inputs

    out_dir = (
        Path(output_dir).expanduser()
        if output_dir
        else chunk_glossary_dir(source_dir)
    )
    resolved_glossary = resolve_glossary_path(glossary_path, source_dir)
    lexicon = load_glossary(resolved_glossary)
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
    }

    records: list[ChunkGlossary] = []
    entity_terms: dict[str, list[EntityTerm]] = {}
    errors: list[str] = []

    # -- phase 1: NER over every chunk ------------------------------------ #
    ner_phase_started = clock()
    for chunk in inputs:
        record = ChunkGlossary(chunk=chunk)
        tag_started = clock()
        try:
            if chunk.text.strip():
                entities = backend.entities(chunk.text)
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
        records.append(record)
    ner_phase_seconds = clock() - ner_phase_started

    # -- phase 2: retrieval for every term collected above ----------------- #
    match_phase_started = clock()
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    for completed, record in enumerate(records, start=1):
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
        if progress is not None:
            with contextlib.suppress(Exception):  # a reporting hook must never break a run
                progress(record.to_dict(), completed, len(records))
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
        processed_count=len(records),
        skipped_count=0,
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
                "chunks": len(records),
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
            "run_seconds": round(clock() - run_started, 4),
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
    return result


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
def chunk_glossary_filename(chunk_filename: str) -> str:
    """``00001-01-x-0000.md`` -> ``00001-01-x-0000.md.json``.

    Keeping the full chunk name (including its ``.md``) makes the mapping
    unambiguous and guarantees two chunks cannot collide on one output file. A
    consumer strips the final ``.json`` to find the chunk it describes.
    """
    name = Path(str(chunk_filename or "")).name or "chunk.md"
    return f"{name}.json"


def chunk_glossary_payload(
    record: ChunkGlossary,
    *,
    language: str = "",
    backend: str = "",
    threshold: float = DEFAULT_CHUNK_GLOSSARY_THRESHOLD,
    scorer: str = DEFAULT_CHUNK_GLOSSARY_SCORER,
    glossary_path: str = "",
) -> dict[str, Any]:
    """The per-chunk file: the chunk's identity, its terms and their matches."""
    return {
        "stage": "chunk-glossary",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "chunk": record.chunk.to_dict(),
        "language": language,
        "backend": backend,
        "threshold": threshold,
        "scorer": scorer,
        "glossary_path": glossary_path,
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
) -> Path:
    """Write one chunk's glossary next to the others, under the chunk's name."""
    folder = Path(out_dir)
    path = folder / chunk_glossary_filename(record.chunk.filename)
    payload = chunk_glossary_payload(
        record,
        language=language,
        backend=backend,
        threshold=threshold,
        scorer=scorer,
        glossary_path=glossary_path,
    )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_chunk_glossary_manifest(
    out_dir: str | Path,
    result: ChunkGlossaryResult,
    records: Sequence[ChunkGlossary] | None = None,
) -> Path:
    """Write ``_chunk_glossary_manifest.json``: the book-level index."""
    path = Path(out_dir) / CHUNK_GLOSSARY_MANIFEST_FILENAME
    payload = {
        "stage": "chunk-glossary",
        "ok": result.ok,
        "dry_run": result.dry_run,
        "limited": result.limited,
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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_chunk_glossary_trace(
    out_dir: str | Path,
    result: ChunkGlossaryResult,
    records: Sequence[ChunkGlossary] | None = None,
) -> Path:
    """Write the audit trail: the same run plus each chunk's terms and log."""
    path = Path(out_dir) / CHUNK_GLOSSARY_TRACE_FILENAME
    payload = result.to_dict()
    if records is not None:
        payload["detail"] = [record.to_dict() for record in records]
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return path


def load_chunk_glossary_manifest(chunks_path: str | Path) -> dict[str, Any]:
    """Read ``_chunk_glossary_manifest.json`` from an output folder (or a file)."""
    path = Path(chunks_path).expanduser()
    manifest = (
        path / CHUNK_GLOSSARY_MANIFEST_FILENAME if path.is_dir() else path
    )
    if not manifest.is_file():
        return {}
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


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
