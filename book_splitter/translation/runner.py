"""Stage 7: translate every body chunk, in reading order, with the chunk's glossary.

The first six stages produce translation material. Stage 6b localizes the front
and back matter with its own agent; this is the stage that turns the remaining
body chunks into the translated book:

```
05-chunks/00001-01-chapter-1-0000.md              06-chunk-glossary/00001-….md.json
        │                                              │
        └──────────────┬───────────────────────────────┘
                       ▼
             prompt: system + glossary rows + source chunk
                       │
                       ▼   one structured-output call per chunk
             07-translations/00001-01-chapter-1-0000.md
             07-translation-prompts/00001-….md.json          (the whole input)
             07-translation-prompts/00001-….md.prompt.md     (the same, readable)
```

Front/back matter chunks (``00-front-matter.md``, a preface, a notes section…) are
filtered out unless ``include_matter=True``: stage 6b already wrote their
localized chunks to the same ``07-translations/`` folder with a different prompt and
**without** the dictionary.

**The order matters.** Chunks are visited in reading order, one model call at a
time, because each call is handed the tail of the previous translation when
continuity context is enabled and because a run that is interrupted leaves a
readable prefix of the book. The loop is a plain ``for``, not a pool: chunk 137's
text does not exist until chunk 136 has been translated.

**The glossary is per chunk.** Stage 6 wrote ``matched_terms`` for exactly the
entities a chunk mentions; that list is put in the prompt verbatim. Point
``glossary_scope="book"`` at ``glossary.json`` instead to send the whole
dictionary with every chunk (more consistent, more input tokens).

**Every call is structured and validated.** The model is asked for
``{"translation": "..."}``, Pydantic validates it, and a refusal, an empty answer,
an unchanged source or a summary is retried with the failure named. A chunk that
still fails is reported and the book continues: a hole is visible in the manifest,
an aborted run is not.

**The whole input is kept.** For every chunk, ``07-translation-prompts/<chunk>.json``
holds the system prompt, the user prompt, the serialised messages, the glossary
rows that were sent, the source chunk and the answer (or the error);
``<chunk>.prompt.md`` holds the same two prompts in a form a human can read. That
is what makes a bad translation debuggable after the fact.

``limit`` translates only the first N chunks, and ``resume`` (``skip_if_present``)
reuses the translations an earlier run already wrote — both are there so a book
can be tested and a killed run can be continued without paying twice.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..chunk_glossary.matching import clean_term, load_glossary, match_key
from ..chunk_glossary.runner import (
    ChunkInput,
    chunk_glossary_filename,
    discover_chunk_inputs,
    resolve_chunks_dir,
    resolve_glossary_path,
)
from ..chunking.runner import load_chunk_manifest
from ..core.constants import (
    CHAPTERS_DIRNAME,
    CHUNK_GLOSSARY_DIRNAME,
    DEFAULT_SOURCE_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
    DEFAULT_TRANSLATION_ATTEMPTS,
    DEFAULT_TRANSLATION_CONTEXT_CHARS,
    DEFAULT_TRANSLATION_MAX_TOKENS,
    GLOSSARY_DIRNAME,
    GLOSSARY_FILENAME,
    TRANSLATION_GLOSSARY_SCOPES,
    TRANSLATION_MANIFEST_FILENAME,
    TRANSLATION_METHODS,
    TRANSLATION_TRACE_FILENAME,
)
from ..core.errors import ConfigError, ToolError
from ..documents.chapters import (
    chunk_glossary_dir,
    load_split_manifest,
    translation_prompts_dir,
    translations_dir,
)
from ..glossary.extract import TokenUsage
from .prompts import build_chunk_prompt, build_system_prompt
from .translator import ChunkTranslator, make_translator

#: Monotonic clock used for the timings; injectable so tests can be exact.
Clock = Callable[[], float]
#: Called once per chunk as ``(chunk_record, completed, total)``.
ProgressHook = Callable[[dict[str, Any], int, int], None]
#: Called *before* a chunk's model call, with where it sits in the run.
ChunkStartHook = Callable[[dict[str, Any]], None]


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GlossaryRow:
    """One binding rendering, as the prompt consumes it."""

    term: str
    translation: str
    category: str = "terminology"

    def to_dict(self) -> dict[str, str]:
        return {
            "term": self.term,
            "translation": self.translation,
            "category": self.category,
        }


@dataclass
class ChunkTranslation:
    """One chunk's translation: what it cost, where it went, and why it failed."""

    chunk: ChunkInput
    terms: list[GlossaryRow] = field(default_factory=list)
    glossary_missing: bool = False
    context_chars: int = 0
    source_chars: int = 0
    translation_chars: int = 0
    translation_sha256: str = ""
    attempts: int = 0
    method: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    elapsed_seconds: float = 0.0
    output_path: str | None = None
    prompt_path: str | None = None
    debug_path: str | None = None
    ok: bool = True
    error: str | None = None
    skipped: bool = False
    reused: bool = False
    log: list[str] = field(default_factory=list)
    #: The translated text. Kept out of every serialised record (it is the file).
    translation: str = field(default="", repr=False, compare=False)
    #: The accepted answer as the model wrote it, clipped. For the trace only.
    raw_answer: str = field(default="", repr=False, compare=False)
    #: The full prompt/input payload written to the debug folder.
    debug: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def glossary_term_count(self) -> int:
        return len(self.terms)

    def record(self) -> dict[str, Any]:
        """The flat per-chunk record stored in the manifest."""
        return {
            **self.chunk.to_dict(),
            "glossary_terms": self.glossary_term_count,
            "glossary_missing": self.glossary_missing,
            "context_chars": self.context_chars,
            "source_chars": self.source_chars,
            "translation_chars": self.translation_chars,
            "translation_sha256": self.translation_sha256,
            "attempts": self.attempts,
            "method": self.method,
            "tokens": self.usage.to_dict(),
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "output_path": self.output_path,
            "prompt_path": self.prompt_path,
            "debug_path": self.debug_path,
            "ok": self.ok,
            "error": self.error,
            "skipped": self.skipped,
            "reused": self.reused,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.record(),
            "terms": [term.to_dict() for term in self.terms],
            "log": list(self.log),
            "raw_answer": self.raw_answer,
        }


@dataclass
class TranslationResult:
    """Outcome of a whole stage-7 run."""

    ok: bool
    chunks_path: str
    chunk_glossary_path: str
    output_dir: str
    prompts_dir: str
    source_language: str = DEFAULT_SOURCE_LANGUAGE
    target_language: str = DEFAULT_TARGET_LANGUAGE
    book_title: str = ""
    source_file: str = ""
    translator: str = ""
    provider: str = ""
    model: str = ""
    glossary_scope: str = "chunk"
    glossary_entries: int = 0
    chunk_count: int = 0
    ok_count: int = 0
    processed_count: int = 0
    skipped_count: int = 0
    reused_count: int = 0
    dry_run: bool = False
    save: bool = True
    resumed: bool = False
    limited: bool = False
    settings: dict[str, Any] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)
    totals: dict[str, int] = field(default_factory=dict)
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
def resolve_chunk_glossary_dir(path: str | Path, chunks_path: str | Path) -> Path:
    """Resolve a per-chunk glossary folder: explicit, or the chunks' sibling."""
    if not path:
        return chunk_glossary_dir(resolve_chunks_dir(chunks_path))
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate.parent
    if candidate.name == CHUNK_GLOSSARY_DIRNAME:
        return candidate
    nested = candidate / CHUNK_GLOSSARY_DIRNAME
    if nested.is_dir():
        return nested
    return candidate


def has_chunk_glossary_files(path: str | Path) -> bool:
    """True when ``path`` holds at least one per-chunk glossary file."""
    folder = Path(path).expanduser()
    if not folder.is_dir():
        return False
    return any(
        item.is_file() and item.name.endswith(".json") and not item.name.startswith("_")
        for item in folder.glob("*.json")
    )


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _source_file(chunks_path: Path) -> str:
    """The source document path, from whichever manifest still records it.

    Stage 5's own manifest is the first place to look, but a book chunked from an
    older run may only have stage 1's ``source_path``, and a folder whose
    manifests were lost can still be identified by the glossary's ``book_title``.
    """
    manifest = load_chunk_manifest(chunks_path)
    source_file = str(manifest.get("source_file") or manifest.get("source_path") or "")
    if source_file:
        return source_file
    split = load_split_manifest(chunks_path.parent / CHAPTERS_DIRNAME)
    source_file = str(split.get("source_file") or split.get("source_path") or "")
    if source_file:
        return source_file
    glossary_file = chunks_path.parent / GLOSSARY_DIRNAME / GLOSSARY_FILENAME
    if glossary_file.is_file():
        try:
            payload = json.loads(glossary_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ""
        if isinstance(payload, dict):
            return str(payload.get("source_file") or "")
    return ""


def _book_title(chunks_path: Path, book_title: str = "") -> str:
    """The book's name for the prompt: explicit, the source file, or the folder."""
    if book_title.strip():
        return book_title.strip()
    source_file = _source_file(chunks_path)
    name = Path(source_file).name if source_file else chunks_path.parent.name
    for suffix in (".md", ".txt", ".markdown"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
    return name.replace("_", " ").strip()


def load_chunk_glossary_rows(path: str | Path) -> list[GlossaryRow] | None:
    """The binding rows of one chunk, or ``None`` when the file is missing.

    ``matched_terms`` is stage 6's accepted retrieval and the authoritative list.
    The glossary term is used as the row's term (not the inflected NER mention):
    a prompt that lists ``Džordža → Džordž`` teaches the translator the wrong
    citation form.
    """
    file = Path(path).expanduser()
    if not file.is_file():
        return None
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    raw = payload.get("matched_terms")
    if not isinstance(raw, list):
        raw = []
    rows: list[GlossaryRow] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        term = clean_term(item.get("glossary_term") or item.get("term"))
        translation = " ".join(str(item.get("translation") or "").split())
        if not term or not translation:
            continue
        key = match_key(term)
        if not key or key in seen:
            continue
        seen.add(key)
        rows.append(
            GlossaryRow(
                term=term,
                translation=translation,
                category=str(item.get("category") or "terminology"),
            )
        )
    return rows


def load_book_glossary_rows(glossary_path: str | Path) -> list[GlossaryRow]:
    """The whole book dictionary, for ``glossary_scope="book"``."""
    rows: list[GlossaryRow] = []
    seen: set[str] = set()
    for entry in load_glossary(glossary_path):
        key = match_key(entry.term)
        if not key or key in seen:
            continue
        seen.add(key)
        rows.append(
            GlossaryRow(
                term=clean_term(entry.term),
                translation=" ".join(str(entry.translation).split()),
                category=entry.category or "terminology",
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def run_translation(
    chunks_path: str,
    output_dir: str | None = None,
    *,
    prompts_dir: str | Path | None = None,
    chunk_glossary_path: str | Path | None = None,
    glossary_path: str | Path | None = None,
    glossary_scope: str = "chunk",
    source_language: str = DEFAULT_SOURCE_LANGUAGE,
    target_language: str = DEFAULT_TARGET_LANGUAGE,
    book_title: str = "",
    translator: ChunkTranslator | None = None,
    settings: Any = None,
    settings_factory: Callable[[], Any] | None = None,
    max_attempts: int = DEFAULT_TRANSLATION_ATTEMPTS,
    method: str = "auto",
    max_tokens: int | None = DEFAULT_TRANSLATION_MAX_TOKENS,
    context_chars: int = DEFAULT_TRANSLATION_CONTEXT_CHARS,
    save: bool = True,
    dry_run: bool = False,
    limit: int | None = None,
    chapter_files: Sequence[str] | None = None,
    include_matter: bool = False,
    resume: bool = False,
    write_manifest: bool = True,
    write_prompts: bool = True,
    progress: ProgressHook | None = None,
    chunk_start: ChunkStartHook | None = None,
    clock: Clock = time.perf_counter,
) -> TranslationResult:
    """Translate the chunks of one book, in reading order, one call at a time.

    ``translator`` is injectable, which is how the offline tests run the whole
    stage without a provider. A real run builds it from ``settings`` (or
    ``settings_factory``, resolved only when at least one chunk still needs a
    call, so a fully resumed run needs no key).

    ``resume`` reuses a translation an earlier run wrote when the manifest records
    it as successful, the chunk has not changed (its SHA-256 matches), and the
    language pair is the same. ``chapter_files`` restricts the run to chunks
    belonging to those chapter filenames (the chapter-aligned window); ``limit``
    then translates only the first N of the filtered chunks. ``dry_run`` still
    calls the model — the answer *is* the artifact, so there is nothing to
    validate without asking — but writes no file.

    Front and back matter are skipped by default: they are localized by the
    separate :mod:`book_splitter.matter_agent` stage, with a different prompt and
    without the dictionary. Pass ``include_matter=True`` only to reproduce the
    older behaviour for a book whose matter the caller has already handled.
    """
    run_started = clock()
    scope = str(glossary_scope or "chunk").strip().lower()
    if scope not in TRANSLATION_GLOSSARY_SCOPES:
        raise ToolError(
            f"unknown glossary scope {glossary_scope!r}; pick one of "
            f"{', '.join(TRANSLATION_GLOSSARY_SCOPES)}"
        )
    method = str(method or "auto").strip().lower()
    if method not in TRANSLATION_METHODS:
        raise ToolError(
            f"unknown translation method {method!r}; pick one of "
            f"{', '.join(TRANSLATION_METHODS)}"
        )
    attempts = max(1, _int(max_attempts, DEFAULT_TRANSLATION_ATTEMPTS))
    context_size = max(0, _int(context_chars, DEFAULT_TRANSLATION_CONTEXT_CHARS))

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
    # Front and back matter belong to the localization stage that runs before
    # this one. It translated their chunks with a different prompt and without
    # the dictionary, so translating them again here would overwrite that work.
    matter_skipped = 0
    if not include_matter:
        from ..matter.inspect import matter_chapter_names

        matter_names = matter_chapter_names(
            source_dir, source_dir.parent / CHAPTERS_DIRNAME
        )
        if matter_names:
            kept = [
                chunk
                for chunk in all_inputs
                if chunk.chapter_filename not in matter_names
            ]
            matter_skipped = len(all_inputs) - len(kept)
            all_inputs = kept
    # A chunk's identity for resume is its *actual* current text, not the hash a
    # manifest remembered: a chunk file edited on disk must be translated again,
    # even when its old manifest entry still carries the old hash.
    for chunk in all_inputs:
        chunk.sha256 = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
    limited = limit is not None and 0 < _int(limit) < len(all_inputs)
    inputs = all_inputs[: _int(limit)] if limit is not None and _int(limit) > 0 else all_inputs

    out_dir = (
        Path(output_dir).expanduser() if output_dir else translations_dir(source_dir)
    )
    prompt_dir = (
        Path(prompts_dir).expanduser()
        if prompts_dir
        else translation_prompts_dir(out_dir)
    )
    title = _book_title(source_dir, book_title)
    source_file = _source_file(source_dir)

    if not all_inputs:
        # Nothing but front/back matter in the window: the matter stage already
        # wrote those translations. The body stage has no work, and demanding a
        # per-chunk glossary it would never use would be a false failure.
        return TranslationResult(
            ok=True,
            chunks_path=str(source_dir),
            chunk_glossary_path=str(
                resolve_chunk_glossary_dir(chunk_glossary_path or "", source_dir)
            ),
            output_dir=str(out_dir),
            prompts_dir=str(prompt_dir),
            source_language=str(source_language),
            target_language=str(target_language),
            book_title=title,
            source_file=source_file,
            glossary_scope=scope,
            chunk_count=0,
            ok_count=0,
            processed_count=0,
            skipped_count=matter_skipped,
            dry_run=dry_run,
            save=save,
            resumed=False,
            limited=limited,
            settings={
                "glossary_scope": scope,
                "include_matter": bool(include_matter),
                "matter_skipped": matter_skipped,
            },
            tokens={"calls": 0, "chunks": 0},
            totals={"translated_chunks": 0, "matter_skipped": matter_skipped},
            timing={"run_seconds": round(clock() - run_started, 4)},
            chunks=[],
            errors=[],
        )

    resolved_chunk_glossary = resolve_chunk_glossary_dir(
        chunk_glossary_path or "", source_dir
    )
    resolved_glossary = resolve_glossary_path(glossary_path, source_dir)
    if scope == "chunk":
        if not has_chunk_glossary_files(resolved_chunk_glossary):
            raise ToolError(
                f"no per-chunk glossary in {resolved_chunk_glossary} — run "
                "`chunk-glossary` first, or pass --glossary-scope book to send the "
                "whole glossary.json with every chunk"
            )
        book_rows: list[GlossaryRow] = []
    else:
        book_rows = load_book_glossary_rows(resolved_glossary)

    system_prompt = build_system_prompt(source_language, target_language, title)

    # -- resume: which chunks does this run not have to ask about? ---------- #
    # The manifest is read even without ``resume``: its records for chunks this
    # run does not touch are still the only proof those chunks were translated,
    # and a limited or interrupted run must not erase them.
    previous_manifest = load_translation_manifest(out_dir) if not dry_run else {}
    same_pair = bool(
        previous_manifest
        and str(previous_manifest.get("target_language") or "") == str(target_language)
        and str(previous_manifest.get("source_language") or "") == str(source_language)
    )
    resume_enabled = bool(resume and not dry_run and previous_manifest and same_pair)
    previous_files = {
        str(entry.get("filename")): entry
        for entry in previous_manifest.get("files") or []
        if isinstance(entry, dict) and entry.get("filename")
    }
    #: Records from an earlier run that this run may keep. They are carried into
    #: every manifest snapshot, not only the final one, so Ctrl-C after one chunk
    #: does not lose the resume record of the rest of the book. A different
    #: language pair invalidates the whole manifest, so nothing is carried then.
    carried_from = previous_files if same_pair else {}
    reused: list[bool] = [
        _is_reusable(chunk, previous_files.get(chunk.filename), out_dir)
        if resume_enabled
        else False
        for chunk in inputs
    ]
    resumed = bool(resume_enabled)
    pending = [chunk for chunk, is_reused in zip(inputs, reused, strict=True) if not is_reused]

    if translator is None and pending:
        if settings is None and settings_factory is not None:
            settings = settings_factory()
        if settings is None:
            raise ConfigError(
                "the translation stage needs a model: pass settings (provider/model) "
                "or a translator"
            )
        translator = make_translator(
            settings, max_attempts=attempts, method=method, max_tokens=max_tokens
        )
    translator_name = getattr(translator, "name", "injected") if translator is not None else ""

    settings_payload = {
        "glossary_scope": scope,
        "max_attempts": attempts,
        "method": method,
        "max_tokens": int(max_tokens) if max_tokens else None,
        "context_chars": context_size,
        "save": bool(save),
        "write_prompts": bool(write_prompts),
        "limit": int(limit) if limit is not None else None,
        "chapter_files": list(chapter_files) if chapter_files is not None else None,
        "include_matter": bool(include_matter),
        "matter_skipped": matter_skipped,
        "resume": bool(resume),
        "json_only": bool(getattr(translator, "json_only", False)),
    }

    if pending and not dry_run and save:
        out_dir.mkdir(parents=True, exist_ok=True)
        if write_prompts:
            prompt_dir.mkdir(parents=True, exist_ok=True)

    records: list[ChunkTranslation] = []
    totals = (
        _previous_usage(previous_manifest)
        if resume_enabled
        else TokenUsage()
    )
    errors: list[str] = []
    previous_translation = _last_reused_translation(inputs, reused, out_dir)
    completed = 0

    for position, (chunk, is_reused) in enumerate(zip(inputs, reused, strict=True), start=1):
        if scope == "chunk":
            rows = load_chunk_glossary_rows(
                resolved_chunk_glossary / chunk_glossary_filename(chunk.filename)
            )
            glossary_missing = rows is None
            terms = rows or []
        else:
            terms = list(book_rows)
            glossary_missing = False

        context_text = _context_tail(previous_translation, context_size)
        record = ChunkTranslation(
            chunk=chunk,
            terms=terms,
            glossary_missing=glossary_missing,
            context_chars=len(context_text),
            source_chars=len(chunk.text),
        )
        records.append(record)
        if glossary_missing:
            record.log.append(
                f"glossary: no file for this chunk in {resolved_chunk_glossary}; "
                "translating without it"
            )

        if is_reused:
            record.reused = True
            _load_existing_translation(record, out_dir / chunk.filename)
            previous_translation = record.translation or previous_translation
            completed += 1
            _snapshot_manifest(
                out_dir,
                records,
                source_language=source_language,
                target_language=target_language,
                tokens=totals,
                enabled=write_manifest and save and not dry_run,
            )
            if progress is not None:
                with contextlib.suppress(Exception):
                    progress(record.to_dict(), completed, len(inputs))
            continue

        if not chunk.text.strip():
            record.translation = chunk.text
            _finish_record(record, record.translation)
            if not dry_run and save:
                _write_translation_file(out_dir, record)
            record.log.append("empty chunk — nothing to translate")
            completed += 1
            _snapshot_manifest(
                out_dir,
                records,
                source_language=source_language,
                target_language=target_language,
                tokens=totals,
                enabled=write_manifest and save and not dry_run,
            )
            if progress is not None:
                with contextlib.suppress(Exception):
                    progress(record.to_dict(), completed, len(inputs))
            continue

        user_prompt = build_chunk_prompt(
            chunk_index=position,
            total_chunks=len(inputs),
            chunk_filename=chunk.filename,
            chapter_filename=chunk.chapter_filename,
            source_text=chunk.text,
            source_language=source_language,
            target_language=target_language,
            terms=terms,
            context_text=context_text,
            book_title=title,
        )
        record.debug = _debug_payload(
            record,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context_text=context_text,
            source_language=source_language,
            target_language=target_language,
            scope=scope,
            chunk_glossary_path=str(resolved_chunk_glossary),
            glossary_path=str(resolved_glossary),
            provider=str(getattr(settings, "provider", "") or ""),
            model=str(getattr(settings, "model", "") or ""),
            translator=translator_name,
        )
        if not dry_run and save and write_prompts:
            _write_prompt_files(prompt_dir, record)

        if chunk_start is not None:
            with contextlib.suppress(Exception):
                chunk_start(
                    {
                        "index": position,
                        "total": len(inputs),
                        "done": completed,
                        "filename": chunk.filename,
                        "chapter_filename": chunk.chapter_filename,
                        "tokens": totals.to_dict(),
                    }
                )

        call_started = clock()
        try:
            if translator is None:  # pragma: no cover - guarded above
                raise ConfigError("no translator was provided for the translation stage")
            outcome = translator.translate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                source_text=chunk.text,
            )
            record.attempts = int(outcome.attempts)
            record.method = outcome.method
            record.usage = outcome.usage
            record.raw_answer = outcome.raw_text
            totals += outcome.usage
            record.translation = preserve_trailing_newlines(outcome.text, chunk.text)
            if outcome.error:
                record.log.append(f"retried after: {outcome.error}")
            if outcome.recovered:
                record.log.append("recovered the translation from malformed JSON")
            _finish_record(record, record.translation)
            if not dry_run and save:
                _write_translation_file(out_dir, record)
                if write_prompts:
                    _write_prompt_files(prompt_dir, record)
            record.log.append(
                f"translated {record.source_chars:,} -> {record.translation_chars:,} chars "
                f"with {int(outcome.usage.total_tokens):,} token(s), "
                f"{outcome.attempts} attempt(s)"
            )
            previous_translation = record.translation
        except Exception as exc:  # one bad chunk must not stop the book
            record.ok = False
            record.error = f"{type(exc).__name__}: {exc}"
            # A failed chunk still spent its attempts. The translator carries the
            # usage on the exception so the run's token total stays honest.
            failed_usage = getattr(exc, "usage", None)
            if isinstance(failed_usage, TokenUsage):
                record.usage = failed_usage
                totals += failed_usage
            record.attempts = int(getattr(exc, "attempts", 0) or 0)
            errors.append(f"{chunk.filename}: {record.error}")
            record.log.append(f"translation failed: {record.error}")
            if not dry_run and save:
                _update_debug_answer(record)
                if write_prompts:
                    _write_prompt_files(prompt_dir, record)
        record.elapsed_seconds = clock() - call_started
        completed += 1
        _snapshot_manifest(
            out_dir,
            records,
            source_language=source_language,
            target_language=target_language,
            tokens=totals,
            enabled=write_manifest and save and not dry_run,
            extra_files=_carried_records(
                carried_from, {record.chunk.filename for record in records}, out_dir
            ),
        )
        if progress is not None:
            with contextlib.suppress(Exception):
                progress(record.to_dict(), completed, len(inputs))

    totals_payload = _totals(records)
    # A provider that rejected the schema mid-run leaves the translator in JSON
    # mode; recording it here is what lets the summary say so instead of leaving
    # a run's fallback invisible.
    settings_payload["json_only"] = bool(getattr(translator, "json_only", False))
    result = TranslationResult(
        ok=bool(records) and all(record.ok for record in records) and not errors,
        chunks_path=str(source_dir),
        chunk_glossary_path=str(resolved_chunk_glossary),
        output_dir=str(out_dir),
        prompts_dir=str(prompt_dir) if write_prompts else "",
        source_language=str(source_language),
        target_language=str(target_language),
        book_title=title,
        source_file=source_file,
        translator=translator_name,
        provider=str(getattr(settings, "provider", "") or ""),
        model=str(getattr(settings, "model", "") or ""),
        glossary_scope=scope,
        glossary_entries=len(book_rows) if scope == "book" else 0,
        chunk_count=len(records),
        ok_count=sum(1 for record in records if record.ok),
        processed_count=len(pending),
        skipped_count=matter_skipped,
        reused_count=sum(1 for record in records if record.reused),
        dry_run=dry_run,
        save=save,
        resumed=resumed,
        limited=limited,
        settings=settings_payload,
        tokens={**totals.to_dict(), "chunks": len(pending)},
        totals=totals_payload,
        timing=_timing(records, run_seconds=clock() - run_started),
        chunks=[record.record() for record in records],
        errors=errors,
    )
    if write_manifest and not dry_run and save:
        # A run limited to a smaller prefix than the previous one, or one that
        # replaces only its own window, must not drop the manifest entries of the
        # chunks it did not touch: their files are still valid and the next
        # resume reads this file.
        carried = _carried_records(
            carried_from, {str(record.chunk.filename) for record in records}, out_dir
        )
        result.manifest_path = str(
            write_translation_manifest(out_dir, result, records, extra_files=carried)
        )
        result.trace_path = str(write_translation_trace(out_dir, result, records))
    return result


# --------------------------------------------------------------------------- #
# resume helpers
# --------------------------------------------------------------------------- #
def _is_reusable(chunk: ChunkInput, previous: dict[str, Any] | None, out_dir: Path) -> bool:
    """True when an earlier run's translation can stand in for this chunk.

    The manifest is the record of *why* a file can be trusted: a file that exists
    but is not recorded as successful (or was translated from a different chunk,
    or an older chunk hash) is translated again. A file with no manifest is not
    reused at all — a stray empty file would otherwise become a silent hole.
    """
    if not previous or not previous.get("ok"):
        return False
    recorded_sha = str(previous.get("sha256") or "")
    if recorded_sha and recorded_sha != chunk.sha256:
        return False
    path = out_dir / chunk.filename
    return path.is_file() and path.stat().st_size > 0


def _previous_usage(manifest: dict[str, Any]) -> TokenUsage:
    tokens = manifest.get("tokens") or {}
    return TokenUsage(
        input_tokens=_int(tokens.get("input_tokens")),
        output_tokens=_int(tokens.get("output_tokens")),
        total_tokens=_int(tokens.get("total_tokens")),
        calls=_int(tokens.get("calls")),
    )


def _last_reused_translation(
    inputs: Sequence[ChunkInput], reused: Sequence[bool], out_dir: Path
) -> str:
    """The last reused translation before the first pending chunk, for context."""
    text = ""
    for chunk, is_reused in zip(inputs, reused, strict=True):
        if not is_reused:
            break
        path = out_dir / chunk.filename
        with contextlib.suppress(OSError):
            text = path.read_text(encoding="utf-8")
    return text


def _load_existing_translation(record: ChunkTranslation, path: Path) -> None:
    record.output_path = str(path)
    try:
        record.translation = path.read_text(encoding="utf-8")
    except OSError as exc:
        record.ok = False
        record.error = f"{type(exc).__name__}: {exc}"
        record.log.append(f"could not read the reused translation: {record.error}")
        return
    record.translation_chars = len(record.translation)
    record.translation_sha256 = hashlib.sha256(record.translation.encode("utf-8")).hexdigest()
    record.method = "reused"
    record.log.append("reused: the chunk is already translated and unchanged")


def _context_tail(text: str, size: int) -> str:
    """The last ``size`` characters of the previous translation, cut at a word."""
    if size <= 0 or not text:
        return ""
    if len(text) <= size:
        return text
    tail = text[-size:]
    newline = tail.find("\n")
    if 0 <= newline < size // 2:
        return tail[newline + 1 :]
    space = tail.find(" ")
    if 0 <= space < size // 2:
        return tail[space + 1 :]
    return tail


def _finish_record(record: ChunkTranslation, translation: str) -> None:
    record.translation_chars = len(translation)
    record.translation_sha256 = hashlib.sha256(translation.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# output helpers
# --------------------------------------------------------------------------- #
def preserve_trailing_newlines(translation: str, source: str) -> str:
    """Give the translation the source chunk's trailing newlines.

    Chunks are concatenated in filename order, so the *edges* matter: a chunk
    that ended with a blank line must still end with one in the translation, or
    two paragraphs merge when the files are joined. The model's own trailing
    newlines are normalised to the source's, which is the only count that keeps
    the reconstruction identical in shape.
    """
    stripped = translation.rstrip("\n")
    if not stripped:
        return translation
    trailing = source[len(source.rstrip("\n")) :]
    return stripped + trailing


def prompt_filename(chunk_filename: str) -> str:
    """``00001-x.md`` -> ``00001-x.md.prompt.md`` (the human-readable prompt)."""
    name = Path(str(chunk_filename or "")).name or "chunk.md"
    return f"{name}.prompt.md"


def debug_filename(chunk_filename: str) -> str:
    """``00001-x.md`` -> ``00001-x.md.json`` (the full prompt/input record)."""
    name = Path(str(chunk_filename or "")).name or "chunk.md"
    return f"{name}.json"


def _write_translation_file(out_dir: Path, record: ChunkTranslation) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / record.chunk.filename
    path.write_text(record.translation, encoding="utf-8")
    record.output_path = str(path)
    return path


def _prompt_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    """The messages exactly as the model receives them, for the debug file."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _prompt_sha256(system_prompt: str, user_prompt: str) -> str:
    payload = f"{system_prompt}\n\u0000\n{user_prompt}".encode()
    return hashlib.sha256(payload).hexdigest()


def _debug_payload(
    record: ChunkTranslation,
    *,
    system_prompt: str,
    user_prompt: str,
    context_text: str,
    source_language: str,
    target_language: str,
    scope: str,
    chunk_glossary_path: str,
    glossary_path: str,
    provider: str,
    model: str,
    translator: str,
) -> dict[str, Any]:
    """Everything one model call was made of: the whole input, plus its answer."""
    return {
        "stage": "translate",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "chunk": record.chunk.to_dict(),
        "language": {"source": source_language, "target": target_language},
        "provider": provider,
        "model": model,
        "translator": translator,
        "glossary": {
            "scope": scope,
            "chunk_glossary_path": chunk_glossary_path,
            "glossary_path": glossary_path,
            "missing": record.glossary_missing,
            "entry_count": record.glossary_term_count,
            "entries": [term.to_dict() for term in record.terms],
        },
        "context": {"chars": len(context_text), "text": context_text},
        "prompt": {
            "system": system_prompt,
            "user": user_prompt,
            "messages": _prompt_messages(system_prompt, user_prompt),
            "chars": {
                "system": len(system_prompt),
                "user": len(user_prompt),
                "total": len(system_prompt) + len(user_prompt),
            },
            "sha256": _prompt_sha256(system_prompt, user_prompt),
        },
        "source_chunk": {**record.chunk.to_dict(), "text": record.chunk.text},
        "answer": None,
        "raw_answer": "",
        "ok": record.ok,
        "error": record.error,
        "output_path": record.output_path,
    }


def _update_debug_answer(record: ChunkTranslation) -> None:
    """Fold the accepted (or failed) answer into the payload before writing it."""
    if not record.debug:
        return
    record.debug["answer"] = (
        {
            "text": record.translation,
            "chars": record.translation_chars,
            "sha256": record.translation_sha256,
            "attempts": record.attempts,
            "method": record.method,
            "error": record.error,
            "usage": record.usage.to_dict(),
        }
        if record.attempts or record.translation
        else None
    )
    record.debug["raw_answer"] = record.raw_answer
    record.debug["ok"] = record.ok
    record.debug["error"] = record.error
    record.debug["output_path"] = record.output_path


def _write_prompt_files(prompt_dir: Path, record: ChunkTranslation) -> tuple[Path, Path]:
    """Write both halves of the debug trail: the readable prompt and the JSON."""
    prompt_dir.mkdir(parents=True, exist_ok=True)
    _update_debug_answer(record)
    payload = record.debug or {}
    debug_path = prompt_dir / debug_filename(record.chunk.filename)
    debug_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    record.debug_path = str(debug_path)

    prompt = payload.get("prompt") or {}
    system_prompt = str(prompt.get("system") or "")
    user_prompt = str(prompt.get("user") or "")
    readable = (
        f"<!-- book_splitter · translate · {record.chunk.filename} · "
        f"{payload.get('generated_at') or ''} -->\n"
        f"<!-- prompt sha256: {prompt.get('sha256') or ''} -->\n\n"
        f"# System prompt\n\n{system_prompt}\n\n"
        f"# User prompt\n\n{user_prompt}\n"
    )
    prompt_path = prompt_dir / prompt_filename(record.chunk.filename)
    prompt_path.write_text(readable, encoding="utf-8")
    record.prompt_path = str(prompt_path)
    payload["output_path"] = record.output_path
    payload["debug_path"] = record.debug_path
    payload["prompt_path"] = record.prompt_path
    return prompt_path, debug_path


# --------------------------------------------------------------------------- #
# bookkeeping
# --------------------------------------------------------------------------- #
def _totals(records: Sequence[ChunkTranslation]) -> dict[str, int]:
    called = [record for record in records if record.attempts]
    return {
        "chunks": len(records),
        "ok_chunks": sum(1 for record in records if record.ok),
        "failed_chunks": sum(1 for record in records if not record.ok),
        "reused_chunks": sum(1 for record in records if record.reused),
        "translated_chunks": len(called),
        "source_chars": sum(record.source_chars for record in records),
        "translation_chars": sum(record.translation_chars for record in records),
        "glossary_rows": sum(record.glossary_term_count for record in records),
        "missing_glossary_chunks": sum(1 for record in records if record.glossary_missing),
        "attempts": sum(record.attempts for record in records),
    }


def _timing(records: Sequence[ChunkTranslation], *, run_seconds: float) -> dict[str, float]:
    called = [record for record in records if record.attempts]
    model_seconds = sum(record.elapsed_seconds for record in called)
    return {
        "run_seconds": round(float(run_seconds), 4),
        "model_seconds": round(float(model_seconds), 4),
        "seconds_per_chunk": round(model_seconds / len(called), 4) if called else 0.0,
        "model_calls": sum(record.attempts for record in records),
    }


def _carried_records(
    previous: dict[str, dict[str, Any]],
    current: set[str],
    out_dir: Path,
) -> list[dict[str, Any]]:
    """Earlier records this run does not replace, for the manifest to keep.

    A record is kept only when this run did not write a record for the same
    filename, it was successful, and its translation file is still there. Those
    are exactly the chunks whose saved translation remains valid while a limited
    or interrupted run is in progress.
    """
    carried: list[dict[str, Any]] = []
    for filename, entry in previous.items():
        if filename in current or not entry.get("ok"):
            continue
        path = out_dir / filename
        if path.is_file() and path.stat().st_size > 0:
            carried.append(entry)
    return carried


def _snapshot_manifest(
    out_dir: Path,
    records: Sequence[ChunkTranslation],
    *,
    source_language: str,
    target_language: str,
    tokens: TokenUsage,
    enabled: bool,
    extra_files: Sequence[dict[str, Any]] = (),
) -> None:
    """Rewrite a resumable manifest after every finished chunk.

    A translation run is dozens to hundreds of paid calls. Without a snapshot,
    Ctrl-C (or a killed terminal) would leave the translated files on disk but no
    record of them, and ``--resume`` would translate the whole book again. The
    snapshot is the small subset of the final manifest that resume reads —
    language pair, token totals and one record per finished chunk — so the cost
    of an interruption is the chunk in flight, not the run. ``extra_files``
    keeps the records of chunks outside this run's window, so a limited or
    interrupted run cannot lose them either.
    """
    if not enabled or not records:
        return
    payload = {
        "stage": "translate",
        "partial": True,
        "source_language": source_language,
        "target_language": target_language,
        "tokens": tokens.to_dict(),
        "files": [record.record() for record in records] + list(extra_files),
    }
    with contextlib.suppress(OSError):  # a snapshot must never end the run
        path = Path(out_dir) / TRANSLATION_MANIFEST_FILENAME
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_translation_manifest(
    out_dir: str | Path,
    result: TranslationResult,
    records: Sequence[ChunkTranslation] | None = None,
    *,
    extra_files: Sequence[dict[str, Any]] = (),
) -> Path:
    """Write ``_translation_manifest.json``: the book-level index of the run."""
    path = Path(out_dir) / TRANSLATION_MANIFEST_FILENAME
    files = (
        [record.record() for record in records]
        if records is not None
        else list(result.chunks)
    )
    files.extend(extra_files)
    payload = {
        "stage": "translate",
        "ok": result.ok,
        "dry_run": result.dry_run,
        "resumed": result.resumed,
        "limited": result.limited,
        "book_title": result.book_title,
        "source_file": result.source_file,
        "source_language": result.source_language,
        "target_language": result.target_language,
        "translator": result.translator,
        "provider": result.provider,
        "model": result.model,
        "chunks_path": result.chunks_path,
        "chunk_glossary_path": result.chunk_glossary_path,
        "output_dir": result.output_dir,
        "prompts_dir": result.prompts_dir,
        "glossary_scope": result.glossary_scope,
        "glossary_entries": result.glossary_entries,
        "settings": result.settings,
        "chunk_count": result.chunk_count,
        "ok_count": result.ok_count,
        "processed_count": result.processed_count,
        "reused_count": result.reused_count,
        "tokens": result.tokens,
        "totals": result.totals,
        "timing": result.timing,
        "files": files,
        "errors": result.errors,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_translation_trace(
    out_dir: str | Path,
    result: TranslationResult,
    records: Sequence[ChunkTranslation] | None = None,
) -> Path:
    """Write the audit trail: the run plus each chunk's terms, log and raw answer."""
    path = Path(out_dir) / TRANSLATION_TRACE_FILENAME
    payload = result.to_dict()
    if records is not None:
        payload["detail"] = [record.to_dict() for record in records]
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return path


def load_translation_manifest(translations_path: str | Path) -> dict[str, Any]:
    """Read ``_translation_manifest.json`` from an output folder (or a file)."""
    path = Path(translations_path).expanduser()
    manifest = path / TRANSLATION_MANIFEST_FILENAME if path.is_dir() else path
    if not manifest.is_file():
        return {}
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


__all__ = [
    "ChunkStartHook",
    "ChunkTranslation",
    "Clock",
    "GlossaryRow",
    "ProgressHook",
    "TranslationResult",
    "debug_filename",
    "has_chunk_glossary_files",
    "load_book_glossary_rows",
    "load_chunk_glossary_rows",
    "load_translation_manifest",
    "preserve_trailing_newlines",
    "prompt_filename",
    "resolve_chunk_glossary_dir",
    "run_translation",
    "write_translation_manifest",
    "write_translation_trace",
]
