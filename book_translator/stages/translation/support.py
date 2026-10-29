"""Inputs, resume logic and artifacts for the translation stage."""

from __future__ import annotations

import contextlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from book_translator.core.artifacts import (
    sha256_text,
    timestamp,
    write_json,
    write_manifest,
    write_text,
    write_trace,
)
from book_translator.core.coerce import as_int as _int
from book_translator.core.constants import (
    CHAPTERS_DIRNAME,
    CHUNK_GLOSSARY_DIRNAME,
    GLOSSARY_DIRNAME,
    GLOSSARY_FILENAME,
    TRANSLATION_MANIFEST_FILENAME,
    TRANSLATION_TRACE_FILENAME,
)
from book_translator.core.llm import TokenUsage
from book_translator.documents.chapters import (
    chunk_glossary_dir,
)
from book_translator.documents.chunks import (
    ChunkInput,
    chunk_glossary_filename,
    resolve_chunks_dir,
)
from book_translator.documents.manifest import (
    load_chunk_manifest,
    load_split_manifest,
)
from book_translator.documents.text import (
    book_title_from_source,
    preserve_trailing_newlines,
)
from book_translator.stages.chunk_glossary.matching import clean_term, load_glossary, match_key

from .records import (
    ChunkTranslation,
    GlossaryRow,
    TranslationResult,
)
from .translator import ChunkTranslator


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
    return book_title_from_source(_source_file(chunks_path), chunks_path, book_title)


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
# resume helpers
# --------------------------------------------------------------------------- #
def _glossary_fingerprint(
    chunk: ChunkInput,
    *,
    scope: str,
    chunk_glossary_dir: Path,
    glossary_path: Path,
) -> str:
    """A stable digest of the glossary rows this chunk's prompt would receive.

    Chunk scope hashes the chunk's own rows, so a change to one chunk's terms
    invalidates that chunk only; book scope hashes the whole dictionary, which is
    the same input for every chunk. The rows are hashed, not the files: a
    per-chunk JSON also records absolute paths and timestamps that do not change
    the prompt, and hashing them would make a resumed run depend on where the
    book folder lives.
    """
    if scope == "chunk":
        rows = load_chunk_glossary_rows(
            chunk_glossary_dir / chunk_glossary_filename(chunk.filename)
        )
    else:
        rows = load_book_glossary_rows(glossary_path)
    payload = [row.to_dict() for row in rows or []]
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _model_identity(
    settings: Any | None,
    translator: ChunkTranslator | None,
    previous_manifest: dict[str, Any],
    *,
    label: str = "",
) -> str:
    """Who would answer this run's calls, as a stable string.

    The real translator's ``name`` is ``provider/model``. A caller that knows the
    configured label (the pipeline does, without resolving a key) can pass it as
    ``label``. When a fully resumed run has neither, the previous manifest's
    identity stands in: nothing will be called, and treating the old answer as
    the current model is what lets a resume finish without a key.
    """
    if translator is not None:
        return str(getattr(translator, "name", "") or "")
    if label:
        return str(label)
    if settings is not None:
        provider = str(getattr(settings, "provider", "") or "")
        model = str(getattr(settings, "model", "") or "")
        if provider or model:
            return f"{provider}/{model}" if provider and model else provider or model
    recorded = previous_manifest.get("settings") or {}
    return str(recorded.get("model_identity") or "")


def _translation_resume_key(
    chunk: ChunkInput,
    *,
    scope: str,
    glossary_sha256: str,
    context_chars: int,
    method: str,
    model_identity: str,
    max_tokens: int | None,
    source_language: str,
    target_language: str,
) -> str:
    """A digest of every input that changes what the model would be asked."""
    payload = {
        "chunk_sha256": chunk.sha256,
        "glossary_scope": scope,
        "glossary_sha256": glossary_sha256,
        "context_chars": int(context_chars),
        "method": str(method),
        "model": str(model_identity),
        "max_tokens": int(max_tokens) if max_tokens else None,
        "source_language": str(source_language),
        "target_language": str(target_language),
    }
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _is_reusable(
    chunk: ChunkInput,
    previous: dict[str, Any] | None,
    out_dir: Path,
    resume_key: str,
) -> bool:
    """True when an earlier run's translation can stand in for this chunk.

    The manifest is the record of *why* a file can be trusted: a file that exists
    but is not recorded as successful (or was translated from a different chunk,
    an older chunk hash, or with different glossary/context/model settings) is
    translated again. A file with no manifest is not reused at all — a stray
    empty file would otherwise become a silent hole. A manifest written before
    ``resume_key`` existed is also translated again: it does not record the
    settings that produced the text, so reuse cannot be proven safe.
    """
    if not previous or not previous.get("ok"):
        return False
    recorded_sha = str(previous.get("sha256") or "")
    if recorded_sha and recorded_sha != chunk.sha256:
        return False
    if str(previous.get("resume_key") or "") != resume_key:
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
    record.translation_sha256 = sha256_text(record.translation)
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
    record.translation_sha256 = sha256_text(translation)


# --------------------------------------------------------------------------- #
# output helpers
# --------------------------------------------------------------------------- #
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
    write_text(path, record.translation)
    record.output_path = str(path)
    return path


def _prompt_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    """The messages exactly as the model receives them, for the debug file."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _prompt_sha256(system_prompt: str, user_prompt: str) -> str:
    return sha256_text(f"{system_prompt}\n\u0000\n{user_prompt}")


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
        "generated_at": timestamp(),
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
    write_json(debug_path, payload)
    record.debug_path = str(debug_path)

    prompt = payload.get("prompt") or {}
    system_prompt = str(prompt.get("system") or "")
    user_prompt = str(prompt.get("user") or "")
    readable = (
        f"<!-- book_translator · translate · {record.chunk.filename} · "
        f"{payload.get('generated_at') or ''} -->\n"
        f"<!-- prompt sha256: {prompt.get('sha256') or ''} -->\n\n"
        f"# System prompt\n\n{system_prompt}\n\n"
        f"# User prompt\n\n{user_prompt}\n"
    )
    prompt_path = prompt_dir / prompt_filename(record.chunk.filename)
    write_text(prompt_path, readable)
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
        write_manifest(out_dir, TRANSLATION_MANIFEST_FILENAME, payload)


def write_translation_manifest(
    out_dir: str | Path,
    result: TranslationResult,
    records: Sequence[ChunkTranslation] | None = None,
    *,
    extra_files: Sequence[dict[str, Any]] = (),
) -> Path:
    """Write ``_translation_manifest.json``: the book-level index of the run."""
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
    return write_manifest(out_dir, TRANSLATION_MANIFEST_FILENAME, payload)


def write_translation_trace(
    out_dir: str | Path,
    result: TranslationResult,
    records: Sequence[ChunkTranslation] | None = None,
) -> Path:
    """Write the audit trail: the run plus each chunk's terms, log and raw answer."""
    payload = result.to_dict()
    if records is not None:
        payload["detail"] = [record.to_dict() for record in records]
    return write_trace(out_dir, TRANSLATION_TRACE_FILENAME, payload)

