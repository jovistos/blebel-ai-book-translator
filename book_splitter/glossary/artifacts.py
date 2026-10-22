"""Evidence lookup, per-chapter extraction and the glossary artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..core.artifacts import (
    read_json,
    read_manifest,
    timestamp,
    write_json,
    write_manifest,
    write_text,
    write_trace,
)
from ..core.constants import (
    GLOSSARY_FILENAME,
    GLOSSARY_MANIFEST_FILENAME,
    GLOSSARY_MARKDOWN_FILENAME,
    GLOSSARY_TRACE_FILENAME,
)
from ..core.errors import ConfigError, ExtractionError, ToolError
from ..core.llm import TokenUsage
from ..documents.chapters import reduced_dir, unique_dir
from ..documents.io import load_lines
from .extract import GlossaryExtractor
from .prompts import build_chapter_prompt
from .records import (
    ChapterGlossary,
    ChapterPlan,
    Clock,
    GlossaryBook,
    GlossaryResult,
    GlossaryTerm,
)


# --------------------------------------------------------------------------- #
# finding the evidence
# --------------------------------------------------------------------------- #
def resolve_sentences_dir(
    chapters_path: Path, *, sentences_dir: str | None, sentences: str
) -> Path:
    """Where the sentences to mine come from: explicit, or the sibling folder.

    ``unique`` prefers ``03-reduced-unique/`` and falls back to ``03-reduced/`` when the
    first-mention folder was never written (a stage-3 run with
    ``--no-save-unique``). ``reduced`` goes straight to ``03-reduced/``. Either way
    the folder has to hold at least one chapter file, or the run says which folder
    it looked in instead of silently building an empty dictionary.
    """
    if sentences_dir:
        folder = Path(sentences_dir).expanduser()
        if not _has_markdown(folder):
            raise ToolError(f"no sentence files in {folder}")
        return folder
    wanted = str(sentences or "unique").strip().lower()
    if wanted not in {"unique", "reduced"}:
        raise ToolError(f"unknown sentences source {sentences!r}; use 'unique' or 'reduced'")
    candidates = (
        [unique_dir(chapters_path), reduced_dir(chapters_path)]
        if wanted == "unique"
        else [reduced_dir(chapters_path)]
    )
    for folder in candidates:
        if _has_markdown(folder):
            return folder
    looked = " and ".join(str(folder) for folder in candidates)
    raise ToolError(
        f"no reduced sentences found in {looked} — run `dictionary` first, "
        "or point --sentences-in at the folder that holds them"
    )


def _has_markdown(folder: Path) -> bool:
    return folder.is_dir() and any(
        path.is_file() and not path.name.startswith("_") for path in folder.glob("*.md")
    )


def _book_title(source_file: str, chapters_path: Path) -> str:
    """The book's name, for the prompt: the source filename without its suffix."""
    name = Path(source_file).name if source_file else chapters_path.parent.name
    for suffix in (".md", ".txt", ".markdown"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
    return name.replace("_", " ").strip()


def chunk_sentences(sentences: Sequence[str], limit: int) -> list[list[str]]:
    """Split a chapter's sentences into model-sized parts, on line boundaries.

    Only a chapter whose reduced file is larger than ``limit`` is split at all —
    the reduced folders are small enough that most chapters are one call. The
    limit exists so that a single enormous chapter cannot overflow a context
    window and fail: the parts are processed in order and the dictionary is
    carried from one to the next, exactly as it is carried between chapters.
    """
    lines = [line.rstrip() for line in sentences if line.strip()]
    if not lines:
        return [[]]
    if limit <= 0:
        return [lines]
    chunks: list[list[str]] = []
    current: list[str] = []
    size = 0
    for line in lines:
        if current and size + len(line) + 1 > limit:
            chunks.append(current)
            current = []
            size = 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append(current)
    return chunks


# --------------------------------------------------------------------------- #
# one chapter
# --------------------------------------------------------------------------- #
def _extract_chapter(
    plan: ChapterPlan,
    *,
    sentences_path: Path,
    extractor: GlossaryExtractor | None,
    system_prompt: str,
    book: GlossaryBook,
    chunk_chars: int,
    max_context_entries: int,
    clock: Clock,
) -> ChapterGlossary:
    """Read one reduced chapter, ask the model about it, fold the answer in.

    The dictionary is passed to the prompt *as it is at the start of each part*,
    so a long chapter's later parts see the entities its earlier parts just locked
    in — the same compatibility rule as between chapters, applied one level down.
    """
    chapter = plan.chapter
    chapter_started = clock()
    path = Path(sentences_path) / chapter.filename
    record = ChapterGlossary(
        index=plan.index,
        body_index=plan.body_index,
        chapter_id=chapter.chapter_id,
        filename=chapter.filename,
        title=plan.title,
        kind=plan.kind,
        sentences_path=str(path),
    )
    if not path.is_file():
        record.ok = False
        record.error = f"no reduced chapter at {path}"
        return record
    sentences = load_lines(path)
    chunks = chunk_sentences(sentences, chunk_chars)
    record.output_lines = len([line for line in sentences if line.strip()])
    record.source_chars = sum(len(line) for line in sentences)
    record.chunks = len(chunks)
    for part, chunk in enumerate(chunks, start=1):
        if extractor is None:  # pragma: no cover - guarded by run_glossary
            raise ConfigError("no extractor was provided for the glossary stage")
        user_prompt = build_chapter_prompt(
            chapter_index=plan.body_index or plan.index,
            chapter_filename=chapter.filename,
            chapter_title=plan.title,
            sentences=chunk,
            dictionary=book.terms(),
            max_dictionary_entries=max_context_entries,
            part=part,
            parts=len(chunks),
        )
        try:
            outcome = extractor.extract(system_prompt=system_prompt, user_prompt=user_prompt)
        except ExtractionError as exc:
            record.ok = False
            record.error = str(exc)
            record.log.append(f"part {part}/{len(chunks)}: extraction failed: {exc}")
            break
        record.models_called += 1
        record.attempts += int(outcome.attempts)
        record.method = outcome.method or record.method
        record.usage += outcome.usage
        if outcome.error:
            record.log.append(f"part {part}/{len(chunks)}: retried after {outcome.error}")
        if outcome.raw_text:
            record.raw_answers.append(outcome.raw_text)
        # Provenance is the *body* chapter number, the one the prompt used and the
        # one a translator looking at the glossary will count to.
        merge = book.merge(
            outcome.entries,
            chapter_index=plan.body_index or plan.index,
            chapter_filename=chapter.filename,
        )
        record.added += len(merge.added)
        record.repeats += len(merge.repeats)
        record.conflicts += len(merge.conflicts)
        record.added_terms.extend(term.to_dict() for term in merge.added)
        record.conflict_detail.extend(merge.conflicts)
        record.log.append(
            f"part {part}/{len(chunks)}: {len(outcome.entries)} entr"
            f"{'y' if len(outcome.entries) == 1 else 'ies'} returned -> "
            f"{len(merge.added)} new, {len(merge.repeats)} already locked, "
            f"{len(merge.conflicts)} conflicting"
        )
    record.elapsed_seconds = clock() - chapter_started
    return record


def _skipped_record(plan: ChapterPlan) -> ChapterGlossary:
    return ChapterGlossary(
        index=plan.index,
        body_index=plan.body_index,
        chapter_id=plan.chapter.chapter_id,
        filename=plan.chapter.filename,
        title=plan.title,
        kind=plan.kind,
        skipped=True,
        skip_reason=plan.skip_reason,
        log=[f"skipped: {plan.skip_reason}"],
    )


def _reused_record(plan: ChapterPlan, previous: dict[str, Any] | None) -> ChapterGlossary:
    """A chapter a resumed run did not visit, described from the old manifest."""
    record = ChapterGlossary(
        index=plan.index,
        body_index=plan.body_index,
        chapter_id=plan.chapter.chapter_id,
        filename=plan.chapter.filename,
        title=plan.title,
        kind=plan.kind,
        reused=True,
    )
    previous = previous or {}
    record.added = int(previous.get("added") or 0)
    record.repeats = int(previous.get("repeats") or 0)
    record.conflicts = int(previous.get("conflicts") or 0)
    record.output_lines = int(previous.get("output_lines") or 0)
    record.sentences_path = str(previous.get("sentences_path") or "")
    record.log.append("reused: the chapter is already in the glossary manifest")
    return record


# --------------------------------------------------------------------------- #
# resuming
# --------------------------------------------------------------------------- #
def _previous_manifest(out_dir: Path) -> dict[str, Any]:
    return read_manifest(out_dir, GLOSSARY_MANIFEST_FILENAME)


def _resume_state(
    out_dir: Path, previous_manifest: dict[str, Any], *, enabled: bool
) -> tuple[GlossaryBook, TokenUsage, list[dict[str, Any]], set[str]]:
    """The dictionary, cost and finished chapters of an earlier run.

    The entries come from ``glossary.json`` and the bookkeeping from the
    manifest, so a resumed run continues with the same locked renderings and
    reports the *whole* run's token total rather than only its own chapters.
    """
    book = GlossaryBook()
    totals = TokenUsage()
    conflicts: list[dict[str, Any]] = []
    done: set[str] = set()
    if not enabled:
        return book, totals, conflicts, done

    glossary_path = Path(out_dir) / GLOSSARY_FILENAME
    if glossary_path.is_file():
        data = read_json(glossary_path, default={})
        for entry in (data.get("entries") if isinstance(data, dict) else None) or []:
            if isinstance(entry, dict) and entry.get("term"):
                book.lock(GlossaryTerm.from_dict(entry))
    tokens = previous_manifest.get("tokens") or {}
    totals = TokenUsage(
        input_tokens=int(tokens.get("input_tokens") or 0),
        output_tokens=int(tokens.get("output_tokens") or 0),
        total_tokens=int(tokens.get("total_tokens") or 0),
        calls=int(tokens.get("calls") or 0),
    )
    for conflict in previous_manifest.get("conflicts") or []:
        if isinstance(conflict, dict):
            conflicts.append(conflict)
    for entry in previous_manifest.get("files") or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("ok") and not entry.get("skipped"):
            done.add(str(entry.get("filename") or ""))
    return book, totals, conflicts, done


# --------------------------------------------------------------------------- #
# writing the artifacts
# --------------------------------------------------------------------------- #
def glossary_payload(result: GlossaryResult, book: GlossaryBook) -> dict[str, Any]:
    """The dictionary itself — the file the translation stage consumes."""
    return {
        "stage": "glossary",
        "generated_at": timestamp(),
        "book_title": result.book_title,
        "source_file": result.source_file,
        "source_language": result.source_language,
        "target_language": result.target_language,
        "provider": result.provider,
        "model": result.model,
        "chapters_path": result.chapters_path,
        "sentences_dir": result.sentences_dir,
        "chapter_count": result.body_chapter_count,
        "entry_count": len(book),
        "categories": book.categories(),
        "tokens": result.tokens,
        "entries": book.to_list(),
    }


def write_glossary(out_dir: Path, result: GlossaryResult, book: GlossaryBook) -> Path:
    return write_json(Path(out_dir) / GLOSSARY_FILENAME, glossary_payload(result, book))


def write_glossary_markdown(out_dir: Path, result: GlossaryResult, book: GlossaryBook) -> Path:
    """The human-readable view: a table per category, in the order categories fill.

    Meant to be read in a diff or an editor while the book is being translated —
    ``glossary.json`` is the machine's copy, this is the reviewer's.
    """
    lines = [
        f"# Glossary — {result.book_title or 'the book'}",
        "",
        f"{result.source_language} → {result.target_language} · "
        f"{len(book)} entr{'y' if len(book) == 1 else 'ies'} · "
        f"built by {result.provider or '—'}/{result.model or '—'} from "
        f"{result.body_chapter_count} chapter(s)",
        "",
        f"Tokens: {result.tokens.get('total_tokens', 0):,} total "
        f"({result.tokens.get('input_tokens', 0):,} in, "
        f"{result.tokens.get('output_tokens', 0):,} out)",
        "",
    ]
    by_category: dict[str, list[GlossaryTerm]] = {}
    for entry in book:
        by_category.setdefault(entry.category, []).append(entry)
    for category, entries in by_category.items():
        lines.append(f"## {category}")
        lines.append("")
        lines.append("| Term | Translation | First seen |")
        lines.append("| --- | --- | --- |")
        for entry in entries:
            term = entry.term.replace("|", "\\|")
            translation = entry.translation.replace("|", "\\|")
            lines.append(f"| {term} | {translation} | {entry.chapter_filename} |")
        lines.append("")
    return write_text(Path(out_dir) / GLOSSARY_MARKDOWN_FILENAME, "\n".join(lines))


def write_glossary_manifest(
    out_dir: Path,
    result: GlossaryResult,
    records: Sequence[ChapterGlossary] | None = None,
) -> Path:
    """The run's bookkeeping: per chapter, what it cost and what it added."""
    if records is None:
        records = []
    payload = {
        "stage": "glossary",
        "ok": result.ok,
        "dry_run": result.dry_run,
        "limited": result.limited,
        "resumed": result.resumed,
        "book_title": result.book_title,
        "source_file": result.source_file,
        "source_language": result.source_language,
        "target_language": result.target_language,
        "provider": result.provider,
        "model": result.model,
        "extractor": result.extractor,
        "chapters_path": result.chapters_path,
        "sentences_dir": result.sentences_dir,
        "settings": result.settings,
        "chapter_count": result.chapter_count,
        "body_chapter_count": result.body_chapter_count,
        "processed_count": result.processed_count,
        "skipped_count": result.skipped_count,
        "reused_count": result.reused_count,
        "entry_count": result.entry_count,
        "categories": result.categories,
        "tokens": result.tokens,
        "timing": result.timing,
        "conflicts": result.conflicts,
        "files": [record.record() for record in records],
        "errors": result.errors,
    }
    return write_manifest(out_dir, GLOSSARY_MANIFEST_FILENAME, payload)


def write_glossary_trace(
    out_dir: Path,
    result: GlossaryResult,
    records: Sequence[ChapterGlossary] | None = None,
) -> Path:
    """The audit trail: per chapter, the entries it added and the answers it gave."""
    payload = result.to_dict()
    if records is not None:
        payload["detail"] = [record.to_dict() for record in records]
    return write_trace(out_dir, GLOSSARY_TRACE_FILENAME, payload)


def _timing(records: Sequence[ChapterGlossary], *, run_seconds: float) -> dict[str, float]:
    """Wall clock, plus how many provider requests the book actually cost.

    ``model_calls`` counts *attempts*, not chapters: a chapter that was retried
    (or that had to fall back to JSON mode after a provider refusal) made more
    than one request, and that is the number a cost estimate needs.
    """
    called = [record for record in records if record.models_called]
    model_seconds = sum(record.elapsed_seconds for record in called)
    total_attempts = sum(record.attempts for record in records)
    return {
        "run_seconds": round(float(run_seconds), 4),
        "model_seconds": round(float(model_seconds), 4),
        "seconds_per_chapter": round(model_seconds / len(called), 4) if called else 0.0,
        "model_calls": total_attempts,
    }




__all__ = [
    "_extract_chapter",
    "_previous_manifest",
    "_reused_record",
    "_resume_state",
    "_skipped_record",
    "_timing",
    "chunk_sentences",
    "glossary_payload",
    "resolve_sentences_dir",
    "write_glossary",
    "write_glossary_manifest",
    "write_glossary_markdown",
    "write_glossary_trace",
]
