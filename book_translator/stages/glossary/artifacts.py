"""Evidence lookup, per-chapter extraction and the glossary artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from book_translator.core.artifacts import (
    read_json,
    read_manifest,
    timestamp,
    write_json,
    write_manifest,
    write_text,
    write_trace,
)
from book_translator.core.constants import (
    GLOSSARY_CSV_FILENAME,
    GLOSSARY_FILENAME,
    GLOSSARY_MANIFEST_FILENAME,
    GLOSSARY_MARKDOWN_FILENAME,
    GLOSSARY_TRACE_FILENAME,
)
from book_translator.core.errors import ConfigError, ExtractionError, ToolError
from book_translator.core.llm import TokenUsage
from book_translator.documents.chapters import reduced_dir, unique_dir
from book_translator.documents.io import load_lines

from .book import GlossaryBook
from .extract import GlossaryExtractor
from .prompts import build_chapter_prompt
from .records import (
    ChapterGlossary,
    ChapterPlan,
    Clock,
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
    from book_translator.documents.text import book_title_from_source

    return book_title_from_source(source_file, chapters_path)


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
    conflicts.extend(
        conflict
        for conflict in previous_manifest.get("conflicts") or []
        if isinstance(conflict, dict)
    )
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


def write_glossary_csv(out_dir: Path, result: GlossaryResult, book: GlossaryBook) -> Path:
    """The spreadsheet view: term, category, translation, chapter_filename.

    Meant to be inspected or edited by hand; user changes to translations
    can be synced back into the pipeline dictionary.
    """
    import csv

    target = Path(out_dir) / GLOSSARY_CSV_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["term", "category", "translation", "chapter_filename"])
        for entry in book:
            writer.writerow([entry.term, entry.category, entry.translation, entry.chapter_filename])
    return target


def sync_glossary_from_csv(glossary_dir_or_file: Path | str) -> tuple[int, int]:
    """Sync manual edits from glossary.csv into glossary.json and glossary.md.

    Returns (updated_count, added_count).
    """
    import csv

    from .models import entry_key

    path = Path(glossary_dir_or_file)
    out_dir = path if path.is_dir() else path.parent
    csv_file = out_dir / GLOSSARY_CSV_FILENAME
    json_file = out_dir / GLOSSARY_FILENAME
    md_file = out_dir / GLOSSARY_MARKDOWN_FILENAME

    if not csv_file.is_file():
        return 0, 0

    csv_rows: list[dict[str, str]] = []
    try:
        with csv_file.open("r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for r in reader:
                cleaned = {
                    str(k).strip().lower(): str(v).strip()
                    for k, v in r.items()
                    if k is not None and v is not None
                }
                if cleaned.get("term") and cleaned.get("translation"):
                    csv_rows.append(cleaned)
    except Exception:
        return 0, 0

    if not csv_rows and not json_file.is_file():
        return 0, 0

    data: dict[str, Any] = {}
    if json_file.is_file():
        try:
            data = read_json(json_file, default={})
        except Exception:
            data = {}

    raw_entries = data.get("entries") if isinstance(data, dict) else []
    if raw_entries is None or not isinstance(raw_entries, list):
        raw_entries = []

    entries_by_key: dict[str, dict[str, Any]] = {}
    for entry in raw_entries:
        if isinstance(entry, dict) and entry.get("term"):
            k = entry_key(str(entry["term"]))
            if k:
                entries_by_key[k] = entry

    updated_count = 0
    added_count = 0

    for row in csv_rows:
        term = row["term"]
        trans = row["translation"]
        cat = row.get("category", "") or "terminology"
        ch_file = row.get("chapter_filename", "")
        k = entry_key(term)
        if not k:
            continue
        if k in entries_by_key:
            entry = entries_by_key[k]
            changed = False
            if entry.get("translation") != trans:
                entry["translation"] = trans
                changed = True
            if cat and entry.get("category") != cat:
                entry["category"] = cat
                changed = True
            if changed:
                updated_count += 1
        else:
            new_entry = {
                "term": term,
                "category": cat,
                "translation": trans,
                "chapter_index": 0,
                "chapter_filename": ch_file,
                "also_returned_in": [],
            }
            raw_entries.append(new_entry)
            entries_by_key[k] = new_entry
            added_count += 1

    if updated_count > 0 or added_count > 0 or not json_file.is_file():
        data["entries"] = raw_entries
        data["entry_count"] = len(raw_entries)
        cats: dict[str, int] = {}
        for e in raw_entries:
            c = str(e.get("category") or "terminology")
            cats[c] = cats.get(c, 0) + 1
        data["categories"] = dict(sorted(cats.items()))
        write_json(json_file, data)

        book = GlossaryBook()
        for e in raw_entries:
            book.lock(GlossaryTerm.from_dict(e))

        write_text(
            md_file,
            render_glossary_markdown(
                book,
                book_title=str(data.get("book_title") or ""),
                source_language=str(data.get("source_language") or "en"),
                target_language=str(data.get("target_language") or "sr"),
                provider=str(data.get("provider") or ""),
                model=str(data.get("model") or ""),
                chapter_count=int(data.get("chapter_count") or 0),
                tokens=data.get("tokens") or {},
                include_tokens_Only_if_present=True,
            ),
        )

    return updated_count, added_count


def render_glossary_markdown(
    book: GlossaryBook,
    *,
    book_title: str = "",
    source_language: str = "en",
    target_language: str = "sr",
    provider: str = "",
    model: str = "",
    chapter_count: int = 0,
    tokens: dict[str, Any] | None = None,
    include_tokens_Only_if_present: bool = False,
) -> str:
    """Render the human-readable Markdown glossary table per category."""
    lines = [
        f"# Glossary — {book_title or 'the book'}",
        "",
        f"{source_language} → {target_language} · "
        f"{len(book)} entr{'y' if len(book) == 1 else 'ies'} · "
        f"built by {provider or '—'}/{model or '—'} from "
        f"{chapter_count} chapter(s)",
        "",
    ]
    tok = tokens or {}
    if tok or not include_tokens_Only_if_present:
        lines.extend(
            [
                f"Tokens: {tok.get('total_tokens', 0):,} total "
                f"({tok.get('input_tokens', 0):,} in, "
                f"{tok.get('output_tokens', 0):,} out)",
                "",
            ]
        )
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
    return "\n".join(lines)


def write_glossary_markdown(out_dir: Path, result: GlossaryResult, book: GlossaryBook) -> Path:
    """The human-readable view: a table per category, in the order categories fill.

    Meant to be read in a diff or an editor while the book is being translated —
    ``glossary.json`` is the machine's copy, this is the reviewer's.
    """
    return write_text(
        Path(out_dir) / GLOSSARY_MARKDOWN_FILENAME,
        render_glossary_markdown(
            book,
            book_title=result.book_title,
            source_language=result.source_language,
            target_language=result.target_language,
            provider=result.provider,
            model=result.model,
            chapter_count=result.body_chapter_count,
            tokens=result.tokens,
        ),
    )


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
    "write_glossary_csv",
    "sync_glossary_from_csv",
    "write_glossary_manifest",
    "write_glossary_markdown",
    "write_glossary_trace",
]
