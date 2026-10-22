"""Human-facing reporting for every stage: durations, bars and summaries.

The CLI and the pipeline used to grow their own formatting next to every stage:
one ``_print_*_summary`` per command, one ``Console.kv`` block per phase, and two
slightly different ``human_seconds`` implementations. This module owns the two
things both front ends genuinely share:

* **formatting** — :func:`human_seconds`, :func:`human_bytes`;
* **a stage summary** — :func:`summarize` turns one stage result into a
  :class:`StageSummary`, a value the CLI prints and the pipeline records.

Nothing here prints on its own. A summary is data, so the same numbers can be
rendered to a terminal, written to a log, or embedded in ``--json`` without the
renderer and the stage drifting apart.

The summary handlers import the stage result classes directly. That is
deliberate: only the CLI and the pipeline driver import this module (it is not
re-exported from :mod:`book_splitter.core`), so a library user who imports
``core.run`` never pays for the stage packages.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import singledispatch
from typing import Any, TextIO

# Stage result types. Importing them here is safe: no result module imports
# ``core.report``, and ``core.__init__`` does not import this module.
from ..agent.result import SplitterResult
from ..assembly.models import AssemblyResult
from ..binding.runner import BindResult
from ..chunk_glossary.models import ChunkGlossaryResult
from ..chunking.models import ChunkingResult
from ..conversion.runner import ConversionResult
from ..dictionary.models import DictionaryResult
from ..epub.result import EpubResult
from ..glossary.records import GlossaryResult
from ..matter_agent.models import MatterResult
from ..openings.phase import OpeningAuditResult
from ..postprocess_agent.models import PostprocessResult
from ..preprocess_agent.models import PreprocessResult
from ..translation.records import TranslationResult


# --------------------------------------------------------------------------- #
# numbers
# --------------------------------------------------------------------------- #
def human_seconds(seconds: float) -> str:
    """``0.0213`` -> ``21 ms``; ``389.42`` -> ``6m 29.4s``."""
    value = max(0.0, float(seconds or 0.0))
    if value < 1:
        return f"{value * 1000:.0f} ms"
    if value < 60:
        return f"{value:.2f} s"
    minutes, rest = divmod(value, 60)
    if minutes < 60:
        return f"{int(minutes)}m {rest:04.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes:02d}m {rest:04.1f}s"


def human_bytes(size: float) -> str:
    """``411144`` -> ``401.5 KiB``."""
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024:
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def human_count(value: Any) -> str:
    """``12345`` -> ``12,345``; ``None`` -> ``0``."""
    try:
        return f"{int(value or 0):,}"
    except (TypeError, ValueError):
        return str(value)


def json_text(payload: Any) -> str:
    """The one JSON encoding used by every ``--json`` surface."""
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


# --------------------------------------------------------------------------- #
# one stage's summary
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StageSummary:
    """The numbers one stage run produced, ready to render anywhere.

    ``rows`` are the normal ``label: value`` lines. ``detail`` is debug-only
    (per-chapter files, per-chunk lines), ``artifacts`` are the paths a caller can
    open, and ``next_steps`` are the "what now" hints. Keeping them separate lets
    ``--debug`` add noise without changing the normal output.
    """

    headline: str
    status: str = "ok"
    rows: tuple[tuple[str, str], ...] = ()
    detail: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    artifacts: tuple[tuple[str, str], ...] = ()
    next_steps: tuple[str, ...] = ()

    @property
    def failed(self) -> bool:
        return self.status == "failed" or bool(self.errors)

    @property
    def skipped(self) -> bool:
        return self.status == "skipped"

    def render(self, *, debug: bool = False, indent: int = 2) -> list[str]:
        """The summary as text lines; no trailing newline."""
        pad = " " * indent
        lines = [f"{pad}{label:<16}: {value}" for label, value in self.rows]
        if debug and self.detail:
            lines.extend(f"{pad}  {line}" if line else "" for line in self.detail)
        lines.extend(f"{pad}warning         : {warning}" for warning in self.warnings)
        lines.extend(f"{pad}error           : {error}" for error in self.errors)
        lines.extend(f"{pad}{label:<16}: {path}" for label, path in self.artifacts)
        lines.extend(f"{pad}{step}" for step in self.next_steps)
        return lines


def _tokens(result: Any, key: str = "total_tokens") -> int:
    usage = getattr(result, "usage", None) or getattr(result, "tokens", None) or {}
    try:
        return int(usage.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def _status(result: Any) -> str:
    if getattr(result, "skipped", False):
        return "skipped"
    if not getattr(result, "ok", True):
        return "failed"
    if getattr(result, "dry_run", False):
        return "dry-run"
    return "ok"


def _artifact(label: str, path: Any) -> tuple[str, str] | None:
    if not path:
        return None
    return (label, str(path))


def _artifacts(*items: tuple[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(entry for entry in (_artifact(*item) for item in items) if entry)


def _errors(result: Any) -> tuple[str, ...]:
    errors = tuple(str(item) for item in (getattr(result, "errors", None) or []))
    error = getattr(result, "error", None)
    if error and str(error) not in errors:
        errors = (str(error), *errors)
    return errors


# --------------------------------------------------------------------------- #
# per-stage summaries
# --------------------------------------------------------------------------- #
@singledispatch
def summarize(result: Any) -> StageSummary:
    """Turn a stage result into a :class:`StageSummary`.

    Unknown result types get a generic summary from their ``to_dict`` so a new
    stage is never a crash; add a registration for the numbers that matter.
    """
    payload = result.to_dict() if hasattr(result, "to_dict") else {}
    ok = bool(payload.get("ok", True))
    return StageSummary(
        headline=str(payload.get("summary") or ("ok" if ok else "failed")),
        status="ok" if ok else "failed",
        rows=tuple(
            (str(key), str(value))
            for key, value in payload.items()
            if isinstance(value, (str, int, float)) and key != "ok"
        ),
    )


@summarize.register
def _summarize_conversion(result: ConversionResult) -> StageSummary:
    method = result.method or "—"
    rows = [
        ("source", result.source),
        ("kind", result.source_kind),
        ("method", method),
        ("markdown", result.markdown_path or "—"),
    ]
    if result.pages is not None:
        rows.append(
            (
                "text layer",
                f"{result.text_pages or 0}/{result.sampled_pages or 0} sampled page(s) "
                "carry text · OCR off",
            )
        )
    if result.images or result.image_refs:
        rows.append(
            ("images", f"{result.images:,} file(s) · {result.image_refs:,} reference(s)")
        )
    if result.dry_run:
        rows.append(("mode", "dry run — source validated, nothing written"))
    return StageSummary(
        headline=f"{result.source_kind} -> Markdown · {method}",
        status=_status(result),
        rows=tuple(rows),
        warnings=tuple(result.log),
        errors=(result.error,) if result.error else (),
        artifacts=_artifacts(("markdown", result.markdown_path)),
    )


@summarize.register
def _summarize_split(result: SplitterResult) -> StageSummary:
    tokens = _tokens(result)
    verification = result.verification or {}
    rows = [
        ("source", result.source_path),
        ("output", result.output_dir),
        ("strategy", result.strategy or "—"),
        ("chapters", str(result.chapter_count)),
        ("llm", f"{result.attempts} call(s) · {tokens:,} tokens"),
    ]
    if "reassembles_byte_for_byte" in verification:
        rows.append(
            (
                "byte-for-byte",
                f"{verification.get('reassembles_byte_for_byte')} "
                f"({verification.get('output_lines')}/{verification.get('source_lines')} lines)",
            )
        )
    elif verification.get("reused_existing_chapters"):
        rows.append(("reused", "existing chapters — nothing was re-sliced"))
    rows.append(("files", f"{len(result.files)} file(s)"))
    detail = tuple(
        f"{entry.get('filename', '?'):<52} lines {entry.get('start_line', '?')}-"
        f"{entry.get('end_line', '?')}  {entry.get('bytes', 0)} B"
        for entry in result.files
    )
    return StageSummary(
        headline=(
            f"{result.chapter_count} chapters · strategy {result.strategy or '—'} · "
            f"{tokens:,} tokens"
        ),
        status=_status(result),
        rows=tuple(rows),
        detail=detail,
        warnings=tuple(result.log),
        errors=(result.error,) if result.error else (),
        artifacts=_artifacts(("trace", result.trace_path)),
    )


@summarize.register
def _summarize_preprocess(result: PreprocessResult) -> StageSummary:
    calls = sum(int(chapter.get("attempts") or 0) for chapter in result.chapters)
    rows = [
        (
            "chapters",
            f"{result.ok_count}/{result.chapter_count} clean"
            + (f" · {result.skipped_count} reused" if result.skipped_count else ""),
        ),
        ("llm", f"{calls} call(s) · {_tokens(result):,} tokens"),
        ("output", result.output_dir),
    ]
    counts = (result.corpus_scan or {}).get("counts") or {}
    if counts:
        rows.append(
            ("damage", ", ".join(f"{key}={value}" for key, value in counts.items() if value))
        )
    openings = result.openings or {}
    if openings and not openings.get("skipped"):
        rows.append(
            (
                "openings",
                f"{int(openings.get('inspected_count') or 0)} checked · "
                f"{int(openings.get('fixed_count') or 0)} fixed · "
                f"{int(openings.get('remaining_count') or 0)} need attention",
            )
        )
    if result.parallel:
        rows.append(("parallel", "worker pool · completion order"))
    detail = tuple(
        f"{chapter.get('filename', '?'):<52} "
        f"{int(chapter.get('source_lines') or 0)} -> {int(chapter.get('output_lines') or 0)} lines"
        for chapter in result.chapters
    )
    return StageSummary(
        headline=f"{result.ok_count}/{result.chapter_count} chapters clean",
        status=_status(result),
        rows=tuple(rows),
        detail=detail,
        errors=_errors(result),
        artifacts=_artifacts(
            ("manifest", result.manifest_path), ("trace", result.trace_path)
        ),
    )


@summarize.register
def _summarize_dictionary(result: DictionaryResult) -> StageSummary:
    language = result.language or {}
    totals = result.totals or {}
    unique = result.unique_totals or {}
    timing = result.timing or {}
    rows = [
        (
            "language",
            f"{language.get('language') or '—'} via {language.get('method') or '—'} "
            f"(confidence {float(language.get('confidence') or 0):.2f}) · "
            f"backend {result.backend or '—'}"
            + (f" on {result.device}" if result.device else ""),
        ),
        (
            "chapters",
            f"{result.ok_count}/{result.chapter_count} ok"
            + (f" · {result.skipped_count} reused" if result.skipped_count else ""),
        ),
        (
            "sentences",
            f"{human_count(totals.get('sentences'))} -> "
            f"{human_count(totals.get('kept'))} kept",
        ),
        (
            "dropped",
            f"short={human_count(totals.get('dropped_short'))} "
            f"no-uppercase={human_count(totals.get('dropped_no_uppercase'))} "
            f"no-entity={human_count(totals.get('dropped_no_entity'))}",
        ),
    ]
    if result.save_unique:
        rows.append(
            (
                "first mentions",
                f"{human_count(unique.get('kept'))} kept · "
                f"{human_count(unique.get('nouns'))} noun(s) · "
                f"{human_count(unique.get('dropped_duplicate_noun'))} repeated",
            )
        )
    if result.entities:
        rows.append(
            ("entities", ", ".join(f"{key}={value:,}" for key, value in result.entities.items()))
        )
    if timing:
        rows.append(("ner time", human_seconds(float(timing.get("ner_seconds") or 0.0))))
        rows.append(("total time", human_seconds(float(timing.get("run_seconds") or 0.0))))
    if result.skipped:
        rows.append(("note", "every reduced chapter is already present"))
    detail = tuple(
        f"{chapter.get('filename', '?'):<48} "
        f"{human_count(chapter.get('sentences'))} sentences -> "
        f"{human_count(chapter.get('kept'))} kept"
        for chapter in result.chapters
    )
    return StageSummary(
        headline=(
            f"{result.ok_count}/{result.chapter_count} chapters · "
            f"{human_count(totals.get('sentences'))} sentences -> "
            f"{human_count(totals.get('kept'))} kept"
        ),
        status=_status(result),
        rows=tuple(rows),
        detail=detail,
        errors=_errors(result),
        artifacts=_artifacts(
            ("reduced", result.output_dir),
            ("unique", result.unique_dir),
            ("manifest", result.manifest_path),
            ("trace", result.trace_path),
        ),
    )


@summarize.register
def _summarize_glossary(result: GlossaryResult) -> StageSummary:
    rows = [
        (
            "chapters",
            f"{result.ok_count}/{result.chapter_count} ok"
            + (f" · {result.skipped_count} skipped" if result.skipped_count else "")
            + (f" · {result.reused_count} reused" if result.reused_count else ""),
        ),
        ("terms", human_count(result.entry_count)),
        ("llm", f"{result.tokens.get('calls', 0)} call(s) · {_tokens(result):,} tokens"),
    ]
    if result.categories:
        rows.append(
            ("categories", ", ".join(f"{key}={value}" for key, value in result.categories.items()))
        )
    if result.conflicts:
        rows.append(("conflicts", str(len(result.conflicts))))
    if result.provider or result.model:
        rows.append(("model", f"{result.provider or '—'}/{result.model or 'auto'}"))
    if result.resumed:
        rows.append(("resume", f"{result.reused_count} chapter(s) reused"))
    detail = tuple(
        f"{chapter.get('filename', '?'):<48} "
        f"{human_count(chapter.get('added'))} new · "
        f"{human_count(chapter.get('repeats'))} locked"
        for chapter in result.chapters
    )
    return StageSummary(
        headline=f"{human_count(result.entry_count)} terms · "
        f"{result.ok_count}/{result.chapter_count} chapters",
        status=_status(result),
        rows=tuple(rows),
        detail=detail,
        errors=_errors(result),
        artifacts=_artifacts(
            ("glossary", result.glossary_path),
            ("markdown", result.markdown_path),
            ("manifest", result.manifest_path),
            ("trace", result.trace_path),
        ),
    )


@summarize.register
def _summarize_chunking(result: ChunkingResult) -> StageSummary:
    verification = result.verification or {}
    settings = result.settings or {}
    tokenizer = settings.get("tokenizer") or {}
    rows = [
        (
            "chapters",
            f"{result.ok_count}/{result.chapter_count} ok"
            + (f" · {result.skipped_count} reused" if result.skipped_count else ""),
        ),
        ("chunks", human_count(result.chunk_count)),
        (
            "verification",
            "slices ok"
            if verification.get("chunks_are_source_slices")
            else "FAILED slice check",
        ),
        (
            "byte-for-byte",
            f"book={verification.get('book_reconstructs_byte_for_byte')} · "
            f"size violations={verification.get('chunk_size_violations', 0)}",
        ),
        ("tokenizer", str(tokenizer.get("name") or tokenizer.get("kind") or "—")),
    ]
    detail = tuple(
        f"{chapter.get('filename', '?'):<52} "
        f"{human_count(chapter.get('chunk_count'))} chunk(s) · "
        f"{human_count(chapter.get('token_count'))} tok"
        for chapter in result.chapters
    )
    return StageSummary(
        headline=f"{human_count(result.chunk_count)} chunks · "
        f"{result.chapter_count} chapters",
        status=_status(result),
        rows=tuple(rows),
        detail=detail,
        errors=_errors(result),
        artifacts=_artifacts(
            ("output", result.output_dir),
            ("manifest", result.manifest_path),
            ("trace", result.trace_path),
        ),
    )


@summarize.register
def _summarize_chunk_glossary(result: ChunkGlossaryResult) -> StageSummary:
    rows = [
        ("chunks", human_count(result.chunk_count)),
        ("entities", human_count(result.entity_count)),
        ("terms", human_count(result.term_count)),
        ("matched", human_count(result.matched_term_count)),
        ("unmatched", human_count(result.unmatched_term_count)),
        ("matched chunks", human_count(result.matched_chunk_count)),
        ("glossary", result.glossary_path),
        (
            "language",
            f"{result.language.get('language') or '—'} via "
            f"{result.language.get('method') or '—'} · backend {result.backend or '—'}",
        ),
        ("output", result.output_dir),
    ]
    detail = tuple(
        f"{chunk.get('filename', '?'):<44} "
        f"entities={int(chunk.get('entity_count') or 0)} "
        f"terms={int(chunk.get('term_count') or 0)} "
        f"matched={int(chunk.get('matched_count') or 0)}"
        for chunk in result.chunks
    )
    return StageSummary(
        headline=f"{human_count(result.matched_term_count)}/"
        f"{human_count(result.term_count)} terms matched · "
        f"{human_count(result.chunk_count)} chunks",
        status=_status(result),
        rows=tuple(rows),
        detail=detail,
        errors=_errors(result),
        artifacts=_artifacts(
            ("manifest", result.manifest_path), ("trace", result.trace_path)
        ),
    )


@summarize.register
def _summarize_matter(result: MatterResult) -> StageSummary:
    totals = result.totals or {}
    rows = [
        ("sections", human_count(result.section_count)),
        (
            "chunks",
            f"{result.ok_count}/{result.chunk_count} ok"
            + (f" · {result.reused_count} reused" if result.reused_count else ""),
        ),
        ("omitted", human_count(totals.get("omitted"))),
        ("llm", f"{_tokens(result):,} tokens"),
        ("title", result.book_title or "—"),
        ("author", result.book_author or "—"),
    ]
    if result.book_plan.get("title"):
        rows.append(("localized title", str(result.book_plan["title"])))
    if result.synthetic_sections:
        rows.append(("synthetic", f"{len(result.synthetic_sections)} section(s)"))
    detail = tuple(
        f"{chunk.get('filename', '?'):<46} "
        f"{chunk.get('section_kind', '?')} "
        f"{human_count(chunk.get('source_chars'))} -> "
        f"{human_count(chunk.get('translation_chars'))} chars"
        for chunk in result.chunks
    )
    return StageSummary(
        headline=f"{result.ok_count}/{result.chunk_count} matter chunks localized",
        status=_status(result),
        rows=tuple(rows),
        detail=detail,
        errors=_errors(result),
        artifacts=_artifacts(
            ("translations", result.output_dir),
            ("prompts", result.prompts_dir),
            ("plan", result.matter_path),
            ("manifest", result.manifest_path),
            ("trace", result.trace_path),
        ),
    )


@summarize.register
def _summarize_translation(result: TranslationResult) -> StageSummary:
    rows = [
        (
            "chunks",
            f"{result.ok_count}/{result.chunk_count} ok"
            + (f" · {result.reused_count} reused" if result.reused_count else ""),
        ),
        ("llm", f"{_tokens(result):,} tokens"),
        ("glossary", result.glossary_scope),
        (
            "model",
            f"{result.translator or '—'} · {result.provider or '—'}/{result.model or 'auto'}",
        ),
        ("output", result.output_dir),
    ]
    if result.resumed:
        rows.append(("resume", f"{result.reused_count} chunk(s) reused"))
    if result.limited:
        rows.append(("window", "a limited prefix — the manifest is partial"))
    detail = tuple(
        f"{chunk.get('filename', '?'):<46} "
        f"{human_count(chunk.get('source_chars'))} -> "
        f"{human_count(chunk.get('translation_chars'))} chars · "
        f"{human_count((chunk.get('tokens') or {}).get('total_tokens'))} tok"
        for chunk in result.chunks
    )
    return StageSummary(
        headline=f"{result.ok_count}/{result.chunk_count} chunks translated",
        status=_status(result),
        rows=tuple(rows),
        detail=detail,
        errors=_errors(result),
        artifacts=_artifacts(
            ("translations", result.output_dir),
            ("prompts", result.prompts_dir),
            ("manifest", result.manifest_path),
            ("trace", result.trace_path),
        ),
    )


@summarize.register
def _summarize_assembly(result: AssemblyResult) -> StageSummary:
    rows = [
        (
            "chapters",
            f"{result.ok_count}/{result.chapter_count} written"
            + (f" · {result.reused_count} reused" if result.reused_count else "")
            + (
                f" · {result.incomplete_count} incomplete"
                if result.incomplete_count
                else ""
            ),
        ),
        (
            "chunks",
            f"{human_count((result.totals or {}).get('translated_chunks'))} translated "
            f"of {human_count((result.totals or {}).get('chunks'))} in the window",
        ),
        ("output", result.output_dir),
    ]
    if result.limited:
        rows.append(("window", f"first {result.first_chapters} chapter(s)"))
    detail = tuple(
        f"incomplete    : {chapter.get('filename')} — {chapter.get('note') or 'not translated'}"
        for chapter in result.chapters
        if chapter.get("skipped")
    )
    return StageSummary(
        headline=f"{result.ok_count}/{result.chapter_count} chapters written",
        status=_status(result),
        rows=tuple(rows),
        detail=detail,
        errors=_errors(result),
        artifacts=_artifacts(
            ("manifest", result.manifest_path), ("trace", result.trace_path)
        ),
    )


@summarize.register
def _summarize_postprocess(result: PostprocessResult) -> StageSummary:
    validation = result.validation or {}
    toc = result.toc or {}
    rows = [
        (
            "chapters",
            f"{result.ok_count}/{result.chapter_count} ok"
            + (f" · {result.reused_count} reused" if result.reused_count else ""),
        ),
        (
            "book plan",
            str(result.book_plan.get("policy") or result.book_plan.get("title") or "—"),
        ),
        ("validation", "ok" if validation.get("ok") else "not validated"),
        ("toc", "updated" if toc.get("ok") else "unchanged"),
        ("llm", f"{_tokens(result):,} tokens"),
        ("output", result.output_dir),
    ]
    if result.limited:
        rows.append(("window", f"first {result.first_chapters} chapter(s)"))
    detail = tuple(
        f"{chapter.get('filename', '?'):<48} "
        f"{'ok' if chapter.get('ok') else 'FAIL'} · "
        f"{len(chapter.get('operations') or [])} operation(s)"
        for chapter in result.chapters
    )
    return StageSummary(
        headline=f"{result.ok_count}/{result.chapter_count} chapters postprocessed",
        status=_status(result),
        rows=tuple(rows),
        detail=detail,
        errors=_errors(result),
        artifacts=_artifacts(
            ("manifest", result.manifest_path), ("trace", result.trace_path)
        ),
    )


@summarize.register
def _summarize_bind(result: BindResult) -> StageSummary:
    rows = [
        ("files", str(result.chapter_count)),
        ("table of contents", "included" if result.toc_included else "not found"),
        ("size", f"{result.chars:,} characters · {result.lines:,} lines"),
        ("output", result.output_path),
    ]
    return StageSummary(
        headline=f"{result.chapter_count} files · {result.chars:,} characters",
        status=_status(result),
        rows=tuple(rows),
        errors=_errors(result),
        artifacts=_artifacts(
            ("manifest", result.manifest_path), ("trace", result.trace_path)
        ),
    )


@summarize.register
def _summarize_epub(result: EpubResult) -> StageSummary:
    verification = result.verification or {}
    rows = [
        ("title", result.title),
        ("author", result.author or "(none recorded)"),
        ("language", result.language),
        ("identifier", result.identifier),
        ("documents", str(result.document_count)),
        ("pictures", str(result.image_count)),
        (
            "size",
            f"{result.bytes:,} bytes · {result.chars:,} characters of Markdown",
        ),
        (
            "verification",
            f"mimetype {'ok' if verification.get('mimetype_first') else 'FAILED'} · "
            f"container {'ok' if verification.get('container_ok') else 'FAILED'} · "
            f"package {'ok' if verification.get('opf_ok') else 'FAILED'} · "
            f"navigation {'ok' if verification.get('nav_ok') else 'FAILED'} · "
            f"{verification.get('xhtml_valid', 0)}/{verification.get('documents', 0)} XHTML",
        ),
        ("output", result.output_path),
    ]
    if result.cover:
        rows.insert(6, ("cover", result.cover))
    if result.toc_dropped:
        rows.append(("table of contents", "dropped (the reader builds its own)"))
    return StageSummary(
        headline=f"{result.document_count} documents · {result.bytes:,} bytes",
        status=_status(result),
        rows=tuple(rows),
        warnings=tuple(result.warnings),
        errors=_errors(result),
        artifacts=_artifacts(
            ("manifest", result.manifest_path), ("trace", result.trace_path)
        ),
    )


@summarize.register
def _summarize_openings(result: OpeningAuditResult) -> StageSummary:
    rows = [
        ("chapters", str(result.chapter_count)),
        ("inspected", str(result.inspected_count)),
        ("clean", str(result.clean_count)),
        ("issues", str(result.issue_count)),
        ("fixable", str(result.fixable_count)),
        ("fixed", str(result.fixed_count)),
        ("need attention", str(result.remaining_count)),
        ("folder", result.folder),
    ]
    if result.skipped:
        rows.append(("skipped", result.skipped))
    detail = tuple(
        f"{chapter.get('filename', '?'):<52} "
        f"{chapter.get('issue_count', 0)} issue(s)"
        for chapter in result.chapters
    )
    return StageSummary(
        headline=(
            f"{result.inspected_count} opening(s) · {result.fixed_count} fixed · "
            f"{result.remaining_count} need attention"
        ),
        status=_status(result),
        rows=tuple(rows),
        detail=detail,
        warnings=tuple(result.errors) if result.ok else (),
        errors=tuple(result.errors) if not result.ok else (),
    )


# --------------------------------------------------------------------------- #
# deterministic phases that return plain payloads
# --------------------------------------------------------------------------- #
def summarize_payload(name: str, payload: Mapping[str, Any]) -> StageSummary:
    """Summarize the deterministic ``recon``/``scan`` payloads.

    Those phases do not return a typed result; their payload is the report. The
    rows here are the same numbers the pipeline used to print inline.
    """
    payload = dict(payload or {})
    if name == "recon":
        assessment = payload.get("header_assessment") or {}
        toc = payload.get("printed_toc_block") or {}
        rows = [
            ("source", str(payload.get("source") or "—")),
            ("lines", human_count(payload.get("line_count"))),
            ("headers", human_count(payload.get("header_count"))),
            ("verdict", str(assessment.get("verdict") or "—")),
            ("candidates", human_count(payload.get("candidate_count"))),
            ("printed TOC", "found" if toc.get("found") else "not found"),
        ]
        return StageSummary(
            headline=(
                f"{human_count(payload.get('line_count'))} lines · "
                f"{human_count(payload.get('header_count'))} headers "
                f"({assessment.get('verdict') or '—'}) · "
                f"{human_count(payload.get('candidate_count'))} candidates"
            ),
            rows=tuple(rows),
        )
    if name == "scan":
        counts = payload.get("counts") or {}
        rows = [
            ("chapters", human_count(payload.get("chapter_count"))),
            ("findings", human_count(payload.get("total_findings"))),
            ("shapes", str(len(counts))),
        ]
        rows.extend((str(key), human_count(value)) for key, value in sorted(counts.items()))
        return StageSummary(
            headline=(
                f"{human_count(payload.get('total_findings'))} findings in "
                f"{len(counts)} shape(s)"
            ),
            rows=tuple(rows),
        )
    generic_rows = tuple(
        (str(key), str(value))
        for key, value in payload.items()
        if isinstance(value, (str, int, float))
    )
    return StageSummary(headline=name, rows=generic_rows)


# --------------------------------------------------------------------------- #
# progress bar
# --------------------------------------------------------------------------- #
RecordFormatter = Callable[[dict, int, int], str]
LineFormatter = Callable[[dict], str]


class Bar:
    """A one-line progress bar that also works when stdout is not a terminal.

    On a terminal it redraws in place about once a second while a model call is
    in flight. When piped (or captured by a test) it prints one durable line per
    finished item instead: a ``\\r`` bar is noise in a log, and the durable line
    carries the same counts. The numbers are the ones that matter for a paid run:
    done/total, elapsed, ETA, tokens and the unit currently in flight.

    The bar owns its thread but not its formatting: ``format_record`` renders one
    finished item (the durable line) and ``format_line`` renders the live state.
    """

    WIDTH = 22
    HEARTBEAT_SECONDS = 10.0

    def __init__(
        self,
        total: int = 0,
        *,
        label: str = "",
        format_record: RecordFormatter | None = None,
        format_line: LineFormatter | None = None,
        stream: TextIO | None = None,
        tty: bool | None = None,
        heartbeat: float = HEARTBEAT_SECONDS,
        width: int = WIDTH,
    ) -> None:
        self.total = max(0, int(total))
        self.label = label
        self._format_record = format_record or self._default_record
        self._format_line = format_line or self._default_line
        self._stream = stream or sys.stdout
        self.tty = self._stream.isatty() if tty is None else bool(tty)
        self.heartbeat = max(0.1, float(heartbeat))
        self.width = max(4, int(width))

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._drawn = 0
        self._started = time.monotonic()
        self._done = 0
        self._tokens = 0
        self._current = ""
        self._current_started = self._started
        self._durations: list[float] = []
        self._state: dict[str, Any] = {}

    # -- hooks a runner calls ---------------------------------------------- #
    def start(self, info: Mapping[str, Any] | None = None) -> None:
        """A unit of work is about to start."""
        info = dict(info or {})
        now = time.monotonic()
        with self._lock:
            self._state = info
            self._current = str(info.get("filename") or info.get("label") or self.label)
            self._current_started = now
            self._done = int(info.get("done") or self._done)
            self.total = int(info.get("total") or self.total)
            prior = int((info.get("tokens") or {}).get("total_tokens") or 0)
            self._tokens = max(self._tokens, prior)
        if self.tty and (self._thread is None or not self._thread.is_alive()):
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._loop, name="book-splitter-bar", daemon=True
            )
            self._thread.start()

    def update(self, **state: Any) -> None:
        """Merge extra state the line formatter may render."""
        with self._lock:
            self._state.update(state)

    def finish(self, record: Mapping[str, Any] | dict, done: int, total: int) -> str:
        """One unit finished: print its durable line and update the bar."""
        record = dict(record or {})
        now = time.monotonic()
        with self._lock:
            if self._current and self._current_started:
                self._durations.append(max(0.0, now - self._current_started))
            self._done = int(done)
            self.total = int(total or self.total)
            tokens = record.get("tokens") or record.get("usage") or {}
            self._tokens += int((tokens or {}).get("total_tokens") or 0)
            self._current = ""
            self._current_started = 0.0
        line = self._format_record(record, done, max(0, int(total)))
        if self.tty:
            self._clear()
        print(line, file=self._stream, flush=True)
        if self.tty:
            self._draw()
        return line

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None
        if self.tty:
            self._clear()

    # -- rendering ---------------------------------------------------------- #
    def line(self) -> str:
        with self._lock:
            state = dict(self._state)
            state.setdefault("done", self._done)
            state.setdefault("total", self.total)
            state.setdefault("tokens", self._tokens)
            if self._current:
                state["current"] = self._current
                state["current_seconds"] = time.monotonic() - self._current_started
            state["elapsed"] = time.monotonic() - self._started
            state["durations"] = list(self._durations)
        return self._format_line(state)

    def _default_record(self, record: dict, done: int, total: int) -> str:
        flag = "reuse" if record.get("reused") else ("ok   " if record.get("ok", True) else "FAIL ")
        name = str(record.get("filename") or record.get("label") or "?")
        tokens = (record.get("tokens") or record.get("usage") or {}).get("total_tokens") or 0
        return (
            f"  [{done:>3}/{total}] {flag} {name:<46} "
            f"{int(tokens):>7,} tok"
        )

    def _default_line(self, state: dict) -> str:
        done = int(state.get("done") or 0)
        total = int(state.get("total") or 0)
        fill = min(self.width, int(self.width * done / total)) if total else 0
        bar = "█" * fill + "░" * (self.width - fill)
        elapsed = float(state.get("elapsed") or 0.0)
        durations = state.get("durations") or []
        average = (
            sum(durations) / len(durations)
            if durations
            else (elapsed / done if done else None)
        )
        eta = average * max(total - done, 0) if average is not None else None
        parts = [
            f"  {bar} {done:>3}/{total}",
            f"{elapsed:.0f}s elapsed",
            f"ETA ~{eta:.0f}s" if eta is not None else "ETA …",
            f"{int(state.get('tokens') or 0):,} tok",
        ]
        current = str(state.get("current") or "")
        if current:
            parts.append(f"now {current} ({float(state.get('current_seconds') or 0):.0f}s)")
        return " · ".join(parts)

    def _draw(self) -> None:
        line = self.line()
        with self._lock:
            pad = max(0, self._drawn - len(line))
            print("\r" + line + " " * pad, end="", file=self._stream, flush=True)
            self._drawn = max(self._drawn, len(line))

    def _clear(self) -> None:
        with self._lock:
            if self._drawn:
                print("\r" + " " * self._drawn + "\r", end="", file=self._stream, flush=True)
                self._drawn = 0

    def _loop(self) -> None:
        last_heartbeat = 0.0
        while not self._stop.wait(1.0):
            if self.tty:
                self._draw()
                continue
            now = time.monotonic()
            if now - last_heartbeat >= self.heartbeat:
                last_heartbeat = now
                print(self.line(), file=self._stream, flush=True)


__all__ = [
    "Bar",
    "LineFormatter",
    "RecordFormatter",
    "StageSummary",
    "human_bytes",
    "human_count",
    "human_seconds",
    "json_text",
    "summarize",
    "summarize_payload",
]
