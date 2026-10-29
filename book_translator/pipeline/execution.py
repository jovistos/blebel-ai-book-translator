"""Phase execution mixin for the pipeline driver.

This module houses the individual phase executor implementations (stages 0 through 11).
PipelineDriver inherits from PhaseExecutionMixin so all phase methods remain fully available
on the driver instance.
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from book_translator.core.constants import (
    DEFAULT_GPU_INDEX,
    MATTER_TRANSLATOR_NAME,
)
from book_translator.core.errors import MissingDependencyError, ToolError
from book_translator.core.report import human_bytes, human_seconds
from book_translator.core.settings import (
    TRANSLATION_CHOICES,
    stage_model_label,
)
from book_translator.documents.chapters import (
    discover_chapters,
    has_chapter_files,
    has_chunk_files,
    has_postprocessed_chapter_files,
    has_translated_chapter_files,
    has_translation_files,
    load_split_manifest,
)
from book_translator.documents.io import load_lines, source_size_bytes
from book_translator.documents.window import (
    chapter_files_for_window,
)
from book_translator.paths import (
    repo_root,
)
from book_translator.stages.analysis.candidates import extract_candidates
from book_translator.stages.analysis.headers import assess_headers
from book_translator.stages.analysis.toc import detect_toc_block
from book_translator.stages.conversion import (
    is_markdown_source,
    run_conversion,
)
from book_translator.stages.preprocess.corpus import scan_corpus_damage

ROOT = repo_root()

def has_chapters(path: Path) -> bool:
    return has_chapter_files(path)

def rel(path: str | Path) -> str:
    resolved = Path(path).expanduser()
    try:
        return str(resolved.resolve().relative_to(ROOT))
    except ValueError:
        return str(resolved)

class PhaseFailed(RuntimeError):
    def __init__(self, message: str, summary: str = 'failed') -> None:
        super().__init__(message)
        self.summary = summary

class PhaseExecutionMixin:
    """Mixin providing execution implementations for all pipeline phases."""

    def _ensure_markdown_source(self) -> Path:
        """Run stage 1 on demand for a phase started past ``convert``.

        Idempotent: once ``self.source`` is Markdown this is a no-op. A dry run
        converts into a temporary folder, so ``data/output/`` keeps its promise
        of no artifacts while ``recon``/``split`` still have real text to read.
        """
        if is_markdown_source(self.source):
            return self.source
        out_dir = self.markdown_out
        if self.args.dry_run:
            if self._convert_temp_dir is None:
                self._convert_temp_dir = Path(
                    tempfile.mkdtemp(prefix="book-translator-convert-")
                )
            out_dir = self._convert_temp_dir
        result = run_conversion(
            self.source,
            out_dir,
            force=self.args.force_convert,
            images=getattr(self.args, "images", True),
        )
        self.source = Path(result.markdown_path)
        self.console.kv(
            "stage 1",
            f"{result.source_kind} -> {rel(result.markdown_path)} ({result.method})",
        )
        return self.source

    def phase_convert(self) -> tuple[str, dict[str, Any]]:
        """Stage 1: EPUB/PDF/TXT -> the Markdown every later phase reads."""
        console = self.console
        target_dir = self.markdown_out
        if self.args.dry_run:
            if self._convert_temp_dir is None:
                self._convert_temp_dir = Path(
                    tempfile.mkdtemp(prefix="book-translator-convert-")
                )
            target_dir = self._convert_temp_dir
        runtime = self._stage_runtime(
            original_source=self.original_source,
            conversion_target=target_dir,
            conversion_live=bool(self.args.dry_run),
        )
        result = self.options.stage.call("convert", runtime)()
        if result.markdown_path:
            self.source = Path(result.markdown_path)

        console.kv("source", rel(result.source))
        console.kv("kind", result.source_kind)
        if result.pages is not None:
            console.kv(
                "text layer",
                f"{result.text_pages}/{result.sampled_pages} sampled PDF page(s) carry text "
                "· OCR off",
            )
        console.kv("method", result.method)
        console.kv("markdown", rel(result.markdown_path))
        if result.images or result.image_refs:
            console.kv(
                "images",
                f"{result.images} file(s) · {result.image_refs} reference(s) in the Markdown",
            )
        for line in result.log:
            console.info(f"     · {line}")
        if self.args.dry_run:
            console.info(
                "     · dry run: converted into a temporary folder, not under data/output/"
            )
        summary = f"{result.source_kind} -> Markdown · {result.method}"
        if result.reused:
            summary += " · reused"
        elif self.args.dry_run:
            summary += " · dry run"
        return summary, result.to_dict()

    # -- phase 1: recon ---------------------------------------------------- #
    def phase_recon(self) -> tuple[str, dict[str, Any]]:
        """Deterministic structural report: no LLM, no key, no writes."""
        from book_translator.stages.agent.tools import scan_markdown_headers

        args, console = self.args, self.console
        self._ensure_markdown_source()
        lines = load_lines(self.source)
        headers = scan_markdown_headers(str(self.source))["headers"]
        assessment = assess_headers(headers, len(lines))
        toc = detect_toc_block(lines, scan_lines=args.toc_scan_lines)
        candidate_report = extract_candidates(lines, max_candidates=args.max_candidates)
        candidates = candidate_report["candidates"]
        candidate_total = int(candidate_report.get("count") or len(candidates))

        console.kv(
            "headers",
            f"{len(headers):,} · verdict {assessment['verdict']!r} "
            f"(score {assessment['score']:.2f}) · "
            f"{assessment['chapter_like_count']} look like chapters",
        )
        if toc.get("found"):
            entries = toc.get("entries") or []
            console.kv(
                "printed TOC",
                f"lines {toc['start_line']}–{toc['end_line']} via {toc.get('method')} "
                f"· {len(entries)} entries",
            )
        else:
            console.kv("printed TOC", "not found (no dot-leader block in the front matter)")
        shown = (
            f"{candidate_total:,} ({len(candidates):,} shown, raise --max-candidates for more)"
            if candidate_report.get("truncated")
            else f"{candidate_total:,}"
        )
        console.kv("candidates", f"{shown} unmarked title-looking lines")
        console.kv("why", "; ".join(assessment.get("reasons") or []))

        console.debug("")
        console.debug("    headers:")
        for header in headers[:20]:
            console.debug(f"      L{header['line_number']:<5} h{header['level']}  {header['text']}")
        if len(headers) > 20:
            console.debug(f"      … {len(headers) - 20} more")
        if toc.get("found") and toc.get("entries"):
            console.debug("    printed TOC entries:")
            for entry in (toc.get("entries") or [])[:20]:
                console.debug(
                    f"      L{entry['line_number']:<5} {entry['title']}  (page {entry['page']})"
                )
        if candidates:
            console.debug("    heuristic candidates:")
            for candidate in candidates[:20]:
                flags = ",".join(
                    name
                    for name, on in (
                        ("standalone", candidate["standalone"]),
                        ("keyword", candidate["matches_chapter_keyword"]),
                        ("numbered", candidate["numbered"]),
                        ("caps", candidate["all_caps"]),
                    )
                    if on
                )
                console.debug(
                    f"      L{candidate['line_number']:<5} {candidate['text']}  [{flags}]"
                )

        summary = (
            f"{len(lines):,} lines · {len(headers):,} headers "
            f"({assessment['verdict']}) · {len(candidates):,} candidates"
        )
        payload = {
            "source": str(self.source),
            "line_count": len(lines),
            "size_bytes": source_size_bytes(self.source),
            "header_count": len(headers),
            "header_assessment": assessment,
            "printed_toc_block": toc,
            "candidate_count": len(candidates),
        }
        return summary, payload

    # -- phase 2: split ---------------------------------------------------- #
    def phase_split(self) -> tuple[str, dict[str, Any]]:
        """Stage 1: the agent chooses the cuts, a plain node copies the lines.

        With ``--map`` the cuts come from a JSON file instead of the agent, so
        this phase needs no provider, no key and no network. With
        ``--skip-if-present`` and chapters already on disk this phase is decided
        *before* any provider setting is resolved, so reusing a finished split
        needs no API key either.
        """
        from book_translator.stages.agent import result_from_existing_chapters

        args, console = self.args, self.console

        if args.skip_if_present and not args.dry_run and has_chapters(self.chapters_dir):
            console.kv("source", rel(self.source))
            console.kv("output", rel(self.chapters_dir))
            result = result_from_existing_chapters(self.chapters_dir)
            for line in result.log:
                console.info(f"     · {line}")
            console.kv("strategy", result.strategy or "—")
            console.kv("chapters", result.chapter_count)
            console.kv("llm", "not used — the chapters were reused")
            console.kv("files", f"{len(result.files)} reused")
            self._log_paths("trace", result.trace_path)
            return (
                f"{result.chapter_count} chapters · reused (nothing re-sliced, no model)",
                result.to_dict(),
            )

        self._ensure_markdown_source()
        console.kv("source", rel(self.source))
        console.kv("output", rel(self.chapters_dir))

        if args.map:
            from book_translator.stages.slicing.mapfile import load_split_map

            mapping, notes = load_split_map(args.map)
            console.kv("map", f"{rel(args.map)} · {len(mapping)} split point(s)")
            for line in notes:
                console.info(f"     · {line}")
            console.kv("budget", f"no agent turns · min {args.min_chapters} chapters")
            console.info("     the map is validated, then a Python node copies the source lines")
            runtime = self._stage_runtime(source=self.source)
            result = self.options.stage.call("split_map", runtime, mapping=mapping)()
        else:
            settings = None if args.dry_run else self.settings
            if not args.dry_run:
                assert settings is not None
                console.kv("provider", f"{settings.provider} · {settings.model}")
            console.kv(
                "budget",
                f"max {args.max_attempts} agent turn(s), min {args.min_chapters} chapters",
            )
            console.info(
                "     the model only routes; a Python node copies the source lines verbatim,"
            )
            console.info("     and the verifier proves the chapters reassemble byte for byte")
            runtime = self._stage_runtime(settings=settings, source=self.source)
            result = self.options.stage.call("split", runtime)()

        for line in result.log:
            console.info(f"     · {line}")

        tokens = int((result.usage or {}).get("total_tokens", 0) or 0)
        console.kv("strategy", result.strategy or "—")
        console.kv("chapters", result.chapter_count)
        console.kv("llm", f"{result.attempts} call(s) · {tokens:,} tokens")
        verification = result.verification or {}
        if "reassembles_byte_for_byte" in verification:
            console.kv(
                "byte-for-byte",
                f"{verification.get('reassembles_byte_for_byte')} "
                f"({verification.get('output_lines')}/{verification.get('source_lines')} lines)",
            )
        elif verification.get("reused_existing_chapters"):
            console.kv("reused", "chapter files were already there — nothing was re-sliced")
        console.kv(
            "files",
            f"{len(result.files)} {'planned' if args.dry_run else 'written'}"
            f"{' (dry run)' if args.dry_run else ''}",
        )
        for entry in result.files:
            console.debug(
                f"      {entry['filename']:<52} lines {entry['start_line']:>5}-"
                f"{entry['end_line']:<5} {entry['bytes']:>8} B"
            )
        self._log_paths("trace", result.trace_path)

        byte_exact = bool(verification.get("reassembles_byte_for_byte"))
        summary = (
            f"{result.chapter_count} chapters · strategy {result.strategy or '—'} · "
            f"{tokens:,} tokens" + ("" if args.dry_run else f" · byte-for-byte {byte_exact}")
        )
        if not result.ok:
            issues = (result.validation or {}).get("issues") or []
            detail = "\n".join(f"     - {issue}" for issue in issues)
            raise PhaseFailed(
                f"{result.error or 'the split was not verified'}"
                + (f"\n{detail}" if detail else ""),
                summary=summary,
            )
        return summary, result.to_dict()

    # -- phase 3: scan ----------------------------------------------------- #
    def phase_scan(self) -> tuple[str, dict[str, Any]]:
        """Deterministic damage profile of the chapters stage 2 just wrote."""
        chapters = discover_chapters(self.chapters_dir)
        if self.chapter_limit is not None:
            chapters = chapters[: self.chapter_limit]
        scan = scan_corpus_damage(chapters, max_examples_per_pattern=self.args.examples)
        manifest = load_split_manifest(self.chapters_dir)
        counts = {name: value for name, value in (scan.get("counts") or {}).items() if value}

        console = self.console
        console.kv(
            "chapters",
            f"{len(chapters)} files · {int(scan.get('total_lines', 0)):,} lines · "
            f"{human_bytes(sum(c.bytes for c in chapters))}",
        )
        console.kv("strategy", self._recorded_strategy(manifest))
        window = self._window_text()
        if window is not None:
            console.kv("window", window)
        console.kv(
            "damage",
            f"{int(scan.get('total_findings', 0)):,} findings in {len(counts)} shape(s)",
        )
        for name, value in sorted(counts.items(), key=lambda item: -item[1]):
            console.info(f"      {name:<26} {value:>5}")
        if not counts:
            console.info("      none recognised — stage 3 has nothing to repair")

        for name in sorted(counts, key=lambda key: -counts[key]):
            examples = (scan.get("examples") or {}).get(name) or []
            for example in examples[: self.args.examples]:
                console.debug(
                    f"      {name} · {example['chapter']}:{example['line_number']} "
                    f"→ {example['text']!r}"
                )

        per_chapter = scan.get("per_chapter") or {}
        for filename, chapter_counts in per_chapter.items():
            rendered = ", ".join(f"{k}={v}" for k, v in sorted(chapter_counts.items()))
            console.debug(f"      {filename:<52} {rendered}")

        summary = (
            f"{int(scan.get('total_findings', 0)):,} findings in "
            f"{len(counts)} shape(s) over {len(chapters)} chapters"
        )
        return summary, scan

    # -- phase 4: dictionary ----------------------------------------------- #
    def phase_dictionary(self) -> tuple[str, dict[str, Any]]:
        """Stage 3: reduce each chapter to the sentences that carry an entity.

        Then reduce again to the sentences that introduce a noun. No LLM: a
        deterministic sentence funnel plus an optional NER tagger. When the tagger
        is not installed the phase reports a skip with the install command instead
        of failing the run, the way stage 3 has its off switch. It reads the
        cleaned chapters whenever stage 3 covers the window, so page numbers and
        running headers never reach the sentence funnel or the glossary.
        """
        args, console = self.args, self.console
        save_unique = not args.no_save_unique
        input_dir = self.text_input()
        self._kv_text_input(console, input_dir)
        console.kv("output", rel(self.reduced_out))
        if save_unique:
            console.kv("unique", rel(self.unique_out))
        console.kv("provider", "none — deterministic, no LLM")
        window = self._window_text()
        if window is not None:
            console.kv("window", window)
        console.kv(
            "filters",
            f"uppercase={'off' if args.no_uppercase_filter else 'on'} · "
            f"min_words={args.min_words} · types={args.keep_types}",
        )
        console.kv(
            "device",
            "cpu"
            if args.cpu
            else f"gpu {args.gpu if args.gpu is not None else DEFAULT_GPU_INDEX}",
        )
        console.kv("policy", "keep a sentence only if it carries PER/LOC/ORG/MISC")
        if save_unique:
            console.kv(
                "first mentions",
                "keep a sentence only if it introduces a noun the book has not used yet",
            )

        try:
            runtime = self._stage_runtime(chapter_limit=self.chapter_limit)
            result = self.options.stage.call("dictionary", runtime)()
        except MissingDependencyError as exc:
            console.info(f"  · skipped: {exc}")
            payload: dict[str, Any] = {"missing_dependency": str(exc)}
            try:
                gloss_summary, gloss_payload = self._glossary_if_enabled()
            except ToolError as gloss_exc:
                console.info(f"  · glossary skipped: {gloss_exc}")
                return "skipped — the optional tagger is not installed", payload
            if gloss_payload is None:
                return "skipped — the optional tagger is not installed", payload
            payload["glossary"] = gloss_payload
            return (
                "reduction skipped (no tagger) · glossary: " + gloss_summary,
                payload,
            )

        language = result.language or {}
        totals = result.totals or {}
        unique_totals = result.unique_totals or {}
        console.kv(
            "language",
            f"{language.get('language') or '—'} via {language.get('method')} "
            f"(confidence {language.get('confidence', 0):.2f}) · backend {result.backend or '—'}"
            + (f" on {result.device}" if result.device else ""),
        )
        if language.get("note"):
            console.info(f"      note: {language['note']}")
        console.kv(
            "chapters",
            f"{result.ok_count}/{result.chapter_count} ok"
            + (f" · {result.skipped_count} reused" if result.skipped_count else ""),
        )
        if not result.skipped:
            console.kv(
                "sentences",
                f"{int(totals.get('sentences', 0)):,} → {int(totals.get('kept', 0)):,} kept",
            )
            console.kv(
                "dropped",
                f"short={int(totals.get('dropped_short', 0)):,} "
                f"no-uppercase={int(totals.get('dropped_no_uppercase', 0)):,} "
                f"no-entity={int(totals.get('dropped_no_entity', 0)):,}"
                + (
                    f" temporal={int(totals.get('dropped_temporal', 0)):,}"
                    if args.drop_temporal_only
                    else ""
                ),
            )
            if save_unique:
                console.kv(
                    "first mentions",
                    f"{int(unique_totals.get('kept', 0)):,} kept · "
                    f"{int(unique_totals.get('nouns', 0)):,} noun(s) · "
                    f"{int(unique_totals.get('dropped_duplicate_noun', 0)):,} dropped as repeats",
                )
            if result.entities:
                console.kv(
                    "entities", ", ".join(f"{k}={v:,}" for k, v in result.entities.items())
                )
            timing = result.timing or {}
            tagged = int(totals.get("sentences_tagged", 0))
            console.kv(
                "ner time",
                f"{human_seconds(float(timing.get('ner_seconds', 0.0)))} for "
                f"{tagged:,} sentence(s)"
                + (
                    f"  ({float(timing.get('milliseconds_per_sentence', 0.0)):.1f} ms each, "
                    f"{float(timing.get('sentences_per_second', 0.0)):.0f}/s)"
                    if tagged
                    else ""
                ),
            )
            if timing.get("backend_load_seconds"):
                where = result.device or ("cpu" if result.gpu is None else f"gpu {result.gpu}")
                console.kv(
                    "model load",
                    f"{human_seconds(float(timing['backend_load_seconds']))} "
                    f"({result.backend or '—'}, {where})",
                )
            console.kv(
                "total time",
                f"{human_seconds(float(timing.get('run_seconds', 0.0)))} "
                f"(chapters {human_seconds(float(timing.get('chapter_seconds', 0.0)))})",
            )
        self._log_paths("manifest", result.manifest_path)
        self._log_paths("trace", result.trace_path)

        if not result.ok:
            detail = "\n".join(f"     - {error}" for error in result.errors)
            raise PhaseFailed(
                f"{len(result.errors)} chapter(s) failed\n{detail}"
                if detail
                else "the run failed",
                summary=f"{result.ok_count}/{result.chapter_count} ok",
            )

        if result.skipped:
            summary = "nothing to do — every reduced chapter already exists"
        else:
            summary = (
                f"{result.ok_count}/{result.chapter_count} chapters · "
                f"{int(totals.get('sentences', 0)):,} sentences → "
                f"{int(totals.get('kept', 0)):,} kept"
            )
            if save_unique:
                summary += (
                    f" → {int(unique_totals.get('kept', 0)):,} first mentions "
                    f"({int(unique_totals.get('nouns', 0)):,} nouns)"
                )
        payload = result.to_dict()
        gloss_summary, gloss_payload = self._glossary_if_enabled()
        if gloss_payload is not None:
            payload["glossary"] = gloss_payload
            summary += " · glossary: " + gloss_summary
        if args.dry_run:
            summary += " · dry run"
        return summary, payload

    def _glossary_if_enabled(self) -> tuple[str, dict[str, Any] | None]:
        """The glossary half of the phase, or ``(reason, None)`` when switched off."""
        if not self.glossary_enabled:
            return "off (--no-glossary)", None
        return self._build_glossary()

    # -- the glossary, run inside the dictionary phase --------------------- #
    def _build_glossary(self) -> tuple[str, dict[str, Any]]:
        """Stage 4: grow one book-wide glossary, one chapter at a time.

        Part of the ``dictionary`` phase: the reduction produces the sentences,
        this turns them into the dictionary. It is the only step that calls a paid
        model per chapter, so it narrates itself: a live bar while the run is in
        flight, one line per finished chapter, and the token bill at the end.
        ``--dry-run`` calls the model and writes nothing, exactly like the
        ``glossary`` command; it is a validation run, not a free price quote.
        """
        from book_translator.stages.glossary.runner import resolve_sentences_dir

        args, console = self.args, self.console
        chapter_source = self.text_input()
        sentences_path = resolve_sentences_dir(
            chapter_source,
            sentences_dir=getattr(args, "sentences_in", None),
            sentences=args.sentences,
        )
        # Deliberately *not* resolved here: a run with nothing left to mine must
        # finish without a key. The label is flags > env > provider table, and
        # ``run_glossary`` resolves the real settings only when a chapter needs one.
        console.info("")
        console.info("  glossary — one chapter at a time, the dictionary carried forward")
        console.info(
            "             (Ctrl-C is safe: every finished chapter is already in the manifest)"
        )
        console.kv("sentences", rel(sentences_path) + f"  ({args.sentences})")
        console.kv("output", rel(self.glossary_out))
        console.kv("languages", f"{args.source_language} → {args.target_language}")
        console.kv("model", self._stage_model_label("dictionary"))
        console.kv(
            "skips",
            ("front matter, " if not getattr(args, "include_front_matter", False) else "")
            + ("back matter" if not getattr(args, "include_back_matter", False) else "")
            or "nothing",
        )

        self._bar_state = {"active": False, "total": 0, "done": 0, "current": ""}
        try:
            runtime = self._stage_runtime(
                settings_factory=lambda: self.settings_for("dictionary"),
                progress=self._on_glossary_chapter,
                chapter_start=self._on_glossary_start,
                chapter_limit=self.chapter_limit,
                sentences_dir=str(sentences_path),
            )
            result = self.options.stage.call("glossary", runtime)()
        finally:
            self._stop_glossary_bar()

        categories = result.categories or {}
        tokens = result.tokens or {}
        console.kv(
            "chapters",
            f"{result.ok_count}/{result.body_chapter_count} processed"
            + (f" · {result.reused_count} reused" if result.reused_count else "")
            + (
                f" · {result.skipped_count} skipped (front/back matter)"
                if result.skipped_count
                else ""
            ),
        )
        console.kv(
            "entries",
            f"{result.entry_count:,}"
            + (
                "  (" + ", ".join(f"{k}={v:,}" for k, v in categories.items()) + ")"
                if categories
                else ""
            ),
        )
        if result.conflicts:
            console.kv(
                "conflicts",
                f"{len(result.conflicts)} (a later chapter proposed a different rendering; "
                "the first one stands)",
            )
        chapters_with_calls = int(tokens.get("chapters") or 0)
        console.kv(
            "tokens",
            f"{int(tokens.get('total_tokens') or 0):,} total "
            f"({int(tokens.get('input_tokens') or 0):,} in, "
            f"{int(tokens.get('output_tokens') or 0):,} out) over "
            f"{chapters_with_calls} chapter(s), {int(tokens.get('calls') or 0)} call(s)",
        )
        timing = result.timing or {}
        if timing:
            console.kv(
                "time",
                f"{human_seconds(float(timing.get('run_seconds', 0.0)))} "
                f"(model {human_seconds(float(timing.get('model_seconds', 0.0)))})",
            )
        self._log_paths("glossary", result.glossary_path)
        self._log_paths("markdown", result.markdown_path)
        self._log_paths("manifest", result.manifest_path)
        self._log_paths("trace", result.trace_path)

        if not result.ok:
            detail = "\n".join(f"     - {error}" for error in result.errors)
            raise PhaseFailed(
                f"{len(result.errors)} chapter(s) failed\n{detail}"
                if detail
                else "the glossary run failed",
                summary=f"{result.ok_count}/{result.body_chapter_count} processed",
            )

        if result.processed_count == 0:
            summary = f"nothing to do — {result.reused_count} chapter(s) already in the glossary"
        else:
            summary = (
                f"{result.ok_count}/{result.body_chapter_count} chapters · "
                f"{result.entry_count:,} entr{'y' if result.entry_count == 1 else 'ies'}"
            )
            if result.skipped_count:
                summary += f" · {result.skipped_count} block(s) skipped"
        return summary, result.to_dict()

    def _stage_model_label(self, stage: str) -> str:
        """One stage's provider/model, without a key or an LM Studio request."""
        return stage_model_label(
            stage,
            shared=self._shared_model_args(),
            config=self.llm_config,
            overrides=self._llm_overrides.get(stage),
        )

    def _translation_model_label(self) -> str:
        """Stage 7's label, without resolving a key or asking LM Studio for its model."""
        return stage_model_label(
            "translate",
            provider=self.args.translate_provider,
            model=self.args.translate_model,
            base_url=self.args.translate_base_url,
            shared=self._shared_model_args(),
            config=self.llm_config,
            overrides=self._llm_overrides.get("translate"),
            allowed=TRANSLATION_CHOICES,
        )

    # -- live progress for the glossary ------------------------------------ #
    #: Seconds between heartbeat lines when stdout is not a terminal (a piped or
    #: logged run), and the redraw interval when it is one.
    GLOSSARY_HEARTBEAT_SECONDS = 10.0

    def _on_glossary_start(self, chapter: dict[str, Any]) -> None:
        """A chapter's model call is about to start: say which one, and tick."""
        self._bar_state.update(
            {
                "current": str(chapter.get("filename") or ""),
                "chapter_started": time.monotonic(),
                "done": int(chapter.get("done") or 0),
                "total": int(chapter.get("total") or 0),
            }
        )
        if self._bar_thread is None or not self._bar_thread.is_alive():
            self._bar_state["started"] = time.monotonic()
            self._bar_state["durations"] = []
            self._bar_state["active"] = True
            self._bar_stop.clear()
            self._bar_thread = threading.Thread(
                target=self._glossary_bar_loop, name="glossary-bar", daemon=True
            )
            self._bar_thread.start()

    def _on_glossary_chapter(self, chapter: dict[str, Any], done: int, total: int) -> None:
        """One chapter is in the dictionary: one durable line, then the bar resumes."""
        state = self._bar_state
        now = time.monotonic()
        elapsed = now - float(state.get("chapter_started") or now)
        durations = state.setdefault("durations", [])
        durations.append(elapsed)
        state["done"], state["total"], state["current"] = done, total, ""
        tokens = chapter.get("tokens") or {}
        self.console.clear_bar()
        self.console.info(
            f"  [{done:>2}/{total}] {chapter.get('filename') or '?'!s:<44} "
            f"{int(chapter.get('added') or 0):>3} new · "
            f"{int(chapter.get('repeats') or 0):>2} locked · "
            f"{int(chapter.get('conflicts') or 0):>2} conflict · "
            f"{int(tokens.get('total_tokens') or 0):>7,} tok · {elapsed:5.1f}s"
        )

    def _glossary_bar_line(self) -> str:
        state = self._bar_state
        total = int(state.get("total") or 0)
        if total <= 0:
            return "  starting the glossary run…"
        done = int(state.get("done") or 0)
        now = time.monotonic()
        elapsed = now - float(state.get("started") or now)
        durations = state.get("durations") or []
        average = (
            sum(durations) / len(durations)
            if durations
            else (elapsed / done if done else None)
        )
        remaining = max(total - done, 0)
        eta = average * remaining if average is not None else None
        width = 22
        filled = min(width, int(width * done / total)) if total else 0
        bar = "█" * filled + "░" * (width - filled)
        parts = [
            f"  {bar} {done:>2}/{total}",
            f"{human_seconds(elapsed)} elapsed",
            f"ETA ~{human_seconds(eta)}" if eta is not None else "ETA …",
        ]
        current = str(state.get("current") or "")
        if current:
            chapter_elapsed = now - float(state.get("chapter_started") or now)
            parts.append(f"now {current} ({chapter_elapsed:.0f}s)")
        return " · ".join(parts)

    def _glossary_bar_loop(self) -> None:
        last_heartbeat = 0.0
        while not self._bar_stop.wait(1.0):
            if not self._bar_state.get("active"):
                continue
            line = self._glossary_bar_line()
            if self.console.bar_enabled:
                self.console.bar(line)
                continue
            now = time.monotonic()
            if now - last_heartbeat >= self.GLOSSARY_HEARTBEAT_SECONDS:
                last_heartbeat = now
                self.console.stream_line(line)

    def _stop_glossary_bar(self) -> None:
        self._bar_state["active"] = False
        self._bar_stop.set()
        thread = self._bar_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._bar_thread = None
        self.console.clear_bar()

    # -- phase 5: preprocess ----------------------------------------------- #
    def phase_preprocess(self) -> tuple[str, dict[str, Any]]:
        """Stage 2: one agent run per chapter, each plan validated before it is applied."""

        args, console = self.args, self.console
        chapters = discover_chapters(self.chapters_dir)
        console.kv("input", rel(self.chapters_dir))
        console.kv("output", rel(self.preprocessed_out))
        preprocess_settings = None if args.dry_run else self.settings_for("preprocess")
        if preprocess_settings is not None:
            console.kv(
                "provider",
                f"{preprocess_settings.provider} · {preprocess_settings.model}",
            )
        shown = (
            len(chapters)
            if self.chapter_limit is None
            else min(len(chapters), self.chapter_limit)
        )
        console.kv("chapters", f"{shown} · one agent run each · workers={args.workers}")
        window = self._window_text()
        if window is not None:
            console.kv("window", window)
        console.kv(
            "policy",
            "validate the plan, apply it, then prove only the declared lines changed",
        )
        console.info(
            "     this is the token-heavy phase: the per-chapter lines below are live progress"
        )

        self._count_width = len(str(len(chapters)))
        self._name_width = max((len(c.filename) for c in chapters), default=24)
        self._last_tick = time.monotonic()

        runtime = self._stage_runtime(
            settings=preprocess_settings,
            progress=self._on_chapter,
            chapter_limit=self.chapter_limit,
        )
        result = self.options.stage.call("preprocess", runtime)()

        records = result.chapters or []
        operations = sum(len(r.get("operations") or []) for r in records)
        fixes = sum(len(r.get("text_fixes") or []) for r in records)
        dropped = sum(len(r.get("removed_artefacts") or []) for r in records)
        lines_in = sum(int(r.get("source_lines") or 0) for r in records)
        lines_out = sum(int(r.get("output_lines") or 0) for r in records)
        calls = sum(int(r.get("attempts") or 0) for r in records)
        tokens = int((result.usage or {}).get("total_tokens", 0) or 0)

        console.kv("chapters", f"{result.ok_count}/{result.chapter_count} clean")
        if getattr(result, "skipped_count", 0):
            console.kv("reused", f"{result.skipped_count} chapter(s)")
        console.kv("edits", f"{operations} operations · {dropped} lines dropped · {fixes} fixes")
        if not args.dry_run:
            console.kv("lines", f"{lines_in:,} → {lines_out:,}")
        openings = getattr(result, "openings", {}) or {}
        if openings and not openings.get("skipped"):
            console.kv(
                "openings",
                f"{int(openings.get('inspected_count') or 0)} checked · "
                f"{int(openings.get('fixed_count') or 0)} line(s) removed · "
                f"{int(openings.get('remaining_count') or 0)} need attention",
            )
        console.kv("llm", f"{calls} call(s) · {tokens:,} tokens")
        if result.parallel:
            console.kv("parallel", f"{args.workers} workers · completion order")
        self._log_paths("manifest", result.manifest_path)
        self._log_paths("trace", result.trace_path)

        if not result.ok:
            failed = [r for r in records if not r.get("ok")]
            detail = "\n".join(
                f"     - {r.get('filename')}: {r.get('error') or 'not verified'}" for r in failed
            )
            raise PhaseFailed(
                f"{len(failed)} chapter(s) failed\n{detail}"
                if detail
                else "the run was not verified",
                summary=f"{result.ok_count}/{result.chapter_count} clean",
            )

        summary = (
            f"{result.ok_count}/{result.chapter_count} clean · {operations} operation(s) · "
            f"{tokens:,} tokens"
        )
        if args.dry_run:
            summary += " · dry run"
        return summary, result.to_dict()

    def _on_chapter(self, chapter: dict[str, Any], done: int, total: int) -> None:
        """Live progress line, printed as each chapter finishes."""
        with self._progress_lock:
            now = time.monotonic()
            delta = now - self._last_tick
            self._last_tick = now

        name = str(chapter.get("filename") or "?")
        if chapter.get("ok"):
            out_lines = int(chapter.get("output_lines") or 0)
            rendered = f"{out_lines:<4}" if out_lines else "—   "
            count = len(chapter.get("operations") or [])
            fixes = len(chapter.get("text_fixes") or [])
            self.console.info(
                f"  [{done:>{self._count_width}}/{total}] ok   {name:<{self._name_width}} "
                f"{int(chapter.get('source_lines') or 0):>4} → {rendered} "
                f"ops={count:<3} fixes={fixes:<3} {delta:5.1f}s"
            )
        else:
            self.console.error(
                f"  [{done:>{self._count_width}}/{total}] FAIL {name:<{self._name_width}} "
                f"{chapter.get('error') or 'not verified'}"
            )
        for line in chapter.get("log") or []:
            self.console.debug(f"          · {line}")

    # -- phase 6: chunk ---------------------------------------------------- #
    def phase_chunk(self) -> tuple[str, dict[str, Any]]:
        """Stage 5: cut every chapter into reversible, translation-sized chunks."""

        args, console = self.args, self.console
        input_dir = self.chunk_input()
        if not has_chapter_files(input_dir):
            raise PhaseFailed(
                f"no chapter files in {rel(input_dir)} — run split first",
                summary="no chapters to chunk",
            )
        if self.chunk_in is not None:
            console.kv("input", rel(input_dir) + "  (--chunk-in)")
        else:
            self._kv_text_input(console, input_dir)
        console.kv("output", rel(self.chunks_out))
        console.kv(
            "chunk size",
            f"{args.chunk_size:,} tokens · overlap 0 · recursive",
        )
        console.kv("tokenizer", args.chunk_tokenizer)
        console.kv(
            "policy",
            "paragraph then sentence then line then word; every chunk is an exact source slice",
        )

        try:
            runtime = self._stage_runtime(chapter_limit=self.chapter_limit)
            result = self.options.stage.call("chunk", runtime)()
        except MissingDependencyError as exc:
            console.info(f"  · skipped: {exc}")
            return (
                "skipped — the optional chunker is not installed",
                {"missing_dependency": str(exc)},
            )

        totals = result.totals or {}
        verification = result.verification or {}
        chunks = int(result.chunk_count or 0)
        console.kv(
            "chapters",
            f"{result.ok_count}/{result.chapter_count} ok"
            + (f" · {result.skipped_count} reused" if result.skipped_count else ""),
        )
        console.kv("chunks", f"{chunks:,} · {int(totals.get('chunk_tokens') or 0):,} tokens")
        tokenizer = (result.settings or {}).get("tokenizer") or {}
        used = tokenizer.get("used") or tokenizer.get("requested") or "—"
        console.kv(
            "tokenizer",
            used + ("  (local approximation — exact tokenizer unavailable)" if tokenizer.get("fallback") else ""),
        )
        if tokenizer.get("note"):
            console.info(f"      note: {tokenizer['note']}")
        kinds = verification.get("boundary_counts") or {}
        shown = ", ".join(f"{kind}={value:,}" for kind, value in kinds.items() if value)
        if shown:
            console.kv("boundaries", shown)
        console.kv(
            "reconstruction",
            f"chapters={verification.get('chapters_reconstruct_byte_for_byte')} "
            f"book={verification.get('book_reconstructs_byte_for_byte')} "
            f"word-breaks={verification.get('word_breaking_chunks', 0)}",
        )
        self._log_paths("manifest", result.manifest_path)
        self._log_paths("trace", result.trace_path)

        if not result.ok:
            detail = "\n".join(f"     - {error}" for error in result.errors)
            raise PhaseFailed(
                f"{len(result.errors)} chapter(s) failed\n{detail}"
                if detail
                else "the chunk run was not verified",
                summary=f"{result.ok_count}/{result.chapter_count} ok",
            )

        if result.skipped:
            summary = f"nothing to do — {chunks} chunk(s) already on disk"
        else:
            summary = (
                f"{result.ok_count}/{result.chapter_count} chapters · {chunks:,} chunks · "
                f"max {args.chunk_size:,} tokens"
            )
        if args.dry_run:
            summary += " · dry run"
        return summary, result.to_dict()

    # -- phase 7: chunk glossary ------------------------------------------- #
    def phase_chunk_glossary(self) -> tuple[str, dict[str, Any]]:
        """Stage 6: tag every chunk first, then attach the glossary terms.

        The two phases are one call into the runner, but the narration keeps them
        apart because the order is the contract: no glossary lookup happens until
        the last chunk has been tagged.
        """
        from book_translator.core.constants import GLOSSARY_FILENAME

        args, console = self.args, self.console
        if not has_chunk_files(self.chunks_out):
            raise PhaseFailed(
                f"no chunk files in {rel(self.chunks_out)} — run chunk first",
                summary="no chunks to annotate",
            )
        glossary_path = self.glossary_out / GLOSSARY_FILENAME
        if not glossary_path.is_file():
            console.info(
                f"  · skipped: no glossary at {rel(glossary_path)} — run the "
                "dictionary/glossary phase first, or point --glossary-out at an existing "
                "dictionary"
            )
            return (
                "skipped — no glossary to attach",
                {"missing_glossary": str(glossary_path)},
            )

        console.kv("input", rel(self.chunks_out))
        console.kv("glossary", rel(glossary_path))
        console.kv("output", rel(self.chunk_glossary_out))
        console.kv(
            "match",
            f"threshold {args.match_threshold}% · scorer {args.match_scorer} · "
            f"candidates {args.match_candidates}",
        )
        console.kv(
            "phases",
            "ner over every chunk first, then fuzzy retrieval for every collected term",
        )
        console.kv("provider", "none — deterministic, no LLM")
        chapter_files = chapter_files_for_window(self.chunks_out, self.first_chapters)
        if chapter_files is not None:
            console.kv(
                "window",
                f"chunks of the first {self.first_chapters} body chapter(s)",
            )
        if args.limit is not None:
            console.kv("limit", f"first {args.limit:,} chunk(s) in the window")
        console.kv(
            "resume",
            "reuse finished chunks" if args.skip_if_present else "off",
        )

        try:
            runtime = self._stage_runtime(chapter_files=chapter_files)
            result = self.options.stage.call("chunk_glossary", runtime)()
        except MissingDependencyError as exc:
            console.info(f"  · skipped: {exc}")
            return (
                "skipped — the optional tagger or rapidfuzz is not installed",
                {"missing_dependency": str(exc)},
            )

        totals = result.totals or {}
        console.kv("chunks", f"{result.ok_count}/{result.chunk_count} ok")
        console.kv(
            "entities",
            f"{int(totals.get('entity_mentions') or 0):,} mention(s) -> "
            f"{int(totals.get('terms') or 0):,} distinct term(s)",
        )
        console.kv(
            "matched",
            f"{int(totals.get('matched_terms') or 0):,} term(s) in "
            f"{int(totals.get('matched_chunks') or 0):,} chunk(s) · "
            f"{int(totals.get('unique_matched_glossary_terms') or 0):,} dictionary entries",
        )
        console.kv("unmatched", f"{int(totals.get('unmatched_terms') or 0):,} term(s)")
        for phase in result.phases or []:
            console.kv(
                phase.get("name", "phase"),
                f"{human_seconds(float(phase.get('seconds') or 0.0))} for "
                f"{int(phase.get('chunks') or 0):,} chunk(s)",
            )
        self._log_paths("manifest", result.manifest_path)
        self._log_paths("trace", result.trace_path)

        if not result.ok:
            detail = "\n".join(f"     - {error}" for error in result.errors)
            raise PhaseFailed(
                f"{len(result.errors)} chunk(s) failed\n{detail}"
                if detail
                else "the chunk-glossary run was not verified",
                summary=f"{result.ok_count}/{result.chunk_count} ok",
            )

        summary = (
            f"{result.chunk_count:,} chunks · "
            f"{int(totals.get('matched_terms') or 0):,} matched term(s) · "
            f"threshold {args.match_threshold}%"
        )
        if args.dry_run:
            summary += " · dry run"
        return summary, result.to_dict()

    # -- phase 7: front/back matter localization -------------------------- #
    def phase_matter(self) -> tuple[str, dict[str, Any]]:
        """Stage 7: localize front/back matter with its own agent, before translation.

        The stage is intentionally independent of the dictionary. It first asks
        one book-level agent for a localization plan (published title if one can
        be found, the BleBel credit, the original-edition material to omit, the
        section titles), then reads every matter chunk through the same
        agent-with-tools loop with that plan binding. The translated chunks are
        written to ``08-translations/`` under the chunk names stage 9 expects, so
        assembly is unchanged.
        """

        args, console = self.args, self.console
        if not has_chunk_files(self.chunks_out):
            raise PhaseFailed(
                f"no chunk files in {rel(self.chunks_out)} — run chunk first",
                summary="no chunks to localize",
            )

        # Matter keeps stage 2's folder, not the cleaned one: it creates the
        # synthetic front/back matter files and appends them to stage 2's
        # ``_manifest.json``, so it needs the folder that manifest belongs to.
        # Its chunk text still comes from ``05-chunks/``, cleaned or not.
        matter_chapters = self.chapters_dir
        console.kv("input", rel(matter_chapters))
        console.kv("chunks", rel(self.chunks_out))
        console.kv("output", rel(self.translations_out))
        console.kv("plan", rel(self.matter_out))
        console.kv("prompts", rel(self.matter_prompts_out))
        console.kv("languages", f"{args.source_language} → {args.target_language}")
        console.kv("model", self._stage_model_label("matter"))
        console.kv(
            "policy",
            "no dictionary · web search for an existing translated title · "
            f"credit {MATTER_TRANSLATOR_NAME} · original-edition material omitted",
        )
        console.kv(
            "resume",
            "reuse finished matter chunks" if args.skip_if_present else "off",
        )

        runtime = self._stage_runtime(
            settings_factory=lambda: self.settings_for("matter"),
            progress=None if getattr(args, "quiet", False) else self._on_matter_chunk,
        )
        result = self.options.stage.call("matter", runtime)()

        plan = result.book_plan or {}
        totals = result.totals or {}
        tokens = result.tokens or {}
        if result.synthetic_sections:
            console.kv(
                "created",
                ", ".join(result.synthetic_sections)
                + " from the input filename",
            )
        console.kv(
            "sections",
            f"{len(result.sections)} · "
            f"{int(totals.get('matter_chunks') or 0):,} chunk(s) in the matter window",
        )
        if result.skipped:
            console.info("  · skipped: the book has no front/back matter chunks")
            return "skipped — no front/back matter", result.to_dict()
        console.kv("title", plan.get("translated_title") or "—")
        console.kv(
            "credit",
            plan.get("translation_credit") or f"Translated by {MATTER_TRANSLATOR_NAME}",
        )
        if plan.get("title_evidence"):
            console.kv("evidence", plan.get("title_evidence"))
        console.kv(
            "omitted",
            f"{len(plan.get('omit') or []):,} source phrase(s)"
            f" · {int(totals.get('omitted') or 0)} chunk(s) omitted",
        )
        console.kv(
            "chunks",
            f"{result.ok_count}/{result.chunk_count} localized"
            + (f" · {result.reused_count} reused" if result.reused_count else ""),
        )
        console.kv(
            "tokens",
            f"{int(tokens.get('total_tokens') or 0):,} total "
            f"({int(tokens.get('input_tokens') or 0):,} in, "
            f"{int(tokens.get('output_tokens') or 0):,} out) over "
            f"{int(tokens.get('llm_calls') or 0):,} call(s)",
        )
        timing = result.timing or {}
        if timing:
            console.kv(
                "time",
                f"{human_seconds(float(timing.get('run_seconds', 0.0)))} "
                f"({float(timing.get('seconds_per_chunk', 0.0)):.1f} s/chunk)",
            )
        self._log_paths("manifest", result.manifest_path)
        self._log_paths("trace", result.trace_path)
        self._log_paths("plan file", self.matter_out / "book-plan.json")

        if not result.ok:
            detail = "\n".join(f"     - {error}" for error in result.errors)
            raise PhaseFailed(
                f"{len(result.errors)} matter chunk(s) failed\n{detail}"
                if detail
                else "the matter localization run was not verified",
                summary=f"{result.ok_count}/{result.chunk_count} localized",
            )

        summary = (
            f"{result.ok_count}/{result.chunk_count} matter chunk(s) · "
            f"title {(plan.get('translated_title') or '—')!r} · "
            f"{int(tokens.get('total_tokens') or 0):,} tokens"
        )
        if result.reused_count:
            summary += f" · {result.reused_count:,} reused"
        return summary, result.to_dict()

    def _on_matter_chunk(
        self, chunk: dict[str, Any], done: int, total: int
    ) -> None:
        """One durable line per localized matter chunk."""
        if chunk.get("reused"):
            flag = "reuse"
        elif chunk.get("omitted"):
            flag = "omit "
        elif chunk.get("ok"):
            flag = "ok   "
        else:
            flag = "FAIL "
        usage = chunk.get("usage") or {}
        self.console.info(
            f"  [{done:>3}/{total}] {flag} {chunk.get('filename') or '?'!s:<46} "
            f"{chunk.get('section_kind') or '?'!s:<5} "
            f"{int(chunk.get('source_chars') or 0):>6,} → "
            f"{int(chunk.get('translation_chars') or 0):>6,} chars · "
            f"{int(usage.get('total_tokens') or 0):>7,} tok · "
            f"{int(chunk.get('attempts') or 0)} attempt(s)"
            + (f" · {chunk.get('omit_reason')}" if chunk.get("omitted") else "")
        )

    # -- phase 8: translate ------------------------------------------------ #
    def phase_translate(self) -> tuple[str, dict[str, Any]]:
        """Stage 7: translate every chunk, glossary first, debug prompt kept.

        Opt-in (``--with-translate``) because it is the paid end product: one
        model call per chunk. The phase reads the chunks and their per-chunk
        glossary, and writes ``07-translations/`` plus ``07-translation-prompts/`` —
        the latter holds the whole input of every call, for debugging.
        """
        from book_translator.core.constants import GLOSSARY_FILENAME
        from book_translator.stages.translation.runner import has_chunk_glossary_files

        args, console = self.args, self.console
        if not has_chunk_files(self.chunks_out):
            raise PhaseFailed(
                f"no chunk files in {rel(self.chunks_out)} — run chunk first",
                summary="no chunks to translate",
            )
        if not has_chunk_glossary_files(self.chunk_glossary_out):
            console.info(
                f"  · skipped: no per-chunk glossary in {rel(self.chunk_glossary_out)} — "
                "run the chunk_glossary phase first"
            )
            return (
                "skipped — no per-chunk glossary to send",
                {"missing_chunk_glossary": str(self.chunk_glossary_out)},
            )
        glossary_path = self.glossary_out / GLOSSARY_FILENAME
        if args.translate_glossary_scope == "book" and not glossary_path.is_file():
            console.info(
                f"  · skipped: no book glossary at {rel(glossary_path)} — run the "
                "dictionary/glossary phase first, or use --translate-glossary-scope chunk"
            )
            return (
                "skipped — no book glossary to send",
                {"missing_glossary": str(glossary_path)},
            )

        console.kv("input", rel(self.chunks_out))
        console.kv("glossary", rel(self.chunk_glossary_out) + f"  ({args.translate_glossary_scope})")
        console.kv("output", rel(self.translations_out))
        console.kv("prompts", rel(self.translation_prompts_out))
        console.kv("languages", f"{args.source_language} → {args.target_language}")
        console.kv("model", self._translation_model_label())
        chapter_files = chapter_files_for_window(
            self.chunks_out, self.translate_first_chapters
        )
        if chapter_files is not None:
            console.kv(
                "window",
                f"chunks of the first {self.translate_first_chapters} body chapter(s)",
            )
        console.kv(
            "limit",
            f"{self.translate_limit} chunk(s) in the window"
            if self.translate_limit
            else "the whole body",
        )
        console.kv(
            "resume",
            "reuse finished chunks" if args.skip_if_present else "off",
        )
        console.kv(
            "matter",
            "localized by stage 7 and skipped here"
            if not args.include_matter
            else "also translated here (--include-matter)",
        )

        try:
            runtime = self._stage_runtime(
                settings_factory=lambda: self.translation_settings,
                progress=None
                if getattr(args, "quiet", False)
                else self._on_translation_chunk,
                chapter_files=chapter_files,
                model_identity=self._translation_model_label(),
            )
            result = self.options.stage.call("translate", runtime)()
        except MissingDependencyError as exc:
            console.info(f"  · skipped: {exc}")
            return (
                "skipped — a translation dependency is not installed",
                {"missing_dependency": str(exc)},
            )

        totals = result.totals or {}
        tokens = result.tokens or {}
        if result.chunk_count == 0:
            console.kv(
                "chunks",
                f"0 body chunks · {result.skipped_count} front/back matter chunk(s) "
                "left to the matter stage",
            )
            self._log_paths("manifest", result.manifest_path)
            self._log_paths("trace", result.trace_path)
            return (
                "no body chunks — front/back matter is localized by stage 7",
                result.to_dict(),
            )
        console.kv(
            "chunks",
            f"{result.ok_count}/{result.chunk_count} ok"
            + (f" · {result.reused_count:,} reused" if result.reused_count else ""),
        )
        if result.provider or result.model:
            console.kv("model used", f"{result.provider or '—'}/{result.model or '—'}")
        console.kv(
            "text",
            f"{int(totals.get('source_chars') or 0):,} → "
            f"{int(totals.get('translation_chars') or 0):,} characters",
        )
        console.kv(
            "tokens",
            f"{int(tokens.get('total_tokens') or 0):,} total "
            f"({int(tokens.get('input_tokens') or 0):,} in, "
            f"{int(tokens.get('output_tokens') or 0):,} out) over "
            f"{int(tokens.get('calls') or 0):,} call(s)",
        )
        timing = result.timing or {}
        if timing:
            console.kv(
                "time",
                f"{human_seconds(float(timing.get('run_seconds', 0.0)))} "
                f"({float(timing.get('seconds_per_chunk', 0.0)):.1f} s/chunk)",
            )
        self._log_paths("manifest", result.manifest_path)
        self._log_paths("trace", result.trace_path)

        if not result.ok:
            detail = "\n".join(f"     - {error}" for error in result.errors)
            raise PhaseFailed(
                f"{len(result.errors)} chunk(s) failed\n{detail}"
                if detail
                else "the translation run was not verified",
                summary=f"{result.ok_count}/{result.chunk_count} ok",
            )

        summary = (
            f"{int(totals.get('translated_chunks') or 0):,} chunk(s) translated · "
            f"{int(tokens.get('total_tokens') or 0):,} tokens · "
            f"{args.source_language} → {args.target_language}"
        )
        if result.reused_count:
            summary += f" · {result.reused_count:,} reused"
        if args.dry_run:
            summary += " · dry run"
        return summary, result.to_dict()

    # -- phase 9: concatenate ---------------------------------------------- #
    def phase_concatenate(self) -> tuple[str, dict[str, Any]]:
        """Stage 8: glue the translated chunks back into one file per chapter.

        Deterministic and cheap, but it runs after the paid translation phase so
        a translated run ends with a readable book. It is also reachable on its
        own with ``--start-at concatenate`` over an earlier translation.
        """

        args, console = self.args, self.console
        if not has_translation_files(self.translations_out):
            console.info(
                f"  · skipped: no translated chunks in {rel(self.translations_out)} — "
                "run the translate phase first"
            )
            return (
                "skipped — no translated chunks to assemble",
                {"missing_translations": str(self.translations_out)},
            )

        console.kv("input", rel(self.translations_out))
        console.kv("chunks", rel(self.chunks_out))
        console.kv("output", rel(self.translated_chapters_out))
        if self.translate_first_chapters is not None:
            console.kv("window", f"first {self.translate_first_chapters} chapter(s)")
        console.kv(
            "resume",
            "reuse finished chapters" if args.skip_if_present else "off",
        )

        runtime = self._stage_runtime(
            progress=None
            if getattr(args, "quiet", False)
            else self._on_assembly_chapter
        )
        result = self.options.stage.call("concatenate", runtime)()

        totals = result.totals or {}
        console.kv(
            "chapters",
            f"{result.ok_count}/{result.chapter_count} written"
            + (f" · {result.reused_count} reused" if result.reused_count else "")
            + (
                f" · {result.incomplete_count} incomplete"
                if result.incomplete_count
                else ""
            ),
        )
        console.kv(
            "chunks",
            f"{int(totals.get('translated_chunks') or 0):,} translated of "
            f"{int(totals.get('chunks') or 0):,} in the window",
        )
        if result.incomplete_count:
            console.kv(
                "incomplete",
                "a chapter is written only when every chunk was translated",
            )
        self._log_paths("manifest", result.manifest_path)
        self._log_paths("trace", result.trace_path)

        if not result.ok:
            detail = "\n".join(f"     - {error}" for error in result.errors)
            raise PhaseFailed(
                f"{len(result.errors)} chapter(s) failed\n{detail}"
                if detail
                else "the assembly run failed",
                summary=f"{result.ok_count}/{result.chapter_count} written",
            )

        summary = (
            f"{result.ok_count}/{result.chapter_count} chapters · "
            f"{int(totals.get('translated_chunks') or 0):,} translated chunk(s)"
        )
        if result.reused_count:
            summary += f" · {result.reused_count} reused"
        if result.incomplete_count:
            summary += f" · {result.incomplete_count} incomplete"
        if args.dry_run:
            summary += " · dry run"
        return summary, result.to_dict()

    # -- phase 10: postprocess --------------------------------------------- #
    def phase_postprocess(self) -> tuple[str, dict[str, Any]]:
        """Stage 9: book-level naming/TOC plan, then one chapter at a time."""

        args, console = self.args, self.console
        if not has_translated_chapter_files(self.translated_chapters_out):
            console.info(
                f"  · skipped: no translated chapters in "
                f"{rel(self.translated_chapters_out)} — run concatenate first"
            )
            return (
                "skipped — no translated chapters to postprocess",
                {"missing_translated_chapters": str(self.translated_chapters_out)},
            )

        console.kv("input", rel(self.translated_chapters_out))
        console.kv("output", rel(self.postprocess_out))
        console.kv("model", self._stage_model_label("postprocess"))
        console.kv(
            "policy",
            "one naming policy from the full title list; mandatory web search for the "
            "published chapter list; chapter openings and heading records only, never prose",
        )
        window = (
            f"first {self.translate_first_chapters} chapter(s)"
            if self.translate_first_chapters is not None
            else "whole translated book"
        )
        console.kv("window", window)

        runtime = self._stage_runtime(
            settings=None if args.dry_run else self.settings_for("postprocess"),
            progress=None
            if getattr(args, "quiet", False)
            else self._on_postprocess_chapter,
        )
        result = self.options.stage.call("postprocess", runtime)()
        totals = result.totals or {}
        toc = result.toc or {}
        console.kv(
            "chapters",
            f"{result.ok_count}/{result.chapter_count} postprocessed"
            + (f" · {result.reused_count} reused" if result.reused_count else ""),
        )
        console.kv(
            "titles",
            f"{int(totals.get('title_set') or 0)} set · "
            f"{int(totals.get('title_inserted') or 0)} inserted",
        )
        console.kv(
            "headings",
            f"{int(totals.get('demoted_headings') or 0)} demoted · "
            f"{int(totals.get('merged_subtitles') or 0)} subtitle(s) folded into a title",
        )
        console.kv(
            "toc",
            f"{toc.get('action') or '—'} · {int(totals.get('toc_entries') or 0)} entries"
            + (f" · {toc.get('filename')}" if toc.get("filename") else ""),
        )
        openings = getattr(result, "openings", {}) or {}
        if openings and not openings.get("skipped"):
            console.kv(
                "openings",
                f"{int(openings.get('inspected_count') or 0)} checked · "
                f"{int(openings.get('fixed_count') or 0)} line(s) removed · "
                f"{int(openings.get('remaining_count') or 0)} need attention",
            )
        web = result.web or {}
        console.kv(
            "web",
            f"{len(web.get('searches') or [])} lookup(s) · "
            + ("search attempted" if web.get("attempted") else "search NOT attempted"),
        )
        tokens = int((result.usage or {}).get("total_tokens", 0) or 0)
        console.kv("llm", f"{tokens:,} tokens")
        self._log_paths("manifest", result.manifest_path)
        self._log_paths("trace", result.trace_path)

        if not result.ok:
            detail = "\n".join(f"     - {error}" for error in result.errors)
            raise PhaseFailed(
                f"{len(result.errors)} error(s)\n{detail}" if detail else "the run failed",
                summary=f"{result.ok_count}/{result.chapter_count} postprocessed",
            )
        summary = (
            f"{result.ok_count}/{result.chapter_count} chapters · "
            f"{int(totals.get('demoted_headings') or 0)} heading(s) demoted · "
            f"toc {toc.get('action') or '—'}"
        )
        if args.dry_run:
            summary += " · dry run"
        return summary, result.to_dict()

    def _on_postprocess_chapter(
        self, chapter: dict[str, Any], done: int, total: int
    ) -> None:
        """One durable line per postprocessed (or reused) chapter."""
        flag = "reuse" if chapter.get("reused") else ("ok   " if chapter.get("ok") else "FAIL ")
        self.console.info(
            f"  [{done:>3}/{total}] {flag} {chapter.get('filename') or '?'!s:<46} "
            f"{chapter.get('title_action') or '—'!s:<8} "
            f"{int(chapter.get('heading_decisions') or 0):>3} heading decision(s) · "
            f"{int(chapter.get('output_lines') or 0):>5} lines"
        )

    # -- phase 11: bind ---------------------------------------------------- #
    def phase_bind(self) -> tuple[str, dict[str, Any]]:
        """Stage 10: concatenate the postprocessed chapters into one book."""

        args, console = self.args, self.console
        from book_translator.documents.chapters import has_translated_chapter_files

        if has_postprocessed_chapter_files(self.postprocess_out):
            source = self.postprocess_out
        elif has_translated_chapter_files(self.translated_chapters_out):
            # `--with-translate` must still leave one readable book when the
            # optional heading pass was not selected.
            source = self.translated_chapters_out
        else:
            console.info(
                f"  · skipped: no postprocessed or translated chapters in "
                f"{rel(self.translated_chapters_out)} — run translate first"
            )
            return (
                "skipped — no chapters to bind",
                {"missing_chapters": str(self.translated_chapters_out)},
            )
        console.kv("input", rel(source))
        console.kv("output", rel(self.translated_book_out))
        runtime = self._stage_runtime(postprocess=source)
        result = self.options.stage.call("bind", runtime)()
        console.kv("files", f"{result.chapter_count} · toc {result.toc_included}")
        console.kv("size", f"{result.chars:,} chars · {result.lines:,} lines")
        self._log_paths("manifest", result.manifest_path)
        self._log_paths("trace", result.trace_path)
        if not result.ok:
            detail = "\n".join(f"     - {error}" for error in result.errors)
            raise PhaseFailed(detail or "the bind run failed", summary="bind failed")
        summary = f"{result.chapter_count} files · {result.chars:,} characters"
        if args.dry_run:
            summary += " · dry run"
        return summary, result.to_dict()

    def phase_epub(self) -> tuple[str, dict[str, Any]]:
        """Stage 11: package the postprocessed chapters as an EPUB 3 book."""

        args, console = self.args, self.console
        from book_translator.documents.chapters import has_translated_chapter_files

        if has_postprocessed_chapter_files(self.postprocess_out):
            source = self.postprocess_out
        elif has_translated_chapter_files(self.translated_chapters_out):
            # Same fallback as stage 11: an EPUB is still worth having when the
            # optional heading pass was not selected.
            source = self.translated_chapters_out
        else:
            console.info(
                f"  · skipped: no postprocessed or translated chapters in "
                f"{rel(self.translated_chapters_out)} — run translate first"
            )
            return (
                "skipped — no chapters to package",
                {"missing_chapters": str(self.translated_chapters_out)},
            )
        console.kv("input", rel(source))
        console.kv("output", rel(self.epub_out))
        try:
            runtime = self._stage_runtime(postprocess=source)
            result = self.options.stage.call("epub", runtime)()
        except ToolError as exc:
            raise PhaseFailed(str(exc), summary="epub failed") from exc
        console.kv(
            "book",
            f"{result.title!r} · {result.language} · {result.document_count} document(s)",
        )
        console.kv(
            "size",
            f"{result.bytes:,} bytes · {result.image_count} picture(s)",
        )
        if result.toc_dropped:
            console.kv("toc", "dropped — the reader's navigation is the table of contents")
        for warning in result.warnings:
            console.info(f"  · warning: {warning}")
        self._log_paths("manifest", result.manifest_path)
        self._log_paths("trace", result.trace_path)
        if not result.ok:
            detail = "\n".join(f"     - {error}" for error in result.errors)
            raise PhaseFailed(detail or "the epub run failed", summary="epub failed")
        summary = (
            f"{result.document_count} document(s) · {result.bytes:,} bytes · "
            f"{result.image_count} picture(s)"
        )
        if args.dry_run:
            summary += " · dry run"
        return summary, result.to_dict()

    def _on_assembly_chapter(
        self, chapter: dict[str, Any], done: int, total: int
    ) -> None:
        """One durable line per assembled (or skipped) chapter."""
        if chapter.get("skipped"):
            flag = "skip"
            tail = f" · {chapter.get('note') or 'not translated'}"
        elif chapter.get("reused"):
            flag = "reuse"
            tail = ""
        else:
            flag = "ok   "
            tail = ""
        self.console.info(
            f"  [{done:>3}/{total}] {flag} {chapter.get('filename') or '?'!s:<46} "
            f"{int(chapter.get('chunk_count') or 0):>3} chunk(s) · "
            f"{int(chapter.get('translation_chars') or 0):>7,} chars{tail}"
        )

    def _on_translation_chunk(
        self, chunk: dict[str, Any], done: int, total: int
    ) -> None:
        """One durable line per translated chunk, with its tokens."""
        if chunk.get("reused"):
            flag = "reuse"
        elif chunk.get("ok"):
            flag = "ok   "
        else:
            flag = "FAIL "
        tokens = chunk.get("tokens") or {}
        self.console.info(
            f"  [{done:>3}/{total}] {flag} {chunk.get('filename') or '?'!s:<46} "
            f"{int(chunk.get('source_chars') or 0):>6,} → "
            f"{int(chunk.get('translation_chars') or 0):>6,} chars · "
            f"{int(tokens.get('total_tokens') or 0):>7,} tok · "
            f"{int(chunk.get('attempts') or 0)} attempt(s)"
        )

    def _recorded_strategy(self, manifest: dict[str, Any]) -> str:
        """Which strategy won.

        Stage 1's manifest does not carry it (it records line ranges and hashes),
        so fall back to the strategy in ``_agent_trace.json``.
        """
        from book_translator.core.constants import TRACE_FILENAME

        if manifest.get("strategy"):
            return str(manifest["strategy"])
        trace = self.chapters_dir / TRACE_FILENAME
        if not trace.is_file():
            return "—"
        try:
            payload = json.loads(trace.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return "—"
        return str((payload.get("result") or {}).get("strategy") or "—")

    def _log_paths(self, label: str, path: str | Path | None) -> None:
        if path and Path(path).exists():
            self.console.debug(f"      {label}: {rel(path)}")

    # -- closing report ---------------------------------------------------- #
