"""Pipeline reporting, banner generation, next-steps diagnostics and terminal UI."""

from __future__ import annotations

import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..conversion import is_markdown_source
from ..core.constants import (
    ASSEMBLY_MANIFEST_FILENAME,
    ASSEMBLY_TRACE_FILENAME,
    CHUNK_GLOSSARY_MANIFEST_FILENAME,
    CHUNK_GLOSSARY_TRACE_FILENAME,
    CHUNK_MANIFEST_FILENAME,
    CHUNK_TRACE_FILENAME,
    CONVERT_MANIFEST_FILENAME,
    EPUB_MANIFEST_FILENAME,
    EPUB_TRACE_FILENAME,
    GLOSSARY_FILENAME,
    GLOSSARY_MANIFEST_FILENAME,
    GLOSSARY_MARKDOWN_FILENAME,
    GLOSSARY_TRACE_FILENAME,
    MANIFEST_FILENAME,
    MATTER_MANIFEST_FILENAME,
    MATTER_TRACE_FILENAME,
    MATTER_TRANSLATOR_NAME,
    PREPROCESS_MANIFEST_FILENAME,
    PREPROCESS_TRACE_FILENAME,
    REDUCED_MANIFEST_FILENAME,
    REDUCED_TRACE_FILENAME,
    TRACE_FILENAME,
    TRANSLATION_MANIFEST_FILENAME,
    TRANSLATION_TRACE_FILENAME,
)
from ..core.report import human_bytes, human_seconds
from ..core.settings import LLM_CONFIG_FILENAME, llm_config_path
from ..documents.io import load_lines, source_size_bytes
from ..paths import book_slug
from .console import RULE
from .report import PhaseRecord, PhaseState

if TYPE_CHECKING:
    from .console import Console
    from .driver import PipelineDriver


def rel(path: str | Path) -> str:
    """A human-readable path string relative to cwd when possible."""
    try:
        return str(Path(path).relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def print_banner(driver: PipelineDriver, phases: list[str]) -> None:
    """Print the startup banner and configured pipeline stages."""
    console = driver.console
    console.log_only(f"# run_pipeline.py · {datetime.now().isoformat(timespec='seconds')}")
    console.log_only(f"# argv: {' '.join(sys.argv[1:]) or '(none)'}")
    console.log_only("")
    console.info(RULE)
    console.info(" book_translator · the whole pipeline, one command")
    console.info(RULE)
    try:
        size = source_size_bytes(driver.source)
        if is_markdown_source(driver.source):
            lines = len(load_lines(driver.source))
            console.kv("book", f"{rel(driver.source)} — {human_bytes(size)}, {lines:,} lines")
        else:
            console.kv(
                "book",
                f"{rel(driver.source)} — {human_bytes(size)}, "
                f"{driver.source_kind} (stage 0 converts it to Markdown)",
            )
    except OSError as exc:
        console.kv("book", f"{rel(driver.source)} — unreadable: {exc}")
    console.kv("output", rel(driver.book_dir))
    model_config = llm_config_path(driver.args.llm_config)
    console.kv(
        "llm config",
        rel(model_config)
        if model_config
        else f"built-in defaults (no {LLM_CONFIG_FILENAME})",
    )
    if driver.convert_needed:
        console.kv("stage 0", rel(driver.markdown_out) + "/  (converted source)")
    console.kv("chapters", rel(driver.chapters_dir))
    console.kv("reduced", rel(driver.reduced_out))
    console.kv("unique", rel(driver.unique_out))
    console.kv("preprocessed", rel(driver.preprocessed_out))
    console.kv("chunks", rel(driver.chunks_out))
    console.kv("chunk gloss", rel(driver.chunk_glossary_out))
    console.kv("translations", rel(driver.translations_out))
    console.kv("prompts", rel(driver.translation_prompts_out))
    console.kv("matter", rel(driver.matter_out))
    console.kv("matter prompts", rel(driver.matter_prompts_out))
    console.kv("translated chapters", rel(driver.translated_chapters_out))
    console.kv("postprocessed", rel(driver.postprocess_out))
    console.kv("book", rel(driver.translated_book_out))
    if not driver.chunk_enabled:
        console.kv("stage 5", "chunking switched off — remove --no-chunk to run it")
    else:
        console.kv(
            "chunk size",
            f"{driver.args.chunk_size:,} tokens · tokenizer {driver.args.chunk_tokenizer}",
        )
    if not driver.chunk_glossary_enabled:
        console.kv(
            "stage 6",
            "chunk glossary switched off — remove --no-chunk-glossary to run it",
        )
    else:
        console.kv(
            "match",
            f"{driver.args.match_threshold}% · scorer {driver.args.match_scorer}",
        )
    if not driver.translate_enabled:
        console.kv(
            "stage 7",
            "translation switched off — add --with-translate to translate the chunks",
        )
    else:
        console.kv(
            "translate",
            f"{driver.args.source_language} → {driver.args.target_language} · "
            f"glossary {driver.args.translate_glossary_scope} · "
            f"limit {driver.translate_limit if driver.translate_limit else 'all chunks'}",
        )
    if not driver.matter_enabled:
        console.kv(
            "stage 6b",
            "front/back matter localization switched off — add --with-matter "
            "or --start-at matter",
        )
    else:
        console.kv(
            "matter",
            "front/back matter localized without a dictionary · "
            f"credit {MATTER_TRANSLATOR_NAME} · web title search on",
        )
    if driver.first_chapters is not None:
        console.kv(
            "chapter window",
            f"first {driver.first_chapters} chapter(s) for every stage after split "
            "(split itself is always the whole book)",
        )
    if (
        driver.translate_first_chapters is not None
        and driver.translate_first_chapters != driver.first_chapters
    ):
        console.kv(
            "translation window",
            f"first {driver.translate_first_chapters} chapter(s)",
        )
    if not driver.concat_enabled:
        console.kv(
            "stage 8",
            "assembly switched off — add --with-translate or --start-at concatenate",
        )
    else:
        console.kv("stage 8", rel(driver.translated_chapters_out))
    if not driver.postprocess_enabled:
        console.kv(
            "stage 9",
            "postprocess switched off — add --with-postprocess or --start-at postprocess",
        )
    else:
        console.kv("stage 9", rel(driver.postprocess_out))
    if not driver.bind_enabled:
        console.kv("stage 10", "bind switched off — remove --no-bind to write one book file")
    else:
        console.kv("stage 10", rel(driver.translated_book_out))
    if not driver.epub_enabled:
        console.kv("stage 11", "epub switched off — remove --no-epub to write one")
    else:
        console.kv("stage 11", rel(driver.epub_out))
    console.kv("phases", " → ".join(phases) or "(none — every selected phase is switched off)")
    if not driver.preprocess_enabled:
        console.kv("stage 2", "preprocess switched off — remove --no-preprocess to run it")
    if not driver.dictionary_enabled:
        console.kv("stage 3", "dictionary switched off — --with-dictionary to run it")
    elif driver.glossary_enabled:
        console.kv("glossary", rel(driver.glossary_out) + "  (built by the dictionary phase)")
    else:
        console.kv("glossary", "off — --no-glossary is set (reduction only)")
    mode = "DRY RUN — nothing is written" if driver.args.dry_run else "live"
    console.kv("mode", mode)
    if console.log_path is not None:
        console.kv("log", rel(console.log_path))


def print_finish_header(console: Console, total: float, aborted: bool, records_count: int, selected_count: int) -> None:
    label = "STOPPED" if aborted else "DONE"
    console.log_only("")
    console.log_only(RULE)
    console.log_only(
        f" {label} · {human_seconds(total)} total · "
        f"{records_count}/{selected_count} phase(s) run"
    )
    console.log_only(RULE)

    if console.quiet:
        return

    rich_c = getattr(console, "_rich_console", None)
    if getattr(console, "is_rich", False) and rich_c is not None:
        from rich.panel import Panel

        border = "red" if aborted else "green"
        text = f"[bold {border}]{label}[/bold {border}] · {human_seconds(total)} total · {records_count}/{selected_count} phase(s) run"
        rich_c.print(Panel(text, border_style=border, expand=False))
    else:
        console.info("")
        console.info(RULE)
        console.info(
            f" {label} · {human_seconds(total)} total · "
            f"{records_count}/{selected_count} phase(s) run"
        )
        console.info(RULE)


def print_phase_table(console: Console, records: list[PhaseRecord]) -> None:
    for record in records:
        console.log_only(
            f"  {'✓' if record.ok else '✗'} {record.name:<11} "
            f"{human_seconds(record.seconds):>9}   {record.summary}"
        )
        if record.error:
            console.log_only(f"      {record.error}")
    console.log_only("")

    if console.quiet:
        return

    rich_c = getattr(console, "_rich_console", None)
    if getattr(console, "is_rich", False) and rich_c is not None:
        from rich.table import Table

        table = Table(title="Pipeline Summary", border_style="dim", show_header=True)
        table.add_column("Status", justify="center", width=8)
        table.add_column("Phase", style="bold cyan", width=16)
        table.add_column("Duration", justify="right", width=12)
        table.add_column("Summary")

        for record in records:
            status = "[green]✓ OK[/green]" if record.ok else "[red]✗ FAIL[/red]"
            summary_text = record.summary
            if record.error:
                summary_text = f"{summary_text} ([red]{record.error}[/red])" if summary_text else f"[red]{record.error}[/red]"
            table.add_row(
                status,
                record.name,
                human_seconds(record.seconds),
                summary_text or "",
            )
        rich_c.print(table)
        rich_c.print()
    else:
        for record in records:
            console.info(
                f"  {'✓' if record.ok else '✗'} {record.name:<11} "
                f"{human_seconds(record.seconds):>9}   {record.summary}"
            )
            if record.error:
                console.error(f"      {record.error}")
        console.info("")


def print_artifacts(driver: PipelineDriver, console: Console) -> None:
    console.log_only("  artifacts")
    if driver.convert_needed and not driver.args.dry_run:
        console.log_only(f"    markdown      {rel(driver.markdown_out)}/")
    console.log_only(f"    chapters      {rel(driver.chapters_dir)}/")
    console.log_only(f"    reduced       {rel(driver.reduced_out)}/")
    console.log_only(f"    unique        {rel(driver.unique_out)}/")
    console.log_only(f"    glossary      {rel(driver.glossary_out)}/")
    console.log_only(f"    preprocessed  {rel(driver.preprocessed_out)}/")
    console.log_only(f"    chunks        {rel(driver.chunks_out)}/")
    console.log_only(f"    chunk gloss   {rel(driver.chunk_glossary_out)}/")
    console.log_only(f"    translations  {rel(driver.translations_out)}/")
    console.log_only(f"    prompts       {rel(driver.translation_prompts_out)}/")
    console.log_only(f"    matter        {rel(driver.matter_out)}/")
    console.log_only(f"    matter prompts {rel(driver.matter_prompts_out)}/")
    console.log_only(f"    translated ch {rel(driver.translated_chapters_out)}/")
    console.log_only(f"    postprocessed {rel(driver.postprocess_out)}/")
    console.log_only(f"    book          {rel(driver.translated_book_out)}")
    if driver.epub_enabled:
        console.log_only(f"    epub          {rel(driver.epub_out)}")
    if console.log_path is not None:
        console.log_only(f"    log           {rel(console.log_path)}")
    for label, path in find_artifact_files(driver):
        console.log_only(f"    {label:<16} {rel(path)}")
    console.log_only("")

    if console.quiet:
        return

    rich_c = getattr(console, "_rich_console", None)
    if getattr(console, "is_rich", False) and rich_c is not None:
        from rich.tree import Tree

        tree = Tree(f"[bold cyan]Artifacts[/bold cyan] [dim]({rel(driver.book_dir)})[/dim]")
        if driver.convert_needed and not driver.args.dry_run:
            tree.add(f"[yellow]00-markdown[/yellow] → {rel(driver.markdown_out)}/")
        tree.add(f"[blue]01-chapters[/blue] → {rel(driver.chapters_dir)}/")
        tree.add(f"[blue]02-preprocessed[/blue] → {rel(driver.preprocessed_out)}/")
        tree.add(f"[blue]03-reduced[/blue] → {rel(driver.reduced_out)}/")
        tree.add(f"[blue]04-glossary[/blue] → {rel(driver.glossary_out)}/")
        tree.add(f"[blue]05-chunks[/blue] → {rel(driver.chunks_out)}/")
        tree.add(f"[blue]06-chunk-glossary[/blue] → {rel(driver.chunk_glossary_out)}/")
        if driver.matter_enabled:
            tree.add(f"[magenta]06b-matter[/magenta] → {rel(driver.matter_out)}/")
        if driver.translate_enabled:
            tree.add(f"[green]07-translations[/green] → {rel(driver.translations_out)}/")
            tree.add(f"[green]08-translated-chapters[/green] → {rel(driver.translated_chapters_out)}/")
        if driver.postprocess_enabled:
            tree.add(f"[cyan]09-postprocess[/cyan] → {rel(driver.postprocess_out)}/")
        if driver.bind_enabled:
            tree.add(f"[bold green]10-translated-book.md[/bold green] → {rel(driver.translated_book_out)}")
        if driver.epub_enabled:
            tree.add(f"[bold magenta]11-translated-book.epub[/bold magenta] → {rel(driver.epub_out)}")
        if console.log_path is not None:
            tree.add(f"[dim]pipeline.log[/dim] → {rel(console.log_path)}")
        rich_c.print(tree)
        rich_c.print()
    else:
        console.info("  artifacts")
        if driver.convert_needed and not driver.args.dry_run:
            console.info(f"    markdown      {rel(driver.markdown_out)}/")
        console.info(f"    chapters      {rel(driver.chapters_dir)}/")
        console.info(f"    reduced       {rel(driver.reduced_out)}/")
        console.info(f"    unique        {rel(driver.unique_out)}/")
        console.info(f"    glossary      {rel(driver.glossary_out)}/")
        console.info(f"    preprocessed  {rel(driver.preprocessed_out)}/")
        console.info(f"    chunks        {rel(driver.chunks_out)}/")
        console.info(f"    chunk gloss   {rel(driver.chunk_glossary_out)}/")
        console.info(f"    translations  {rel(driver.translations_out)}/")
        console.info(f"    prompts       {rel(driver.translation_prompts_out)}/")
        console.info(f"    matter        {rel(driver.matter_out)}/")
        console.info(f"    matter prompts {rel(driver.matter_prompts_out)}/")
        console.info(f"    translated ch {rel(driver.translated_chapters_out)}/")
        console.info(f"    postprocessed {rel(driver.postprocess_out)}/")
        console.info(f"    book          {rel(driver.translated_book_out)}")
        if driver.epub_enabled:
            console.info(f"    epub          {rel(driver.epub_out)}")
        if console.log_path is not None:
            console.info(f"    log           {rel(console.log_path)}")
        for label, path in find_artifact_files(driver):
            console.info(f"    {label:<16} {rel(path)}")
        console.info("")


def derive_phase_state(records: list[PhaseRecord]) -> PhaseState:
    """Derive the closing state from the phase records in one pass."""

    def payloads(name: str, *, ok_only: bool = True) -> list[dict[str, Any]]:
        return [
            record.payload or {}
            for record in records
            if record.name == name and (record.ok or not ok_only)
        ]

    chunk_payloads = payloads("chunk")
    dictionary_payloads = payloads("dictionary")
    matter_payloads = payloads("matter")
    return PhaseState(
        cleaned=any(record.name == "preprocess" and record.ok for record in records),
        chunked=any(
            not payload.get("missing_dependency") and int(payload.get("chunk_count") or 0) > 0
            for payload in chunk_payloads
        ),
        chunk_missing=any(
            bool(payload.get("missing_dependency"))
            for payload in payloads("chunk", ok_only=False)
        ),
        chunk_glossaried=any(
            not payload.get("missing_dependency") and int(payload.get("chunk_count") or 0) > 0
            for payload in payloads("chunk_glossary")
        ),
        translated=any(
            int(payload.get("chunk_count") or 0) > 0 for payload in payloads("translate")
        ),
        mattered=any(int(payload.get("chunk_count") or 0) > 0 for payload in matter_payloads),
        matter_plan=next(
            (
                payload.get("book_plan") or {}
                for payload in matter_payloads
                if isinstance(payload.get("book_plan"), dict)
            ),
            {},
        ),
        concatenated=any(
            int(payload.get("ok_count") or 0) > 0 or int(payload.get("reused_count") or 0) > 0
            for payload in payloads("concatenate")
        ),
        postprocessed=any(
            int(payload.get("ok_count") or 0) > 0 for payload in payloads("postprocess")
        ),
        bound=any(int(payload.get("chapter_count") or 0) > 0 for payload in payloads("bind")),
        epubbed=any(int(payload.get("document_count") or 0) > 0 for payload in payloads("epub")),
        glossaried=any(
            isinstance(payload.get("glossary"), dict)
            and not (payload.get("glossary") or {}).get("dry_run")
            for payload in dictionary_payloads
        ),
        reduced=any(not payload.get("missing_dependency") for payload in dictionary_payloads),
        tagger_missing=any(
            bool(payload.get("missing_dependency"))
            for record in records
            for payload in [record.payload or {}]
        ),
    )


def print_next_steps(driver: PipelineDriver, console: Console, state: PhaseState, aborted: bool) -> None:
    if aborted:
        console.info(
            "  stopped — fix the error above and re-run; the phases before it are done"
        )
        return
    if driver.args.dry_run:
        console.info("  dry run — every plan was validated, no chapter file was written")
        return
    if state.epubbed:
        console.info(
            f"  next: {rel(driver.epub_out)} is the translated book — open it in any EPUB reader"
        )
    elif state.bound:
        console.info(
            f"  next: {rel(driver.translated_book_out)} is the whole translated book — "
            "one Markdown file assembled from the postprocessed chapters"
        )
    elif state.postprocessed:
        console.info(
            f"  next: {rel(driver.postprocess_out)}/ holds the postprocessed chapters — "
            "run bind (or --start-at bind) to concatenate them into one book"
        )
    elif state.concatenated:
        console.info(
            f"  next: {rel(driver.translated_chapters_out)}/ holds the translated "
            "book, one file per chapter — the source chapters were not touched"
        )
        if driver.translation_prompts_out.is_dir():
            console.info(
                f"  debug: {rel(driver.translation_prompts_out)}/ holds the whole "
                "prompt of every chunk (system, user, glossary and source text)"
            )
        if not driver.postprocess_enabled:
            console.info(
                "  next: make the headings consistent with --with-postprocess "
                "(or --start-at postprocess)"
            )
    elif state.translated:
        console.info(
            f"  next: {rel(driver.translations_out)}/ holds the translated chunks — "
            f"run --start-at concatenate to rebuild the translated chapters"
        )
        console.info(
            f"  debug: {rel(driver.translation_prompts_out)}/ holds the whole prompt "
            "of every chunk (system, user, glossary and source text)"
        )
    elif state.mattered:
        console.info(
            f"  next: {rel(driver.translations_out)}/ holds the localized front/back "
            "matter — run --with-translate (or --start-at translate) for the body"
        )
        console.info(
            f"  plan: {rel(driver.matter_out)}/book-plan.json records the localized "
            f"title, the BleBel credit and the "
            f"{len(state.matter_plan.get('omit') or [])} omitted source phrase(s)"
        )
    elif state.chunk_glossaried:
        console.info(
            f"  next: {rel(driver.chunk_glossary_out)}/ holds one glossary file per chunk — "
            "send a chunk with its matched terms to keep the translation consistent"
        )
    elif state.chunk_missing:
        console.info(
            "  next: install the chunker (pip install -r requirements-chunk.txt) "
            "and re-run --start-at chunk"
        )
    elif state.chunked:
        console.info(
            f"  next: {rel(driver.chunks_out)}/ holds the reversible chunks — run "
            "--with-translate (or --start-at chunk_glossary) to continue"
        )
    elif state.cleaned:
        console.info(
            f"  next: {rel(driver.preprocessed_out)}/ holds the cleaned chapters — "
            "stages 3–5 will read them by default"
        )
    elif state.glossaried:
        console.info(
            f"  next: {rel(driver.glossary_out)}/{GLOSSARY_FILENAME} is the book glossary — "
            "run --start-at chunk to cut the chapters into translation-sized chunks"
        )
    elif state.reduced:
        console.info(
            f"  next: {rel(driver.unique_out)}/ holds the unique sentences — "
            "run glossary (or --start-at glossary) to turn them into a book dictionary"
        )
    else:
        console.info(
            f"  next: {rel(driver.chapters_dir)}/ holds the chapter files — "
            "clean them with preprocess (or re-run without --no-preprocess)"
        )

    if state.tagger_missing:
        console.info(
            "  note: install stage 3's tagger (pip install -r "
            "requirements-dictionary.txt) and re-run --start-at dictionary"
        )
    if not driver.preprocess_enabled:
        console.info(
            "  stage 2 (preprocess) is switched off — remove --no-preprocess to run it"
        )
    if not driver.translate_enabled:
        console.info(
            "  stage 7 (translation) is switched off — --with-translate to translate "
            "the chunks (one paid model call per chunk)"
        )


def find_artifact_files(driver: PipelineDriver) -> list[tuple[str, str | Path]]:
    """Manifest/trace files written by whichever phases ran, for the report."""
    candidates = [
        ("stage-0 markdown", driver.markdown_out / f"{book_slug(driver.original_source)}.md"),
        ("stage-0 manifest", driver.markdown_out / CONVERT_MANIFEST_FILENAME),
        ("split trace", driver.chapters_dir / TRACE_FILENAME),
        ("split manifest", driver.chapters_dir / MANIFEST_FILENAME),
        ("stage-3 manifest", driver.reduced_out / REDUCED_MANIFEST_FILENAME),
        ("stage-3 trace", driver.reduced_out / REDUCED_TRACE_FILENAME),
        ("stage-4 glossary", driver.glossary_out / GLOSSARY_FILENAME),
        ("stage-4 markdown", driver.glossary_out / GLOSSARY_MARKDOWN_FILENAME),
        ("stage-4 manifest", driver.glossary_out / GLOSSARY_MANIFEST_FILENAME),
        ("stage-4 trace", driver.glossary_out / GLOSSARY_TRACE_FILENAME),
        ("stage-2 manifest", driver.preprocessed_out / PREPROCESS_MANIFEST_FILENAME),
        ("stage-2 trace", driver.preprocessed_out / PREPROCESS_TRACE_FILENAME),
        ("stage-5 manifest", driver.chunks_out / CHUNK_MANIFEST_FILENAME),
        ("stage-5 trace", driver.chunks_out / CHUNK_TRACE_FILENAME),
        ("stage-6 manifest", driver.chunk_glossary_out / CHUNK_GLOSSARY_MANIFEST_FILENAME),
        ("stage-6 trace", driver.chunk_glossary_out / CHUNK_GLOSSARY_TRACE_FILENAME),
        ("stage-6b plan", driver.matter_out / "book-plan.json"),
        ("stage-6b manifest", driver.matter_out / MATTER_MANIFEST_FILENAME),
        ("stage-6b trace", driver.matter_out / MATTER_TRACE_FILENAME),
        ("stage-7 manifest", driver.translations_out / TRANSLATION_MANIFEST_FILENAME),
        ("stage-7 trace", driver.translations_out / TRANSLATION_TRACE_FILENAME),
        ("stage-8 manifest", driver.translated_chapters_out / ASSEMBLY_MANIFEST_FILENAME),
        ("stage-8 trace", driver.translated_chapters_out / ASSEMBLY_TRACE_FILENAME),
        (
            "stage-9 manifest",
            driver.postprocess_out / "_postprocess_manifest.json",
        ),
        ("stage-9 trace", driver.postprocess_out / "_postprocess_trace.json"),
        ("stage-10 manifest", driver.translated_book_out.parent / "_bind_manifest.json"),
        ("stage-10 trace", driver.translated_book_out.parent / "_bind_trace.json"),
        ("stage-11 manifest", driver.epub_out.parent / EPUB_MANIFEST_FILENAME),
        ("stage-11 trace", driver.epub_out.parent / EPUB_TRACE_FILENAME),
    ]
    return [(label, path) for label, path in candidates if path.is_file()]


def build_summary_json(driver: PipelineDriver, total: float) -> dict[str, Any]:
    return {
        "book": str(driver.source),
        "source": str(driver.original_source),
        "markdown_dir": str(driver.markdown_out),
        "markdown_path": (
            str(driver.source)
            if not is_markdown_source(driver.original_source) and not driver.args.dry_run
            else None
        ),
        "output_dir": str(driver.book_dir),
        "chapters_dir": str(driver.chapters_dir),
        "reduced_dir": str(driver.reduced_out),
        "unique_dir": str(driver.unique_out),
        "glossary_dir": str(driver.glossary_out),
        "preprocessed_dir": str(driver.preprocessed_out),
        "text_input": str(driver.text_input()),
        "text_source": driver._text_choice()[1],
        "chunks_dir": str(driver.chunks_out),
        "chunk_glossary_dir": str(driver.chunk_glossary_out),
        "translations_dir": str(driver.translations_out),
        "translation_prompts_dir": str(driver.translation_prompts_out),
        "matter_dir": str(driver.matter_out),
        "matter_prompts_dir": str(driver.matter_prompts_out),
        "translated_chapters_dir": str(driver.translated_chapters_out),
        "postprocessed_chapters_dir": str(driver.postprocess_out),
        "translated_book_path": str(driver.translated_book_out),
        "translated_epub_path": str(driver.epub_out),
        "first_chapters": driver.first_chapters,
        "translate_first_chapters": driver.translate_first_chapters,
        "chapter_limit": driver.chapter_limit,
        "dry_run": bool(driver.args.dry_run),
        "start_at": driver.args.start_at,
        "stop_after": driver.args.stop_after,
        "ok": not any(not record.ok for record in driver.records),
        "elapsed_seconds": round(total, 3),
        "log": str(driver.console.log_path) if driver.console.log_path else None,
        "phases": [
            {
                "name": record.name,
                "ok": record.ok,
                "seconds": round(record.seconds, 3),
                "summary": record.summary,
                "error": record.error,
                "result": record.payload,
            }
            for record in driver.records
        ],
    }


class GlossaryProgressBar:
    """Manages the background loop and terminal updates for glossary building."""

    HEARTBEAT_SECONDS = 10.0

    def __init__(self, console: Console) -> None:
        self.console = console
        self.state: dict[str, Any] = {}
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def on_start(self, chapter: dict[str, Any]) -> None:
        self.state.update(
            {
                "current": str(chapter.get("filename") or ""),
                "chapter_started": time.monotonic(),
                "done": int(chapter.get("done") or 0),
                "total": int(chapter.get("total") or 0),
            }
        )
        if self.thread is None or not self.thread.is_alive():
            self.state["started"] = time.monotonic()
            self.state["durations"] = []
            self.state["active"] = True
            self.stop_event.clear()
            self.thread = threading.Thread(
                target=self._loop, name="glossary-bar", daemon=True
            )
            self.thread.start()

    def on_chapter(self, chapter: dict[str, Any], done: int, total: int) -> None:
        now = time.monotonic()
        elapsed = now - float(self.state.get("chapter_started") or now)
        durations = self.state.setdefault("durations", [])
        durations.append(elapsed)
        self.state["done"], self.state["total"], self.state["current"] = done, total, ""
        tokens = chapter.get("tokens") or {}
        self.console.clear_bar()
        self.console.info(
            f"  [{done:>2}/{total}] {chapter.get('filename') or '?'!s:<44} "
            f"{int(chapter.get('added') or 0):>3} new · "
            f"{int(chapter.get('repeats') or 0):>2} locked · "
            f"{int(chapter.get('conflicts') or 0):>2} conflict · "
            f"{int(tokens.get('total_tokens') or 0):>7,} tok · {elapsed:5.1f}s"
        )

    def _line(self) -> str:
        total = int(self.state.get("total") or 0)
        if total <= 0:
            return "  starting the glossary run…"
        done = int(self.state.get("done") or 0)
        now = time.monotonic()
        elapsed = now - float(self.state.get("started") or now)
        durations = self.state.get("durations") or []
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
        current = str(self.state.get("current") or "")
        if current:
            chapter_elapsed = now - float(self.state.get("chapter_started") or now)
            parts.append(f"now {current} ({chapter_elapsed:.0f}s)")
        return " · ".join(parts)

    def _loop(self) -> None:
        last_heartbeat = 0.0
        while not self.stop_event.wait(1.0):
            if not self.state.get("active"):
                continue
            line = self._line()
            if self.console.bar_enabled:
                self.console.bar(line)
                continue
            now = time.monotonic()
            if now - last_heartbeat >= self.HEARTBEAT_SECONDS:
                last_heartbeat = now
                self.console.stream_line(line)

    def stop(self) -> None:
        self.state["active"] = False
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
        self.console.clear_bar()


class PipelineReporter:
    """Orchestrates all narration and artifact reporting for a pipeline run."""

    def __init__(self, driver: PipelineDriver) -> None:
        self.driver = driver
        self.console = driver.console
        self.glossary_bar = GlossaryProgressBar(self.console)

    def banner(self, phases: list[str]) -> None:
        print_banner(self.driver, phases)

    def finish(self, total: float, aborted: bool) -> int:
        print_finish_header(
            self.console,
            total,
            aborted,
            len(self.driver.records),
            len(self.driver._selected_phases()),
        )
        print_phase_table(self.console, self.driver.records)
        print_artifacts(self.driver, self.console)
        state = derive_phase_state(self.driver.records)
        print_next_steps(self.driver, self.console, state, aborted)
        if self.driver.args.json:
            self.console.json(build_summary_json(self.driver, total))
        if self.driver.exit_code:
            return self.driver.exit_code
        return 1 if aborted else 0


__all__ = [
    "GlossaryProgressBar",
    "PipelineReporter",
    "build_summary_json",
    "derive_phase_state",
    "find_artifact_files",
    "print_artifacts",
    "print_banner",
    "print_finish_header",
    "print_next_steps",
    "print_phase_table",
    "rel",
]
