"""Command line interface for the multi-stage book pipeline.

Documents live in dedicated folders: sources are read from ``data/input/`` and
every generated artifact is written under ``data/output/<book>/``. Both can be
overridden with ``--in-dir`` / ``--out``.

Stage 0 (`convert`) turns an EPUB, a text-based PDF or a TXT into the Markdown
the rest of the pipeline reads: docling does EPUB/PDF (an image-based PDF is
rejected instead of OCR'd), and a TXT is copied unchanged under a ``.md`` name.
It writes to ``00-markdown/``. `inspect` and `split` run it automatically when
the source is not Markdown, so every later stage keeps reading plain Markdown.

Stage 1 turns the book into chapter files; stage 2 cleans each chapter, one
chapter at a time; stage 3 reduces each chapter to the raw sentences that carry a
named entity and then to the sentences that introduce a noun; stage 4 is the LLM
pass that folder was built for — it walks the first mentions first to last and
locks in one target-language rendering per entity, feeding each chapter the
dictionary the earlier ones produced; stage 5 cuts each chapter into
translation-sized chunks without breaking paragraphs or sentences, with the
chapter and order of every chunk recorded so the text can be concatenated back;
stage 6 runs NER over every chunk first, then fuzzy-matches each extracted term
against the glossary and writes the relevant terms per chunk under the chunk's
own name; stage 6b (`matter`) is a separate LangGraph agent that localizes front
and back matter before translation — it uses no dictionary, searches the web for
an existing translated title, omits original-edition imprint material, credits
BleBel and formats the result as published front/back matter; stage 7 translates
the body chunks in reading order, sends each one the glossary terms it mentions,
validates a Pydantic translation answer (retrying a failed call), writes the
translated chunk under the chunk's own name and keeps the whole prompt of every
call in a separate debug folder; stage 8 concatenates the translated chunks back
into one file per chapter under ``08-translated-chapters/``, using stage 5's chapter
membership and chunk order; stage 9 postprocesses those translated chapters'
headings and table of contents, one chapter at a time and without reading their
prose, into ``09-postprocessed-chapters/``; stage 10 concatenates that folder into
one ``10-translated-book.md``; stage 11 (`epub`) packages the same folder as an
EPUB 3 book, ``11-translated-book.epub`` — one XHTML document per chapter, a
navigation document instead of the printed table of contents, the referenced
pictures embedded once, and the metadata stage 6b recorded.

Both agent stages end with a deterministic **opening audit** (`openings`): it
walks the first lines of every chapter, removes a stray copy of the chapter
number under the title, a duplicated title line or a leftover page number, and
reports what still needs a human. `book_splitter openings` runs that same pass on
its own, with no model and no key, over any chapter folder.

The body stages after splitting accept ``--first-chapters N``: they run over the
first N body chapters only. The front/back matter localizer (``matter``) always
works on the whole matter set, because it is not a chapter window. Splitting
itself always runs over the whole book, because
its chapter boundaries are the coordinate system every later stage reads. When a
stage's unit is a chunk (``chunk-glossary``, ``translate``), N is translated into
the matching chunk prefix through stage 5's manifest, so a chapter window never
stops mid-chapter. ``--skip-if-present`` (``--resume``) makes the same command
reuse the work already on disk; increasing N therefore processes only the new
chapters.

Examples::

    python -m book_splitter convert mybook.epub            # -> data/output/mybook/00-markdown/
    python -m book_splitter convert scan.pdf --dry-run     # text-layer check, no writes
    python -m book_splitter inspect                       # report on the only book in data/input
    python -m book_splitter inspect mybook.md --candidates
    python -m book_splitter split mybook.md               # -> data/output/mybook/01-chapters/
    python -m book_splitter split mybook.md --out /tmp/out
    python -m book_splitter split mybook.md --dry-run
    python -m book_splitter split mybook.md --skip-if-present   # reuse existing chapters
    python -m book_splitter split mybook.md --map my-map.json   # no LLM, no network
    python -m book_splitter preprocess mybook              # -> data/output/mybook/02-preprocessed/
    python -m book_splitter preprocess mybook --dry-run
    python -m book_splitter scan mybook                    # damage profile, no LLM, no writes
    python -m book_splitter dictionary mybook              # -> 03-reduced/ and 03-reduced-unique/
    python -m book_splitter dictionary mybook --language sr --gpu 1
    python -m book_splitter dictionary mybook --skip-if-present --first-chapters 3
    python -m book_splitter dictionary mybook --unique-only     # only the first mentions
    python -m book_splitter glossary mybook                # -> 04-glossary/ (en -> sr)
    python -m book_splitter glossary mybook -s sr -t en    # the other direction
    python -m book_splitter glossary mybook --first-chapters 2 --dry-run  # price it first
    python -m book_splitter glossary mybook --resume       # continue where it stopped
    python -m book_splitter chunk mybook                   # -> 05-chunks/ (1000-token default)
    python -m book_splitter chunk mybook --chunk-size 700  # smaller translation units
    python -m book_splitter chunk mybook --dry-run         # verify the cuts, write nothing
    python -m book_splitter chunk-glossary mybook          # -> 06-chunk-glossary/ (per chunk)
    python -m book_splitter chunk-glossary mybook --match-threshold 90
    python -m book_splitter chunk-glossary mybook --first-chapters 3 --resume
    python -m book_splitter matter mybook                  # -> 06b-matter/ + localized chunks
    python -m book_splitter matter mybook -t sr --limit 3  # first three matter chunks
    python -m book_splitter translate mybook              # -> 07-translations/ (body chunks)
    python -m book_splitter translate mybook -s sr -t en  # the other direction
    python -m book_splitter translate mybook --first-chapters 3  # first three chapters
    python -m book_splitter translate mybook --limit 3    # translate three chunks first
    python -m book_splitter translate mybook --resume     # continue where it stopped
    python -m book_splitter translate mybook --translator lmstudio  # local model, stage 7 only
    python -m book_splitter concatenate mybook            # -> 08-translated-chapters/
    python -m book_splitter concatenate mybook --first-chapters 3 --resume
    python -m book_splitter postprocess mybook             # -> 09-postprocessed-chapters/
    python -m book_splitter postprocess mybook --first-chapters 3 --resume
    python -m book_splitter openings mybook                # opening audit/repair (no LLM)
    python -m book_splitter openings mybook --dry-run
    python -m book_splitter bind mybook                  # -> 10-translated-book.md
    python -m book_splitter bind mybook --dry-run
    python -m book_splitter epub mybook                   # -> 11-translated-book.epub
    python -m book_splitter epub mybook --cover cover.jpg # with a cover picture
    python -m book_splitter epub mybook --title "Naslov" --author "Autor"
    python -m book_splitter tools --stage preprocess       # list the toolset per stage
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

from .analysis.candidates import extract_candidates
from .analysis.headers import assess_headers
from .analysis.toc import detect_toc_block
from .assembly import run_assembly
from .conversion import is_markdown_source, run_conversion
from .core.constants import (
    CHAPTERS_DIRNAME,
    CHUNK_GLOSSARY_SCORERS,
    DEFAULT_CHUNK_GLOSSARY_CANDIDATES,
    DEFAULT_CHUNK_GLOSSARY_MIN_TERM_CHARS,
    DEFAULT_CHUNK_GLOSSARY_SCORER,
    DEFAULT_CHUNK_GLOSSARY_THRESHOLD,
    DEFAULT_CHUNK_TOKENIZER,
    DEFAULT_CHUNK_TOKENS,
    DEFAULT_FRONT_MATTER_LINES,
    DEFAULT_GLOSSARY_ATTEMPTS,
    DEFAULT_GLOSSARY_CHUNK_CHARS,
    DEFAULT_GPU_INDEX,
    DEFAULT_KEEP_ENTITY_TYPES,
    DEFAULT_MATTER_ATTEMPTS,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MAX_HEADERS,
    DEFAULT_MIN_CHAPTERS,
    DEFAULT_MIN_SENTENCE_WORDS,
    DEFAULT_POSTPROCESS_ATTEMPTS,
    DEFAULT_RECURSION_LIMIT,
    DEFAULT_SOURCE_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
    DEFAULT_TRANSLATION_ATTEMPTS,
    DEFAULT_TRANSLATION_CONTEXT_CHARS,
    DEFAULT_TRANSLATION_MAX_TOKENS,
    MAX_FEEDBACK_LOOPS,
    POSTPROCESSED_CHAPTERS_DIRNAME,
    PREPROCESSED_DIRNAME,
    TOC_SCAN_LINES,
    TRANSLATED_CHAPTERS_DIRNAME,
    TRANSLATION_GLOSSARY_SCOPES,
    TRANSLATION_METHODS,
)
from .core.errors import ConfigError, MissingDependencyError, ToolError
from .core.settings import PROVIDERS, TRANSLATION_CHOICES
from .documents.chapters import (
    chapters_dir,
    chunk_glossary_dir,
    chunks_dir,
    discover_chapters,
    glossary_dir,
    has_chapter_files,
    has_chunk_files,
    has_postprocessed_chapter_files,
    has_translated_chapter_files,
    has_translation_files,
    load_split_manifest,
    matter_dir,
    matter_prompts_dir,
    postprocessed_chapters_dir,
    preprocessed_dir,
    reduced_dir,
    translated_book_path,
    translated_chapters_dir,
    translated_epub_path,
    translation_prompts_dir,
    translations_dir,
    unique_dir,
)
from .documents.io import load_lines, source_size_bytes
from .documents.window import (
    chapter_files_for_window,
    chapter_window_limit_for_path,
)
from .paths import default_input_dir, default_output_dir, resolve_book_path, resolve_output_dir


def _print_json(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _resolve_book(args: argparse.Namespace) -> Path:
    return resolve_book_path(getattr(args, "book", None), input_dir=args.in_dir)


def _resolve_chapters_dir(book: str | None, in_dir: str | None, out: str | None) -> Path:
    """Where stage 1's chapters are.

    Resolution order: an explicit ``--out``, a ``--book`` / positional argument
    that is already a folder, then the default output folder of the named book
    (or of the only book in ``data/input/``).
    """
    if out:
        return Path(out).expanduser()
    if book:
        candidate = Path(book).expanduser()
        if candidate.is_dir():
            return candidate
    source = resolve_book_path(book, input_dir=in_dir)
    return resolve_output_dir(source, None)


def _ensure_markdown_source(source: Path, out: str | Path | None = None) -> Path:
    """Run stage 0 when the source is not Markdown and return what stage 1 reads.

    A Markdown source is returned untouched: stage 0 only exists to make the
    other formats readable by the line-based stages.
    """
    if is_markdown_source(source):
        return source
    result = run_conversion(source, out)
    for line in result.log:
        print(f"  · {line}", file=sys.stderr)
    if result.markdown_path:
        print(f"converted: {result.source} -> {result.markdown_path}", file=sys.stderr)
    return Path(result.markdown_path)


def _model_settings(
    args: argparse.Namespace,
    stage: str,
    *,
    translation: bool = False,
):
    """Resolve one command's model, honouring ``--llm-config`` and ``--llm``.

    The settings module is imported at call time so the offline tests keep
    patching ``book_splitter.core.settings.resolve_settings`` (or
    ``resolve_translation_settings``) at the exact seam this command uses.
    """
    from .core.settings import (
        parse_llm_overrides,
        resolve_settings,
        resolve_translation_settings,
        stage_overrides,
    )

    overrides = stage_overrides(
        parse_llm_overrides(getattr(args, "llm", None), default_stage=stage),
        stage,
    )
    config_path = getattr(args, "llm_config", None)
    if translation:
        return resolve_translation_settings(
            provider=getattr(args, "translator", None),
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            temperature=args.temperature,
            config_path=config_path,
            overrides=overrides,
        )
    return resolve_settings(
        stage=stage,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        config_path=config_path,
        overrides=overrides,
    )


def cmd_convert(args: argparse.Namespace) -> int:
    """Stage 0: EPUB/PDF/TXT -> Markdown under ``00-markdown/``."""
    source = _resolve_book(args)
    result = run_conversion(
        source,
        args.out,
        dry_run=args.dry_run,
        force=args.force,
        images=getattr(args, "images", True),
    )

    if args.json:
        _print_json(result.to_dict())
        return 0

    if not args.quiet:
        for line in result.log:
            print(f"  · {line}")
        print()
        print(f"source          : {result.source}")
        print(f"kind            : {result.source_kind}")
        if result.pages is not None:
            print(
                f"pdf pages       : {result.pages} "
                f"({result.text_pages}/{result.sampled_pages} sampled page(s) with text)"
            )
        print(f"method          : {result.method}")
        print(f"markdown        : {result.markdown_path}")
        if result.images or result.image_refs:
            print(f"images          : {result.images} file(s), {result.image_refs} reference(s)")
            if result.images_dir:
                print(f"images folder   : {result.images_dir}")
        if result.reused:
            print("reused          : existing Markdown still matches the source hash")
        print(
            "written         : no (dry run — nothing written)"
            if args.dry_run
            else f"written         : {'yes' if result.written else 'no'}"
        )
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    from .agent.tools import grouped_tools

    wanted = getattr(args, "stage", None)
    printed = 0
    for stage, items in grouped_tools():
        if wanted and stage != wanted:
            continue
        print(f"{stage}:")
        if not items:
            print("    (none)")
        for name, summary in items:
            print(f"    {name}")
            print(f"        {summary}")
        printed += len(items)
        print()
    if printed == 0:
        print(f"no tools registered for stage {wanted!r}", file=sys.stderr)
        return 1
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Deterministic pre-flight: no LLM, no API key.

    Stage 0 runs first when the source is EPUB/PDF/TXT, so the report exists for
    every supported format; a Markdown source is read as-is.
    """
    from .agent.tools import scan_markdown_headers

    source = _ensure_markdown_source(_resolve_book(args))
    lines = load_lines(source)
    headers = scan_markdown_headers(str(source))["headers"]
    toc = detect_toc_block(lines, scan_lines=args.toc_scan_lines)

    report = {
        "source": str(source),
        "input_dir": str(default_input_dir()),
        "line_count": len(lines),
        "size_bytes": source_size_bytes(source),
        "header_count": len(headers),
        "header_assessment": assess_headers(headers, len(lines)),
        "headers": headers[: args.max_headers],
        "printed_toc_block": toc,
        "first_lines": [
            {"line_number": i + 1, "text": line.strip()[:120]}
            for i, line in enumerate(lines[: args.head_lines])
            if line.strip()
        ],
    }
    if args.candidates:
        report["heuristic_candidates"] = extract_candidates(
            lines, max_candidates=args.max_candidates
        )["candidates"]
    _print_json(report)
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    """Deterministic damage profile of a split book: no LLM, no writes."""
    from .preprocess.corpus import scan_corpus_damage
    from .preprocess.inspect import chapter_digest, inspect_chapter

    chapters_dir = _resolve_chapters_dir(args.book, args.in_dir, args.out)
    chapters = discover_chapters(chapters_dir)
    window = chapter_window_limit_for_path(
        chapters_dir,
        getattr(args, "first_chapters", None),
        getattr(args, "limit", None),
    )
    if window is not None:
        chapters = chapters[:window]
    if args.headers:
        for chapter in chapters:
            _print_json(inspect_chapter(chapter.path, include_all=args.all_headers))
        return 0
    if args.digest:
        for chapter in chapters:
            report = inspect_chapter(chapter.path, include_all=args.all_headers)
            print(chapter_digest(report))
            print()
    report = scan_corpus_damage(chapters, max_examples_per_pattern=args.examples)
    report["chapters_path"] = str(chapters_dir)
    report["manifest"] = {
        "strategy": load_split_manifest(chapters_dir).get("strategy"),
        "chapter_count": len(chapters),
    }
    _print_json(report)
    return 0


def cmd_preprocess(args: argparse.Namespace) -> int:
    from .preprocess_agent import run_preprocess

    chapters_dir = _resolve_chapters_dir(args.book, args.in_dir, args.out)
    out_dir = args.preprocessed_out or None

    try:
        settings = _model_settings(args, "preprocess")
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    if not args.quiet:
        print(f"provider={settings.provider} model={settings.model} base_url={settings.base_url}")

    if not args.quiet:
        print(f"chapters={chapters_dir.expanduser().resolve()}")
        print(f"output={out_dir or 'data/output/<book>/02-preprocessed/'}")
    window = chapter_window_limit_for_path(
        chapters_dir, getattr(args, "first_chapters", None), args.limit
    )
    if not args.quiet:
        label = _window_label(args, window)
        if label is not None:
            print(f"window      : {label}")
    if not args.quiet and args.skip_if_present:
        print("resume      : chapters already cleaned and verified will be reused")

    result = run_preprocess(
        str(chapters_dir),
        out_dir,
        settings=settings,
        max_attempts=args.max_attempts,
        dry_run=args.dry_run,
        recursion_limit=args.recursion_limit,
        workers=args.workers,
        limit=window,
        skip_if_present=args.skip_if_present,
        write_manifest=not args.no_manifest,
    )

    if not args.quiet:
        for chapter in result.chapters:
            for line in chapter.get("log", []):
                print(f"  · {chapter['filename']} {line}")

    if args.json:
        _print_json(result.to_dict())
        return 0 if result.ok else 1

    _print_preprocess_summary(result, dry_run=args.dry_run)
    return 0 if result.ok else 1


def _print_preprocess_summary(result, dry_run: bool = False) -> None:
    print()
    print(
        f"chapters        : {result.ok_count}/{result.chapter_count} clean"
        + (f"  reused: {result.skipped_count}" if result.skipped_count else "")
    )
    print(f"llm calls       : {sum(c.get('attempts', 0) for c in result.chapters)}"
          f"  tokens: {result.usage.get('total_tokens', 0)}")
    if result.corpus_scan:
        counts = result.corpus_scan.get("counts", {})
        print("damage found    : " + ", ".join(f"{k}={v}" for k, v in counts.items() if v))
    _print_openings_line(result.openings)
    print(f"output dir      : {result.output_dir}")
    if result.chapters:
        print("files:")
        for chapter in result.chapters:
            flag = "ok " if chapter["ok"] else "FAIL"
            print(
                f"  {flag} {chapter['filename']:<52} lines "
                f"{chapter['source_lines']:>5} -> {chapter['output_lines']:<5} "
                f"ops={len(chapter['operations'])} fixes={len(chapter['text_fixes'])}"
            )
    if result.manifest_path:
        print(f"manifest        : {result.manifest_path}")
    if result.trace_path:
        print(f"trace           : {result.trace_path}")
    if not result.ok:
        print("\nFAILED chapters:", file=sys.stderr)
        for error in result.errors:
            print(f"  - {error}", file=sys.stderr)
        return
    print(
        "\nDRY RUN — every plan validated, nothing written."
        if dry_run
        else "\nOK — every chapter cleaned and verified."
    )


def _window_label(args: argparse.Namespace, limit: int | None) -> str | None:
    """Human description of the chapter window, without exposing the file count."""
    first = getattr(args, "first_chapters", None)
    if first is not None:
        return f"first {first} body chapter(s)"
    if limit is not None:
        return f"first {limit} chapter(s)"
    return None


def _parse_entity_types(value: str | None) -> tuple[str, ...]:
    """``"PER, loc ,ORG"`` -> ``("PER", "LOC", "ORG")``."""
    if not value:
        return DEFAULT_KEEP_ENTITY_TYPES
    wanted = tuple(part.strip().upper() for part in str(value).split(",") if part.strip())
    return wanted or DEFAULT_KEEP_ENTITY_TYPES


def cmd_dictionary(args: argparse.Namespace) -> int:
    """Stage 3: reduce each chapter to its entity-bearing sentences, then to first mentions."""
    from .dictionary import run_dictionary

    # Accept either the chapters folder or the book folder that holds it, exactly
    # like `scan` and `preprocess` do: `--out data/output/<book>` is a natural
    # thing to type, and `discover_chapters` resolves the nested `01-chapters/`
    # anyway, so the pre-flight check must agree with it.
    chapters_path = chapters_dir(_resolve_chapters_dir(args.book, args.in_dir, args.out))
    if not has_chapter_files(chapters_path):
        print(
            f"no chapter files in {chapters_path.expanduser().resolve()} — run `split` first",
            file=sys.stderr,
        )
        return 2
    out_dir = (
        Path(args.reduced_out).expanduser() if args.reduced_out else reduced_dir(chapters_path)
    )
    # `--unique-only` is the shortcut for "I only want the small list"; otherwise
    # both folders are written and each can still be switched off on its own.
    save_reduced = not args.no_save_reduced and not args.unique_only
    save_unique = not args.no_save_unique
    unique_out_dir = (
        Path(args.unique_out).expanduser() if args.unique_out else unique_dir(out_dir)
    )
    keep_types = _parse_entity_types(args.keep_types)
    limit = chapter_window_limit_for_path(
        chapters_path, getattr(args, "first_chapters", None), args.limit
    )

    if not args.quiet:
        device = "cpu" if args.cpu else f"gpu {args.gpu if args.gpu is not None else DEFAULT_GPU_INDEX}"
        print(f"chapters={chapters_path.expanduser().resolve()}")
        if save_reduced:
            print(f"reduced={out_dir.expanduser().resolve()}")
        if save_unique:
            print(f"unique={unique_out_dir.expanduser().resolve()}")
        print(f"language={args.language} device={device} keep_types={','.join(keep_types)}")
        label = _window_label(args, limit)
        if label is not None:
            print(f"window       : {label}")
        if not save_reduced and not save_unique:
            print("save         : nothing is written except the manifest")
        elif not save_reduced:
            print("save         : first mentions only (--no-save-reduced)")
        elif not save_unique:
            print("save         : reduced chapters only (--no-save-unique)")

    try:
        result = run_dictionary(
            str(chapters_path),
            str(out_dir),
            unique_output_dir=str(unique_out_dir),
            language=args.language,
            gpu=args.gpu,
            use_gpu=not args.cpu,
            min_words=args.min_words,
            require_uppercase=not args.no_uppercase_filter,
            keep_types=keep_types,
            drop_temporal_only=args.drop_temporal_only,
            skip_if_present=args.skip_if_present,
            save_reduced=save_reduced,
            save_unique=save_unique,
            dry_run=args.dry_run,
            limit=limit,
            example_limit=args.examples,
            unique_example_limit=args.examples,
            progress=None if args.quiet else _dictionary_progress,
        )
    except MissingDependencyError as exc:
        print(f"missing dependency: {exc}", file=sys.stderr)
        return 2
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        _print_json(result.to_dict())
        return 0 if result.ok else 1
    _print_dictionary_summary(result, dry_run=args.dry_run)
    return 0 if result.ok else 1


def _dictionary_progress(chapter: dict, done: int, total: int) -> None:
    """One line per reduced chapter, printed as it finishes."""
    unique = chapter.get("unique")
    tail = f" -> {chapter.get('unique_kept', 0):<5} unique" if unique is not None else ""
    print(
        f"  [{done}/{total}] {chapter['filename']:<48} "
        f"{chapter['sentences']:>5} sentences -> {chapter['kept']:<5} kept{tail}"
        f"  ner {_human_seconds(float(chapter.get('ner_seconds') or 0.0)):>8}"
    )


def _human_seconds(value: float) -> str:
    """``0.0213`` -> ``21 ms``; ``389.4`` -> ``6m 29.4s``."""
    if value < 1:
        return f"{value * 1000:.0f} ms"
    if value < 60:
        return f"{value:.2f} s"
    minutes, seconds = divmod(value, 60)
    if minutes < 60:
        return f"{int(minutes)}m {seconds:04.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes:02d}m {seconds:04.1f}s"


def _print_dictionary_summary(result, dry_run: bool = False) -> None:
    language = result.language or {}
    print()
    print(
        f"language        : {language.get('language') or '—'} "
        f"({language.get('method')}, confidence {language.get('confidence', 0):.2f})"
        f"  backend: {result.backend or '—'}"
        + (f" on {result.device}" if result.device else "")
    )
    if language.get("note"):
        print(f"  note          : {language['note']}")
    totals = result.totals or {}
    unique_totals = result.unique_totals or {}
    timing = result.timing or {}
    save_unique = getattr(result, "save_unique", True)

    def _unique_line(label: str) -> str:
        dropped = unique_totals.get("dropped_duplicate_noun", 0)
        return (
            f"{label:<16}: {unique_totals.get('sentences', 0)} -> "
            f"{unique_totals.get('kept', 0)} kept "
            f"(duplicate-noun={dropped}, nouns={unique_totals.get('nouns', 0)})"
        )

    print(f"chapters        : {result.ok_count}/{result.chapter_count} ok"
          + (f"  reused: {result.skipped_count}" if result.skipped_count else ""))
    if result.skipped:
        print(
            "nothing to do   : every reduced chapter is already present"
            + (" (both folders)" if save_unique else "")
        )
    else:
        print(
            f"sentences       : {totals.get('sentences', 0)} -> {totals.get('kept', 0)} kept"
        )
        print(
            "dropped         : "
            f"short={totals.get('dropped_short', 0)} "
            f"no-uppercase={totals.get('dropped_no_uppercase', 0)} "
            f"no-entity={totals.get('dropped_no_entity', 0)}"
            + (
                f" temporal-only={totals.get('dropped_temporal', 0)}"
                if result.settings.get("drop_temporal_only")
                else ""
            )
        )
        if save_unique:
            print(_unique_line("first mentions"))
        entities = result.entities or {}
        if entities:
            print("entities        : " + ", ".join(f"{k}={v}" for k, v in entities.items()))
        if timing:
            tagged = int(totals.get("sentences_tagged", 0))
            print(
                f"ner time        : {_human_seconds(float(timing.get('ner_seconds', 0.0)))} "
                f"for {tagged:,} sentence(s)"
                + (
                    f"  ({float(timing.get('milliseconds_per_sentence', 0.0)):.1f} ms each, "
                    f"{float(timing.get('sentences_per_second', 0.0)):.0f}/s)"
                    if tagged
                    else ""
                )
            )
            if timing.get("backend_load_seconds"):
                where = result.device or ("cpu" if result.gpu is None else f"gpu {result.gpu}")
                print(
                    f"model load      : "
                    f"{_human_seconds(float(timing['backend_load_seconds']))} "
                    f"({result.backend or '—'}, {where})"
                )
            print(
                f"total time      : {_human_seconds(float(timing.get('run_seconds', 0.0)))}"
                f"  (chapters {_human_seconds(float(timing.get('chapter_seconds', 0.0)))})"
            )
    print(f"output dir      : {result.output_dir}")
    if save_unique and getattr(result, "unique_dir", ""):
        print(f"unique dir      : {result.unique_dir}")
    if result.manifest_path:
        print(f"manifest        : {result.manifest_path}")
    if result.trace_path:
        print(f"trace           : {result.trace_path}")
    if not result.ok:
        print("\nFAILED chapters:", file=sys.stderr)
        for error in result.errors:
            print(f"  - {error}", file=sys.stderr)
        return
    if result.skipped:
        print("\nOK — nothing was written: the existing folders were reused.")
        return
    if dry_run:
        print("\nDRY RUN — every chapter was reduced, nothing was written.")
        return
    lines = []
    if result.save_reduced:
        lines.append("OK — reduced chapters written.")
    if save_unique:
        lines.append("OK — first-mention chapters written.")
    if not lines:
        lines.append("OK — the manifest was written; no chapter files.")
    print("\n" + "\n".join(lines))


def cmd_chunk(args: argparse.Namespace) -> int:
    """Stage 5: cut each chapter into chunks that concatenate back exactly."""
    from .chunking import run_chunking

    chapters_path = chapters_dir(_resolve_chapters_dir(args.book, args.in_dir, args.out))
    if not has_chapter_files(chapters_path):
        print(
            f"no chapter files in {chapters_path.expanduser().resolve()} — run `split` first",
            file=sys.stderr,
        )
        return 2
    if int(args.chunk_size) <= 0:
        print("error: --chunk-size must be greater than 0 tokens", file=sys.stderr)
        return 2
    out_dir = (
        Path(args.chunks_out).expanduser() if args.chunks_out else chunks_dir(chapters_path)
    )
    limit = chapter_window_limit_for_path(
        chapters_path, getattr(args, "first_chapters", None), args.limit
    )

    if not args.quiet:
        print(f"chapters={chapters_path.expanduser().resolve()}")
        print(f"chunks={out_dir.expanduser().resolve()}")
        print(
            f"chunk size={args.chunk_size} tokens  "
            f"tokenizer={args.chunk_tokenizer}  overlap=0  method=recursive"
        )
        label = _window_label(args, limit)
        if label is not None:
            print(f"window      : {label}")

    try:
        result = run_chunking(
            str(chapters_path),
            str(out_dir),
            chunk_size=args.chunk_size,
            tokenizer=args.chunk_tokenizer,
            skip_if_present=args.skip_if_present,
            dry_run=args.dry_run,
            limit=limit,
            write_manifest=not args.no_manifest and not args.dry_run,
            progress=None if args.quiet else _chunk_progress,
        )
    except MissingDependencyError as exc:
        print(f"missing dependency: {exc}", file=sys.stderr)
        return 2
    except (ToolError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        _print_json(result.to_dict())
        return 0 if result.ok else 1
    _print_chunk_summary(result, dry_run=args.dry_run)
    return 0 if result.ok else 1


def _chunk_progress(chapter: dict, done: int, total: int) -> None:
    """One line per chunked (or reused) chapter, printed as it finishes."""
    flag = "reuse" if chapter.get("skipped") else "ok   "
    print(
        f"  [{done}/{total}] {flag} {chapter['filename']:<44} "
        f"{chapter.get('chunk_count', 0):>4} chunk(s)  "
        f"{int(chapter.get('token_count') or 0):>8,} tok"
    )


def _print_chunk_summary(result, dry_run: bool = False) -> None:
    settings = result.settings or {}
    tokenizer = settings.get("tokenizer") or {}
    verification = result.verification or {}
    print()
    print(
        f"chapters        : {result.ok_count}/{result.chapter_count} ok"
        + (f"  reused: {result.skipped_count}" if result.skipped_count else "")
    )
    print(
        f"chunks          : {result.chunk_count:,}  "
        f"(max {int(settings.get('chunk_size') or 0):,} tokens each, overlap 0)"
    )
    print(
        f"tokenizer       : {tokenizer.get('used') or tokenizer.get('requested') or '—'}"
        + (
            f"  (requested {tokenizer.get('requested')}; local approximation is in use)"
            if tokenizer.get("fallback")
            else ""
        )
    )
    if tokenizer.get("note"):
        print(f"  note          : {tokenizer['note']}")
    totals = result.totals or {}
    print(
        f"tokens          : {int(totals.get('chapter_tokens') or 0):,} in chapters "
        f"-> {int(totals.get('chunk_tokens') or 0):,} across chunks"
    )
    kinds = verification.get("boundary_counts") or {}
    shown = ", ".join(f"{kind}={count:,}" for kind, count in kinds.items() if count)
    if shown:
        print(f"boundaries      : {shown}")
    print(
        f"reconstruction  : chapters={verification.get('chapters_reconstruct_byte_for_byte')} "
        f"book={verification.get('book_reconstructs_byte_for_byte')} "
        f"word-breaks={verification.get('word_breaking_chunks', 0)}"
    )
    if verification.get("chunk_size_violations"):
        print(f"  warning       : {verification['chunk_size_violations']} chunk(s) exceed the limit")
    print(f"output dir      : {result.output_dir}")
    if result.manifest_path:
        print(f"manifest        : {result.manifest_path}")
    if result.trace_path:
        print(f"trace           : {result.trace_path}")
    if not result.ok:
        print("\nFAILED chapters:", file=sys.stderr)
        for error in result.errors:
            print(f"  - {error}", file=sys.stderr)
        return
    if dry_run:
        print("\nDRY RUN — every cut was verified, nothing was written.")
        return
    if result.skipped:
        print("\nOK — nothing was written: the existing chunks were reused.")
        return
    print(
        "\nOK — chunks written; concatenate them in filename order to rebuild the book, "
        "or per chapter via reconstruct_chapter()."
    )


def cmd_chunk_glossary(args: argparse.Namespace) -> int:
    """Stage 6: NER over every chunk, then fuzzy-match each term to the glossary."""
    from .chunk_glossary import run_chunk_glossary
    from .chunk_glossary.runner import resolve_chunks_dir

    chapters_path = chapters_dir(_resolve_chapters_dir(args.book, args.in_dir, args.out))
    chunks_path = (
        resolve_chunks_dir(Path(args.chunks_in).expanduser())
        if args.chunks_in
        else chunks_dir(chapters_path)
    )
    if not has_chunk_files(chunks_path):
        print(
            f"no chunk files in {chunks_path.expanduser().resolve()} — run `chunk` first",
            file=sys.stderr,
        )
        return 2
    glossary_path = (
        Path(args.glossary_in).expanduser()
        if args.glossary_in
        else glossary_dir(chapters_path)
    )
    out_dir = (
        Path(args.chunk_glossary_out).expanduser()
        if args.chunk_glossary_out
        else chunk_glossary_dir(chunks_path)
    )
    keep_types = _parse_entity_types(args.keep_types)

    if not 0.0 <= float(args.threshold) <= 100.0:
        print(
            "error: --match-threshold is a percentage between 0 and 100",
            file=sys.stderr,
        )
        return 2
    if int(args.candidates) <= 0:
        print("error: --match-candidates must be greater than 0", file=sys.stderr)
        return 2
    if int(args.min_term_chars) < 0:
        print("error: --min-term-chars cannot be negative", file=sys.stderr)
        return 2

    try:
        chapter_files = chapter_files_for_window(
            chunks_path, getattr(args, "first_chapters", None)
        )
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    limit = args.limit

    if not args.quiet:
        print(f"chapters={chapters_path.expanduser().resolve()}")
        print(f"chunks={chunks_path.expanduser().resolve()}")
        print(f"glossary={glossary_path.expanduser().resolve()}")
        print(f"chunk-glossary={out_dir.expanduser().resolve()}")
        print(
            f"language={args.language}  threshold={args.threshold}  "
            f"scorer={args.scorer}  candidates={args.candidates}  "
            f"keep_types={','.join(keep_types)}  min_term_chars={args.min_term_chars}"
        )
        print("phases      : ner over every chunk, then fuzzy retrieval for every term")
        if chapter_files is not None:
            print(
                f"window      : chunks of the first {args.first_chapters} body chapter(s)"
            )
        if limit is not None:
            print(f"limit       : first {limit} chunk(s) in the window")
        if not args.drop_temporal_only:
            print("              dates/times are kept; pass --drop-temporal-only to skip them")

    try:
        result = run_chunk_glossary(
            str(chunks_path),
            str(out_dir),
            glossary_path=str(glossary_path),
            language=args.language,
            gpu=args.gpu,
            use_gpu=not args.cpu,
            keep_types=keep_types,
            min_term_chars=args.min_term_chars,
            drop_temporal_only=args.drop_temporal_only,
            threshold=args.threshold,
            scorer=args.scorer,
            candidates=args.candidates,
            dry_run=args.dry_run,
            limit=limit,
            chapter_files=chapter_files,
            skip_if_present=args.skip_if_present,
            write_manifest=not args.no_manifest and not args.dry_run,
            progress=None if args.quiet else _chunk_glossary_progress,
        )
    except MissingDependencyError as exc:
        print(f"missing dependency: {exc}", file=sys.stderr)
        return 2
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        _print_json(result.to_dict())
        return 0 if result.ok else 1
    _print_chunk_glossary_summary(result, dry_run=args.dry_run)
    return 0 if result.ok else 1


def _chunk_glossary_progress(chunk: dict, done: int, total: int) -> None:
    """One line per chunk, printed as the retrieval phase finishes it."""
    flag = "ok   " if chunk.get("ok") else "FAIL "
    print(
        f"  [{done}/{total}] {flag} {chunk['filename']:<44} "
        f"entities={chunk.get('entity_count', 0):>3}  "
        f"terms={chunk.get('term_count', 0):>3}  "
        f"matched={chunk.get('matched_count', 0):>3}  "
        f"unmatched={chunk.get('unmatched_count', 0):>3}"
    )


def _print_chunk_glossary_summary(result, dry_run: bool = False) -> None:
    language = result.language or {}
    settings = result.settings or {}
    totals = result.totals or {}
    timing = result.timing or {}
    print()
    print(
        f"language        : {language.get('language') or '—'} "
        f"({language.get('method')}, confidence {language.get('confidence', 0):.2f})"
        f"  backend: {result.backend or '—'}"
        + (f" on {result.device}" if result.device else "")
    )
    if language.get("note"):
        print(f"  note          : {language['note']}")
    print(f"chunks          : {result.ok_count}/{result.chunk_count} ok")
    print(
        f"entities        : {int(totals.get('entity_mentions') or 0):,} mention(s) "
        f"-> {int(totals.get('terms') or 0):,} distinct term(s)"
    )
    print(f"glossary        : {result.entry_count:,} entr"
          f"{'y' if result.entry_count == 1 else 'ies'} loaded")
    print(
        f"matched         : {int(totals.get('matched_terms') or 0):,} term(s) in "
        f"{int(totals.get('matched_chunks') or 0):,} chunk(s)  "
        f"(threshold {settings.get('threshold')}%, scorer {settings.get('scorer')}, "
        f"unique dictionary entries {int(totals.get('unique_matched_glossary_terms') or 0):,})"
    )
    print(f"unmatched       : {int(totals.get('unmatched_terms') or 0):,} term(s)")
    phases = result.phases or []
    for phase in phases:
        print(
            f"{phase.get('name', 'phase'):<16}: "
            f"{_human_seconds(float(phase.get('seconds') or 0.0))} "
            f"for {int(phase.get('chunks') or 0):,} chunk(s)"
        )
    if timing:
        print(f"total time      : {_human_seconds(float(timing.get('run_seconds') or 0.0))}")
    print(f"output dir      : {result.output_dir}")
    if result.manifest_path:
        print(f"manifest        : {result.manifest_path}")
    if result.trace_path:
        print(f"trace           : {result.trace_path}")
    if not result.ok:
        print("\nFAILED chunks:", file=sys.stderr)
        for error in result.errors:
            print(f"  - {error}", file=sys.stderr)
        return
    if dry_run:
        print("\nDRY RUN — both phases ran, nothing was written.")
        return
    if result.matched_term_count == 0:
        print(
            "\nOK — no chunk term reached the glossary; every chunk file records the "
            "unmatched terms and their closest candidate."
        )
        return
    print(
        f"\nOK — {int(totals.get('matched_terms') or 0):,} relevant term(s) saved, "
        "one file per chunk under the chunk's own name."
    )


class _TranslationBar:
    """A one-line progress bar for the translate command, with live token totals.

    On a terminal it redraws in place about once a second while a chunk's model
    call is in flight, so a long book shows that something is happening rather
    than a silent cursor. When stdout is piped (or captured by a test) it prints
    one durable line per finished chunk instead: a ``\\r`` bar is noise in a log,
    and the durable line carries the same counts.

    The numbers are the ones that matter for a paid run: chunks done, elapsed
    time, an ETA from the completed calls' average, the tokens spent so far, and
    the chunk currently in flight.
    """

    WIDTH = 22
    HEARTBEAT_SECONDS = 10.0

    def __init__(self, total: int) -> None:
        self.total = max(0, int(total))
        self.tty = sys.stdout.isatty()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._width = 0
        self._started = time.monotonic()
        self._done = 0
        self._tokens = 0
        self._current = ""
        self._current_started = self._started
        self._durations: list[float] = []

    # -- the hooks the runner calls ---------------------------------------- #
    def start(self, info: dict) -> None:
        """A chunk's model call is about to start."""
        with self._lock:
            self._current = str(info.get("filename") or "")
            self._current_started = time.monotonic()
            self._done = int(info.get("done") or 0)
            self.total = int(info.get("total") or self.total)
            # The runner hands over the cumulative total, which includes a
            # resumed run's earlier spend; a finished chunk then adds its own.
            prior = int((info.get("tokens") or {}).get("total_tokens") or 0)
            self._tokens = max(self._tokens, prior)
        if self.tty and (self._thread is None or not self._thread.is_alive()):
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._loop, name="translate-bar", daemon=True
            )
            self._thread.start()

    def finish(self, record: dict, done: int, total: int) -> None:
        """One chunk is finished (translated, reused or failed)."""
        now = time.monotonic()
        with self._lock:
            if self._current and self._current_started:
                self._durations.append(max(0.0, now - self._current_started))
            self._done = int(done)
            self.total = int(total or self.total)
            tokens = record.get("tokens") or {}
            self._tokens += int(tokens.get("total_tokens") or 0)
            self._current = ""
            self._current_started = 0.0
        if self.tty:
            self._clear()
        print(self._durable_line(record, done, total), flush=True)
        if self.tty:
            self._draw()

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None
        if self.tty:
            self._clear()

    # -- rendering ---------------------------------------------------------- #
    def _durable_line(self, record: dict, done: int, total: int) -> str:
        if record.get("reused"):
            flag = "reuse"
        elif record.get("ok"):
            flag = "ok   "
        else:
            flag = "FAIL "
        usage = record.get("tokens") or {}
        elapsed = float(record.get("elapsed_seconds") or 0.0)
        return (
            f"  [{done:>3}/{total}] {flag} {str(record.get('filename') or '?'):<46} "
            f"{int(record.get('source_chars') or 0):>6,} -> "
            f"{int(record.get('translation_chars') or 0):>6,} chars · "
            f"{int(usage.get('total_tokens') or 0):>7,} tok · "
            f"{int(record.get('attempts') or 0)} attempt(s) · {elapsed:5.1f}s"
            + (
                f"\n           {record.get('error')}"
                if record.get("error") and not record.get("ok")
                else ""
            )
        )

    def _line(self) -> str:
        with self._lock:
            done, total = self._done, self.total
            tokens = self._tokens
            current = self._current
            current_elapsed = time.monotonic() - self._current_started
            durations = list(self._durations)
        fill = min(self.WIDTH, int(self.WIDTH * done / total)) if total else 0
        bar = "█" * fill + "░" * (self.WIDTH - fill)
        elapsed = time.monotonic() - self._started
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
            f"{tokens:,} tok",
        ]
        if current:
            parts.append(f"now {current} ({current_elapsed:.0f}s)")
        return " · ".join(parts)

    def _draw(self) -> None:
        line = self._line()
        with self._lock:
            pad = max(0, self._width - len(line))
            print("\r" + line + " " * pad, end="", file=sys.stdout, flush=True)
            self._width = max(self._width, len(line))

    def _clear(self) -> None:
        with self._lock:
            if self._width:
                print("\r" + " " * self._width + "\r", end="", file=sys.stdout, flush=True)
                self._width = 0

    def _loop(self) -> None:
        last_heartbeat = 0.0
        while not self._stop.wait(1.0):
            if self.tty:
                self._draw()
                continue
            now = time.monotonic()
            if now - last_heartbeat >= self.HEARTBEAT_SECONDS:
                last_heartbeat = now
                print(self._line(), flush=True)


def cmd_matter(args: argparse.Namespace) -> int:
    """Stage 6b: localize front/back matter with its own agent, before translation."""
    from .chunk_glossary.runner import resolve_chunks_dir
    from .matter_agent import run_matter

    if args.chapters_in:
        chapters_path = chapters_dir(Path(args.chapters_in).expanduser())
    else:
        chapters_path = chapters_dir(
            _resolve_chapters_dir(args.book, args.in_dir, args.out)
        )
    chunks_path = (
        resolve_chunks_dir(Path(args.chunks_in).expanduser())
        if args.chunks_in
        else chunks_dir(chapters_path)
    )
    if not has_chunk_files(chunks_path):
        print(
            f"no chunk files in {chunks_path.expanduser().resolve()} — run `chunk` first",
            file=sys.stderr,
        )
        return 2
    translations_out = (
        Path(args.translations_out).expanduser()
        if args.translations_out
        else translations_dir(chunks_path)
    )
    matter_out = (
        Path(args.matter_out).expanduser()
        if args.matter_out
        else matter_dir(chunks_path)
    )
    prompts_out = (
        Path(args.prompts_out).expanduser()
        if args.prompts_out
        else matter_prompts_dir(chunks_path)
    )

    if args.limit is not None and int(args.limit) < 0:
        print("error: --limit cannot be negative", file=sys.stderr)
        return 2
    if int(args.max_attempts) <= 0:
        print("error: --max-attempts must be greater than 0", file=sys.stderr)
        return 2

    def factory():
        return _model_settings(args, "matter")

    # A fully resumed run needs no key, so a missing configuration is carried
    # until a pending chunk actually needs the model.
    settings = None
    config_error: ConfigError | None = None
    try:
        settings = factory()
    except ConfigError as exc:
        config_error = exc

    if not args.quiet:
        where = (
            f"{settings.provider}/{settings.model}"
            if settings is not None
            else "no model resolved"
        )
        print(f"chapters={chapters_path.expanduser().resolve()}")
        print(f"chunks={chunks_path.expanduser().resolve()}")
        print(f"translations={translations_out.expanduser().resolve()}")
        print(f"matter={matter_out.expanduser().resolve()}")
        if not args.no_prompts:
            print(f"prompts={prompts_out.expanduser().resolve()}")
        print(
            f"language={args.source_language} -> {args.target_language}  "
            f"model={where}  attempts<={args.max_attempts}  no dictionary"
        )
        print(
            "policy      : web-search the published title, credit BleBel, omit the "
            "original edition's imprint material, format as published front/back matter"
        )
        if args.limit:
            print(f"limit       : first {args.limit} matter chunk(s)")
        if args.resume:
            print("resume      : matter chunks already localized and unchanged are reused")

    try:
        result = run_matter(
            str(chapters_path),
            str(chunks_path),
            str(translations_out),
            prompts_dir=None if args.no_prompts else str(prompts_out),
            matter_path=str(matter_out),
            source_language=args.source_language,
            target_language=args.target_language,
            book_title=args.book_title or "",
            settings=settings,
            settings_factory=None if settings is not None else factory,
            max_attempts=args.max_attempts,
            recursion_limit=args.recursion_limit,
            dry_run=args.dry_run,
            resume=args.resume,
            limit=args.limit,
            write_manifest=not args.no_manifest,
            write_prompts=not args.no_prompts,
            progress=None if args.quiet or args.json else _matter_progress,
        )
    except ConfigError as exc:
        print(f"error: {config_error or exc}", file=sys.stderr)
        return 2
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        _print_json(result.to_dict())
        return 0 if result.ok else 1
    _print_matter_summary(result, dry_run=args.dry_run)
    return 0 if result.ok else 1


def _matter_progress(chunk: dict, done: int, total: int) -> None:
    """One durable line per localized (or omitted) matter chunk."""
    if chunk.get("reused"):
        flag = "reuse"
    elif chunk.get("omitted"):
        flag = "omit "
    elif chunk.get("ok"):
        flag = "ok   "
    else:
        flag = "FAIL "
    usage = chunk.get("usage") or {}
    print(
        f"  [{done:>3}/{total}] {flag} {str(chunk.get('filename') or '?'):<46} "
        f"{str(chunk.get('section_kind') or '?'):<5} "
        f"{int(chunk.get('source_chars') or 0):>6,} -> "
        f"{int(chunk.get('translation_chars') or 0):>6,} chars · "
        f"{int(usage.get('total_tokens') or 0):>7,} tok · "
        f"{int(chunk.get('attempts') or 0)} attempt(s)"
        + (f" · {chunk.get('omit_reason')}" if chunk.get("omitted") else "")
    )


def _print_matter_summary(result, dry_run: bool = False) -> None:
    plan = result.book_plan or {}
    validation = result.validation or {}
    totals = result.totals or {}
    tokens = result.tokens or {}
    timing = result.timing or {}
    print()
    print(
        f"title           : {plan.get('translated_title') or '—'}"
        + (
            f"  (original: {plan.get('original_title')!r})"
            if plan.get("original_title")
            else ""
        )
    )
    print(
        f"credit          : {plan.get('translation_credit') or 'Translated by BleBel'}"
    )
    if plan.get("title_evidence"):
        print(f"title evidence  : {plan['title_evidence']}")
    print(
        f"sections        : {result.section_count} front/back matter file(s), "
        f"{int(totals.get('matter_chunks') or 0):,} chunk(s)"
    )
    print(
        f"chunks          : {result.ok_count}/{result.chunk_count} localized"
        + (f"  reused: {result.reused_count:,}" if result.reused_count else "")
        + (f"  omitted: {int(totals.get('omitted') or 0)}" if totals.get("omitted") else "")
    )
    omitted = plan.get("omit") or []
    if omitted:
        print(f"omitted source  : {len(omitted)} phrase(s); the first few:")
        for phrase in omitted[:5]:
            print(f"    - {phrase}")
        if len(omitted) > 5:
            print(f"    … and {len(omitted) - 5} more (see the plan)")
    if validation.get("warnings"):
        for warning in validation["warnings"][:5]:
            print(f"  warning       : {warning}")
    print(
        f"tokens used     : {int(tokens.get('total_tokens') or 0):,} total "
        f"({int(tokens.get('input_tokens') or 0):,} in, "
        f"{int(tokens.get('output_tokens') or 0):,} out) over "
        f"{int(tokens.get('llm_calls') or 0):,} call(s)"
    )
    if timing:
        print(
            f"time            : {_human_seconds(float(timing.get('run_seconds', 0.0)))}"
            f"  {float(timing.get('seconds_per_chunk', 0.0)):.1f} s/chunk"
        )
    print(f"output dir      : {result.output_dir}")
    if not dry_run and result.prompts_dir:
        print(f"prompts         : {result.prompts_dir}")
    print(f"plan dir        : {result.matter_path}")
    if result.manifest_path:
        print(f"manifest        : {result.manifest_path}")
    if result.trace_path:
        print(f"trace           : {result.trace_path}")
    if result.skipped:
        print("\nOK — no front/back matter chunks to localize.")
        return
    if not result.ok:
        print("\nFAILED matter chunks:", file=sys.stderr)
        for error in result.errors:
            print(f"  - {error}", file=sys.stderr)
        return
    if dry_run:
        print("\nDRY RUN — every matter chunk was localized, nothing was written.")
        return
    print(
        "\nOK — front/back matter localized without a dictionary; the translated chunks "
        "are in 07-translations/ and stage 8 will assemble them with the body."
    )


def cmd_translate(args: argparse.Namespace) -> int:
    """Stage 7: translate each chunk in reading order, with its glossary rows."""
    from .chunk_glossary.runner import resolve_chunks_dir
    from .translation import run_translation
    from .translation.runner import resolve_chunk_glossary_dir

    chapters_path = chapters_dir(_resolve_chapters_dir(args.book, args.in_dir, args.out))
    chunks_path = (
        resolve_chunks_dir(Path(args.chunks_in).expanduser())
        if args.chunks_in
        else chunks_dir(chapters_path)
    )
    if not has_chunk_files(chunks_path):
        print(
            f"no chunk files in {chunks_path.expanduser().resolve()} — run `chunk` first",
            file=sys.stderr,
        )
        return 2
    chunk_glossary_path = resolve_chunk_glossary_dir(
        args.chunk_glossary_in or "", chunks_path
    )
    glossary_path = (
        Path(args.glossary_in).expanduser()
        if args.glossary_in
        else glossary_dir(chapters_path)
    )
    translations_out = (
        Path(args.translations_out).expanduser()
        if args.translations_out
        else translations_dir(chunks_path)
    )
    prompts_out = (
        Path(args.prompts_out).expanduser()
        if args.prompts_out
        else translation_prompts_dir(chunks_path)
    )

    if args.limit is not None and int(args.limit) < 0:
        print("error: --limit cannot be negative", file=sys.stderr)
        return 2
    if int(args.max_attempts) <= 0:
        print("error: --max-attempts must be greater than 0", file=sys.stderr)
        return 2
    if int(args.context_chars) < 0:
        print("error: --context-chars cannot be negative", file=sys.stderr)
        return 2

    try:
        chapter_files = chapter_files_for_window(
            chunks_path, getattr(args, "first_chapters", None)
        )
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    limit = args.limit

    # The key is resolved before the run so a setup problem is one line, not a
    # traceback from inside a provider client. A fully resumed run needs no key;
    # the runner only builds a translator when a chunk is actually pending.
    settings = None
    config_error: ConfigError | None = None
    try:
        settings = _model_settings(args, "translate", translation=True)
    except ConfigError as exc:
        config_error = exc

    if not args.quiet:
        where = (
            f"{settings.provider}/{settings.model}"
            if settings is not None
            else "no model resolved"
        )
        print(f"chapters={chapters_path.expanduser().resolve()}")
        print(f"chunks={chunks_path.expanduser().resolve()}")
        print(f"chunk-glossary={chunk_glossary_path.expanduser().resolve()}")
        print(f"translations={translations_out.expanduser().resolve()}")
        if not args.no_prompts:
            print(f"prompts={prompts_out.expanduser().resolve()}")
        print(
            f"language={args.source_language} -> {args.target_language}  "
            f"model={where}  method={args.method}  attempts<={args.max_attempts}  "
            f"scope={args.glossary_scope}"
        )
        if chapter_files is not None:
            print(
                f"window      : chunks of the first {args.first_chapters} body chapter(s)"
            )
        if limit:
            print(f"limit       : first {limit} chunk(s) in the window")
        if args.resume:
            print("resume      : chunks already translated and unchanged will be reused")

    bar = None if args.quiet or args.json else _TranslationBar(total=0)
    try:
        result = run_translation(
            str(chunks_path),
            str(translations_out),
            prompts_dir=None if args.no_prompts else str(prompts_out),
            chunk_glossary_path=str(chunk_glossary_path),
            glossary_path=str(glossary_path),
            glossary_scope=args.glossary_scope,
            source_language=args.source_language,
            target_language=args.target_language,
            book_title=args.book_title or "",
            settings=settings,
            max_attempts=args.max_attempts,
            method=args.method,
            max_tokens=args.max_tokens or None,
            context_chars=args.context_chars,
            dry_run=args.dry_run,
            limit=limit,
            chapter_files=chapter_files,
            include_matter=args.include_matter,
            resume=args.resume,
            write_manifest=not args.no_manifest,
            write_prompts=not args.no_prompts,
            progress=bar.finish if bar is not None else None,
            chunk_start=bar.start if bar is not None else None,
        )
    except ConfigError as exc:
        print(f"error: {config_error or exc}", file=sys.stderr)
        return 2
    except (MissingDependencyError, ToolError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        if bar is not None:
            bar.close()

    if args.json:
        _print_json(result.to_dict())
        return 0 if result.ok else 1
    _print_translation_summary(result, dry_run=args.dry_run)
    return 0 if result.ok else 1


def _print_translation_summary(result, dry_run: bool = False) -> None:
    settings = result.settings or {}
    totals = result.totals or {}
    tokens = result.tokens or {}
    timing = result.timing or {}
    print()
    print(
        f"language        : {result.source_language} -> {result.target_language}"
        f"  {result.provider or '—'}/{result.model or '—'}"
    )
    print(
        f"chunks          : {result.ok_count}/{result.chunk_count} ok"
        + (f"  reused: {result.reused_count:,}" if result.reused_count else "")
        + (
            f"  failed: {int(totals.get('failed_chunks') or 0):,}"
            if int(totals.get("failed_chunks") or 0)
            else ""
        )
    )
    if result.limited:
        print(f"  limit         : first {result.chunk_count:,} chunk(s) only")
    if result.glossary_scope == "book":
        print(
            f"glossary        : whole book glossary · "
            f"{result.glossary_entries:,} entr{'y' if result.glossary_entries == 1 else 'ies'} "
            "per chunk"
        )
    else:
        print(
            f"glossary        : per-chunk matched terms · "
            f"{int(totals.get('glossary_rows') or 0):,} row(s) sent"
            + (
                f" · {int(totals.get('missing_glossary_chunks') or 0):,} chunk(s) had none"
                if int(totals.get("missing_glossary_chunks") or 0)
                else ""
            )
        )
    print(
        f"text            : {int(totals.get('source_chars') or 0):,} -> "
        f"{int(totals.get('translation_chars') or 0):,} characters"
    )
    print(
        f"tokens used     : {int(tokens.get('total_tokens') or 0):,} total "
        f"({int(tokens.get('input_tokens') or 0):,} in, "
        f"{int(tokens.get('output_tokens') or 0):,} out) over "
        f"{int(tokens.get('calls') or 0):,} call(s), "
        f"{int(tokens.get('chunks') or 0):,} chunk(s) asked"
    )
    if timing:
        print(
            f"time            : {_human_seconds(float(timing.get('run_seconds', 0.0)))}"
            f"  (model {_human_seconds(float(timing.get('model_seconds', 0.0)))})"
            f"  {float(timing.get('seconds_per_chunk', 0.0)):.1f} s/chunk"
        )
    print(f"output dir      : {result.output_dir}")
    if not dry_run and result.prompts_dir:
        print(f"prompts         : {result.prompts_dir}")
    if result.manifest_path:
        print(f"manifest        : {result.manifest_path}")
    if result.trace_path:
        print(f"trace           : {result.trace_path}")
    if settings.get("json_only"):
        print(
            "  note          : the provider rejected native structured output; "
            "the run used JSON mode"
        )
    if not result.ok:
        print("\nFAILED chunks:", file=sys.stderr)
        for error in result.errors:
            print(f"  - {error}", file=sys.stderr)
        return
    if dry_run:
        print("\nDRY RUN — every chunk was translated, nothing was written.")
        return
    if result.reused_count == result.chunk_count and result.processed_count == 0:
        print("\nOK — nothing to do: every chunk is already translated.")
        return
    print(
        f"\nOK — {int(totals.get('translated_chunks') or 0):,} chunk(s) translated; "
        "concatenate 07-translations/ in filename order to rebuild the translated book."
    )


def cmd_concatenate(args: argparse.Namespace) -> int:
    """Stage 8: concatenate the translated chunks into one file per chapter."""
    from .assembly.runner import resolve_chunks_dir

    if args.chunks_in:
        chunks_path = resolve_chunks_dir(Path(args.chunks_in).expanduser())
        chapters_path = chunks_path.parent / CHAPTERS_DIRNAME
    else:
        chapters_path = chapters_dir(_resolve_chapters_dir(args.book, args.in_dir, args.out))
        chunks_path = chunks_dir(chapters_path)
    if not has_chunk_files(chunks_path):
        print(
            f"no chunk files in {chunks_path.expanduser().resolve()} — run `chunk` first",
            file=sys.stderr,
        )
        return 2
    translations_path = (
        Path(args.translations_in).expanduser()
        if args.translations_in
        else translations_dir(chunks_path)
    )
    if not has_translation_files(translations_path):
        print(
            f"no translated chunks in {translations_path.expanduser().resolve()} — "
            "run `translate` first",
            file=sys.stderr,
        )
        return 2
    out_dir = (
        Path(args.chapters_out).expanduser()
        if args.chapters_out
        else translated_chapters_dir(chunks_path)
    )

    if not args.quiet:
        print(f"chapters={chapters_path.expanduser().resolve()}")
        print(f"chunks={chunks_path.expanduser().resolve()}")
        print(f"translations={translations_path.expanduser().resolve()}")
        print(f"translated-chapters={out_dir.expanduser().resolve()}")
        if args.first_chapters:
            print(f"window      : first {args.first_chapters} chapter(s)")
        if args.skip_if_present:
            print("resume      : chapter files that already match will be reused")

    try:
        result = run_assembly(
            str(chunks_path),
            str(translations_path),
            str(out_dir),
            first_chapters=args.first_chapters,
            skip_if_present=args.skip_if_present,
            dry_run=args.dry_run,
            write_manifest=not args.no_manifest,
            progress=None if args.quiet else _assembly_progress,
        )
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        _print_json(result.to_dict())
        return 0 if result.ok else 1
    _print_assembly_summary(result, dry_run=args.dry_run)
    return 0 if result.ok else 1


def _assembly_progress(chapter: dict, done: int, total: int) -> None:
    """One line per assembled (or skipped) chapter."""
    if chapter.get("skipped"):
        flag = "skip"
        tail = f"  {chapter.get('note') or 'not translated'}"
    elif chapter.get("reused"):
        flag = "reuse"
        tail = ""
    else:
        flag = "ok   "
        tail = ""
    print(
        f"  [{done}/{total}] {flag} {chapter['filename']:<44} "
        f"{int(chapter.get('chunk_count') or 0):>3} chunk(s)  "
        f"{int(chapter.get('translation_chars') or 0):>7,} chars{tail}"
    )


def _print_assembly_summary(result, dry_run: bool = False) -> None:
    print()
    print(
        f"chapters        : {result.ok_count}/{result.chapter_count} written"
        + (f"  reused: {result.reused_count}" if result.reused_count else "")
        + (f"  incomplete: {result.incomplete_count}" if result.incomplete_count else "")
    )
    if result.limited:
        print(f"window          : first {result.first_chapters} chapter(s)")
    print(
        f"chunks          : {int(result.totals.get('translated_chunks') or 0):,} translated "
        f"of {int(result.totals.get('chunks') or 0):,} in the window"
    )
    print(
        f"output          : {result.output_dir}"
        + ("  (dry run — nothing written)" if dry_run else "")
    )
    if result.manifest_path:
        print(f"manifest        : {result.manifest_path}")
    if result.trace_path:
        print(f"trace           : {result.trace_path}")
    incomplete = [chapter for chapter in result.chapters if chapter.get("skipped")]
    for chapter in incomplete:
        print(
            f"  incomplete    : {chapter['filename']} — "
            f"{chapter.get('note') or 'not translated'}"
        )
    if not result.ok:
        print("\nFAILED chapters:", file=sys.stderr)
        for error in result.errors:
            print(f"  - {error}", file=sys.stderr)
        return
    if dry_run:
        print("\nDRY RUN — every chapter was assembled in memory, nothing was written.")
        return
    if result.chapter_count == 0:
        print("\nOK — no chapters in the requested window.")
        return
    print(
        "\nOK — translated chapters written to a separate folder; "
        "the source chapters were not touched."
    )


def cmd_postprocess(args: argparse.Namespace) -> int:
    """Stage 9: postprocess the translated chapters' headings and table of contents."""
    from .postprocess_agent import run_postprocess
    from .translation.runner import resolve_chunks_dir as _resolve_chunks

    if args.translated_chapters_in:
        source_path = Path(args.translated_chapters_in).expanduser()
        if source_path.is_file():
            source_path = source_path.parent
        if (source_path / TRANSLATED_CHAPTERS_DIRNAME).is_dir():
            source_path = source_path / TRANSLATED_CHAPTERS_DIRNAME
    elif args.chunks_in:
        source_path = translated_chapters_dir(
            _resolve_chunks(Path(args.chunks_in).expanduser())
        )
    else:
        chapters_path = chapters_dir(_resolve_chapters_dir(args.book, args.in_dir, args.out))
        source_path = translated_chapters_dir(chapters_path)
    if not has_translated_chapter_files(source_path):
        print(
            f"no translated chapter files in {source_path.expanduser().resolve()} — "
            "run `concatenate` first",
            file=sys.stderr,
        )
        return 2
    out_dir = (
        Path(args.postprocessed_out).expanduser()
        if args.postprocessed_out
        else postprocessed_chapters_dir(source_path)
    )
    if int(args.max_attempts) <= 0:
        print("error: --max-attempts must be greater than 0", file=sys.stderr)
        return 2
    # Stage 9 counts body chapters directly (front matter is always included and
    # does not consume one of the N), so its limit is the user's N, not a file count.
    limit = args.first_chapters if args.first_chapters is not None else args.limit
    if limit is not None:
        limit = int(limit)

    try:
        settings = _model_settings(args, "postprocess")
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    if not args.quiet:
        print(f"provider={settings.provider} model={settings.model} base_url={settings.base_url}")
        print(f"translated-chapters={source_path.expanduser().resolve()}")
        print(f"postprocessed-chapters={out_dir.expanduser().resolve()}")
        print(
            f"budget      : max {args.max_attempts} model turn(s) per book plan and per chapter; "
            "headings/opening only, no chapter prose"
        )
        label = _window_label(args, limit)
        if label is not None:
            print(f"window      : {label}")
        if args.skip_if_present:
            print("resume      : chapters whose source hash and title still match are reused")

    try:
        result = run_postprocess(
            str(source_path),
            str(out_dir),
            settings=settings,
            max_attempts=args.max_attempts,
            dry_run=args.dry_run,
            recursion_limit=args.recursion_limit,
            limit=limit,
            skip_if_present=args.skip_if_present,
            write_manifest=not args.no_manifest,
        )
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not args.quiet:
        for chapter in result.chapters:
            for line in chapter.get("log") or []:
                print(f"  · {chapter['filename']} {line}")

    if args.json:
        _print_json(result.to_dict())
        return 0 if result.ok else 1
    _print_postprocess_summary(result, dry_run=args.dry_run)
    return 0 if result.ok else 1


def _print_postprocess_summary(result, dry_run: bool = False) -> None:
    validation = result.validation or {}
    totals = result.totals or {}
    toc = result.toc or {}
    print()
    print(
        f"book plan       : "
        f"{(result.book_plan.get('naming') or {}).get('mode', '—')} "
        f"label={(result.book_plan.get('naming') or {}).get('label', '—')!r} "
        f"· toc {(result.book_plan.get('toc') or {}).get('action', '—')}"
    )
    print(
        f"chapters        : {result.ok_count}/{result.chapter_count} postprocessed"
        + (f"  reused: {result.reused_count}" if result.reused_count else "")
    )
    print(
        f"headings        : {int(totals.get('demoted_headings') or 0)} demoted · "
        f"{int(totals.get('promoted_headings') or 0)} promoted · "
        f"{int(totals.get('merged_subtitles') or 0)} subtitle(s) folded into a title"
    )
    print(
        f"titles          : {int(totals.get('title_set') or 0)} set · "
        f"{int(totals.get('title_inserted') or 0)} inserted"
    )
    web = result.web or {}
    print(
        f"web search      : {len(web.get('searches') or [])} lookup(s) · "
        + ("attempted" if web.get("attempted") else "NOT attempted")
        + f" · source {(result.book_plan.get('title_source') or '—')}"
    )
    evidence = str(result.book_plan.get("title_evidence") or "").strip()
    if evidence:
        print(f"title evidence  : {evidence[:200]}")
    if toc:
        print(
            f"table of contents: {toc.get('action') or '—'} "
            f"({int(totals.get('toc_entries') or 0)} entries)"
            + (f" · {toc.get('filename')}" if toc.get("filename") else "")
            + (" · dry run" if dry_run else "")
        )
    if validation.get("warnings"):
        for warning in validation["warnings"][:5]:
            print(f"  warning       : {warning}")
    print(f"output dir      : {result.output_dir}")
    _print_openings_line(result.openings)
    if result.manifest_path:
        print(f"manifest        : {result.manifest_path}")
    if result.trace_path:
        print(f"trace           : {result.trace_path}")
    if not result.ok:
        print("\nFAILED chapters:", file=sys.stderr)
        for error in result.errors:
            print(f"  - {error}", file=sys.stderr)
        return
    if dry_run:
        print("\nDRY RUN — every plan was validated, nothing was written.")
        return
    print(
        "\nOK — postprocessed chapters written to a separate folder; the translated "
        "chapters were not touched."
    )


def _print_openings_line(openings: dict | None, *, detail: bool = True) -> None:
    """One line (and the chapters that still need a look) for the opening audit."""
    data = openings or {}
    if not data or data.get("skipped"):
        return
    dry = bool(data.get("dry_run"))
    removed = int(data.get("fixed_count") or 0)
    planned = sum(len(chapter.get("fixes") or []) for chapter in data.get("chapters") or [])
    print(
        f"openings        : {int(data.get('inspected_count') or 0)} checked · "
        f"{int(data.get('clean_count') or 0)} clean · "
        + (
            f"{planned} line(s) would be removed · "
            if dry
            else f"{removed} line(s) removed · "
        )
        + f"{int(data.get('remaining_count') or 0)} need attention"
    )
    if not detail:
        return
    verb = "would remove" if dry else "removed"
    for chapter in data.get("chapters") or []:
        fixes = chapter.get("fixes") or []
        if fixes:
            text = ", ".join(
                f"line {fix.get('line_number')} {str(fix.get('text') or '')!r}"
                for fix in fixes
            )
            print(f"  · {verb} {chapter.get('filename')}: {text}")
    for chapter in data.get("chapters") or []:
        remaining = chapter.get("remaining") or []
        for issue in remaining[:4]:
            print(
                f"  · {chapter.get('filename')} line {issue.get('line_number')}: "
                f"{issue.get('kind')} — {issue.get('detail')}"
            )


def cmd_openings(args: argparse.Namespace) -> int:
    """Final phase of stages 2 and 9: audit every chapter opening (no LLM)."""
    from .openings import audit_openings

    folder = _resolve_openings_dir(args)
    if not folder.is_dir():
        print(f"error: chapter folder not found: {folder}", file=sys.stderr)
        return 2
    result = audit_openings(
        folder,
        dry_run=bool(args.dry_run),
        keep_blank_slot=bool(args.keep_blank_slot),
    )
    if args.json:
        _print_json(result.to_dict())
        return 0 if result.ok else 1
    if not args.quiet:
        print(f"chapters        : {folder.expanduser().resolve()}")
        print(
            "policy          : "
            + (
                "keep the line count (stage-2 blank slot)"
                if args.keep_blank_slot
                else "remove the line and collapse the blank (stage-9 output)"
            )
        )
    print()
    _print_openings_line(result.summary(), detail=not args.quiet)
    if not args.quiet:
        print(f"output dir      : {result.folder}")
    if not result.ok:
        print("\nFAILED openings:", file=sys.stderr)
        for error in result.errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    if result.errors:
        for error in result.errors:
            print(f"  warning         : {error}", file=sys.stderr)
    print(
        "\nDRY RUN — every opening audited, nothing was written."
        if args.dry_run
        else "\nOK — every chapter opening audited; only provable leftovers were removed."
    )
    return 0


#: Chapter folders the ``openings`` command understands, most advanced first.
_CHAPTER_STAGE_FOLDERS = (
    POSTPROCESSED_CHAPTERS_DIRNAME,
    TRANSLATED_CHAPTERS_DIRNAME,
    PREPROCESSED_DIRNAME,
    CHAPTERS_DIRNAME,
)


def _resolve_openings_dir(args: argparse.Namespace) -> Path:
    """The folder to audit: an explicit one, or the book's most advanced stage."""
    if getattr(args, "chapters_in", None):
        path = Path(args.chapters_in).expanduser()
        if path.is_file():
            path = path.parent
        # A stage folder is taken as given (the caller chose that stage); a book
        # folder is descended into, preferring the most advanced stage present.
        if path.name not in _CHAPTER_STAGE_FOLDERS:
            for name in _CHAPTER_STAGE_FOLDERS:
                candidate = path / name
                if has_chapter_files(candidate):
                    return candidate
        return path
    chapters = _resolve_chapters_dir(args.book, args.in_dir, None)
    for candidate in (
        postprocessed_chapters_dir(chapters),
        translated_chapters_dir(chapters),
        preprocessed_dir(chapters),
        chapters,
    ):
        if has_chapter_files(candidate):
            return candidate
    return chapters


def cmd_bind(args: argparse.Namespace) -> int:
    """Stage 10: concatenate the postprocessed chapters into one Markdown book."""
    from .binding import run_bind

    if args.postprocessed_in:
        source_path = Path(args.postprocessed_in).expanduser()
        if source_path.is_file():
            source_path = source_path.parent
        if (source_path / POSTPROCESSED_CHAPTERS_DIRNAME).is_dir():
            source_path = source_path / POSTPROCESSED_CHAPTERS_DIRNAME
    else:
        chapters_path = chapters_dir(_resolve_chapters_dir(args.book, args.in_dir, args.out))
        source_path = postprocessed_chapters_dir(chapters_path)
    if not has_postprocessed_chapter_files(source_path):
        print(
            f"no postprocessed chapter files in {source_path.expanduser().resolve()} — "
            "run `postprocess` first",
            file=sys.stderr,
        )
        return 2
    output_path = (
        Path(args.book_out).expanduser()
        if args.book_out
        else translated_book_path(source_path)
    )
    if not args.quiet:
        print(f"postprocessed-chapters={source_path.expanduser().resolve()}")
        print(f"book={output_path.expanduser().resolve()}")
    try:
        result = run_bind(
            str(source_path),
            str(output_path),
            dry_run=args.dry_run,
            write_manifest=not args.no_manifest,
        )
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        _print_json(result.to_dict())
        return 0 if result.ok else 1
    _print_bind_summary(result, dry_run=args.dry_run)
    return 0 if result.ok else 1


def _print_bind_summary(result, dry_run: bool = False) -> None:
    print()
    print(f"files           : {result.chapter_count}")
    print(f"table of contents: {'included' if result.toc_included else 'not found'}")
    print(f"size            : {result.chars:,} characters · {result.lines:,} lines")
    print(f"output          : {result.output_path}")
    if result.manifest_path:
        print(f"manifest        : {result.manifest_path}")
    if result.trace_path:
        print(f"trace           : {result.trace_path}")
    if not result.ok:
        print("\nFAILED:", file=sys.stderr)
        for error in result.errors:
            print(f"  - {error}", file=sys.stderr)
        return
    if dry_run:
        print("\nDRY RUN — the book was assembled and verified, nothing was written.")
        return
    print("\nOK — the postprocessed chapters were concatenated into one Markdown book.")


def cmd_epub(args: argparse.Namespace) -> int:
    """Stage 11: package the postprocessed chapters as an EPUB 3 book."""
    from .epub import run_epub

    if args.postprocessed_in:
        source_path = Path(args.postprocessed_in).expanduser()
        if source_path.is_file():
            source_path = source_path.parent
        if (source_path / POSTPROCESSED_CHAPTERS_DIRNAME).is_dir():
            source_path = source_path / POSTPROCESSED_CHAPTERS_DIRNAME
    else:
        chapters_path = chapters_dir(_resolve_chapters_dir(args.book, args.in_dir, args.out))
        source_path = postprocessed_chapters_dir(chapters_path)
    if not has_postprocessed_chapter_files(source_path):
        print(
            f"no postprocessed chapter files in {source_path.expanduser().resolve()} — "
            "run `postprocess` first",
            file=sys.stderr,
        )
        return 2
    output_path = (
        Path(args.epub_out).expanduser()
        if args.epub_out
        else translated_epub_path(source_path)
    )
    if not args.quiet:
        print(f"postprocessed-chapters={source_path.expanduser().resolve()}")
        print(f"epub={output_path.expanduser().resolve()}")
    try:
        result = run_epub(
            str(source_path),
            str(output_path),
            title=args.title,
            author=args.author,
            language=args.language,
            identifier=args.identifier,
            description=args.description,
            publisher=args.publisher,
            cover=args.cover,
            dry_run=args.dry_run,
            write_manifest=not args.no_manifest,
        )
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        _print_json(result.to_dict())
        return 0 if result.ok else 1
    _print_epub_summary(result, dry_run=args.dry_run)
    return 0 if result.ok else 1


def _print_epub_summary(result, dry_run: bool = False) -> None:
    print()
    print(f"title           : {result.title}")
    print(f"author          : {result.author or '(none recorded)'}")
    print(f"language        : {result.language}")
    print(f"identifier      : {result.identifier}")
    print(f"documents       : {result.document_count}")
    print(f"pictures        : {result.image_count}")
    if result.cover:
        print(f"cover           : {result.cover}")
    print(
        f"size            : {result.bytes:,} bytes · "
        f"{result.chars:,} characters of Markdown"
    )
    verification = result.verification or {}
    print(
        "verification    : "
        f"mimetype {'ok' if verification.get('mimetype_first') else 'FAILED'} · "
        f"container {'ok' if verification.get('container_ok') else 'FAILED'} · "
        f"package {'ok' if verification.get('opf_ok') else 'FAILED'} · "
        f"navigation {'ok' if verification.get('nav_ok') else 'FAILED'} · "
        f"{int(verification.get('xhtml_valid') or 0)}/"
        f"{int(verification.get('documents') or 0)} XHTML"
    )
    if result.toc_dropped:
        print("table of contents: dropped (the reader's navigation is the table of contents)")
    print(f"output          : {result.output_path}")
    if result.manifest_path:
        print(f"manifest        : {result.manifest_path}")
    if result.trace_path:
        print(f"trace           : {result.trace_path}")
    for warning in result.warnings:
        print(f"warning         : {warning}", file=sys.stderr)
    if not result.ok:
        print("\nFAILED:", file=sys.stderr)
        for error in result.errors:
            print(f"  - {error}", file=sys.stderr)
        return
    if dry_run:
        print("\nDRY RUN — the package was built and verified, nothing was written.")
        return
    print("\nOK — the postprocessed chapters were packaged as an EPUB 3 book.")


def cmd_glossary(args: argparse.Namespace) -> int:
    """Stage 4: build the book-wide glossary, one chapter at a time, first to last."""
    from .glossary import run_glossary
    from .glossary.runner import resolve_sentences_dir

    chapters_path = chapters_dir(_resolve_chapters_dir(args.book, args.in_dir, args.out))
    if not has_chapter_files(chapters_path):
        print(
            f"no chapter files in {chapters_path.expanduser().resolve()} — run `split` first",
            file=sys.stderr,
        )
        return 2
    try:
        sentences_path = resolve_sentences_dir(
            chapters_path, sentences_dir=args.sentences_in, sentences=args.sentences
        )
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out_dir = (
        Path(args.glossary_out).expanduser() if args.glossary_out else glossary_dir(chapters_path)
    )
    limit = chapter_window_limit_for_path(
        chapters_path, getattr(args, "first_chapters", None), args.limit
    )

    # The key is resolved before the run so a setup problem is one line, not a
    # traceback from inside a provider client. A fully resumed run needs no key —
    # it never calls the model — so a missing credential is carried, not fatal.
    settings = None
    config_error: ConfigError | None = None
    try:
        settings = _model_settings(args, "dictionary")
    except ConfigError as exc:
        config_error = exc

    if not args.quiet:
        where = (
            f"{settings.provider}/{settings.model}"
            if settings is not None
            else "no model resolved"
        )
        print(f"chapters={chapters_path.expanduser().resolve()}")
        print(f"sentences={sentences_path.expanduser().resolve()} ({args.sentences})")
        print(f"glossary={out_dir.expanduser().resolve()}")
        print(
            f"language={args.source_language} -> {args.target_language}  "
            f"model={where}  method={args.extraction_method}  attempts<={args.max_attempts}"
        )
        skipped = []
        if not args.include_front_matter:
            skipped.append("front matter")
        if not args.include_back_matter:
            skipped.append("back matter")
        if skipped:
            print(f"skip        : {', '.join(skipped)}")
        label = _window_label(args, limit)
        if label is not None:
            print(f"window      : {label}")

    try:
        result = run_glossary(
            str(chapters_path),
            str(out_dir),
            sentences_dir=str(sentences_path),
            sentences=args.sentences,
            source_language=args.source_language,
            target_language=args.target_language,
            book_title=args.book_title or "",
            settings=settings,
            max_attempts=args.max_attempts,
            method=args.extraction_method,
            max_tokens=args.max_tokens or None,
            chunk_chars=args.max_chapter_chars,
            max_context_entries=args.max_context_entries,
            include_front_matter=args.include_front_matter,
            include_back_matter=args.include_back_matter,
            skip_if_present=args.resume,
            dry_run=args.dry_run,
            limit=limit,
            progress=None if args.quiet else _glossary_progress,
        )
    except ConfigError as exc:
        print(f"error: {config_error or exc}", file=sys.stderr)
        return 2
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        _print_json(result.to_dict())
        return 0 if result.ok else 1
    _print_glossary_summary(result, dry_run=args.dry_run)
    return 0 if result.ok else 1


def _glossary_progress(chapter: dict, done: int, total: int) -> None:
    """One line per chapter, printed as its answer is folded into the dictionary."""
    tokens = chapter.get("tokens") or {}
    print(
        f"  [{done}/{total}] {chapter['filename']:<44} "
        f"{chapter['added']:>4} new  {chapter['repeats']:>3} locked  "
        f"{chapter['conflicts']:>2} conflict  "
        f"{int(tokens.get('total_tokens') or 0):>8,} tok  "
        f"{chapter['attempts']} attempt(s)"
    )


def _print_glossary_summary(result, dry_run: bool = False) -> None:
    print()
    print(
        f"glossary        : {result.source_language} -> {result.target_language}"
        f"  {result.provider or '—'}/{result.model or '—'}"
    )
    if (getattr(result, "settings", None) or {}).get("json_only"):
        print(
            "  note          : the provider rejected native structured output; "
            "the run used JSON mode"
        )
    print(
        f"chapters        : {result.ok_count}/{result.body_chapter_count} body chapter(s) "
        "processed"
        + (f"  reused: {result.reused_count}" if result.reused_count else "")
        + (f"  skipped: {result.skipped_count}" if result.skipped_count else "")
    )
    categories = result.categories or {}
    print(
        f"entries         : {result.entry_count}"
        + (
            "  (" + ", ".join(f"{name}={count}" for name, count in categories.items()) + ")"
            if categories
            else ""
        )
    )
    if result.conflicts:
        print(f"conflicts       : {len(result.conflicts)} (the first rendering stands)")
        for conflict in result.conflicts[:3]:
            print(
                f"    - {conflict.get('term')}: locked {conflict.get('locked')!r}, "
                f"{conflict.get('chapter_filename')} proposed {conflict.get('proposed')!r}"
            )
        if len(result.conflicts) > 3:
            print(f"    … and {len(result.conflicts) - 3} more (see the manifest)")
    tokens = result.tokens or {}
    chapters_with_calls = int(tokens.get("chapters") or 0)
    print(
        f"tokens used     : {int(tokens.get('total_tokens') or 0):,} total "
        f"({int(tokens.get('input_tokens') or 0):,} in, "
        f"{int(tokens.get('output_tokens') or 0):,} out) "
        f"over {chapters_with_calls} chapter(s), {int(tokens.get('calls') or 0)} call(s)"
    )
    timing = result.timing or {}
    if timing:
        print(
            f"time            : {_human_seconds(float(timing.get('run_seconds', 0.0)))}"
            f"  (model {_human_seconds(float(timing.get('model_seconds', 0.0)))})"
            f"  {float(timing.get('seconds_per_chapter', 0.0)):.1f} s/chapter"
        )
    if result.resumed:
        print(
            f"resumed         : {result.reused_count} chapter(s) reused from the manifest"
        )
    print(f"output dir      : {result.output_dir}")
    if result.glossary_path:
        print(f"glossary        : {result.glossary_path}")
    if result.markdown_path:
        print(f"markdown        : {result.markdown_path}")
    if result.manifest_path:
        print(f"manifest        : {result.manifest_path}")
    if result.trace_path:
        print(f"trace           : {result.trace_path}")
    if not result.ok:
        print("\nFAILED chapters:", file=sys.stderr)
        for error in result.errors:
            print(f"  - {error}", file=sys.stderr)
        return
    if dry_run:
        print("\nDRY RUN — every chapter was asked, nothing was written.")
        return
    if result.entry_count == 0:
        print("\nOK — no entity was found; the glossary is empty.")
        return
    print(f"\nOK — {result.entry_count} entr{'y' if result.entry_count == 1 else 'ies'} locked in.")


def cmd_split(args: argparse.Namespace) -> int:
    from .agent import run_split, run_split_from_map
    from .core.errors import ToolError
    from .slicing.mapfile import load_split_map

    source = _resolve_book(args)
    out_dir = resolve_output_dir(source, args.out)

    if getattr(args, "skip_if_present", False) and not args.dry_run and has_chapter_files(out_dir):
        # Checked here, before any provider settings are resolved: a run that is
        # going to be skipped must not need an API key either.
        from .agent import result_from_existing_chapters

        if not args.quiet:
            print(f"source={source}")
            print(f"output={out_dir.expanduser().resolve()}")
        return _report_split(result_from_existing_chapters(out_dir), args)

    # Stage 0: the splitter only ever reads Markdown. A TXT is copied and an
    # EPUB/text-based PDF is converted before anything tries to count lines.
    source = _ensure_markdown_source(source)

    if getattr(args, "map", None):
        # A supplied map needs no provider, no key and no network.
        try:
            mapping, notes = load_split_map(args.map)
        except ToolError as exc:
            print(f"map error: {exc}", file=sys.stderr)
            return 2
        if not args.quiet:
            print(f"map={Path(args.map).expanduser()}")
            print(f"source={source}")
            print(f"output={out_dir.expanduser().resolve()}")
            for note in notes:
                print(f"  · {note}")
        result = run_split_from_map(
            str(source),
            str(out_dir),
            mapping,
            min_chapters=args.min_chapters,
            include_front_matter=not args.no_front_matter,
            dry_run=args.dry_run,
        )
        return _report_split(result, args)

    try:
        settings = _model_settings(args, "split")
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    if not args.quiet:
        print(f"provider={settings.provider} model={settings.model} base_url={settings.base_url}")
        print(f"source={source}")
        print(f"output={out_dir.expanduser().resolve()}")

    result = run_split(
        str(source),
        str(out_dir),
        settings=settings,
        front_matter_lines=args.front_matter_lines,
        max_attempts=args.max_attempts,
        min_chapters=args.min_chapters,
        include_front_matter=not args.no_front_matter,
        dry_run=args.dry_run,
        recursion_limit=args.recursion_limit,
    )
    return _report_split(result, args)


def _report_split(result, args: argparse.Namespace) -> int:
    """Print one split run (agent or supplied map) and return the exit code."""
    if not args.quiet:
        for line in result.log:
            print(f"  · {line}")

    if args.json:
        _print_json(result.to_dict())
        return 0 if result.ok else 1

    _print_summary(result, dry_run=args.dry_run)
    return 0 if result.ok else 1


def _print_summary(result, dry_run: bool = False) -> None:
    print()
    print(f"strategy        : {result.strategy}")
    print(f"llm calls       : {result.attempts}  tokens: {result.usage.get('total_tokens', 0)}")
    print(f"chapters        : {result.chapter_count}")
    print(f"output dir      : {result.output_dir}")
    if result.verification:
        if "reassembles_byte_for_byte" in result.verification:
            print(f"byte-for-byte   : {result.verification.get('reassembles_byte_for_byte')}")
            print(
                f"lines           : {result.verification.get('output_lines')}"
                f"/{result.verification.get('source_lines')}"
            )
        elif result.verification.get("reused_existing_chapters"):
            print("reused          : chapter files were already there; nothing was re-sliced")
    if result.files:
        print("files:")
        for f in result.files:
            start, end = f.get("start_line"), f.get("end_line")
            span = f"{start:>5}-{end:<5}" if start is not None and end is not None else " " * 11
            print(
                f"  {f.get('filename', '?'):<52} lines {span} "
                f"{int(f.get('bytes') or 0):>8} B"
            )
    if result.trace_path:
        print(f"trace           : {result.trace_path}")
    if not result.ok:
        print(f"\nFAILED: {result.error}", file=sys.stderr)
        for issue in (result.validation or {}).get("issues", []):
            print(f"  - {issue}", file=sys.stderr)
        return
    print(
        "\nDRY RUN — plan validated, nothing written."
        if dry_run
        else "\nOK — every source line is accounted for."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="book_splitter",
        description="Stage 0 `convert` turns an EPUB, a text-based PDF or a TXT into "
        "Markdown (docling, no OCR for image-based PDFs), then four LangGraph agents, a "
        "dictionary funnel, a glossary pass, a reversible chunker, a per-chunk glossary "
        "matcher, a front/back matter localizer, a chunk translator, a heading "
        "postprocessor and a binder turn that Markdown book into a consistent translation: "
        "`split` cuts it into chapters "
        "without reading the body, "
        "`preprocess` repairs each chapter's headers and conversion damage one chapter "
        "at a time, `dictionary` reduces each chapter to the sentences that carry a "
        "named entity, `glossary` locks one rendering per term, `chunk` cuts each "
        "chapter into sentence-safe chunks that concatenate back exactly, "
        "`chunk-glossary` tags every chunk and attaches the glossary terms it mentions, "
        "`matter` localizes the front/back matter before translation (no dictionary, "
        "web-searched title, BleBel credit, original-edition material omitted), "
        "`translate` translates the body chunks in reading order with those terms, "
        "keeping each call's whole prompt for debugging, `concatenate` glues the "
        "translated chunks back into one file per chapter, `postprocess` makes the "
        "translated chapter titles, other headings and table of contents consistent "
        "without reading the prose, `openings` audits (and repairs) every chapter's "
        "first lines with no model at all, and `bind` concatenates the postprocessed "
        "chapters into one Markdown book. Sources come from data/input/, output goes to "
        "data/output/<book>/.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_book(parser_: argparse.ArgumentParser, required: bool) -> None:
        parser_.add_argument(
            "book",
            nargs="?" if not required else None,
            help="source document — a bare filename or a path "
            "(default: the only book found in data/input/)",
        )
        parser_.add_argument(
            "--in-dir",
            default=None,
            help=f"folder holding the source documents (default: {default_input_dir()})",
        )

    def add_llm_flags(parser_: argparse.ArgumentParser) -> None:
        parser_.add_argument(
            "--llm-config",
            default=None,
            metavar="FILE",
            help="per-stage model config (default: ./book_splitter.toml, or "
            "$BOOK_SPLITTER_CONFIG); one [llm] table plus an optional [llm.<stage>] "
            "table per stage",
        )
        parser_.add_argument(
            "--llm",
            action="append",
            default=[],
            metavar="[STAGE.]FIELD=VALUE",
            help="override one model field for this stage, repeatable; fields: "
            "provider, model, base_url, temperature, timeout, max_retries, "
            "api_key_env (e.g. --llm model=deepseek-flash or "
            "--llm translate.provider=lmstudio)",
        )

    def add_model_flags(parser_: argparse.ArgumentParser) -> None:
        parser_.add_argument("--provider", choices=sorted(PROVIDERS), default=None)
        parser_.add_argument("--model", default=None)
        parser_.add_argument("--base-url", default=None)
        parser_.add_argument("--api-key", default=None)
        parser_.add_argument(
            "--temperature",
            type=float,
            default=None,
            help="sampling temperature for this stage (default: the config file, else 0.0)",
        )
        add_llm_flags(parser_)
        parser_.add_argument("--recursion-limit", type=int, default=DEFAULT_RECURSION_LIMIT)
        parser_.add_argument("--json", action="store_true", help="machine-readable result")
        parser_.add_argument("--quiet", "-q", action="store_true")

    p_convert = sub.add_parser(
        "convert",
        help="stage 0: convert an EPUB, a text-based PDF or a TXT to Markdown "
        "(docling; image-based PDFs are rejected instead of OCR'd)",
    )
    add_book(p_convert, required=False)
    p_convert.add_argument(
        "--out",
        "-o",
        default=None,
        help="folder for the converted Markdown (default: data/output/<book>/00-markdown)",
    )
    p_convert.add_argument(
        "--force",
        action="store_true",
        help="convert again even when the existing Markdown still matches the source hash",
    )
    p_convert.add_argument(
        "--no-images",
        dest="images",
        action="store_false",
        help="do not extract pictures from the EPUB/PDF; the Markdown stays text-only "
        "(images are extracted by default)",
    )
    p_convert.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the source (including a PDF's text layer) and report what would "
        "happen; run no docling conversion and write nothing",
    )
    p_convert.add_argument("--json", action="store_true", help="machine-readable result")
    p_convert.add_argument("--quiet", "-q", action="store_true")
    p_convert.set_defaults(func=cmd_convert, images=True)

    p_tools = sub.add_parser("tools", help="list the tools, grouped by pipeline stage")
    p_tools.add_argument(
        "--stage",
        choices=["shared", "split", "preprocess", "matter", "postprocess"],
        default=None,
        help="show only one stage group (shared lists the generic tools)",
    )
    p_tools.set_defaults(func=cmd_tools)

    p_inspect = sub.add_parser(
        "inspect",
        help="deterministic structural report (no LLM; runs stage 0 for a "
        "non-Markdown source)",
    )
    add_book(p_inspect, required=False)
    p_inspect.add_argument("--max-headers", type=int, default=DEFAULT_MAX_HEADERS)
    p_inspect.add_argument("--head-lines", type=int, default=40)
    p_inspect.add_argument("--toc-scan-lines", type=int, default=TOC_SCAN_LINES)
    p_inspect.add_argument(
        "--candidates", action="store_true", help="also list heuristic candidate lines"
    )
    p_inspect.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    p_inspect.set_defaults(func=cmd_inspect)

    p_split = sub.add_parser("split", help="stage 1: run the agent and write chapter files")
    add_book(p_split, required=True)
    p_split.add_argument(
        "--out",
        "-o",
        default=None,
        help="output folder for chapter files "
        f"(default: {default_output_dir()}/<book>/{CHAPTERS_DIRNAME})",
    )
    p_split.add_argument(
        "--map",
        default=None,
        metavar="FILE",
        help="replay a chapter map from a JSON file instead of running the agent "
        "(no API key, no network); see book_splitter/slicing/mapfile.py for the format",
    )
    p_split.add_argument("--provider", choices=sorted(PROVIDERS), default=None)
    p_split.add_argument("--model", default=None)
    p_split.add_argument("--base-url", default=None)
    p_split.add_argument("--api-key", default=None)
    p_split.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="sampling temperature for stage 1 (default: the config file, else 0.0)",
    )
    add_llm_flags(p_split)
    p_split.add_argument(
        "--front-matter-lines",
        type=int,
        default=DEFAULT_FRONT_MATTER_LINES,
        help="max lines the agent may read from the front of the book",
    )
    p_split.add_argument(
        "--max-attempts",
        type=int,
        default=MAX_FEEDBACK_LOOPS,
        help="max LLM turns before giving up",
    )
    p_split.add_argument(
        "--min-chapters",
        type=int,
        default=DEFAULT_MIN_CHAPTERS,
        help="reject plans with fewer split points than this",
    )
    p_split.add_argument("--recursion-limit", type=int, default=DEFAULT_RECURSION_LIMIT)
    p_split.add_argument(
        "--no-front-matter",
        action="store_true",
        help="do not write 00-front-matter.md (content before chapter 1)",
    )
    p_split.add_argument(
        "--skip-if-present",
        "--resume",
        dest="skip_if_present",
        action="store_true",
        help="reuse the chapter files already in the output folder instead of "
        "splitting again (no model, no key, and the existing trace is left alone)",
    )
    p_split.add_argument(
        "--dry-run", action="store_true", help="validate the plan but write no chapter files"
    )
    p_split.add_argument("--json", action="store_true", help="machine-readable result")
    p_split.add_argument("--quiet", "-q", action="store_true")
    p_split.set_defaults(func=cmd_split)

    p_scan = sub.add_parser(
        "scan",
        help="stage 2 pre-flight: header evidence and damage profile (no LLM, no writes)",
    )
    add_book(p_scan, required=False)
    p_scan.add_argument(
        "--out",
        "-o",
        default=None,
        help="the chapters folder from stage 1 (default: data/output/<book>/01-chapters)",
    )
    p_scan.add_argument(
        "--headers", action="store_true", help="print the full header record per chapter"
    )
    p_scan.add_argument(
        "--all-headers", action="store_true", help="include headers that look fine"
    )
    p_scan.add_argument("--digest", action="store_true", help="print the compact text digest")
    p_scan.add_argument("--examples", type=int, default=3)
    p_scan.add_argument(
        "--first-chapters",
        "--chapters",
        dest="first_chapters",
        type=int,
        default=None,
        help="scan only the first N body chapters (the whole book when omitted)",
    )
    p_scan.set_defaults(func=cmd_scan)

    p_pre = sub.add_parser(
        "preprocess",
        help="stage 2: clean each chapter, one chapter at a time, and write the result",
    )
    add_book(p_pre, required=False)
    p_pre.add_argument(
        "--out",
        "-o",
        default=None,
        help="the chapters folder from stage 1 (default: data/output/<book>/01-chapters)",
    )
    p_pre.add_argument(
        "--preprocessed-out",
        default=None,
        help="where to write cleaned chapters (default: data/output/<book>/02-preprocessed)",
    )
    add_model_flags(p_pre)
    p_pre.add_argument(
        "--max-attempts",
        type=int,
        default=MAX_FEEDBACK_LOOPS,
        help="max LLM turns per chapter before giving up on it",
    )
    p_pre.add_argument(
        "--workers",
        type=int,
        default=1,
        help="chapters processed at once (default 1: strictly one chapter at a time)",
    )
    p_pre.add_argument(
        "--no-manifest",
        action="store_true",
        help="do not write _preprocess_manifest.json",
    )
    p_pre.add_argument(
        "--first-chapters",
        "--chapters",
        dest="first_chapters",
        type=int,
        default=None,
        help="process only the first N body chapters (the whole book when omitted)",
    )
    p_pre.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process only the first N body chapters (kept as an alias of --first-chapters)",
    )
    p_pre.add_argument(
        "--skip-if-present",
        "--resume",
        dest="skip_if_present",
        action="store_true",
        help="reuse a chapter whose cleaned file a previous run verified, and start "
        "with the first chapter that is missing",
    )
    p_pre.add_argument(
        "--dry-run",
        action="store_true",
        help="validate every chapter's plan but write no files",
    )
    p_pre.set_defaults(func=cmd_preprocess)

    p_dict = sub.add_parser(
        "dictionary",
        help="stage 3: reduce each chapter to its entity-bearing sentences "
        "(sentence split -> uppercase filter -> 4-word floor -> NER gate), then to "
        "the sentences that introduce a noun (first mentions only)",
    )
    add_book(p_dict, required=False)
    p_dict.add_argument(
        "--out",
        "-o",
        default=None,
        help="the chapters folder from stage 1 (default: data/output/<book>/01-chapters); "
        "point it at data/output/<book>/02-preprocessed to reduce the cleaned text instead",
    )
    p_dict.add_argument(
        "--reduced-out",
        default=None,
        help="where to write the reduced chapters (default: data/output/<book>/03-reduced)",
    )
    p_dict.add_argument(
        "--unique-out",
        default=None,
        help="where to write the first-mention chapters "
        "(default: a sibling of the reduced folder, data/output/<book>/03-reduced-unique)",
    )
    p_dict.add_argument(
        "--language",
        default="auto",
        help="'auto' samples a few chapters and detects the language; otherwise an "
        "ISO code: sr, hr, sl, bg (CLASSLA) or en (stanza)",
    )
    p_dict.add_argument(
        "--gpu",
        type=int,
        default=None,
        help=f"physical GPU to pin the tagger to (default: GPU {DEFAULT_GPU_INDEX}, or "
        "$CUDA_VISIBLE_DEVICES when that is already set)",
    )
    p_dict.add_argument("--cpu", action="store_true", help="run the tagger on the CPU")
    p_dict.add_argument(
        "--min-words",
        type=int,
        default=DEFAULT_MIN_SENTENCE_WORDS,
        help=f"drop sentences with fewer words than this (default: {DEFAULT_MIN_SENTENCE_WORDS})",
    )
    p_dict.add_argument(
        "--no-uppercase-filter",
        action="store_true",
        help="do not drop sentences whose only capital letter is the first word",
    )
    p_dict.add_argument(
        "--keep-types",
        default=",".join(DEFAULT_KEEP_ENTITY_TYPES),
        help="entity types that make a sentence survive "
        f"(default: {','.join(DEFAULT_KEEP_ENTITY_TYPES)})",
    )
    p_dict.add_argument(
        "--drop-temporal-only",
        action="store_true",
        help="also drop a sentence when every entity in it is a date or a time",
    )
    p_dict.add_argument(
        "--skip-if-present",
        "--resume",
        dest="skip_if_present",
        action="store_true",
        help="reuse a chapter whose reduced (and first-mention) file already exists; a "
        "fully reduced book returns without loading a tagger",
    )
    p_dict.add_argument(
        "--no-save-reduced",
        action="store_true",
        help="write no reduced chapters; the manifest and the first-mention chapters "
        "still land",
    )
    p_dict.add_argument(
        "--no-save-unique",
        action="store_true",
        help="write no first-mention chapters, only the reduced ones",
    )
    p_dict.add_argument(
        "--unique-only",
        action="store_true",
        help="write only the first-mention chapters (implies --no-save-reduced)",
    )
    p_dict.add_argument(
        "--first-chapters",
        "--chapters",
        dest="first_chapters",
        type=int,
        default=None,
        help="reduce only the first N body chapters of the book (the whole book when omitted)",
    )
    p_dict.add_argument(
        "--limit",
        type=int,
        default=None,
        help="reduce only the first N body chapters (kept as a unit-limit alias)",
    )
    p_dict.add_argument(
        "--examples",
        type=int,
        default=5,
        help="dropped-sentence examples kept per reason in the trace (default: 5)",
    )
    p_dict.add_argument(
        "--dry-run",
        action="store_true",
        help="reduce every chapter but write no reduced file, manifest or trace",
    )
    p_dict.add_argument("--json", action="store_true", help="machine-readable result")
    p_dict.add_argument("--quiet", "-q", action="store_true")
    p_dict.set_defaults(func=cmd_dictionary)

    p_chunk = sub.add_parser(
        "chunk",
        help="stage 5: cut each chapter into translation-sized chunks without "
        "breaking paragraphs or sentences; every chunk is an exact source slice "
        "and the manifest records its chapter and global order",
    )
    add_book(p_chunk, required=False)
    p_chunk.add_argument(
        "--out",
        "-o",
        default=None,
        help="the chapters folder from stage 1 (default: data/output/<book>/01-chapters); "
        "point it at data/output/<book>/02-preprocessed to chunk the cleaned text instead",
    )
    p_chunk.add_argument(
        "--chunks-out",
        default=None,
        help="where to write the chunk files "
        "(default: data/output/<book>/05-chunks, a sibling of the input folder)",
    )
    p_chunk.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_TOKENS,
        help=f"maximum chunk length in tokens (default: {DEFAULT_CHUNK_TOKENS})",
    )
    p_chunk.add_argument(
        "--chunk-tokenizer",
        default=DEFAULT_CHUNK_TOKENIZER,
        help="tokenizer used to measure a chunk: a tiktoken encoding "
        f"(default: {DEFAULT_CHUNK_TOKENIZER}), 'word', 'character', 'byte', or a "
        "Hugging Face tokenizer name; falls back to a local approximation offline",
    )
    p_chunk.add_argument(
        "--skip-if-present",
        "--resume",
        dest="skip_if_present",
        action="store_true",
        help="reuse a chapter whose recorded chunk files still concatenate to its "
        "current text; a fully chunked book needs no tokenizer at all",
    )
    p_chunk.add_argument(
        "--first-chapters",
        "--chapters",
        dest="first_chapters",
        type=int,
        default=None,
        help="chunk only the first N body chapters (the whole book when omitted)",
    )
    p_chunk.add_argument(
        "--limit",
        type=int,
        default=None,
        help="chunk only the first N body chapters (kept as a unit-limit alias)",
    )
    p_chunk.add_argument(
        "--no-manifest",
        action="store_true",
        help="do not write _chunk_manifest.json / _chunk_trace.json",
    )
    p_chunk.add_argument(
        "--dry-run",
        action="store_true",
        help="compute and verify every cut but write no chunk file, manifest or trace",
    )
    p_chunk.add_argument("--json", action="store_true", help="machine-readable result")
    p_chunk.add_argument("--quiet", "-q", action="store_true")
    p_chunk.set_defaults(func=cmd_chunk)

    p_cg = sub.add_parser(
        "chunk-glossary",
        help="stage 6: run NER over every chunk first, then fuzzy-match each "
        "extracted term against glossary.json (rapidfuzz) and write the relevant "
        "terms per chunk, under the chunk's own name",
    )
    add_book(p_cg, required=False)
    p_cg.add_argument(
        "--out",
        "-o",
        default=None,
        help="the chapters folder from stage 1 (default: data/output/<book>/01-chapters); "
        "the chunks and glossary folders are derived from the book folder",
    )
    p_cg.add_argument(
        "--chunks-in",
        default=None,
        help="the chunks folder from stage 5 (default: data/output/<book>/05-chunks); "
        "a book folder or the _chunk_manifest.json is also accepted",
    )
    p_cg.add_argument(
        "--glossary-in",
        default=None,
        help="the glossary folder or glossary.json from stage 4 "
        "(default: data/output/<book>/04-glossary/glossary.json)",
    )
    p_cg.add_argument(
        "--chunk-glossary-out",
        default=None,
        help="where the per-chunk JSON files land "
        "(default: data/output/<book>/06-chunk-glossary)",
    )
    p_cg.add_argument(
        "--language",
        default="auto",
        help="'auto' samples a few chunks and detects the language; otherwise an "
        "ISO code: sr, hr, sl, bg (CLASSLA) or en (stanza)",
    )
    p_cg.add_argument(
        "--gpu",
        type=int,
        default=None,
        help=f"physical GPU to pin the tagger to (default: GPU {DEFAULT_GPU_INDEX}, or "
        "$CUDA_VISIBLE_DEVICES when that is already set)",
    )
    p_cg.add_argument("--cpu", action="store_true", help="run the tagger on the CPU")
    p_cg.add_argument(
        "--keep-types",
        default=",".join(DEFAULT_KEEP_ENTITY_TYPES),
        help="entity types that count as a term "
        f"(default: {','.join(DEFAULT_KEEP_ENTITY_TYPES)})",
    )
    p_cg.add_argument(
        "--min-term-chars",
        type=int,
        default=DEFAULT_CHUNK_GLOSSARY_MIN_TERM_CHARS,
        help="drop extracted terms shorter than this "
        f"(default: {DEFAULT_CHUNK_GLOSSARY_MIN_TERM_CHARS})",
    )
    p_cg.add_argument(
        "--drop-temporal-only",
        action="store_true",
        help="do not extract dates and times (they cannot be in the glossary)",
    )
    p_cg.add_argument(
        "--threshold",
        "--match-threshold",
        dest="threshold",
        type=float,
        default=DEFAULT_CHUNK_GLOSSARY_THRESHOLD,
        help="minimum rapidfuzz score, as a percentage "
        f"(default: {DEFAULT_CHUNK_GLOSSARY_THRESHOLD}); lower is more permissive",
    )
    p_cg.add_argument(
        "--scorer",
        choices=CHUNK_GLOSSARY_SCORERS,
        default=DEFAULT_CHUNK_GLOSSARY_SCORER,
        help="rapidfuzz scorer: ratio is strict; token_sort_ratio ignores word "
        "order; wratio tolerates partial names "
        f"(default: {DEFAULT_CHUNK_GLOSSARY_SCORER})",
    )
    p_cg.add_argument(
        "--candidates",
        type=int,
        default=DEFAULT_CHUNK_GLOSSARY_CANDIDATES,
        help="glossary candidates retrieved per term; the best is the match and the "
        f"rest are recorded as alternatives (default: {DEFAULT_CHUNK_GLOSSARY_CANDIDATES})",
    )
    p_cg.add_argument(
        "--first-chapters",
        "--chapters",
        dest="first_chapters",
        type=int,
        default=None,
        help="tag and match only the chunks of the first N body chapters (the whole "
        "book when omitted)",
    )
    p_cg.add_argument(
        "--limit",
        type=int,
        default=None,
        help="tag and match only the first N chunks (a quick look before a full run)",
    )
    p_cg.add_argument(
        "--skip-if-present",
        "--resume",
        dest="skip_if_present",
        action="store_true",
        help="reuse a per-chunk glossary file whose chunk hash, threshold, scorer and "
        "glossary still match; a fully covered window needs no tagger",
    )
    p_cg.add_argument(
        "--no-manifest",
        action="store_true",
        help="do not write _chunk_glossary_manifest.json / _chunk_glossary_trace.json",
    )
    p_cg.add_argument(
        "--dry-run",
        action="store_true",
        help="run both phases but write no per-chunk file, manifest or trace",
    )
    p_cg.add_argument("--json", action="store_true", help="machine-readable result")
    p_cg.add_argument("--quiet", "-q", action="store_true")
    p_cg.set_defaults(func=cmd_chunk_glossary)

    p_matter = sub.add_parser(
        "matter",
        help="stage 6b: localize the front/back matter before translation with its "
        "own LangGraph agent — no dictionary, a web-searched published title, the "
        "BleBel credit, original-edition material omitted, published formatting",
    )
    add_book(p_matter, required=False)
    p_matter.add_argument(
        "--out",
        "-o",
        default=None,
        help="the chapters folder from stage 1 (default: data/output/<book>/01-chapters); "
        "the chunks, translations and matter folders are derived from the book folder",
    )
    p_matter.add_argument(
        "--chapters-in",
        default=None,
        help="the chapter folder the chunks were cut from (default: derived from the "
        "book); point it at 02-preprocessed/ when that is what was chunked",
    )
    p_matter.add_argument(
        "--chunks-in",
        default=None,
        help="the chunks folder from stage 5 (default: data/output/<book>/05-chunks); "
        "a book folder or _chunk_manifest.json is also accepted",
    )
    p_matter.add_argument(
        "--translations-out",
        default=None,
        help="where the localized matter chunks land; this is the same folder stage 7 "
        "writes, so stage 8 assembles them with the body "
        "(default: data/output/<book>/07-translations)",
    )
    p_matter.add_argument(
        "--matter-out",
        default=None,
        help="where the book plan, manifest and trace land "
        "(default: data/output/<book>/06b-matter)",
    )
    p_matter.add_argument(
        "--prompts-out",
        default=None,
        help="where the whole agent input of each matter chunk lands "
        "(default: data/output/<book>/06b-matter-prompts)",
    )
    p_matter.add_argument(
        "--source-language",
        "-s",
        default=DEFAULT_SOURCE_LANGUAGE,
        help=f"the language the book is written in (default: {DEFAULT_SOURCE_LANGUAGE})",
    )
    p_matter.add_argument(
        "--target-language",
        "-t",
        default=DEFAULT_TARGET_LANGUAGE,
        help=f"the language of the translated edition (default: {DEFAULT_TARGET_LANGUAGE})",
    )
    p_matter.add_argument(
        "--book-title",
        default=None,
        help="the source title hint (default: the front matter's title or the filename)",
    )
    add_model_flags(p_matter)
    p_matter.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MATTER_ATTEMPTS,
        help="model turns allowed for the book plan and for each matter chunk "
        f"(default: {DEFAULT_MATTER_ATTEMPTS})",
    )
    p_matter.add_argument(
        "--limit",
        type=int,
        default=None,
        help="localize only the first N matter chunks (a cheap look before the run)",
    )
    p_matter.add_argument(
        "--resume",
        "--skip-if-present",
        dest="resume",
        action="store_true",
        help="reuse a matter chunk whose source hash and language pair still match the "
        "manifest, and continue with the first pending chunk",
    )
    p_matter.add_argument(
        "--no-prompts",
        action="store_true",
        help="do not write the per-chunk agent input records",
    )
    p_matter.add_argument(
        "--no-manifest",
        action="store_true",
        help="do not write _matter_manifest.json / _matter_trace.json",
    )
    p_matter.add_argument(
        "--dry-run",
        action="store_true",
        help="ask the agent for every matter chunk but write no translation, prompt, "
        "manifest or plan file",
    )
    p_matter.set_defaults(func=cmd_matter)

    p_tr = sub.add_parser(
        "translate",
        help="stage 7: translate every chunk in reading order, with the glossary "
        "rows it mentions, into 07-translations/ (English -> Serbian by default) and "
        "keep the whole prompt of each call in 07-translation-prompts/",
    )
    add_book(p_tr, required=False)
    p_tr.add_argument(
        "--out",
        "-o",
        default=None,
        help="the chapters folder from stage 1 (default: data/output/<book>/01-chapters); "
        "the chunks, glossary and translation folders are derived from the book folder",
    )
    p_tr.add_argument(
        "--chunks-in",
        default=None,
        help="the chunks folder from stage 5 (default: data/output/<book>/05-chunks); "
        "a book folder or _chunk_manifest.json is also accepted",
    )
    p_tr.add_argument(
        "--chunk-glossary-in",
        default=None,
        help="stage 6's per-chunk glossary folder "
        "(default: data/output/<book>/06-chunk-glossary)",
    )
    p_tr.add_argument(
        "--glossary-in",
        default=None,
        help="stage 4's glossary folder or glossary.json, used when --glossary-scope "
        "is book (default: data/output/<book>/04-glossary/glossary.json)",
    )
    p_tr.add_argument(
        "--glossary-scope",
        choices=TRANSLATION_GLOSSARY_SCOPES,
        default="chunk",
        help="'chunk' sends the matched terms stage 6 attached to the chunk "
        "(default); 'book' sends the whole glossary with every chunk",
    )
    p_tr.add_argument(
        "--translations-out",
        default=None,
        help="where the translated chunks land (default: data/output/<book>/07-translations)",
    )
    p_tr.add_argument(
        "--prompts-out",
        default=None,
        help="where the per-chunk prompt/input records land "
        "(default: data/output/<book>/07-translation-prompts)",
    )
    p_tr.add_argument(
        "--source-language",
        "-s",
        default=DEFAULT_SOURCE_LANGUAGE,
        help=f"the language the book is written in (default: {DEFAULT_SOURCE_LANGUAGE})",
    )
    p_tr.add_argument(
        "--target-language",
        "-t",
        default=DEFAULT_TARGET_LANGUAGE,
        help=f"the language being translated into (default: {DEFAULT_TARGET_LANGUAGE})",
    )
    p_tr.add_argument(
        "--book-title",
        default=None,
        help="the title the prompt uses (default: the source filename from stage 1)",
    )
    p_tr.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_TRANSLATION_ATTEMPTS,
        help="model calls allowed per chunk before it is reported as failed "
        f"(default: {DEFAULT_TRANSLATION_ATTEMPTS})",
    )
    p_tr.add_argument(
        "--method",
        "--translation-method",
        dest="method",
        choices=TRANSLATION_METHODS,
        default="auto",
        help="'auto' asks for native structured output and moves to JSON mode only "
        "when the provider rejects the schema (default); 'json' starts in JSON mode",
    )
    p_tr.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_TRANSLATION_MAX_TOKENS,
        help="response cap per model call (default: "
        f"{DEFAULT_TRANSLATION_MAX_TOKENS}); 0 leaves it to the provider",
    )
    p_tr.add_argument(
        "--context-chars",
        type=int,
        default=DEFAULT_TRANSLATION_CONTEXT_CHARS,
        help="send the last N characters of the previous translation as continuity "
        f"context (default: {DEFAULT_TRANSLATION_CONTEXT_CHARS} = off)",
    )
    p_tr.add_argument(
        "--first-chapters",
        "--chapters",
        dest="first_chapters",
        type=int,
        default=None,
        help="translate only the chunks of the first N body chapters (the whole book when "
        "omitted); a chapter window, unlike --limit, never stops mid-chapter",
    )
    p_tr.add_argument(
        "--limit",
        type=int,
        default=None,
        help="translate only the first N chunks (a cheap look before a full book)",
    )
    p_tr.add_argument(
        "--resume",
        "--skip-if-present",
        dest="resume",
        action="store_true",
        help="skip chunks that an earlier run already translated (same language pair, "
        "same chunk hash), and continue with the first pending chunk",
    )
    p_tr.add_argument(
        "--no-prompts",
        action="store_true",
        help="do not write the per-chunk prompt/input records",
    )
    p_tr.add_argument(
        "--include-matter",
        action="store_true",
        help="also translate the front/back matter chunks, with this stage's prompt "
        "(the old behaviour; stage 6b normally localizes them first)",
    )
    p_tr.add_argument(
        "--no-manifest",
        action="store_true",
        help="do not write _translation_manifest.json / _translation_trace.json",
    )
    p_tr.add_argument(
        "--dry-run",
        action="store_true",
        help="ask the model for every chunk but write no translation, prompt, "
        "manifest or trace",
    )
    p_tr.add_argument(
        "--translator",
        "--provider",
        dest="translator",
        choices=sorted(TRANSLATION_CHOICES),
        default=None,
        help="model that translates the body chunks: 'deepseek' (the cloud default) "
        "or 'lmstudio' (your local OpenAI-compatible server, "
        "http://127.0.0.1:1234/v1 by default; the loaded model is auto-detected). "
        "This choice affects stage 7 only — front/back matter and the other stages "
        "keep the shared SPLITTER_* settings",
    )
    p_tr.add_argument("--model", default=None, help="the model id (e.g. the id LM Studio shows)")
    p_tr.add_argument("--base-url", default=None)
    p_tr.add_argument("--api-key", default=None)
    p_tr.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="sampling temperature for stage 7 (default: the config file, else 0.0)",
    )
    add_llm_flags(p_tr)
    p_tr.add_argument("--json", action="store_true", help="machine-readable result")
    p_tr.add_argument("--quiet", "-q", action="store_true")
    p_tr.set_defaults(func=cmd_translate)

    p_concat = sub.add_parser(
        "concatenate",
        aliases=["assemble"],
        help="stage 8: concatenate the translated chunks back into one file per "
        "chapter, in a separate 08-translated-chapters/ folder",
    )
    add_book(p_concat, required=False)
    p_concat.add_argument(
        "--out",
        "-o",
        default=None,
        help="the chapters folder from stage 1 (default: data/output/<book>/01-chapters); "
        "used to locate the chunks, translations and book folder",
    )
    p_concat.add_argument(
        "--chunks-in",
        default=None,
        help="the chunks folder from stage 5 (default: data/output/<book>/05-chunks); "
        "a book folder or _chunk_manifest.json is also accepted",
    )
    p_concat.add_argument(
        "--translations-in",
        default=None,
        help="the translated chunks from stage 7 (default: a sibling of 05-chunks/)",
    )
    p_concat.add_argument(
        "--chapters-out",
        default=None,
        help="where the translated chapter files go "
        "(default: data/output/<book>/08-translated-chapters)",
    )
    p_concat.add_argument(
        "--first-chapters",
        "--chapters",
        dest="first_chapters",
        type=int,
        default=None,
        help="assemble only the first N body chapters (the whole translated book when omitted)",
    )
    p_concat.add_argument(
        "--skip-if-present",
        "--resume",
        dest="skip_if_present",
        action="store_true",
        help="reuse a translated chapter file whose bytes already match the current "
        "chunks' translations; only new chapters are written",
    )
    p_concat.add_argument(
        "--no-manifest",
        action="store_true",
        help="do not write _translated_chapters_manifest.json / _translated_chapters_trace.json",
    )
    p_concat.add_argument(
        "--dry-run",
        action="store_true",
        help="assemble every chapter in memory but write no chapter, manifest or trace",
    )
    p_concat.add_argument("--json", action="store_true", help="machine-readable result")
    p_concat.add_argument("--quiet", "-q", action="store_true")
    p_concat.set_defaults(func=cmd_concatenate)

    p_post = sub.add_parser(
        "postprocess",
        help="stage 9: postprocess the translated chapters' headings and table of "
        "contents in a separate folder, without reading chapter prose",
    )
    add_book(p_post, required=False)
    p_post.add_argument(
        "--out",
        "-o",
        default=None,
        help="the chapters folder from stage 1 (default: data/output/<book>/01-chapters); "
        "the translated-chapters and postprocessed-chapters folders are derived from it",
    )
    p_post.add_argument(
        "--translated-chapters-in",
        default=None,
        help="stage 8's output folder (default: data/output/<book>/08-translated-chapters)",
    )
    p_post.add_argument(
        "--chunks-in",
        default=None,
        help="alternative way to locate the book: stage 5's chunks folder; its sibling "
        "08-translated-chapters/ is used as input",
    )
    p_post.add_argument(
        "--postprocessed-out",
        default=None,
        help="where the postprocessed chapters go "
        "(default: data/output/<book>/09-postprocessed-chapters)",
    )
    add_model_flags(p_post)
    p_post.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_POSTPROCESS_ATTEMPTS,
        help="model turns allowed for the book plan and for each chapter "
        f"(default: {DEFAULT_POSTPROCESS_ATTEMPTS})",
    )
    p_post.add_argument(
        "--first-chapters",
        "--chapters",
        dest="first_chapters",
        type=int,
        default=None,
        help="postprocess only front matter plus the first N body chapters "
        "(the whole translated book when omitted)",
    )
    p_post.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process only the first N body chapters (kept as a unit-limit alias)",
    )
    p_post.add_argument(
        "--skip-if-present",
        "--resume",
        dest="skip_if_present",
        action="store_true",
        help="reuse a postprocessed chapter whose source hash and canonical title still "
        "match the manifest, and start with the first chapter that is missing",
    )
    p_post.add_argument(
        "--no-manifest",
        action="store_true",
        help="do not write _postprocess_manifest.json / _postprocess_trace.json",
    )
    p_post.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the book plan and every chapter plan, and compute the TOC, "
        "but write nothing",
    )
    p_post.set_defaults(func=cmd_postprocess)

    p_open = sub.add_parser(
        "openings",
        help="final phase of stages 2 and 9: audit every chapter's first lines and "
        "remove the safe leftovers (no LLM)",
    )
    add_book(p_open, required=False)
    p_open.add_argument(
        "--chapters-in",
        default=None,
        help="the chapter folder to audit (default: the book's most advanced stage "
        "folder: 09-postprocessed-chapters/ → 08-translated-chapters/ → "
        "02-preprocessed/ → 01-chapters/)",
    )
    p_open.add_argument(
        "--keep-blank-slot",
        action="store_true",
        help="stage-2 semantics: a removed line leaves an empty line, so line numbers "
        "keep pointing at the same text (default: the line is removed and the blank "
        "around it collapsed, which is stage 9's behaviour)",
    )
    p_open.add_argument(
        "--dry-run",
        action="store_true",
        help="report every opening finding and write nothing",
    )
    p_open.add_argument("--json", action="store_true", help="machine-readable result")
    p_open.add_argument("--quiet", "-q", action="store_true")
    p_open.set_defaults(func=cmd_openings)

    p_bind = sub.add_parser(
        "bind",
        aliases=["full-book", "compile"],
        help="stage 10: concatenate the postprocessed chapters into one Markdown book",
    )
    add_book(p_bind, required=False)
    p_bind.add_argument(
        "--out",
        "-o",
        default=None,
        help="the chapters folder from stage 1 (default: data/output/<book>/01-chapters)",
    )
    p_bind.add_argument(
        "--postprocessed-in",
        default=None,
        help="stage 9's output folder (default: data/output/<book>/09-postprocessed-chapters)",
    )
    p_bind.add_argument(
        "--book-out",
        default=None,
        help="where the single Markdown book goes "
        "(default: data/output/<book>/10-translated-book.md)",
    )
    p_bind.add_argument(
        "--no-manifest",
        action="store_true",
        help="do not write _bind_manifest.json / _bind_trace.json",
    )
    p_bind.add_argument(
        "--dry-run",
        action="store_true",
        help="assemble and verify the book in memory but write no file",
    )
    p_bind.add_argument("--json", action="store_true", help="machine-readable result")
    p_bind.add_argument("--quiet", "-q", action="store_true")
    p_bind.set_defaults(func=cmd_bind)

    p_epub = sub.add_parser(
        "epub",
        aliases=["ebook"],
        help="stage 11: package the postprocessed chapters as an EPUB 3 book "
        "(images embedded, reader navigation instead of the printed TOC)",
    )
    add_book(p_epub, required=False)
    p_epub.add_argument(
        "--out",
        "-o",
        default=None,
        help="the chapters folder from stage 1 (default: data/output/<book>/01-chapters)",
    )
    p_epub.add_argument(
        "--postprocessed-in",
        default=None,
        help="stage 9's output folder (default: data/output/<book>/09-postprocessed-chapters)",
    )
    p_epub.add_argument(
        "--epub-out",
        default=None,
        help="where the EPUB goes (default: data/output/<book>/11-translated-book.epub)",
    )
    p_epub.add_argument(
        "--title",
        default=None,
        help="override the book title (default: the title stage 6b recorded)",
    )
    p_epub.add_argument(
        "--author",
        default=None,
        help="override the author (default: the author stage 6b recorded)",
    )
    p_epub.add_argument(
        "--language",
        default=None,
        help="override the book language, an ISO 639 code (default: the target "
        "language stage 6b recorded)",
    )
    p_epub.add_argument(
        "--identifier",
        default=None,
        help="override the publication identifier (default: a UUID derived from the "
        "book folder, so rebuilds keep the same book identity)",
    )
    p_epub.add_argument("--description", default=None, help="book description/blurb")
    p_epub.add_argument("--publisher", default=None, help="publisher for the metadata")
    p_epub.add_argument(
        "--cover",
        default=None,
        help="cover picture — a path, or a file name inside the book's "
        "00-markdown/images/ folder",
    )
    p_epub.add_argument(
        "--no-manifest",
        action="store_true",
        help="do not write _epub_manifest.json / _epub_trace.json",
    )
    p_epub.add_argument(
        "--dry-run",
        action="store_true",
        help="build and verify the package in memory but write no file",
    )
    p_epub.add_argument("--json", action="store_true", help="machine-readable result")
    p_epub.add_argument("--quiet", "-q", action="store_true")
    p_epub.set_defaults(func=cmd_epub)

    p_gloss = sub.add_parser(
        "glossary",
        help="stage 4: walk the reduced chapters first to last and build one "
        "book-wide dictionary (term, category, locked target-language translation, "
        "context) with an LLM, feeding each chapter the dictionary so far",
    )
    add_book(p_gloss, required=False)
    p_gloss.add_argument(
        "--out",
        "-o",
        default=None,
        help="the chapters folder from stage 1 (default: data/output/<book>/01-chapters)",
    )
    p_gloss.add_argument(
        "--sentences",
        choices=("unique", "reduced"),
        default="unique",
        help="which stage-3 folder to mine: 'unique' first mentions (default) or "
        "'reduced' every entity-bearing sentence",
    )
    p_gloss.add_argument(
        "--sentences-in",
        default=None,
        help="read the sentences from this folder instead of deriving it from --out",
    )
    p_gloss.add_argument(
        "--glossary-out",
        default=None,
        help="where to write glossary.json / glossary.md "
        "(default: data/output/<book>/04-glossary)",
    )
    p_gloss.add_argument(
        "--source-language",
        "-s",
        default=DEFAULT_SOURCE_LANGUAGE,
        help=f"the language the book is written in (default: {DEFAULT_SOURCE_LANGUAGE})",
    )
    p_gloss.add_argument(
        "--target-language",
        "-t",
        default=DEFAULT_TARGET_LANGUAGE,
        help=f"the language being translated into (default: {DEFAULT_TARGET_LANGUAGE})",
    )
    p_gloss.add_argument(
        "--book-title",
        default=None,
        help="the title the prompt uses (default: the source filename from stage 1)",
    )
    p_gloss.add_argument(
        "--include-front-matter",
        action="store_true",
        help="also mine the front matter (skipped by default)",
    )
    p_gloss.add_argument(
        "--include-back-matter",
        action="store_true",
        help="also mine the back matter (skipped by default)",
    )
    p_gloss.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_GLOSSARY_ATTEMPTS,
        help="model calls allowed per chapter before it is reported as failed "
        f"(default: {DEFAULT_GLOSSARY_ATTEMPTS})",
    )
    p_gloss.add_argument(
        "--extraction-method",
        choices=("auto", "structured", "json"),
        default="auto",
        help="'auto' asks for native structured output and falls back to JSON mode on "
        "the last attempt (default)",
    )
    p_gloss.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="response cap per model call (default: 8192); 0 leaves it to the provider",
    )
    p_gloss.add_argument(
        "--max-chapter-chars",
        type=int,
        default=DEFAULT_GLOSSARY_CHUNK_CHARS,
        help="split a chapter larger than this into parts, carrying the dictionary "
        f"across them (default: {DEFAULT_GLOSSARY_CHUNK_CHARS})",
    )
    p_gloss.add_argument(
        "--max-context-entries",
        type=int,
        default=0,
        help="cap the dictionary sent with each chapter (0 = send all; a cap trades a "
        "little consistency for input tokens on a long book)",
    )
    p_gloss.add_argument(
        "--resume",
        "--skip-if-present",
        dest="resume",
        action="store_true",
        help="reuse the glossary and the chapters already in the manifest, and start "
        "at the first chapter that is missing",
    )
    p_gloss.add_argument(
        "--first-chapters",
        "--chapters",
        dest="first_chapters",
        type=int,
        default=None,
        help="build the dictionary from only the first N body chapters (the whole book "
        "when omitted)",
    )
    p_gloss.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process only the first N body chapters (kept as a unit-limit alias)",
    )
    p_gloss.add_argument(
        "--dry-run",
        action="store_true",
        help="ask the model for every chapter but write no glossary, manifest or trace",
    )
    p_gloss.add_argument("--provider", choices=sorted(PROVIDERS), default=None)
    p_gloss.add_argument("--model", default=None)
    p_gloss.add_argument("--base-url", default=None)
    p_gloss.add_argument("--api-key", default=None)
    p_gloss.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="sampling temperature for stage 4 (default: the config file, else 0.0)",
    )
    add_llm_flags(p_gloss)
    p_gloss.add_argument("--json", action="store_true", help="machine-readable result")
    p_gloss.add_argument("--quiet", "-q", action="store_true")
    p_gloss.set_defaults(func=cmd_glossary)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    first_chapters = getattr(args, "first_chapters", None)
    if first_chapters is not None and int(first_chapters) < 0:
        print("error: --first-chapters cannot be negative", file=sys.stderr)
        return 2
    # Validate --llm syntax even when the command ends up needing no model (a
    # --map replay, a fully resumed run); the command's own stage is filled in
    # when the setting is actually resolved.
    try:
        from .core.settings import parse_llm_overrides

        parse_llm_overrides(getattr(args, "llm", None), default_stage="split")
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ToolError as exc:
        # Stage 0's unsupported/image-based source and the other deterministic
        # stages report one actionable line instead of a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        # Includes FileNotFoundError: any disk-level problem (a read-only cache, a
        # missing folder on a computed path) is one actionable line, not a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
