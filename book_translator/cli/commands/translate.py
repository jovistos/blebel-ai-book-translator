"""translate command: implementation and its human summary."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Any

from book_translator.core.constants import (
    DEFAULT_SOURCE_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
    DEFAULT_TRANSLATION_ATTEMPTS,
    DEFAULT_TRANSLATION_CONTEXT_CHARS,
    DEFAULT_TRANSLATION_MAX_TOKENS,
    TRANSLATION_GLOSSARY_SCOPES,
    TRANSLATION_METHODS,
)
from book_translator.core.errors import ConfigError, MissingDependencyError, ToolError
from book_translator.core.report import human_seconds
from book_translator.core.settings import TRANSLATION_CHOICES
from book_translator.documents.chapters import (
    chapters_dir,
    chunks_dir,
    glossary_dir,
    has_chunk_files,
    translation_prompts_dir,
    translations_dir,
)
from book_translator.documents.window import (
    chapter_files_for_window,
)

from .. import parents
from ..common import (
    _model_settings,
    _print_json,
    _resolve_text_dir,
)


def add_parser(sub: Any) -> None:
    """Register this command's parser (the help text is snapshot-tested)."""
    p_tr = sub.add_parser(
        "translate",
        help="stage 7: translate every chunk in reading order, with the glossary "
        "rows it mentions, into 07-translations/ (English -> Serbian by default) and "
        "keep the whole prompt of each call in 07-translation-prompts/",
    )
    parents.add_book(p_tr, required=False)
    p_tr.add_argument(
        "--out",
        "-o",
        default=None,
        help="the chapter folder (default: the cleaned data/output/<book>/02-preprocessed "
        "when stage 2 wrote it, else data/output/<book>/01-chapters); "
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
    parents.add_llm_flags(p_tr)
    p_tr.add_argument("--json", action="store_true", help="machine-readable result")
    p_tr.add_argument("--quiet", "-q", action="store_true")
    p_tr.add_argument(
        "--text-in",
        default=None,
        help="chapter text folder to read (default: 02-preprocessed/ when it covers "
        "the window, else 01-chapters/); overrides --out",
    )
    p_tr.set_defaults(func=cmd_translate)


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
            f"  [{done:>3}/{total}] {flag} {record.get('filename') or '?'!s:<46} "
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


def cmd_translate(args: argparse.Namespace) -> int:
    """Stage 7: translate each chunk in reading order, with its glossary rows."""
    from book_translator.stages.chunk_glossary.runner import resolve_chunks_dir
    from book_translator.stages.translation import run_translation
    from book_translator.stages.translation.runner import resolve_chunk_glossary_dir

    chapters_path = chapters_dir(_resolve_text_dir(
            args.book,
            args.in_dir,
            args.out,
            getattr(args, "output_root", None),
            getattr(args, "text_in", None),
        ))
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
            f"time            : {human_seconds(float(timing.get('run_seconds', 0.0)))}"
            f"  (model {human_seconds(float(timing.get('model_seconds', 0.0)))})"
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
