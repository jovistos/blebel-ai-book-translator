#!/usr/bin/env python
"""The whole book pipeline in one command: recon → split → scan → dictionary
(→ preprocess) → chunk → chunk-glossary.

`split`, `preprocess`, `dictionary`, `chunk` and `chunk-glossary` are separate
stages behind their own CLIs (the dictionary stage is also the book's glossary
builder). This script runs them in the only order that works, from one place, and
narrates each phase with the numbers that matter: how big the book is, which
strategy won, whether the chapters reassemble byte for byte, what damage was found,
how many sentences survived the dictionary funnel, how many entities were locked
into the glossary, what the LLM pass cost in tokens, and how many glossary terms
each chunk mentions.

The ``dictionary`` phase has two halves. The first is deterministic — the sentence
funnel and the NER gate, no model. The second is the glossary: one paid model call
per chapter, each handed the dictionary the earlier chapters produced. The second
half is skipped by ``--no-glossary`` (the phase stays deterministic and keyless)
and by ``--dry-run`` (nothing to validate without asking the model, and a dry run
must not spend).

    python scripts/run_pipeline.py                        # the only book in data/input/
    python scripts/run_pipeline.py BOOK.md --debug        # + headers, files, damage examples
    python scripts/run_pipeline.py --dry-run              # validate every plan, write no artifacts
    python scripts/run_pipeline.py --stop-after split     # stop once the chapters exist
    python scripts/run_pipeline.py --start-at dictionary  # stage 3 only, over existing chapters
    python scripts/run_pipeline.py --start-at chunk        # stage 5 only, over existing chapters
    python scripts/run_pipeline.py --start-at chunk_glossary
                                                          # stage 6 only, over existing chunks
    python scripts/run_pipeline.py --chunk-size 700        # smaller translation chunks
    python scripts/run_pipeline.py --match-threshold 90    # more permissive glossary retrieval
    python scripts/run_pipeline.py BOOK.md --map map.json --stop-after split
                                                          # replay a supplied map: no LLM at all

Stage 2 (``preprocess``) is **switched off for now**: a plain run skips it and
reports it as such, while the phases after it still run. The phase and everything
it needs are still in the tree — ``--with-preprocess`` runs it again, and setting
``PREPROCESS_ENABLED = True`` at the top of this file makes it the default again.

Stage 3 (``dictionary``) is on by default but needs an optional tagger
(``pip install -r requirements-dictionary.txt``). When that dependency is missing
the phase reports itself as skipped with the install command instead of failing
the run, so a split-only workflow keeps working.

The chunk phase (stage 5, ``chunk``) cuts every chapter into translation-sized
chunks with chonkie, preferring paragraph over sentence over line over word
boundaries. Every chunk is an exact slice of the chapter it came from, so the
chunks concatenate back to the chapter (and to the book) byte for byte. The
manifest records the chapter membership, the global order, the character offsets
and the boundary kind of every cut. ``--chunk-size`` is in tokens, default 1000.
A missing ``chonkie`` installation is reported as a skipped phase, not a failure,
so a split-only run keeps working.

The last phase (stage 6, ``chunk_glossary``) is the bridge to translation. It runs
NER over **every** chunk first, then fuzzy-matches each extracted term against the
glossary with rapidfuzz (95 % by default) and writes the terms a chunk mentions to
``chunk-glossary/<chunk>.json``. It needs both stage 4's ``glossary.json`` and
stage 5's chunks; without them it reports a skip/actionable error. ``--no-chunk-glossary``
drops it, and ``--no-chunk`` drops it too.

Everything printed is also written to ``data/output/<book>/pipeline.log``
(``--log-file`` to move it, ``--no-log`` to skip it). The log always keeps the
debug level, even when the console does not.

Phases are independent of each other only in one direction: ``scan``,
``dictionary``, ``preprocess``, ``chunk`` and ``chunk_glossary`` read folders an
earlier phase writes, so ``--start-at dictionary`` needs a previous run. A
``--dry-run`` split writes no chapters, and the later phases are then reported as
skipped instead of failing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from book_splitter.analysis.candidates import extract_candidates  # noqa: E402
from book_splitter.analysis.headers import assess_headers  # noqa: E402
from book_splitter.analysis.toc import detect_toc_block  # noqa: E402
from book_splitter.core.constants import (  # noqa: E402
    CHUNK_GLOSSARY_SCORERS,
    DEFAULT_CHUNK_GLOSSARY_CANDIDATES,
    DEFAULT_CHUNK_GLOSSARY_MIN_TERM_CHARS,
    DEFAULT_CHUNK_GLOSSARY_SCORER,
    DEFAULT_CHUNK_GLOSSARY_THRESHOLD,
    DEFAULT_CHUNK_TOKENIZER,
    DEFAULT_CHUNK_TOKENS,
    DEFAULT_FRONT_MATTER_LINES,
    DEFAULT_GPU_INDEX,
    DEFAULT_KEEP_ENTITY_TYPES,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MIN_CHAPTERS,
    DEFAULT_MIN_SENTENCE_WORDS,
    DEFAULT_RECURSION_LIMIT,
    DEFAULT_SOURCE_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
    MAX_FEEDBACK_LOOPS,
    TOC_SCAN_LINES,
)
from book_splitter.core.errors import ConfigError, MissingDependencyError, ToolError  # noqa: E402
from book_splitter.core.settings import (  # noqa: E402
    PROVIDERS,
    load_dotenv_files,
    resolve_settings,
)
from book_splitter.documents.chapters import (  # noqa: E402
    chapters_dir,
    chunk_glossary_dir,
    chunks_dir,
    discover_chapters,
    glossary_dir,
    has_chapter_files,
    has_chunk_files,
    load_split_manifest,
    preprocessed_dir,
    reduced_dir,
    unique_dir,
)
from book_splitter.documents.io import load_lines, source_size_bytes  # noqa: E402
from book_splitter.paths import (  # noqa: E402
    book_output_dir,
    default_input_dir,
    default_output_dir,
    resolve_book_path,
    resolve_output_dir,
)
from book_splitter.preprocess.corpus import scan_corpus_damage  # noqa: E402

#: The phases, in the order they must run. The glossary is not a phase of its
#: own: it is the second half of ``dictionary`` (reduce the chapters, then turn
#: them into the book's dictionary). ``chunk`` is the translation-sized cut that
#: runs last: it does not depend on the dictionary, but it reads the chapter folder
#: those phases start from. ``chunk_glossary`` joins the two: it tags every chunk
#: first, then attaches the glossary entries the chunk mentions.
PHASES = ("recon", "split", "scan", "dictionary", "preprocess", "chunk", "chunk_glossary")

#: Phases that need the folder stage 1 writes.
NEEDS_CHAPTERS = ("scan", "dictionary", "preprocess", "chunk", "chunk_glossary")

# TEMPORARY: stage 2 (preprocess) is switched off, so a plain run stops after
# ``dictionary``. Nothing was removed — ``phase_preprocess``, the stage-2 tools and
# the stage-2 agent are all still here. Set this to ``True`` (or pass
# ``--with-preprocess``) to run it again.
PREPROCESS_ENABLED = False

# Stage 3 (dictionary) needs an optional tagger. It stays selected, and reports
# itself as skipped when the dependency is absent, so a split-only run still
# finishes. ``--no-dictionary`` drops the phase entirely.
DICTIONARY_ENABLED = True

# The glossary is built by the ``dictionary`` phase, after the reduction: that is
# the one step that calls a paid model once per chapter. ``--no-glossary`` keeps
# the phase deterministic (reduction only), and ``--dry-run`` skips the glossary
# half, because there is nothing to validate without asking the model.
GLOSSARY_ENABLED = True

# Stage 5 (chunking) is on by default and needs the optional ``chonkie`` package.
# A missing dependency is reported as a skip, the way the dictionary stage reports
# a missing tagger, so the earlier phases still finish. ``--no-chunk`` drops it.
CHUNK_ENABLED = True

# Stage 6 (chunk glossary) is on by default. It reuses stage 3's tagger and needs
# stage 4's glossary.json, so a missing dependency or a missing glossary is
# reported as a skip/actionable error rather than a traceback. ``--no-chunk-glossary``
# drops it; ``--no-chunk`` drops it too, because there are no chunks to annotate.
CHUNK_GLOSSARY_ENABLED = True

TITLES = {
    "recon": "what the book actually contains (no LLM, no writes)",
    "split": "stage 1 — the agent routes, the lines are copied verbatim",
    "scan": "stage 2 pre-flight — the damage profile (no LLM)",
    "dictionary": "stage 3 — reduce the chapters, then build the book's dictionary",
    "preprocess": "stage 2 — clean each chapter, then prove the edit",
    "chunk": "stage 5 — cut each chapter into reversible, sentence-safe chunks (no LLM)",
    "chunk_glossary": "stage 6 — tag every chunk, then attach the glossary terms it mentions",
}

RULE = "─" * 72


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #
def human_bytes(size: float) -> str:
    """``411144`` -> ``401.5 KiB``."""
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024:
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def human_seconds(seconds: float) -> str:
    """``389.42`` -> ``6m 29.4s``."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {rest:04.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes:02d}m {rest:04.1f}s"


def rel(path: str | Path) -> str:
    """A path relative to the repository root when it lives inside it."""
    resolved = Path(path).expanduser()
    try:
        return str(resolved.resolve().relative_to(ROOT))
    except ValueError:
        return str(resolved)


def has_chapters(path: Path) -> bool:
    """True when ``path`` holds at least one chapter file."""
    return has_chapter_files(path)


def _entity_types(value: str | None) -> tuple[str, ...]:
    """``"PER, loc"`` -> ``("PER", "LOC")``; empty means the default four."""
    if not value:
        return DEFAULT_KEEP_ENTITY_TYPES
    wanted = tuple(part.strip().upper() for part in str(value).split(",") if part.strip())
    return wanted or DEFAULT_KEEP_ENTITY_TYPES


# --------------------------------------------------------------------------- #
# console: stdout + the run log
# --------------------------------------------------------------------------- #
class Console:
    """Narrates the run to stdout **and** to the run's log file.

    The log receives every line, debug included; the console only shows debug
    lines with ``--debug``. Writes are locked, because stage 2 reports progress
    from worker threads when ``--workers`` is greater than one.
    """

    def __init__(self, log_path: Path | None, debug: bool = False) -> None:
        self.debug_enabled = debug
        self.log_path = log_path
        self._lock = threading.Lock()
        self._log = None
        self.bar_enabled = sys.stdout.isatty()
        self._bar_drawn = False
        self._bar_width = 0
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = log_path.open("w", encoding="utf-8")

    def emit(self, text: str = "", level: str = "info") -> None:
        with self._lock:
            self._clear_bar_locked()
            if self._log is not None:
                self._log.write(text + "\n")
                self._log.flush()
            if level == "debug" and not self.debug_enabled:
                return
            stream = sys.stderr if level == "error" else sys.stdout
            print(text, file=stream, flush=True)

    # -- the live bar ----------------------------------------------------- #
    def bar(self, text: str) -> None:
        """Redraw the one-line progress bar in place (terminal only).

        The bar is deliberately *not* written to the log: it is redrawn several
        times a second, and the log is a transcript of the run, not a recording
        of a cursor. Anything that must survive the run goes through ``info``.
        """
        if not self.bar_enabled:
            return
        with self._lock:
            padded = text + " " * max(0, self._bar_width - len(text))
            print("\r" + padded, end="", file=sys.stdout, flush=True)
            self._bar_width = max(self._bar_width, len(text))
            self._bar_drawn = True

    def stream_line(self, text: str) -> None:
        """A transient progress line for a non-terminal stdout (piped / logged).

        Same job as ``bar`` where a cursor cannot be moved: print the state, but
        keep it out of the log — the log keeps the durable per-chapter lines.
        """
        with self._lock:
            self._clear_bar_locked()
            print(text, file=sys.stdout, flush=True)

    def clear_bar(self) -> None:
        with self._lock:
            self._clear_bar_locked()

    def _clear_bar_locked(self) -> None:
        if self._bar_drawn:
            print("\r" + " " * self._bar_width + "\r", end="", file=sys.stdout, flush=True)
            self._bar_drawn = False

    def info(self, text: str = "") -> None:
        self.emit(text)

    def debug(self, text: str = "") -> None:
        self.emit(text, "debug")

    def error(self, text: str = "") -> None:
        self.emit(text, "error")

    def log_only(self, text: str = "") -> None:
        """Write to the log without touching the console (used for provenance)."""
        with self._lock:
            if self._log is not None:
                self._log.write(text + "\n")
                self._log.flush()

    def kv(self, key: str, value: Any, indent: int = 2, level: str = "info") -> None:
        self.emit(f"{' ' * indent}{key:<14}: {value}", level)

    def phase(self, index: int, total: int, name: str, title: str) -> None:
        self.emit("")
        self.emit(RULE)
        self.emit(f" [{index}/{total}] {name.upper()}")
        self.emit(f"       {title}")
        self.emit(RULE)

    def close(self) -> None:
        with self._lock:
            if self._log is not None:
                self._log.close()
                self._log = None


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #
class PhaseFailed(RuntimeError):
    """A phase ran and did not succeed; the pipeline stops here."""

    def __init__(self, message: str, summary: str = "failed") -> None:
        super().__init__(message)
        self.summary = summary


@dataclass
class PhaseRecord:
    """What one phase did, for the closing summary and ``--json``."""

    name: str
    ok: bool = False
    seconds: float = 0.0
    summary: str = ""
    error: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


class Pipeline:
    """Runs the selected phases in order and reports on each one."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        # The flag above is the default; ``--with-preprocess`` overrides it for
        # this run only, so the phase can still be exercised while it is off.
        self.preprocess_enabled = bool(
            PREPROCESS_ENABLED or getattr(args, "with_preprocess", False)
        )
        self.dictionary_enabled = bool(
            DICTIONARY_ENABLED and not getattr(args, "no_dictionary", False)
        )
        # The glossary belongs to the dictionary phase; ``--with-glossary`` is
        # still accepted so an older command line keeps working.
        self.glossary_enabled = bool(
            GLOSSARY_ENABLED
            and not getattr(args, "no_glossary", False)
            and self.dictionary_enabled
        )
        self.chunk_enabled = bool(CHUNK_ENABLED and not getattr(args, "no_chunk", False))
        self.chunk_glossary_enabled = bool(
            CHUNK_GLOSSARY_ENABLED
            and not getattr(args, "no_chunk_glossary", False)
            and self.chunk_enabled
        )
        self.records: list[PhaseRecord] = []
        self.exit_code = 0
        self.started = time.monotonic()
        self._settings = None
        self._progress_lock = threading.Lock()
        self._bar_stop = threading.Event()
        self._bar_thread: threading.Thread | None = None
        self._bar_state: dict[str, Any] = {"active": False, "total": 0, "done": 0}
        self._last_tick = self.started
        self._count_width = 1
        self._name_width = 24

        load_dotenv_files()  # so the banner's model label matches the real resolution
        self.source = resolve_book_path(args.book, input_dir=args.in_dir)
        self.book_dir = book_output_dir(self.source, output_root=args.output_root)
        self.chapters_dir = resolve_output_dir(self.source, args.out, output_root=args.output_root)
        self.preprocessed_out = (
            Path(args.preprocessed_out).expanduser()
            if args.preprocessed_out
            else preprocessed_dir(self.chapters_dir)
        )
        self.reduced_out = (
            Path(args.reduced_out).expanduser()
            if args.reduced_out
            else reduced_dir(self.chapters_dir)
        )
        self.unique_out = (
            Path(args.unique_out).expanduser()
            if args.unique_out
            else unique_dir(self.reduced_out)
        )
        self.glossary_out = (
            Path(args.glossary_out).expanduser()
            if getattr(args, "glossary_out", None)
            else glossary_dir(self.chapters_dir)
        )
        # The chunker cuts what stage 1 wrote by default. ``--chunk-in`` points it
        # at the cleaned text (or any other chapters folder) instead.
        self.chunk_in = (
            Path(args.chunk_in).expanduser()
            if getattr(args, "chunk_in", None)
            else self.chapters_dir
        )
        self.chunks_out = (
            Path(args.chunks_out).expanduser()
            if getattr(args, "chunks_out", None)
            else chunks_dir(self.chapters_dir)
        )
        self.chunk_glossary_out = (
            Path(args.chunk_glossary_out).expanduser()
            if getattr(args, "chunk_glossary_out", None)
            else chunk_glossary_dir(self.chunks_out)
        )

        log_path = None
        if not args.no_log:
            log_path = (
                Path(args.log_file).expanduser()
                if args.log_file
                else self.book_dir / "pipeline.log"
            )
        self.console = Console(log_path, debug=args.debug)

    # -- setup ------------------------------------------------------------- #
    @property
    def settings(self):
        """Resolve the model connection once, and only when a phase needs it."""
        if self._settings is None:
            self._settings = resolve_settings(
                provider=self.args.provider,
                model=self.args.model,
                base_url=self.args.base_url,
                api_key=self.args.api_key,
                temperature=self.args.temperature,
            )
        return self._settings

    def _selected_phases(self) -> list[str]:
        start = PHASES.index(self.args.start_at)
        stop = PHASES.index(self.args.stop_after)
        phases = list(PHASES[start : stop + 1])
        if not self.preprocess_enabled:
            # Switched off for now: the phase is dropped, not reordered, so the
            # remaining phases and the order they run in are unchanged.
            phases = [name for name in phases if name != "preprocess"]
        if not self.dictionary_enabled:
            phases = [name for name in phases if name != "dictionary"]
        if not self.chunk_enabled:
            phases = [name for name in phases if name != "chunk"]
        if not self.chunk_glossary_enabled:
            phases = [name for name in phases if name != "chunk_glossary"]
        return phases

    def _switched_off_phases(self) -> list[str]:
        """Phases the user asked for that a temporary switch dropped."""
        start = PHASES.index(self.args.start_at)
        stop = PHASES.index(self.args.stop_after)
        return [name for name in PHASES[start : stop + 1] if name not in self._selected_phases()]

    def close(self) -> None:
        self.console.close()

    # -- the run ----------------------------------------------------------- #
    def run(self) -> int:
        phases = self._selected_phases()
        self._banner(phases)
        for name in self._switched_off_phases():
            flag = {
                "preprocess": "--with-preprocess",
                "dictionary": "--with-dictionary",
                "chunk": "--no-chunk",
                "chunk_glossary": "--no-chunk-glossary",
            }.get(name, "--with-" + name)
            self.console.info(f"  · {name}: switched off for now — skipped (see {flag})")
        for index, name in enumerate(phases, start=1):
            record = PhaseRecord(name=name)
            self.records.append(record)
            title = TITLES[name]
            if name == "split" and getattr(self.args, "map", None):
                title = "stage 1 — a supplied map routes, the lines are copied verbatim"
            self.console.phase(index, len(phases), name, title)

            skip_for_dry_run = (
                name in NEEDS_CHAPTERS and self.args.dry_run and not has_chapters(self.chapters_dir)
            )
            if skip_for_dry_run:
                record.ok = True
                record.summary = "skipped — a dry run writes no chapters"
                self.console.info("  · skipped: a --dry-run split writes no chapter files to scan")
                continue

            if name == "chunk_glossary" and not has_chunk_files(self.chunks_out):
                record.ok = True
                record.summary = "skipped — no chunks on disk to annotate"
                self.console.info(
                    "  · skipped: no chunk files — run the chunk phase (or a real run) first"
                )
                continue

            tick = time.monotonic()
            try:
                summary, payload = getattr(self, f"phase_{name}")()
            except PhaseFailed as exc:
                record.summary, record.error = exc.summary, str(exc)
            except ConfigError as exc:
                record.error, self.exit_code = str(exc), 2
            except (ToolError, FileNotFoundError) as exc:
                record.error = str(exc)
            except KeyboardInterrupt:  # pragma: no cover - interactive only
                self.console.error("\n  interrupted")
                self.exit_code = 130
                break
            except Exception as exc:  # a crash must still be reported and logged
                record.error = f"{type(exc).__name__}: {exc}"
                self.console.debug(traceback.format_exc())
                if self.args.debug:
                    traceback.print_exc()
            finally:
                record.seconds = time.monotonic() - tick

            if record.error:
                self.console.error(f"  ✗ {name}: {record.error}")
                break
            record.ok, record.summary, record.payload = True, summary, payload
            self.console.info(f"  ✓ {name} finished in {human_seconds(record.seconds)}")

        return self._finish()

    def _banner(self, phases: list[str]) -> None:
        console = self.console
        console.log_only(f"# run_pipeline.py · {datetime.now().isoformat(timespec='seconds')}")
        console.log_only(f"# argv: {' '.join(sys.argv[1:]) or '(none)'}")
        console.log_only("")
        console.info(RULE)
        console.info(" book_splitter · the whole pipeline, one command")
        console.info(RULE)
        try:
            lines = len(load_lines(self.source))
            size = source_size_bytes(self.source)
            console.kv("book", f"{rel(self.source)} — {human_bytes(size)}, {lines:,} lines")
        except OSError as exc:  # the path was resolved, so this is unusual
            console.kv("book", f"{rel(self.source)} — unreadable: {exc}")
        console.kv("output", rel(self.book_dir))
        console.kv("chapters", rel(self.chapters_dir))
        console.kv("reduced", rel(self.reduced_out))
        console.kv("unique", rel(self.unique_out))
        console.kv("preprocessed", rel(self.preprocessed_out))
        console.kv("chunks", rel(self.chunks_out))
        console.kv("chunk gloss", rel(self.chunk_glossary_out))
        if not self.chunk_enabled:
            console.kv("stage 5", "chunking switched off — remove --no-chunk to run it")
        else:
            console.kv(
                "chunk size",
                f"{self.args.chunk_size:,} tokens · tokenizer {self.args.chunk_tokenizer}",
            )
        if not self.chunk_glossary_enabled:
            console.kv(
                "stage 6",
                "chunk glossary switched off — remove --no-chunk-glossary to run it",
            )
        else:
            console.kv(
                "match",
                f"{self.args.match_threshold}% · scorer {self.args.match_scorer}",
            )
        console.kv("phases", " → ".join(phases) or "(none — every selected phase is switched off)")
        if not self.preprocess_enabled:
            console.kv("stage 2", "preprocess switched off — --with-preprocess to run it")
        if not self.dictionary_enabled:
            console.kv("stage 3", "dictionary switched off — --with-dictionary to run it")
        elif self.glossary_enabled:
            console.kv("glossary", rel(self.glossary_out) + "  (built by the dictionary phase)")
        else:
            console.kv("glossary", "off — --no-glossary is set (reduction only)")
        mode = "DRY RUN — nothing is written" if self.args.dry_run else "live"
        console.kv("mode", mode)
        if console.log_path is not None:
            console.kv("log", rel(console.log_path))

    # -- phase 1: recon ---------------------------------------------------- #
    def phase_recon(self) -> tuple[str, dict[str, Any]]:
        """Deterministic structural report: no LLM, no key, no writes."""
        from book_splitter.agent.tools import scan_markdown_headers

        args, console = self.args, self.console
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
        from book_splitter.agent import (
            result_from_existing_chapters,
            run_split,
            run_split_from_map,
        )

        args, console = self.args, self.console
        console.kv("source", rel(self.source))
        console.kv("output", rel(self.chapters_dir))

        if args.skip_if_present and not args.dry_run and has_chapters(self.chapters_dir):
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

        if args.map:
            from book_splitter.slicing.mapfile import load_split_map

            mapping, notes = load_split_map(args.map)
            console.kv("map", f"{rel(args.map)} · {len(mapping)} split point(s)")
            for line in notes:
                console.info(f"     · {line}")
            console.kv("budget", f"no agent turns · min {args.min_chapters} chapters")
            console.info("     the map is validated, then a Python node copies the source lines")
            result = run_split_from_map(
                str(self.source),
                str(self.chapters_dir),
                mapping,
                min_chapters=args.min_chapters,
                include_front_matter=not args.no_front_matter,
                dry_run=args.dry_run,
                skip_if_present=args.skip_if_present,
            )
        else:
            console.kv("provider", f"{self.settings.provider} · {self.settings.model}")
            console.kv(
                "budget",
                f"max {args.max_attempts} agent turn(s), min {args.min_chapters} chapters",
            )
            console.info(
                "     the model only routes; a Python node copies the source lines verbatim,"
            )
            console.info("     and the verifier proves the chapters reassemble byte for byte")
            result = run_split(
                str(self.source),
                str(self.chapters_dir),
                settings=self.settings,
                front_matter_lines=args.front_matter_lines,
                max_attempts=args.max_attempts,
                min_chapters=args.min_chapters,
                include_front_matter=not args.no_front_matter,
                dry_run=args.dry_run,
                recursion_limit=args.recursion_limit,
                skip_if_present=args.skip_if_present,
            )

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
        """Deterministic damage profile of the chapters stage 1 just wrote."""
        chapters = discover_chapters(self.chapters_dir)
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
        console.kv(
            "damage",
            f"{int(scan.get('total_findings', 0)):,} findings in {len(counts)} shape(s)",
        )
        for name, value in sorted(counts.items(), key=lambda item: -item[1]):
            console.info(f"      {name:<26} {value:>5}")
        if not counts:
            console.info("      none recognised — stage 2 has nothing to repair")

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
        of failing the run, the way stage 2 has its off switch.
        """
        from book_splitter.dictionary import run_dictionary

        args, console = self.args, self.console
        save_unique = not args.no_save_unique
        save_reduced = not args.no_save_reduced and not args.unique_only
        console.kv("input", rel(self.chapters_dir))
        console.kv("output", rel(self.reduced_out))
        if save_unique:
            console.kv("unique", rel(self.unique_out))
        console.kv("provider", "none — deterministic, no LLM")
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
            result = run_dictionary(
                str(self.chapters_dir),
                str(self.reduced_out),
                unique_output_dir=str(self.unique_out),
                language=args.language,
                gpu=args.gpu,
                use_gpu=not args.cpu,
                min_words=args.min_words,
                require_uppercase=not args.no_uppercase_filter,
                keep_types=_entity_types(args.keep_types),
                drop_temporal_only=args.drop_temporal_only,
                skip_if_present=args.skip_if_present,
                save_reduced=save_reduced,
                save_unique=save_unique,
                dry_run=args.dry_run,
                limit=args.limit,
                write_manifest=not args.no_manifest and not args.dry_run,
            )
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
        if self.args.dry_run:
            self.console.info("  · glossary skipped: a --dry-run does not spend tokens")
            return "skipped (dry run)", {"dry_run": True}
        return self._build_glossary()

    # -- the glossary, run inside the dictionary phase --------------------- #
    def _build_glossary(self) -> tuple[str, dict[str, Any]]:
        """Stage 4: grow one book-wide glossary, one chapter at a time.

        Part of the ``dictionary`` phase: the reduction produces the sentences,
        this turns them into the dictionary. It is the only step that calls a paid
        model per chapter, so it narrates itself: a live bar while the run is in
        flight, one line per finished chapter, and the token bill at the end.
        ``--dry-run`` reports itself as skipped instead of calling — an extraction
        has nothing to validate without asking the model, and a dry run must not
        spend.
        """
        from book_splitter.glossary import run_glossary
        from book_splitter.glossary.runner import resolve_sentences_dir

        args, console = self.args, self.console
        sentences_path = resolve_sentences_dir(
            self.chapters_dir,
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
        console.kv("model", self._model_label())
        console.kv(
            "skips",
            ("front matter, " if not getattr(args, "include_front_matter", False) else "")
            + ("back matter" if not getattr(args, "include_back_matter", False) else "")
            or "nothing",
        )

        self._bar_state = {"active": False, "total": 0, "done": 0, "current": ""}
        try:
            result = run_glossary(
                str(self.chapters_dir),
                str(self.glossary_out),
                sentences_dir=str(sentences_path),
                sentences=args.sentences,
                source_language=args.source_language,
                target_language=args.target_language,
                settings_factory=lambda: self.settings,
                max_attempts=args.max_attempts,
                max_context_entries=args.max_context_entries,
                include_front_matter=getattr(args, "include_front_matter", False),
                include_back_matter=getattr(args, "include_back_matter", False),
                skip_if_present=args.skip_if_present,
                limit=args.limit,
                progress=self._on_glossary_chapter,
                chapter_start=self._on_glossary_start,
            )
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

    def _model_label(self) -> str:
        """The configured provider/model, without resolving (or requiring) a key."""
        provider = (
            self.args.provider or os.environ.get("SPLITTER_PROVIDER") or "deepseek"
        ).lower()
        spec = PROVIDERS.get(provider, {})
        model = self.args.model or os.environ.get("SPLITTER_MODEL") or spec.get("model") or "—"
        return f"{provider}/{model}"

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
            f"  [{done:>2}/{total}] {str(chapter.get('filename') or '?'):<44} "
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
        from book_splitter.preprocess_agent import run_preprocess

        args, console = self.args, self.console
        chapters = discover_chapters(self.chapters_dir)
        console.kv("input", rel(self.chapters_dir))
        console.kv("output", rel(self.preprocessed_out))
        console.kv("provider", f"{self.settings.provider} · {self.settings.model}")
        console.kv("chapters", f"{len(chapters)} · one agent run each · workers={args.workers}")
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

        result = run_preprocess(
            str(self.chapters_dir),
            str(self.preprocessed_out),
            settings=self.settings,
            max_attempts=args.max_attempts,
            dry_run=args.dry_run,
            recursion_limit=args.recursion_limit,
            workers=args.workers,
            write_manifest=not args.no_manifest and not args.dry_run,
            progress=self._on_chapter,
        )

        records = result.chapters or []
        operations = sum(len(r.get("operations") or []) for r in records)
        fixes = sum(len(r.get("text_fixes") or []) for r in records)
        dropped = sum(len(r.get("removed_artefacts") or []) for r in records)
        lines_in = sum(int(r.get("source_lines") or 0) for r in records)
        lines_out = sum(int(r.get("output_lines") or 0) for r in records)
        calls = sum(int(r.get("attempts") or 0) for r in records)
        tokens = int((result.usage or {}).get("total_tokens", 0) or 0)

        console.kv("chapters", f"{result.ok_count}/{result.chapter_count} clean")
        console.kv("edits", f"{operations} operations · {dropped} lines dropped · {fixes} fixes")
        if not args.dry_run:
            console.kv("lines", f"{lines_in:,} → {lines_out:,}")
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
        from book_splitter.chunking import run_chunking

        args, console = self.args, self.console
        input_dir = chapters_dir(self.chunk_in)
        if not has_chapter_files(input_dir):
            raise PhaseFailed(
                f"no chapter files in {rel(input_dir)} — run split first",
                summary="no chapters to chunk",
            )
        console.kv("input", rel(input_dir))
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
            result = run_chunking(
                str(input_dir),
                str(self.chunks_out),
                chunk_size=args.chunk_size,
                tokenizer=args.chunk_tokenizer,
                skip_if_present=args.skip_if_present,
                dry_run=args.dry_run,
                limit=args.limit,
                write_manifest=not args.no_manifest and not args.dry_run,
            )
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
        from book_splitter.chunk_glossary import run_chunk_glossary
        from book_splitter.core.constants import GLOSSARY_FILENAME

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

        try:
            result = run_chunk_glossary(
                str(self.chunks_out),
                str(self.chunk_glossary_out),
                glossary_path=str(glossary_path),
                language=args.language,
                gpu=args.gpu,
                use_gpu=not args.cpu,
                keep_types=_entity_types(args.keep_types),
                min_term_chars=args.min_term_chars,
                drop_temporal_only=args.drop_temporal_only,
                threshold=args.match_threshold,
                scorer=args.match_scorer,
                candidates=args.match_candidates,
                dry_run=args.dry_run,
                limit=args.limit,
                write_manifest=not args.no_manifest and not args.dry_run,
            )
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

    def _recorded_strategy(self, manifest: dict[str, Any]) -> str:
        """Which strategy won.

        Stage 1's manifest does not carry it (it records line ranges and hashes),
        so fall back to the strategy in ``_agent_trace.json``.
        """
        from book_splitter.core.constants import TRACE_FILENAME

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

    def _log_paths(self, label: str, path: str | None) -> None:
        if path and Path(path).exists():
            self.console.debug(f"      {label}: {rel(path)}")

    # -- closing report ---------------------------------------------------- #
    def _finish(self) -> int:
        from book_splitter.core.constants import GLOSSARY_FILENAME

        console = self.console
        total = time.monotonic() - self.started
        failed = [r for r in self.records if not r.ok]
        aborted = bool(failed)
        console.info("")
        console.info(RULE)
        console.info(
            f" {'STOPPED' if aborted else 'DONE'} · {human_seconds(total)} total · "
            f"{len(self.records)}/{len(self._selected_phases())} phase(s) run"
        )
        console.info(RULE)
        for record in self.records:
            console.info(
                f"  {'✓' if record.ok else '✗'} {record.name:<11} "
                f"{human_seconds(record.seconds):>9}   {record.summary}"
            )
            if record.error:
                console.error(f"      {record.error}")
        console.info("")
        console.info("  artifacts")
        console.info(f"    chapters      {rel(self.chapters_dir)}/")
        console.info(f"    reduced       {rel(self.reduced_out)}/")
        console.info(f"    unique        {rel(self.unique_out)}/")
        console.info(f"    glossary      {rel(self.glossary_out)}/")
        console.info(f"    preprocessed  {rel(self.preprocessed_out)}/")
        console.info(f"    chunks        {rel(self.chunks_out)}/")
        console.info(f"    chunk gloss   {rel(self.chunk_glossary_out)}/")
        if console.log_path is not None:
            console.info(f"    log           {rel(console.log_path)}")
        for label, path in self._artifact_files():
            console.info(f"    {label:<16} {rel(path)}")
        console.info("")
        cleaned = any(record.name == "preprocess" and record.ok for record in self.records)
        chunked = any(
            record.name == "chunk"
            and record.ok
            and not (record.payload or {}).get("missing_dependency")
            and int((record.payload or {}).get("chunk_count") or 0) > 0
            for record in self.records
        )
        chunk_missing = any(
            record.name == "chunk" and bool((record.payload or {}).get("missing_dependency"))
            for record in self.records
        )
        chunk_glossaried = any(
            record.name == "chunk_glossary"
            and record.ok
            and not (record.payload or {}).get("missing_dependency")
            and int((record.payload or {}).get("chunk_count") or 0) > 0
            for record in self.records
        )
        glossaried = any(
            record.name == "dictionary"
            and record.ok
            and isinstance((record.payload or {}).get("glossary"), dict)
            and not (record.payload or {}).get("glossary", {}).get("dry_run")
            for record in self.records
        )
        reduced = any(
            record.name == "dictionary"
            and record.ok
            and not (record.payload or {}).get("missing_dependency")
            for record in self.records
        )
        tagger_missing = any(
            bool((record.payload or {}).get("missing_dependency")) for record in self.records
        )
        if aborted:
            console.info(
                "  stopped — fix the error above and re-run; the phases before it are done"
            )
        elif self.args.dry_run:
            console.info("  dry run — every plan was validated, no chapter file was written")
        else:
            if chunk_glossaried:
                console.info(
                    f"  next: {rel(self.chunk_glossary_out)}/ holds one glossary file per chunk — "
                    "send a chunk with its matched terms to keep the translation consistent"
                )
            elif chunked:
                console.info(
                    f"  next: {rel(self.chunks_out)}/ holds the translation-sized chunks — "
                    "concatenate them in filename order (or per chapter) to rebuild the text"
                )
            elif cleaned:
                console.info(f"  next: {rel(self.preprocessed_out)}/ holds the cleaned book")
            elif glossaried:
                console.info(
                    f"  next: {rel(self.glossary_out / GLOSSARY_FILENAME)} is the dictionary "
                    "the translation is bound to"
                )
            elif reduced:
                console.info(
                    f"  next: {rel(self.reduced_out)}/ holds the reduced chapters and "
                    f"{rel(self.unique_out)}/ the first mentions — "
                    "the LLM dictionary pass reads the first mentions"
                )
                if not self.glossary_enabled:
                    console.info(
                        "  next: build that dictionary with --start-at glossary --with-glossary "
                        "(it calls the model once per chapter)"
                    )
            elif chunk_missing:
                console.info(
                    "  next: install stage 5's chunker (pip install chonkie), then re-run "
                    "with --start-at chunk"
                )
            elif tagger_missing:
                console.info(
                    "  next: install stage 3's tagger (pip install -r "
                    "requirements-dictionary.txt), then re-run with --start-at dictionary"
                )
            elif has_chapters(self.chapters_dir):
                console.info(
                    f"  next: reduce them with --start-at dictionary  →  "
                    f"{rel(self.reduced_out)}/"
                )
            else:
                console.info("  next: re-run without --stop-after to reach the later stages")
            if tagger_missing and chunked:
                # The chunks are usable, but the dictionary half never ran: keep the
                # install command visible instead of hiding it behind the chunk hint.
                console.info(
                    "  note: install stage 3's tagger (pip install -r "
                    "requirements-dictionary.txt) and re-run --start-at dictionary"
                )
            if not self.preprocess_enabled:
                console.info(
                    "  stage 2 (preprocess) is switched off — --with-preprocess to run it"
                )

        if self.args.json:
            console.info("")
            console.info(json.dumps(self._summary_json(total), ensure_ascii=False, indent=2))
        if self.exit_code:
            return self.exit_code
        return 1 if aborted else 0

    def _artifact_files(self) -> list[tuple[str, str]]:
        """Manifest/trace files written by whichever phases ran, for the report."""
        from book_splitter.core.constants import (
            CHUNK_GLOSSARY_MANIFEST_FILENAME,
            CHUNK_GLOSSARY_TRACE_FILENAME,
            CHUNK_MANIFEST_FILENAME,
            CHUNK_TRACE_FILENAME,
            GLOSSARY_FILENAME,
            GLOSSARY_MANIFEST_FILENAME,
            GLOSSARY_MARKDOWN_FILENAME,
            GLOSSARY_TRACE_FILENAME,
            MANIFEST_FILENAME,
            PREPROCESS_MANIFEST_FILENAME,
            PREPROCESS_TRACE_FILENAME,
            REDUCED_MANIFEST_FILENAME,
            REDUCED_TRACE_FILENAME,
            TRACE_FILENAME,
        )

        candidates = [
            ("split trace", self.chapters_dir / TRACE_FILENAME),
            ("split manifest", self.chapters_dir / MANIFEST_FILENAME),
            ("stage-3 manifest", self.reduced_out / REDUCED_MANIFEST_FILENAME),
            ("stage-3 trace", self.reduced_out / REDUCED_TRACE_FILENAME),
            ("stage-4 glossary", self.glossary_out / GLOSSARY_FILENAME),
            ("stage-4 markdown", self.glossary_out / GLOSSARY_MARKDOWN_FILENAME),
            ("stage-4 manifest", self.glossary_out / GLOSSARY_MANIFEST_FILENAME),
            ("stage-4 trace", self.glossary_out / GLOSSARY_TRACE_FILENAME),
            ("stage-2 manifest", self.preprocessed_out / PREPROCESS_MANIFEST_FILENAME),
            ("stage-2 trace", self.preprocessed_out / PREPROCESS_TRACE_FILENAME),
            ("stage-5 manifest", self.chunks_out / CHUNK_MANIFEST_FILENAME),
            ("stage-5 trace", self.chunks_out / CHUNK_TRACE_FILENAME),
            ("stage-6 manifest", self.chunk_glossary_out / CHUNK_GLOSSARY_MANIFEST_FILENAME),
            ("stage-6 trace", self.chunk_glossary_out / CHUNK_GLOSSARY_TRACE_FILENAME),
        ]
        return [(label, path) for label, path in candidates if path.is_file()]

    def _summary_json(self, total: float) -> dict[str, Any]:
        return {
            "book": str(self.source),
            "output_dir": str(self.book_dir),
            "chapters_dir": str(self.chapters_dir),
            "reduced_dir": str(self.reduced_out),
            "unique_dir": str(self.unique_out),
            "glossary_dir": str(self.glossary_out),
            "preprocessed_dir": str(self.preprocessed_out),
            "chunks_dir": str(self.chunks_out),
            "chunk_glossary_dir": str(self.chunk_glossary_out),
            "dry_run": bool(self.args.dry_run),
            "start_at": self.args.start_at,
            "stop_after": self.args.stop_after,
            "ok": not any(not record.ok for record in self.records),
            "elapsed_seconds": round(total, 3),
            "log": str(self.console.log_path) if self.console.log_path else None,
            "phases": [
                {
                    "name": record.name,
                    "ok": record.ok,
                    "seconds": round(record.seconds, 3),
                    "summary": record.summary,
                    "error": record.error,
                    "result": record.payload,
                }
                for record in self.records
            ],
        }


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "book",
        nargs="?",
        default=None,
        help="source document — filename or path (default: the only book in data/input/)",
    )
    parser.add_argument(
        "--in-dir", default=None, help=f"folder holding the source (default: {default_input_dir()})"
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=f"root of the generated artifacts (default: {default_output_dir()})",
    )
    parser.add_argument(
        "--out",
        "-o",
        default=None,
        help="stage-1 chapters folder (default: <output-root>/<book>/chapters)",
    )
    parser.add_argument(
        "--map",
        default=None,
        metavar="FILE",
        help="stage 1: replay a chapter map from a JSON file instead of running the agent "
        "(no provider, no key, no network)",
    )
    parser.add_argument(
        "--preprocessed-out", default=None, help="stage-2 folder (default: a sibling of chapters/)"
    )
    parser.add_argument(
        "--reduced-out", default=None, help="stage-3 folder (default: a sibling of chapters/)"
    )
    parser.add_argument(
        "--unique-out",
        default=None,
        help="stage-3 first-mention folder (default: a sibling of the reduced folder, "
        "reduced-unique/)",
    )
    parser.add_argument(
        "--chunk-in",
        default=None,
        help="stage-5 input folder (default: chapters/); point it at preprocessed/ "
        "to chunk the cleaned text instead",
    )
    parser.add_argument(
        "--chunks-out",
        default=None,
        help="stage-5 output folder (default: <book>/chunks/)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_TOKENS,
        help=f"stage-5 maximum chunk length in tokens (default: {DEFAULT_CHUNK_TOKENS})",
    )
    parser.add_argument(
        "--chunk-tokenizer",
        default=DEFAULT_CHUNK_TOKENIZER,
        help="stage-5 tokenizer used to measure a chunk: a tiktoken encoding "
        f"(default: {DEFAULT_CHUNK_TOKENIZER}), 'word', 'character', 'byte', or a "
        "Hugging Face tokenizer name; falls back to a local approximation offline",
    )
    parser.add_argument(
        "--chunk-glossary-out",
        default=None,
        help="stage 6: where the per-chunk glossary files go "
        "(default: <book>/chunk-glossary)",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=DEFAULT_CHUNK_GLOSSARY_THRESHOLD,
        help="stage 6: minimum rapidfuzz score as a percentage "
        f"(default: {DEFAULT_CHUNK_GLOSSARY_THRESHOLD}); lower is more permissive",
    )
    parser.add_argument(
        "--match-scorer",
        choices=CHUNK_GLOSSARY_SCORERS,
        default=DEFAULT_CHUNK_GLOSSARY_SCORER,
        help=f"stage 6: rapidfuzz scorer (default: {DEFAULT_CHUNK_GLOSSARY_SCORER})",
    )
    parser.add_argument(
        "--match-candidates",
        type=int,
        default=DEFAULT_CHUNK_GLOSSARY_CANDIDATES,
        help="stage 6: glossary candidates retrieved per term "
        f"(default: {DEFAULT_CHUNK_GLOSSARY_CANDIDATES})",
    )
    parser.add_argument(
        "--min-term-chars",
        type=int,
        default=DEFAULT_CHUNK_GLOSSARY_MIN_TERM_CHARS,
        help="stage 6: drop extracted terms shorter than this "
        f"(default: {DEFAULT_CHUNK_GLOSSARY_MIN_TERM_CHARS})",
    )
    parser.add_argument(
        "--no-chunk-glossary",
        action="store_true",
        help="drop stage 6 (per-chunk NER + glossary retrieval) from this run",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate every plan; write no artifact except this log",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="stage 2: chapters at once (default 1)"
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=MAX_FEEDBACK_LOOPS,
        help="max agent turns per unit of work (stage 4: extraction attempts per chapter)",
    )
    parser.add_argument(
        "--front-matter-lines", type=int, default=DEFAULT_FRONT_MATTER_LINES, help="stage 1 budget"
    )
    parser.add_argument(
        "--min-chapters", type=int, default=DEFAULT_MIN_CHAPTERS, help="stage 1 floor"
    )
    parser.add_argument(
        "--no-front-matter", action="store_true", help="stage 1: do not write 00-front-matter.md"
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="stages 2/3/6: do not write their manifest files",
    )
    parser.add_argument(
        "--language",
        default="auto",
        help="stages 3/6: 'auto' detects the language by sampling chapters/chunks; otherwise "
        "an ISO code (sr, hr, sl, bg via CLASSLA or en via stanza)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help=f"stages 3/6: physical GPU for the tagger (default: GPU {DEFAULT_GPU_INDEX}, or "
        "$CUDA_VISIBLE_DEVICES when that is already set)",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="stages 3/6: run the tagger on the CPU"
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=DEFAULT_MIN_SENTENCE_WORDS,
        help=f"stage 3: drop sentences shorter than this (default: {DEFAULT_MIN_SENTENCE_WORDS})",
    )
    parser.add_argument(
        "--no-uppercase-filter",
        action="store_true",
        help="stage 3: keep sentences whose only capital letter is the first word",
    )
    parser.add_argument(
        "--keep-types",
        default=",".join(DEFAULT_KEEP_ENTITY_TYPES),
        help=f"stages 3/6: entity types that count "
        f"(default: {','.join(DEFAULT_KEEP_ENTITY_TYPES)})",
    )
    parser.add_argument(
        "--drop-temporal-only",
        action="store_true",
        help="stages 3/6: drop a sentence/term when every entity in it is a date or a time",
    )
    parser.add_argument(
        "--skip-if-present",
        action="store_true",
        help="reuse work already on disk: existing chapters for stage 1, existing "
        "reduced chapters for stage 3 (no model is loaded for the latter), and "
        "existing chunks for stage 5 (no tokenizer is loaded when all chapters match)",
    )
    parser.add_argument(
        "--no-save-reduced",
        action="store_true",
        help="stage 3: run the filters but write no reduced chapters",
    )
    parser.add_argument(
        "--no-save-unique",
        action="store_true",
        help="stage 3: write no first-mention chapters, only the reduced ones",
    )
    parser.add_argument(
        "--unique-only",
        action="store_true",
        help="stage 3: write only the first-mention chapters (implies --no-save-reduced)",
    )
    parser.add_argument(
        "--no-dictionary",
        action="store_true",
        help="drop stage 3 from this run (it is on by default)",
    )
    parser.add_argument(
        "--no-chunk",
        action="store_true",
        help="drop stage 5 (chunking) from this run (it is on by default)",
    )
    parser.add_argument(
        "--with-glossary",
        action="store_true",
        help="build the glossary (the default; kept so an older command line still works)",
    )
    parser.add_argument(
        "--no-glossary",
        action="store_true",
        help="stage 3: reduce only, do not build the glossary (no model call, no key needed)",
    )
    parser.add_argument(
        "--glossary-out",
        default=None,
        help="stage 4: where glossary.json / glossary.md go (default: <book>/glossary)",
    )
    parser.add_argument(
        "--source-language",
        default=DEFAULT_SOURCE_LANGUAGE,
        help=f"stage 4: the language the book is written in (default: {DEFAULT_SOURCE_LANGUAGE})",
    )
    parser.add_argument(
        "--target-language",
        default=DEFAULT_TARGET_LANGUAGE,
        help=f"stage 4: the language being translated into (default: {DEFAULT_TARGET_LANGUAGE})",
    )
    parser.add_argument(
        "--sentences",
        choices=("unique", "reduced"),
        default="unique",
        help="stage 4: mine the first mentions (default) or every entity-bearing sentence",
    )
    parser.add_argument(
        "--sentences-in",
        default=None,
        help="stage 4: read the sentences from this folder instead of deriving it",
    )
    parser.add_argument(
        "--max-context-entries",
        type=int,
        default=0,
        help="stage 4: cap the dictionary re-sent with each chapter (0 = all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stages 3/4/5/6: only the first N units (chapters, or chunks for stage 6) — a "
        "quick look, and how to price the glossary or a chunk run before the whole book",
    )
    parser.add_argument(
        "--toc-scan-lines",
        type=int,
        default=TOC_SCAN_LINES,
        help="recon: front matter lines to scan",
    )
    parser.add_argument(
        "--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES, help="recon: candidate cap"
    )
    parser.add_argument(
        "--examples", type=int, default=3, help="recon/debug: damage examples per shape"
    )

    parser.add_argument("--provider", choices=sorted(PROVIDERS), default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--recursion-limit", type=int, default=DEFAULT_RECURSION_LIMIT)

    parser.add_argument(
        "--start-at",
        choices=PHASES,
        default=PHASES[0],
        help="begin here, skipping earlier phases (default: recon)",
    )
    parser.add_argument(
        "--stop-after",
        choices=PHASES,
        default=PHASES[-1],
        help=f"stop after this phase (default: {PHASES[-1]})",
    )
    parser.add_argument(
        "--with-preprocess",
        action="store_true",
        help="run stage 2 even though it is switched off for now (see PREPROCESS_ENABLED)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="also print headers, files and damage examples"
    )
    parser.add_argument(
        "--log-file", default=None, help="where to write the log (default: <book>/pipeline.log)"
    )
    parser.add_argument("--no-log", action="store_true", help="do not write a log file")
    parser.add_argument("--json", action="store_true", help="append a machine-readable run summary")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if PHASES.index(args.start_at) > PHASES.index(args.stop_after):
        print(
            f"error: --start-at {args.start_at} comes after --stop-after {args.stop_after}",
            file=sys.stderr,
        )
        return 2
    if int(args.chunk_size) <= 0:
        print("error: --chunk-size must be greater than 0 tokens", file=sys.stderr)
        return 2
    try:
        pipeline = Pipeline(args)
    except (FileNotFoundError, ToolError, ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        return pipeline.run()
    finally:
        pipeline.close()


if __name__ == "__main__":
    raise SystemExit(main())
