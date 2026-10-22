#!/usr/bin/env python
"""The whole book pipeline in one command: convert → recon → split → scan →
preprocess → dictionary → chunk → chunk-glossary → matter (→ translate →
concatenate → postprocess → bind → epub).

`convert` (stage 0), `split`, `preprocess`, `dictionary`, `chunk`, `chunk-glossary`,
`matter`, `translate`, `concatenate`, `postprocess`, `bind` and `epub` are separate
stages behind their own CLIs (the dictionary stage is also the book's glossary
builder).
Stage 0 is skipped for a Markdown source and runs automatically for an EPUB, a
text-based PDF or a TXT: docling converts EPUB/PDF to `00-markdown/`, an
image-based PDF is rejected instead of OCR'd, and a TXT is copied under a `.md`
name. Pictures are extracted to `00-markdown/images/` by default (`--no-images`
switches that off) and the Markdown references them, so they survive translation
and binding into the final `10-translated-book.md` and into the EPUB 3 book
`11-translated-book.epub`. This script runs the stages
in the only order that works, from one place,
and narrates each
phase with the numbers that matter: how big the book is, which strategy won,
whether the chapters reassemble byte for byte, what damage was found, how many
sentences survived the dictionary funnel, how many entities were locked into the
glossary, what the LLM pass cost in tokens, how many glossary terms each chunk
mentions, which front/back matter sections were localized, and — when translation
is switched on — how many chunks were translated, what they cost, and how many
chapters were assembled from them.

    python scripts/run_pipeline.py                        # the only book in data/input/
    python scripts/run_pipeline.py BOOK.epub              # stage 0 converts it, then the usual run
    python scripts/run_pipeline.py BOOK.pdf --stop-after convert
                                                          # just the text-layer check + Markdown
    python scripts/run_pipeline.py BOOK.md --debug        # + headers, files, damage examples
    python scripts/run_pipeline.py --dry-run              # validate every plan, write no artifacts
    python scripts/run_pipeline.py --stop-after split     # stop once the chapters exist
    python scripts/run_pipeline.py --start-at dictionary  # stage 3 only, over existing chapters
    python scripts/run_pipeline.py --start-at chunk        # stage 5 only, over existing chapters
    python scripts/run_pipeline.py --text-in 01-chapters   # read the raw text again in stages 3-5
    python scripts/run_pipeline.py --start-at chunk_glossary
                                                          # stage 6 only, over existing chunks
    python scripts/run_pipeline.py --start-at matter       # stage 6b only, over existing chunks
    python scripts/run_pipeline.py --start-at matter --matter-limit 3
                                                          # localize three matter chunks
    python scripts/run_pipeline.py --start-at translate --with-translate
                                                          # stage 7 only, over existing chunks
    python scripts/run_pipeline.py --start-at translate --with-translate \
        --translate-provider lmstudio                     # stage 7 on the local LM Studio model
                                                          # (front/back matter and other stages
                                                          # stay on the shared provider)
    python scripts/run_pipeline.py --llm split.model=deepseek-flash \
        --llm translate.provider=lmstudio                 # one override per stage, no file edit
    python scripts/run_pipeline.py --llm-config /tmp/cheap.toml
                                                          # another per-stage config file
    python scripts/run_pipeline.py --start-at concatenate
                                                          # stage 8 only, over existing translations
    python scripts/run_pipeline.py --start-at postprocess --with-postprocess
                                                          # stage 9 only, over existing chapters
    python scripts/run_pipeline.py --start-at bind         # stage 10 only, over existing output
    python scripts/run_pipeline.py --start-at epub         # stage 11 only: -> 11-translated-book.epub
    python scripts/run_pipeline.py --no-epub               # write the Markdown book, no EPUB
    python scripts/run_pipeline.py --epub-cover cover.jpg  # pack the book with a cover
    python scripts/run_pipeline.py --with-translate --translate-limit 3
                                                          # translate three chunks, then stop
    python scripts/run_pipeline.py --first-chapters 5 --with-translate
                                                          # split in full; every later stage
                                                          # only the first 5 chapters
    python scripts/run_pipeline.py --first-chapters 10 --skip-if-present --with-translate
                                                          # widen the window: reuse 1–5,
                                                          # process only 6–10
    python scripts/run_pipeline.py --with-translate --translate-first-chapters 3
                                                          # translate/assemble only 3 chapters
    python scripts/run_pipeline.py --chunk-size 700        # smaller translation chunks
    python scripts/run_pipeline.py --match-threshold 90    # more permissive glossary retrieval
    python scripts/run_pipeline.py BOOK.md --map map.json --stop-after split
                                                          # replay a supplied map: no LLM at all

Stage 2 (``preprocess``) is **on by default** and runs right after the split's
damage scan: it repairs the converter's damage (page numbers, split headings,
figure captions, words cut by a page number) before any later stage reads a
chapter. The dictionary, the glossary and the chunker read ``02-preprocessed/``
by default whenever stage 2 covered the chapters in the window; ``--text-in``
overrides that folder and ``--no-preprocess`` keeps the raw ``01-chapters/``
text for every stage. It costs one model call per chapter, so ``--no-preprocess``
also drops the cleaning itself for a book whose chapters are already clean.

Stage 3 (``dictionary``) is on by default but needs an optional tagger
(``pip install -r requirements-dictionary.txt``). When that dependency is missing
the phase reports itself as skipped with the install command instead of failing
the run, so a split-only workflow keeps working. It reads the cleaned chapters
when stage 2 produced them, so page numbers and running headers never reach the
sentence funnel or the glossary prompt.

The ``dictionary`` phase has two halves. The first is deterministic — the sentence
funnel and the NER gate, no model. The second is the glossary: one paid model call
per chapter, each handed the dictionary the earlier chapters produced. The second
half is skipped by ``--no-glossary`` (the phase stays deterministic and keyless)
and by ``--dry-run`` (nothing to validate without asking the model, and a dry run
must not spend).

The chunk phase (stage 5, ``chunk``) cuts every chapter into translation-sized
chunks with chonkie, preferring paragraph over sentence over line over word
boundaries. Every chunk is an exact slice of the chapter it came from, so the
chunks concatenate back to the chapter (and to the book) byte for byte. It reads
the cleaned chapters too, so the page-number artefacts preprocess removed never
become chunks; ``--chunk-in`` points it somewhere else (for example back at raw
``01-chapters/``). The manifest records the chapter membership, the global order,
the character offsets and the boundary kind of every cut. ``--chunk-size`` is in
tokens, default 1000. A missing ``chonkie`` installation is reported as a skipped
phase, not a failure, so a split-only run keeps working.

The chunk-glossary phase (stage 6, ``chunk_glossary``) is the bridge to translation. It runs
NER over **every** chunk first, then fuzzy-matches each extracted term against the
glossary with rapidfuzz (95 % by default) and writes the terms a chunk mentions to
``06-chunk-glossary/<chunk>.json``. It needs both stage 4's ``glossary.json`` and
stage 5's chunks; without them it reports a skip/actionable error. ``--no-chunk-glossary``
drops it, and ``--no-chunk`` drops it too.

The matter phase is the separate agent that runs **before** translation. Front
and back matter are localized, not translated like body chunks: a book-level
LangGraph plan searches the web for an existing published translation of the
title, fixes the ``BleBel`` credit, lists the original edition's imprint/ISBN/CIP
material to omit, and decides the section titles; then one agent run per matter
chunk localizes it with that binding plan, without any dictionary. Translated
chunks land in ``07-translations/`` under the usual names, so assembly needs no new
rule, while the plan and every prompt are kept in ``06b-matter/`` and
``06b-matter-prompts/``. It runs with ``--with-translate`` (or ``--with-matter`` /
``--start-at matter``); ``--no-matter`` drops it and ``--include-matter`` makes
stage 7 translate the matter chunks the old way.

Stage 7 (``translate``) is the end product and is **opt-in**: ``--with-translate``
sends every **body** chunk, in reading order, to the model with the glossary
terms stage 6 attached, validates a Pydantic ``{"translation": ...}`` answer
(retrying a failed call), and writes ``07-translations/<chunk>.md``. Front/back
matter chunks are skipped by default because the matter stage already localized
them. The whole input of every call — system prompt, user prompt, messages,
glossary and source chunk — is kept in ``07-translation-prompts/`` for debugging.
English into Serbian by default; ``--translate-limit N`` translates a prefix, and
``--skip-if-present`` resumes a run that was interrupted. It is off by default
because it is one paid call per chunk; ``--no-translate`` is accepted for an
explicit command line.

Everything printed is also written to ``data/output/<book>/pipeline.log``
(``--log-file`` to move it, ``--no-log`` to skip it). The log always keeps the
debug level, even when the console does not.

Phases are independent of each other only in one direction: ``scan``,
``preprocess``, ``dictionary``, ``chunk``, ``chunk_glossary``, ``translate`` and
``concatenate`` read folders an earlier phase writes, so ``--start-at dictionary``
needs a previous run. A ``--dry-run`` split writes no chapters, and the later
phases are then reported as skipped instead of failing.

Stage 8 (``concatenate``) is the inverse of stage 5's cut, applied to the
translated chunks: it groups ``07-translations/<chunk>.md`` by the chapter stage 5's
manifest says each chunk belongs to, joins each group in chunk order, and writes
one file per chapter to ``08-translated-chapters/``. The source ``01-chapters/`` folder
is never touched. The phase is selected whenever translation is selected, and it
can be run on its own over an earlier translation with ``--start-at concatenate``.
A chapter whose chunks are only partly translated is reported as incomplete and
skipped, not written as a truncated chapter.

Stage 9 (``postprocess``) is **opt-in** with ``--with-postprocess`` (or
``--start-at postprocess``). It inventories all translated chapter titles,
chooses one naming policy, then walks the chapters one at a time: it reads only
the short opening and the heading records, sets/inserts/keeps the chapter title,
and keeps or demotes every other heading (a demotion removes the ``#`` markup,
never the text). It finishes by creating or updating the table of contents from
the final titles and writes everything to ``09-postprocessed-chapters/``.

Stage 10 (``bind``) is deterministic: it concatenates the postprocessed chapters,
in reading order, into one ``10-translated-book.md``. It is selected with stage 9,
or on its own with ``--start-at bind``.

Stage 11 (``epub``) is deterministic too: it packages the same chapter folder as
an EPUB 3 book, ``11-translated-book.epub`` — one XHTML document per chapter, a
navigation document built from the chapters' headings, the referenced pictures
embedded once, and the title/author/language stage 6b recorded. The printed table
of contents is not shipped (the reader's navigation replaces it). It is selected
with stage 10, or on its own with ``--start-at epub``, and ``--no-epub`` drops it.

``--first-chapters N`` is the one window that applies to every **body** stage
after splitting: the preprocessing, the dictionary reduction, the glossary, the
chunker, the per-chunk glossary, translation and assembly all see only the first
N **body** chapters (front/back matter does not count as one of the N). The matter
stage is not windowed: it always localizes the whole front/back matter set,
because those files are not chapters. Splitting itself always runs
over the whole book, because its boundaries are the coordinate system every later
stage reads. For the chunk-based stages the window is turned into the exact set of
chunk files belonging to those chapters through stage 5's manifest, so it never
stops in the middle of a chapter and never starts in the front matter.
``--translate-first-chapters N`` overrides the window for translation and
assembly alone. Combined with ``--skip-if-present`` (``--resume``), widening N
reuses the finished chapters and chunks and processes only the new ones.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
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
from book_splitter.conversion import (  # noqa: E402
    classify_source,
    is_markdown_source,
    run_conversion,
)
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
    DEFAULT_TRANSLATION_ATTEMPTS,
    DEFAULT_TRANSLATION_CONTEXT_CHARS,
    DEFAULT_TRANSLATION_MAX_TOKENS,
    MATTER_TRANSLATOR_NAME,
    MAX_FEEDBACK_LOOPS,
    TOC_SCAN_LINES,
    TRANSLATION_GLOSSARY_SCOPES,
    TRANSLATION_METHODS,
)
from book_splitter.core.errors import ConfigError, MissingDependencyError, ToolError  # noqa: E402
from book_splitter.core.settings import (  # noqa: E402
    LLM_CONFIG_FILENAME,
    PROVIDERS,
    STAGES,
    TRANSLATION_CHOICES,
    llm_config_path,
    load_dotenv_files,
    load_llm_config,
    normalize_stage,
    parse_llm_overrides,
    resolve_settings,
    resolve_translation_settings,
    stage_model_label,
)
from book_splitter.documents.chapters import (  # noqa: E402
    chapters_dir,
    chunk_glossary_dir,
    chunks_dir,
    discover_chapters,
    file_sha256,
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
from book_splitter.documents.io import load_lines, source_size_bytes  # noqa: E402
from book_splitter.documents.window import (  # noqa: E402
    chapter_files_for_window,
    chapter_window_limit_for_path,
    normalize_limit,
)
from book_splitter.paths import (  # noqa: E402
    book_markdown_dir,
    book_output_dir,
    book_slug,
    default_input_dir,
    default_output_dir,
    resolve_book_path,
    resolve_output_dir,
)
from book_splitter.preprocess.corpus import scan_corpus_damage  # noqa: E402

#: The phases, in the order they must run. ``convert`` is stage 0: it turns a
#: non-Markdown source (EPUB, text-based PDF, TXT) into the Markdown every later
#: phase reads, and is dropped from the list for a Markdown book. ``preprocess``
#: comes next, right after ``scan`` produced the damage profile, so the repairs
#: land before any text consumer runs. The glossary is not a phase of its own: it
#: is the second half of ``dictionary`` (reduce the chapters, then turn them into
#: the book's dictionary). ``chunk`` is the translation-sized cut: it does not
#: depend on the dictionary, but it reads the same chapter folder those phases
#: start from. ``chunk_glossary`` joins the two: it tags every chunk first, then
#: attaches the glossary entries the chunk mentions. ``matter`` is
#: the separate front/back matter localization agent: it is not dictionary-driven,
#: it runs before translation, and it writes the matter translations stage 8
#: already knows how to assemble. ``translate`` is the body end product: it sends
#: every body chunk with those entries to the model and writes the translated
#: chunk plus the whole prompt that produced it.
PHASES = (
    "convert",
    "recon",
    "split",
    "scan",
    "preprocess",
    "dictionary",
    "chunk",
    "chunk_glossary",
    "matter",
    "translate",
    "concatenate",
    "postprocess",
    "bind",
    "epub",
)

#: Phases that need the folder stage 1 writes.
NEEDS_CHAPTERS = (
    "scan",
    "dictionary",
    "preprocess",
    "chunk",
    "chunk_glossary",
    "matter",
    "translate",
    "concatenate",
    "postprocess",
    "bind",
    "epub",
)

# Stage 2 (preprocess) repairs the converter's artefacts before anything reads
# the chapters, so it is on by default. ``--no-preprocess`` drops the phase;
# ``--with-preprocess`` is still accepted so an older command line keeps working.
PREPROCESS_ENABLED = True

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

# The matter stage is the localization of front/back matter by its own agent.
# It is not dictionary-driven, so it runs after the chunk glossary (it needs the
# chunk files) and before translation. It is selected with translation, by
# ``--with-matter``, or by ``--start-at matter``; ``--no-matter`` drops it.
MATTER_ENABLED = True

# Stage 7 (translation) is the end product and the most expensive phase: one paid
# model call per chunk (183 calls for the sample English novel). It is therefore
# **opt-in**: a plain run stops after the chunk glossary, and ``--with-translate``
# runs the book's translation. ``--no-translate`` is accepted so an explicit
# command line keeps working, and ``--translate-limit N`` translates a prefix for
# a cheap test of the whole path.
TRANSLATE_ENABLED = False

# Stage 9 (postprocess) is the new heading/TOC pass. It is **opt-in** in the
# one-command pipeline, like translation: `--with-postprocess` runs it, and
# `--start-at postprocess` runs it alone over an earlier translated book. The
# standalone `postprocess` CLI command is always available.
POSTPROCESS_ENABLED = False

TITLES = {
    "convert": "stage 0 — convert the EPUB/PDF/TXT source to Markdown (no LLM, no OCR)",
    "recon": "what the book actually contains (no LLM, no writes)",
    "split": "stage 1 — the agent routes, the lines are copied verbatim",
    "scan": "stage 2 pre-flight — the damage profile (no LLM)",
    "dictionary": "stage 3 — reduce the chapters, then build the book's dictionary",
    "preprocess": "stage 2 — clean each chapter, then prove the edit",
    "chunk": "stage 5 — cut each chapter into reversible, sentence-safe chunks (no LLM)",
    "chunk_glossary": "stage 6 — tag every chunk, then attach the glossary terms it mentions",
    "matter": "stage 6b — localize the front/back matter with its own agent (no dictionary)",
    "translate": "stage 7 — translate every body chunk, one call at a time, and keep its prompt",
    "concatenate": "stage 8 — glue the translated chunks back into chapters (no LLM)",
    "postprocess": "stage 9 — make the translated headings and table of contents consistent",
    "bind": "stage 10 — concatenate the postprocessed chapters into one Markdown book",
    "epub": "stage 11 — package the chapters as an EPUB 3 book (no LLM)",
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
        # Stage 2 is on by default; ``--no-preprocess`` drops it for this run,
        # and ``--with-preprocess`` is still accepted as the older spelling.
        self.preprocess_enabled = bool(
            (PREPROCESS_ENABLED or getattr(args, "with_preprocess", False))
            and not getattr(args, "no_preprocess", False)
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
        # Stage 7 is opt-in and needs the chunks (and their glossary) to exist.
        self.translate_enabled = bool(
            (TRANSLATE_ENABLED or getattr(args, "with_translate", False))
            and not getattr(args, "no_translate", False)
            and self.chunk_enabled
        )
        self.translate_limit = (
            getattr(args, "translate_limit", None)
            if getattr(args, "translate_limit", None) is not None
            else getattr(args, "limit", None)
        )
        # The matter stage is part of the translation product: --with-translate
        # runs it first, and it can be run on its own with --with-matter or
        # --start-at matter. --no-translate drops translation but an explicit
        # matter run still stands.
        wants_matter = bool(
            MATTER_ENABLED
            and (
                self.translate_enabled
                or getattr(args, "with_matter", False)
                or self.args.start_at == "matter"
            )
        )
        self.matter_enabled = bool(
            wants_matter and not getattr(args, "no_matter", False) and self.chunk_enabled
        )
        # The chapter window. Splitting always runs over the whole book; every
        # stage after it can be narrowed to the first N chapters. Stage 7 has its
        # own override (``--translate-first-chapters``) because translation is the
        # paid step and may be priced over a smaller prefix than the dictionary.
        self.first_chapters = normalize_limit(getattr(args, "first_chapters", None))
        self.translate_first_chapters = (
            normalize_limit(getattr(args, "translate_first_chapters", None))
            or self.first_chapters
        )
        self.chapter_limit = None
        # Stage 8 is the tail of the translation output. It is selected when
        # translation is selected, or when the user explicitly starts at it (so a
        # previously translated book can be assembled without re-translating).
        self.concat_enabled = bool(
            not getattr(args, "no_concat", False)
            and (self.translate_enabled or self.args.start_at == "concatenate")
        )
        # Stage 9 runs after the chapters are assembled. It is selected when
        # translation/assembly is selected, or when the user starts at it (over an
        # earlier translated book).
        self.postprocess_enabled = bool(
            not getattr(args, "no_postprocess", False)
            and (
                POSTPROCESS_ENABLED
                or getattr(args, "with_postprocess", False)
                or self.args.start_at == "postprocess"
            )
        )
        # Stage 10 is the deterministic single-file concatenation of stage 9.
        self.bind_enabled = bool(
            not getattr(args, "no_bind", False)
            and (
                self.concat_enabled
                or self.postprocess_enabled
                or self.args.start_at == "bind"
            )
        )
        # Stage 11 packages the same chapters as an EPUB: it follows the bind, and
        # `--start-at epub` runs it alone over an existing chapter folder.
        self.epub_enabled = bool(
            not getattr(args, "no_epub", False)
            and (self.bind_enabled or self.args.start_at == "epub")
        )
        self.records: list[PhaseRecord] = []
        self.exit_code = 0
        self.started = time.monotonic()
        self._settings = None
        self._translation_settings = None
        self._stage_settings: dict[str, Any] = {}
        self._llm_config = None
        # ``--llm <stage>.<field>=VALUE``: parsed once, applied per phase.
        self._llm_overrides = parse_llm_overrides(getattr(args, "llm", None))
        self._progress_lock = threading.Lock()
        self._bar_stop = threading.Event()
        self._bar_thread: threading.Thread | None = None
        self._bar_state: dict[str, Any] = {"active": False, "total": 0, "done": 0}
        self._last_tick = self.started
        self._count_width = 1
        self._name_width = 24

        load_dotenv_files()  # so the banner's model label matches the real resolution
        self.source = resolve_book_path(args.book, input_dir=args.in_dir)
        # Stage 0 converts a non-Markdown source before any phase reads its lines.
        # ``self.source`` is replaced by the converted Markdown once it has run;
        # the original path is kept for the banner and the JSON summary.
        self.original_source = self.source
        # Validation up front: an unsupported format is one clear error from
        # ``main`` instead of a traceback inside the banner.
        self.source_kind = classify_source(self.source)
        self.convert_needed = self.source_kind != "markdown"
        self.markdown_out = book_markdown_dir(self.source, output_root=args.output_root)
        self._convert_temp_dir: Path | None = None
        self.book_dir = book_output_dir(self.source, output_root=args.output_root)
        self.chapters_dir = resolve_output_dir(self.source, args.out, output_root=args.output_root)
        self._refresh_chapter_limit()
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
        # The chapter text stages 3-5 read. Stage 2's output is the default once
        # it covers the window; ``--text-in`` overrides it, and ``--chunk-in``
        # overrides it for the chunker alone. ``--no-preprocess`` forces raw
        # ``01-chapters/``. Both are kept as ``None`` when not given so the choice
        # is re-evaluated after the preprocess phase has written its files.
        self.text_in = (
            Path(args.text_in).expanduser()
            if getattr(args, "text_in", None)
            else None
        )
        self.chunk_in = (
            Path(args.chunk_in).expanduser()
            if getattr(args, "chunk_in", None)
            else None
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
        self.translations_out = (
            Path(args.translations_out).expanduser()
            if getattr(args, "translations_out", None)
            else translations_dir(self.chunks_out)
        )
        self.translation_prompts_out = (
            Path(args.translation_prompts_out).expanduser()
            if getattr(args, "translation_prompts_out", None)
            else translation_prompts_dir(self.chunks_out)
        )
        self.matter_out = (
            Path(args.matter_out).expanduser()
            if getattr(args, "matter_out", None)
            else matter_dir(self.chunks_out)
        )
        self.matter_prompts_out = (
            Path(args.matter_prompts_out).expanduser()
            if getattr(args, "matter_prompts_out", None)
            else matter_prompts_dir(self.chunks_out)
        )
        self.translated_chapters_out = (
            Path(args.translated_chapters_out).expanduser()
            if getattr(args, "translated_chapters_out", None)
            else translated_chapters_dir(self.chunks_out)
        )
        self.postprocess_out = (
            Path(args.postprocess_out).expanduser()
            if getattr(args, "postprocess_out", None)
            else postprocessed_chapters_dir(self.translated_chapters_out)
        )
        self.translated_book_out = (
            Path(args.book_out).expanduser()
            if getattr(args, "book_out", None)
            else translated_book_path(self.postprocess_out)
        )
        self.epub_out = (
            Path(args.epub_out).expanduser()
            if getattr(args, "epub_out", None)
            else translated_epub_path(self.postprocess_out)
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
    def llm_config(self):
        """The per-stage model file, read once and only when a phase needs it."""
        if self._llm_config is None:
            self._llm_config = load_llm_config(self.args.llm_config)
        return self._llm_config

    def _shared_model_args(self) -> dict[str, Any]:
        """The flags shared by every stage.

        The resolver applies them only to a stage whose provider matches the one
        they were written for, so ``--model`` never leaks a cloud model id into a
        stage that a config section pinned to a local server.
        """
        return {
            "provider": self.args.provider,
            "model": self.args.model,
            "base_url": self.args.base_url,
            "api_key": self.args.api_key,
            "temperature": self.args.temperature,
        }

    @property
    def settings(self):
        """Stage 1's model connection (the shared default stage)."""
        return self.settings_for("split")

    def settings_for(self, stage: str):
        """One stage's connection, resolved lazily, cached, and keyless until needed.

        ``stage`` is one of :data:`STAGES` (``split``, ``preprocess``,
        ``dictionary``, ``matter``, ``translate``, ``postprocess``). Each stage
        reads its own ``[llm.<stage>]`` section from ``book_splitter.toml``.
        """
        canonical = normalize_stage(stage)
        if canonical is None:
            raise ConfigError(
                f"unknown pipeline stage {stage!r}; choose one of {', '.join(STAGES)}"
            )
        if canonical == "translate":
            return self.translation_settings
        if canonical == "split" and self._settings is not None:
            return self._settings
        if canonical in self._stage_settings:
            return self._stage_settings[canonical]
        resolved = resolve_settings(
            stage=canonical,
            shared=self._shared_model_args(),
            config=self.llm_config,
            overrides=self._llm_overrides.get(canonical),
        )
        if canonical == "split":
            self._settings = resolved
        self._stage_settings[canonical] = resolved
        return resolved

    @property
    def translation_settings(self):
        """Stage 7's model: DeepSeek, or the local LM Studio server.

        Its ``[llm.translate]`` section is independent of every other stage, so
        switching translation to LM Studio never moves the split, dictionary,
        matter or postprocess stages onto the local model. Resolution stays lazy
        so a fully resumed translation needs no key and a run without translation
        never touches it.
        """
        if self._translation_settings is None:
            self._translation_settings = resolve_translation_settings(
                provider=self.args.translate_provider,
                model=self.args.translate_model,
                base_url=self.args.translate_base_url,
                api_key=self.args.translate_api_key,
                temperature=self.args.translate_temperature,
                shared=self._shared_model_args(),
                config=self.llm_config,
                overrides=self._llm_overrides.get("translate"),
            )
        return self._translation_settings

    def _refresh_chapter_limit(self) -> None:
        """Turn the body-chapter window into the file count each stage takes.

        Called once at construction (for existing chapters) and again after the
        split phase, because a fresh run does not know whether the split will
        write ``00-front-matter.md`` until the phase has run. The effective limit
        deliberately includes any front matter *before* chapter 1 while not
        counting it as one of the N body chapters.
        """
        self.chapter_limit = chapter_window_limit_for_path(
            self.chapters_dir,
            self.first_chapters,
            getattr(self.args, "limit", None),
        )

    def _window_text(self) -> str | None:
        """Human label for the active window; the file count stays internal."""
        if self.first_chapters is not None:
            return f"first {self.first_chapters} body chapter(s)"
        if self.chapter_limit is not None:
            return f"first {self.chapter_limit} chapter(s)"
        return None

    def _cleaned_window(self) -> Path | None:
        """Stage 2's folder when it covers exactly the chapters this run reads.

        ``--first-chapters`` makes stage 2 clean only the window, so
        ``02-preprocessed/`` is a prefix of ``01-chapters/``, not a copy. It is
        usable only when every chapter in the window has a cleaned file, the
        stage-2 manifest was written for the current chapter set (same source
        line counts) and neither the source chapter nor the cleaned file changed
        since it was verified. Anything else falls back to the raw chapters
        rather than silently cleaning some chapters and not others.
        """
        folder = self.preprocessed_out
        if not folder.is_dir() or not has_chapter_files(folder):
            return None
        manifest = load_split_manifest(folder)
        if not manifest or not manifest.get("ok") or manifest.get("dry_run"):
            return None
        entries = {
            str(entry.get("filename")): entry
            for entry in manifest.get("files") or []
            if isinstance(entry, dict) and entry.get("filename")
        }
        if not entries:
            return None
        try:
            chapters = discover_chapters(self.chapters_dir)
        except (ToolError, OSError):
            return None
        window = chapters[: self.chapter_limit] if self.chapter_limit else chapters
        if not window:
            return None
        for chapter in window:
            entry = entries.get(chapter.filename)
            if entry is None or not (folder / chapter.filename).is_file():
                return None
            recorded_lines = entry.get("source_lines")
            if recorded_lines is not None and int(recorded_lines) != chapter.line_count:
                return None
            recorded_source = str(entry.get("source_sha256") or "")
            if recorded_source and chapter.sha256 and recorded_source != chapter.sha256:
                return None
            recorded_clean = str(entry.get("sha256") or "")
            if recorded_clean and file_sha256(folder / chapter.filename) != recorded_clean:
                return None
        return folder

    def _text_choice(self) -> tuple[Path, str]:
        """The folder stages 3-5 read, plus why it was chosen."""
        if self.text_in is not None:
            return chapters_dir(self.text_in), "explicit --text-in"
        if not self.preprocess_enabled:
            return self.chapters_dir, "raw chapters (--no-preprocess)"
        cleaned = self._cleaned_window()
        if cleaned is not None:
            return cleaned, "cleaned by stage 2"
        if has_chapter_files(self.preprocessed_out):
            return self.chapters_dir, "cleaned text does not cover this window"
        return self.chapters_dir, "raw chapters"

    def text_input(self) -> Path:
        """The chapter folder stages 3-5 read: the cleaned text when usable."""
        return self._text_choice()[0]

    def chunk_input(self) -> Path:
        """Stage 5's input: ``--chunk-in`` wins, else the resolved text folder."""
        if self.chunk_in is not None:
            return chapters_dir(self.chunk_in)
        return self.text_input()

    def _kv_text_input(self, console: Any, input_dir: Path) -> None:
        """Report the text folder and, when relevant, why it is not the cleaned one."""
        label = rel(input_dir)
        if input_dir == self.preprocessed_out:
            label += "  (cleaned by stage 2)"
        elif (
            self.chunk_in is None
            and self.text_in is None
            and self.preprocess_enabled
            and has_chapter_files(self.preprocessed_out)
        ):
            label += "  (cleaned text does not cover this window)"
        console.kv("input", label)

    def _selected_phases(self) -> list[str]:
        start = PHASES.index(self.args.start_at)
        stop = PHASES.index(self.args.stop_after)
        phases = list(PHASES[start : stop + 1])
        if not self.convert_needed:
            # A Markdown source needs no stage 0; dropping it keeps the phase
            # numbering of an existing Markdown run unchanged.
            phases = [name for name in phases if name != "convert"]
        if not self.preprocess_enabled:
            # Disabled: the phase is dropped, not reordered, so the remaining
            # phases and the order they run in are unchanged.
            phases = [name for name in phases if name != "preprocess"]
        if not self.dictionary_enabled:
            phases = [name for name in phases if name != "dictionary"]
        if not self.chunk_enabled:
            phases = [name for name in phases if name != "chunk"]
        if not self.chunk_glossary_enabled:
            phases = [name for name in phases if name != "chunk_glossary"]
        if not self.matter_enabled:
            phases = [name for name in phases if name != "matter"]
        if not self.translate_enabled:
            phases = [name for name in phases if name != "translate"]
        if not self.concat_enabled:
            phases = [name for name in phases if name != "concatenate"]
        if not self.postprocess_enabled:
            phases = [name for name in phases if name != "postprocess"]
        if not self.bind_enabled:
            phases = [name for name in phases if name != "bind"]
        if not self.epub_enabled:
            phases = [name for name in phases if name != "epub"]
        return phases

    def _switched_off_phases(self) -> list[str]:
        """Phases the user asked for that a temporary switch dropped."""
        start = PHASES.index(self.args.start_at)
        stop = PHASES.index(self.args.stop_after)
        candidates = list(PHASES[start : stop + 1])
        if not self.convert_needed:
            # Not a switched-off phase: stage 0 simply has nothing to do here.
            candidates = [name for name in candidates if name != "convert"]
        return [name for name in candidates if name not in self._selected_phases()]

    def close(self) -> None:
        self.console.close()
        if self._convert_temp_dir is not None:
            shutil.rmtree(self._convert_temp_dir, ignore_errors=True)
            self._convert_temp_dir = None

    # -- the run ----------------------------------------------------------- #
    def run(self) -> int:
        phases = self._selected_phases()
        self._banner(phases)
        for name in self._switched_off_phases():
            if name == "concatenate" and not self.translate_enabled:
                self.console.info(
                    "  · concatenate: switched off — add --with-translate to assemble "
                    "the translation, or --start-at concatenate"
                )
                continue
            flag = {
                "preprocess": "--no-preprocess",
                "dictionary": "--with-dictionary",
                "chunk": "--no-chunk",
                "chunk_glossary": "--no-chunk-glossary",
                "matter": "--with-matter / --no-matter",
                "translate": "--with-translate",
                "concatenate": "--no-concat",
                "postprocess": "--with-postprocess",
                "bind": "--no-bind",
                "epub": "--no-epub",
            }.get(name, "--with-" + name)
            self.console.info(f"  · {name}: switched off — skipped (see {flag})")
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

            if name == "matter" and not has_chunk_files(self.chunks_out):
                record.ok = True
                record.summary = "skipped — no chunks on disk to localize"
                self.console.info(
                    "  · skipped: no chunk files — run the chunk phase (or a real run) first"
                )
                continue

            if name == "translate":
                from book_splitter.translation.runner import has_chunk_glossary_files

                if not has_chunk_files(self.chunks_out):
                    record.ok = True
                    record.summary = "skipped — no chunks on disk to translate"
                    self.console.info(
                        "  · skipped: no chunk files — run the chunk phase first"
                    )
                    continue
                if not has_chunk_glossary_files(self.chunk_glossary_out):
                    record.ok = True
                    record.summary = "skipped — no per-chunk glossary to send"
                    self.console.info(
                        "  · skipped: no per-chunk glossary — run the chunk_glossary phase first"
                    )
                    continue

            if name == "concatenate" and not has_translation_files(self.translations_out):
                record.ok = True
                record.summary = "skipped — no translated chunks to assemble"
                self.console.info(
                    "  · skipped: no translated chunks — run the translate phase first "
                    "(or --start-at concatenate after a translated run)"
                )
                continue

            if name == "postprocess" and not has_translated_chapter_files(
                self.translated_chapters_out
            ):
                record.ok = True
                record.summary = "skipped — no translated chapters to postprocess"
                self.console.info(
                    "  · skipped: no translated chapters — run the translate/concatenate "
                    "phases first (or --start-at postprocess after a translated run)"
                )
                continue

            if name == "bind" and not has_postprocessed_chapter_files(self.postprocess_out):
                record.ok = True
                record.summary = "skipped — no postprocessed chapters to bind"
                self.console.info(
                    "  · skipped: no postprocessed chapters — run the postprocess phase first"
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
            if name == "split":
                # The split may have just written the front matter; recompute the
                # chapter window now that the real file list exists.
                self._refresh_chapter_limit()
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
            size = source_size_bytes(self.source)
            if is_markdown_source(self.source):
                lines = len(load_lines(self.source))
                console.kv("book", f"{rel(self.source)} — {human_bytes(size)}, {lines:,} lines")
            else:
                console.kv(
                    "book",
                    f"{rel(self.source)} — {human_bytes(size)}, "
                    f"{self.source_kind} (stage 0 converts it to Markdown)",
                )
        except OSError as exc:  # the path was resolved, so this is unusual
            console.kv("book", f"{rel(self.source)} — unreadable: {exc}")
        console.kv("output", rel(self.book_dir))
        model_config = llm_config_path(self.args.llm_config)
        console.kv(
            "llm config",
            rel(model_config)
            if model_config
            else f"built-in defaults (no {LLM_CONFIG_FILENAME})",
        )
        if self.convert_needed:
            console.kv("stage 0", rel(self.markdown_out) + "/  (converted source)")
        console.kv("chapters", rel(self.chapters_dir))
        console.kv("reduced", rel(self.reduced_out))
        console.kv("unique", rel(self.unique_out))
        console.kv("preprocessed", rel(self.preprocessed_out))
        console.kv("chunks", rel(self.chunks_out))
        console.kv("chunk gloss", rel(self.chunk_glossary_out))
        console.kv("translations", rel(self.translations_out))
        console.kv("prompts", rel(self.translation_prompts_out))
        console.kv("matter", rel(self.matter_out))
        console.kv("matter prompts", rel(self.matter_prompts_out))
        console.kv("translated chapters", rel(self.translated_chapters_out))
        console.kv("postprocessed", rel(self.postprocess_out))
        console.kv("book", rel(self.translated_book_out))
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
        if not self.translate_enabled:
            console.kv(
                "stage 7",
                "translation switched off — add --with-translate to translate the chunks",
            )
        else:
            console.kv(
                "translate",
                f"{self.args.source_language} → {self.args.target_language} · "
                f"glossary {self.args.translate_glossary_scope} · "
                f"limit {self.translate_limit if self.translate_limit else 'all chunks'}",
            )
        if not self.matter_enabled:
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
        if self.first_chapters is not None:
            console.kv(
                "chapter window",
                f"first {self.first_chapters} chapter(s) for every stage after split "
                "(split itself is always the whole book)",
            )
        if (
            self.translate_first_chapters is not None
            and self.translate_first_chapters != self.first_chapters
        ):
            console.kv(
                "translation window",
                f"first {self.translate_first_chapters} chapter(s)",
            )
        if not self.concat_enabled:
            console.kv(
                "stage 8",
                "assembly switched off — add --with-translate or --start-at concatenate",
            )
        else:
            console.kv("stage 8", rel(self.translated_chapters_out))
        if not self.postprocess_enabled:
            console.kv(
                "stage 9",
                "postprocess switched off — add --with-postprocess or --start-at postprocess",
            )
        else:
            console.kv("stage 9", rel(self.postprocess_out))
        if not self.bind_enabled:
            console.kv("stage 10", "bind switched off — remove --no-bind to write one book file")
        else:
            console.kv("stage 10", rel(self.translated_book_out))
        if not self.epub_enabled:
            console.kv("stage 11", "epub switched off — remove --no-epub to write one")
        else:
            console.kv("stage 11", rel(self.epub_out))
        console.kv("phases", " → ".join(phases) or "(none — every selected phase is switched off)")
        if not self.preprocess_enabled:
            console.kv("stage 2", "preprocess switched off — remove --no-preprocess to run it")
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

    # -- phase 0: convert -------------------------------------------------- #
    def _ensure_markdown_source(self) -> Path:
        """Run stage 0 on demand for a phase started past ``convert``.

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
                    tempfile.mkdtemp(prefix="book-splitter-convert-")
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
            "stage 0",
            f"{result.source_kind} -> {rel(result.markdown_path)} ({result.method})",
        )
        return self.source

    def phase_convert(self) -> tuple[str, dict[str, Any]]:
        """Stage 0: EPUB/PDF/TXT -> the Markdown every later phase reads."""
        console = self.console
        target_dir = self.markdown_out
        if self.args.dry_run:
            if self._convert_temp_dir is None:
                self._convert_temp_dir = Path(
                    tempfile.mkdtemp(prefix="book-splitter-convert-")
                )
            target_dir = self._convert_temp_dir
        result = run_conversion(
            self.source,
            target_dir,
            force=self.args.force_convert,
            images=getattr(self.args, "images", True),
        )
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
        from book_splitter.agent.tools import scan_markdown_headers

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
        from book_splitter.agent import (
            result_from_existing_chapters,
            run_split,
            run_split_from_map,
        )

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
        of failing the run, the way stage 2 has its off switch. It reads the
        cleaned chapters whenever stage 2 covers the window, so page numbers and
        running headers never reach the sentence funnel or the glossary.
        """
        from book_splitter.dictionary import run_dictionary

        args, console = self.args, self.console
        save_unique = not args.no_save_unique
        save_reduced = not args.no_save_reduced and not args.unique_only
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
            result = run_dictionary(
                str(input_dir),
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
                limit=self.chapter_limit,
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
            result = run_glossary(
                str(chapter_source),
                str(self.glossary_out),
                sentences_dir=str(sentences_path),
                sentences=args.sentences,
                source_language=args.source_language,
                target_language=args.target_language,
                settings_factory=lambda: self.settings_for("dictionary"),
                max_attempts=args.max_attempts,
                max_context_entries=args.max_context_entries,
                include_front_matter=getattr(args, "include_front_matter", False),
                include_back_matter=getattr(args, "include_back_matter", False),
                skip_if_present=args.skip_if_present,
                limit=self.chapter_limit,
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
        preprocess_settings = self.settings_for("preprocess")
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

        result = run_preprocess(
            str(self.chapters_dir),
            str(self.preprocessed_out),
            settings=preprocess_settings,
            max_attempts=args.max_attempts,
            dry_run=args.dry_run,
            recursion_limit=args.recursion_limit,
            workers=args.workers,
            limit=self.chapter_limit,
            skip_if_present=args.skip_if_present,
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
        from book_splitter.chunking import run_chunking

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
            result = run_chunking(
                str(input_dir),
                str(self.chunks_out),
                chunk_size=args.chunk_size,
                tokenizer=args.chunk_tokenizer,
                skip_if_present=args.skip_if_present,
                dry_run=args.dry_run,
                limit=self.chapter_limit,
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
                chapter_files=chapter_files,
                skip_if_present=args.skip_if_present,
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

    # -- phase 7b: front/back matter localization -------------------------- #
    def phase_matter(self) -> tuple[str, dict[str, Any]]:
        """Stage 6b: localize front/back matter with its own agent, before translation.

        The stage is intentionally independent of the dictionary. It first asks
        one book-level agent for a localization plan (published title if one can
        be found, the BleBel credit, the original-edition material to omit, the
        section titles), then reads every matter chunk through the same
        agent-with-tools loop with that plan binding. The translated chunks are
        written to ``07-translations/`` under the chunk names stage 8 expects, so
        assembly is unchanged.
        """
        from book_splitter.matter_agent import run_matter

        args, console = self.args, self.console
        if args.dry_run:
            console.info("  · matter localization skipped: a --dry-run does not spend tokens")
            return "skipped (dry run)", {"dry_run": True}
        if not has_chunk_files(self.chunks_out):
            raise PhaseFailed(
                f"no chunk files in {rel(self.chunks_out)} — run chunk first",
                summary="no chunks to localize",
            )

        # Matter keeps stage 1's folder, not the cleaned one: it creates the
        # synthetic front/back matter files and appends them to stage 1's
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

        result = run_matter(
            str(matter_chapters),
            str(self.chunks_out),
            str(self.translations_out),
            prompts_dir=str(self.matter_prompts_out),
            matter_path=str(self.matter_out),
            source_language=args.source_language,
            target_language=args.target_language,
            settings_factory=lambda: self.settings_for("matter"),
            max_attempts=args.max_attempts,
            recursion_limit=args.recursion_limit,
            resume=args.skip_if_present,
            limit=getattr(args, "matter_limit", None),
            write_manifest=not args.no_manifest,
            progress=None if getattr(args, "quiet", False) else self._on_matter_chunk,
        )

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
            f"  [{done:>3}/{total}] {flag} {str(chunk.get('filename') or '?'):<46} "
            f"{str(chunk.get('section_kind') or '?'):<5} "
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
        from book_splitter.core.constants import GLOSSARY_FILENAME
        from book_splitter.translation import run_translation
        from book_splitter.translation.runner import has_chunk_glossary_files

        args, console = self.args, self.console
        if args.dry_run:
            console.info("  · translation skipped: a --dry-run does not spend tokens")
            return "skipped (dry run)", {"dry_run": True}
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
            "localized by stage 6b and skipped here"
            if not args.include_matter
            else "also translated here (--include-matter)",
        )

        try:
            result = run_translation(
                str(self.chunks_out),
                str(self.translations_out),
                prompts_dir=str(self.translation_prompts_out),
                chunk_glossary_path=str(self.chunk_glossary_out),
                glossary_path=str(glossary_path),
                glossary_scope=args.translate_glossary_scope,
                source_language=args.source_language,
                target_language=args.target_language,
                settings_factory=lambda: self.translation_settings,
                max_attempts=args.translate_attempts,
                method=args.translate_method,
                max_tokens=args.translate_max_tokens or None,
                context_chars=args.translate_context_chars,
                resume=args.skip_if_present,
                limit=self.translate_limit,
                chapter_files=chapter_files,
                include_matter=bool(getattr(args, "include_matter", False)),
                dry_run=args.dry_run,
                write_manifest=not args.no_manifest,
                progress=None
                if getattr(args, "quiet", False)
                else self._on_translation_chunk,
            )
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
                "no body chunks — front/back matter is localized by stage 6b",
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
        from book_splitter.assembly import run_assembly

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

        result = run_assembly(
            str(self.chunks_out),
            str(self.translations_out),
            str(self.translated_chapters_out),
            first_chapters=self.translate_first_chapters,
            skip_if_present=args.skip_if_present,
            dry_run=args.dry_run,
            write_manifest=not args.no_manifest and not args.dry_run,
            progress=None
            if getattr(args, "quiet", False)
            else self._on_assembly_chapter,
        )

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
        from book_splitter.postprocess_agent import run_postprocess

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

        result = run_postprocess(
            str(self.translated_chapters_out),
            str(self.postprocess_out),
            settings=self.settings_for("postprocess"),
            max_attempts=args.max_attempts,
            dry_run=args.dry_run,
            recursion_limit=args.recursion_limit,
            limit=self.translate_first_chapters,
            skip_if_present=args.skip_if_present,
            write_manifest=not args.no_manifest and not args.dry_run,
            progress=None if getattr(args, "quiet", False) else self._on_postprocess_chapter,
        )
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
            f"  [{done:>3}/{total}] {flag} {str(chapter.get('filename') or '?'):<46} "
            f"{str(chapter.get('title_action') or '—'):<8} "
            f"{int(chapter.get('heading_decisions') or 0):>3} heading decision(s) · "
            f"{int(chapter.get('output_lines') or 0):>5} lines"
        )

    # -- phase 11: bind ---------------------------------------------------- #
    def phase_bind(self) -> tuple[str, dict[str, Any]]:
        """Stage 10: concatenate the postprocessed chapters into one book."""
        from book_splitter.binding import run_bind

        args, console = self.args, self.console
        from book_splitter.documents.chapters import has_translated_chapter_files

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
        result = run_bind(
            str(source),
            str(self.translated_book_out),
            dry_run=args.dry_run,
            write_manifest=not args.no_manifest and not args.dry_run,
        )
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
        from book_splitter.epub import run_epub

        args, console = self.args, self.console
        from book_splitter.documents.chapters import has_translated_chapter_files

        if has_postprocessed_chapter_files(self.postprocess_out):
            source = self.postprocess_out
        elif has_translated_chapter_files(self.translated_chapters_out):
            # Same fallback as stage 10: an EPUB is still worth having when the
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
            result = run_epub(
                str(source),
                str(self.epub_out),
                cover=getattr(args, "epub_cover", None),
                title=getattr(args, "epub_title", None),
                author=getattr(args, "epub_author", None),
                language=getattr(args, "epub_language", None),
                dry_run=args.dry_run,
                write_manifest=not args.no_manifest and not args.dry_run,
            )
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
            f"  [{done:>3}/{total}] {flag} {str(chapter.get('filename') or '?'):<46} "
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
            f"  [{done:>3}/{total}] {flag} {str(chunk.get('filename') or '?'):<46} "
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
        if self.convert_needed and not self.args.dry_run:
            console.info(f"    markdown      {rel(self.markdown_out)}/")
        console.info(f"    chapters      {rel(self.chapters_dir)}/")
        console.info(f"    reduced       {rel(self.reduced_out)}/")
        console.info(f"    unique        {rel(self.unique_out)}/")
        console.info(f"    glossary      {rel(self.glossary_out)}/")
        console.info(f"    preprocessed  {rel(self.preprocessed_out)}/")
        console.info(f"    chunks        {rel(self.chunks_out)}/")
        console.info(f"    chunk gloss   {rel(self.chunk_glossary_out)}/")
        console.info(f"    translations  {rel(self.translations_out)}/")
        console.info(f"    prompts       {rel(self.translation_prompts_out)}/")
        console.info(f"    matter        {rel(self.matter_out)}/")
        console.info(f"    matter prompts {rel(self.matter_prompts_out)}/")
        console.info(f"    translated ch {rel(self.translated_chapters_out)}/")
        console.info(f"    postprocessed {rel(self.postprocess_out)}/")
        console.info(f"    book          {rel(self.translated_book_out)}")
        if self.epub_enabled:
            console.info(f"    epub          {rel(self.epub_out)}")
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
        translated = any(
            record.name == "translate"
            and record.ok
            and int((record.payload or {}).get("chunk_count") or 0) > 0
            for record in self.records
        )
        mattered = any(
            record.name == "matter"
            and record.ok
            and int((record.payload or {}).get("chunk_count") or 0) > 0
            for record in self.records
        )
        matter_plan = next(
            (
                (record.payload or {}).get("book_plan") or {}
                for record in self.records
                if record.name == "matter" and record.ok
            ),
            {},
        )
        concatenated = any(
            record.name == "concatenate"
            and record.ok
            and (
                int((record.payload or {}).get("ok_count") or 0) > 0
                or int((record.payload or {}).get("reused_count") or 0) > 0
            )
            for record in self.records
        )
        postprocessed = any(
            record.name == "postprocess"
            and record.ok
            and int((record.payload or {}).get("ok_count") or 0) > 0
            for record in self.records
        )
        bound = any(
            record.name == "bind"
            and record.ok
            and int((record.payload or {}).get("chapter_count") or 0) > 0
            for record in self.records
        )
        epubbed = any(
            record.name == "epub"
            and record.ok
            and int((record.payload or {}).get("document_count") or 0) > 0
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
            if epubbed:
                console.info(
                    f"  next: {rel(self.epub_out)} is the translated book — "
                    "open it in any EPUB reader"
                )
            elif bound:
                console.info(
                    f"  next: {rel(self.translated_book_out)} is the whole translated book — "
                    "one Markdown file assembled from the postprocessed chapters"
                )
            elif postprocessed:
                console.info(
                    f"  next: {rel(self.postprocess_out)}/ holds the postprocessed chapters — "
                    "run bind (or --start-at bind) to concatenate them into one book"
                )
            elif concatenated:
                console.info(
                    f"  next: {rel(self.translated_chapters_out)}/ holds the translated "
                    "book, one file per chapter — the source chapters were not touched"
                )
                if self.translation_prompts_out.is_dir():
                    console.info(
                        f"  debug: {rel(self.translation_prompts_out)}/ holds the whole "
                        "prompt of every chunk (system, user, glossary and source text)"
                    )
                if not self.postprocess_enabled:
                    console.info(
                        "  next: make the headings consistent with --with-postprocess "
                        "(or --start-at postprocess)"
                    )
            elif translated:
                console.info(
                    f"  next: {rel(self.translations_out)}/ holds the translated chunks — "
                    f"run --start-at concatenate to rebuild the translated chapters"
                )
                console.info(
                    f"  debug: {rel(self.translation_prompts_out)}/ holds the whole prompt "
                    "of every chunk (system, user, glossary and source text)"
                )
            elif mattered:
                console.info(
                    f"  next: {rel(self.translations_out)}/ holds the localized front/back "
                    "matter — run --with-translate (or --start-at translate) for the body"
                )
                console.info(
                    f"  plan: {rel(self.matter_out)}/book-plan.json records the localized "
                    f"title, the BleBel credit and the "
                    f"{len(matter_plan.get('omit') or [])} omitted source phrase(s)"
                )
            elif chunk_glossaried:
                console.info(
                    f"  next: {rel(self.chunk_glossary_out)}/ holds one glossary file per chunk — "
                    "send a chunk with its matched terms to keep the translation consistent"
                )
                if not self.translate_enabled:
                    console.info(
                        "  next: translate the book with --with-translate "
                        "(one paid model call per chunk; --translate-limit N for a test)"
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
                    "  stage 2 (preprocess) is switched off — remove --no-preprocess to run it"
                )
            if not self.translate_enabled:
                console.info(
                    "  stage 7 (translation) is switched off — --with-translate to translate "
                    "the chunks (one paid model call per chunk)"
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
            PREPROCESS_MANIFEST_FILENAME,
            PREPROCESS_TRACE_FILENAME,
            REDUCED_MANIFEST_FILENAME,
            REDUCED_TRACE_FILENAME,
            TRACE_FILENAME,
            TRANSLATION_MANIFEST_FILENAME,
            TRANSLATION_TRACE_FILENAME,
        )

        candidates = [
            ("stage-0 markdown", self.markdown_out / f"{book_slug(self.original_source)}.md"),
            ("stage-0 manifest", self.markdown_out / CONVERT_MANIFEST_FILENAME),
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
            ("stage-6b plan", self.matter_out / "book-plan.json"),
            ("stage-6b manifest", self.matter_out / MATTER_MANIFEST_FILENAME),
            ("stage-6b trace", self.matter_out / MATTER_TRACE_FILENAME),
            ("stage-7 manifest", self.translations_out / TRANSLATION_MANIFEST_FILENAME),
            ("stage-7 trace", self.translations_out / TRANSLATION_TRACE_FILENAME),
            ("stage-8 manifest", self.translated_chapters_out / ASSEMBLY_MANIFEST_FILENAME),
            ("stage-8 trace", self.translated_chapters_out / ASSEMBLY_TRACE_FILENAME),
            (
                "stage-9 manifest",
                self.postprocess_out / "_postprocess_manifest.json",
            ),
            ("stage-9 trace", self.postprocess_out / "_postprocess_trace.json"),
            ("stage-10 manifest", self.translated_book_out.parent / "_bind_manifest.json"),
            ("stage-10 trace", self.translated_book_out.parent / "_bind_trace.json"),
            ("stage-11 manifest", self.epub_out.parent / EPUB_MANIFEST_FILENAME),
            ("stage-11 trace", self.epub_out.parent / EPUB_TRACE_FILENAME),
        ]
        return [(label, path) for label, path in candidates if path.is_file()]

    def _summary_json(self, total: float) -> dict[str, Any]:
        return {
            "book": str(self.source),
            "source": str(self.original_source),
            "markdown_dir": str(self.markdown_out),
            "markdown_path": (
                str(self.source)
                if not is_markdown_source(self.original_source) and not self.args.dry_run
                else None
            ),
            "output_dir": str(self.book_dir),
            "chapters_dir": str(self.chapters_dir),
            "reduced_dir": str(self.reduced_out),
            "unique_dir": str(self.unique_out),
            "glossary_dir": str(self.glossary_out),
            "preprocessed_dir": str(self.preprocessed_out),
            "text_input": str(self.text_input()),
            "text_source": self._text_choice()[1],
            "chunks_dir": str(self.chunks_out),
            "chunk_glossary_dir": str(self.chunk_glossary_out),
            "translations_dir": str(self.translations_out),
            "translation_prompts_dir": str(self.translation_prompts_out),
            "matter_dir": str(self.matter_out),
            "matter_prompts_dir": str(self.matter_prompts_out),
            "translated_chapters_dir": str(self.translated_chapters_out),
            "postprocessed_chapters_dir": str(self.postprocess_out),
            "translated_book_path": str(self.translated_book_out),
            "translated_epub_path": str(self.epub_out),
            "first_chapters": self.first_chapters,
            "translate_first_chapters": self.translate_first_chapters,
            "chapter_limit": self.chapter_limit,
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
        help="stage-1 chapters folder (default: <output-root>/<book>/01-chapters)",
    )
    parser.add_argument(
        "--map",
        default=None,
        metavar="FILE",
        help="stage 1: replay a chapter map from a JSON file instead of running the agent "
        "(no provider, no key, no network)",
    )
    parser.add_argument(
        "--preprocessed-out", default=None, help="stage-2 folder (default: a sibling of 01-chapters/)"
    )
    parser.add_argument(
        "--reduced-out", default=None, help="stage-3 folder (default: a sibling of 01-chapters/)"
    )
    parser.add_argument(
        "--unique-out",
        default=None,
        help="stage-3 first-mention folder (default: a sibling of the reduced folder, "
        "03-reduced-unique/)",
    )
    parser.add_argument(
        "--text-in",
        default=None,
        help="chapters folder stages 3-5 read as text (default: 02-preprocessed/ when "
        "stage 2 covers the window, else 01-chapters/); point it at 01-chapters/ to "
        "force the raw text",
    )
    parser.add_argument(
        "--chunk-in",
        default=None,
        help="stage-5 input folder, overriding --text-in (default: the text folder "
        "above, i.e. the cleaned 02-preprocessed/ when stage 2 covered the window)",
    )
    parser.add_argument(
        "--chunks-out",
        default=None,
        help="stage-5 output folder (default: <book>/05-chunks/)",
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
        "(default: <book>/06-chunk-glossary)",
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
        "--translations-out",
        default=None,
        help="stage 7: where the translated chunks go (default: <book>/07-translations)",
    )
    parser.add_argument(
        "--translation-prompts-out",
        default=None,
        help="stage 7: where the per-chunk prompt/input records go "
        "(default: <book>/07-translation-prompts)",
    )
    parser.add_argument(
        "--matter-out",
        default=None,
        help="stage 6b: where the front/back matter plan, manifest and trace go "
        "(default: <book>/06b-matter)",
    )
    parser.add_argument(
        "--matter-prompts-out",
        default=None,
        help="stage 6b: where the matter agent's whole prompt/input records go "
        "(default: <book>/06b-matter-prompts)",
    )
    parser.add_argument(
        "--with-matter",
        action="store_true",
        help="run the front/back matter localization agent even when translation is "
        "not selected (it is selected with --with-translate by default)",
    )
    parser.add_argument(
        "--no-matter",
        action="store_true",
        help="drop the front/back matter localization stage; stage 7 then skips the "
        "matter chunks and stages 8-10 do not assemble them",
    )
    parser.add_argument(
        "--matter-limit",
        type=int,
        default=None,
        help="stage 6b: localize only the first N matter chunks (a cheap test of the "
        "plan and the localization loop)",
    )
    parser.add_argument(
        "--translated-chapters-out",
        default=None,
        help="stage 8: where the translated chapter files go "
        "(default: <book>/08-translated-chapters)",
    )
    parser.add_argument(
        "--postprocess-out",
        default=None,
        help="stage 9: where the postprocessed chapter files go "
        "(default: <book>/09-postprocessed-chapters)",
    )
    parser.add_argument(
        "--book-out",
        default=None,
        help="stage 10: where the single Markdown book goes "
        "(default: <book>/10-translated-book.md)",
    )
    parser.add_argument(
        "--with-postprocess",
        action="store_true",
        help="run stage 9 even when neither translation nor assembly is selected "
        "(over an existing translated-chapters folder)",
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="drop stage 9 (heading/TOC postprocessing) from this run",
    )
    parser.add_argument(
        "--no-bind",
        action="store_true",
        help="drop stage 10 (single Markdown book concatenation) from this run",
    )
    parser.add_argument(
        "--epub-out",
        default=None,
        help="stage 11: where the EPUB 3 book goes "
        "(default: <book>/11-translated-book.epub)",
    )
    parser.add_argument(
        "--no-epub",
        action="store_true",
        help="drop stage 11 (EPUB 3 packaging) from this run",
    )
    parser.add_argument(
        "--epub-cover",
        default=None,
        help="stage 11: cover picture — a path, or a file name inside the book's "
        "00-markdown/images/ folder",
    )
    parser.add_argument("--epub-title", default=None, help="stage 11: override the title")
    parser.add_argument("--epub-author", default=None, help="stage 11: override the author")
    parser.add_argument(
        "--epub-language",
        default=None,
        help="stage 11: override the language, an ISO 639 code",
    )
    parser.add_argument(
        "--first-chapters",
        "--chapters",
        dest="first_chapters",
        type=int,
        default=None,
        help="run every body stage after splitting over only the first N body chapters; "
        "front/back matter does not count as a chapter, and splitting itself "
        "always runs over the whole book (the matter stage is never windowed)",
    )
    parser.add_argument(
        "--translate-first-chapters",
        type=int,
        default=None,
        help="stage 7/8: translate and assemble only the first N body chapters "
        "(default: --first-chapters)",
    )
    parser.add_argument(
        "--no-concat",
        action="store_true",
        help="drop stage 8 (concatenating the translated chunks into chapters) "
        "from this run",
    )
    parser.add_argument(
        "--with-translate",
        action="store_true",
        help="run stage 7 (translate every chunk) — off by default because it is one "
        "paid model call per chunk",
    )
    parser.add_argument(
        "--no-translate",
        action="store_true",
        help="drop stage 7 from this run (it is off unless --with-translate is given)",
    )
    parser.add_argument(
        "--translate-limit",
        type=int,
        default=None,
        help="stage 7: translate only the first N chunks (defaults to --limit); the "
        "cheap way to test the whole translation path",
    )
    parser.add_argument(
        "--translate-glossary-scope",
        choices=TRANSLATION_GLOSSARY_SCOPES,
        default="chunk",
        help="stage 7: send the matched terms stage 6 attached to each chunk (default) "
        "or the whole glossary.json with every chunk",
    )
    parser.add_argument(
        "--translate-attempts",
        type=int,
        default=DEFAULT_TRANSLATION_ATTEMPTS,
        help="stage 7: model calls allowed per chunk before it is reported as failed "
        f"(default: {DEFAULT_TRANSLATION_ATTEMPTS})",
    )
    parser.add_argument(
        "--translate-method",
        choices=TRANSLATION_METHODS,
        default="auto",
        help="stage 7: 'auto' asks for native structured output and falls back to JSON "
        "mode when the provider rejects the schema (default)",
    )
    parser.add_argument(
        "--translate-max-tokens",
        type=int,
        default=DEFAULT_TRANSLATION_MAX_TOKENS,
        help="stage 7: response cap per model call "
        f"(default: {DEFAULT_TRANSLATION_MAX_TOKENS})",
    )
    parser.add_argument(
        "--translate-context-chars",
        type=int,
        default=DEFAULT_TRANSLATION_CONTEXT_CHARS,
        help="stage 7: send the last N characters of the previous translation as "
        f"continuity context (default: {DEFAULT_TRANSLATION_CONTEXT_CHARS} = off)",
    )
    parser.add_argument(
        "--translate-provider",
        choices=sorted(TRANSLATION_CHOICES),
        default=None,
        help="stage 7 only: 'deepseek' (the cloud default) or 'lmstudio' (the local "
        "OpenAI-compatible server at http://127.0.0.1:1234/v1, whose loaded model "
        "is auto-detected). Front/back matter and every other stage keep the "
        "shared provider",
    )
    parser.add_argument(
        "--translate-model",
        default=None,
        help="stage 7 only: the model id; for LM Studio this is the id the server "
        "lists (e.g. gemma-4-31b-en2sr-v3@q4_k_m). Default: discover it",
    )
    parser.add_argument(
        "--translate-base-url",
        default=None,
        help="stage 7 only: OpenAI-compatible base URL (default: the provider's; "
        "LM Studio uses http://127.0.0.1:1234/v1)",
    )
    parser.add_argument(
        "--translate-api-key",
        default=None,
        help="stage 7 only: API key; a local LM Studio endpoint needs none",
    )
    parser.add_argument(
        "--translate-temperature",
        type=float,
        default=None,
        help="stage 7 only: sampling temperature (default: --temperature, 0.0)",
    )
    parser.add_argument(
        "--include-matter",
        action="store_true",
        help="stage 7: translate the front/back matter chunks too, with the body "
        "prompt (the old behaviour; overwrites stage 6b's localized chunks)",
    )

    parser.add_argument(
        "--force-convert",
        action="store_true",
        help="stage 0: convert again even when the Markdown in 00-markdown/ still "
        "matches the source hash",
    )
    parser.add_argument(
        "--no-images",
        dest="images",
        action="store_false",
        help="stage 0: do not extract pictures from the EPUB/PDF (images are "
        "extracted into 00-markdown/images/ by default)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate every plan; write no artifact except this log (stage 0 converts "
        "into a temporary folder so the later plans still have text to read)",
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
        help="stages 3/6: 'auto' detects the language by sampling 01-chapters/chunks; otherwise "
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
        "--resume",
        dest="skip_if_present",
        action="store_true",
        help="reuse work already on disk: existing chapters for stage 1, existing "
        "reduced chapters for stage 3 (no model is loaded for the latter), "
        "existing chunk glossaries for stage 6, existing matter localizations for "
        "stage 6b, existing translations for stage 7, "
        "and existing translated chapters for stage 8",
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
        help="stage 4: where glossary.json / glossary.md go (default: <book>/04-glossary)",
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
        help="stages 2/3/4/5/6/7: only the first N units (chapters, or chunks for "
        "stages 6/7) — a quick look, and how to price the glossary or a "
        "chunk/translation run before the whole book; --first-chapters is the "
        "chapter-aligned alternative",
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
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="shared sampling temperature for every stage (default: the config file, "
        "else 0.0); use --llm <stage>.temperature to set one stage only",
    )
    parser.add_argument(
        "--llm-config",
        default=None,
        metavar="FILE",
        help=f"per-stage model config (default: ./{LLM_CONFIG_FILENAME}, or "
        f"$BOOK_SPLITTER_CONFIG); one [llm] table plus an optional [llm.<stage>] "
        "table per stage",
    )
    parser.add_argument(
        "--llm",
        action="append",
        default=[],
        metavar="STAGE.FIELD=VALUE",
        help="override one model field for one stage, repeatable; stages: "
        f"{', '.join(STAGES)}; fields: provider, model, base_url, temperature, "
        "timeout, max_retries, api_key_env (e.g. --llm translate.provider=lmstudio "
        "--llm translate.model=my-model)",
    )
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
        help="accepted for older command lines; stage 2 is on by default",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="drop stage 2 (chapter cleaning) from this run (it is on by default)",
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
    for flag in ("first_chapters", "translate_first_chapters"):
        value = getattr(args, flag, None)
        if value is not None and int(value) < 0:
            print(f"error: --{flag.replace('_', '-')} cannot be negative", file=sys.stderr)
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
