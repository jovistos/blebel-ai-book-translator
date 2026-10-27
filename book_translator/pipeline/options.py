"""Normalized options and the exact runner call each phase makes.

The CLI and the pipeline used to build runner calls twice: once per ``cmd_*`` and
once per ``Pipeline.phase_*``. They drifted (a ``write_manifest`` condition here,
an ``identifier`` there, ``translate_first_chapters`` in one place and
``first_chapters`` in the other). This module is the one builder:

* :class:`PhaseSelection` decides *which* phases run, from the switches and the
  start/stop window;
* :class:`StageOptions` holds the normalized per-stage knobs;
* :meth:`StageOptions.call` returns a :class:`StageCall` with the exact
  positional arguments, runner function and keyword arguments.

No ``**vars(args)``: every runner kwarg is written out, so a test can assert the
whole contract and a new option cannot silently leak into an unrelated stage.
Runner lookup is lazy, which keeps the existing test seam (monkeypatching
``book_translator.stages.agent.run_split``) working.
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, cast

from book_translator.stages.dictionary.reduce import DEFAULT_EXAMPLE_LIMIT
from book_translator.stages.dictionary.unique import DEFAULT_UNIQUE_EXAMPLES

from ..core.constants import (
    DEFAULT_CHUNK_GLOSSARY_CANDIDATES,
    DEFAULT_CHUNK_GLOSSARY_MIN_TERM_CHARS,
    DEFAULT_CHUNK_GLOSSARY_SCORER,
    DEFAULT_CHUNK_GLOSSARY_THRESHOLD,
    DEFAULT_CHUNK_TOKENIZER,
    DEFAULT_CHUNK_TOKENS,
    DEFAULT_FRONT_MATTER_LINES,
    DEFAULT_GLOSSARY_CHUNK_CHARS,
    DEFAULT_KEEP_ENTITY_TYPES,
    DEFAULT_MIN_CHAPTERS,
    DEFAULT_MIN_SENTENCE_WORDS,
    DEFAULT_RECURSION_LIMIT,
    DEFAULT_SOURCE_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
    DEFAULT_TRANSLATION_ATTEMPTS,
    DEFAULT_TRANSLATION_CONTEXT_CHARS,
    DEFAULT_TRANSLATION_MAX_TOKENS,
    MAX_FEEDBACK_LOOPS,
    TRANSLATION_GLOSSARY_SCOPES,
    TRANSLATION_METHODS,
)
from ..core.errors import ConfigError, ToolError
from ..core.settings import parse_llm_overrides
from ..paths import default_input_dir, default_output_dir
from .paths import StagePaths
from .phases import NEEDS_CHAPTERS, PHASES
from .sources import StageSources

#: The phases, in the order they must run. ``convert`` is dropped for a Markdown
#: source; every other phase keeps its position so the numbering is stable.
#: Feature defaults. These used to live in ``scripts/run_pipeline.py``; keeping
#: them here means the driver and the CLI cannot disagree about what is on.
PREPROCESS_ENABLED = True
DICTIONARY_ENABLED = True
GLOSSARY_ENABLED = True
CHUNK_ENABLED = True
CHUNK_GLOSSARY_ENABLED = True
MATTER_ENABLED = True
TRANSLATE_ENABLED = False
POSTPROCESS_ENABLED = False

#: Stage name -> (module, function). Resolved lazily by :func:`resolve_runner`.
_RUNNERS: dict[str, tuple[str, str]] = {
    "convert": ("book_translator.stages.conversion", "run_conversion"),
    "split": ("book_translator.stages.agent", "run_split"),
    "split_map": ("book_translator.stages.agent", "run_split_from_map"),
    "dictionary": ("book_translator.stages.dictionary", "run_dictionary"),
    "glossary": ("book_translator.stages.glossary", "run_glossary"),
    "preprocess": ("book_translator.stages.preprocess_agent", "run_preprocess"),
    "chunk": ("book_translator.stages.chunking", "run_chunking"),
    "chunk_glossary": ("book_translator.stages.chunk_glossary", "run_chunk_glossary"),
    "matter": ("book_translator.stages.matter_agent", "run_matter"),
    "translate": ("book_translator.stages.translation", "run_translation"),
    "concatenate": ("book_translator.stages.assembly", "run_assembly"),
    "postprocess": ("book_translator.stages.postprocess_agent", "run_postprocess"),
    "bind": ("book_translator.stages.binding", "run_bind"),
    "epub": ("book_translator.stages.epub", "run_epub"),
}


def resolve_runner(stage: str) -> Callable[..., Any]:
    """Resolve a runner lazily, so tests can still patch the package attribute."""
    try:
        module_name, function_name = _RUNNERS[stage]
    except KeyError as exc:  # pragma: no cover - programming error
        raise ToolError(f"unknown pipeline stage {stage!r}") from exc
    module = importlib.import_module(module_name)
    return cast(Callable[..., Any], getattr(module, function_name))


def _get(ns: Any, name: str, default: Any = None) -> Any:
    value = getattr(ns, name, None)
    return default if value is None else value


def _int(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def _first(ns: Any, names: Sequence[str], default: Any = None) -> Any:
    """The first non-None attribute among the aliases (``--translate-method`` etc.)."""
    for name in names:
        value = getattr(ns, name, None)
        if value is not None:
            return value
    return default


def _entity_types(value: Any) -> tuple[str, ...]:
    if not value:
        return DEFAULT_KEEP_ENTITY_TYPES
    wanted = tuple(part.strip().upper() for part in str(value).split(",") if part.strip())
    return wanted or DEFAULT_KEEP_ENTITY_TYPES


# --------------------------------------------------------------------------- #
# which phases run
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PhaseSelection:
    """The permanent enable switches, plus the start/stop window."""

    preprocess: bool = True
    dictionary: bool = True
    glossary: bool = True
    chunk: bool = True
    chunk_glossary: bool = True
    matter: bool = False
    translate: bool = False
    concatenate: bool = False
    postprocess: bool = False
    bind: bool = False
    epub: bool = False
    start_at: str = PHASES[0]
    stop_after: str = PHASES[-1]

    @classmethod
    def from_namespace(cls, ns: Any) -> PhaseSelection:
        start_at = str(_get(ns, "start_at", PHASES[0]))
        stop_after = str(_get(ns, "stop_after", PHASES[-1]))
        preprocess = bool(PREPROCESS_ENABLED and not _get(ns, "no_preprocess", False))
        dictionary = bool(DICTIONARY_ENABLED and not _get(ns, "no_dictionary", False))
        glossary = bool(
            GLOSSARY_ENABLED and dictionary and not _get(ns, "no_glossary", False)
        )
        chunk = bool(CHUNK_ENABLED and not _get(ns, "no_chunk", False))
        chunk_glossary = bool(
            CHUNK_GLOSSARY_ENABLED and chunk and not _get(ns, "no_chunk_glossary", False)
        )
        translate = bool(
            (TRANSLATE_ENABLED or _get(ns, "with_translate", False))
            and not _get(ns, "no_translate", False)
            and chunk
        )
        wants_matter = bool(
            MATTER_ENABLED
            and (
                translate
                or _get(ns, "with_matter", False)
                or start_at == "matter"
            )
        )
        matter = bool(wants_matter and not _get(ns, "no_matter", False) and chunk)
        concatenate = bool(
            not _get(ns, "no_concat", False)
            and (translate or start_at == "concatenate")
        )
        postprocess = bool(
            not _get(ns, "no_postprocess", False)
            and (
                POSTPROCESS_ENABLED
                or _get(ns, "with_postprocess", False)
                or start_at == "postprocess"
            )
        )
        bind = bool(
            not _get(ns, "no_bind", False)
            and (concatenate or postprocess or start_at == "bind")
        )
        epub = bool(
            not _get(ns, "no_epub", False) and (bind or start_at == "epub")
        )
        return cls(
            preprocess=preprocess,
            dictionary=dictionary,
            glossary=glossary,
            chunk=chunk,
            chunk_glossary=chunk_glossary,
            matter=matter,
            translate=translate,
            concatenate=concatenate,
            postprocess=postprocess,
            bind=bind,
            epub=epub,
            start_at=start_at,
            stop_after=stop_after,
        )

    def _switches(self) -> dict[str, bool]:
        return {
            "preprocess": self.preprocess,
            "dictionary": self.dictionary,
            "chunk": self.chunk,
            "chunk_glossary": self.chunk_glossary,
            "matter": self.matter,
            "translate": self.translate,
            "concatenate": self.concatenate,
            "postprocess": self.postprocess,
            "bind": self.bind,
            "epub": self.epub,
        }

    def _window(self) -> tuple[int, int]:
        try:
            return PHASES.index(self.start_at), PHASES.index(self.stop_after)
        except ValueError as exc:
            raise ConfigError(
                f"unknown phase; choose one of {', '.join(PHASES)}"
            ) from exc

    def selected(self, *, convert_needed: bool = True) -> tuple[str, ...]:
        """The phases this run will actually execute, in order."""
        start, stop = self._window()
        phases = list(PHASES[start : stop + 1])
        if not convert_needed:
            phases = [name for name in phases if name != "convert"]
        switches = self._switches()
        phases = [
            name for name in phases if name in {"convert", "recon", "split", "scan"} or switches[name]
        ]
        return tuple(phases)

    def switched_off(self, *, convert_needed: bool = True) -> tuple[str, ...]:
        """Phases in the requested window that a switch dropped."""
        start, stop = self._window()
        candidates = list(PHASES[start : stop + 1])
        if not convert_needed:
            candidates = [name for name in candidates if name != "convert"]
        selected = set(self.selected(convert_needed=convert_needed))
        return tuple(name for name in candidates if name not in selected)


# --------------------------------------------------------------------------- #
# one stage's normalized options
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StageOptions:
    """Every per-stage knob, normalized once.

    The fields are deliberately flat: they mirror the CLI flags and the runner
    parameters, and ``from_namespace`` is the only place that knows the aliases
    (``--translator``/``--provider``, ``--method``/``--translation-method``,
    ``--threshold``/``--match-threshold``).
    """

    mode: str = "pipeline"
    # windows / writing
    first_chapters: int | None = None
    translate_first_chapters: int | None = None
    limit: int | None = None
    dry_run: bool = False
    skip_if_present: bool = False
    resume: bool = False
    no_manifest: bool = False
    no_prompts: bool = False
    # split
    front_matter_lines: int = DEFAULT_FRONT_MATTER_LINES
    max_attempts: int = MAX_FEEDBACK_LOOPS
    min_chapters: int = DEFAULT_MIN_CHAPTERS
    split_include_front_matter: bool = True
    recursion_limit: int = DEFAULT_RECURSION_LIMIT
    workers: int = 1
    map_path: str | None = None
    # dictionary
    language: str = "auto"
    gpu: int | None = None
    use_gpu: bool = True
    min_words: int = DEFAULT_MIN_SENTENCE_WORDS
    require_uppercase: bool = True
    keep_types: tuple[str, ...] = DEFAULT_KEEP_ENTITY_TYPES
    drop_temporal_only: bool = False
    save_reduced: bool = True
    save_unique: bool = True
    example_limit: int = DEFAULT_EXAMPLE_LIMIT
    unique_example_limit: int = DEFAULT_UNIQUE_EXAMPLES
    # chunking
    chunk_size: int = DEFAULT_CHUNK_TOKENS
    chunk_tokenizer: str = DEFAULT_CHUNK_TOKENIZER
    # chunk glossary
    threshold: float = DEFAULT_CHUNK_GLOSSARY_THRESHOLD
    scorer: str = DEFAULT_CHUNK_GLOSSARY_SCORER
    candidates: int = DEFAULT_CHUNK_GLOSSARY_CANDIDATES
    min_term_chars: int = DEFAULT_CHUNK_GLOSSARY_MIN_TERM_CHARS
    # glossary
    sentences: str = "unique"
    sentences_dir: str | None = None
    source_language: str = DEFAULT_SOURCE_LANGUAGE
    target_language: str = DEFAULT_TARGET_LANGUAGE
    book_title: str = ""
    max_context_entries: int = 0
    glossary_include_front_matter: bool = False
    glossary_include_back_matter: bool = False
    glossary_method: str = "auto"
    glossary_max_tokens: int | None = None
    glossary_chunk_chars: int = DEFAULT_GLOSSARY_CHUNK_CHARS
    # matter / translation
    matter_limit: int | None = None
    translate_limit: int | None = None
    glossary_scope: str = "chunk"
    translation_method: str = "auto"
    translation_max_tokens: int | None = DEFAULT_TRANSLATION_MAX_TOKENS
    translation_attempts: int = DEFAULT_TRANSLATION_ATTEMPTS
    translation_context_chars: int = DEFAULT_TRANSLATION_CONTEXT_CHARS
    include_matter: bool = False
    # postprocess
    require_web_search: bool = True
    # epub
    epub_title: str | None = None
    epub_author: str | None = None
    epub_language: str | None = None
    epub_identifier: str | None = None
    epub_description: str | None = None
    epub_publisher: str | None = None
    epub_cover: str | None = None
    force_convert: bool = False
    images: bool = True
    ns: Any = None

    @classmethod
    def from_namespace(cls, ns: Any, *, mode: str = "pipeline") -> StageOptions:
        skip_if_present = bool(_get(ns, "skip_if_present", False))
        resume = bool(_get(ns, "resume", False)) or skip_if_present
        threshold = _get(ns, "threshold", None)
        if threshold is None:
            threshold = _get(ns, "match_threshold", DEFAULT_CHUNK_GLOSSARY_THRESHOLD)
        method = _first(
            ns, ("translate_method", "method", "translation_method"), "auto"
        )
        context_chars = _first(
            ns,
            ("translate_context_chars", "context_chars"),
            DEFAULT_TRANSLATION_CONTEXT_CHARS,
        )
        max_tokens = _first(
            ns,
            ("translate_max_tokens", "max_tokens"),
            DEFAULT_TRANSLATION_MAX_TOKENS,
        )
        glossary_scope = _first(
            ns, ("translate_glossary_scope", "glossary_scope"), "chunk"
        )
        translation_attempts = _first(
            ns, ("translate_attempts", "max_attempts"), DEFAULT_TRANSLATION_ATTEMPTS
        )
        glossary_method = (
            _get(ns, "extraction_method", None)
            or _get(ns, "method", None)
            or "auto"
        )
        cover = _get(ns, "cover", None) or _get(ns, "epub_cover", None)
        return cls(
            mode=mode,
            first_chapters=_int(_get(ns, "first_chapters")),
            translate_first_chapters=_int(_get(ns, "translate_first_chapters")),
            limit=_int(_get(ns, "limit")),
            dry_run=bool(_get(ns, "dry_run", False)),
            skip_if_present=skip_if_present,
            resume=resume,
            no_manifest=bool(_get(ns, "no_manifest", False)),
            no_prompts=bool(_get(ns, "no_prompts", False)),
            front_matter_lines=int(
                _get(ns, "front_matter_lines", DEFAULT_FRONT_MATTER_LINES)
            ),
            max_attempts=int(_get(ns, "max_attempts", MAX_FEEDBACK_LOOPS)),
            min_chapters=int(_get(ns, "min_chapters", DEFAULT_MIN_CHAPTERS)),
            split_include_front_matter=not bool(_get(ns, "no_front_matter", False)),
            recursion_limit=int(
                _get(ns, "recursion_limit", DEFAULT_RECURSION_LIMIT)
            ),
            workers=max(1, int(_get(ns, "workers", 1))),
            map_path=_get(ns, "map", None),
            language=str(_get(ns, "language", "auto")),
            gpu=_int(_get(ns, "gpu")),
            use_gpu=not bool(_get(ns, "cpu", False)),
            min_words=int(_get(ns, "min_words", DEFAULT_MIN_SENTENCE_WORDS)),
            require_uppercase=not bool(_get(ns, "no_uppercase_filter", False)),
            keep_types=_entity_types(_get(ns, "keep_types", None)),
            drop_temporal_only=bool(_get(ns, "drop_temporal_only", False)),
            save_reduced=not bool(_get(ns, "no_save_reduced", False))
            and not bool(_get(ns, "unique_only", False)),
            save_unique=not bool(_get(ns, "no_save_unique", False)),
            example_limit=int(_get(ns, "examples", DEFAULT_EXAMPLE_LIMIT)),
            unique_example_limit=int(_get(ns, "examples", DEFAULT_UNIQUE_EXAMPLES)),
            chunk_size=int(_get(ns, "chunk_size", DEFAULT_CHUNK_TOKENS)),
            chunk_tokenizer=str(
                _get(ns, "chunk_tokenizer", DEFAULT_CHUNK_TOKENIZER)
            ),
            threshold=float(threshold),
            scorer=str(
                _get(ns, "scorer", DEFAULT_CHUNK_GLOSSARY_SCORER)
            ),
            candidates=int(
                _get(ns, "candidates", DEFAULT_CHUNK_GLOSSARY_CANDIDATES)
            ),
            min_term_chars=int(
                _get(ns, "min_term_chars", DEFAULT_CHUNK_GLOSSARY_MIN_TERM_CHARS)
            ),
            sentences=str(_get(ns, "sentences", "unique")),
            sentences_dir=_get(ns, "sentences_in", None),
            source_language=str(
                _get(ns, "source_language", DEFAULT_SOURCE_LANGUAGE)
            ),
            target_language=str(
                _get(ns, "target_language", DEFAULT_TARGET_LANGUAGE)
            ),
            book_title=str(_get(ns, "book_title", "") or ""),
            max_context_entries=int(_get(ns, "max_context_entries", 0)),
            glossary_include_front_matter=bool(
                _get(ns, "include_front_matter", False)
            ),
            glossary_include_back_matter=bool(
                _get(ns, "include_back_matter", False)
            ),
            glossary_method=str(glossary_method),
            glossary_max_tokens=_int(_get(ns, "max_tokens")),
            glossary_chunk_chars=int(
                _get(ns, "max_chapter_chars", DEFAULT_GLOSSARY_CHUNK_CHARS)
            ),
            matter_limit=_int(_get(ns, "matter_limit")),
            translate_limit=_int(_get(ns, "translate_limit")),
            glossary_scope=str(glossary_scope),
            translation_method=str(method),
            translation_max_tokens=_int(max_tokens) or None,
            translation_attempts=int(translation_attempts),
            translation_context_chars=int(context_chars),
            include_matter=bool(_get(ns, "include_matter", False)),
            require_web_search=bool(_get(ns, "require_web_search", True)),
            epub_title=_get(ns, "epub_title", _get(ns, "title", None)),
            epub_author=_get(ns, "epub_author", _get(ns, "author", None)),
            epub_language=_get(ns, "epub_language", _get(ns, "language", None)),
            epub_identifier=_get(ns, "identifier", None),
            epub_description=_get(ns, "description", None),
            epub_publisher=_get(ns, "publisher", None),
            epub_cover=cover,
            force_convert=bool(_get(ns, "force_convert", False)),
            images=bool(_get(ns, "images", True)),
            ns=ns,
        )

    # -- window helpers ---------------------------------------------------- #
    def chapter_limit(self, runtime: StageRuntime) -> int | None:
        return (
            runtime.chapter_limit
            if runtime.chapter_limit is not None
            else self.limit
        )

    def assembly_first_chapters(self) -> int | None:
        return self.translate_first_chapters or self.first_chapters

    def postprocess_limit(self) -> int | None:
        return (
            self.translate_first_chapters
            or self.first_chapters
            or self.limit
        )

    def translation_limit(self) -> int | None:
        if self.mode == "pipeline":
            return self.translate_limit if self.translate_limit is not None else self.limit
        return self.limit

    def matter_limit_for(self) -> int | None:
        if self.mode == "pipeline":
            return self.matter_limit if self.matter_limit is not None else None
        return self.limit

    # -- the call builders ------------------------------------------------- #
    def _call_convert(self, runtime: StageRuntime, write_manifest: bool, write_prompts: bool, chapter_limit: int | None):
        paths = runtime.paths
        positional = (
            self._original_source(runtime),
            runtime.conversion_target or paths.markdown_dir,
        )
        kwargs = {
            "dry_run": False if runtime.conversion_live else self.dry_run,
            "force": self.force_convert,
            "images": self.images,
        }
        return positional, kwargs

    def _call_split(self, runtime: StageRuntime, write_manifest: bool, write_prompts: bool, chapter_limit: int | None):
        paths = runtime.paths
        positional = (self._source(runtime), paths.chapters_dir)
        kwargs = {
            "settings": runtime.settings,
            "front_matter_lines": self.front_matter_lines,
            "max_attempts": self.max_attempts,
            "min_chapters": self.min_chapters,
            "include_front_matter": self.split_include_front_matter,
            "dry_run": self.dry_run,
            "recursion_limit": self.recursion_limit,
            "skip_if_present": self.skip_if_present,
        }
        return positional, kwargs

    def _call_dictionary(self, runtime: StageRuntime, write_manifest: bool, write_prompts: bool, chapter_limit: int | None):
        paths, sources = runtime.paths, runtime.sources
        positional = (sources.text, paths.reduced_dir)
        kwargs = {
            "unique_output_dir": str(paths.unique_dir),
            "language": self.language,
            "gpu": self.gpu,
            "use_gpu": self.use_gpu,
            "min_words": self.min_words,
            "require_uppercase": self.require_uppercase,
            "keep_types": self.keep_types,
            "drop_temporal_only": self.drop_temporal_only,
            "skip_if_present": self.skip_if_present,
            "save_reduced": self.save_reduced,
            "save_unique": self.save_unique,
            "dry_run": self.dry_run,
            "limit": chapter_limit,
            "example_limit": self.example_limit,
            "unique_example_limit": self.unique_example_limit,
            "write_manifest": write_manifest,
            "progress": runtime.progress,
        }
        return positional, kwargs

    def _call_glossary(self, runtime: StageRuntime, write_manifest: bool, write_prompts: bool, chapter_limit: int | None):
        paths, sources = runtime.paths, runtime.sources
        positional = (sources.text, paths.glossary_dir)
        kwargs = {
            "sentences_dir": runtime.sentences_dir or self.sentences_dir,
            "sentences": self.sentences,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "book_title": self.book_title,
            "settings": runtime.settings,
            "settings_factory": runtime.settings_factory,
            "max_attempts": self.max_attempts,
            "method": self.glossary_method,
            "max_tokens": self.glossary_max_tokens,
            "chunk_chars": self.glossary_chunk_chars,
            "max_context_entries": self.max_context_entries,
            "include_front_matter": self.glossary_include_front_matter,
            "include_back_matter": self.glossary_include_back_matter,
            "skip_if_present": self.skip_if_present,
            "save": True,
            "dry_run": self.dry_run,
            "limit": chapter_limit,
            "progress": runtime.progress,
            "chapter_start": runtime.chapter_start,
            "clock": runtime.clock,
        }
        return positional, kwargs

    def _call_preprocess(self, runtime: StageRuntime, write_manifest: bool, write_prompts: bool, chapter_limit: int | None):
        paths = runtime.paths
        positional = (paths.chapters_dir, paths.preprocessed_dir)
        kwargs = {
            "settings": runtime.settings,
            "max_attempts": self.max_attempts,
            "dry_run": self.dry_run,
            "recursion_limit": self.recursion_limit,
            "workers": self.workers,
            "limit": chapter_limit,
            "skip_if_present": self.skip_if_present,
            "write_manifest": write_manifest,
            "progress": runtime.progress,
        }
        return positional, kwargs

    def _call_chunk(self, runtime: StageRuntime, write_manifest: bool, write_prompts: bool, chapter_limit: int | None):
        paths, sources = runtime.paths, runtime.sources
        positional = (sources.chunk_input, paths.chunks_dir)
        kwargs = {
            "chunk_size": self.chunk_size,
            "tokenizer": self.chunk_tokenizer,
            "skip_if_present": self.skip_if_present,
            "dry_run": self.dry_run,
            "limit": chapter_limit,
            "write_manifest": write_manifest,
            "progress": runtime.progress,
            "clock": runtime.clock,
        }
        return positional, kwargs

    def _call_chunk_glossary(self, runtime: StageRuntime, write_manifest: bool, write_prompts: bool, chapter_limit: int | None):
        paths, sources = runtime.paths, runtime.sources
        positional = (sources.chunks, paths.chunk_glossary_dir)
        kwargs = {
            "glossary_path": str(sources.glossary),
            "language": self.language,
            "gpu": self.gpu,
            "use_gpu": self.use_gpu,
            "keep_types": self.keep_types,
            "min_term_chars": self.min_term_chars,
            "drop_temporal_only": self.drop_temporal_only,
            "threshold": self.threshold,
            "scorer": self.scorer,
            "candidates": self.candidates,
            "dry_run": self.dry_run,
            "limit": self.limit,
            "chapter_files": runtime.chapter_files,
            "skip_if_present": self.skip_if_present,
            "write_manifest": write_manifest,
            "progress": runtime.progress,
        }
        return positional, kwargs

    def _call_matter(self, runtime: StageRuntime, write_manifest: bool, write_prompts: bool, chapter_limit: int | None):
        paths, sources = runtime.paths, runtime.sources
        positional = (sources.chapters, sources.chunks, paths.translations_dir)
        kwargs = {
            "prompts_dir": None if self.no_prompts else str(paths.matter_prompts_dir),
            "matter_path": str(paths.matter_dir),
            "source_language": self.source_language,
            "target_language": self.target_language,
            "book_title": self.book_title,
            "settings": runtime.settings,
            "settings_factory": runtime.settings_factory,
            "max_attempts": self.max_attempts,
            "recursion_limit": self.recursion_limit,
            "dry_run": self.dry_run,
            "save": True,
            "resume": self.resume,
            "limit": self.matter_limit_for(),
            "chapter_files": runtime.chapter_files,
            "write_manifest": write_manifest,
            "write_prompts": write_prompts,
            "progress": runtime.progress,
            "clock": runtime.clock,
        }
        return positional, kwargs

    def _call_translate(self, runtime: StageRuntime, write_manifest: bool, write_prompts: bool, chapter_limit: int | None):
        paths, sources = runtime.paths, runtime.sources
        positional = (sources.chunks, paths.translations_dir)
        kwargs = {
            "prompts_dir": None if self.no_prompts else str(paths.translation_prompts_dir),
            "chunk_glossary_path": str(sources.chunk_glossary),
            "glossary_path": str(sources.glossary),
            "glossary_scope": self.glossary_scope,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "book_title": self.book_title,
            "settings": runtime.settings,
            "settings_factory": runtime.settings_factory,
            "model_identity": runtime.model_identity,
            "max_attempts": self.translation_attempts,
            "method": self.translation_method,
            "max_tokens": self.translation_max_tokens,
            "context_chars": self.translation_context_chars,
            "save": True,
            "dry_run": self.dry_run,
            "limit": self.translation_limit(),
            "chapter_files": runtime.chapter_files,
            "include_matter": self.include_matter,
            "resume": self.resume,
            "write_manifest": write_manifest,
            "write_prompts": write_prompts,
            "progress": runtime.progress,
            "chunk_start": runtime.chunk_start,
            "clock": runtime.clock,
        }
        return positional, kwargs

    def _call_concatenate(self, runtime: StageRuntime, write_manifest: bool, write_prompts: bool, chapter_limit: int | None):
        paths, sources = runtime.paths, runtime.sources
        positional = (
            sources.chunks,
            sources.translations,
            paths.translated_chapters_dir,
        )
        kwargs = {
            "first_chapters": self.assembly_first_chapters(),
            "skip_if_present": self.skip_if_present,
            "dry_run": self.dry_run,
            "write_manifest": write_manifest,
            "progress": runtime.progress,
            "clock": runtime.clock,
        }
        return positional, kwargs

    def _call_postprocess(self, runtime: StageRuntime, write_manifest: bool, write_prompts: bool, chapter_limit: int | None):
        paths, sources = runtime.paths, runtime.sources
        positional = (sources.translated_chapters, paths.postprocess_dir)
        kwargs = {
            "settings": runtime.settings,
            "max_attempts": self.max_attempts,
            "dry_run": self.dry_run,
            "recursion_limit": self.recursion_limit,
            "limit": self.postprocess_limit(),
            "skip_if_present": self.skip_if_present,
            "write_manifest": write_manifest,
            "require_web_search": self.require_web_search,
            "progress": runtime.progress,
        }
        return positional, kwargs

    def _call_bind(self, runtime: StageRuntime, write_manifest: bool, write_prompts: bool, chapter_limit: int | None):
        paths, sources = runtime.paths, runtime.sources
        positional = (sources.postprocess, paths.translated_book_path)
        kwargs = {"dry_run": self.dry_run, "write_manifest": write_manifest}
        return positional, kwargs

    def _call_epub(self, runtime: StageRuntime, write_manifest: bool, write_prompts: bool, chapter_limit: int | None):
        paths, sources = runtime.paths, runtime.sources
        positional = (sources.postprocess, paths.epub_path)
        cover = self.epub_cover
        if cover is None and sources.postprocess is not None:
            parent = Path(sources.postprocess).parent
            trans_img = parent / "07-translations" / "images"
            orig_img = parent / "00-markdown" / "images"
            for d in (trans_img, orig_img):
                if d.is_dir():
                    for pat in ("*cover*_translated.*", "image_000000*_translated.*", "*cover*.*", "image_000000*.*"):
                        matches = sorted(d.glob(pat))
                        if matches:
                            cover = str(matches[0])
                            break
                if cover:
                    break
        kwargs = {
            "title": self.epub_title,
            "author": self.epub_author,
            "language": self.epub_language,
            "identifier": self.epub_identifier,
            "description": self.epub_description,
            "publisher": self.epub_publisher,
            "cover": cover,
            "dry_run": self.dry_run,
            "write_manifest": write_manifest,
        }
        return positional, kwargs

    _BUILDERS: ClassVar[dict[str, Any]] = {
        "convert": _call_convert,
        "split": _call_split,
        "dictionary": _call_dictionary,
        "glossary": _call_glossary,
        "preprocess": _call_preprocess,
        "chunk": _call_chunk,
        "chunk_glossary": _call_chunk_glossary,
        "matter": _call_matter,
        "translate": _call_translate,
        "concatenate": _call_concatenate,
        "postprocess": _call_postprocess,
        "bind": _call_bind,
        "epub": _call_epub,
    }

    # -- the call ---------------------------------------------------------- #
    def call(
        self,
        stage: str,
        runtime: StageRuntime,
        *,
        mapping: Sequence[Mapping[str, Any]] | None = None,
    ) -> StageCall:
        """Build the exact positional arguments and kwargs for one runner."""
        paths = runtime.paths
        if stage == "split_map":
            if mapping is None:
                raise ToolError("split_map needs the parsed --map payload")
            runner = resolve_runner("split_map")
            positional = (
                self._source(runtime),
                paths.chapters_dir,
                mapping,
            )
            kwargs = {
                "min_chapters": self.min_chapters,
                "include_front_matter": self.split_include_front_matter,
                "dry_run": self.dry_run,
                "skip_if_present": self.skip_if_present,
            }
            return StageCall("split_map", runner, positional, kwargs)

        builder = self._BUILDERS.get(stage)
        if builder is None:
            raise ToolError(f"no runner call is defined for stage {stage!r}")

        runner = resolve_runner(stage)
        write_manifest = not self.no_manifest and not self.dry_run
        write_prompts = not self.no_prompts and not self.dry_run
        chapter_limit = self.chapter_limit(runtime)

        positional, kwargs = builder(self, runtime, write_manifest, write_prompts, chapter_limit)
        return StageCall(stage, runner, positional, kwargs)


    def _source(self, runtime: StageRuntime) -> Path:
        source = runtime.source or runtime.paths.source or runtime.paths.book
        if source is None:
            raise ToolError("no source book was resolved")
        return Path(source)

    def _original_source(self, runtime: StageRuntime) -> Path:
        source = (
            runtime.original_source
            or runtime.paths.original_source
            or runtime.paths.source
            or runtime.paths.book
        )
        if source is None:
            raise ToolError("no source book was resolved")
        return Path(source)


@dataclass(frozen=True)
class StageRuntime:
    """Everything a runner call needs that is not a CLI option.

    ``settings`` is resolved by the caller (lazily for the stages that support
    ``settings_factory``); ``progress``/``clock`` are the test and UI seams.
    """

    paths: StagePaths
    sources: StageSources
    settings: Any = None
    settings_factory: Callable[[], Any] | None = None
    progress: Any = None
    chunk_start: Any = None
    chapter_start: Any = None
    clock: Any = time.perf_counter
    chapter_limit: int | None = None
    chapter_files: Sequence[str] | None = None
    sentences_dir: str | None = None
    model_identity: str = ""
    source: Path | None = None
    original_source: Path | None = None
    conversion_target: Path | None = None
    conversion_live: bool = False


@dataclass(frozen=True)
class StageCall:
    """One fully-built runner call: ``runner(*positional, **kwargs)``."""

    stage: str
    runner: Callable[..., Any]
    positional: tuple[Any, ...]
    kwargs: dict[str, Any]

    def __call__(self) -> Any:
        return self.runner(*self.positional, **self.kwargs)


# --------------------------------------------------------------------------- #
# the whole pipeline's options
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PipelineOptions:
    """The normalized options of one whole-pipeline run."""

    book: str | None
    in_dir: Path
    output_root: Path
    start_at: str
    stop_after: str
    dry_run: bool
    json: bool
    quiet: bool
    debug: bool
    no_log: bool
    log_file: Path | None
    selection: PhaseSelection
    stage: StageOptions
    llm: list[str] = field(default_factory=list)
    llm_config: str | None = None
    force_convert: bool = False
    images: bool = True
    interactive: bool = False
    ns: Any = None

    @classmethod
    def from_namespace(cls, ns: Any) -> PipelineOptions:
        selection = PhaseSelection.from_namespace(ns)
        stage = StageOptions.from_namespace(ns, mode="pipeline")
        return cls(
            book=_get(ns, "book", None),
            in_dir=Path(_get(ns, "in_dir", None) or default_input_dir()).expanduser(),
            output_root=Path(
                _get(ns, "output_root", None) or default_output_dir()
            ).expanduser(),
            start_at=selection.start_at,
            stop_after=selection.stop_after,
            dry_run=bool(_get(ns, "dry_run", False)),
            json=bool(_get(ns, "json", False)),
            quiet=bool(_get(ns, "quiet", False)),
            debug=bool(_get(ns, "debug", False)),
            no_log=bool(_get(ns, "no_log", False)),
            log_file=Path(ns.log_file).expanduser()
            if _get(ns, "log_file", None)
            else None,
            selection=selection,
            stage=stage,
            llm=list(_get(ns, "llm", []) or []),
            llm_config=_get(ns, "llm_config", None),
            force_convert=bool(_get(ns, "force_convert", False)),
            images=bool(_get(ns, "images", True)),
            interactive=bool(_get(ns, "interactive", False)),
            ns=ns,
        )

    def validate(self) -> list[str]:
        """The usage errors a parser cannot catch; each line is one message."""
        errors: list[str] = []
        try:
            if PHASES.index(self.start_at) > PHASES.index(self.stop_after):
                errors.append(
                    f"--start-at {self.start_at} comes after --stop-after {self.stop_after}"
                )
        except ValueError:
            errors.append(f"unknown phase; choose one of {', '.join(PHASES)}")
        if int(self.stage.chunk_size) <= 0:
            errors.append("--chunk-size must be greater than 0 tokens")
        for flag, value in (
            ("--first-chapters", self.stage.first_chapters),
            ("--translate-first-chapters", self.stage.translate_first_chapters),
            ("--limit", self.stage.limit),
        ):
            if value is not None and int(value) < 0:
                errors.append(f"{flag} cannot be negative")
        if self.stage.translation_method not in TRANSLATION_METHODS:
            errors.append(
                f"--translate-method must be one of {', '.join(TRANSLATION_METHODS)}"
            )
        if self.stage.glossary_scope not in TRANSLATION_GLOSSARY_SCOPES:
            errors.append(
                f"--translate-glossary-scope must be one of "
                f"{', '.join(TRANSLATION_GLOSSARY_SCOPES)}"
            )
        try:
            parse_llm_overrides(self.llm, default_stage="split")
        except ConfigError as exc:
            errors.append(str(exc))
        return errors


__all__ = [
    "MATTER_ENABLED",
    "NEEDS_CHAPTERS",
    "PHASES",
    "PipelineOptions",
    "PhaseSelection",
    "StageCall",
    "StageOptions",
    "StageRuntime",
    "resolve_runner",
]
