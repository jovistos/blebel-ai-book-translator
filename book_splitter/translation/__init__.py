"""Stage 7: translate the chunks into the target language, one call at a time.

The first six stages end with translation material: chapters cut into
translation-sized chunks, a book-wide glossary of locked renderings, and one
per-chunk list of the glossary rows each chunk mentions. This package is the
stage those artifacts were built for. It walks the chunks in reading order, sends
each one with its glossary rows, validates a structured answer with Pydantic,
and writes the translated chunk plus a record of the *whole* model input.

Why one call per chunk rather than one for the chapter:

* **Context.** A chunk is sized to fit a translation prompt with room for the
  glossary and the answer; a chapter is not.
* **Recoverability.** A failed chunk is a retry and then one visible hole, not a
  lost chapter. The chunks before it are already on disk.
* **Reviewability.** ``translation-prompts/<chunk>.json`` holds the exact system
  prompt, user prompt, messages, glossary rows and source chunk of every call, so
  one bad paragraph can be debugged without reconstructing the request.

The pieces:

:mod:`~book_splitter.translation.models`
    the Pydantic shape of the answer: ``{"translation": "..."}``;
:mod:`~book_splitter.translation.prompts`
    the constant system prompt and the per-chunk prompt (glossary first, then the
    previous translation's tail for continuity, then the source chunk);
:mod:`~book_splitter.translation.translator`
    one model call, native structured output with a JSON-mode fallback, retries
    with the failure named, token accounting;
:mod:`~book_splitter.translation.runner`
    the ordered run over the chunks, the ``--limit`` and resume rules, and the
    files that come out: ``translations/<chunk>.md``,
    ``translation-prompts/<chunk>.md.json`` and ``<chunk>.md.prompt.md``,
    ``_translation_manifest.json`` and ``_translation_trace.json``.
"""

from .models import ChunkTranslationAnswer
from .prompts import (
    build_chunk_prompt,
    build_repair_note,
    build_system_prompt,
    render_glossary,
    target_style_note,
)
from .runner import (
    ChunkStartHook,
    ChunkTranslation,
    GlossaryRow,
    ProgressHook,
    TranslationResult,
    debug_filename,
    has_chunk_glossary_files,
    load_book_glossary_rows,
    load_chunk_glossary_rows,
    load_translation_manifest,
    preserve_trailing_newlines,
    prompt_filename,
    resolve_chunk_glossary_dir,
    run_translation,
    write_translation_manifest,
    write_translation_trace,
)
from .translator import (
    METHODS,
    ChunkTranslator,
    LangChainTranslator,
    TranslationOutcome,
    clean_translation,
    make_translator,
    translation_problem,
)

__all__ = [
    "METHODS",
    "ChunkStartHook",
    "ChunkTranslation",
    "ChunkTranslationAnswer",
    "ChunkTranslator",
    "GlossaryRow",
    "LangChainTranslator",
    "ProgressHook",
    "TranslationOutcome",
    "TranslationResult",
    "build_chunk_prompt",
    "build_repair_note",
    "build_system_prompt",
    "clean_translation",
    "debug_filename",
    "has_chunk_glossary_files",
    "load_book_glossary_rows",
    "load_chunk_glossary_rows",
    "load_translation_manifest",
    "make_translator",
    "preserve_trailing_newlines",
    "prompt_filename",
    "render_glossary",
    "resolve_chunk_glossary_dir",
    "run_translation",
    "target_style_note",
    "translation_problem",
    "write_translation_manifest",
    "write_translation_trace",
]
