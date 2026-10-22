"""Stage-4 entry point: turn the reduced chapters into one book-wide glossary.

Stage 3 left a folder of first mentions — ``03-reduced-unique/`` — one sentence per
noun, in reading order, and stopped there on purpose. This is the pass that folder
was built for.

The run walks the **body** chapters from first to last, one model call per
chapter, and grows a single dictionary:

```
03-reduced-unique/01-chapter-1.md   -> call 1: the dictionary is empty, the model
                                    finds the entities and fixes a rendering
                                    for each
03-reduced-unique/02-chapter-2.md   -> call 2: the prompt now carries the whole
                                    dictionary so far, and the model is told
                                    only to add what is new
...
03-reduced-unique/41-chapter-41.md  -> call 41
                                 -> 04-glossary/glossary.json + glossary.md
```

That order is the whole design. A dictionary built chapter by chapter *with the
earlier decisions in the prompt* is what makes the renderings compatible: the
model that names a place in chapter 30 can see how the place was named in chapter
2, and it is instructed to treat those as final. A dictionary built in one pass
over the whole book would be cheaper and would drift exactly where it matters.

Front matter and back matter are **skipped by default**: the copyright page, the
table of contents and the author's note carry names (a publisher, a city, an
acknowledgement) that must not be locked into the book's terminology. The stage-1
manifest records each file's ``kind``, so the decision is read rather than guessed
— with a filename fallback (``00-front-matter.md``, ``*-back-matter.md``) for a
folder whose manifest is missing. ``include_front_matter`` /
``include_back_matter`` put either block back.

Every chapter after the first sees the dictionary *as it stands*, and the calls
are counted: ``result.tokens`` is the sum of every attempt of every chapter, which
is the number the run reports last.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ..core.constants import (
    DEFAULT_GLOSSARY_ATTEMPTS,
    DEFAULT_GLOSSARY_CHUNK_CHARS,
    DEFAULT_SOURCE_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
)
from .artifacts import (
    chunk_sentences,
    resolve_sentences_dir,
)
from .book import GlossaryBook, MergeOutcome
from .execute import execute_glossary
from .extract import GlossaryExtractor, make_extractor
from .records import (
    ChapterGlossary,
    ChapterPlan,
    ChapterStartHook,
    Clock,
    GlossaryResult,
    GlossaryTerm,
    ProgressHook,
    chapter_kind,
    select_chapters,
)


# the run
# --------------------------------------------------------------------------- #
def run_glossary(
    chapters_path: str,
    output_dir: str | None = None,
    *,
    sentences_dir: str | None = None,
    sentences: str = "unique",
    source_language: str = DEFAULT_SOURCE_LANGUAGE,
    target_language: str = DEFAULT_TARGET_LANGUAGE,
    book_title: str = "",
    extractor: GlossaryExtractor | None = None,
    settings: Any = None,
    settings_factory: Callable[[], Any] | None = None,
    max_attempts: int = DEFAULT_GLOSSARY_ATTEMPTS,
    method: str = "auto",
    max_tokens: int | None = None,
    chunk_chars: int = DEFAULT_GLOSSARY_CHUNK_CHARS,
    max_context_entries: int = 0,
    include_front_matter: bool = False,
    include_back_matter: bool = False,
    skip_if_present: bool = False,
    save: bool = True,
    dry_run: bool = False,
    limit: int | None = None,
    progress: ProgressHook | None = None,
    chapter_start: ChapterStartHook | None = None,
    clock: Clock = time.perf_counter,
) -> GlossaryResult:
    """Build the glossary of one book, one chapter at a time, first to last.

    ``sentences="unique"`` reads the first-mention reduction (the default, and the
    cheap one); ``"reduced"`` reads every entity-bearing sentence instead. An
    explicit ``sentences_dir`` overrides both. ``extractor`` is injected by the
    tests and by anything that wants a different model; when it is ``None`` the
    chat model is built from ``settings``. ``settings_factory`` defers resolving
    those settings until a chapter actually needs a model, which is what lets a
    fully resumed run finish without a provider.

    ``skip_if_present`` resumes: the dictionary and the cost of the chapters
    already recorded in ``_glossary_manifest.json`` are loaded, and the run starts
    at the first body chapter that is missing from it. That is what makes a run
    that died on chapter 30 cost the price of chapters 30..N, not of the book. A
    manifest that names a different chapter folder (raw ``01-chapters/`` vs cleaned
    ``02-preprocessed/``) is not resumed: the filenames match but the text does not.

    ``dry_run`` still calls the model — the answer *is* the artifact, so there is
    nothing to validate without asking — but writes no file.
    """
    return execute_glossary(
        chapters_path,
        output_dir,
        sentences_dir=sentences_dir,
        sentences=sentences,
        source_language=source_language,
        target_language=target_language,
        book_title=book_title,
        extractor=extractor,
        settings=settings,
        settings_factory=settings_factory,
        max_attempts=max_attempts,
        method=method,
        max_tokens=max_tokens,
        chunk_chars=chunk_chars,
        max_context_entries=max_context_entries,
        include_front_matter=include_front_matter,
        include_back_matter=include_back_matter,
        skip_if_present=skip_if_present,
        save=save,
        dry_run=dry_run,
        limit=limit,
        progress=progress,
        chapter_start=chapter_start,
        clock=clock,
        extractor_factory=make_extractor,
    )


__all__ = [
    "ChapterGlossary",
    "ChapterPlan",
    "ChapterStartHook",
    "GlossaryBook",
    "GlossaryResult",
    "GlossaryTerm",
    "MergeOutcome",
    "ProgressHook",
    "chapter_kind",
    "chunk_sentences",
    "resolve_sentences_dir",
    "run_glossary",
    "select_chapters",
]
