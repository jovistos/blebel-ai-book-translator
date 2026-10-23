"""Stage 4: build the book-wide glossary the translation is bound to.

Stage 3 reduced every chapter to its first mentions and stopped there. This
package is the LLM pass that folder was built for: it walks the body chapters in
reading order and grows **one dictionary** for the whole book, asking the model
about one chapter at a time and handing it everything already decided.

Why one call per chapter rather than one call for the book:

* **Compatibility.** Every call after the first sees the renderings locked in by
  the earlier chapters and is told they are final, so the name a chapter 30
  sentence uses is the name chapter 2 chose. A single pass over a whole book
  cannot make that promise — it is one chance to be consistent, with nothing to
  be consistent *with*.
* **Context.** A chapter's first mentions are a few kilobytes; a book's are not.
  Chapter-sized calls keep the evidence inside the window and the failures small:
  a chapter that fails validation is retried (and, if it still fails, reported)
  without costing the rest of the book.

The pieces:

:mod:`~book_translator.glossary.models`
    the Pydantic shape of the answer — ``term``, ``category``, ``translation``;
:mod:`~book_translator.glossary.prompts`
    the prompts, which state that a whole book is being translated and that the
    renderings produced here are the ones the translator must reuse;
:mod:`~book_translator.glossary.extract`
    one model call, structured output, retries, token accounting;
:mod:`~book_translator.glossary.book`
    the live dictionary and its merge rules (first rendering wins; a later
    disagreement is a recorded conflict);
:mod:`~book_translator.glossary.records`
    the four records the run writes, and the front/back-matter chapter
    selection;
:mod:`~book_translator.glossary.runner`
    the ordered run over the chapters, the front/back-matter skip, and the
    files that come out:
    ``glossary.json``, ``glossary.md``, ``_glossary_manifest.json`` and
    ``_glossary_trace.json``.
"""

from .extract import (
    ExtractionOutcome,
    GlossaryExtractor,
    LangChainExtractor,
    TokenUsage,
    make_extractor,
)
from .models import (
    ChapterExtraction,
    GlossaryEntry,
    TermCategory,
    entry_key,
    normalize_category,
)
from .prompts import build_chapter_prompt, build_system_prompt, render_dictionary
from .runner import (
    ChapterGlossary,
    ChapterPlan,
    ChapterStartHook,
    GlossaryBook,
    GlossaryResult,
    GlossaryTerm,
    MergeOutcome,
    ProgressHook,
    chapter_kind,
    chunk_sentences,
    resolve_sentences_dir,
    run_glossary,
    select_chapters,
)

__all__ = [
    "ChapterExtraction",
    "ChapterGlossary",
    "ChapterStartHook",
    "ChapterPlan",
    "ExtractionOutcome",
    "GlossaryBook",
    "GlossaryEntry",
    "GlossaryExtractor",
    "GlossaryResult",
    "GlossaryTerm",
    "LangChainExtractor",
    "MergeOutcome",
    "ProgressHook",
    "TermCategory",
    "TokenUsage",
    "build_chapter_prompt",
    "build_system_prompt",
    "chapter_kind",
    "chunk_sentences",
    "entry_key",
    "make_extractor",
    "normalize_category",
    "render_dictionary",
    "resolve_sentences_dir",
    "run_glossary",
    "select_chapters",
]
