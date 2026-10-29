"""Four LangGraph agents, a dictionary funnel, a glossary, a chunker, a per-chunk glossary, a front/back matter localizer, a translator, an assembler, a heading postprocessor, a binder and an EPUB packager.

The pipeline turns one long book into translation-ready material.

Stage 1 — :mod:`book_translator.stages.conversion` — turns a source EPUB, a text-based
PDF or a TXT into Markdown under ``01-markdown/``. EPUB/PDF go through docling
with OCR disabled, so an image-based PDF is rejected instead of OCR'd; a TXT is
copied under a ``.md`` name. Pictures are extracted to ``01-markdown/images/``
and referenced from the Markdown (``--no-images`` turns that off), so they reach
the final translated book. A Markdown source skips this stage entirely.

Stage 2 — :mod:`book_translator.stages.agent` — splits one long Markdown book into one
file per chapter **without reading its body**. Design contract:

* The LLM never reads the body of the book. It only ever sees line numbers and
  short candidate/header strings produced by deterministic tools.
* Every byte of the output files is copied verbatim from the source file by the
  slicer; the LLM only decides *where* to cut.
* Documents live in dedicated folders: sources in ``data/input/``, every
  generated artifact under ``data/output/<book>/`` (see :mod:`book_translator.paths`).

Stage 3 — :mod:`book_translator.stages.preprocess_agent` — repairs each chapter file,
**one chapter at a time**, with its own specialised agent and toolset:

* It starts from the chapter's *headers*, not from its prose: page numbers,
  figure captions marked up as headings, and headings the PDF conversion split
  across two lines are detected by shape and confirmed by the agent.
* It never edits text; it submits a plan in a fixed vocabulary
  (``keep`` / ``promote`` / ``demote`` / ``remove`` / ``merge_with_next`` plus a
  few named text fixes) and a deterministic validator checks each entry against
  the real file.
* A per-line provenance map proves that only the declared lines changed, and
  that the only lines deleted were page artefacts.

Stage 4 — :mod:`book_translator.stages.dictionary` and :mod:`book_translator.stages.glossary` —
extracts named entities and builds the book's locked glossary:
* Step 1 (:mod:`book_translator.stages.dictionary`) reduces every chapter to the raw
  sentences that carry a named entity, one chapter at a time. No LLM and no
  LangGraph here: a deterministic sentence funnel plus an optional NER tagger
  (CLASSLA for South Slavic, stanza for English) used purely as a gatekeeper.
* Step 2 (:mod:`book_translator.stages.glossary`) is the LLM pass that walks the
  first mentions first to last and grows **one book-wide dictionary**
  for a translation: for every person, place, organisation, artifact and recurring
  term of art it locks in a single rendering in the target language (Pydantic
  structured output, retries, token accounting), handing each chapter the
  dictionary the earlier ones produced so the renderings stay compatible. English
  into Serbian by default.

Stage 5 — :mod:`book_translator.stages.chunking` — cuts each chapter into
translation-sized chunks (1000 tokens by default) with chonkie's recursive
chunker, preferring paragraph over sentence over line over word boundaries. Every
chunk is an exact slice of its chapter, the manifest records its chapter and
global order, and concatenating the chunks recreates the chapter byte for byte.

Stage 6 — :mod:`book_translator.stages.chunk_glossary` — joins the two halves of the
translation material. It runs stage 4's NER tagger over **every** chunk first,
collecting the entity terms each chunk mentions; only then does it fuzzy-match
each term against stage 4's glossary (``rapidfuzz``, 95 % by default) and write
the relevant terms per chunk, under the chunk's own name in ``06-chunk-glossary/``.

Stage 7 — :mod:`book_translator.stages.matter_agent` — localizes the front and back
matter **before** translation, as its own LangGraph agent and **without the
dictionary**. A book-level plan searches the web for an existing published
translation of the title, fixes the translator credit (**BleBel**), lists the
original edition's imprint/ISBN/CIP material to omit, and decides the section
titles; then one agent run per matter chunk localizes it with that binding plan
and published Markdown formatting. The translated chunks land in the same
``08-translations/`` folder stage 8 uses, so stage 9 needs no new join rule; stage 8
skips those chapters. The plan, manifest and every prompt are kept in ``07-matter/``
and ``07-matter-prompts/``.

Stage 8 — :mod:`book_translator.stages.translation` — is the end product for the body. It
walks the chunks in reading order, skips the front/back matter stage 7 already
localized, and translates each body chunk into the target language (English into
Serbian by default), sending the glossary rows stage 6 attached and asking for a
Pydantic-validated ``{"translation": ...}`` answer, with retries and token
accounting. The translated chunks land in ``08-translations/`` under the chunk's own
name; the whole prompt of every call — system prompt, user prompt, messages,
glossary and source chunk — is kept in ``08-translation-prompts/`` for debugging.
``--limit`` translates only the first N chunks, and ``--resume`` skips the chunks
an earlier run already finished.

Stage 9 — :mod:`book_translator.stages.assembly` — puts the translated chunks back
together. Stage 5's manifest says which chapter each chunk belongs to and its
order inside that chapter; this stage concatenates each chapter's translated
chunks and writes ``09-translated-chapters/<chapter>.md`` in a folder separate from
the source ``02-chapters/``. A chapter is written only when every one of its chunks
was translated, so a partial translation is reported rather than silently
truncated. ``--first-chapters N`` assembles a prefix, and ``--skip-if-present``
reuses a chapter whose bytes already match the current translations.

Stage 10 — :mod:`book_translator.stages.postprocess_agent` — postprocesses the assembled
**translated** chapters, one chapter at a time, with a book-level naming pass:

* it first inventories every chapter title, chooses one naming policy and makes
  the chapter headings consistent;
* it inspects only each chapter's short opening and its heading records, then
  sets/inserts/keeps the chapter title and ``keep``/``demote``/``promote``s the
  other headings (a demotion removes the ``#`` markup, a promotion adds it back
  to a plain ALL-CAPS heading, and neither changes the text);
* it creates or updates the table of contents from the final titles, placing a
  missing one right after the front matter;
* the result lands in ``10-postprocessed-chapters/`` — the translated chapters are
  never edited in place.

Stage 11 — :mod:`book_translator.stages.binding` — concatenates the postprocessed
chapters, in reading order, into one ``11-translated-book.md``. It is deterministic:
the TOC and chapters are joined with one blank line and the result is verified.

Stage 12 — :mod:`book_translator.stages.epub` — packages the same chapter folder as an
**EPUB 3** book, ``12-translated-book.epub``, the artifact a reader actually
opens: one XHTML document per chapter, a navigation document built from the
chapters' own headings, the pictures the Markdown references embedded once under
``images/``, and the metadata stage 7 already recorded (title, author, target
language) plus an identifier that survives a rebuild. The printed table of
contents is deliberately not shipped — every reader builds its own from the
navigation document. The package is written from the chapter folder, not from
stage 11's single file, and it is verified after writing.

Package layout::

    book_translator/
        paths.py        where documents live (input/output folders)
        cli/            one module per command, shared helpers and root parser
                        (`convert` / `inspect` / `split` / `preprocess` /
                        `dictionary` / `glossary` / `chunk` / `chunk-glossary` /
                        `matter` / `translate` / `concatenate` / `postprocess` /
                        `openings` / `bind` / `epub` / `tools`)
        core/           constants, errors, LLM settings
        documents/      reading, markdown/bodyline shapes, chapter discovery,
                        the shared BookMetadata record
        pipeline/       orchestration, drivers, progress reporting
        stage_graph/    shared agent graph primitives
        toolkit/        stage tools and inspection helpers
        stages/         the pipeline stages:
            conversion/     stage 1: EPUB/PDF/TXT -> Markdown (docling; no OCR)
            agent/          stage 2 agent: prompts, model, tools, graph
            split/          stage 2 splitter exports
            slicing/        stage 2: planning, verbatim slicing, verification
            analysis/       TOC detection, header assessment, title location
            preprocess/     stage 3: header inspection, damage scan, and the
                            plan/apply/verify repair engine
            preprocess_agent/  stage 3 agent: prompts, model, tools binding, graph
            openings/       the final phase of stages 3 and 10: every chapter opening
                            audited, the provable leftovers (a stray number under the
                            title, a duplicated title, a page number) removed
            dictionary/     stage 4 (step 1): sentence funnel, language routing, NER gate
            glossary/       stage 4 (step 2): ordered LLM pass that builds the glossary
            chunking/       stage 5: reversible, sentence-safe chapter chunks
            chunk_glossary/ stage 6: per-chunk NER + fuzzy glossary retrieval
            matter/         stage 7: front/back matter discovery, plan validation,
                            publication formatting (no LLM)
            matter_agent/   stage 7 agent: book plan + per-chunk localization graphs
            translation/    stage 8: per-chunk structured translation + prompt record
            assembly/       stage 9: translated chunks concatenated into chapters
            postprocess/    stage 10: heading/title/TOC planning, rule-group
                            validation and application
            postprocess_agent/ stage 10 agent: book + per-chapter graphs
            binding/        stage 11: postprocessed chapters -> one Markdown book
            epub/           stage 12: postprocessed chapters -> one EPUB 3 book
                            (render, metadata, package, verification; optional
                            markdown-it-py, imported lazily)
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "3.0.0"

if TYPE_CHECKING:  # pragma: no cover - typing surface only
    from .paths import (
        default_input_dir,
        default_output_dir,
        resolve_book_path,
        resolve_output_dir,
    )
    from .stages.agent import SplitterResult, SplitterState, build_graph, run_split
    from .stages.assembly import AssemblyResult, run_assembly
    from .stages.binding import BindResult, run_bind
    from .stages.chunk_glossary import ChunkGlossaryResult, run_chunk_glossary
    from .stages.chunking import ChunkingResult, run_chunking
    from .stages.conversion import ConversionResult, run_conversion
    from .stages.dictionary import DictionaryResult, run_dictionary
    from .stages.epub import EpubResult, run_epub
    from .stages.glossary import GlossaryResult, run_glossary
    from .stages.matter_agent import (
        MatterChunkState,
        MatterPlanState,
        MatterResult,
        build_matter_chunk_graph,
        build_matter_graph,
        run_matter,
    )
    from .stages.openings import OpeningAuditResult, audit_openings
    from .stages.postprocess_agent import (
        ChapterPostprocessState,
        PostprocessResult,
        build_chapter_postprocess_graph,
        build_postprocess_graph,
        run_postprocess,
    )
    from .stages.preprocess_agent import (
        PreprocessResult,
        PreprocessState,
        build_preprocess_graph,
        run_preprocess,
    )
    from .stages.translation import TranslationResult, run_translation

__all__ = [
    "AssemblyResult",
    "BindResult",
    "ChunkGlossaryResult",
    "ChunkingResult",
    "ConversionResult",
    "DictionaryResult",
    "EpubResult",
    "GlossaryResult",
    "MatterChunkState",
    "MatterPlanState",
    "MatterResult",
    "OpeningAuditResult",
    "PostprocessResult",
    "ChapterPostprocessState",
    "PreprocessResult",
    "PreprocessState",
    "SplitterResult",
    "SplitterState",
    "TranslationResult",
    "build_chapter_postprocess_graph",
    "build_graph",
    "build_matter_chunk_graph",
    "build_matter_graph",
    "build_preprocess_graph",
    "build_postprocess_graph",
    "run_assembly",
    "audit_openings",
    "run_bind",
    "run_chunk_glossary",
    "run_chunking",
    "run_conversion",
    "run_postprocess",
    "default_input_dir",
    "default_output_dir",
    "resolve_book_path",
    "resolve_output_dir",
    "run_dictionary",
    "run_epub",
    "run_glossary",
    "run_matter",
    "run_preprocess",
    "run_split",
    "run_translation",
]

#: Where each public name lives. Importing this mapping is free; the modules are
#: only imported when the name is first read. That keeps a small import such as
#: ``book_translator.documents.text`` from dragging in every LangGraph stage (and
#: the OpenAI client) through the package's own ``__init__``.
_LAZY_GROUPS: dict[str, tuple[str, ...]] = {
    "book_translator.stages.agent": (
        "SplitterResult",
        "SplitterState",
        "build_graph",
        "run_split",
    ),
    "book_translator.stages.assembly": ("AssemblyResult", "run_assembly"),
    "book_translator.stages.binding": ("BindResult", "run_bind"),
    "book_translator.stages.chunk_glossary": ("ChunkGlossaryResult", "run_chunk_glossary"),
    "book_translator.stages.chunking": ("ChunkingResult", "run_chunking"),
    "book_translator.stages.conversion": ("ConversionResult", "run_conversion"),
    "book_translator.stages.dictionary": ("DictionaryResult", "run_dictionary"),
    "book_translator.stages.epub": ("EpubResult", "run_epub"),
    "book_translator.stages.glossary": ("GlossaryResult", "run_glossary"),
    "book_translator.stages.matter_agent": (
        "MatterChunkState",
        "MatterPlanState",
        "MatterResult",
        "build_matter_chunk_graph",
        "build_matter_graph",
        "run_matter",
    ),
    "book_translator.stages.openings": ("OpeningAuditResult", "audit_openings"),
    "book_translator.paths": (
        "default_input_dir",
        "default_output_dir",
        "resolve_book_path",
        "resolve_output_dir",
    ),
    "book_translator.stages.postprocess_agent": (
        "ChapterPostprocessState",
        "PostprocessResult",
        "build_chapter_postprocess_graph",
        "build_postprocess_graph",
        "run_postprocess",
    ),
    "book_translator.stages.preprocess_agent": (
        "PreprocessResult",
        "PreprocessState",
        "build_preprocess_graph",
        "run_preprocess",
    ),
    "book_translator.stages.translation": ("TranslationResult", "run_translation"),
}

_LAZY_EXPORTS: dict[str, str] = {
    name: module_name for module_name, names in _LAZY_GROUPS.items() for name in names
}


def __getattr__(name: str) -> Any:
    """Resolve one public name on first access (PEP 562)."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
