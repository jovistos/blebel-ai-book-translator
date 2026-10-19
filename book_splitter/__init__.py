"""Two LangGraph agents, a dictionary funnel, a glossary, a chunker, a per-chunk glossary, a translator and an assembler.

The pipeline turns one long Markdown book into translation-ready material.

Stage 1 — :mod:`book_splitter.agent` — splits one long Markdown book into one
file per chapter **without reading its body**. Design contract:

* The LLM never reads the body of the book. It only ever sees line numbers and
  short candidate/header strings produced by deterministic tools.
* Every byte of the output files is copied verbatim from the source file by the
  slicer; the LLM only decides *where* to cut.
* Documents live in dedicated folders: sources in ``data/input/``, every
  generated artifact under ``data/output/<book>/`` (see :mod:`book_splitter.paths`).

Stage 2 — :mod:`book_splitter.preprocess_agent` — repairs each chapter file,
**one chapter at a time**, with its own specialised agent and toolset:

* It starts from the chapter's *headers*, not from its prose: page numbers,
  figure captions marked up as headings, and headings the PDF conversion split
  across two lines are detected by shape and confirmed by the agent.
* It never edits text; it submits a plan in a fixed vocabulary
  (``keep`` / ``demote`` / ``remove`` / ``merge_with_next`` plus a few named text
  fixes) and a deterministic validator checks each entry against the real file.
* A per-line provenance map proves that only the declared lines changed, and
  that the only lines deleted were page artefacts.

Stage 3 — :mod:`book_splitter.dictionary` — reduces every chapter to the raw
sentences that carry a named entity, one chapter at a time. No LLM and no
LangGraph here: a deterministic sentence funnel plus an optional NER tagger
(CLASSLA for South Slavic, stanza for English) used purely as a gatekeeper. The
tagger only filters; it cannot recognise an invented term, so it never builds the
dictionary.

Stage 4 — :mod:`book_splitter.glossary` — is the LLM pass stage 3 was built for.
It walks the first mentions first to last and grows **one book-wide dictionary**
for a translation: for every person, place, organisation, artifact and recurring
term of art it locks in a single rendering in the target language (Pydantic
structured output, retries, token accounting), handing each chapter the
dictionary the earlier ones produced so the renderings stay compatible. English
into Serbian by default.

Stage 5 — :mod:`book_splitter.chunking` — cuts each chapter into
translation-sized chunks (1000 tokens by default) with chonkie's recursive
chunker, preferring paragraph over sentence over line over word boundaries. Every
chunk is an exact slice of its chapter, the manifest records its chapter and
global order, and concatenating the chunks recreates the chapter byte for byte.

Stage 6 — :mod:`book_splitter.chunk_glossary` — joins the two halves of the
translation material. It runs stage 3's NER tagger over **every** chunk first,
collecting the entity terms each chunk mentions; only then does it fuzzy-match
each term against stage 4's glossary (``rapidfuzz``, 95 % by default) and write
the relevant terms per chunk, under the chunk's own name in ``chunk-glossary/``.

Stage 7 — :mod:`book_splitter.translation` — is the end product. It walks the
chunks in reading order and translates each one into the target language
(English into Serbian by default), sending the glossary rows stage 6 attached
and asking for a Pydantic-validated ``{"translation": ...}`` answer, with retries
and token accounting. The translated chunks land in ``translations/`` under the
chunk's own name; the whole prompt of every call — system prompt, user prompt,
messages, glossary and source chunk — is kept in ``translation-prompts/`` for
debugging. ``--limit`` translates only the first N chunks, and ``--resume``
skips the chunks an earlier run already finished.

Stage 8 — :mod:`book_splitter.assembly` — puts the translated chunks back
together. Stage 5's manifest says which chapter each chunk belongs to and its
order inside that chapter; this stage concatenates each chapter's translated
chunks and writes ``translated-chapters/<chapter>.md`` in a folder separate from
the source ``chapters/``. A chapter is written only when every one of its chunks
was translated, so a partial translation is reported rather than silently
truncated. ``--first-chapters N`` assembles a prefix, and ``--skip-if-present``
reuses a chapter whose bytes already match the current translations.

Package layout::

    book_splitter/
        paths.py        where documents live (input/output folders)
        cli.py          `inspect` / `split` / `preprocess` / `dictionary` /
                        `glossary` / `chunk` / `chunk-glossary` / `translate` / `tools`
        core/           constants, errors, LLM settings
        documents/      reading, markdown/bodyline shapes, chapter discovery
        analysis/       TOC detection, header assessment, title location
        slicing/        stage 1: planning, verbatim slicing, verification
        preprocess/     stage 2: header inspection, damage scan, repair engine
        dictionary/     stage 3: sentence funnel, language routing, NER gate
        glossary/       stage 4: ordered LLM pass that builds the glossary
        chunking/       stage 5: reversible, sentence-safe chapter chunks
        chunk_glossary/ stage 6: per-chunk NER + fuzzy glossary retrieval
        translation/    stage 7: per-chunk structured translation + prompt record
        assembly/       stage 8: translated chunks concatenated into chapters
        agent/          stage 1 agent: prompts, model, tools, graph
        preprocess_agent/  stage 2 agent: prompts, model, tools binding, graph
"""

from .agent import SplitterResult, SplitterState, build_graph, run_split
from .assembly import AssemblyResult, run_assembly
from .chunk_glossary import ChunkGlossaryResult, run_chunk_glossary
from .chunking import ChunkingResult, run_chunking
from .dictionary import DictionaryResult, run_dictionary
from .glossary import GlossaryResult, run_glossary
from .paths import (
    default_input_dir,
    default_output_dir,
    resolve_book_path,
    resolve_output_dir,
)
from .preprocess_agent import (
    PreprocessResult,
    PreprocessState,
    build_preprocess_graph,
    run_preprocess,
)
from .translation import TranslationResult, run_translation

__all__ = [
    "AssemblyResult",
    "ChunkGlossaryResult",
    "ChunkingResult",
    "DictionaryResult",
    "GlossaryResult",
    "PreprocessResult",
    "PreprocessState",
    "SplitterResult",
    "SplitterState",
    "TranslationResult",
    "build_graph",
    "build_preprocess_graph",
    "run_assembly",
    "run_chunk_glossary",
    "run_chunking",
    "default_input_dir",
    "default_output_dir",
    "resolve_book_path",
    "resolve_output_dir",
    "run_dictionary",
    "run_glossary",
    "run_preprocess",
    "run_split",
    "run_translation",
]
__version__ = "2.5.0"
