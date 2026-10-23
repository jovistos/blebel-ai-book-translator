"""Stage 6: attach the book glossary to every translation-sized chunk.

Stage 5 gave every chapter a flat set of chunks; stage 4 gave the book one locked
rendering per entity. This package joins them in two phases, and the order is the
contract:

1. :func:`~book_translator.stages.chunk_glossary.runner.run_chunk_glossary` runs NER over
   **every** chunk first, using stage 3's tagger
   (:mod:`book_translator.stages.dictionary.ner`). The result is a per-chunk list of
   distinct entity terms — no glossary lookup happens yet.
2. Only when the last chunk has been tagged does the retrieval phase start. Each
   term is scored against the glossary entries with ``rapidfuzz`` and the best
   candidate is accepted when it reaches the configured threshold (95 % by
   default, and a flag). The accepted terms are written next to the chunk, under
   the chunk's own name::

       05-chunks/00001-01-predgovor-0000.md
       06-chunk-glossary/00001-01-predgovor-0000.md.json

That file is what a translation prompt needs: the chunk's identity (global
order, chapter, character and line spans), every NER term with its match or near
miss, and a ``matched_terms`` list of ``term`` / ``translation`` / ``category``
rows. The book-level index is ``_chunk_glossary_manifest.json``; the full audit
trail is ``_chunk_glossary_trace.json``.

``rapidfuzz`` and the taggers are imported lazily, so a split-only install keeps
working and a missing dependency is one actionable line instead of a traceback.
"""

from .matching import (
    LexiconEntry,
    MatchCandidate,
    RapidFuzzMatcher,
    TermMatcher,
    clean_term,
    glossary_file,
    load_glossary,
    match_key,
    normalize_scorer,
    normalize_threshold,
)
from .runner import (
    ChunkGlossary,
    ChunkGlossaryResult,
    ChunkInput,
    EntityTerm,
    ProgressHook,
    TermMatch,
    aggregate_entities,
    chunk_glossary_filename,
    chunk_glossary_payload,
    collect_matched_terms,
    discover_chunk_inputs,
    load_chunk_glossary_manifest,
    match_term,
    resolve_chunks_dir,
    resolve_glossary_path,
    run_chunk_glossary,
    write_chunk_glossary,
    write_chunk_glossary_manifest,
    write_chunk_glossary_trace,
)

__all__ = [
    "ChunkGlossary",
    "ChunkGlossaryResult",
    "ChunkInput",
    "EntityTerm",
    "LexiconEntry",
    "MatchCandidate",
    "ProgressHook",
    "RapidFuzzMatcher",
    "TermMatch",
    "TermMatcher",
    "aggregate_entities",
    "chunk_glossary_filename",
    "chunk_glossary_payload",
    "clean_term",
    "collect_matched_terms",
    "discover_chunk_inputs",
    "glossary_file",
    "load_chunk_glossary_manifest",
    "load_glossary",
    "match_key",
    "match_term",
    "normalize_scorer",
    "normalize_threshold",
    "resolve_chunks_dir",
    "resolve_glossary_path",
    "run_chunk_glossary",
    "write_chunk_glossary",
    "write_chunk_glossary_manifest",
    "write_chunk_glossary_trace",
]
