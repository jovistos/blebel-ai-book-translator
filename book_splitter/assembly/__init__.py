"""Stage 8: concatenate the translated chunks back into chapter files.

The translation stage writes one file per chunk, because one model call per chunk
is what makes a long book recoverable. A reader, however, wants the book. This
stage joins the two views:

* stage 5's manifest says which chapter each chunk belongs to, and where it sits
  inside that chapter;
* ``translations/<chunk>.md`` holds the translated text of that chunk;
* :func:`~book_splitter.assembly.runner.run_assembly` concatenates each chapter's
  translated chunks, in chunk order, into
  ``translated-chapters/<chapter>.md``.

The output is a separate folder on purpose. ``chapters/`` is the source text and
must never be overwritten by a translation; ``translated-chapters/`` is the
product and can be deleted and rebuilt at any time.

The stage is resumable: ``--first-chapters N`` narrows it to a prefix, and
``--skip-if-present`` reuses a chapter file whose bytes already equal the current
concatenation. Growing ``N`` therefore assembles only the new chapters, never the
whole book again. A chapter whose chunks are only partially translated is
reported as incomplete and skipped rather than written as a truncated chapter.

:mod:`~book_splitter.assembly.runner` also writes
``_translated_chapters_manifest.json`` (one entry per chapter: chunk count,
character counts, hash, output path, reused/skipped state) and
``_translated_chapters_trace.json`` (the audit trail) next to the chapters.
"""

from .runner import (
    AssemblyResult,
    Clock,
    ProgressHook,
    TranslatedChapter,
    load_assembly_manifest,
    resolve_chunks_dir,
    resolve_output_dir,
    resolve_translations_dir,
    run_assembly,
    write_assembly_manifest,
    write_assembly_trace,
)

__all__ = [
    "AssemblyResult",
    "Clock",
    "ProgressHook",
    "TranslatedChapter",
    "load_assembly_manifest",
    "resolve_chunks_dir",
    "resolve_output_dir",
    "resolve_translations_dir",
    "run_assembly",
    "write_assembly_manifest",
    "write_assembly_trace",
]
