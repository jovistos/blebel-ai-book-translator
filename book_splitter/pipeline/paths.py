"""One value for every folder a run reads or writes.

The pipeline used to carry twenty path attributes built in ``Pipeline.__init__``,
while each stage command rebuilt the same layout from ``--out`` and its own
``--*-out`` flags. Both surfaces now start from :class:`StagePaths`:

``for_pipeline``
    A source book plus ``--output-root`` (and the optional ``--out`` /
    ``--*-out`` overrides) produce the canonical book layout.

``from_base``
    A stage command hands in the folder it is anchored to (the chapters folder,
    the chunks folder, the translated-chapters folder, …) and gets the same
    layout, rooted at that folder's book folder. This preserves the long-standing
    stage-command behaviour where ``chunk --out /tmp/book/01-chapters`` writes
    ``/tmp/book/05-chunks``.

The folder names themselves stay in :mod:`book_splitter.documents.layout`; this
module only applies the pipeline's override rules. Nothing here creates a
folder.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..core.coerce import as_path as _path
from ..core.constants import CHAPTERS_DIRNAME, MARKDOWN_DIRNAME
from ..documents.chapters import (
    chapters_dir,
    chunk_glossary_dir,
    chunks_dir,
    glossary_dir,
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
from ..documents.layout import book_root, resolve_stage_dir
from ..paths import (
    book_output_dir,
    default_input_dir,
    default_output_dir,
    resolve_book_path,
    resolve_output_dir,
)

#: StagePaths field -> the argparse destination of its long flag. ``--out`` is
#: the chapters anchor, not a separate folder override. A field missing here has
#: no command-line flag (``markdown_dir`` is always the book folder).
OVERRIDE_FLAGS: dict[str, str] = {
    "chapters_dir": "out",
    "preprocessed_dir": "preprocessed_out",
    "reduced_dir": "reduced_out",
    "unique_dir": "unique_out",
    "glossary_dir": "glossary_out",
    "chunks_dir": "chunks_out",
    "chunk_glossary_dir": "chunk_glossary_out",
    "translations_dir": "translations_out",
    "translation_prompts_dir": "translation_prompts_out",
    "matter_dir": "matter_out",
    "matter_prompts_dir": "matter_prompts_out",
    "translated_chapters_dir": "translated_chapters_out",
    "postprocess_dir": "postprocess_out",
    "translated_book_path": "book_out",
    "epub_path": "epub_out",
}

#: Every field an override may target.
OVERRIDE_KEYS = ("markdown_dir", *OVERRIDE_FLAGS)


def _override(overrides: Mapping[str, Any], key: str) -> Path | None:
    return _path(overrides.get(key))


@dataclass(frozen=True)
class StagePaths:
    """Where a pipeline run (or one stage command) reads and writes.

    The field names match the long flags and the JSON report. ``base`` is the
    chapters folder the rest of the layout is anchored to; ``book_dir`` is the
    book folder that owns the numbered stage folders.
    """

    input_dir: Path
    output_root: Path
    book_dir: Path
    base: Path
    markdown_dir: Path
    chapters_dir: Path
    preprocessed_dir: Path
    reduced_dir: Path
    unique_dir: Path
    glossary_dir: Path
    chunks_dir: Path
    chunk_glossary_dir: Path
    translations_dir: Path
    translation_prompts_dir: Path
    matter_dir: Path
    matter_prompts_dir: Path
    translated_chapters_dir: Path
    postprocess_dir: Path
    translated_book_path: Path
    epub_path: Path
    book: Path | None = None
    source: Path | None = None
    original_source: Path | None = None

    # -- constructors ------------------------------------------------------ #
    @classmethod
    def for_pipeline(
        cls,
        ns: Any,
        *,
        source: Path,
        original_source: Path | None = None,
        input_dir: Path | None = None,
        output_root: Path | None = None,
    ) -> StagePaths:
        """The canonical layout for a whole-pipeline run.

        ``source`` is the Markdown stage 1 reads (the original source for a
        Markdown book, the converted file for an EPUB/PDF/TXT). ``--out`` moves
        the chapters anchor; every ``--*-out`` moves one folder; everything else
        follows the book folder under ``--output-root``.
        """
        book = Path(source).expanduser()
        in_dir = Path(input_dir).expanduser() if input_dir else Path("data/input")
        root = Path(output_root).expanduser() if output_root else default_output_dir()
        book_dir = book_output_dir(book, output_root=root)
        explicit_base = _path(getattr(ns, "out", None))
        base = (
            chapters_dir(explicit_base)
            if explicit_base is not None
            else book_dir / CHAPTERS_DIRNAME
        )
        overrides = {
            field: getattr(ns, flag, None) for field, flag in OVERRIDE_FLAGS.items()
        }
        # `--out` is the chapters anchor, not a separate override.
        overrides["chapters_dir"] = base
        return cls.from_base(
            base,
            book=book,
            source=book,
            original_source=original_source or book,
            input_dir=in_dir,
            output_root=root,
            overrides=overrides,
            book_dir=book_dir,
            markdown_dir=overrides.get("markdown_dir") or book_dir / MARKDOWN_DIRNAME,
        )

    @classmethod
    def for_command(
        cls,
        ns: Any,
        *,
        book: Path | None = None,
        input_dir: Path | None = None,
        output_root: Path | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> StagePaths:
        """The layout for one standalone stage command.

        A command anchors on ``--out`` when it is given, on a positional book
        folder, or on the default output folder of the named book — the same
        order :func:`book_splitter.paths.resolve_output_dir` has always used.
        Every ``--*-out`` flag then overrides its own folder, and the resulting
        value is the one both the command and the shared source resolvers read.
        """
        explicit_out = _path(getattr(ns, "out", None))
        book_arg = (
            Path(book).expanduser()
            if book is not None
            else _path(getattr(ns, "book", None))
        )
        in_dir = (
            Path(input_dir).expanduser()
            if input_dir is not None
            else _path(getattr(ns, "in_dir", None))
        )
        root = (
            Path(output_root).expanduser()
            if output_root is not None
            else _path(getattr(ns, "output_root", None))
        )
        source: Path | None
        if explicit_out is not None:
            base = chapters_dir(explicit_out)
            source = None
        elif book_arg is not None and book_arg.is_dir():
            base = chapters_dir(book_arg)
            source = None
        else:
            source = resolve_book_path(book_arg, input_dir=in_dir)
            base = chapters_dir(resolve_output_dir(source, None, output_root=root))

        merged = {
            field: getattr(ns, flag, None) for field, flag in OVERRIDE_FLAGS.items()
        }
        if overrides:
            merged.update(overrides)
        # ``--out`` is the chapters anchor, not a separate override.
        merged["chapters_dir"] = base
        return cls.from_base(
            base,
            book=book_arg if book_arg is not None else source,
            source=source,
            input_dir=in_dir,
            output_root=root,
            overrides=merged,
        )

    @classmethod
    def from_base(
        cls,
        base: Path | str,
        *,
        book: Path | None = None,
        source: Path | None = None,
        original_source: Path | None = None,
        input_dir: Path | None = None,
        output_root: Path | None = None,
        overrides: Mapping[str, Any] | None = None,
        book_dir: Path | None = None,
        markdown_dir: Path | None = None,
    ) -> StagePaths:
        """The layout for a stage command anchored at ``base``.

        ``base`` may be any stage folder (or the book folder): the existing
        :func:`resolve_stage_dir` rule steps to the book folder and appends the
        requested stage's name. Explicit ``overrides`` win.
        """
        overrides = overrides or {}
        anchor = Path(base).expanduser()
        root = (
            Path(output_root).expanduser()
            if output_root
            else default_output_dir()
        )
        resolved_book_dir = (
            Path(book_dir).expanduser() if book_dir is not None else book_root(anchor)
        )
        resolved_markdown = (
            Path(markdown_dir).expanduser()
            if markdown_dir is not None
            else resolved_book_dir / MARKDOWN_DIRNAME
        )

        resolved_chapters = _override(overrides, "chapters_dir") or resolve_stage_dir(
            anchor, "chapters"
        )
        resolved_preprocessed = _override(
            overrides, "preprocessed_dir"
        ) or preprocessed_dir(anchor)
        resolved_reduced = _override(overrides, "reduced_dir") or reduced_dir(anchor)
        resolved_unique = _override(overrides, "unique_dir") or unique_dir(
            resolved_reduced
        )
        resolved_glossary = _override(overrides, "glossary_dir") or glossary_dir(anchor)
        resolved_chunks = _override(overrides, "chunks_dir") or chunks_dir(anchor)
        resolved_chunk_glossary = _override(
            overrides, "chunk_glossary_dir"
        ) or chunk_glossary_dir(resolved_chunks)
        resolved_translations = _override(
            overrides, "translations_dir"
        ) or translations_dir(resolved_chunks)
        resolved_prompts = _override(
            overrides, "translation_prompts_dir"
        ) or translation_prompts_dir(resolved_chunks)
        resolved_matter = _override(overrides, "matter_dir") or matter_dir(
            resolved_chunks
        )
        resolved_matter_prompts = _override(
            overrides, "matter_prompts_dir"
        ) or matter_prompts_dir(resolved_chunks)
        resolved_translated_chapters = _override(
            overrides, "translated_chapters_dir"
        ) or translated_chapters_dir(resolved_chunks)
        resolved_postprocess = _override(
            overrides, "postprocess_dir"
        ) or postprocessed_chapters_dir(resolved_translated_chapters)
        resolved_book_file = _override(
            overrides, "translated_book_path"
        ) or translated_book_path(resolved_postprocess)
        resolved_epub = _override(overrides, "epub_path") or translated_epub_path(
            resolved_postprocess
        )

        return cls(
            input_dir=Path(input_dir).expanduser()
            if input_dir
            else default_input_dir(),
            output_root=root,
            book_dir=resolved_book_dir,
            base=anchor,
            markdown_dir=resolved_markdown,
            chapters_dir=resolved_chapters,
            preprocessed_dir=resolved_preprocessed,
            reduced_dir=resolved_reduced,
            unique_dir=resolved_unique,
            glossary_dir=resolved_glossary,
            chunks_dir=resolved_chunks,
            chunk_glossary_dir=resolved_chunk_glossary,
            translations_dir=resolved_translations,
            translation_prompts_dir=resolved_prompts,
            matter_dir=resolved_matter,
            matter_prompts_dir=resolved_matter_prompts,
            translated_chapters_dir=resolved_translated_chapters,
            postprocess_dir=resolved_postprocess,
            translated_book_path=resolved_book_file,
            epub_path=resolved_epub,
            book=Path(book).expanduser() if book else None,
            source=Path(source).expanduser() if source else None,
            original_source=Path(original_source).expanduser()
            if original_source
            else None,
        )

    # -- transitions ------------------------------------------------------- #
    def with_source(self, source: Path | str) -> StagePaths:
        """Swap the readable Markdown source after stage 0, keeping the layout."""
        return replace(self, source=Path(source).expanduser())

    def with_overrides(self, **overrides: Any) -> StagePaths:
        """Return a copy with extra folder overrides (mostly for tests/tools)."""
        return self.from_base(
            self.base,
            book=self.book,
            source=self.source,
            original_source=self.original_source,
            input_dir=self.input_dir,
            output_root=self.output_root,
            overrides={**self.to_dict_paths(), **overrides},
            book_dir=self.book_dir,
            markdown_dir=self.markdown_dir,
        )

    # -- reporting --------------------------------------------------------- #
    def to_dict(self) -> dict[str, str | None]:
        """The JSON ``paths`` block. Paths are strings, missing books are null."""
        return {
            "input_dir": str(self.input_dir),
            "output_root": str(self.output_root),
            "book_dir": str(self.book_dir),
            "base": str(self.base),
            "book": str(self.book) if self.book else None,
            "source": str(self.source) if self.source else None,
            "original_source": str(self.original_source)
            if self.original_source
            else None,
            "markdown_dir": str(self.markdown_dir),
            "chapters_dir": str(self.chapters_dir),
            "preprocessed_dir": str(self.preprocessed_dir),
            "reduced_dir": str(self.reduced_dir),
            "unique_dir": str(self.unique_dir),
            "glossary_dir": str(self.glossary_dir),
            "chunks_dir": str(self.chunks_dir),
            "chunk_glossary_dir": str(self.chunk_glossary_dir),
            "translations_dir": str(self.translations_dir),
            "translation_prompts_dir": str(self.translation_prompts_dir),
            "matter_dir": str(self.matter_dir),
            "matter_prompts_dir": str(self.matter_prompts_dir),
            "translated_chapters_dir": str(self.translated_chapters_dir),
            "postprocess_dir": str(self.postprocess_dir),
            "translated_book_path": str(self.translated_book_path),
            "epub_path": str(self.epub_path),
        }

    def to_dict_paths(self) -> dict[str, Path]:
        """The folder fields as paths, for an override round-trip."""
        return {
            "markdown_dir": self.markdown_dir,
            "chapters_dir": self.chapters_dir,
            "preprocessed_dir": self.preprocessed_dir,
            "reduced_dir": self.reduced_dir,
            "unique_dir": self.unique_dir,
            "glossary_dir": self.glossary_dir,
            "chunks_dir": self.chunks_dir,
            "chunk_glossary_dir": self.chunk_glossary_dir,
            "translations_dir": self.translations_dir,
            "translation_prompts_dir": self.translation_prompts_dir,
            "matter_dir": self.matter_dir,
            "matter_prompts_dir": self.matter_prompts_dir,
            "translated_chapters_dir": self.translated_chapters_dir,
            "postprocess_dir": self.postprocess_dir,
            "translated_book_path": self.translated_book_path,
            "epub_path": self.epub_path,
        }

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return f"StagePaths(book_dir={self.book_dir}, base={self.base})"


__all__ = ["OVERRIDE_KEYS", "StagePaths"]
