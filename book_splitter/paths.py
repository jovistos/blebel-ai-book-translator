"""Where documents live: one folder for inputs, one for outputs.

The project keeps generated documents away from the source code:

``data/input/``
    Source books (Markdown/TXT/PDF/EPUB) handed to the splitter. Read-only inputs.

``data/output/``
    Everything the splitter generates. Per book, the artifacts live under
    ``data/output/<book-stem>/``::

        data/output/Aleksandar_Loven_-_Ljubav_seks_i_vase_srce/
            00-markdown/my-book.md
            01-chapters/00-front-matter.md
            01-chapters/01-predgovor.md
            ...
            01-chapters/_manifest.json
            01-chapters/_agent_trace.json
            05-chunks/00001-chapter-1-0000.md
            06-chunk-glossary/00001-chapter-1-0000.md.json
            07-translations/00001-chapter-1-0000.md
            07-translation-prompts/00001-chapter-1-0000.md.json
            08-translated-chapters/01-chapter-1.md

Every artifact folder is prefixed with the number of the stage that writes it, so
a plain sort of a book's output folder follows the pipeline order.

No module may hard-code those paths: everything goes through this module so the
layout stays in one place.
"""

from __future__ import annotations

import re
from pathlib import Path

#: Directory holding the source documents. Never written to.
INPUT_DIRNAME = "data/input"
#: Directory holding every generated artifact. Never committed.
OUTPUT_DIRNAME = "data/output"
#: Sub-directory of an output folder that receives the chapter files.
#: Stage numbers keep a book folder sorted in pipeline order.
CHAPTERS_DIRNAME = "01-chapters"
#: Sub-directory that receives stage 0's converted Markdown.
MARKDOWN_DIRNAME = "00-markdown"
#: Sub-directory of an output folder that receives stage 3's reduced chapters.
REDUCED_DIRNAME = "03-reduced"
#: Sub-directory that receives stage 3's further-reduced chapters (first mentions).
UNIQUE_DIRNAME = "03-reduced-unique"
#: Sub-directory that receives stage 5's translation-sized chunks.
CHUNKS_DIRNAME = "05-chunks"

_MARKDOWN_SUFFIXES = (".md", ".markdown")
#: Plain text is already the content stage 0 writes; only the name changes.
_TEXT_SUFFIXES = (".txt",)
#: Formats docling converts to Markdown in stage 0.
_DOCLING_SUFFIXES = (".pdf", ".epub")
#: Every source format the pipeline accepts.
SOURCE_SUFFIXES = _MARKDOWN_SUFFIXES + _TEXT_SUFFIXES + _DOCLING_SUFFIXES


def repo_root(start: Path | str | None = None) -> Path:
    """The project root: a checkout, a book project, or the current directory.

    A source checkout is recognised by ``pyproject.toml`` next to the
    ``book_splitter/`` package; a book project is recognised by its
    ``book_splitter.toml`` runtime config. The package's own location is only an
    anchor, never proof of a project root: once the package is installed,
    ``site-packages`` contains ``book_splitter/`` but is not a project, so the
    current directory is the only sensible fallback there.
    """
    anchor = Path(start).expanduser().resolve() if start else Path(__file__).resolve()
    anchor_candidates = [anchor, *anchor.parents]
    cwd = Path.cwd().resolve()
    cwd_candidates = [cwd, *cwd.parents]

    def source_checkout(candidates: list[Path]) -> Path | None:
        for candidate in candidates:
            if (candidate / "pyproject.toml").is_file() and (
                candidate / "book_splitter"
            ).is_dir():
                return candidate
        return None

    def book_project(candidates: list[Path]) -> Path | None:
        for candidate in candidates:
            if (candidate / "book_splitter.toml").is_file():
                return candidate
        return None

    # The package's own ancestor chain wins for a source checkout, then for a
    # project whose config sits next to it. Only then does the working directory
    # get a say, which is the installed-package case.
    for search in (source_checkout, book_project):
        found = search(anchor_candidates)
        if found is not None:
            return found
    for search in (source_checkout, book_project):
        found = search(cwd_candidates)
        if found is not None:
            return found
    return cwd


def default_input_dir(root: Path | str | None = None) -> Path:
    """Absolute path of the shared input folder."""
    return (Path(root).expanduser() if root else repo_root()) / INPUT_DIRNAME


def default_output_dir(root: Path | str | None = None) -> Path:
    """Absolute path of the shared output folder."""
    return (Path(root).expanduser() if root else repo_root()) / OUTPUT_DIRNAME


def book_slug(path: Path | str) -> str:
    """A filesystem-safe identifier for a source document (its stem)."""
    stem = Path(path).name
    for suffix in SOURCE_SUFFIXES:
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._")
    return slug or "book"


def book_output_dir(
    book: Path | str,
    root: Path | str | None = None,
    *,
    output_root: Path | str | None = None,
) -> Path:
    """``data/output/<book-stem>/`` — the artifact folder for one book.

    ``output_root`` overrides ``data/output`` itself, which is what tests and
    library users want.
    """
    base = Path(output_root).expanduser() if output_root else default_output_dir(root)
    return base / book_slug(book)


def book_chapters_dir(
    book: Path | str,
    root: Path | str | None = None,
    *,
    output_root: Path | str | None = None,
) -> Path:
    """``data/output/<book-stem>/01-chapters/`` — where the chapter files land."""
    return book_output_dir(book, root, output_root=output_root) / CHAPTERS_DIRNAME


def book_markdown_dir(
    book: Path | str,
    root: Path | str | None = None,
    *,
    output_root: Path | str | None = None,
) -> Path:
    """``data/output/<book-stem>/00-markdown/`` — stage 0's converted source."""
    return book_output_dir(book, root, output_root=output_root) / MARKDOWN_DIRNAME


def resolve_book_path(
    book: str | Path | None = None,
    *,
    input_dir: Path | str | None = None,
    root: Path | str | None = None,
) -> Path:
    """Resolve a book argument to an absolute path inside the input folder.

    Accepted forms, in order:

    1. an existing path, used as-is (``data/input/foo.md``, ``/tmp/foo.md``)
    2. a bare filename or stem resolved inside ``data/input/``
    3. ``None`` — the single Markdown file in ``data/input/``, when unambiguous

    A source may be Markdown, plain text, PDF or EPUB: the non-Markdown ones are
    converted by stage 0 before the splitter reads them.

    Raises :class:`FileNotFoundError` with an actionable message otherwise.
    """
    in_dir = Path(input_dir).expanduser() if input_dir else default_input_dir(root)

    if book is None or str(book).strip() == "":
        return _only_book_in(in_dir)

    raw = Path(str(book)).expanduser()
    if raw.is_file():
        return raw.resolve()

    candidate = (in_dir / raw.name).resolve()
    if candidate.is_file():
        return candidate

    if raw.suffix == "":
        for suffix in SOURCE_SUFFIXES:
            with_suffix = (in_dir / f"{raw.name}{suffix}").resolve()
            if with_suffix.is_file():
                return with_suffix

    available = sorted(p.name for p in in_dir.glob("*") if p.is_file()) if in_dir.is_dir() else []
    hint = f"available: {', '.join(available)}" if available else f"{in_dir} is empty"
    raise FileNotFoundError(f"book not found: {book!r} (looked in {in_dir}; {hint})")


def resolve_output_dir(
    book: Path | str,
    out: str | Path | None = None,
    *,
    root: Path | str | None = None,
    output_root: Path | str | None = None,
) -> Path:
    """Resolve ``--out``, defaulting to ``data/output/<book-stem>/01-chapters``."""
    if out is not None and str(out).strip():
        return Path(out).expanduser()
    return book_chapters_dir(book, root, output_root=output_root)


def _only_book_in(in_dir: Path) -> Path:
    if not in_dir.is_dir():
        raise FileNotFoundError(
            f"input folder not found: {in_dir} — create it and drop the source book inside"
        )
    books = sorted(
        p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in _MARKDOWN_SUFFIXES
    )
    if not books:
        books = sorted(p for p in in_dir.iterdir() if p.is_file())
    if not books:
        raise FileNotFoundError(f"no source document found in {in_dir}")
    if len(books) > 1:
        names = ", ".join(p.name for p in books)
        raise FileNotFoundError(
            f"{in_dir} holds several documents ({names}); pass the book to use"
        )
    return books[0].resolve()
