"""Deterministic discovery of front and back matter.

The localization stage runs after chunking, so it has two sources of truth: the
chapter files stage 1 wrote (which carry the titles the splitter found) and
stage 5's chunk manifest (which says which chunk belongs to which chapter). This
module joins them and answers one question:

    which chapters are front matter, which are back matter, and which chunks
    must the localization agent translate instead of stage 7?

The classifier is the same one stage 9 uses (:func:`postprocess.inspect.matter_kind`)
because the two stages must agree: a preface that stage 9 keeps out of the body
numbering is exactly a section stage 7 must not translate as a body chunk.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from book_translator.core.constants import (
    CHAPTERS_DIRNAME,
    CHUNKS_DIRNAME,
    MAX_MATTER_READ_CHARS,
)
from book_translator.core.errors import ToolError
from book_translator.documents.chapters import Chapter, discover_chapters
from book_translator.documents.chunks import chunk_files
from book_translator.documents.manifest import load_chunk_manifest, load_split_manifest
from book_translator.stages.postprocess.inspect import matter_kind, scan_chapter_structure

from .metadata import parse_book_metadata

#: Stage 1's ``kind`` values, kept here so a manifest's wording is not guessed.
_FRONT_KINDS = frozenset({"front-matter", "frontmatter", "front matter", "front_matter", "front"})
_BACK_KINDS = frozenset({"back-matter", "backmatter", "back matter", "back_matter", "back"})


@dataclass
class MatterSection:
    """One front/back matter chapter and the chunks stage 5 cut from it."""

    filename: str
    path: str
    kind: str
    title: str = ""
    chars: int = 0
    lines: int = 0
    chunk_filenames: list[str] = field(default_factory=list)
    chunk_count: int = 0
    manifest_kind: str = ""
    title_source: str = ""

    @property
    def is_front(self) -> bool:
        return self.kind == "front"

    @property
    def is_back(self) -> bool:
        return self.kind == "back"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MatterBook:
    """The book-level inventory the plan agent sees (metadata, never body prose)."""

    chapters_path: str
    chunks_path: str = ""
    source_file: str = ""
    book_title: str = ""
    book_author: str = ""
    source_language: str = ""
    target_language: str = ""
    sections: list[MatterSection] = field(default_factory=list)
    body_chapters: list[dict[str, Any]] = field(default_factory=list)
    chapter_count: int = 0

    @property
    def front_sections(self) -> list[MatterSection]:
        return [section for section in self.sections if section.is_front]

    @property
    def back_sections(self) -> list[MatterSection]:
        return [section for section in self.sections if section.is_back]

    @property
    def chunk_count(self) -> int:
        return sum(section.chunk_count for section in self.sections)

    def section(self, filename: str) -> MatterSection | None:
        wanted = Path(str(filename or "")).name
        for section in self.sections:
            if section.filename == wanted:
                return section
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chapters_path": self.chapters_path,
            "chunks_path": self.chunks_path,
            "source_file": self.source_file,
            "book_title": self.book_title,
            "book_author": self.book_author,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "chapter_count": self.chapter_count,
            "matter_section_count": len(self.sections),
            "matter_chunk_count": self.chunk_count,
            "sections": [section.to_dict() for section in self.sections],
            "body_chapters": list(self.body_chapters),
        }


def _chunk_map(chunks_path: str | Path | None) -> dict[str, list[str]]:
    """``chapter filename -> chunk filenames`` from stage 5's manifest."""
    if chunks_path is None or not str(chunks_path).strip():
        return {}
    folder = Path(chunks_path).expanduser()
    if folder.is_file():
        folder = folder.parent
    if not folder.is_dir():
        return {}
    try:
        entries = chunk_files(folder)
    except (ToolError, OSError):
        return {}
    grouped: dict[str, list[str]] = {}
    for entry in entries:
        chapter = str(entry.get("chapter_filename") or "")
        filename = str(entry.get("filename") or "")
        if chapter and filename:
            grouped.setdefault(chapter, []).append(filename)
    for filenames in grouped.values():
        filenames.sort()
    return grouped


def _manifest_kind(chapter: Chapter) -> str:
    return str(chapter.extras.get("kind") or "").strip().lower()


def _kind_of(filename: str, index: int, title: str, manifest_kind: str) -> str:
    """Classify one chapter, preferring stage 1's explicit ``kind``.

    The fallback is stage 9's classifier: it notices a preface or a notes section
    whose filename looks numbered, which is the case that made this stage
    necessary (``01-predgovor.md`` must not be translated with a dictionary).
    """
    if manifest_kind in _FRONT_KINDS:
        return "front"
    if manifest_kind in _BACK_KINDS:
        return "back"
    return matter_kind(filename, index, title)


def _resolve_chapters_dir(chapters_path: str | Path, chunks_path: str | Path | None) -> Path:
    candidate = Path(chapters_path).expanduser()
    if candidate.is_file():
        candidate = candidate.parent
    if candidate.is_dir() and candidate.name == CHAPTERS_DIRNAME:
        return candidate
    if candidate.is_dir():
        nested = candidate / CHAPTERS_DIRNAME
        if nested.is_dir():
            return nested
    if chunks_path is not None:
        sibling = Path(chunks_path).expanduser().parent / CHAPTERS_DIRNAME
        if sibling.is_dir():
            return sibling
    return candidate


def _chapter_title(chapter: Chapter) -> tuple[str, str]:
    """The chapter's title and where it came from (manifest, then headings)."""
    if str(chapter.title or "").strip():
        return str(chapter.title).strip(), "manifest"
    path = Path(chapter.path)
    if path.is_file():
        try:
            structure = scan_chapter_structure(path)
        except (OSError, ToolError):
            structure = None
        if structure is not None and structure.title:
            return structure.title, structure.title_source
    return Path(chapter.filename).stem.replace("-", " ").strip(), "filename"


def discover_matter_sections(
    chapters_path: str | Path,
    chunks_path: str | Path | None = None,
) -> list[MatterSection]:
    """Every front/back matter chapter, with its chunks, in reading order.

    Body chapters are omitted. The result is empty for a book whose front matter
    the splitter did not write and whose chapters carry no preface/back-matter
    title; that is a legitimate book, and the stage then reports a skip.
    """
    base = _resolve_chapters_dir(chapters_path, chunks_path)
    if not base.is_dir():
        raise ToolError(f"chapters folder not found: {base} — run `split` first")
    chapters = discover_chapters(base)
    grouped = _chunk_map(chunks_path if chunks_path is not None else base.parent / CHUNKS_DIRNAME)
    sections: list[MatterSection] = []
    for index, chapter in enumerate(chapters):
        title, title_source = _chapter_title(chapter)
        manifest_kind = _manifest_kind(chapter)
        kind = _kind_of(chapter.filename, index, title, manifest_kind)
        if kind == "body":
            continue
        chunks = grouped.get(chapter.filename, [])
        try:
            chars = len(Path(chapter.path).read_text(encoding="utf-8"))
        except OSError:
            chars = chapter.bytes
        sections.append(
            MatterSection(
                filename=chapter.filename,
                path=str(chapter.path),
                kind=kind,
                title=title,
                chars=chars,
                lines=chapter.line_count,
                chunk_filenames=list(chunks),
                chunk_count=len(chunks),
                manifest_kind=manifest_kind,
                title_source=title_source,
            )
        )
    return sections


def body_chapter_titles(chapters_path: str | Path) -> list[dict[str, Any]]:
    """The body chapters' titles, for a front-matter table of contents.

    Only titles and filenames are returned; the agent must not read chapter
    prose to localize a title page.
    """
    base = Path(chapters_path).expanduser()
    if base.is_file():
        base = base.parent
    try:
        chapters = discover_chapters(base)
    except (ToolError, OSError):
        return []
    out: list[dict[str, Any]] = []
    for index, chapter in enumerate(chapters):
        title, _source = _chapter_title(chapter)
        if _kind_of(chapter.filename, index, title, _manifest_kind(chapter)) != "body":
            continue
        out.append(
            {
                "filename": chapter.filename,
                "title": title,
                "chapter_index": index,
                "start_line": chapter.start_line,
            }
        )
    return out


def matter_chapter_names(
    chapters_path: str | Path,
    chunks_path: str | Path | None = None,
) -> dict[str, str]:
    """``chapter filename -> "front" | "back"`` for stage 7's filter.

    A folder without a readable chapters directory falls back to the titles
    stage 5 recorded in its chunk manifest, so a resumed translation still knows
    which chunks belong to the matter stage even if ``01-chapters/`` was moved.
    """
    base = _resolve_chapters_dir(chapters_path, chunks_path)
    try:
        sections = discover_matter_sections(base, chunks_path)
    except (ToolError, OSError):
        sections = []
    if sections:
        return {section.filename: section.kind for section in sections}

    # Fallback: stage 5's manifest carries the chapter titles it chunked.
    folder = Path(chunks_path).expanduser() if chunks_path is not None else base.parent / CHUNKS_DIRNAME
    if folder.is_file():
        folder = folder.parent
    manifest = load_chunk_manifest(folder)
    records = [
        record for record in manifest.get("chapters") or [] if isinstance(record, dict)
    ]
    names: dict[str, str] = {}
    for index, record in enumerate(records):
        filename = str(record.get("filename") or "")
        if not filename:
            continue
        title = str(record.get("title") or "")
        kind = _kind_of(filename, index, title, "")
        if kind != "body":
            names[filename] = kind
    if names:
        return names
    # A folder with neither a chapters folder nor a manifest: filename rules only.
    for entry in chunk_files(folder) if folder.is_dir() else []:
        filename = str(entry.get("chapter_filename") or "")
        if filename and filename not in names:
            kind = _kind_of(filename, len(names), "", "")
            if kind != "body":
                names[filename] = kind
    return names


def read_section_text(
    section: MatterSection,
    *,
    max_chars: int = MAX_MATTER_READ_CHARS,
) -> dict[str, Any]:
    """One matter section's text, clipped only at the documented hard cap.

    The agent must be able to read a title page and a copyright page: unlike the
    splitting and postprocessing agents, this stage is *allowed* to read prose.
    The cap exists so a pathological file cannot exhaust the context window, and
    ``truncated`` tells the runner not to localize a clipped section silently.
    """
    path = Path(section.path).expanduser()
    if not path.is_file():
        raise ToolError(f"matter section not found: {path}")
    text = path.read_text(encoding="utf-8")
    limit = max(1, int(max_chars or MAX_MATTER_READ_CHARS))
    truncated = len(text) > limit
    return {
        "filename": section.filename,
        "path": str(path),
        "kind": section.kind,
        "title": section.title,
        "chars": len(text),
        "truncated": truncated,
        "text": text[:limit],
    }


def _source_file(chapters_path: Path, manifest: dict[str, Any]) -> str:
    source = str(manifest.get("source_file") or manifest.get("source_path") or "")
    if source:
        return source
    return ""


def _original_source_filename(chapters_path: Path, source_file: str) -> str:
    """The original document filename (e.g. EPUB) if converted through 00-markdown."""
    if source_file:
        p = Path(source_file)
        if p.name.endswith(".md") and (p.parent.name == "00-markdown" or (chapters_path.parent / "00-markdown").is_dir()):
            convert_manifest = (
                p.parent / "_convert_manifest.json"
                if p.parent.name == "00-markdown"
                else chapters_path.parent / "00-markdown" / "_convert_manifest.json"
            )
            if convert_manifest.is_file():
                try:
                    import json
                    cdata = json.loads(convert_manifest.read_text(encoding="utf-8"))
                    orig = cdata.get("source") or cdata.get("source_file") or cdata.get("source_path")
                    if orig:
                        return Path(orig).name
                except Exception:
                    pass
    return ""


def build_matter_book(
    chapters_path: str | Path,
    chunks_path: str | Path | None = None,
    *,
    source_language: str = "",
    target_language: str = "",
    book_title: str = "",
    book_author: str = "",
) -> MatterBook:
    """The whole inventory the plan agent starts from."""
    base = _resolve_chapters_dir(chapters_path, chunks_path)
    sections = discover_matter_sections(base, chunks_path)
    body = body_chapter_titles(base)
    manifest = load_split_manifest(base)
    source_file = _source_file(base, manifest)
    orig_name = _original_source_filename(base, source_file)
    metadata = parse_book_metadata(
        orig_name or source_file or base.parent.name.replace("-", " ")
    )
    title = str(book_title or "").strip() or metadata.title
    author = str(book_author or "").strip() or metadata.author
    if not title:
        front = next((section for section in sections if section.is_front), None)
        title = (front.title if front else "") or (
            Path(source_file).stem.replace("_", " ") if source_file else base.parent.name
        )
    return MatterBook(
        chapters_path=str(base),
        chunks_path=str(
            Path(chunks_path).expanduser()
            if chunks_path is not None
            else base.parent / CHUNKS_DIRNAME
        ),
        source_file=source_file,
        book_title=title,
        book_author=author,
        source_language=source_language,
        target_language=target_language,
        sections=sections,
        body_chapters=body,
        chapter_count=len(sections) + len(body),
    )


__all__ = [
    "MatterBook",
    "MatterSection",
    "body_chapter_titles",
    "build_matter_book",
    "discover_matter_sections",
    "matter_chapter_names",
    "read_section_text",
]
