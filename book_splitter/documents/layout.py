"""The one stage-directory registry and the one path resolver.

Every stage folder is named with its stage number (``00-markdown/``,
``01-chapters/``, …, ``09-postprocessed-chapters/``). Before this module each
helper in :mod:`book_splitter.documents.chapters` carried its own copy of "if
the caller handed me one of my siblings, step up to the book folder, then
append my own name", and each copy had a slightly different list of siblings it
recognised. That is how a new stage folder gets written inside another stage's
folder.

:data:`STAGE_DIRS` is now the only registry. :func:`resolve_stage_dir` is the
only rule: a stage folder, a sibling stage folder, the book folder and a file
inside the book all resolve to the requested stage's folder.
"""

from __future__ import annotations

from pathlib import Path

from ..core.constants import (
    CHAPTERS_DIRNAME,
    CHUNK_GLOSSARY_DIRNAME,
    CHUNKS_DIRNAME,
    GLOSSARY_DIRNAME,
    MARKDOWN_DIRNAME,
    MATTER_DIRNAME,
    MATTER_PROMPTS_DIRNAME,
    POSTPROCESSED_CHAPTERS_DIRNAME,
    PREPROCESSED_DIRNAME,
    REDUCED_DIRNAME,
    TRANSLATED_CHAPTERS_DIRNAME,
    TRANSLATION_PROMPTS_DIRNAME,
    TRANSLATIONS_DIRNAME,
    UNIQUE_DIRNAME,
)
from ..core.errors import ToolError

#: Stage key -> folder name. Keys are the names code and CLI options use; the
#: values are what a sorted book folder shows. Adding a stage means adding one
#: line here, not editing a dozen sibling lists.
STAGE_DIRS: dict[str, str] = {
    "markdown": MARKDOWN_DIRNAME,
    "chapters": CHAPTERS_DIRNAME,
    "preprocessed": PREPROCESSED_DIRNAME,
    "reduced": REDUCED_DIRNAME,
    "unique": UNIQUE_DIRNAME,
    "glossary": GLOSSARY_DIRNAME,
    "chunks": CHUNKS_DIRNAME,
    "chunk_glossary": CHUNK_GLOSSARY_DIRNAME,
    "matter": MATTER_DIRNAME,
    "matter_prompts": MATTER_PROMPTS_DIRNAME,
    "translations": TRANSLATIONS_DIRNAME,
    "translation_prompts": TRANSLATION_PROMPTS_DIRNAME,
    "translated_chapters": TRANSLATED_CHAPTERS_DIRNAME,
    "postprocessed_chapters": POSTPROCESSED_CHAPTERS_DIRNAME,
}

#: Every folder name the registry knows, for "is this a stage folder?" checks.
STAGE_DIRNAMES: frozenset[str] = frozenset(STAGE_DIRS.values())


def stage_dir_name(stage: str) -> str:
    """The folder name for a stage key, with an actionable error otherwise."""
    try:
        return STAGE_DIRS[str(stage)]
    except KeyError as exc:
        known = ", ".join(sorted(STAGE_DIRS))
        raise ToolError(f"unknown stage {stage!r}; known stages: {known}") from exc


def is_stage_dirname(name: str | Path) -> bool:
    """True when ``name`` is one of the registered stage folder names."""
    return Path(str(name)).name in STAGE_DIRNAMES


def book_root(path: str | Path) -> Path:
    """The book folder for a file, a stage folder, or the book folder itself."""
    candidate = Path(path).expanduser()
    if candidate.is_file() or candidate.suffix.lower() in {".md", ".json", ".epub"}:
        candidate = candidate.parent
    if candidate.name in STAGE_DIRNAMES:
        return candidate.parent
    return candidate


def resolve_stage_dir(path: str | Path, stage: str) -> Path:
    """Resolve ``stage``'s folder from a book folder, a sibling folder, or a file.

    The rule is uniform and deliberately has no per-stage exceptions:

    * the path already *is* the requested folder -> return it unchanged;
    * the path is any other registered stage folder -> resolve from its parent;
    * the path is a file (a chapter, a manifest, the bound book) -> its folder;
    * anything else is treated as the book folder.

    Returned paths are not created; the stage decides whether it will write.
    """
    name = stage_dir_name(stage)
    candidate = Path(path).expanduser()
    if candidate.is_file() or candidate.suffix.lower() in {".md", ".json", ".epub"}:
        candidate = candidate.parent
    if candidate.name == name:
        return candidate
    if candidate.name in STAGE_DIRNAMES:
        candidate = candidate.parent
    return candidate / name


def stage_of(path: str | Path) -> str | None:
    """The stage key a folder belongs to, or ``None`` for the book folder."""
    name = Path(str(path)).expanduser().name
    for stage, dirname in STAGE_DIRS.items():
        if dirname == name:
            return stage
    return None


__all__ = [
    "STAGE_DIRS",
    "STAGE_DIRNAMES",
    "book_root",
    "is_stage_dirname",
    "resolve_stage_dir",
    "stage_dir_name",
    "stage_of",
]
