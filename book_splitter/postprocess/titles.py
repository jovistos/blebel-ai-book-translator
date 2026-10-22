"""Title text and the canonical chapter-title policy (stage 9).

``clean_title`` and ``render_heading`` are the two text primitives every title
decision uses; :func:`canonical_titles` is the book-wide policy that derives the
final name of every chapter from the book plan. The rules that *validate* a
plan against them live in :mod:`book_splitter.postprocess.rules`.

Keeping the derivation separate from the validation is also what breaks the old
plan <-> toc import: the TOC generator needs the canonical titles, and it can now
import them without importing the validator.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..core.constants import DEFAULT_CHAPTER_TITLE_LEVEL, MAX_POSTPROCESS_TITLE_CHARS
from ..documents.markdown import page_number_of
from ..documents.text import excerpt
from .inspect import is_bare_chapter_label, strip_chapter_label

_MULTISPACE_RE = re.compile(r"\s+")


def clean_title(text: str, limit: int = MAX_POSTPROCESS_TITLE_CHARS) -> str:
    """One-line title text: no markup, no newlines, collapsed spaces, capped."""
    value = (text or "").strip()
    value = re.sub(r"^#{1,6}\s*", "", value)
    value = _MULTISPACE_RE.sub(" ", value).strip()
    return excerpt(value, limit)


def render_heading(text: str, level: int = DEFAULT_CHAPTER_TITLE_LEVEL) -> str:
    """A Markdown ATX heading line for ``text``."""
    safe_level = max(1, min(int(level or 1), 6))
    return f"{'#' * safe_level} {clean_title(text)}".rstrip()


def is_bare_number_title(text: str) -> bool:
    """True when a title is only its chapter number (``2``, ``## 2``).

    A translated untitled chapter keeps the number but loses the label. That
    number is the chapter's *number*, never its name: it must not be appended to
    the heading as ``Poglavlje 2: 2``.
    """
    value = clean_title(text)
    return bool(value) and not any(char.isalpha() for char in value) and (
        page_number_of(value) is not None
    )


def title_part_of(summary: dict[str, Any]) -> str:
    """The chapter's own title words, with the label/number prefix removed."""
    title = clean_title(str(summary.get("title") or ""))
    if title:
        rest = strip_chapter_label(title)
        if rest and not is_bare_number_title(title):
            return rest
    return clean_title(str(summary.get("subtitle") or ""))


def number_of(summary: dict[str, Any], fallback: int) -> int:
    for key in ("chapter_number", "body_index"):
        value = summary.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                break
    return int(fallback) + 1


def normalise_overrides(value: Any) -> dict[str, str]:
    """Accept ``{file: title}`` or ``[{"filename", "title"}]``; return a dict."""
    out: dict[str, str] = {}
    if isinstance(value, dict):
        for filename, title in value.items():
            name = Path(str(filename or "")).name
            if name:
                out[name] = clean_title(str(title or ""))
    elif isinstance(value, list):
        for entry in value:
            if not isinstance(entry, dict):
                continue
            name = Path(str(entry.get("filename") or entry.get("chapter") or "")).name
            title = clean_title(str(entry.get("title") or ""))
            if name:
                out[name] = title
    return out


def canonical_titles(
    book: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, str]:
    """The final title of every chapter, derived from the book plan.

    The model does not have to echo the whole chapter list: it picks a policy and
    the titles are computed from the inventory. Per-chapter overrides are the
    escape hatch for a name that genuinely cannot follow the policy.
    """
    naming = plan.get("naming") or {}
    mode = str(naming.get("mode") or "keep").strip().lower()
    label = clean_title(str(naming.get("label") or ""))
    include_title = bool(naming.get("include_title"))
    separator = str(naming.get("separator") or ": ")
    if "\n" in separator:
        separator = ": "
    overrides = normalise_overrides(naming.get("overrides"))

    out: dict[str, str] = {}
    for index, summary in enumerate(book.get("chapters") or []):
        filename = str(summary.get("filename") or "")
        if not filename:
            continue
        if overrides.get(filename):
            out[filename] = overrides[filename]
            continue
        if summary.get("matter") != "body":
            out[filename] = clean_title(str(summary.get("title") or "")) or clean_title(
                Path(filename).stem
            )
            continue
        current = title_part_of(summary)
        number = number_of(summary, index)
        if mode == "normalize":
            prefix = f"{label} {number}".strip()
            if include_title and current:
                out[filename] = clean_title(f"{prefix}{separator}{current}")
            else:
                out[filename] = clean_title(prefix) or current
        else:
            base = clean_title(str(summary.get("title") or "")) or clean_title(
                f"{label or 'Poglavlje'} {number}"
            )
            # ``keep`` keeps the title line, but a bare label such as
            # ``Poglavlje 1`` owns the chapter name on the next line. Carry that
            # name into the heading (and so into the TOC) instead of dropping it.
            if current and is_bare_chapter_label(base):
                base = clean_title(f"{base}{separator}{current}")
            out[filename] = base
    return out

__all__ = [
    "canonical_titles",
    "clean_title",
    "is_bare_number_title",
    "normalise_overrides",
    "number_of",
    "render_heading",
    "title_part_of",
]
