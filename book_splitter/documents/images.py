"""Markdown/HTML image references: find them, protect them, put them back.

Images are the one part of a book a language model must not touch. A translator
that "improves" an image path, drops a decorative rule or rewrites an ``alt``
attribute corrupts the published Markdown even when its prose is perfect. Every
stage therefore treats an image reference as an opaque token:

* **Stage 0** writes them, relative to ``00-markdown/images/``;
* **stage 2** refuses any cleaning operation aimed at a line that carries one;
* **stage 5** does not cut through one;
* **stage 7** swaps each reference for ``[[IMG_1]]`` before the model sees the
  chunk and restores it afterwards. The model is told to copy the marker; the
  runner verifies every marker came back exactly once.
* **stage 10** rewrites the relative prefix for the book-root file.

The module has no dependency on docling, a provider or the pipeline's records:
it works on one string at a time, which is what makes it testable offline.
"""

from __future__ import annotations

import re

#: ``![alt](path)`` and ``![alt](<path with spaces>)``. Deliberately conservative:
#: an image reference is one line and never nests parentheses more than once.
MARKDOWN_IMAGE_RE = re.compile(
    r"!\[[^\]\n]*\]\(\s*(?:<[^>\n]+>|[^)\s]+)(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)"
)
#: An HTML image tag (Markdown permits raw HTML). The pipeline preserves it as-is.
HTML_IMAGE_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE | re.DOTALL)
#: Either shape, in reading order.
IMAGE_RE = re.compile(
    f"(?:{MARKDOWN_IMAGE_RE.pattern}|{HTML_IMAGE_RE.pattern})",
    re.IGNORECASE | re.DOTALL,
)

#: The marker a chunk is translated with. ASCII on purpose: every provider and
#: every target script can copy it, and it cannot be confused with prose.
IMAGE_TOKEN_TEMPLATE = "[[IMG_{index}]]"
IMAGE_TOKEN_RE = re.compile(r"\[\[IMG_(\d+)\]\]")

MISSING_IMAGE_MESSAGE = "the translation dropped image marker(s)"
UNKNOWN_IMAGE_MESSAGE = "the translation invented an image marker"


def image_matches(text: str) -> list[re.Match[str]]:
    """Every image reference in ``text``, in reading order."""
    return list(IMAGE_RE.finditer(str(text or "")))


def image_refs(text: str) -> list[str]:
    """The image references themselves, in reading order."""
    return [match.group(0) for match in image_matches(text)]


def image_count(text: str) -> int:
    return len(image_matches(text))


def has_images(text: str) -> bool:
    return IMAGE_RE.search(str(text or "")) is not None


def is_image_line(line: str) -> bool:
    """True when a line holds nothing but image references and whitespace."""
    text = str(line or "").strip()
    if not text or not IMAGE_RE.search(text):
        return False
    return not IMAGE_RE.sub("", text).strip()


def strip_images(text: str) -> str:
    """Remove image references, leaving the surrounding prose alone."""
    return IMAGE_RE.sub("", str(text or ""))


def image_spans(text: str) -> list[tuple[int, int]]:
    """``(start, end)`` character spans of every image reference."""
    return [(match.start(), match.end()) for match in image_matches(text)]


def image_tokens(text: str) -> list[str]:
    """The ``[[IMG_n]]`` markers present in ``text``."""
    return [match.group(0) for match in IMAGE_TOKEN_RE.finditer(str(text or ""))]


def protect_images(text: str) -> tuple[str, dict[str, str]]:
    """Replace every image reference with a unique ``[[IMG_n]]`` marker.

    Returns the protected text and the marker -> original-reference map. The map
    is what :func:`restore_images` consumes after the model answered; it is also
    saved in the stage-7 debug record so a surprising translation can be traced
    back to the picture it stood for.
    """
    source = str(text or "")
    matches = image_matches(source)
    if not matches:
        return source, {}
    parts: list[str] = []
    mapping: dict[str, str] = {}
    cursor = 0
    for index, match in enumerate(matches, start=1):
        token = IMAGE_TOKEN_TEMPLATE.format(index=index)
        # A marker must be unique in the chunk. If the author's own text happens
        # to contain it, step over that number until it does not.
        bump = index
        while token in source or token in mapping:
            bump += 1
            token = IMAGE_TOKEN_TEMPLATE.format(index=bump)
        mapping[token] = match.group(0)
        parts.append(source[cursor : match.start()])
        parts.append(token)
        cursor = match.end()
    parts.append(source[cursor:])
    return "".join(parts), mapping


def restore_images(text: str, mapping: dict[str, str]) -> tuple[str, list[str], list[str]]:
    """Put the original references back.

    Returns ``(restored, missing, duplicated)``. ``missing`` lists markers the
    model dropped; ``duplicated`` lists markers it emitted more than once. Both
    are failures: the caller must not publish a chunk with a lost or doubled
    picture.
    """
    value = str(text or "")
    missing: list[str] = []
    duplicated: list[str] = []
    for token in mapping:
        count = value.count(token)
        if count == 0:
            missing.append(token)
        elif count > 1:
            duplicated.append(token)
    restored = value
    for token, original in mapping.items():
        restored = restored.replace(token, original)
    return restored, missing, duplicated


def image_token_problems(text: str, mapping: dict[str, str]) -> list[str]:
    """Human-readable reasons a translated chunk cannot keep its pictures.

    Empty when every marker of the source appears exactly once and no marker the
    source did not contain was invented. The messages name counts, never the
    image paths, so they can be handed back to the model in a repair note.
    """
    value = str(text or "")
    if not mapping:
        known = set(image_tokens(value))
        return (
            [f"{UNKNOWN_IMAGE_MESSAGE}: {', '.join(sorted(known))}"] if known else []
        )
    problems: list[str] = []
    missing = [token for token in mapping if value.count(token) == 0]
    if missing:
        problems.append(
            f"{MISSING_IMAGE_MESSAGE}: {', '.join(missing)}; copy every marker exactly once"
        )
    duplicated = [token for token in mapping if value.count(token) > 1]
    if duplicated:
        problems.append(
            f"the translation repeated image marker(s): {', '.join(duplicated)}; "
            "each marker belongs in exactly one place"
        )
    unknown = [token for token in image_tokens(value) if token not in mapping]
    if unknown:
        problems.append(
            f"{UNKNOWN_IMAGE_MESSAGE}: {', '.join(unknown)}; do not add image markers"
        )
    return problems


__all__ = [
    "IMAGE_RE",
    "IMAGE_TOKEN_RE",
    "IMAGE_TOKEN_TEMPLATE",
    "MARKDOWN_IMAGE_RE",
    "HTML_IMAGE_RE",
    "MISSING_IMAGE_MESSAGE",
    "UNKNOWN_IMAGE_MESSAGE",
    "has_images",
    "image_count",
    "image_matches",
    "image_refs",
    "image_spans",
    "image_token_problems",
    "image_tokens",
    "is_image_line",
    "protect_images",
    "restore_images",
    "strip_images",
]
