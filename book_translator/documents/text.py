"""Text normalisation: comparison folding, excerpts, slugs and serialisation."""

from __future__ import annotations

import html
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ..core.constants import EXCERPT_CHARS, MAX_TOOL_CHARS

#: Diacritics folded to ASCII so Serbian/Croatian/Czech titles compare cleanly.
DIACRITICS = str.maketrans(
    {
        "č": "c", "ć": "c", "š": "s", "ž": "z", "đ": "dj",
        "Č": "c", "Ć": "c", "Š": "s", "Ž": "z", "Đ": "dj",
        "ä": "a", "ö": "o", "ü": "u", "ß": "ss", "é": "e", "è": "e",
        "á": "a", "à": "a", "í": "i", "ó": "o", "ú": "u", "ñ": "n",
    }
)


def jdump(payload: Any, limit: int | None = MAX_TOOL_CHARS) -> str:
    """Serialise a tool result and enforce the context guardrail."""
    text = json.dumps(payload, ensure_ascii=False, indent=1, default=str)
    if limit is not None and limit > 0 and len(text) > limit:
        text = text[:limit] + f"\n... [tool output truncated at {limit} chars]"
    return text


def norm(text: str) -> str:
    """Normalise a string for fuzzy comparison (diacritics/punctuation folded)."""
    text = html.unescape(text or "").translate(DIACRITICS).lower()
    text = re.sub(r"[\u2018\u2019\u201c\u201d\"'`´]", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def similarity(a: str, b: str) -> float:
    """0..1 similarity between two strings, robust to partial titles."""
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        shorter, longer = sorted((na, nb), key=len)
        return min(0.97, 0.80 + 0.20 * (len(shorter) / max(len(longer), 1)))
    return SequenceMatcher(None, na, nb).ratio()


def excerpt(line: str, limit: int = EXCERPT_CHARS) -> str:
    """A short, single-line preview of a line (safe for a title, not a paragraph)."""
    text = re.sub(r"\s+", " ", line.strip())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def slugify(text: str, max_words: int = 8, max_len: int = 60) -> str:
    """Filesystem-safe ASCII slug (Serbian diacritics folded)."""
    folded = text.translate(DIACRITICS)
    folded = unicodedata.normalize("NFKD", folded).encode("ascii", "ignore").decode()
    folded = re.sub(r"[^\w\s-]", " ", folded).strip().lower()
    words = [w for w in re.split(r"[\s_]+", folded) if w][:max_words]
    slug = "-".join(words)[:max_len].strip("-")
    return slug or "section"


def normalize_filename(name: str, fallback: str = "section") -> str:
    """Force a filename into ``NN-slug.md`` shape, stripping directory parts."""
    stem = Path(str(name or "")).name  # strip any directory component
    if not stem:
        stem = fallback
    if not stem.lower().endswith(".md"):
        stem = f"{stem}.md"
    m = re.match(r"^(\d{1,3})[-_\s]*(.*)$", stem)
    prefix, rest = (m.group(1), m.group(2)) if m else ("", stem)
    rest = rest[:-3] if rest.lower().endswith(".md") else rest
    slug = slugify(rest, max_words=10, max_len=70)
    if prefix:
        return f"{int(prefix):02d}-{slug}.md"
    return f"{prefix}{slug}.md"


def short_args(args: dict[str, Any], limit: int = 80) -> str:
    """One-line rendering of tool arguments for the run log (long lists elided)."""
    parts = []
    for key, value in args.items():
        text = json.dumps(value, ensure_ascii=False, default=str)
        if key in {"chapter_titles", "split_mapping"} and isinstance(value, list):
            text = f"<{len(value)} items>"
        parts.append(f"{key}={text[:60]}")
    out = ", ".join(parts)
    return out if len(out) <= limit else out[: limit - 1] + "…"


def summarise_payload(payload: str) -> str:
    """One-line, content-free summary of a tool result for the run log."""
    try:
        data = json.loads(payload)
    except Exception:
        return f"{len(payload)} chars"
    if not isinstance(data, dict):
        return f"{len(payload)} chars"
    if "error" in data:
        return f"error: {str(data['error'])[:80]}"
    for key in ("count", "found_count", "chapter_count", "file_count", "header_count"):
        if key in data:
            return f"{key}={data[key]}"
    if "ok" in data:
        return f"ok={data['ok']}"
    if "accepted" in data:
        return f"accepted={data['accepted']}"
    return f"{len(payload)} chars"
