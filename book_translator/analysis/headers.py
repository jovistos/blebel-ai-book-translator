"""Judging whether the Markdown headers are a trustworthy chapter map.

PDF->Markdown conversions keep the ``#`` markers but add junk headers (page
numbers, figure captions, running heads), so this returns a graded verdict
instead of a boolean.
"""

from __future__ import annotations

from typing import Any

from ..core.constants import MAX_TITLE_LINE_CHARS
from ..documents.text import excerpt
from .patterns import CHAPTER_RE, NUMERIC_RE


def assess_headers(headers: list[dict[str, Any]], line_count: int) -> dict[str, Any]:
    """Score the header set: ``strong``, ``noisy`` or ``absent``."""
    if not headers:
        return {
            "verdict": "absent",
            "score": 0.0,
            "header_count": 0,
            "chapter_like_count": 0,
            "reasons": ["no Markdown headers found"],
            "noise_examples": [],
        }

    chapter_like, noise = [], []
    for h in headers:
        text = str(h.get("text", ""))
        if NUMERIC_RE.match(text) or len(text) < 3:
            noise.append(text)
        elif CHAPTER_RE.match(text) or (
            text.isupper() and len(text) > 3 and len(text) <= MAX_TITLE_LINE_CHARS
        ):
            chapter_like.append(h)
        elif len(text) > MAX_TITLE_LINE_CHARS or text.endswith("."):
            noise.append(text)

    density = len(headers) / max(line_count, 1)
    noise_ratio = len(noise) / max(len(headers), 1)

    score = 0.0
    score += min(len(chapter_like), 10) / 10 * 0.60
    score += 0.25 if density <= 0.05 else (0.10 if density <= 0.12 else 0.0)
    score += 0.15 * (1 - noise_ratio)
    score = round(min(score, 1.0), 3)

    reasons: list[str] = []
    if len(chapter_like) < 3:
        reasons.append(f"only {len(chapter_like)} header(s) look like chapters")
    if noise_ratio > 0.35:
        reasons.append(
            f"{noise_ratio:.0%} of headers look like conversion noise "
            "(page numbers / captions / running heads)"
        )
    if density > 0.12:
        reasons.append(f"header density is high ({density:.1%} of all lines)")

    if len(chapter_like) >= 3 and noise_ratio <= 0.25 and density <= 0.08:
        verdict = "strong"
    else:
        # Headers exist but cannot be trusted as-is: the model must filter them
        # or fall back to the TOC/heuristics routes.
        verdict = "noisy"

    return {
        "verdict": verdict,
        "score": score,
        "header_count": len(headers),
        "chapter_like_count": len(chapter_like),
        "chapter_like": [
            {"line_number": h["line_number"], "text": excerpt(str(h.get("text", "")))}
            for h in chapter_like[:40]
        ],
        "noise_examples": [excerpt(n) for n in noise[:10]],
        "noise_ratio": round(noise_ratio, 3),
        "reasons": reasons or ["headers look consistent"],
    }
