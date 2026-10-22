"""Naming the chapters a damaged conversion left as gibberish.

Splitting is deterministic: a line that looks like a marker *is* a marker. What
the marker *means* is not — `## lO` is chapter 10, `## '3` is chapter 13, and the
only evidence is the sequence around them (`I, 4, 5, 7, lO, II, 12, 13`).

That one decision is what the model is for. It receives each section's marker
sequence as written and returns the integer each unclear marker stands for.
Deterministic code then checks the answer — strictly increasing within a section,
no duplicate numbers, nothing invented — and only a passing answer is used, in the
filename. Everything here is testable without a provider: pass ``model``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..core.settings import LLMSettings

#: A marker that needs no interpretation.
PLAIN_NUMBER_RE = re.compile(r"^\d{1,3}$")

RENUMBER_PROMPT = """You are naming the chapters of a book whose text was damaged by a PDF/EPUB \
conversion. Chapter markers are the numbers that survived, and some are mangled: `lO` is 10, `II` \
is 11, `'3` is 13, `I` is 1.

Below is the marker sequence of each section, in file order. Within one section the chapter \
numbers increase. For every marker that is NOT a plain integer, state which integer it means, \
using the clean numbers around it as evidence. If a marker cannot be read as a number at all \
(`T)ear Vincent`, `Dear Theo:`), give it `null` — never invent one.

Answer with JSON only, no prose, no code fences:
{"chapters": [{"book": "<section name>", "as_written": "<marker>", "number": <integer or null>}]}

Sections:
<<SECTIONS>>
"""


def ambiguous_markers(sequence: list[str]) -> list[str]:
    """The markers in ``sequence`` that are not already plain integers."""
    return [marker for marker in sequence if not PLAIN_NUMBER_RE.match(marker.strip())]


def build_prompt(sequences: dict[str, list[str]]) -> str:
    """Render the marker sequences for the model."""
    lines = [
        f"- {book}: " + ", ".join(f"`{marker}`" for marker in markers)
        for book, markers in sequences.items()
    ]
    return RENUMBER_PROMPT.replace("<<SECTIONS>>", "\n".join(lines))


def parse_reply(reply: str) -> list[dict[str, Any]]:
    """Read ``{"chapters": [...]}`` out of a model reply, tolerating chatty wrapping."""
    text = (reply or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return []
    try:
        payload = json.loads(text[start : end + 1])
    except ValueError:
        return []
    chapters = payload.get("chapters") if isinstance(payload, dict) else None
    return [item for item in chapters or [] if isinstance(item, dict)]


def validate_corrections(
    sequences: dict[str, list[str]],
    entries: list[dict[str, Any]],
    *,
    max_chapter: int = 200,
) -> tuple[dict[tuple[str, str], str], list[str]]:
    """Check the model's reading of the mangled markers.

    Returns ``((book, as_written) -> number, problems)``. The corrections are
    accepted only as a whole: a single inconsistency would renumber a chapter
    wrongly, which is worse than leaving the marker as written.
    """
    wanted = {
        (book.upper(), marker)
        for book, markers in sequences.items()
        for marker in ambiguous_markers(markers)
    }
    proposed: dict[tuple[str, str], int] = {}
    problems: list[str] = []

    for entry in entries:
        book = str(entry.get("book") or "").strip()
        marker = str(entry.get("as_written") or "").strip()
        number = entry.get("number")
        key = (book.upper(), marker)
        if key not in wanted:
            continue  # a plain marker, another section, or an invention
        if number is None:
            continue  # the model says this one is not a number: leave it alone
        if isinstance(number, bool) or not isinstance(number, int):
            problems.append(f"{book}: {marker!r} -> {number!r} is not an integer")
            continue
        if not 1 <= number <= max_chapter:
            problems.append(f"{book}: {marker!r} -> {number} is out of range")
            continue
        proposed[key] = number

    for book, markers in sequences.items():
        substituted: list[int] = []
        for marker in markers:
            if PLAIN_NUMBER_RE.match(marker):
                substituted.append(int(marker))
                continue
            number = proposed.get((book.upper(), marker))
            if number is not None:
                substituted.append(number)
        if substituted != sorted(substituted):
            problems.append(f"{book}: the corrected sequence is not increasing ({substituted})")
        duplicates = {n for n in substituted if substituted.count(n) > 1}
        if duplicates:
            problems.append(f"{book}: duplicate chapter number(s) {sorted(duplicates)}")

    if problems:
        return {}, problems
    return {(book, marker): str(number) for (book, marker), number in proposed.items()}, []


def resolve_marker_names(
    sequences: dict[str, list[str]],
    *,
    settings: LLMSettings | None = None,
    model: Any | None = None,
) -> tuple[dict[tuple[str, str], str], list[str]]:
    """Ask the model to read the mangled markers, then validate its answer."""
    if not any(ambiguous_markers(markers) for markers in sequences.values()):
        return {}, []
    if model is None:
        from langchain_openai import ChatOpenAI
        from pydantic import SecretStr

        from ..core.settings import resolve_settings

        settings = settings or resolve_settings(stage="split")
        model = ChatOpenAI(
            model=settings.model,
            base_url=settings.base_url,
            # ChatOpenAI requires SecretStr; keep a missing key as None (as before).
            api_key=SecretStr(settings.api_key) if settings.api_key else None,
            temperature=settings.temperature,
            timeout=settings.timeout,
            max_retries=settings.max_retries,
        )
    reply = model.invoke(build_prompt(sequences))
    text = getattr(reply, "content", reply)
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False)
    return validate_corrections(sequences, parse_reply(text))


__all__ = [
    "PLAIN_NUMBER_RE",
    "RENUMBER_PROMPT",
    "ambiguous_markers",
    "build_prompt",
    "parse_reply",
    "resolve_marker_names",
    "validate_corrections",
]
