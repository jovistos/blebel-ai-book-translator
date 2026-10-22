"""The reversible recursive cut: chonkie rules, span enforcement and previews."""

from __future__ import annotations

import bisect
from collections.abc import Callable, Iterable

from ..core.constants import DEFAULT_CHUNK_TOKENIZER
from ..core.errors import MissingDependencyError, ToolError
from ..documents.images import image_spans
from .boundaries import boundary_kind
from .tokenizer import TokenizerChoice, resolve_tokenizer

#: Chonkie recurses on these, in order. A paragraph is kept whole when it fits.
PARAGRAPH_DELIMITERS: tuple[str, ...] = ("\r\n\r\n", "\n\n", "\r\r")
#: Sentence ends, including the common closing punctuation after the mark.
SENTENCE_END_MARKS: tuple[str, ...] = (".", "!", "?", "…")
SENTENCE_CLOSERS: tuple[str, ...] = ('"', "”", "'", "’", "»", "›", ")", "]", "}")
#: The maximum text a chunk preview (first/last line) keeps in the manifest.
PREVIEW_CHARS = 120


def sentence_delimiters() -> tuple[str, ...]:
    """Every literal delimiter chonkie should treat as a sentence end.

    Delimiters include the whitespace that follows the mark because that is where
    the next sentence starts; ``include_delim="prev"`` then puts the delimiter in
    the previous chunk, which is what keeps concatenation exact.
    """
    delimiters: list[str] = []
    for mark in SENTENCE_END_MARKS:
        for closer in ("", *SENTENCE_CLOSERS):
            delimiters.append(f"{mark}{closer} ")
            delimiters.append(f"{mark}{closer}\n")
            delimiters.append(f"{mark}{closer}\r\n")
    # De-duplicate while keeping the "plain mark" forms first.
    return tuple(dict.fromkeys(delimiters))


def _recursive_rules():
    """The four-level rule set: paragraph, sentence, line, whitespace.

    There is deliberately no final token-level rule. If a single word is longer
    than the requested chunk, the post-split in :func:`_enforce_limit` cuts it,
    and that cut is a source slice too; chonkie's token decode would not be.
    """
    from chonkie.types import RecursiveLevel, RecursiveRules

    return RecursiveRules(
        levels=[
            RecursiveLevel(delimiters=list(PARAGRAPH_DELIMITERS), include_delim="prev"),
            RecursiveLevel(delimiters=list(sentence_delimiters()), include_delim="prev"),
            RecursiveLevel(delimiters=["\n", "\r"], include_delim="prev"),
            RecursiveLevel(whitespace=True, include_delim="prev"),
        ]
    )


def build_chunker(chunk_size: int, tokenizer: str | None = None):
    """Build the recursive chunker and resolve the tokenizer it measures with.

    ``chonkie`` is imported here, not at module import, because the splitter, the
    preprocessor and the glossary must keep working without the chunking extra.
    """
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be greater than 0")
    try:
        from chonkie import RecursiveChunker
    except Exception as exc:  # noqa: BLE001 - an ImportError is the normal case
        raise MissingDependencyError(
            "the chunking stage needs chonkie: pip install chonkie"
        ) from exc

    choice = resolve_tokenizer(tokenizer or DEFAULT_CHUNK_TOKENIZER)
    chunker = RecursiveChunker(
        tokenizer=choice.tokenizer,
        chunk_size=int(chunk_size),
        rules=_recursive_rules(),
        min_characters_per_chunk=1,
    )
    return chunker, choice


def _spans_from_chonkie(text: str, chunks) -> list[tuple[int, int]]:
    """Turn chonkie chunks into contiguous source spans, or explain why not."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for chunk in chunks:
        start = int(getattr(chunk, "start_index", cursor))
        end = int(getattr(chunk, "end_index", start))
        if start != cursor or not (cursor < end <= len(text)):
            raise ToolError(
                "chonkie returned an invalid chunk span "
                f"({start}:{end} for a chapter of {len(text)} characters, "
                f"expected to start at {cursor})"
            )
        spans.append((start, end))
        cursor = end
    if cursor != len(text):
        raise ToolError(
            "chonkie did not cover the whole chapter "
            f"({cursor} of {len(text)} characters); refusing to write a cut "
            "that would not reconstruct"
        )
    return spans


def _largest_prefix(
    text: str,
    start: int,
    end: int,
    count_tokens: Callable[[str], int],
    limit: int,
    *,
    window: int = 256,
) -> int:
    """The largest cut in ``(start, end]`` that keeps the prefix at or under ``limit``.

    A binary search finds the token boundary; the small backwards walk then moves
    the cut to whitespace when one is close, so an oversized paragraph is still
    not cut inside a word unless it has no whitespace at all.
    """
    low, high = start + 1, end
    best = start + 1
    while low <= high:
        middle = (low + high) // 2
        if count_tokens(text[start:middle]) <= limit:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    floor = max(start + 1, best - window)
    for cut in range(best, floor - 1, -1):
        if text[cut - 1].isspace():
            return cut
    return best


def _enforce_limit(
    text: str,
    spans: Iterable[tuple[int, int]],
    count_tokens: Callable[[str], int],
    limit: int,
) -> list[tuple[int, int]]:
    """Split any chunk chonkie left over the limit, at whitespace where possible."""
    result: list[tuple[int, int]] = []
    for start, end in spans:
        if count_tokens(text[start:end]) <= limit:
            result.append((start, end))
            continue
        cursor = start
        while cursor < end:
            if count_tokens(text[cursor:end]) <= limit:
                result.append((cursor, end))
                break
            cut = _largest_prefix(text, cursor, end, count_tokens, limit)
            if cut <= cursor:  # pragma: no cover - defensive: a single character
                cut = min(cursor + 1, end)
            result.append((cursor, cut))
            cursor = cut
    return result


def line_numbers(text: str, newlines: list[int], start: int, end: int) -> tuple[int, int]:
    """1-based line span of ``[start, end)`` in ``text``."""
    first = bisect.bisect_left(newlines, start) + 1
    last_position = max(end - 1, start)
    last = bisect.bisect_left(newlines, last_position) + 1
    return first, last


def preview(value: str) -> str:
    first = next((line for line in value.splitlines() if line.strip()), "")
    return first.strip()[:PREVIEW_CHARS]


def last_preview(value: str) -> str:
    lines = [line for line in value.splitlines() if line.strip()]
    return lines[-1].strip()[:PREVIEW_CHARS] if lines else ""


def chunk_text(
    text: str,
    *,
    chunker,
    tokenizer: TokenizerChoice,
    chunk_size: int,
) -> list[tuple[int, int, int, str, str]]:
    """Cut one chapter into spans, returning ``(start, end, tokens, before, after)``.

    The token count is measured by the resolved tokenizer, not read from chonkie,
    so the number in the manifest is the number the stage actually enforced.
    """
    if not text:
        return []
    spans = _spans_from_chonkie(text, chunker.chunk(text))
    spans = _enforce_limit(text, spans, tokenizer.count_tokens, int(chunk_size))
    spans = _image_safe_spans(text, spans, tokenizer.count_tokens, int(chunk_size))
    if "".join(text[start:end] for start, end in spans) != text:  # pragma: no cover
        raise ToolError("the computed chunk spans do not cover the chapter exactly")
    output: list[tuple[int, int, int, str, str]] = []
    for position, (start, end) in enumerate(spans):
        before = "chapter-start" if position == 0 else boundary_kind(text, start)
        after = "chapter-end" if position == len(spans) - 1 else boundary_kind(text, end)
        output.append((start, end, tokenizer.count_tokens(text[start:end]), before, after))
    return output


def _image_safe_spans(
    text: str,
    spans: list[tuple[int, int]],
    count_tokens,
    limit: int,
) -> list[tuple[int, int]]:
    """Move a cut that landed inside ``![…](…)`` to the nearest image edge.

    A split image reference is no longer an image: the translator sees half a
    path in one chunk and half in the next, and neither half survives. Moving the
    cut keeps both chunks exact slices. The side is chosen so both resulting
    chunks stay inside the token limit when possible; if the picture is larger
    than the requested chunk, the image wins and the caller's verification says
    so instead of silently corrupting it.
    """
    if len(spans) <= 1:
        return spans
    ranges = image_spans(text)
    if not ranges:
        return spans

    cuts: list[int] = []
    cursor = 0
    for position, (_start, end) in enumerate(spans[:-1]):
        start = cursor
        cut = end
        next_end = spans[position + 1][1]
        inside = next(
            ((left, right) for left, right in ranges if left < cut < right and right <= next_end),
            None,
        )
        if inside is not None:
            left_edge, right_edge = inside
            candidates = (left_edge, right_edge)
            viable = [
                candidate
                for candidate in candidates
                if count_tokens(text[start:candidate]) <= limit
                and count_tokens(text[candidate:next_end]) <= limit
            ]
            cut = (
                viable[0]
                if viable
                else min(
                    candidates,
                    key=lambda candidate: max(
                        count_tokens(text[start:candidate]),
                        count_tokens(text[candidate:next_end]),
                    ),
                )
            )
        cuts.append(cut)
        cursor = cut

    if any(cut <= 0 or cut >= len(text) for cut in cuts):
        return spans
    if any(left >= right for left, right in zip(cuts, cuts[1:], strict=False)):
        return spans

    adjusted = [(0, cuts[0])]
    adjusted.extend(zip(cuts, cuts[1:], strict=False))
    adjusted.append((cuts[-1], len(text)))
    if any(start >= end for start, end in adjusted):
        return spans
    return adjusted


__all__ = [
    "build_chunker",
    "chunk_text",
    "last_preview",
    "line_numbers",
    "preview",
    "sentence_delimiters",
]
