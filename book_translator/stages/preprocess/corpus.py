"""Corpus sampling: enough text to find a *repeating* defect, never a chapter.

A per-chapter inspector sees one chapter's headers. Some defects only show up
across the book — one chapter has a figure caption, another has a page number
inside a word, a third has a running head. The sampling tools exist so an agent
can ask "what damage does this book have?" and get a bounded, sampled answer.

Every tool here is deliberately lossy and says so in its result: samples are
clipped, strided and capped by a character budget. A sample is *evidence of a
pattern*, never a substitute for reading a chapter, and the returned text is
always marked as a fragment.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from book_translator.core.constants import (
    DEFAULT_SAMPLE_LINES_PER_CHAPTER,
    MAX_SAMPLE_CHARS,
    MAX_SAMPLE_LINE_CHARS,
)
from book_translator.documents.bodyline import (
    count_space_before_punct,
    has_midword_number,
    is_caption_line,
    is_word_continuation,
)
from book_translator.documents.io import load_lines
from book_translator.documents.markdown import header_of, is_page_number_line
from book_translator.documents.text import excerpt

from .damage import inline_page_number_candidate

#: Sampling strategies, in the order the tool tries them.
SAMPLE_HEAD = "head"
SAMPLE_TAIL = "tail"
SAMPLE_STRIDE = "stride"
SAMPLE_LONGEST = "longest"
SAMPLE_HEADERS = "headers"
SAMPLE_STRATEGIES = (
    SAMPLE_HEAD,
    SAMPLE_TAIL,
    SAMPLE_STRIDE,
    SAMPLE_LONGEST,
    SAMPLE_HEADERS,
)


def _pick_lines(
    lines: list[str], strategy: str, max_lines: int, *, skip_blank: bool = True
) -> list[int]:
    """Choose 1-indexed line numbers to sample, deterministically."""
    candidates = [i for i, line in enumerate(lines, start=1) if line.strip() or not skip_blank]
    if not candidates:
        return []
    if strategy == SAMPLE_TAIL:
        return candidates[-max_lines:]
    if strategy == SAMPLE_STRIDE:
        if len(candidates) <= max_lines:
            return candidates
        step = max(1, len(candidates) // max_lines)
        picked = candidates[::step]
        return picked[:max_lines]
    if strategy == SAMPLE_LONGEST:
        ranked = sorted(candidates, key=lambda n: len(lines[n - 1]), reverse=True)
        return sorted(ranked[:max_lines])
    if strategy == SAMPLE_HEADERS:
        return [n for n in candidates if header_of(lines[n - 1])][:max_lines]
    return candidates[:max_lines]


def sample_text(
    path: str | Path,
    max_lines: int = DEFAULT_SAMPLE_LINES_PER_CHAPTER,
    strategy: str = SAMPLE_STRIDE,
    max_chars: int = MAX_SAMPLE_CHARS,
) -> dict[str, Any]:
    """A bounded, clearly-marked sample of one chapter's lines.

    Lines are clipped to :data:`MAX_SAMPLE_LINE_CHARS` characters and the whole
    sample stops at ``max_chars``. Blank-line runs are reported as counts rather
    than as empty strings, so a sample does not waste budget on whitespace.
    """
    p = Path(path).expanduser()
    lines = load_lines(p)
    strategy = strategy if strategy in SAMPLE_STRATEGIES else SAMPLE_STRIDE
    picked = _pick_lines(lines, strategy, max(1, int(max_lines)))
    samples: list[dict[str, Any]] = []
    used = 0
    clipped = False
    for number in picked:
        raw = lines[number - 1].rstrip("\n")
        text = excerpt(raw, MAX_SAMPLE_LINE_CHARS)
        if len(text) < len(raw.strip()):
            clipped = True
        if not text:
            continue
        if used + len(text) > max_chars:
            clipped = True
            break
        used += len(text)
        samples.append(
            {
                "line_number": number,
                "is_header": header_of(raw) is not None,
                "chars": len(raw),
                "text": text,
            }
        )
    return {
        "chapter": {"filename": p.name, "path": str(p), "line_count": len(lines)},
        "strategy": strategy,
        "requested_lines": max_lines,
        "sampled_lines": len(samples),
        "chars_used": used,
        "clipped": clipped,
        "note": (
            "SAMPLES ONLY - lines are clipped and the sample is capped. "
            "Never treat this as the chapter text."
        ),
        "samples": samples,
    }


#: What the corpus scanner can count, and the fix each pattern has.
DAMAGE_PATTERNS: dict[str, dict[str, str]] = {
    "page_number_header": {
        "summary": "a header whose whole body is a printed page number",
        "fix": "remove the line (page artefact)",
        "stage": "preprocess",
    },
    "page_number_in_word": {
        "summary": "a page number printed between the halves of one word",
        "fix": "drop the digits and glue the word back together",
        "stage": "preprocess",
    },
    "split_word": {
        "summary": "a line ending in an unfinished word that continues on the next line",
        "fix": "join the two lines",
        "stage": "preprocess",
    },
    "space_before_punctuation": {
        "summary": "a space in front of , . ; : ! ?",
        "fix": "delete the space",
        "stage": "preprocess",
    },
    "caption_as_header": {
        "summary": "a figure/table caption, marked up as a header or left as a body line",
        "fix": "demote a marked-up caption to a paragraph; keep its words with the image",
        "stage": "preprocess",
    },
    "long_prose_header": {
        "summary": "a header line longer than any plausible heading",
        "fix": "demote to a paragraph (keep the text)",
        "stage": "preprocess",
    },
    "plain_page_number": {
        "summary": "a page number left as a plain body line, with no # markup",
        "fix": "remove the line (page artefact)",
        "stage": "preprocess",
    },
    "inline_page_number": {
        "summary": "a page number merged into a text line instead of standing alone",
        "fix": "drop the number, keep the words on both sides",
        "stage": "preprocess",
    },
    "single_char_header": {
        "summary": "a one-character header, usually the other half of a split heading",
        "fix": "merge with the neighbouring heading",
        "stage": "preprocess",
    },
}

CAPTION_MARKERS = (
    "slika", "tabela", "tabla", "grafikon", "dijagram", "figure", "fig.", "table", "chart",
)


def scan_corpus_damage(
    chapters: Iterable[Any],
    *,
    max_examples_per_pattern: int = 3,
    max_line_chars: int = 120,
) -> dict[str, Any]:
    """Count every recognised damage pattern across a whole book.

    ``chapters`` is anything with ``path``, ``filename`` and ``chapter_id``
    attributes (see :class:`book_translator.documents.chapters.Chapter`).

    The result is a *count per pattern* plus a few short examples, so an agent
    can decide which fixes to request per chapter without reading a chapter.
    """
    counts: dict[str, int] = dict.fromkeys(DAMAGE_PATTERNS, 0)
    per_chapter: dict[str, dict[str, int]] = {}
    examples: dict[str, list[dict[str, Any]]] = {name: [] for name in DAMAGE_PATTERNS}
    total_lines = 0

    for chapter in chapters:
        lines = load_lines(chapter.path)
        total_lines += len(lines)
        chapter_counts: dict[str, int] = {}
        for idx, raw in enumerate(lines):
            line = raw.rstrip("\n")
            number = idx + 1
            head = header_of(line)
            if head:
                body = head[1]
                if is_page_number_line(body):
                    _bump(counts, chapter_counts, "page_number_header")
                    _example(examples, "page_number_header", chapter, number, body, max_examples_per_pattern, max_line_chars)
                elif is_caption_line(body):
                    _bump(counts, chapter_counts, "caption_as_header")
                    _example(examples, "caption_as_header", chapter, number, body, max_examples_per_pattern, max_line_chars)
                elif len(body) > 90:
                    _bump(counts, chapter_counts, "long_prose_header")
                    _example(examples, "long_prose_header", chapter, number, body, max_examples_per_pattern, max_line_chars)
                elif len(body.strip()) <= 1:
                    _bump(counts, chapter_counts, "single_char_header")
                    _example(examples, "single_char_header", chapter, number, body, max_examples_per_pattern, max_line_chars)
            if not head and is_page_number_line(line):
                _bump(counts, chapter_counts, "plain_page_number")
                _example(examples, "plain_page_number", chapter, number, line, max_examples_per_pattern, max_line_chars)
            elif not head and not is_caption_line(line):
                inline = inline_page_number_candidate(line)
                if inline is not None:
                    _bump(counts, chapter_counts, "inline_page_number")
                    _example(examples, "inline_page_number", chapter, number, line, max_examples_per_pattern, max_line_chars)
            if has_midword_number(line):
                _bump(counts, chapter_counts, "page_number_in_word", len(has_midword_number(line)))
                _example(examples, "page_number_in_word", chapter, number, line, max_examples_per_pattern, max_line_chars)
            if count_space_before_punct(line):
                _bump(counts, chapter_counts, "space_before_punctuation")
            if idx + 1 < len(lines) and is_word_continuation(line, lines[idx + 1]):
                _bump(counts, chapter_counts, "split_word")
                _example(examples, "split_word", chapter, number, line, max_examples_per_pattern, max_line_chars)
        per_chapter[getattr(chapter, "filename", str(chapter))] = chapter_counts

    return {
        "chapter_count": len(per_chapter),
        "total_lines": total_lines,
        "total_findings": sum(counts.values()),
        "counts": counts,
        "patterns": DAMAGE_PATTERNS,
        "examples": examples,
        "per_chapter": {k: v for k, v in per_chapter.items() if v},
    }


def _bump(counts: dict[str, int], chapter_counts: dict[str, int], name: str, by: int = 1) -> None:
    counts[name] = counts.get(name, 0) + by
    chapter_counts[name] = chapter_counts.get(name, 0) + by


def _example(
    examples: dict[str, list[dict[str, Any]]],
    name: str,
    chapter: Any,
    line_number: int,
    text: str,
    limit: int,
    max_line_chars: int,
) -> None:
    bucket = examples[name]
    if len(bucket) >= limit:
        return
    bucket.append(
        {
            "chapter": getattr(chapter, "filename", str(chapter)),
            "line_number": line_number,
            "text": excerpt(text, max_line_chars),
        }
    )


def corpus_digest(report: dict[str, Any]) -> str:
    """Compact text rendering of a corpus damage scan for a prompt."""
    counts = report.get("counts") or {}
    out = [
        f"CORPUS DAMAGE SCAN: {report.get('chapter_count', 0)} chapters, "
        f"{report.get('total_lines', 0)} lines, {report.get('total_findings', 0)} findings."
    ]
    for name, count in counts.items():
        if not count:
            continue
        out.append(f"  {name:<26} {count:>5}  fix: {DAMAGE_PATTERNS[name]['fix']}")
    for name, items in (report.get("examples") or {}).items():
        out.extend(
            f"    e.g. {name} :: {item['chapter']}:{item['line_number']} {item['text']!r}"
            for item in items
        )
    return "\n".join(out)


__all__ = [
    "DAMAGE_PATTERNS",
    "SAMPLE_HEADERS",
    "SAMPLE_HEAD",
    "SAMPLE_LONGEST",
    "SAMPLE_STRATEGIES",
    "SAMPLE_STRIDE",
    "SAMPLE_TAIL",
    "corpus_digest",
    "sample_text",
    "scan_corpus_damage",
]
