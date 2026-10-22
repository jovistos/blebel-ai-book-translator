"""Back matter at the tail of a book: *is there any, and where does it start?*

A flat splitter cuts at chapter starts, so everything after the last chapter
start — author biography, publisher catalogue, acknowledgements, a re-printed
table of contents, teaser excerpts — lands inside the last chapter file. That
back matter is not the book's text and should not be translated as if it were.

This module answers the question **deterministically**, from a bounded window at
the very end of the book, and it never guesses above the window: the window is
the last ``chunk_lines`` lines, and a caller that starts short can ask again for
a larger one. Progress is expressed by two flags:

``escalate``
    the window is too small to show where the trailing block begins — either it
    cuts through the block (its first line is itself back-matter evidence) or it
    shows no back matter at all while more lines remain above it. Ask again with
    ``next_chunk_lines``.
``boundary_visible``
    the first line of the trailing block is *inside* the window, preceded by a
    line that is neither blank nor back-matter evidence. Only then is
    ``back_matter.start_line`` a real boundary rather than a truncation artefact.

The evidence vocabulary is deliberately small and shape-based: publisher
keywords (``Also available``, ``About the author``, ``ISBN``, ``Visit:`` …),
whole-line "about the author" headings in the languages the corpus comes in
(``## O AUTORU``, ``Über den Autor``), series headings, ISBN bullets,
author-bio sentences, and **repeated titles** — lines that occur more than once
in the file, which is exactly what a re-printed table of contents looks like. A
trailing run counts as back matter when it holds at least two evidence lines, or
one that is structurally unambiguous (a repeated title, an ISBN, a publisher
catalogue line, an "about the author" heading).

Only predicates and reports live here: the agent decides, the slicer writes.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection
from typing import Any

from ..core.coerce import as_int as _as_int
from ..core.constants import (
    BACK_MATTER_FILENAME,
    DEFAULT_TAIL_CHUNK_LINES,
    MAX_BACK_MATTER_CANDIDATES,
    MAX_BACK_MATTER_FRACTION,
    MAX_BACK_MATTER_LINES,
    MAX_TAIL_CHUNK_LINES,
    MAX_TAIL_LINE_CHARS,
    MAX_TAIL_REPORT_CHARS,
    MAX_TAIL_WINDOW_CHARS,
    MAX_TITLE_LINE_CHARS,
    MIN_BACK_MATTER_LINES,
    MIN_REPEAT_TITLE_CHARS,
    MIN_REPEAT_TITLE_WORDS,
    MIN_TAIL_CHUNK_LINES,
)
from ..documents.text import excerpt, jdump, norm
from .patterns import (
    ABOUT_AUTHOR_HEADING_RE,
    AUTHOR_BIO_RE,
    BACK_MATTER_KEYWORDS,
    ISBN_LINE_RE,
    SERIES_HEADING_RE,
)

#: The role a split point carries when it opens the back matter.
BACK_MATTER_ROLE = "back-matter"

#: Accepted spellings of that role in a submitted map.
BACK_MATTER_ROLES: frozenset[str] = frozenset(
    {BACK_MATTER_ROLE, "back_matter", "backmatter", "back matter"}
)

#: Evidence that is on its own enough to call a trailing block back matter.
STRUCTURAL_SIGNALS: frozenset[str] = frozenset(
    {
        "repeated-title",
        "isbn",
        "publisher-catalog",
        "series-heading",
        "about-author-heading",
    }
)

#: Every signal, i.e. the full back-matter vocabulary.
BACK_MATTER_SIGNALS: frozenset[str] = STRUCTURAL_SIGNALS | {
    "about-author",
    "author-bio",
    "acknowledgements",
    "copyright",
    "excerpt",
    "reading-group",
    "publisher-links",
}


def is_back_matter_role(value: Any) -> bool:
    """True when a map entry declares itself as the back-matter split."""
    return str(value or "").strip().lower() in BACK_MATTER_ROLES


def back_matter_line_limit(line_count: int) -> int:
    """How many trailing lines may still be called back matter.

    Bounded twice on purpose: an absolute cap keeps a 100 000-line book from
    handing a third of itself to the "back matter" role, and a share cap keeps a
    short book honest. The role is the agent's assertion; this is the guardrail
    that stops it from becoming a way to chop a chapter in half.
    """
    share = max(MIN_BACK_MATTER_LINES, int(max(0, line_count) * MAX_BACK_MATTER_FRACTION))
    return max(1, min(MAX_BACK_MATTER_LINES, share, max(1, line_count)))


def repeated_title_set(lines: list[str]) -> set[str]:
    """Normalised text of the title-looking lines that occur MORE THAN ONCE.

    This is how a re-printed table of contents is recognised: the chapter list
    at the back of the book repeats the one at the front, so those lines have a
    twin somewhere else in the file. Counting over the whole file (not just the
    lines above the window) keeps the signal working whatever the window size —
    a window wide enough to swallow the front TOC must not blind the detector.
    """
    counts: Counter[str] = Counter()
    for line in lines:
        text = line.strip()
        if not MIN_REPEAT_TITLE_CHARS <= len(text) <= MAX_TITLE_LINE_CHARS:
            continue
        if len(text.split()) < MIN_REPEAT_TITLE_WORDS:
            continue
        folded = norm(text)
        if folded:
            counts[folded] += 1
    return {text for text, count in counts.items() if count >= 2}


def line_signals(text: str, repeated_titles: Collection[str] = ()) -> list[str]:
    """Every back-matter signal carried by one line, sorted and de-duplicated."""
    stripped = text.strip()
    if not stripped:
        return []
    out: set[str] = set()
    low = stripped.lower()
    for needle, name in BACK_MATTER_KEYWORDS:
        if needle in low:
            out.add(name)
    if ABOUT_AUTHOR_HEADING_RE.match(stripped):
        # A whole-line heading is structural evidence on its own, even when the
        # biography underneath it is written in a language AUTHOR_BIO_RE does
        # not know: ``## O AUTORU`` is never chapter prose.
        out.add("about-author-heading")
    if SERIES_HEADING_RE.match(stripped):
        out.add("series-heading")
    if ISBN_LINE_RE.search(stripped):
        out.add("isbn")
    if AUTHOR_BIO_RE.search(stripped):
        out.add("author-bio")
    folded = norm(stripped)
    if folded and folded in repeated_titles:
        out.add("repeated-title")
    return sorted(out)


def clamp_chunk(chunk_lines: Any, line_count: int) -> int:
    """Force a requested window size into ``[MIN, MAX]`` and inside the file."""
    try:
        wanted = int(chunk_lines)
    except (TypeError, ValueError):
        wanted = DEFAULT_TAIL_CHUNK_LINES
    wanted = max(MIN_TAIL_CHUNK_LINES, min(wanted, MAX_TAIL_CHUNK_LINES))
    return max(1, min(wanted, max(1, line_count)))


def detect_back_matter(
    lines: list[str],
    chunk_lines: int = DEFAULT_TAIL_CHUNK_LINES,
    next_index: int = 0,
    back_matter_filename: str | None = None,
    file_label: str = "",
) -> dict[str, Any]:
    """Report whether the trailing window holds back matter, and where it starts.

    ``next_index`` is the number the agent gave its last chapter plus one, used
    to name the file ``NN-back-matter.md``; ``back_matter_filename`` overrides
    that outright.
    """
    total = len(lines)
    chunk = clamp_chunk(chunk_lines, total)
    start_idx = max(0, total - chunk)
    window = lines[start_idx:]
    repeated = repeated_title_set(lines)
    signals = [line_signals(line, repeated) for line in window]

    # The trailing run: walk up from EOF while lines are blank or evidence.
    # A prose line stops the walk, so a stray keyword inside the chapter cannot
    # pull the boundary down into the story.
    run_start: int | None = None
    for i in range(len(window) - 1, -1, -1):
        if _is_ish(window[i], signals[i]):
            run_start = i
        else:
            break

    run_signals: list[str] = []
    evidence_lines: list[int] = []
    if run_start is not None:
        for i in range(run_start, len(window)):
            if signals[i]:
                evidence_lines.append(start_idx + i + 1)
                run_signals.extend(signals[i])
    distinct = sorted(set(run_signals))
    structural = [s for s in distinct if s in STRUCTURAL_SIGNALS]

    # The boundary is the first line of TEXT in the run: the blank line that
    # separates the story from the block belongs to the chapter, not to the
    # block, and a split point may not be blank.
    block_idx: int | None = None
    if run_start is not None:
        for i in range(run_start, len(window)):
            if window[i].strip():
                block_idx = i
                break

    # Two evidence lines are needed unless the one line is structurally
    # unambiguous (an ISBN bullet or a repeated title is never chapter prose).
    detected = bool(evidence_lines) and (len(evidence_lines) >= 2 or bool(structural))

    more_above = start_idx > 0
    # The run reaching the window's first line means the block may continue
    # above it, blank lines included — the window cannot show the boundary.
    cuts_through_block = detected and run_start == 0
    escalate = bool(
        more_above and chunk < MAX_TAIL_CHUNK_LINES and (not detected or cuts_through_block)
    )
    next_chunk = min(chunk * 2, MAX_TAIL_CHUNK_LINES) if escalate else None
    # Visible boundary = the ish-run starts below the window top (so a real,
    # non-back-matter line was seen above it) and holds a line of text.
    boundary_visible = bool(detected and run_start is not None and run_start > 0 and block_idx is not None)

    first_evidence = evidence_lines[0] if evidence_lines else None
    start_line = start_idx + block_idx + 1 if boundary_visible and block_idx is not None else None

    confidence = "low"
    if detected and run_start is not None:
        repeats = sum(1 for i in range(run_start, len(window)) if "repeated-title" in signals[i])
        confidence = "high" if repeats >= 3 or len(structural) >= 2 else "medium"

    proposed = None
    if start_line is not None:
        proposed = {
            "line_number": start_line,
            "filename": _back_matter_name(next_index, back_matter_filename),
            "role": BACK_MATTER_ROLE,
            "title": excerpt(_first_text(window, block_idx or 0), 80),
        }

    return _fit_report(
        {
            "file": str(file_label or ""),
            "line_count": total,
            "requested_chunk_lines": _as_int(chunk_lines, DEFAULT_TAIL_CHUNK_LINES),
            "chunk_lines": chunk,
            "window": _window_payload(lines, start_idx, signals, keep_line=start_line if proposed else None),
            "more_above": more_above,
            "escalate": escalate,
            "next_chunk_lines": next_chunk,
            "back_matter": {
                "detected": detected,
                "boundary_visible": boundary_visible,
                "start_line": start_line,
                "first_evidence_line": first_evidence,
                "signals": distinct,
                "evidence_line_count": len(evidence_lines),
                "confidence": confidence,
                "reason": _reason(
                    detected=detected,
                    boundary_visible=boundary_visible,
                    escalate=escalate,
                    cuts_through_block=cuts_through_block,
                    distinct=distinct,
                    chunk=chunk,
                    more_above=more_above,
                ),
            },
            "candidates": _candidates(lines, start_idx, signals),
            "proposed_split": proposed,
            "recommended_action": _action(
                detected=detected,
                boundary_visible=boundary_visible,
                escalate=escalate,
                next_chunk=next_chunk,
                start_line=start_line,
                first_evidence=first_evidence,
                more_above=more_above,
            ),
            "note": (
                "TAIL WINDOW ONLY - the last lines of the book, each clipped, the whole "
                "window capped. Not the book text."
            ),
        }
    )


def _is_ish(line: str, signals: list[str]) -> bool:
    """Back-matter-ish: a blank line, or a line carrying any evidence."""
    return (not line.strip()) or bool(signals)


def _report_size(payload: dict[str, Any]) -> int:
    """Exact serialised size, measured the way the tool will send it."""
    return len(jdump(payload, limit=10**9))


def _fit_report(payload: dict[str, Any], limit: int = MAX_TAIL_REPORT_CHARS) -> dict[str, Any]:
    """Trim the report until it fits the per-tool context guardrail.

    The window and the candidate list both repeat line text, so together they can
    outgrow the 24 000-character cap and be cut mid-JSON — which would hand the
    model an unparseable result. Drop the OLDEST lines first, in chunks, so what
    survives is always the true tail — and put the proposed split line back if
    the trimming removed it.
    """
    if _report_size(payload) <= limit:
        return payload

    proposed = payload.get("proposed_split") or {}
    keep = next(
        (
            entry
            for entry in payload["window"]["lines"]
            if entry["line_number"] == proposed.get("line_number")
        ),
        None,
    )

    window = payload["window"]["lines"]
    while window and _report_size(payload) > limit:
        del window[: max(1, len(window) // 8)]
        payload["window"]["start_line"] = window[0]["line_number"] if window else None
        payload["window"]["line_count"] = len(window)
        payload["window"]["truncated_at_chars"] = True

    candidates = payload["candidates"]["lines"]
    while candidates and _report_size(payload) > limit:
        del candidates[: max(1, len(candidates) // 4)]
        payload["candidates"]["truncated"] = True

    if keep is not None and all(e["line_number"] != keep["line_number"] for e in window):
        window.insert(0, keep)
        payload["window"]["start_line"] = keep["line_number"]
        payload["window"]["line_count"] = len(window)
        payload["window"]["truncated_at_chars"] = True
    return payload


def _back_matter_name(next_index: int, back_matter_filename: str | None) -> str:
    if back_matter_filename:
        return str(back_matter_filename)
    try:
        index = int(next_index)
    except (TypeError, ValueError):
        index = 0
    if index > 0:
        return f"{index:02d}-{BACK_MATTER_FILENAME}"
    return BACK_MATTER_FILENAME


def _first_text(window: list[str], index: int) -> str:
    """First non-blank line at or after ``index`` (the block's own first line)."""
    for line in window[max(0, index) :]:
        if line.strip():
            return line
    return ""


def _window_payload(
    lines: list[str],
    start_idx: int,
    signals: list[list[str]],
    keep_line: int | None = None,
) -> dict[str, Any]:
    """The window text, built from the end backwards so the true tail survives.

    ``keep_line`` is force-included even when the character budget would have
    trimmed it away: the agent must always be able to see the line it is told to
    split at.
    """
    shown: list[dict[str, Any]] = []
    used = 0
    truncated = False
    for i in range(len(lines) - 1, start_idx - 1, -1):
        text = excerpt(lines[i].rstrip("\n"), MAX_TAIL_LINE_CHARS)
        if used + len(text) > MAX_TAIL_WINDOW_CHARS:
            truncated = True
            break
        used += len(text)
        shown.append({"line_number": i + 1, "signals": signals[i - start_idx], "text": text})
    shown.reverse()

    if (
        keep_line
        and 1 <= keep_line <= len(lines)
        and all(entry["line_number"] != keep_line for entry in shown)
    ):
        index = keep_line - start_idx - 1
        shown.insert(
            0,
            {
                "line_number": keep_line,
                "signals": signals[index] if 0 <= index < len(signals) else [],
                "text": excerpt(lines[keep_line - 1].rstrip("\n"), MAX_TAIL_LINE_CHARS),
            },
        )
        truncated = True

    return {
        "start_line": shown[0]["line_number"] if shown else None,
        "end_line": len(lines),
        "line_count": len(shown),
        "truncated_at_chars": truncated,
        "lines": shown,
    }


def _candidates(lines: list[str], start_idx: int, signals: list[list[str]]) -> dict[str, Any]:
    """The evidence lines of the window, for a reader who wants to verify."""
    found = [
        {
            "line_number": start_idx + i + 1,
            "text": excerpt(lines[start_idx + i], MAX_TAIL_LINE_CHARS),
            "signals": signals[i],
        }
        for i in range(len(signals))
        if signals[i]
    ]
    return {
        "count": len(found),
        "truncated": len(found) > MAX_BACK_MATTER_CANDIDATES,
        "lines": found[:MAX_BACK_MATTER_CANDIDATES],
    }


def _reason(
    *,
    detected: bool,
    boundary_visible: bool,
    escalate: bool,
    cuts_through_block: bool,
    distinct: list[str],
    chunk: int,
    more_above: bool,
) -> str:
    if not more_above:
        return "the whole book is inside this window; " + (
            "back matter evidence present" if detected else "no back matter evidence"
        )
    if cuts_through_block:
        return (
            f"the window cuts through a back-matter block (signals: "
            f"{', '.join(distinct) or 'none'}); the block starts more than {chunk} lines "
            "above the end"
        )
    if detected and boundary_visible:
        return f"back matter begins inside the window (signals: {', '.join(distinct)})"
    if detected:
        return f"back matter evidence at the very end (signals: {', '.join(distinct)})"
    if escalate:
        return "no back matter evidence in this window; more lines remain above, so look higher"
    return "no back matter evidence in the whole file"


def _action(
    *,
    detected: bool,
    boundary_visible: bool,
    escalate: bool,
    next_chunk: int | None,
    start_line: int | None,
    first_evidence: int | None,
    more_above: bool,
) -> str:
    if escalate:
        return (
            f"window is not decisive: call inspect_book_tail again with "
            f"chunk_lines={next_chunk} (larger window) before deciding"
        )
    if detected and boundary_visible:
        return (
            f"back matter starts at line {start_line}: add the returned proposed_split "
            "(role=back-matter) as the LAST entry of your map"
        )
    if detected:
        return (
            "back matter is present but its first line is above the largest window "
            f"(first evidence seen at line {first_evidence}); use find_lines_matching to "
            "locate where the block opens, then propose that line with role=back-matter "
            "as the last entry"
        )
    if more_above:
        return (
            "no back matter in the largest window checked; do not add a back-matter "
            "split unless find_lines_matching proves otherwise"
        )
    return "no back matter at the tail: do not add a back-matter split"


__all__ = [
    "BACK_MATTER_ROLE",
    "BACK_MATTER_ROLES",
    "BACK_MATTER_SIGNALS",
    "STRUCTURAL_SIGNALS",
    "back_matter_line_limit",
    "clamp_chunk",
    "detect_back_matter",
    "is_back_matter_role",
    "line_signals",
    "repeated_title_set",
]
