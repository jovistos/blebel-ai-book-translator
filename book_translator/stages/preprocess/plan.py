"""What a stage-2 cleaning plan may say, and which plans are legal.

An agent never edits text. It *describes* edits — per line, from a fixed
vocabulary — and this module checks every description against the real file
before :mod:`book_translator.stages.preprocess.apply` performs it. Invalid plans produce
concrete corrections and are sent back to the agent.

This module also owns the text-shape predicates the vocabulary depends on
(``looks_like_page_artefact``, ``looks_like_caption``,
``strip_leading_page_number``, ``next_content_line``), so the validator and the
applier agree on what a legal edit means. Nothing here writes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from book_translator.core.constants import MAX_PAGE_NUMBER_DIGITS
from book_translator.documents.bodyline import is_caption_line
from book_translator.documents.images import has_images
from book_translator.documents.io import load_lines
from book_translator.documents.markdown import (
    header_of,
    is_page_number_line,
    is_plain_heading_candidate,
    is_standalone_line,
)
from book_translator.documents.text import excerpt

from .damage import FIXES, scan_chapter_damage
from .inspect import (
    DECISIONS,
    KEEP,
    MERGE_WITH_NEXT,
    PROMOTE,
    REMOVE,
    REMOVE_CAPTION,
)


def looks_like_page_artefact(text: str) -> bool:
    """True when a line is *only* a printed page number.

    Works with or without ``#`` markup, because the converter marked some page
    numbers as headings and left others as bare body lines.
    """
    return is_page_number_line(text) or is_page_number_line(f"# {text}")


def looks_like_caption(text: str) -> bool:
    """True when a line is a standalone figure/table caption.

    This is the one decision that deletes words, so it is checked twice: once
    from the header record (``has_caption_marker``) and once here, on the raw
    text, right before the delete happens.
    """
    return is_caption_line(text)


def strip_leading_page_number(text: str) -> tuple[str, list[str]]:
    """Drop a printed page number that opens a demoted/merged heading.

    ``"75 lič nosti. Impuls ..."`` → ``("lič nosti. Impuls ...", ["75"])`` and
    ``"1otkucaja u minuti"`` → ``("otkucaja u minuti", ["1"])``. Both shapes are
    page artefacts rather than text: the digits are followed by a *lowercase*
    fragment, which is exactly how a page number interrupts a word or a label.

    A real numbered item (``1. Neki citat``, ``10. POGLAVLJE``) or a heading that
    carries a number inside its own words (``Slika 11. ...``, ``GLAVA 10``) is
    left untouched. Returns the repaired text and the digits removed.
    """
    stripped = text.lstrip()
    if not re.match(r"^\d+\.", stripped):
        # ``76 Slika 11 ...`` — a leading page number followed by a new sentence.
        # A real numbered item (``1. Neki citat``) carries the full stop and is
        # left alone.
        before_sentence = re.match(
            rf"^(\d{{1,{MAX_PAGE_NUMBER_DIGITS}}})\s+(?=[A-Za-zžćčšđ])", stripped
        )
        if before_sentence:
            return stripped[before_sentence.end() :], [before_sentence.group(1)]
    match = re.match(rf"^(\d{{1,{MAX_PAGE_NUMBER_DIGITS}}})\s+(?=[a-zžćčšđ])", stripped)
    if match:
        return stripped[match.end() :], [match.group(1)]
    glued = re.match(rf"^(\d{{1,{MAX_PAGE_NUMBER_DIGITS}}})(?=[a-zžćčšđ])", stripped)
    if glued:
        return stripped[glued.end() :], [glued.group(1)]
    return text, []


def next_content_line(lines: list[str], line_number: int) -> int | None:
    """The first non-blank line number after ``line_number``, or ``None``.

    A split heading is usually separated from its other half by exactly one
    blank line (``## GLAVA 2`` / ```` / ``## SEKS I SRCE``), so a merge has to
    look *through* the blanks.
    """
    for number in range(line_number + 1, len(lines) + 1):
        if lines[number - 1].strip():
            return number
    return None


def _header_lines(lines: list[str]) -> set[int]:
    return {i for i, line in enumerate(lines, start=1) if header_of(line)}


def validate_text_fixes(
    path: str | Path,
    text_fixes: list[dict],
    *,
    max_fixes: int = 500,
) -> tuple[list[dict], list[str], list[str]]:
    """Check requested body-text fixes against a fresh damage scan.

    Returns ``(accepted, issues, warnings)``. A fix is accepted only when the
    scan actually found that damage at that line, so an agent cannot describe a
    repair for a spot that is not damaged.
    """
    issues: list[str] = []
    warnings: list[str] = []
    if not text_fixes:
        return [], issues, warnings
    report = scan_chapter_damage(path)
    by_spot = {(f["kind"], f["line_number"]): f for f in report["findings"]}
    accepted: list[dict] = []
    claimed: set[tuple[str, int]] = set()
    if len(text_fixes) > max_fixes:
        return [], [f"too many text fixes ({len(text_fixes)} > {max_fixes})"], []
    for index, raw in enumerate(text_fixes):
        if not isinstance(raw, dict):
            issues.append(f"text fix #{index + 1} is not an object")
            continue
        kind = str(raw.get("fix") or raw.get("kind") or "").strip()
        raw_line_number = raw.get("line_number")
        if raw_line_number is None:
            issues.append(f"text fix #{index + 1}: line_number must be an integer")
            continue
        try:
            line_number = int(raw_line_number)
        except (TypeError, ValueError):
            issues.append(f"text fix #{index + 1}: line_number must be an integer")
            continue
        if kind not in FIXES:
            issues.append(
                f"text fix #{index + 1}: fix {kind!r} is not one of {sorted(FIXES)}"
            )
            continue
        key = (kind, line_number)
        if key in claimed:
            issues.append(f"text fix #{index + 1}: duplicate fix for line {line_number}")
            continue
        if key not in by_spot:
            issues.append(
                f"text fix #{index + 1}: no {kind} damage was found on line {line_number}"
            )
            continue
        claimed.add(key)
        accepted.append(
            {
                "line_number": line_number,
                "fix": kind,
                "note": excerpt(str(raw.get("note") or ""), 120),
            }
        )
    accepted.sort(key=lambda f: f["line_number"])
    return accepted, issues, warnings


def validate_cleaning_plan(
    path: str | Path,
    operations: list[dict],
    *,
    max_operations: int = 500,
) -> dict[str, Any]:
    """Check every operation against the real chapter; never write anything.

    Returns ``{"ok", "issues", "warnings", "operations", "chapter"}``. Each
    issue names the operation and says exactly what is wrong, so the agent can
    correct the plan without being shown any prose.
    """
    p = Path(path).expanduser()
    lines = load_lines(p)
    headers = _header_lines(lines)
    issues: list[str] = []
    warnings: list[str] = []
    normalised: list[dict] = []

    if not isinstance(operations, list) or not operations:
        return {
            "ok": False,
            "issues": ["operations must be a non-empty list of {line_number, decision}"],
            "warnings": [],
            "operations": [],
            "chapter": {"filename": p.name, "line_count": len(lines)},
        }
    if len(operations) > max_operations:
        return {
            "ok": False,
            "issues": [f"too many operations ({len(operations)} > {max_operations})"],
            "warnings": [],
            "operations": [],
            "chapter": {"filename": p.name, "line_count": len(lines)},
        }

    claimed: set[int] = set()
    for index, raw in enumerate(operations):
        if not isinstance(raw, dict):
            issues.append(f"operation #{index + 1} is not an object")
            continue
        raw_line_number = raw.get("line_number")
        if raw_line_number is None:
            issues.append(f"operation #{index + 1}: line_number must be an integer")
            continue
        try:
            line_number = int(raw_line_number)
        except (TypeError, ValueError):
            issues.append(f"operation #{index + 1}: line_number must be an integer")
            continue
        decision = str(raw.get("decision") or "").strip().lower()
        label = f"line {line_number}"
        if decision not in DECISIONS:
            issues.append(f"{label}: decision {decision!r} is not one of {list(DECISIONS)}")
            continue
        if line_number < 1 or line_number > len(lines):
            issues.append(f"{label}: out of range (chapter has {len(lines)} lines)")
            continue
        if line_number in claimed:
            issues.append(f"{label}: two operations target the same line")
            continue

        current = lines[line_number - 1]
        # An image reference is not text: no cleaning decision may edit, move or
        # delete the line that carries it. This is the stage-2 guarantee that the
        # pictures stage 0 extracted survive into the translated book.
        if has_images(current):
            if decision != KEEP:
                issues.append(
                    f"{label}: holds a Markdown image, and image lines are preserved "
                    "exactly; use keep (or omit the line from the plan)"
                )
            else:
                claimed.add(line_number)
                normalised.append(
                    {
                        "line_number": line_number,
                        "decision": decision,
                        "note": excerpt(str(raw.get("note") or ""), 120),
                    }
                )
            continue
        removing = decision in {REMOVE, REMOVE_CAPTION}
        if decision == PROMOTE:
            if line_number in headers:
                issues.append(f"{label}: already has '#' markup, so it cannot be promoted")
                continue
            if not is_standalone_line(lines, line_number - 1) or not is_plain_heading_candidate(
                current.strip()
            ):
                issues.append(
                    f"{label}: promote is only allowed for a standalone ALL-CAPS heading "
                    "that lost its '#' markup"
                )
                continue
        elif decision != KEEP and not removing and line_number not in headers:
            issues.append(f"{label}: not a header line, so it cannot be {decision}d")
            continue

        if decision == MERGE_WITH_NEXT:
            nxt = next_content_line(lines, line_number)
            if nxt is None or nxt not in headers:
                issues.append(
                    f"{label}: merge_with_next needs the next non-blank line to be a header"
                )
                continue
            if is_page_number_line(lines[nxt - 1]):
                issues.append(
                    f"{label}: the next non-blank line is a page number; remove it instead of merging into it"
                )
                continue
            if len(lines[line_number - 1].rstrip("\n")) > 400:
                warnings.append(f"{label}: merging a long line; check the result by eye")
        if decision == REMOVE:
            head = header_of(current)
            body = head[1] if head else current.strip()
            if not looks_like_page_artefact(body):
                issues.append(
                    f"{label}: remove is only allowed for a line that is only a printed "
                    f"page number; {excerpt(body, 40)!r} is not one"
                )
                continue
            if line_number == len(lines) or lines[line_number].strip():
                warnings.append(
                    f"{label}: the line after the page number is not blank; "
                    "line numbering of the untouched text still shifts by one"
                )
        if decision == REMOVE_CAPTION and not looks_like_caption(current):
            issues.append(
                f"{label}: remove_caption is only allowed for a standalone figure or "
                f"table caption; {excerpt(current, 40)!r} does not look like one"
            )
            continue

        claimed.add(line_number)
        normalised.append(
            {
                "line_number": line_number,
                "decision": decision,
                "note": excerpt(str(raw.get("note") or ""), 120),
            }
        )

    normalised.sort(key=lambda op: op["line_number"])
    return {
        "ok": not issues,
        "issues": issues,
        "warnings": warnings,
        "operation_count": len(normalised),
        "operations": normalised,
        "chapter": {"filename": p.name, "line_count": len(lines)},
    }

__all__ = [
    "looks_like_caption",
    "looks_like_page_artefact",
    "next_content_line",
    "strip_leading_page_number",
    "validate_cleaning_plan",
    "validate_text_fixes",
]
