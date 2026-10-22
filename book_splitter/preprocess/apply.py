"""Applying a validated stage-2 cleaning plan: pure line rewriting.

The applier never guesses. It re-checks each operation against the file it reads
and drops anything that does not match, so calling it directly with an
unvalidated plan can only ever be conservative.

Three guarantees live here:

1. **No invented line numbers.** A merge can only span lines that exist and
   whose next non-blank line is a header.
2. **No arbitrary rewriting.** ``merge_with_next`` glues two lines; ``demote``
   only removes the ``#`` markup; ``promote`` only adds it to a plain ALL-CAPS
   heading; ``remove`` blanks a page-artefact line. There is no operation that
   can rewrite a sentence.
3. **No silent loss.** :class:`ApplyResult` carries a per-line provenance map
   (``kept`` / ``joined_split_word`` / ``merged_pair`` / ``removed_page_number``
   / ``removed_caption`` / ...), which :mod:`book_splitter.preprocess.verify`
   turns into a proof.

Nothing here writes to disk: the caller (a tool) writes the returned lines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..documents.images import has_images
from ..documents.io import load_lines
from ..documents.markdown import (
    header_of,
    is_plain_heading_candidate,
    is_standalone_line,
    page_number_of,
)
from .damage import (
    INLINE_PAGE_NUMBER,
    PAGE_NUMBER_IN_WORD,
    SPACE_BEFORE_PUNCTUATION,
    SPLIT_WORD,
    repair_inline_page_number,
    repair_page_number_in_word,
    repair_space_before_punctuation,
    repair_split_word,
)
from .inspect import (
    DEMOTE,
    KEEP,
    MERGE_DEMOTE,
    MERGE_WITH_NEXT,
    PROMOTE,
    REMOVE,
    REMOVE_CAPTION,
)
from .plan import (
    looks_like_caption,
    looks_like_page_artefact,
    next_content_line,
    strip_leading_page_number,
)


@dataclass
class LineEdit:
    """What happened to one output line, and where it came from."""

    output_line: int
    origin: str
    source_lines: list[int] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "output_line": self.output_line,
            "origin": self.origin,
            "source_lines": self.source_lines,
            "detail": self.detail,
        }


@dataclass
class ApplyResult:
    """The repaired lines plus the provenance needed to verify them."""

    lines: list[str] = field(default_factory=list)
    provenance: list[LineEdit] = field(default_factory=list)
    #: source line number -> {"decision": ..., "kind": ...} for every touched line
    touched: dict[int, dict[str, Any]] = field(default_factory=dict)
    #: source line numbers of page artefacts this plan deletes
    removed_artefacts: list[int] = field(default_factory=list)
    #: source line numbers of figure/table captions this plan deletes
    removed_captions: list[int] = field(default_factory=list)
    #: source line numbers that survive verbatim (mask: 1 keep → 1 output)
    copied_lines: list[int] = field(default_factory=list)
    #: source line numbers of plain ALL-CAPS headings that gained ``##`` markup
    promoted_headings: list[int] = field(default_factory=list)
    #: ``[left_source_line, right_source_line]`` for every stitched word
    joined_words: list[list[int]] = field(default_factory=list)
    #: merge line -> the page-number digits dropped from the appended half
    merged_digits: dict[int, list[str]] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for entry in self.provenance:
            counts[entry.origin] = counts.get(entry.origin, 0) + 1
        return {
            "output_lines": len(self.lines),
            "provenance_counts": counts,
            "removed_artefacts": self.removed_artefacts,
            "removed_captions": self.removed_captions,
            "promoted_headings": self.promoted_headings,
            "copied_lines": len(self.copied_lines),
            "joined_words": self.joined_words,
            "merged_digits": self.merged_digits,
            "touched": len(self.touched),
        }


def apply_cleaning_plan(
    path: str | Path,
    operations: list[dict],
    text_fixes: list[dict] | None = None,
) -> ApplyResult:
    """Apply validated operations to one chapter's lines — pure, no writes.

    ``text_fixes`` are body-text repairs (``split_word`` / ``page_number_in_word``
    / ``space_before_punctuation``) and are applied **first**, in descending line
    order so the header operations keep referring to the file the agent saw.
    Joining two lines is recorded as a merge, never as an edit: the output is the
    concatenation of the two source fragments, which the verifier re-computes.

    The caller (a tool) writes the returned lines. Keeping this pure is what
    makes it testable and what lets the graph refuse to write an unvalidated
    plan: the apply step re-checks each operation and drops anything that does
    not match the file.
    """
    p = Path(path).expanduser()
    lines = load_lines(p)
    plain = [line.rstrip("\n") for line in lines]

    result = ApplyResult()
    working = list(plain)
    merge_pairs: dict[int, list[int]] = {}
    joined_words: list[list[int]] = []

    for fix in _index_fixes(text_fixes):
        number, kind = fix["line_number"], fix["fix"]
        if number < 1 or number > len(working):
            continue
        if has_images(working[number - 1]):
            continue
        if kind == SPACE_BEFORE_PUNCTUATION:
            working[number - 1] = repair_space_before_punctuation(working[number - 1])
            result.touched[number] = {"fix": kind}
        elif kind == PAGE_NUMBER_IN_WORD:
            working[number - 1] = repair_page_number_in_word(working[number - 1])
            result.touched[number] = {"fix": kind}
        elif kind == INLINE_PAGE_NUMBER:
            working[number - 1] = repair_inline_page_number(working[number - 1])
            result.touched[number] = {"fix": kind}

    for fix in _index_fixes(text_fixes):
        number = fix["line_number"]
        if fix["fix"] != SPLIT_WORD or number in merge_pairs:
            continue
        if number < 1 or number + 1 > len(working):
            continue
        left, right = working[number - 1], working[number]
        if not left.strip() or not right.strip():
            continue
        if has_images(left) or has_images(right):
            continue
        joined, _ = repair_split_word(left, right)
        working[number - 1] = joined + right.lstrip()
        merge_pairs[number] = [number, number + 1]
        joined_words.append([number, number + 1])
        result.touched[number] = {"fix": SPLIT_WORD, "joined_with": number + 1}
        result.touched[number + 1] = {"fix": SPLIT_WORD, "joined_into": number}

    headers = {n for n, text in enumerate(plain, start=1) if header_of(text)}
    by_line: dict[int, dict] = {}
    for op in operations or []:
        try:
            by_line[int(op["line_number"])] = op
        except (KeyError, TypeError, ValueError):
            continue

    consumed: set[int] = set()
    for _, pair in merge_pairs.items():
        consumed.add(pair[1])

    idx = 0
    total = len(plain)
    while idx < total:
        number = idx + 1

        if number in consumed:
            idx += 1
            continue

        if number in merge_pairs:
            result.lines.append(working[number - 1])
            result.provenance.append(
                LineEdit(
                    len(result.lines),
                    "joined_split_word",
                    merge_pairs[number],
                    detail="line break removed; the word continues",
                )
            )
            idx += 1
            continue

        line_op = by_line.get(number)
        decision = (line_op or {}).get("decision", KEEP)
        # Refuse an edit aimed at a picture even if apply is called directly with
        # a plan that never went through validate_cleaning_plan.
        if has_images(plain[number - 1]) and decision != KEEP:
            decision = KEEP

        if decision == MERGE_WITH_NEXT and number in headers:
            nxt = next_content_line(plain, number)
            if nxt is not None and nxt in headers:
                span = list(range(number, nxt + 1))
                next_head = header_of(plain[nxt - 1])
                part, merged_digits = strip_leading_page_number(
                    next_head[1] if next_head else ""
                )
                result.merged_digits[number] = merged_digits
                merged = f"{working[number - 1].rstrip()} {part}"
                result.lines.append(merged)
                result.provenance.append(
                    LineEdit(len(result.lines), "merged_pair", span)
                )
                result.touched[number] = {"decision": decision, "merged_with": nxt}
                result.touched[nxt] = {"decision": MERGE_DEMOTE, "merged_into": number}
                consumed.add(nxt)
                idx += 1
                continue

        if decision in {REMOVE, REMOVE_CAPTION}:
            source = plain[number - 1]
            source_head = header_of(source)
            body = source_head[1] if source_head else source.strip()
            if decision == REMOVE and looks_like_page_artefact(body):
                result.provenance.append(
                    LineEdit(len(result.lines) + 1, "removed_page_number", [number])
                )
                result.removed_artefacts.append(number)
                result.touched[number] = {
                    "decision": decision,
                    "page_number": page_number_of(body),
                }
                # The slot stays, blank: the chapter keeps the same line count, so
                # a line number in a log or a trace still points at the same text.
                result.lines.append("")
                idx += 1
                continue
            if decision == REMOVE_CAPTION and looks_like_caption(source):
                result.provenance.append(
                    LineEdit(len(result.lines) + 1, "removed_caption", [number])
                )
                result.removed_captions.append(number)
                result.touched[number] = {"decision": decision}
                result.lines.append("")
                idx += 1
                continue

        if decision == DEMOTE and number in headers:
            demote_head = header_of(plain[number - 1])
            demoted, digits = strip_leading_page_number(
                demote_head[1] if demote_head else ""
            )
            result.lines.append(demoted)
            result.provenance.append(
                LineEdit(len(result.lines), "demoted_header", [number], detail=demoted)
            )
            result.touched[number] = {"decision": decision, "digits_removed": digits}
            idx += 1
            continue

        if (
            decision == PROMOTE
            and number not in headers
            and is_standalone_line(plain, number - 1)
            and is_plain_heading_candidate(plain[number - 1].strip())
        ):
            promoted = f"## {plain[number - 1].strip()}"
            result.lines.append(promoted)
            result.promoted_headings.append(number)
            result.provenance.append(
                LineEdit(len(result.lines), "promoted_heading", [number], detail=promoted)
            )
            result.touched[number] = {"decision": decision}
            idx += 1
            continue

        body = working[number - 1]
        if number in result.touched:
            result.provenance.append(
                LineEdit(len(result.lines) + 1, "kept", [number], detail="body text repaired")
            )
        else:
            result.provenance.append(LineEdit(len(result.lines) + 1, "kept", [number]))
            result.copied_lines.append(number)
        result.lines.append(body)
        idx += 1

    return result


def _index_fixes(text_fixes: list[dict] | None) -> list[dict]:
    """Normalise a text-fix list, ignoring anything malformed."""
    out: list[dict] = []
    for raw in text_fixes or []:
        if not isinstance(raw, dict):
            continue
        raw_line_number = raw.get("line_number")
        if raw_line_number is None:
            continue
        try:
            number = int(raw_line_number)
        except (TypeError, ValueError):
            continue
        kind = str(raw.get("fix") or raw.get("kind") or "")
        if kind:
            out.append({"line_number": number, "fix": kind})
    return out


def text_of(result: ApplyResult) -> str:
    """The repaired chapter as text (newline-terminated, like the source)."""
    out = [line + "\n" for line in result.lines]
    return "".join(out)

__all__ = [
    "ApplyResult",
    "LineEdit",
    "apply_cleaning_plan",
    "text_of",
]
