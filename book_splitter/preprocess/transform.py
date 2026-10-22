"""The cleaning plan: validation and application (stage 2).

Stage 2 borrows stage 1's central safety idea. An agent never edits text; it
*describes* edits — per line, from a fixed vocabulary — and a deterministic
validator checks every description against the real file before a deterministic
applier performs it. Invalid plans produce concrete corrections and are sent
back to the agent.

Three guarantees make the result auditable:

1. **No invented line numbers.** Every referenced line must exist and must
   actually be a header (for header decisions) or carry the damage it claims.
2. **No arbitrary rewriting.** ``merge_with_next`` glues two lines; ``demote``
   only removes the ``#`` markup; ``promote`` only adds it to a plain ALL-CAPS
   heading; ``remove`` deletes a page-artefact line. There is no operation that
   can rewrite a sentence.
3. **No silent loss.** :func:`apply_cleaning_plan` returns a per-line provenance
   map (``kept`` / ``joined_into`` / ``merged_pair`` / ``removed_artefact``), so
   the verifier can prove that the only lines missing from the output are lines
   that were explicitly declared to be page artefacts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.constants import MAX_PAGE_NUMBER_DIGITS
from ..documents.bodyline import (
    count_space_before_punct,
    has_midword_number,
    is_caption_line,
    is_word_continuation,
)
from ..documents.images import has_images, image_refs
from ..documents.io import load_lines
from ..documents.markdown import (
    header_of,
    is_page_number_line,
    is_plain_heading_candidate,
    is_standalone_line,
    page_number_of,
)
from ..documents.text import excerpt
from .damage import (
    FIXES,
    INLINE_PAGE_NUMBER,
    PAGE_NUMBER_IN_WORD,
    SPACE_BEFORE_PUNCTUATION,
    SPLIT_WORD,
    inline_page_number_candidate,
    repair_inline_page_number,
    repair_page_number_in_word,
    repair_space_before_punctuation,
    repair_split_word,
    scan_chapter_damage,
)
from .inspect import (
    DECISIONS,
    DEMOTE,
    KEEP,
    MERGE_DEMOTE,
    MERGE_WITH_NEXT,
    PROMOTE,
    REMOVE,
    REMOVE_CAPTION,
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
        try:
            line_number = int(raw.get("line_number"))
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
        try:
            line_number = int(raw.get("line_number"))
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
            body = header_of(current)[1] if header_of(current) else current.strip()
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

        op = by_line.get(number)
        decision = (op or {}).get("decision", KEEP)
        # Refuse an edit aimed at a picture even if apply is called directly with
        # a plan that never went through validate_cleaning_plan.
        if has_images(plain[number - 1]) and decision != KEEP:
            decision = KEEP

        if decision == MERGE_WITH_NEXT and number in headers:
            nxt = next_content_line(plain, number)
            if nxt is not None and nxt in headers:
                span = list(range(number, nxt + 1))
                part, merged_digits = strip_leading_page_number(
                    header_of(plain[nxt - 1])[1] if header_of(plain[nxt - 1]) else ""
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
            body = header_of(source)[1] if header_of(source) else source.strip()
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
            demoted, digits = strip_leading_page_number(header_of(plain[number - 1])[1])
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
        try:
            number = int(raw.get("line_number"))
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


def verify_cleaning_plan(
    path: str | Path,
    operations: list[dict],
    text_fixes: list[dict] | None = None,
) -> dict[str, Any]:
    """Prove that a plan changed only what its own rules allow.

    The check is a *content* comparison, not a diff of intentions. For every
    source line it computes what the output is allowed to contain and compares
    that with what was written:

    * a line the plan never mentions must appear **verbatim** in the output;
    * a ``demote`` line may lose its ``#`` markers and a leading page number, and
      nothing else;
    * a ``remove`` line must be a pure page number and must be gone;
    * a ``merge_with_next`` pair must be one output line whose text is the two
      halves joined;
    * a ``split_word`` pair must be one output line whose text is the two lines
      joined word by word.

    Any character that falls outside those rules is reported as a problem.
    """
    p = Path(path).expanduser()
    lines = [line.rstrip("\n") for line in load_lines(p)]
    fixes = _index_fixes(text_fixes)
    result = apply_cleaning_plan(p, operations, fixes)
    output = list(result.lines)
    fixed_lines = {fix["line_number"] for fix in fixes}

    problems: list[str] = []
    accounted: set[int] = set()
    for entry in result.provenance:
        accounted.update(entry.source_lines)
    for number in range(1, len(lines) + 1):
        if number not in accounted:
            problems.append(f"line {number} vanished without a provenance entry")

    by_line: dict[int, dict] = {}
    for op in operations or []:
        if isinstance(op, dict) and str(op.get("line_number", "")).strip().lstrip("-").isdigit():
            by_line[int(op["line_number"])] = op

    # Source lines that a fix or a merge legitimately rewrote are checked below.
    rewritten = set(fixed_lines)
    for entry in result.provenance:
        if entry.origin == "merged_pair":
            rewritten.update(entry.source_lines)
    for kind, line_number in {(f["fix"], f["line_number"]) for f in fixes}:
        if kind == SPLIT_WORD:
            rewritten.add(line_number + 1)

    for number, source in enumerate(lines, start=1):
        if number in rewritten:
            continue
        op = by_line.get(number) or {}
        decision = str(op.get("decision") or "").lower()
        head = header_of(source)
        body = head[1] if head else source

        if has_images(source):
            if source not in output:
                problems.append(
                    f"line {number} holds a Markdown image but is not in the output "
                    f"verbatim: {excerpt(source, 60)!r}"
                )
            if decision and decision != "keep":
                problems.append(
                    f"line {number} holds a Markdown image and cannot be {decision}d"
                )
            continue

        if decision == REMOVE:
            if not looks_like_page_artefact(body):
                problems.append(
                    f"line {number} was removed but is not a page artefact: {excerpt(body, 40)!r}"
                )
            if number not in result.removed_artefacts:
                problems.append(f"line {number} was declared remove but is still in the output")
            elif output[entry_line(result, number)] != "":
                problems.append(f"line {number} was removed but its slot is not blank")
            continue

        if decision == REMOVE_CAPTION:
            if not looks_like_caption(source):
                problems.append(
                    f"line {number} was removed as a caption but does not look like one: "
                    f"{excerpt(source, 40)!r}"
                )
            if number not in result.removed_captions:
                problems.append(
                    f"line {number} was declared remove_caption but is still in the output"
                )
            elif output[entry_line(result, number)] != "":
                problems.append(f"line {number} was removed but its slot is not blank")
            continue

        if decision == DEMOTE:
            expected, _ = strip_leading_page_number(body)
            if expected not in output:
                problems.append(
                    f"line {number} was demoted but its text is not in the output: "
                    f"{excerpt(expected, 40)!r}"
                )
            continue

        if decision == PROMOTE:
            expected = f"## {source.strip()}"
            if expected not in output:
                problems.append(
                    f"line {number} was promoted but its heading is not in the output: "
                    f"{excerpt(expected, 40)!r}"
                )
            continue

        if source not in output:
            problems.append(
                f"line {number} was not declared as changed but its text is not in the "
                f"output verbatim: {excerpt(source, 40)!r}"
            )

    # A merge must be exactly the two halves joined, with the second half's
    # leading page number dropped.
    for entry in result.provenance:
        if entry.origin != "merged_pair":
            continue
        first, last = entry.source_lines[0], entry.source_lines[-1]
        left = lines[first - 1].rstrip()
        right = header_of(lines[last - 1])[1] if header_of(lines[last - 1]) else ""
        right, _ = strip_leading_page_number(right)
        expected = f"{left} {right}".strip()
        if expected not in output:
            problems.append(
                f"merged line for {first}+{last} is not the two halves joined: "
                f"{excerpt(expected, 40)!r}"
            )

    fix_report = verify_text_fixes(p, fixes)
    problems.extend(fix_report["problems"])

    declared = {
        int(op["line_number"])
        for op in operations or []
        if str(op.get("line_number", "")).strip().lstrip("-").isdigit()
    }
    undeclared = sorted(set(result.removed_artefacts) - declared)
    undeclared_captions = sorted(set(result.removed_captions) - declared)
    for number in undeclared:
        if number in {i for i, line in enumerate(lines, start=1) if header_of(line)}:
            problems.append(f"line {number} was removed without being declared in the plan")

    for number in undeclared_captions:
        problems.append(f"line {number} was removed as a caption without being declared")

    source_images = [ref for line in lines for ref in image_refs(line)]
    output_images = [ref for line in output for ref in image_refs(line)]
    if source_images != output_images:
        problems.append(
            "the cleaning plan changed the chapter's image references "
            f"({len(source_images)} in, {len(output_images)} out); image lines must "
            "be preserved verbatim and in order"
        )

    return {
        "ok": not problems,
        "problems": problems,
        "removed_artefacts": result.removed_artefacts,
        "removed_captions": result.removed_captions,
        "undeclared_removals": undeclared + undeclared_captions,
        "text_fixes": fix_report,
        "output_lines": len(output),
        "source_lines": len(lines),
        "provenance": result.summary(),
    }


def entry_line(result: ApplyResult, source_line: int) -> int:
    """The output index (0-based) of a source line's provenance entry."""
    for entry in result.provenance:
        if source_line in entry.source_lines:
            return entry.output_line - 1
    return -1


def verify_text_fixes(path: str | Path, text_fixes: list[dict]) -> dict[str, Any]:
    """Re-check every applied body-text fix against its own repair rule.

    Each fix kind has a rule a machine can re-run, so a repair is proven to have
    done *only* what its kind allows: a ``page_number_in_word`` fix must have
    removed digits and nothing else, a ``space_before_punctuation`` fix must have
    removed spaces and nothing else, and a ``split_word`` fix must have joined
    two lines that really are one word.
    """
    p = Path(path).expanduser()
    lines = [line.rstrip("\n") for line in load_lines(p)]
    fixes = _index_fixes(text_fixes)
    checked: list[dict[str, Any]] = []
    problems: list[str] = []

    for fix in fixes:
        number, kind = fix["line_number"], fix["fix"]
        if number < 1 or number > len(lines):
            problems.append(f"text fix for line {number} is out of range")
            continue
        source = lines[number - 1]
        if kind == SPACE_BEFORE_PUNCTUATION:
            expected = repair_space_before_punctuation(source)
            if expected.replace(" ", "") != source.replace(" ", ""):
                problems.append(
                    f"text fix for line {number}: removing spaces changed words, not spacing"
                )
            checked.append(
                {
                    "line_number": number,
                    "fix": kind,
                    "spaces_removed": count_space_before_punct(source)
                    - count_space_before_punct(expected),
                }
            )
        elif kind == PAGE_NUMBER_IN_WORD:
            hits = has_midword_number(source)
            expected = repair_page_number_in_word(source)
            # Re-run the three repair rules by hand: only the digits the detector
            # found, the whitespace they carried, and nothing else, may move. The
            # comparison ignores whitespace because that is the only thing the
            # rules are allowed to shift.
            kept = source
            lead = next((h for h in hits if h["position"] in {"leading", "leading_spaced"}), None)
            if lead is not None:
                kept = kept[lead["offset"] + len(lead["digits"]) :].lstrip()
            for hit in sorted(hits, key=lambda h: -h["offset"]):
                if hit["position"] != "midword":
                    continue
                start = hit["offset"]
                end = start + len(hit["digits"])
                while end < len(kept) and kept[end] in " \t":
                    end += 1
                kept = kept[:start] + kept[end:]
            if not hits:
                problems.append(
                    f"text fix for line {number}: no page number is printed inside a word"
                )
            if kept.replace(" ", "") != expected.replace(" ", ""):
                problems.append(
                    f"text fix for line {number}: the repair changed more than the page numbers"
                )
            checked.append(
                {
                    "line_number": number,
                    "fix": kind,
                    "digits_removed": [h["digits"] for h in hits],
                }
            )
        elif kind == INLINE_PAGE_NUMBER:
            candidate = inline_page_number_candidate(source)
            if candidate is None:
                problems.append(
                    f"text fix for line {number}: no page number is merged into a text line"
                )
                continue
            expected = repair_inline_page_number(source)
            kept = source[: candidate["start"]] + source[candidate["end"] :]
            if re.sub(r"\s+", "", kept) != re.sub(r"\s+", "", expected):
                problems.append(
                    f"text fix for line {number}: the repair changed more than the "
                    "inline page number"
                )
            checked.append(
                {
                    "line_number": number,
                    "fix": kind,
                    "page_number": candidate["page_number"],
                    "token": candidate["token"],
                    "position": candidate["position"],
                }
            )
        elif kind == SPLIT_WORD:
            if number + 1 > len(lines):
                problems.append(f"text fix for line {number}: there is no next line to join")
                continue
            right = lines[number]
            if not is_word_continuation(source, right):
                problems.append(
                    f"text fix for line {number}: the two lines do not form one word"
                )
                continue
            stitched, _ = repair_split_word(source, right)
            checked.append(
                {
                    "line_number": number,
                    "fix": kind,
                    "joined_with": number + 1,
                    "word": stitched.rstrip().split()[-1] if stitched.strip() else "",
                }
            )
        else:
            problems.append(f"text fix for line {number}: unknown fix {kind!r}")

    return {
        "ok": not problems,
        "problems": problems,
        "count": len(checked),
        "checked": checked,
    }


__all__ = [
    "ApplyResult",
    "LineEdit",
    "apply_cleaning_plan",
    "looks_like_caption",
    "looks_like_page_artefact",
    "next_content_line",
    "strip_leading_page_number",
    "text_of",
    "validate_cleaning_plan",
    "validate_text_fixes",
    "verify_cleaning_plan",
    "verify_text_fixes",
]
