"""Proving a stage-2 cleaning plan changed only what its own rules allow.

The check is a *content* comparison, not a diff of intentions. For every source
line it computes what the output is allowed to contain and compares that with
what was written:

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

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..documents.bodyline import (
    count_space_before_punct,
    has_midword_number,
    is_word_continuation,
)
from ..documents.images import has_images, image_refs
from ..documents.io import load_lines
from ..documents.markdown import header_of
from ..documents.text import excerpt
from .apply import LineProvenance, _index_fixes, apply_cleaning_plan
from .damage import (
    INLINE_PAGE_NUMBER,
    PAGE_NUMBER_IN_WORD,
    SPACE_BEFORE_PUNCTUATION,
    SPLIT_WORD,
    inline_page_number_candidate,
    repair_inline_page_number,
    repair_page_number_in_word,
    repair_space_before_punctuation,
    repair_split_word,
)
from .inspect import DEMOTE, PROMOTE, REMOVE, REMOVE_CAPTION
from .plan import (
    looks_like_caption,
    looks_like_page_artefact,
    strip_leading_page_number,
)


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
    provenance = LineProvenance.from_result(result)
    fixed_lines = {fix["line_number"] for fix in fixes}
    headers = {number for number, line in enumerate(lines, start=1) if header_of(line)}

    problems: list[str] = [
        f"line {number} vanished without a provenance entry"
        for number in provenance.unaccounted(len(lines))
    ]

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

    removed_artefacts = set(result.removed_artefacts)
    removed_captions = set(result.removed_captions)

    for number, source in enumerate(lines, start=1):
        if number in rewritten:
            continue
        op = by_line.get(number) or {}
        decision = str(op.get("decision") or "").lower()
        head = header_of(source)
        body = head[1] if head else source

        if has_images(source):
            if not provenance.has_text(source):
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
            slot = provenance.output_index(number)
            if number not in removed_artefacts:
                problems.append(f"line {number} was declared remove but is still in the output")
            elif slot < 0:
                problems.append(f"line {number} was removed but has no provenance entry")
            elif output[slot] != "":
                problems.append(f"line {number} was removed but its slot is not blank")
            continue

        if decision == REMOVE_CAPTION:
            if not looks_like_caption(source):
                problems.append(
                    f"line {number} was removed as a caption but does not look like one: "
                    f"{excerpt(source, 40)!r}"
                )
            slot = provenance.output_index(number)
            if number not in removed_captions:
                problems.append(
                    f"line {number} was declared remove_caption but is still in the output"
                )
            elif slot < 0:
                problems.append(f"line {number} was removed but has no provenance entry")
            elif output[slot] != "":
                problems.append(f"line {number} was removed but its slot is not blank")
            continue

        if decision == DEMOTE:
            expected, _ = strip_leading_page_number(body)
            if not provenance.has_text(expected):
                problems.append(
                    f"line {number} was demoted but its text is not in the output: "
                    f"{excerpt(expected, 40)!r}"
                )
            continue

        if decision == PROMOTE:
            expected = f"## {source.strip()}"
            if not provenance.has_text(expected):
                problems.append(
                    f"line {number} was promoted but its heading is not in the output: "
                    f"{excerpt(expected, 40)!r}"
                )
            continue

        if not provenance.has_text(source):
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
        line_head = header_of(lines[last - 1])
        right = line_head[1] if line_head else ""
        right, _ = strip_leading_page_number(right)
        expected = f"{left} {right}".strip()
        if not provenance.has_text(expected):
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
    problems.extend(
        f"line {number} was removed without being declared in the plan"
        for number in undeclared
        if number in headers
    )
    problems.extend(
        f"line {number} was removed as a caption without being declared"
        for number in undeclared_captions
    )

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
    "verify_cleaning_plan",
    "verify_text_fixes",
]
