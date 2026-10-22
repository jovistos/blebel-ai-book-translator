"""Dry-run scoring of a proposed chapter map, before anything is written.

Every check here is deterministic and content-free: line numbers in range,
strictly increasing, no duplicates, safe/unique filenames, the line actually
looks like a title — or is declared, with evidence, as a ``role: "prose-chapter"``
opening whose heading the conversion dropped — the line is not inside the printed
TOC block, and the resulting gap structure is plausible.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..analysis.backmatter import back_matter_line_limit, is_back_matter_role
from ..analysis.candidates import nearest_title_markers
from ..analysis.prose import (
    entry_evidence,
    is_prose_chapter_role,
    prose_evidence_problem,
    prose_start_problem,
)
from ..analysis.toc import detect_toc_block, line_in_toc
from ..core.constants import DEFAULT_MIN_CHAPTERS, MAX_GAP_LINES
from ..documents.markdown import header_of, is_standalone_line, looks_like_title
from ..documents.text import excerpt, normalize_filename, similarity
from .plan import gap_warnings


def validate_split_map(
    lines: list[str],
    split_mapping: Iterable[dict[str, Any]],
    toc_block: dict[str, Any] | None = None,
    min_chapters: int = DEFAULT_MIN_CHAPTERS,
    min_gap_lines: int = 2,
    max_gap_lines: int = MAX_GAP_LINES,
) -> dict[str, Any]:
    """Score a proposed chapter map *before* anything is written to disk.

    Returns ``{"ok", "issues", "warnings", "points", "chapter_count", ...}``
    where each point echoes the short text found at that line.
    """
    points: list[dict[str, Any]] = []
    issues: list[str] = []
    warnings: list[str] = []
    mapping = list(split_mapping or [])

    if not mapping:
        return {
            "ok": False,
            "issues": ["split_mapping is empty"],
            "warnings": [],
            "points": [],
            "chapter_count": 0,
            "split_count": 0,
            "back_matter_count": 0,
            "prose_chapter_count": 0,
        }

    seen_lines: dict[int, int] = {}
    seen_files: dict[str, int] = {}
    numeric: list[tuple[int, dict[str, Any]]] = []

    for raw in mapping:
        if not isinstance(raw, dict):
            issues.append(f"entry is not an object: {raw!r}")
            continue
        ln = raw.get("line_number", raw.get("line"))
        if ln is None:
            issues.append(f"line_number is not an integer: {raw.get('line_number')!r}")
            continue
        try:
            ln = int(ln)
        except (TypeError, ValueError):
            issues.append(f"line_number is not an integer: {raw.get('line_number')!r}")
            continue
        name = str(raw.get("filename") or raw.get("file") or "").strip()
        if not name:
            issues.append(f"line {ln}: missing filename")
            continue
        numeric.append(
            (
                ln,
                {
                    "line_number": ln,
                    "filename": name,
                    "title": str(raw.get("title") or raw.get("original_header") or ""),
                    "role": str(raw.get("role") or raw.get("kind") or "").strip().lower(),
                    "evidence": entry_evidence(raw),
                },
            )
        )

    if not numeric:
        return {
            "ok": False,
            "issues": issues or ["no usable entries"],
            "warnings": warnings,
            "points": [],
            "chapter_count": 0,
            "split_count": 0,
            "back_matter_count": 0,
        }

    submitted_order = [ln for ln, _ in numeric]
    numeric.sort(key=lambda pair: pair[0])
    if submitted_order != sorted(submitted_order):
        warnings.append("entries were not submitted in ascending line order; they were sorted")

    for ln, item in numeric:
        if ln in seen_lines:
            issues.append(f"line {ln} is used twice")
            continue
        seen_lines[ln] = 1
        name = normalize_filename(item["filename"])
        if name in seen_files:
            issues.append(f"duplicate filename after normalisation: {name}")
        seen_files[name] = 1

        if ln < 1 or ln > len(lines):
            issues.append(f"line {ln} is outside the file (1..{len(lines)})")
            continue

        raw_line = lines[ln - 1]
        text = raw_line.strip()
        head = header_of(raw_line)
        standalone = is_standalone_line(lines, ln - 1)
        title_like = looks_like_title(raw_line)

        point = {
            "line_number": ln,
            "filename": name,
            "title": item["title"] or (head[1] if head else excerpt(text)),
            "matched_text": excerpt(text),
            "is_markdown_header": bool(head),
            "is_standalone": standalone,
            "similarity_to_title": round(similarity(item["title"], text), 3)
            if item["title"]
            else None,
        }

        is_back_matter = is_back_matter_role(item["role"])
        is_prose = is_prose_chapter_role(item["role"])
        if is_back_matter:
            point["role"] = "back-matter"
        elif is_prose:
            point["role"] = "prose-chapter"

        if toc_block is not None and line_in_toc(toc_block, ln):
            issues.append(
                f"line {ln} is inside the printed table of contents "
                f"({toc_block.get('start_line')}-{toc_block.get('end_line')}), not a chapter start"
            )
            point["inside_toc_block"] = True

        if is_back_matter:
            # A back-matter split is the one entry that need not look like a
            # chapter title (an author biography is a paragraph, a catalogue
            # opens with a book list). It is guarded by shape instead: the line
            # must exist in the trailing block, which must be of plausible size.
            if not text:
                issues.append(f"line {ln} is blank; a back-matter split needs a line of text")
            tail = len(lines) - ln + 1
            limit = back_matter_line_limit(len(lines))
            if tail > limit:
                issues.append(
                    f"line {ln} leaves {tail} trailing lines; a back-matter block may be at "
                    f"most {limit} lines here (role=back-matter would cut the story)"
                )
            if not (standalone or title_like):
                warnings.append(
                    f"line {ln} is marked role=back-matter and does not look like a title "
                    f"(text: {excerpt(text, 60)!r}); accepted because back matter need not"
                )
        elif is_prose:
            # A heading-lost chapter start is the second entry that need not look
            # like a title. It is guarded by shape (it must open a paragraph) and
            # by a stated justification: which contents list says this is a chapter.
            problem = prose_start_problem(lines, ln - 1)
            if problem:
                suggestions = nearest_title_markers(lines, ln)
                issues.append(
                    f"line {ln} is marked role=prose-chapter but {problem} "
                    f"(text: {excerpt(text, 60)!r}); nearest markers: {suggestions}"
                )
                point["nearest_markers"] = suggestions
            evidence_problem = prose_evidence_problem(item["evidence"])
            if evidence_problem:
                issues.append(f"line {ln} is marked role=prose-chapter and {evidence_problem}")
            else:
                point["evidence"] = excerpt(item["evidence"], 160)
        elif not head and not (standalone and title_like):
            suggestions = nearest_title_markers(lines, ln)
            issues.append(
                f"line {ln} does not look like a title "
                f"(text: {excerpt(text, 60)!r}); nearest markers: {suggestions}"
            )
            point["nearest_markers"] = suggestions

        points.append(point)

    ordered = sorted(points, key=lambda p: p["line_number"])
    warnings.extend(
        gap_warnings(
            ordered,
            max_gap_lines=max_gap_lines,
            min_gap_lines=min_gap_lines,
        )
    )

    back_matter_points = [i for i, p in enumerate(ordered) if p.get("role") == "back-matter"]
    if back_matter_points and back_matter_points[-1] != len(ordered) - 1:
        issues.append(
            "a role=back-matter split must be the LAST entry of the map: back matter is "
            "whatever follows the final chapter"
        )

    chapter_count = sum(1 for p in ordered if p.get("role") != "back-matter")
    if chapter_count < min_chapters:
        issues.append(
            f"only {chapter_count} chapter split point(s); expected at least {min_chapters}"
        )

    prose_points = [p for p in ordered if p.get("role") == "prose-chapter"]
    anchored_points = [p for p in ordered if p.get("role") not in {"back-matter", "prose-chapter"}]
    if prose_points:
        warnings.append(
            f"{len(prose_points)} of {chapter_count} chapter start(s) are role=prose-chapter: "
            "their headings are missing from the file, so every one of them rests on the "
            "evidence recorded in this map, not on a line of markup"
        )
        if len(prose_points) > len(anchored_points):
            warnings.append(
                "more chapter starts are prose than are anchored to a heading or a title line; "
                "double-check the map against the published contents list before slicing"
            )

    if ordered and ordered[0]["line_number"] > 1:
        warnings.append(
            f"content before line {ordered[0]['line_number']} is front matter and is "
            "written to the front-matter file"
        )

    return {
        "ok": not issues,
        "issues": issues,
        "warnings": warnings,
        "points": ordered,
        "chapter_count": chapter_count,
        "split_count": len(ordered),
        "back_matter_count": len(back_matter_points),
        "prose_chapter_count": len(prose_points),
        "first_split": ordered[0]["line_number"] if ordered else None,
        "last_split": ordered[-1]["line_number"] if ordered else None,
    }


def validate_source(
    source_path: str,
    split_mapping: Iterable[dict[str, Any]],
    min_chapters: int = DEFAULT_MIN_CHAPTERS,
) -> dict[str, Any]:
    """Convenience wrapper: read the file, detect its TOC, then validate."""
    from ..documents.io import load_lines

    lines = load_lines(source_path)
    result = validate_split_map(
        lines, split_mapping, toc_block=detect_toc_block(lines), min_chapters=min_chapters
    )
    result["line_count"] = len(lines)
    return result
