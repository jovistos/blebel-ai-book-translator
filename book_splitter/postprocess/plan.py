"""The stage-9 plans: validation and deterministic application.

As in stages 1 and 2, the agent never edits text. It submits a plan in a fixed
vocabulary and this module checks it against the real chapter:

* the **book plan** chooses one naming policy for the whole book (plus optional
  per-chapter overrides) and a table-of-contents action;
* the **chapter plan** sets/inserts/keeps the chapter title and says ``keep`` or
  ``demote`` for headings.

``demote`` is the only destructive-looking operation, and it deletes no words: it
removes the ``#`` markup and leaves the line as a paragraph. There is deliberately
no "remove this heading" operation that would also delete its text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.constants import DEFAULT_CHAPTER_TITLE_LEVEL, TITLE_ACTIONS
from ..documents.io import load_lines
from ..documents.markdown import header_body, header_of, is_plain_heading_candidate
from ..documents.text import excerpt, norm
from .inspect import ChapterStructure, scan_book_structure, scan_chapter_structure
from .rules import (
    BookPlanDraft,
    ChapterPlanDraft,
    check_titles,
    check_toc_state,
    run_book_rules,
    run_chapter_rules,
)
from .titles import canonical_titles, clean_title, render_heading
from .toc import build_toc_entries


def validate_book_plan(
    chapters_path: str | Path,
    plan: dict[str, Any],
    *,
    book: dict[str, Any] | None = None,
    require_web_search: bool = False,
) -> dict[str, Any]:
    """Check the book-wide naming/TOC plan against the real chapter inventory.

    ``require_web_search`` is set by the graph: the published chapter list lookup
    is mandatory there, so a plan without ``title_evidence`` is rejected.
    """
    if not isinstance(plan, dict):
        return {
            "ok": False,
            "issues": ["the plan must be an object with `naming` and `toc`"],
            "warnings": [],
            "canonical_titles": {},
        }
    book = book or scan_book_structure(chapters_path)
    structures = {
        str(item.get("filename")): item for item in book.get("chapters") or []
    }
    draft = BookPlanDraft.from_plan(plan, book=book, structures=structures)
    run_book_rules(draft, require_web_search=require_web_search)
    normalized = draft.normalized()
    titles = canonical_titles(book, normalized)
    check_titles(draft, titles)
    entries = build_toc_entries(
        book,
        normalized,
        titles=titles,
        include_back_matter=draft.toc_include_back_matter,
    )
    check_toc_state(draft, entries)
    return {
        "ok": not draft.issues,
        "issues": draft.issues,
        "warnings": draft.warnings,
        "naming": normalized["naming"],
        "toc": normalized["toc"],
        "title_evidence": draft.evidence,
        "title_source": draft.title_source,
        "canonical_titles": titles,
        "toc_entries": entries,
        "chapter_count": len(titles),
        "body_chapter_count": int(book.get("body_chapter_count") or 0),
    }


def validate_chapter_plan(
    chapter_path: str | Path,
    plan: dict[str, Any],
    *,
    canonical_title: str = "",
    structure: ChapterStructure | None = None,
) -> dict[str, Any]:
    """Check one chapter's title/heading plan against the real file."""
    path = Path(chapter_path).expanduser()
    structure = structure or scan_chapter_structure(path)
    normalized = normalise_chapter_plan(plan)
    lines = load_lines(path)
    draft = ChapterPlanDraft.from_plan(
        normalized,
        structure=structure,
        lines=lines,
        canonical_title=canonical_title,
    )
    run_chapter_rules(draft)
    return {
        "ok": not draft.issues,
        "issues": draft.issues,
        "warnings": draft.warnings,
        "plan": normalized,
        "canonical_title": clean_title(canonical_title) or draft.title_text,
        "chapter": {
            "filename": structure.filename,
            "line_count": len(lines),
            "title_line": structure.title_line,
            "heading_count": structure.heading_count,
        },
    }


def _normalise_title_plan(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {"action": "set", "text": clean_title(value), "level": None, "line_number": None}
    if not isinstance(value, dict):
        return {}
    action = str(value.get("action") or "").strip().lower()
    try:
        line = int(value["line_number"]) if value.get("line_number") is not None else None
    except (TypeError, ValueError):
        line = None
    try:
        level = int(value["level"]) if value.get("level") is not None else None
    except (TypeError, ValueError):
        level = None
    return {
        "action": action,
        "text": clean_title(str(value.get("text") or "")),
        "line_number": line,
        "level": level,
    }


def normalise_chapter_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Normalise a chapter plan into the shape the validator/applier consume."""
    title = _normalise_title_plan((plan or {}).get("title"))
    headings: list[dict[str, Any]] = []
    for entry in (plan or {}).get("headings") or []:
        if not isinstance(entry, dict):
            continue
        try:
            line = int(entry["line_number"])
        except (KeyError, TypeError, ValueError):
            continue
        headings.append(
            {
                "line_number": line,
                "decision": str(entry.get("decision") or "").strip().lower(),
                "note": excerpt(str(entry.get("note") or ""), 120),
            }
        )
    headings.sort(key=lambda item: item["line_number"])
    return {
        "title": title,
        "headings": headings,
        "notes": excerpt(str((plan or {}).get("notes") or ""), 300),
    }


@dataclass
class ChapterApply:
    """The rewritten lines plus the provenance needed to verify them."""

    lines: list[str] = field(default_factory=list)
    provenance: list[dict[str, Any]] = field(default_factory=list)
    title_action: str = "keep"
    title_line: int | None = None
    title: str = ""
    inserted: bool = False
    demoted_lines: list[int] = field(default_factory=list)
    promoted_lines: list[int] = field(default_factory=list)
    merged_subtitles: list[int] = field(default_factory=list)
    collapsed_blanks: list[int] = field(default_factory=list)
    source_lines: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "source_lines": self.source_lines,
            "output_lines": len(self.lines),
            "title_action": self.title_action,
            "title_line": self.title_line,
            "title": self.title,
            "inserted": self.inserted,
            "demoted_lines": self.demoted_lines,
            "promoted_lines": self.promoted_lines,
            "merged_subtitles": self.merged_subtitles,
            "collapsed_blanks": self.collapsed_blanks,
        }


def apply_chapter_plan(
    chapter_path: str | Path,
    plan: dict[str, Any],
    *,
    canonical_title: str = "",
    structure: ChapterStructure | None = None,
) -> ChapterApply:
    """Apply a validated chapter plan to lines — pure, no writes.

    Line edits are expressed as a rewrite, never as a delete: a ``demote`` keeps
    the text and only removes the ``#`` markup, and a ``promote`` keeps it and
    only adds the markup back to a plain ALL-CAPS heading. A missing title is
    inserted at the top, followed by a blank line when the first source line is
    not blank.
    """
    path = Path(chapter_path).expanduser()
    structure = structure or scan_chapter_structure(path)
    normalized = normalise_chapter_plan(plan)
    title = normalized["title"]
    action = title.get("action") or "keep"
    if action not in TITLE_ACTIONS:
        action = "keep"
    text = clean_title(canonical_title or title.get("text") or structure.title)
    level = int(title.get("level") or structure.title_level or DEFAULT_CHAPTER_TITLE_LEVEL)
    title_line = title.get("line_number") or structure.title_line
    demote = {
        entry["line_number"]
        for entry in normalized["headings"]
        if entry["decision"] == "demote"
    }
    promote = {
        entry["line_number"]
        for entry in normalized["headings"]
        if entry["decision"] == "promote"
    }
    # A promoted heading should look like the chapter's existing section headings.
    # With none to copy, fall back to one level below the title (usually `##`).
    section_levels = [
        int(item.get("level") or 0)
        for item in structure.headings
        if int(item.get("level") or 0) > 0
        and item.get("line_number") != structure.title_line
    ]
    if section_levels:
        promote_level = max(sorted(set(section_levels)), key=section_levels.count)
    else:
        title_is_heading = action in {"set", "insert"} or structure.title_is_heading
        promote_level = min(6, int(level) + 1) if title_is_heading else 2
    # When the book policy folds the subtitle into the chapter title
    # ("Poglavlje 1: Šta vam je na umu?"), the original subtitle line would be a
    # duplicate paragraph. Its words are preserved in the heading, so the line is
    # dropped and recorded as `subtitle_merged` (never silently).
    merge_subtitle: int | None = None
    raw_lines = [line.rstrip("\n") for line in load_lines(path)]
    if (
        action in {"set", "insert"}
        and structure.subtitle_line
        and structure.subtitle
        and structure.subtitle_line != title_line
        and norm(clean_title(structure.subtitle)) in norm(text)
    ):
        merge_subtitle = int(structure.subtitle_line)
    merge_blank: int | None = None
    if merge_subtitle is not None:
        # Keep one blank line between the merged heading and the first paragraph.
        if (
            merge_subtitle - 1 >= 1
            and merge_subtitle - 1 != title_line
            and not raw_lines[merge_subtitle - 2].strip()
        ):
            merge_blank = merge_subtitle - 1
        elif merge_subtitle < len(raw_lines) and not raw_lines[merge_subtitle].strip():
            merge_blank = merge_subtitle + 1
    result = ChapterApply(
        source_lines=len(raw_lines),
        title_action=action,
        title_line=title_line,
        title=text,
        inserted=action == "insert",
    )

    if action == "insert":
        result.lines.append(render_heading(text, level))
        result.provenance.append(
            {"output_line": 1, "origin": "title_inserted", "source_lines": []}
        )
        if raw_lines and raw_lines[0].strip():
            result.lines.append("")
            result.provenance.append(
                {"output_line": 2, "origin": "inserted_separator", "source_lines": []}
            )
        result.title_line = None

    for number, raw in enumerate(raw_lines, start=1):
        if number == merge_subtitle:
            result.merged_subtitles.append(number)
            result.provenance.append(
                {
                    "output_line": len(result.lines) + 1,
                    "origin": "subtitle_merged",
                    "source_lines": [number],
                }
            )
            continue
        if number == merge_blank:
            result.collapsed_blanks.append(number)
            result.provenance.append(
                {
                    "output_line": len(result.lines) + 1,
                    "origin": "subtitle_separator_collapsed",
                    "source_lines": [number],
                }
            )
            continue
        if number == title_line and action == "set":
            result.lines.append(render_heading(text, level))
            result.provenance.append(
                {
                    "output_line": len(result.lines),
                    "origin": "title_set",
                    "source_lines": [number],
                }
            )
            continue
        if number == title_line and action == "keep":
            result.lines.append(raw)
            result.provenance.append(
                {
                    "output_line": len(result.lines),
                    "origin": "title_kept",
                    "source_lines": [number],
                }
            )
            continue
        if number in demote:
            body = header_body(raw) if header_of(raw) else raw.strip()
            result.lines.append(body)
            result.demoted_lines.append(number)
            result.provenance.append(
                {
                    "output_line": len(result.lines),
                    "origin": "demoted_header",
                    "source_lines": [number],
                }
            )
            continue
        if (
            number in promote
            and header_of(raw) is None
            and is_plain_heading_candidate(raw.strip())
        ):
            result.lines.append(render_heading(raw.strip(), promote_level))
            result.promoted_lines.append(number)
            result.provenance.append(
                {
                    "output_line": len(result.lines),
                    "origin": "promoted_heading",
                    "source_lines": [number],
                }
            )
            continue
        result.lines.append(raw)
        result.provenance.append(
            {
                "output_line": len(result.lines),
                "origin": "kept",
                "source_lines": [number],
            }
        )
    return result


def text_of(result: ChapterApply) -> str:
    """The rewritten chapter as text (newline-terminated)."""
    return "".join(line + "\n" for line in result.lines)


def verify_chapter_plan(
    chapter_path: str | Path,
    plan: dict[str, Any],
    *,
    canonical_title: str = "",
    output_path: str | Path | None = None,
    structure: ChapterStructure | None = None,
) -> dict[str, Any]:
    """Prove that the written chapter differs only in the declared ways.

    The expected output is recomputed from the source and the same plan, then
    compared with the file on disk. If no output path is given, the invariants
    are still checked so a dry run and a real run share one verification path.
    """
    path = Path(chapter_path).expanduser()
    result = apply_chapter_plan(
        path, plan, canonical_title=canonical_title, structure=structure
    )
    problems: list[str] = []
    lines = [line.rstrip("\n") for line in load_lines(path)]
    demoted = set(result.demoted_lines)
    promoted = set(result.promoted_lines)
    merged_subtitles = set(result.merged_subtitles)
    collapsed_blanks = set(result.collapsed_blanks)
    accounted: set[int] = set()
    for entry in result.provenance:
        accounted.update(entry.get("source_lines") or [])
    problems.extend(
        f"line {number} vanished without a provenance entry"
        for number in range(1, len(lines) + 1)
        if number not in accounted
    )
    for number in merged_subtitles:
        subtitle = clean_title(lines[number - 1])
        if subtitle and norm(subtitle) not in norm(result.title):
            problems.append(
                f"line {number} was merged into the title but its text is not in the title"
            )
    problems.extend(
        f"line {number} was collapsed as a separator but is not blank"
        for number in collapsed_blanks
        if lines[number - 1].strip()
    )
    title_line = result.title_line
    output_lines = set(result.lines)
    output_headings = {
        clean_title(head[1])
        for line in result.lines
        if (head := header_of(line)) is not None
    }
    for number, raw in enumerate(lines, start=1):
        if number in merged_subtitles or number in collapsed_blanks:
            continue
        if number in demoted:
            expected = header_body(raw) if header_of(raw) else raw.strip()
            if expected not in output_lines:
                problems.append(f"line {number} was demoted but its text is not in the output")
            continue
        if number in promoted:
            promoted_text = clean_title(raw)
            if promoted_text not in output_headings:
                problems.append(
                    f"line {number} was promoted but its Markdown heading is not in the output"
                )
            continue
        if number == title_line:
            continue
        if raw not in output_lines:
            problems.append(
                f"line {number} was not declared as changed but is not in the output verbatim"
            )

    written_text: str | None = None
    if output_path is not None:
        target = Path(output_path).expanduser()
        if not target.is_file():
            problems.append(f"output file is missing: {target}")
        else:
            written_text = target.read_text(encoding="utf-8")
            expected_text = text_of(result)
            if written_text != expected_text:
                problems.append("the written chapter does not match the validated plan")

    return {
        "ok": not problems,
        "problems": problems,
        "source_lines": result.source_lines,
        "output_lines": len(result.lines),
        "title_action": result.title_action,
        "title": result.title,
        "title_line": result.title_line,
        "demoted_lines": result.demoted_lines,
        "provenance": result.summary(),
        "verified_output": output_path is not None and written_text is not None,
    }

__all__ = [
    "ChapterApply",
    "apply_chapter_plan",
    "canonical_titles",
    "clean_title",
    "normalise_chapter_plan",
    "render_heading",
    "text_of",
    "validate_book_plan",
    "validate_chapter_plan",
    "verify_chapter_plan",
]
