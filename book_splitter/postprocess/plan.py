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

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.constants import (
    DEFAULT_CHAPTER_TITLE_LEVEL,
    HEADING_DECISIONS,
    MAX_POSTPROCESS_TITLE_CHARS,
    MAX_TITLE_SCAN_LINES,
    NAMING_MODES,
    TITLE_ACTIONS,
    TOC_ACTIONS,
    TOC_STYLES,
)
from ..documents.io import load_lines
from ..documents.markdown import header_body, header_of
from ..documents.text import excerpt, norm
from .inspect import (
    ChapterStructure,
    is_bare_chapter_label,
    scan_book_structure,
    scan_chapter_structure,
    strip_chapter_label,
)

_MULTISPACE_RE = re.compile(r"\s+")


def clean_title(text: str, limit: int = MAX_POSTPROCESS_TITLE_CHARS) -> str:
    """One-line title text: no markup, no newlines, collapsed spaces, capped."""
    value = (text or "").strip()
    value = re.sub(r"^#{1,6}\s*", "", value)
    value = _MULTISPACE_RE.sub(" ", value).strip()
    return excerpt(value, limit)


def render_heading(text: str, level: int = DEFAULT_CHAPTER_TITLE_LEVEL) -> str:
    """A Markdown ATX heading line for ``text``."""
    safe_level = max(1, min(int(level or 1), 6))
    return f"{'#' * safe_level} {clean_title(text)}".rstrip()


def _title_level(structure: ChapterStructure, plan: dict[str, Any]) -> int:
    level = (plan.get("naming") or {}).get("level")
    if level is None:
        level = structure.title_level or DEFAULT_CHAPTER_TITLE_LEVEL
    try:
        return max(1, min(int(level), 6))
    except (TypeError, ValueError):
        return DEFAULT_CHAPTER_TITLE_LEVEL


def _number(structure: ChapterStructure, index: int) -> int:
    if structure.chapter_number is not None:
        return int(structure.chapter_number)
    if structure.body_index is not None:
        return int(structure.body_index)
    return int(index) + 1


def _title_part(structure: ChapterStructure) -> str:
    """The chapter's own title words, with the label/number prefix removed."""
    title = clean_title(structure.title)
    if title:
        rest = strip_chapter_label(title)
        if rest:
            return rest
    return clean_title(structure.subtitle)


def _normalise_overrides(value: Any) -> dict[str, str]:
    """Accept ``{file: title}`` or ``[{"filename", "title"}]``; return a dict."""
    out: dict[str, str] = {}
    if isinstance(value, dict):
        for filename, title in value.items():
            name = Path(str(filename or "")).name
            if name:
                out[name] = clean_title(str(title or ""))
    elif isinstance(value, list):
        for entry in value:
            if not isinstance(entry, dict):
                continue
            name = Path(str(entry.get("filename") or entry.get("chapter") or "")).name
            title = clean_title(str(entry.get("title") or ""))
            if name:
                out[name] = title
    return out


def _title_part_of(summary: dict[str, Any]) -> str:
    """The chapter's own title words, with the label/number prefix removed."""
    title = clean_title(str(summary.get("title") or ""))
    if title:
        rest = strip_chapter_label(title)
        if rest:
            return rest
    return clean_title(str(summary.get("subtitle") or ""))


def _number_of(summary: dict[str, Any], fallback: int) -> int:
    for key in ("chapter_number", "body_index"):
        value = summary.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                break
    return int(fallback) + 1


def canonical_titles(
    book: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, str]:
    """The final title of every chapter, derived from the book plan.

    The model does not have to echo the whole chapter list: it picks a policy and
    the titles are computed from the inventory. Per-chapter overrides are the
    escape hatch for a name that genuinely cannot follow the policy.
    """
    naming = plan.get("naming") or {}
    mode = str(naming.get("mode") or "keep").strip().lower()
    label = clean_title(str(naming.get("label") or ""))
    include_title = bool(naming.get("include_title"))
    separator = str(naming.get("separator") or ": ")
    if "\n" in separator:
        separator = ": "
    overrides = _normalise_overrides(naming.get("overrides"))

    out: dict[str, str] = {}
    for index, summary in enumerate(book.get("chapters") or []):
        filename = str(summary.get("filename") or "")
        if not filename:
            continue
        if filename in overrides and overrides[filename]:
            out[filename] = overrides[filename]
            continue
        if summary.get("matter") != "body":
            out[filename] = clean_title(str(summary.get("title") or "")) or clean_title(
                Path(filename).stem
            )
            continue
        current = _title_part_of(summary)
        number = _number_of(summary, index)
        if mode == "normalize":
            prefix = f"{label} {number}".strip()
            if include_title and current:
                out[filename] = clean_title(f"{prefix}{separator}{current}")
            else:
                out[filename] = clean_title(prefix) or current
        else:
            base = clean_title(str(summary.get("title") or "")) or clean_title(
                f"{label or 'Poglavlje'} {number}"
            )
            # ``keep`` keeps the title line, but a bare label such as
            # ``Poglavlje 1`` owns the chapter name on the next line. Carry that
            # name into the heading (and so into the TOC) instead of dropping it.
            if current and is_bare_chapter_label(base):
                base = clean_title(f"{base}{separator}{current}")
            out[filename] = base
    return out


def _default_toc_plan(book: dict[str, Any]) -> dict[str, Any]:
    toc = book.get("toc") or {}
    return {
        "action": "update" if toc.get("exists") else "create",
        "heading": "Sadržaj",
        "style": "numbered",
        "include_back_matter": False,
    }


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
    structures = {str(item.get("filename")): item for item in book.get("chapters") or []}
    naming = plan.get("naming")
    if not isinstance(naming, dict):
        naming = {}
    mode = str(naming.get("mode") or "keep").strip().lower()
    label = clean_title(str(naming.get("label") or ""))
    include_title = bool(naming.get("include_title"))
    separator = str(naming.get("separator") or ": ")
    level = naming.get("level", DEFAULT_CHAPTER_TITLE_LEVEL)
    overrides = _normalise_overrides(naming.get("overrides"))
    issues: list[str] = []
    warnings: list[str] = []
    evidence = excerpt(str(plan.get("title_evidence") or "").strip(), 500)
    title_source = str(plan.get("title_source") or "").strip().lower()
    if title_source and title_source not in {"web", "local", "mixed"}:
        warnings.append(
            "title_source should be 'web', 'local' or 'mixed'; it was ignored"
        )
        title_source = ""
    if require_web_search and not evidence:
        issues.append(
            "title_evidence is required: record the URL(s) you read from the published "
            "chapter list, or the failure/no-result outcome of the mandatory web search"
        )

    # A book whose chapters carry names must not lose them just because the
    # plan asked for bare labels: the names are already in the chapter opening.
    # Enable the optional "own title words" part deterministically and say so.
    named_chapters = [
        str(summary.get("filename"))
        for summary in book.get("chapters") or []
        if summary.get("matter") == "body" and _title_part_of(summary)
    ]
    if mode == "normalize" and named_chapters and not include_title:
        include_title = True
        warnings.append(
            f"naming.include_title was false, but {len(named_chapters)} body chapter(s) "
            "carry their own title words; it was enabled so the chapter headings and "
            "the table of contents keep those names"
        )

    if mode not in NAMING_MODES:
        issues.append(f"naming.mode must be one of {list(NAMING_MODES)}, got {mode!r}")
    if mode == "normalize" and not label:
        issues.append("naming.label is required when naming.mode is 'normalize'")
    if len(label) > 40:
        issues.append("naming.label is longer than 40 characters")
    if "\n" in separator or len(separator) > 5:
        issues.append("naming.separator must be at most 5 characters and one line")
    try:
        level = int(level)
    except (TypeError, ValueError):
        issues.append("naming.level must be an integer between 1 and 3")
        level = DEFAULT_CHAPTER_TITLE_LEVEL
    if not 1 <= level <= 3:
        issues.append("naming.level must be between 1 and 3")
    if not book.get("body_chapter_count"):
        issues.append("the folder contains no body chapters to postprocess")

    for filename, title in overrides.items():
        if filename not in structures:
            issues.append(f"naming override targets an unknown chapter: {filename!r}")
            continue
        if not title:
            issues.append(f"naming override for {filename!r} is empty")
        elif len(title) > MAX_POSTPROCESS_TITLE_CHARS:
            issues.append(f"naming override for {filename!r} is too long")

    toc_plan = plan.get("toc")
    if not isinstance(toc_plan, dict):
        toc_plan = _default_toc_plan(book)
        warnings.append("no toc plan was submitted; the deterministic default was used")
    action = str(toc_plan.get("action") or "").strip().lower()
    heading = clean_title(str(toc_plan.get("heading") or "Sadržaj"))
    style = str(toc_plan.get("style") or "numbered").strip().lower()
    include_back = bool(toc_plan.get("include_back_matter"))
    if action not in TOC_ACTIONS:
        issues.append(f"toc.action must be one of {list(TOC_ACTIONS)}, got {action!r}")
    if not heading:
        issues.append("toc.heading cannot be empty")
    if len(heading) > 80:
        issues.append("toc.heading is longer than 80 characters")
    if style not in TOC_STYLES:
        issues.append(f"toc.style must be one of {list(TOC_STYLES)}, got {style!r}")

    normalized = {
        "naming": {
            "mode": mode,
            "label": label,
            "include_title": include_title,
            "separator": separator,
            "level": level,
            "overrides": overrides,
        },
        "toc": {
            "action": action,
            "heading": heading,
            "style": style,
            "include_back_matter": include_back,
        },
        "title_evidence": evidence,
        "title_source": title_source,
    }
    titles = canonical_titles(book, normalized)
    if not issues:
        seen: dict[str, str] = {}
        for filename, title in titles.items():
            if not title:
                issues.append(f"chapter {filename!r} would have an empty title")
                continue
            if len(title) > MAX_POSTPROCESS_TITLE_CHARS:
                issues.append(f"chapter {filename!r} title is too long")
            key = norm(title)
            if key and key in seen and structures.get(filename, {}).get("matter") == "body":
                warnings.append(
                    f"chapters {seen[key]!r} and {filename!r} would share the title {title!r}"
                )
            seen.setdefault(key, filename)

    from .toc import build_toc_entries, toc_matches_entries

    entries = build_toc_entries(
        book, normalized, titles=titles, include_back_matter=include_back
    )
    toc = book.get("toc") or {}
    if action == "create" and toc.get("exists"):
        issues.append(
            "a table of contents already exists; use toc.action='update' (or 'keep' if it "
            "already matches the chapter titles)"
        )
    if action == "update" and not toc.get("exists"):
        issues.append("no table of contents exists; use toc.action='create'")
    if action == "keep":
        if not toc.get("exists"):
            issues.append("no table of contents exists; use toc.action='create'")
        elif not toc_matches_entries(toc, entries):
            issues.append(
                "the existing table of contents does not match the chapter titles; use "
                "toc.action='update'"
            )

    return {
        "ok": not issues,
        "issues": issues,
        "warnings": warnings,
        "naming": normalized["naming"],
        "toc": normalized["toc"],
        "title_evidence": evidence,
        "title_source": title_source,
        "canonical_titles": titles,
        "toc_entries": entries,
        "chapter_count": len(titles),
        "body_chapter_count": int(book.get("body_chapter_count") or 0),
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
    title = normalized["title"]
    headings = normalized["headings"]
    issues: list[str] = []
    warnings: list[str] = []
    lines = load_lines(path)
    title_line = structure.title_line

    action = title.get("action") or "keep"
    text = title.get("text") or clean_title(canonical_title)
    level = title.get("level") or structure.title_level or DEFAULT_CHAPTER_TITLE_LEVEL
    if action not in TITLE_ACTIONS:
        issues.append(f"title.action must be one of {list(TITLE_ACTIONS)}, got {action!r}")
    try:
        level = int(level)
    except (TypeError, ValueError):
        issues.append("title.level must be an integer")
        level = DEFAULT_CHAPTER_TITLE_LEVEL
    if not 1 <= level <= 6:
        issues.append("title.level must be between 1 and 6")
    if canonical_title:
        expected = clean_title(canonical_title)
        if text and norm(text) != norm(expected):
            issues.append(
                f"this chapter's title is fixed by the book plan as {expected!r}; "
                f"the plan says {text!r}"
            )
        text = expected or text
    if action in {"set", "insert"} and not text:
        issues.append("title.text cannot be empty")
    if len(text) > MAX_POSTPROCESS_TITLE_CHARS:
        issues.append(f"title.text is longer than {MAX_POSTPROCESS_TITLE_CHARS} characters")
    if "\n" in text:
        issues.append("title.text must be one line")

    if action == "set":
        line_number = title.get("line_number") or title_line
        if line_number is None:
            issues.append("title.action='set' needs a line_number; this chapter has no title")
        elif line_number < 1 or line_number > len(lines):
            issues.append(f"title line {line_number} is out of range")
        elif line_number > MAX_TITLE_SCAN_LINES:
            issues.append(
                f"title line {line_number} is outside the first {MAX_TITLE_SCAN_LINES} lines"
            )
        elif title_line is not None and line_number != title_line:
            issues.append(
                f"line {line_number} is not the chapter title; the title candidate is "
                f"line {title_line}"
            )
        elif title_line is None:
            issues.append(
                "this chapter has no title candidate; use action='insert' instead of "
                "promoting an arbitrary line"
            )
    elif action == "insert":
        if title_line is not None:
            issues.append(
                f"this chapter already has a title at line {title_line}; use action='set' "
                "instead of inserting a second heading"
            )
    elif action == "keep":
        if title_line is None:
            issues.append("this chapter has no title; use action='insert'")
        elif not structure.title_is_heading:
            issues.append(
                "the existing title is not a Markdown heading; use action='set' to add the "
                "'#' markup"
            )
        elif canonical_title and norm(clean_title(structure.title)) != norm(clean_title(canonical_title)):
            issues.append(
                "the existing title does not match the book plan; use action='set' with the "
                "canonical title"
            )

    header_lines = {int(item["line_number"]) for item in structure.headings}
    claimed: set[int] = set()
    for index, entry in enumerate(headings):
        line = entry["line_number"]
        decision = entry["decision"]
        label = f"heading #{index + 1} (line {line})"
        if decision not in HEADING_DECISIONS:
            issues.append(f"{label}: decision must be one of {list(HEADING_DECISIONS)}")
            continue
        if line < 1 or line > len(lines):
            issues.append(f"{label}: out of range (chapter has {len(lines)} lines)")
            continue
        if line not in header_lines:
            issues.append(f"{label}: not a Markdown header")
            continue
        if line == title_line:
            issues.append(f"{label}: the chapter title is controlled by title.action, not headings")
            continue
        if line in claimed:
            issues.append(f"{label}: another heading decision already targets this line")
            continue
        claimed.add(line)
        suspicious = next(
            (item for item in structure.suspicious_headings if item["line_number"] == line),
            None,
        )
        if decision == "keep" and suspicious is not None:
            warnings.append(
                f"{label}: the inspector suggested {suspicious['suggested']} but the plan keeps it"
            )

    omitted = [
        item
        for item in structure.suspicious_headings
        if int(item["line_number"]) not in claimed
    ]
    if omitted:
        warnings.append(
            f"{len(omitted)} suspicious heading(s) were not reviewed and will be kept as-is: "
            + ", ".join(str(item["line_number"]) for item in omitted[:8])
        )

    return {
        "ok": not issues,
        "issues": issues,
        "warnings": warnings,
        "plan": normalized,
        "canonical_title": clean_title(canonical_title) or text,
        "chapter": {
            "filename": structure.filename,
            "line_count": len(lines),
            "title_line": title_line,
            "heading_count": structure.heading_count,
        },
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
    the text and only removes the ``#`` markup. A missing title is inserted at
    the top, followed by a blank line when the first source line is not blank.
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
    merged_subtitles = set(result.merged_subtitles)
    collapsed_blanks = set(result.collapsed_blanks)
    accounted: set[int] = set()
    for entry in result.provenance:
        accounted.update(entry.get("source_lines") or [])
    for number in range(1, len(lines) + 1):
        if number not in accounted:
            problems.append(f"line {number} vanished without a provenance entry")
    for number in merged_subtitles:
        subtitle = clean_title(lines[number - 1])
        if subtitle and norm(subtitle) not in norm(result.title):
            problems.append(
                f"line {number} was merged into the title but its text is not in the title"
            )
    for number in collapsed_blanks:
        if lines[number - 1].strip():
            problems.append(f"line {number} was collapsed as a separator but is not blank")
    title_line = result.title_line
    for number, raw in enumerate(lines, start=1):
        if number in merged_subtitles or number in collapsed_blanks:
            continue
        if number in demoted:
            expected = header_body(raw) if header_of(raw) else raw.strip()
            if expected not in result.lines:
                problems.append(f"line {number} was demoted but its text is not in the output")
            continue
        if number == title_line:
            continue
        if raw not in result.lines:
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
