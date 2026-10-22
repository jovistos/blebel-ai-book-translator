"""Table-of-contents detection, generation and rewriting for stage 9.

The TOC is book structure, so this module reads only the front matter (and a TOC
file, when one already exists) plus the chapter-title inventory. It never reads a
chapter body.

Three rules keep the result honest:

* an existing TOC is *located* first (a heading, a dot-leader block, or a
  separate file), so an update replaces exactly that block and preserves the rest
  of the front matter;
* a missing TOC is created as ``00-table-of-contents.md``, which sorts right
  after ``00-front-matter.md`` and before ``01-…``;
* the entries are generated from the final chapter titles, and
  :func:`verify_toc` checks that every chapter appears once, in reading order.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..analysis.patterns import TOC_WORDS
from ..analysis.toc import detect_toc_block
from ..core.artifacts import write_text
from ..core.constants import (
    DEFAULT_TOC_HEADING,
    MAX_TOC_ENTRIES,
    TOC_FILENAME,
    TOC_SCAN_LINES,
)
from ..documents.io import load_lines
from ..documents.markdown import header_of
from ..documents.text import excerpt, norm, similarity
from .inspect import is_toc_filename, matter_kind

#: A list marker at the start of a TOC entry (``- ``, ``* ``, ``1. ``, ``1) ``).
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•]|\d{1,3}[.)])\s+")
#: A trailing printed page number, with or without dot leaders.
_TRAILING_PAGE_RE = re.compile(r"(?:\s*[.\-•_]{2,}\s*|\s{3,})(\d{1,4})\s*$")
_MD_LINK_RE = re.compile(r"^\[([^\]]+)\]\([^)]*\)\s*$")
_TOC_WORDS = {key for key in (norm(word) for word in TOC_WORDS) if key}


def _toc_heading_index(lines: list[str]) -> int | None:
    for index, line in enumerate(lines[:TOC_SCAN_LINES]):
        head = header_of(line)
        text = (head[1] if head else line).strip()
        key = norm(text)
        if key and key in _TOC_WORDS:
            return index
    return None


def _entry_title(raw: str) -> str:
    """Strip a list marker, a link wrapper and a page number from one line."""
    text = (raw or "").strip()
    if not text:
        return ""
    link = _MD_LINK_RE.match(text)
    if link:
        return excerpt(link.group(1), 120)
    text = _LIST_MARKER_RE.sub("", text)
    text = _TRAILING_PAGE_RE.sub("", text)
    return excerpt(text.strip(" .\t"), 120)


def _parse_entries(lines: list[str], start: int, end: int) -> list[dict[str, Any]]:
    """Parse TOC entry lines in ``[start, end)`` (0-based)."""
    entries: list[dict[str, Any]] = []
    for index in range(max(0, start), min(len(lines), end)):
        raw = lines[index]
        if not raw.strip():
            continue
        head = header_of(raw)
        if head is not None:
            title = _entry_title(head[1])
            page = None
        else:
            title = _entry_title(raw)
            page = None
            match = _TRAILING_PAGE_RE.search(raw.strip())
            if match:
                page = int(match.group(1))
        if not title or len(title) < 2:
            continue
        if norm(title) in _TOC_WORDS:
            continue
        entries.append({"line_number": index + 1, "title": title, "page": page})
        if len(entries) >= MAX_TOC_ENTRIES:
            break
    return entries


def _match_entries_to_chapters(
    entries: list[dict[str, Any]],
    structures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach the best chapter filename to each TOC entry (for consistency)."""
    if not structures:
        return entries
    out: list[dict[str, Any]] = []
    for entry in entries:
        best: tuple[float, dict[str, Any] | None] = (0.0, None)
        for structure in structures:
            score = similarity(entry["title"], str(structure.get("title") or ""))
            if score > best[0]:
                best = (score, structure)
        matched = dict(entry)
        if best[1] is not None and best[0] >= 0.62:
            matched["chapter_filename"] = best[1].get("filename")
            matched["matched_title"] = best[1].get("title")
            matched["score"] = round(best[0], 3)
        out.append(matched)
    return out


def detect_toc(
    chapters_path: str | Path,
    structures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Locate an existing TOC: a separate file, or a block in the front matter."""
    folder = Path(chapters_path).expanduser()
    if folder.is_file():
        folder = folder.parent
    if structures is None:
        structures = []

    toc_files = [
        path
        for path in folder.glob("*.md")
        if path.is_file() and is_toc_filename(path.name) and not path.name.startswith("_")
    ]
    if toc_files:
        path = sorted(toc_files, key=lambda item: item.name)[0]
        lines = load_lines(path)
        heading_index = _toc_heading_index(lines)
        start = (heading_index + 1) if heading_index is not None else 0
        entries = _parse_entries(lines, start, len(lines))
        heading = ""
        if heading_index is not None:
            head = header_of(lines[heading_index])
            heading = (head[1] if head else lines[heading_index]).strip()
        return {
            "exists": True,
            "location": "file",
            "filename": path.name,
            "path": str(path),
            "heading": heading or DEFAULT_TOC_HEADING,
            "heading_line": (heading_index + 1) if heading_index is not None else None,
            "start_line": start + 1,
            "end_line": len(lines),
            "method": "separate_file",
            "entries": _match_entries_to_chapters(entries, structures),
            "entry_count": len(entries),
        }

    front = next(
        (structure for structure in structures if structure.get("matter") == "front"),
        None,
    )
    if front is None:
        return {
            "exists": False,
            "location": None,
            "method": None,
            "entries": [],
            "entry_count": 0,
        }

    path = folder / str(front.get("filename") or "")
    if not path.is_file():
        return {
            "exists": False,
            "location": None,
            "method": None,
            "entries": [],
            "entry_count": 0,
        }
    lines = load_lines(path)
    block = detect_toc_block(lines, scan_lines=min(len(lines), TOC_SCAN_LINES))
    heading_index = _toc_heading_index(lines)
    if heading_index is None and block.get("heading_line"):
        heading_index = int(block["heading_line"]) - 1
    heading = ""
    if heading_index is not None and 0 <= heading_index < len(lines):
        head = header_of(lines[heading_index])
        heading = (head[1] if head else lines[heading_index]).strip()

    if block.get("found"):
        start = int(block.get("start_line") or 1) - 1
        end = int(block.get("end_line") or start + 1)
        entries = _parse_entries(lines, start, end)
        method = block.get("method") or "block"
    elif heading_index is not None:
        # A TOC heading with no dot leaders: parse the lines after it until two
        # blanks or a second heading, which is the shape of a translated TOC.
        start = heading_index + 1
        end = start
        blanks = 0
        while end < len(lines) and end < start + 120:
            if not lines[end].strip():
                blanks += 1
                if blanks >= 2 and end > start:
                    break
            else:
                blanks = 0
            end += 1
        entries = _parse_entries(lines, start, end)
        method = "keyword"
    else:
        return {
            "exists": False,
            "location": None,
            "method": None,
            "entries": [],
            "entry_count": 0,
        }

    exists = bool(entries) or block.get("found", False)
    return {
        "exists": exists,
        "location": "front-matter",
        "filename": path.name,
        "path": str(path),
        "heading": heading or DEFAULT_TOC_HEADING,
        "heading_line": (heading_index + 1) if heading_index is not None else None,
        "start_line": (start + 1) if exists else None,
        "end_line": end if exists else None,
        "method": method,
        "entries": _match_entries_to_chapters(entries, structures),
        "entry_count": len(entries),
    }


def build_toc_entries(
    book: dict[str, Any],
    plan: dict[str, Any],
    *,
    titles: dict[str, str] | None = None,
    include_back_matter: bool | None = None,
) -> list[dict[str, Any]]:
    """The TOC entries implied by the final chapter titles, in reading order."""
    if titles is None:
        from .plan import canonical_titles

        titles = canonical_titles(book, plan)
    include_back = (
        bool((plan.get("toc") or {}).get("include_back_matter"))
        if include_back_matter is None
        else include_back_matter
    )
    entries: list[dict[str, Any]] = []
    for summary in book.get("chapters") or []:
        matter = summary.get("matter")
        if matter != "body" and not (matter == "back" and include_back):
            continue
        filename = str(summary.get("filename") or "")
        title = titles.get(filename) or str(summary.get("title") or "")
        if not filename or not title:
            continue
        entries.append(
            {
                "title": title,
                "filename": filename,
                "number": len(entries) + 1,
                "matter": matter,
            }
        )
        if len(entries) >= MAX_TOC_ENTRIES:
            break
    return entries


def toc_consistency(
    toc: dict[str, Any],
    expected: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare a detected TOC with the expected chapter entries."""
    actual = list(toc.get("entries") or [])
    expected_titles = [str(item.get("title") or "") for item in expected]
    missing: list[str] = []
    matched_indices: list[int] = []
    for title in expected_titles:
        best_index = None
        best_score = 0.0
        for index, entry in enumerate(actual):
            score = similarity(title, str(entry.get("title") or ""))
            if score > best_score:
                best_score, best_index = score, index
        if best_index is None or best_score < 0.82:
            missing.append(title)
        else:
            matched_indices.append(best_index)
    order_ok = matched_indices == sorted(matched_indices)
    stale = [
        str(entry.get("title") or "")
        for index, entry in enumerate(actual)
        if index not in set(matched_indices)
    ]
    return {
        "ok": bool(toc.get("exists")) and not missing and order_ok and not stale,
        "missing": missing,
        "stale": stale[:20],
        "order_ok": order_ok,
        "expected_count": len(expected),
        "actual_count": len(actual),
    }


def toc_matches_entries(toc: dict[str, Any], expected: list[dict[str, Any]]) -> bool:
    """True when the detected TOC lists exactly the expected titles in order."""
    return bool(toc_consistency(toc, expected)["ok"])


def render_toc(
    entries: list[dict[str, Any]],
    heading: str = DEFAULT_TOC_HEADING,
    style: str = "numbered",
) -> str:
    """Render the TOC block as Markdown (heading, blank line, one entry per line)."""
    title = (heading or DEFAULT_TOC_HEADING).strip()
    out = [f"# {title}", ""]
    for index, entry in enumerate(entries, start=1):
        text = str(entry.get("title") or "").strip()
        if not text:
            continue
        if style == "bullets":
            out.append(f"- {text}")
        else:
            out.append(f"{index}. {text}")
    out.append("")
    return "\n".join(out)


def apply_toc(
    chapters_path: str | Path,
    plan: dict[str, Any],
    entries: list[dict[str, Any]],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Create or update the TOC in the postprocessed chapter folder.

    An existing separate TOC file is rewritten; an existing front-matter block is
    replaced in place, preserving everything before and after it. When neither
    exists, ``00-table-of-contents.md`` is created, which sorts right after the
    front matter.
    """
    folder = Path(chapters_path).expanduser()
    toc_plan = plan.get("toc") or {}
    action = str(toc_plan.get("action") or "create").strip().lower()
    heading = str(toc_plan.get("heading") or DEFAULT_TOC_HEADING)
    style = str(toc_plan.get("style") or "numbered").strip().lower()
    rendered = render_toc(entries, heading=heading, style=style)
    structure = sorted(
        (
            path
            for path in folder.glob("*.md")
            if path.is_file() and not path.name.startswith("_")
        ),
        key=lambda path: path.name,
    )
    structures = [
        {
            "filename": path.name,
            "matter": matter_kind(path.name, index),
            "title": "",
        }
        for index, path in enumerate(structure)
    ]
    toc = detect_toc(folder, structures)
    if action == "keep" and toc_matches_entries(toc, entries):
        return {
            "ok": True,
            "action": "keep",
            "changed": False,
            "location": toc.get("location"),
            "filename": toc.get("filename"),
            "entry_count": len(entries),
            "dry_run": dry_run,
        }

    # A keep plan whose TOC does not match is upgraded to an update: the stage's
    # promise is that the TOC is consistent, not that a stale plan is obeyed.
    if action == "keep":
        action = "update"
    if action == "create" and toc.get("exists"):
        action = "update"

    if toc.get("location") == "file" and toc.get("path"):
        target = Path(str(toc["path"]))
        changed = not target.is_file() or target.read_text(encoding="utf-8") != rendered
        if changed and not dry_run:
            write_text(target, rendered)
        return {
            "ok": True,
            "action": "update",
            "changed": changed,
            "location": "file",
            "filename": target.name,
            "path": str(target),
            "line_count": len(rendered.splitlines()),
            "entry_count": len(entries),
            "dry_run": dry_run,
        }

    if toc.get("location") == "front-matter" and toc.get("path"):
        target = Path(str(toc["path"]))
        lines = [line.rstrip("\n") for line in load_lines(target)]
        heading_line = toc.get("heading_line") or toc.get("start_line")
        end_line = toc.get("end_line") or heading_line
        if heading_line is None:
            start = len(lines)
            end = len(lines)
        else:
            start = max(0, int(heading_line) - 1)
            end = min(len(lines), int(end_line or heading_line))
        new_lines = list(lines[:start])
        new_lines.extend(rendered.rstrip("\n").split("\n"))
        new_lines.extend(lines[end:])
        text = "\n".join(new_lines).rstrip("\n") + "\n"
        changed = text != "".join(line + "\n" for line in lines)
        if changed and not dry_run:
            write_text(target, text)
        return {
            "ok": True,
            "action": "update",
            "changed": changed,
            "location": "front-matter",
            "filename": target.name,
            "path": str(target),
            "line_count": len(new_lines),
            "entry_count": len(entries),
            "dry_run": dry_run,
        }

    target = folder / TOC_FILENAME
    changed = not target.is_file() or target.read_text(encoding="utf-8") != rendered
    if changed and not dry_run:
        write_text(target, rendered)
    return {
        "ok": True,
        "action": "create",
        "changed": changed,
        "location": "file",
        "filename": target.name,
        "path": str(target),
        "line_count": len(rendered.splitlines()),
        "entry_count": len(entries),
        "dry_run": dry_run,
    }


def verify_toc(
    chapters_path: str | Path,
    plan: dict[str, Any],
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Verify that the folder's TOC lists exactly the expected chapter titles."""
    folder = Path(chapters_path).expanduser()
    from .inspect import scan_book_structure

    book = scan_book_structure(folder)
    toc = detect_toc(folder, book.get("chapters") or [])
    consistency = toc_consistency(toc, entries)
    problems: list[str] = []
    if not toc.get("exists"):
        problems.append("no table of contents exists")
    if consistency.get("missing"):
        problems.append(
            "the table of contents is missing: " + ", ".join(consistency["missing"][:5])
        )
    if consistency.get("stale"):
        problems.append(
            "the table of contents has stale entries: "
            + ", ".join(consistency["stale"][:5])
        )
    if not consistency.get("order_ok"):
        problems.append("the table of contents entries are out of chapter order")
    return {
        "ok": not problems,
        "problems": problems,
        "toc": {
            key: toc.get(key)
            for key in ("exists", "location", "filename", "heading", "entry_count")
        },
        "consistency": consistency,
        "expected": [
            {"title": item.get("title"), "filename": item.get("filename")}
            for item in entries
        ],
    }


__all__ = [
    "apply_toc",
    "build_toc_entries",
    "detect_toc",
    "render_toc",
    "toc_consistency",
    "toc_matches_entries",
    "verify_toc",
]
