"""Turning a chapter map into concrete, safe output ranges.

Two responsibilities:

* **Repair** the filenames the model proposed, without ever trusting them as
  paths (``normalize_filename`` strips directory components; duplicates are
  disambiguated instead of overwritten).
* **Plan** the exact ``(filename, start_line, end_line)`` ranges for the whole
  book, front matter included, so a dry run and a real slice agree by
  construction.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from itertools import pairwise
from typing import Any

from ..analysis.backmatter import is_back_matter_role
from ..core.constants import FRONT_MATTER_FILENAME, MAX_GAP_LINES, MIN_GAP_LINES
from ..core.errors import ToolError
from ..documents.markdown import header_of
from ..documents.text import excerpt, norm, normalize_filename, similarity, slugify


def unsafe_filename(name: str) -> str | None:
    """Why a proposed filename cannot be used, or ``None`` when it is fine."""
    if not name or name != name.rsplit("/", 1)[-1] or "/" in name or "\\" in name:
        return "filename must not contain a directory component"
    if name in {".", ".."} or name.startswith("."):
        return "filename must not be hidden or relative"
    if ".." in name:
        return "filename must not contain '..'"
    return None


def reconcile_filename(name: str, line_text: str) -> str:
    """Repair a filename whose slug is a near-miss of its own heading.

    Models occasionally mistype a short heading when naming a file
    (``15-belestke.md`` for ``## BELEŠKE``). Only short, one-or-two-word slugs
    that are neither a substring nor a superstring of the heading are touched,
    so descriptive names such as ``05-glava-2-seks-i-srce.md`` are always kept.
    """
    head = header_of(line_text)
    title = head[1] if head else line_text.strip()
    if not title or not 4 <= len(title) <= 30:
        return name
    m = re.match(r"^(\d{1,3})-(.*)\.md$", name)
    prefix = m.group(1) if m else ""
    slug = m.group(2) if m else (name[:-3] if name.endswith(".md") else name)
    if len(slug.split("-")) > 2:
        return name
    nslug, ntitle = norm(slug), norm(title)
    if not nslug or not ntitle or nslug in ntitle or ntitle in nslug:
        return name
    if similarity(slug, title) >= 0.6:
        fixed = slugify(title)
        return f"{prefix}-{fixed}.md" if prefix else f"{fixed}.md"
    return name


def plan_files(
    lines: list[str],
    split_mapping: Iterable[dict[str, Any]],
    include_front_matter: bool = True,
    front_matter_filename: str = FRONT_MATTER_FILENAME,
) -> list[dict[str, Any]]:
    """Turn a validated chapter map into concrete line ranges.

    Shared by the real slicer and by ``--dry-run`` previews, so what you preview
    is exactly what gets written. Raises :class:`ToolError` for out-of-range
    line numbers.

    An entry carrying ``role: "back-matter"`` is planned with ``kind ==
    "back-matter"`` instead of ``"chapter"``, so the manifest, the chapter count
    and stage 2 all see the trailing publisher/author block for what it is.
    """
    entries = sorted(
        (
            (
                int(e["line_number"]),
                normalize_filename(str(e.get("filename", ""))),
                "back-matter" if is_back_matter_role(e.get("role") or e.get("kind")) else "chapter",
            )
            for e in split_mapping
        ),
        key=lambda entry: entry[0],
    )
    if not entries:
        raise ToolError("split_mapping is empty; nothing to slice")
    for ln, _, _ in entries:
        if not 1 <= ln <= len(lines):
            raise ToolError(f"line {ln} is outside the file (1..{len(lines)})")

    deduped: list[tuple[int, str, str]] = []
    for ln, name, kind in entries:
        if deduped and deduped[-1][0] == ln:
            continue
        deduped.append((ln, name, kind))
    entries = deduped

    used: set[str] = set()
    plan: list[tuple[int, str, str]] = []
    for ln, name, kind in entries:
        name = reconcile_filename(name, lines[ln - 1])
        if name in used or name == normalize_filename(front_matter_filename):
            name = f"{name[:-3]}-{ln}.md"
        used.add(name)
        plan.append((ln, name, kind))

    planned: list[dict[str, Any]] = []
    if include_front_matter and plan[0][0] > 1:
        fm_name = normalize_filename(front_matter_filename)
        if fm_name in used:
            fm_name = f"00-front-matter-{plan[0][0]}.md"
        used.add(fm_name)
        planned.append(
            {
                "filename": fm_name,
                "start_line": 1,
                "end_line": plan[0][0] - 1,
                "kind": "front-matter",
            }
        )

    for i, (start, name, role) in enumerate(plan):
        end = plan[i + 1][0] - 1 if i + 1 < len(plan) else len(lines)
        kind = role if role == "back-matter" else ("front-matter" if i == 0 and start == 1 else "chapter")
        planned.append({"filename": name, "start_line": start, "end_line": end, "kind": kind})
    return planned


def describe_first_line(lines: list[str], start: int, end: int) -> str:
    """Short excerpt of the first non-blank line in a range (never prose)."""
    return next((excerpt(line, 80) for line in lines[start - 1 : end + 3] if line.strip()), "")


def gap_warnings(points: list[dict[str, Any]], max_gap_lines: int = MAX_GAP_LINES) -> list[str]:
    """Plausibility warnings about the distance between consecutive split points."""
    warnings: list[str] = []
    ordered = sorted(points, key=lambda p: p["line_number"])
    for prev, cur in pairwise(ordered):
        gap = cur["line_number"] - prev["line_number"]
        if gap < MIN_GAP_LINES:
            warnings.append(
                f"{prev['filename']} -> {cur['filename']} are only {gap} line(s) apart; "
                "one of them may be a duplicate/noise split"
            )
        if gap > max_gap_lines:
            warnings.append(
                f"gap of {gap} lines between {prev['filename']} and {cur['filename']}; "
                "a chapter may be missing in between"
            )
    return warnings
