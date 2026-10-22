"""Stage-9 tools: postprocessing the translated chapters' headings and TOC.

The tools are split in two families, and the graph binds only the family a node
needs:

**book plan** (one call per book)
    ``list_chapter_titles``, ``inspect_matter_blocks``, ``web_search`` /
    ``fetch_web_page`` (the published chapter list; the search is mandatory),
    ``validate_book_plan_tool``, ``report_book_plan``.

**chapter plan** (one call per chapter)
    ``inspect_chapter_opening``, ``inspect_chapter_headings``,
    ``inspect_heading_context``, ``validate_chapter_postprocess``,
    ``report_chapter_postprocess``.

The write side is deliberately reachable only by the graph, after a plan has
passed validation: ``apply_chapter_postprocess``, ``apply_book_toc`` and
``verify_postprocessed_chapters`` are in the stage's write-tool set.

Everything an agent sees is bounded metadata. ``inspect_chapter_opening`` returns
at most :data:`MAX_POSTPROCESS_OPENING_LINES` clipped lines, and
``inspect_chapter_headings`` returns one record per heading with a one-line
context, never a paragraph. The stage reuses the splitter's shared
``extract_markdown_headers`` / ``inspect_line_window`` / ``list_chapter_files``
tools through the same registry.
"""

from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from ...core.artifacts import write_text
from ...core.constants import (
    DEFAULT_TOC_HEADING,
    HEADING_DECISIONS,
    MAX_POSTPROCESS_TITLE_CHARS,
    NAMING_MODES,
    TITLE_ACTIONS,
    TOC_ACTIONS,
    TOC_STYLES,
)
from ...core.errors import ToolError
from ...documents.chapters import postprocessed_chapters_dir
from ...documents.text import jdump
from ...postprocess.inspect import (
    inspect_heading_context as inspect_heading_context_fn,
)
from ...postprocess.inspect import (
    inspect_matter_blocks as inspect_matter_blocks_fn,
)
from ...postprocess.inspect import (
    scan_book_structure,
    scan_chapter_structure,
)
from ...postprocess.plan import (
    apply_chapter_plan,
    canonical_titles,
    clean_title,
    normalise_chapter_plan,
    text_of,
    validate_chapter_plan,
    verify_chapter_plan,
)
from ...postprocess.plan import (
    validate_book_plan as validate_book_plan_fn,
)
from ...postprocess.toc import (
    apply_toc,
    build_toc_entries,
    detect_toc,
    verify_toc,
)
from .registry import postprocess_stage

#: The chapter folder a chapter graph is working in. Tool calls are resolved
#: against it, so a model that repeats the bare filename it saw in a report still
#: reaches the real file instead of failing four times.
_CHAPTER_SCOPE: ContextVar[str | None] = ContextVar("postprocess_chapter_scope", default=None)


def set_chapter_scope(chapters_path: str | Path) -> Any:
    """Point the chapter tools at the folder the current chapter lives in."""
    return _CHAPTER_SCOPE.set(str(Path(chapters_path).expanduser()))


def reset_chapter_scope(token: Any) -> None:
    """Restore the previous chapter scope (used by the runner around one chapter)."""
    _CHAPTER_SCOPE.reset(token)


def resolve_chapter_path(path: str | Path) -> Path:
    """Resolve a chapter path, accepting a bare filename from the model.

    The prompt shows the absolute path and tells the model to use it. This is the
    safety net for the common failure mode: the model repeats the filename from
    the report (``01-chapter-1.md``) or invents a prefix. The current chapter
    folder is authoritative; a path that does not exist is searched by basename
    there. A genuinely wrong chapter name still raises a clear error.
    """
    candidate = Path(str(path or "")).expanduser()
    if candidate.is_file():
        return candidate
    base = _CHAPTER_SCOPE.get()
    if base:
        for option in (Path(base) / candidate.name, Path(base) / candidate):
            if option.is_file():
                return option
    where = f" (current chapter folder: {base})" if base else ""
    raise ToolError(
        f"chapter file not found: {path}{where}; pass the exact chapter_path from "
        "the CHAPTER REPORT"
    )


#: Tools the book-plan node may call. Kept here so the graph binds a smaller
#: toolset to the book-level model than to a chapter-level one.
BOOK_TOOL_NAMES = (
    "list_chapter_titles",
    "list_chapter_files",
    "extract_markdown_headers",
    "inspect_chapter_opening",
    "inspect_matter_blocks",
    "audit_book_openings",
    "audit_chapter_opening",
    "web_search",
    "fetch_web_page",
    "validate_book_plan_tool",
    "report_book_plan",
)
#: Tools the per-chapter node may call.
CHAPTER_TOOL_NAMES = (
    "inspect_chapter_opening",
    "inspect_chapter_headings",
    "inspect_heading_context",
    "extract_markdown_headers",
    "inspect_line_window",
    "validate_chapter_postprocess",
    "report_chapter_postprocess",
)


@postprocess_stage
@tool
def list_chapter_titles(chapters_path: str) -> str:
    """THE FIRST STEP: list every chapter title in the book, in reading order.

    This is the cheap, book-level view the naming decision needs: one record per
    chapter with its filename, detected title, title line, numbering and whether
    the title is already marked as a heading. It also reports how many titles
    share a label (``Poglavlje``, ``Chapter``) and whether the set is consistent,
    plus front/back matter and the current table-of-contents state.

    No chapter's prose is returned. Use `inspect_chapter_opening` when one
    chapter's opening needs a closer look.
    """
    path = Path(str(chapters_path or "")).expanduser()
    if not path.is_dir():
        base = _CHAPTER_SCOPE.get()
        if base and (Path(base).parent / path.name).is_dir():
            path = Path(base).parent / path.name
    book = scan_book_structure(path)
    book.pop("structures", None)
    return jdump(book)


@postprocess_stage
@tool
def inspect_matter_blocks(chapters_path: str) -> str:
    """Inspect the front and back matter, and the current table of contents.

    Returns at most 40 clipped lines per block (and 6000 characters), its
    headings, and a deterministic TOC detection: whether a TOC exists, where
    (a separate file or a front-matter block), its entries and its line range.
    This is the only tool that reads outside chapter openings, and it cannot
    return a whole chapter.
    """
    return jdump(inspect_matter_blocks_fn(chapters_path))


@postprocess_stage
@tool
def inspect_chapter_opening(chapter_path: str) -> str:
    """Look at only the first few lines of one chapter — where its title lives.

    Pass the exact `chapter_path` from the CHAPTER REPORT; a bare filename is
    not a path. Returns the detected title candidate (or `source: "missing"`),
    its line, level and whether it is already a Markdown header, the optional
    subtitle, and at most 8 clipped opening lines. This is the only chapter-text
    tool, and it is capped hard: never use it to read a chapter.
    """
    path = resolve_chapter_path(chapter_path)
    structure = scan_chapter_structure(path)
    return jdump(
        {
            "chapter_path": str(path),
            "file": structure.filename,
            "line_count": structure.line_count,
            "matter": structure.matter,
            "chapter_number": structure.chapter_number,
            "title": {
                "text": structure.title,
                "line_number": structure.title_line,
                "level": structure.title_level,
                "is_heading": structure.title_is_heading,
                "source": structure.title_source,
            },
            "subtitle": {
                "text": structure.subtitle,
                "line_number": structure.subtitle_line,
            },
            "opening": structure.opening,
            "note": "OPENING ONLY - clipped lines, capped length. Not the chapter text.",
        }
    )


@postprocess_stage
@tool
def inspect_chapter_headings(chapter_path: str) -> str:
    """List every heading in one chapter with its short surrounding evidence.

    Each record has the line number, level, clipped heading text, a deterministic
    suggestion (`keep` / `demote` / `promote`), a risk, and one-line context:
    whether the next non-blank line is a header, starts lowercase, or is long
    prose. A plain ALL-CAPS section heading that lost its `#` markup is included
    with `level: 0` and `suggested: promote`; a chapter's title and subtitle are
    not. Headings that look accidental are also collected under `suspicious`.

    This is the tool to call before deciding `keep`, `demote` or `promote` for a
    heading. It never returns a paragraph. Pass the exact `chapter_path` from the
    CHAPTER REPORT, never a bare filename.
    """
    path = resolve_chapter_path(chapter_path)
    structure = scan_chapter_structure(path)
    return jdump(
        {
            "chapter_path": str(path),
            "file": structure.filename,
            "line_count": structure.line_count,
            "title": {
                "text": structure.title,
                "line_number": structure.title_line,
                "level": structure.title_level,
                "is_heading": structure.title_is_heading,
                "source": structure.title_source,
            },
            "subtitle": {
                "text": structure.subtitle,
                "line_number": structure.subtitle_line,
            },
            "heading_count": structure.heading_count,
            "headings_truncated": structure.headings_truncated,
            "headings": structure.headings,
            "suspicious_count": len(structure.suspicious_headings),
            "suspicious": structure.suspicious_headings,
            "note": "HEADING RECORDS ONLY - context is clipped to one line each.",
        }
    )


@postprocess_stage
@tool
def inspect_heading_context(chapter_path: str, line_number: int, context: int = 1) -> str:
    """Look at one heading and its immediate neighbourhood (clipped).

    Use this when `inspect_chapter_headings` leaves a heading ambiguous. Returns
    the heading, its deterministic suggestion and evidence, and at most two
    clipped non-blank lines before and after. It is not a way to read a chapter.
    """
    path = resolve_chapter_path(chapter_path)
    return jdump(inspect_heading_context_fn(path, line_number, context_lines=context))


@postprocess_stage
@tool
def validate_book_plan_tool(chapters_path: str, plan: dict) -> str:
    """Dry-run the book-wide naming and table-of-contents plan.

    Checks every chapter title the plan would produce, the naming policy, the
    per-chapter overrides, the mandatory web-search evidence and the TOC action
    against the real folder. Every issue is written so the plan can be corrected
    without reading a chapter.
    """
    return jdump(
        validate_book_plan_fn(chapters_path, plan, require_web_search=True)
    )


@postprocess_stage
@tool
def report_book_plan(plan: dict) -> str:
    """Submit the book-wide naming policy and table-of-contents decision.

    THIS ENDS THE BOOK-LEVEL REVIEW. `plan` is an object:

    `naming`: `{"mode": "keep"|"normalize", "label": "Poglavlje",
    "include_title": bool, "separator": ": ", "level": 1, "overrides":
    [{"filename": "...", "title": "..."}]}`. `normalize` rewrites every body
    chapter title from one label + its chapter number; `include_title` appends
    the chapter's own title after the separator (and is enabled automatically
    when the chapters carry names, so the TOC never drops them); `overrides` are
    the exceptions.

    `title_evidence`: **required** when the graph requires the web search. One line
    with the URL(s) you read from the published chapter list, or the failure/no-result
    outcome. `title_source`: `"web"` / `"local"` / `"mixed"`.

    `toc`: `{"action": "create"|"update"|"keep", "heading": "Sadržaj",
    "style": "numbered"|"bullets", "include_back_matter": bool}`. Use `create`
    when the book has no TOC, `update` when it has one that does not match the
    final chapter titles, and `keep` only when it already matches.

    The graph rejects this call until `web_search` has been called at least once.
    The plan is validated before any chapter is written; dry-run it first with
    `validate_book_plan_tool`.
    """
    if not isinstance(plan, dict):
        return jdump({"accepted": False, "error": "plan must be an object"})
    raw_naming = plan.get("naming")
    naming = raw_naming if isinstance(raw_naming, dict) else {}
    raw_toc = plan.get("toc")
    toc = raw_toc if isinstance(raw_toc, dict) else {}
    normalized: dict[str, Any] = {
        "naming": {
            "mode": str(naming.get("mode") or "keep").strip().lower(),
            "label": clean_title(str(naming.get("label") or "")),
            "include_title": bool(naming.get("include_title")),
            "separator": str(naming.get("separator") or ": ")[:5],
            "level": naming.get("level", 1),
            "overrides": naming.get("overrides") or [],
        },
        "title_evidence": str(plan.get("title_evidence") or "").strip()[:500],
        "title_source": str(plan.get("title_source") or "").strip().lower(),
        "toc": {
            "action": str(toc.get("action") or "keep").strip().lower(),
            "heading": clean_title(str(toc.get("heading") or DEFAULT_TOC_HEADING)),
            "style": str(toc.get("style") or "numbered").strip().lower(),
            "include_back_matter": bool(toc.get("include_back_matter")),
        },
    }
    issues = []
    if normalized["naming"]["mode"] not in NAMING_MODES:
        issues.append(f"naming.mode must be one of {list(NAMING_MODES)}")
    if normalized["toc"]["action"] not in TOC_ACTIONS:
        issues.append(f"toc.action must be one of {list(TOC_ACTIONS)}")
    if normalized["toc"]["style"] not in TOC_STYLES:
        issues.append(f"toc.style must be one of {list(TOC_STYLES)}")
    if issues:
        return jdump({"accepted": False, "error": "invalid plan", "issues": issues})
    return jdump({"accepted": True, "plan": normalized})


@postprocess_stage
@tool
def validate_chapter_postprocess(
    chapter_path: str,
    plan: dict,
    canonical_title: str = "",
) -> str:
    """Dry-run one chapter's title/heading plan against the real file.

    Checks the title action (`keep` / `set` / `insert`), its line and text
    against the book's canonical title, and every heading decision against an
    actual Markdown header or a plain ALL-CAPS heading candidate. Issues are
    actionable; warnings flag suspicious headings the plan leaves untouched.
    """
    path = resolve_chapter_path(chapter_path)
    return jdump(
        validate_chapter_plan(
            path,
            plan,
            canonical_title=canonical_title,
        )
    )


@postprocess_stage
@tool
def report_chapter_postprocess(
    chapter_path: str,
    title: dict,
    headings: list[dict] | None = None,
    notes: str = "",
) -> str:
    """Submit one chapter's title and heading plan. THIS ENDS THE CHAPTER.

    `title` is `{"action": "keep"|"set"|"insert", "line_number": int,
    "text": "...", "level": 1}`. `set` rewrites the existing title line (the
    canonical title comes from the book plan); `insert` adds it at the top when
    the chapter has no title at all; `keep` leaves it alone.

    `headings` is a list of `{"line_number": int,
    "decision": "keep"|"demote"|"promote", "note": "..."}` for the chapter's other
    headings. `demote` removes only the `#` markup and keeps the text as a
    paragraph; `promote` is the inverse, adding `##` to a plain ALL-CAPS section
    heading whose markup was lost. There is no delete operation. Omitted
    headings are kept, but the graph warns about suspicious omissions.

    The plan is validated before anything is written; dry-run it with
    `validate_chapter_postprocess` first.
    """
    normalized = normalise_chapter_plan(
        {"title": title, "headings": headings or [], "notes": notes}
    )
    action = normalized["title"].get("action") or ""
    issues: list[str] = []
    if action not in TITLE_ACTIONS:
        issues.append(f"title.action must be one of {list(TITLE_ACTIONS)}")
    if action in {"set", "insert"} and not normalized["title"].get("text"):
        issues.append("title.text cannot be empty for action 'set' or 'insert'")
    if len(normalized["title"].get("text") or "") > MAX_POSTPROCESS_TITLE_CHARS:
        issues.append(f"title.text is longer than {MAX_POSTPROCESS_TITLE_CHARS} characters")
    for entry in normalized["headings"]:
        if entry["decision"] not in HEADING_DECISIONS:
            issues.append(
                f"heading line {entry['line_number']}: decision must be one of "
                f"{list(HEADING_DECISIONS)}"
            )
    if issues:
        return jdump({"accepted": False, "error": "invalid plan", "issues": issues})
    return jdump(
        {
            "accepted": True,
            "chapter_path": str(Path(chapter_path).expanduser()),
            "plan": normalized,
            "title_action": action,
            "heading_decisions": len(normalized["headings"]),
        }
    )


@postprocess_stage
@tool
def apply_chapter_postprocess(
    chapter_path: str,
    plan: dict,
    canonical_title: str = "",
    output_dir: str = "",
) -> str:
    """Write one postprocessed chapter into the output folder.

    The only chapter-writing tool. It applies exactly the validated plan (title
    action + keep/demote/promote decisions), writes the result under the chapter's own
    name in `output_dir` (default: `09-postprocessed-chapters/`), then verifies that
    only the declared lines changed. The graph blocks it until the plan passed
    validation for this chapter.
    """
    path = resolve_chapter_path(chapter_path)
    out_dir = Path(output_dir).expanduser() if output_dir else postprocessed_chapters_dir(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / path.name
    applied = apply_chapter_plan(path, plan, canonical_title=canonical_title)
    write_text(target, text_of(applied))
    verification = verify_chapter_plan(
        path,
        plan,
        canonical_title=canonical_title,
        output_path=target,
    )
    return jdump(
        {
            "ok": bool(verification.get("ok")),
            "chapter_path": str(path),
            "output_path": str(target),
            "output_lines": len(applied.lines),
            "applied": applied.summary(),
            "verification": verification,
        }
    )


@postprocess_stage
@tool
def apply_book_toc(chapters_path: str, plan: dict) -> str:
    """Create or update the table of contents in a postprocessed folder.

    Entries are generated deterministically from the final chapter titles (the
    same titles the chapter plans were held to), so the TOC cannot disagree with
    the chapters. A missing TOC becomes `00-table-of-contents.md`, which sorts
    right after the front matter; an existing block or file is rewritten.

    The graph blocks this tool until the book plan has been validated.
    """
    book = scan_book_structure(chapters_path)
    titles = canonical_titles(book, plan)
    entries = build_toc_entries(book, plan, titles=titles)
    applied = apply_toc(chapters_path, plan, entries)
    verification = verify_toc(chapters_path, plan, entries)
    return jdump(
        {
            "ok": bool(applied.get("ok")) and bool(verification.get("ok")),
            "applied": applied,
            "verification": verification,
            "entries": entries,
            "toc": detect_toc(chapters_path, book.get("chapters") or []),
        }
    )


@postprocess_stage
@tool
def verify_postprocessed_chapters(
    chapters_path: str,
    source_path: str = "",
) -> str:
    """Verify a postprocessed folder against the chapters it came from.

    Every source chapter must have an output file with the same number of
    headings' worth of structure: no output is missing, no chapter is empty, and
    when `source_path` is given every body chapter has a heading and the TOC
    lists every chapter exactly once.
    """
    from ...postprocess.inspect import chapter_files, scan_book_structure
    from ...postprocess.toc import detect_toc

    folder = Path(chapters_path).expanduser()
    if folder.is_file():
        folder = folder.parent
    outputs = chapter_files(folder)
    records: list[dict[str, Any]] = []
    problems: list[str] = []
    for path in outputs:
        lines = path.read_text(encoding="utf-8").splitlines()
        headings = sum(1 for line in lines if line.lstrip().startswith("#"))
        records.append(
            {
                "filename": path.name,
                "lines": len(lines),
                "headings": headings,
            }
        )
        if not lines:
            problems.append(f"{path.name} is empty")
    toc = detect_toc(folder, scan_book_structure(folder).get("chapters") or [])
    if not toc.get("exists"):
        problems.append("the output folder has no table of contents")
    if source_path:
        source = Path(source_path).expanduser()
        source_files = {path.name for path in chapter_files(source)}
        output_files = {path.name for path in outputs}
        for name in sorted(source_files - output_files):
            problems.append(f"missing output chapter: {name}")
    return jdump(
        {
            "ok": not problems and bool(outputs),
            "chapters_path": str(folder),
            "chapter_count": len(outputs),
            "toc": {
                key: toc.get(key)
                for key in ("exists", "location", "filename", "entry_count")
            },
            "chapters": records,
            "problems": problems,
        }
    )


def book_tools() -> list[Any]:
    """The tools the book-plan graph binds (definition order)."""
    from . import TOOL_BY_NAME

    return [TOOL_BY_NAME[name] for name in BOOK_TOOL_NAMES if name in TOOL_BY_NAME]


def chapter_tools() -> list[Any]:
    """The tools the chapter-plan graph binds (definition order)."""
    from . import TOOL_BY_NAME

    return [TOOL_BY_NAME[name] for name in CHAPTER_TOOL_NAMES if name in TOOL_BY_NAME]


__all__ = [
    "BOOK_TOOL_NAMES",
    "reset_chapter_scope",
    "resolve_chapter_path",
    "set_chapter_scope",
    "CHAPTER_TOOL_NAMES",
    "apply_book_toc",
    "apply_chapter_postprocess",
    "book_tools",
    "chapter_tools",
    "inspect_chapter_headings",
    "inspect_chapter_opening",
    "inspect_heading_context",
    "inspect_matter_blocks",
    "list_chapter_titles",
    "report_book_plan",
    "report_chapter_postprocess",
    "validate_book_plan_tool",
    "validate_chapter_postprocess",
    "verify_postprocessed_chapters",
]
