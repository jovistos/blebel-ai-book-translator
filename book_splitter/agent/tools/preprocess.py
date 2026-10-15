"""Stage-2 tools: chapter preprocessing.

The stage has one job: clean *one chapter* of everything that is not part of the
book's text — page numbers, figure captions marked up as headings, headings the
PDF conversion split in two, words a page number was printed inside.

The agent never edits text. It inspects a chapter's headers, then calls
:func:`report_chapter_cleaning` with a list of ``{line_number, decision}``
operations. The graph validates that list against the real file and, only if it
passes, calls :func:`apply_chapter_cleaning`. The decision vocabulary is fixed:

===============  ==========================================================
``keep``         the header is a real heading, leave the line alone
``demote``       remove the ``#`` markup, keep the text as a paragraph
``remove``       the header is only a printed page number — delete the line
``merge_with_next``  this heading continues on the *header* on the next line;
                 the two are joined into one paragraph
================ ==========================================================

The tool that ends the stage for a chapter is ``report_chapter_cleaning``; the
write tools are blocked by the graph until its plan has been validated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from ...core.constants import (
    DEFAULT_MAX_HEADERS,
    PREPROCESS_MANIFEST_FILENAME,
    PREPROCESS_TRACE_FILENAME,
)
from ...documents.chapters import (
    discover_chapters,
    file_sha256,
    preprocessed_dir,
)
from ...documents.io import load_lines
from ...documents.text import excerpt, jdump
from ...preprocess.corpus import scan_corpus_damage
from ...preprocess.damage import damage_digest, scan_chapter_damage
from ...preprocess.inspect import inspect_chapter
from ...preprocess.transform import (
    apply_cleaning_plan,
    text_of,
    validate_cleaning_plan,
    verify_cleaning_plan,
)
from .registry import preprocess_stage


@preprocess_stage
@tool
def extract_headers_from_file(file_path: str, max_headers: int = DEFAULT_MAX_HEADERS) -> str:
    """List every Markdown header in one file, with its exact line number.

    Generic and cheap: it returns `{"line_count", "header_count", "headers":
    [{"line_number", "level", "text"}]}` and nothing else. For a chapter, prefer
    `collect_chapter_headers`, which adds the evidence needed to decide what to
    do with each header.
    """
    from ...documents.markdown import header_of

    lines = load_lines(file_path)
    found: list[dict[str, Any]] = []
    for idx, raw in enumerate(lines, start=1):
        head = header_of(raw)
        if head:
            found.append({"line_number": idx, "level": head[0], "text": excerpt(head[1], 120)})
    return jdump(
        {
            "file": Path(file_path).name,
            "line_count": len(lines),
            "header_count": len(found),
            "headers": found[: max(1, int(max_headers))],
        }
    )


@preprocess_stage
@tool
def collect_chapter_headers(chapter_path: str, include_all: bool = False) -> str:
    """THE FIRST STEP OF PREPROCESSING: extract a chapter's headers with evidence.

    Call this before deciding anything. It returns, for every header in the
    chapter: the line number, the header text (clipped), flags such as
    `is_page_number`, `has_caption_marker`, `next_is_continuation`, and a
    deterministic `recommendation` (`keep` / `demote` / `remove` /
    `remove_caption` / `merge_with_next`) with a `risk` and the `evidence`.

    `review` lists everything that deserves a decision — start there. It also
    includes `body_candidates`: page numbers and figure captions that the PDF
    conversion left as *plain* body lines with no `#` markup, which are just as
    removable as the ones that became headings. Set `include_all: true` to see
    every header record. Reading a chapter's *prose* is never necessary and never
    provided.
    """
    return jdump(inspect_chapter(chapter_path, include_all=include_all))


@preprocess_stage
@tool
def collect_book_headers(chapters_path: str, include_all: bool = False) -> str:
    """Extract the header inventory of every chapter in a split book.

    Same evidence as `collect_chapter_headers`, grouped per chapter, plus the
    book-wide recommendation counts. Use it when a decision in one chapter
    depends on how the rest of the book marks its headings (for example "do
    other chapters keep `## GLAVA n` as a heading?").
    """
    chapters = discover_chapters(chapters_path)
    per_chapter = []
    totals: dict[str, int] = {}
    for chapter in chapters:
        report = inspect_chapter(chapter.path, include_all=include_all)
        for key, value in (report.get("recommendation_counts") or {}).items():
            totals[key] = totals.get(key, 0) + value
        per_chapter.append(
            {
                "chapter_id": chapter.chapter_id,
                "filename": chapter.filename,
                "header_count": report["header_count"],
                "review": report["review"],
                "recommendation_counts": report["recommendation_counts"],
            }
        )
    return jdump(
        {
            "chapters_path": str(chapters_path),
            "chapter_count": len(chapters),
            "recommendation_totals": totals,
            "chapters": per_chapter,
        }
    )


@preprocess_stage
@tool
def inspect_chapter_damage(chapter_path: str, kinds: str = "") -> str:
    """Find conversion damage in one chapter's body text — by shape, not content.

    Returns one finding per damaged spot: the line number, the two fragments and
    a reconstructed word. `kinds` is an optional comma-separated filter from
    `page_number_in_word`, `split_word`, `space_before_punctuation`;
    empty means all of them.

    The chapter's prose is never returned — only the damaged fragment and the
    few characters around it.
    """
    wanted = tuple(k.strip() for k in kinds.split(",") if k.strip()) or None
    report = scan_chapter_damage(chapter_path, kinds=wanted)
    return jdump(report)


@preprocess_stage
@tool
def scan_book_damage(chapters_path: str, max_examples_per_pattern: int = 3) -> str:
    """Count the recognised damage patterns across the whole book.

    Same scan as the generic `scan_corpus_damage_tool`, plus a pre-rendered
    text digest. Use it once per book, before per-chapter work, to learn which
    fixes are actually needed and how common each one is.
    """
    chapters = discover_chapters(chapters_path)
    report = scan_corpus_damage(chapters, max_examples_per_pattern=max_examples_per_pattern)
    report["digest"] = damage_digest(report)
    return jdump(report)


@preprocess_stage
@tool
def validate_chapter_cleaning(chapter_path: str, operations: list[dict]) -> str:
    """Dry-run a cleaning plan for one chapter; checks it against the real file.

    Call this with your candidate operations before reporting them. Every issue
    it lists is fixable from the evidence you already have: an out-of-range
    line, a line that is not a header, a `merge_with_next` whose next line is
    not a header, or a `remove` of a line that is not a page number.
    """

    return jdump(validate_cleaning_plan(chapter_path, operations))

@preprocess_stage
@tool
def report_chapter_cleaning(
    chapter_path: str,
    operations: list[dict],
    text_fixes: list[dict] | None = None,
    notes: str = "",
) -> str:
    """Submit the cleaning plan for one chapter. THIS ENDS THE CHAPTER'S REVIEW.

    `operations` is the complete list of line decisions, each
    `{"line_number": int, "decision": "keep"|"demote"|"remove"|"remove_caption"|
    "merge_with_next", "note": "<why, short>"}`. `remove` deletes a printed page
    number (with or without `#` markup); `remove_caption` deletes a standalone
    figure/table caption, whose picture is not in the Markdown anyway. Lines you
    do not mention are left exactly as they are, so list only what changes.

    `text_fixes` is the optional list of body-text repairs, each
    `{"line_number": int, "fix": "page_number_in_word"|"split_word"|
    "space_before_punctuation"}`. Only request a fix that `inspect_chapter_damage`
    actually reported.

    The plan is validated by the graph before anything is written; if it is
    rejected you will be asked to correct it, so always dry-run it first with
    `validate_chapter_cleaning`.
    """
    cleaned: list[dict[str, Any]] = []
    for entry in list(operations or []):
        if not isinstance(entry, dict):
            continue
        try:
            line_number = int(entry.get("line_number", entry.get("line")))
        except (TypeError, ValueError):
            return jdump(
                {
                    "accepted": False,
                    "error": (
                        "every operation needs an integer line_number, got "
                        f"{entry.get('line_number')!r}"
                    ),
                }
            )
        cleaned.append(
            {
                "line_number": line_number,
                "decision": str(entry.get("decision") or "").strip().lower(),
                "note": excerpt(str(entry.get("note") or ""), 120),
            }
        )
    if not cleaned:
        return jdump({"accepted": False, "error": "operations was empty"})

    fixes: list[dict[str, Any]] = []
    for entry in list(text_fixes or []):
        if not isinstance(entry, dict):
            continue
        try:
            line_number = int(entry.get("line_number", entry.get("line")))
        except (TypeError, ValueError):
            return jdump(
                {
                    "accepted": False,
                    "error": (
                        "every text fix needs an integer line_number, got "
                        f"{entry.get('line_number')!r}"
                    ),
                }
            )
        fixes.append(
            {
                "line_number": line_number,
                "fix": str(entry.get("fix") or entry.get("kind") or "").strip().lower(),
                "note": excerpt(str(entry.get("note") or ""), 120),
            }
        )

    cleaned.sort(key=lambda op: op["line_number"])
    fixes.sort(key=lambda op: op["line_number"])
    return jdump(
        {
            "accepted": True,
            "chapter_path": str(chapter_path),
            "notes": excerpt(str(notes or ""), 300),
            "operation_count": len(cleaned),
            "text_fix_count": len(fixes),
            "operations": cleaned,
            "text_fixes": fixes,
        }
    )


@preprocess_stage
@tool
def apply_chapter_cleaning(
    chapter_path: str,
    operations: list[dict],
    text_fixes: list[dict] | None = None,
    output_dir: str = "",
) -> str:
    """Write one preprocessed chapter file. The only tool that writes content.

    `operations` are the header decisions (see `report_chapter_cleaning`);
    `text_fixes` are optional body-text repairs, each
    `{"line_number": int, "fix": "split_word"|"page_number_in_word"|
    "space_before_punctuation"}` — take them from `inspect_chapter_damage`, and
    remember that `split_word` joins a line with the next one.

    The result is written to `output_dir` (default: the `preprocessed/` folder
    next to `chapters/`). Only bookkeeping is returned — hashes, line counts,
    what was removed, and a per-line provenance map — never the text written.

    The graph blocks this tool until the plan has passed validation for this
    chapter.
    """
    path = Path(chapter_path).expanduser()
    target_dir = Path(output_dir).expanduser() if output_dir else preprocessed_dir(path)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / path.name

    fixes = list(text_fixes or [])
    result = apply_cleaning_plan(path, operations, fixes)
    target.write_text(text_of(result), encoding="utf-8")

    verification = verify_cleaning_plan(path, operations, fixes)
    payload = {
        "ok": bool(verification.get("ok")),
        "chapter_path": str(path),
        "output_path": str(target),
        "output_lines": len(result.lines),
        "source_lines": verification.get("source_lines"),
        "sha256": file_sha256(target),
        "removed_artefacts": result.removed_artefacts,
        "provenance": result.summary(),
        "verification": verification,
    }
    return jdump(payload)


@preprocess_stage
@tool
def verify_preprocessed_chapters(
    chapters_path: str,
    output_dir: str = "",
) -> str:
    """Verify a preprocessed folder against the chapters it came from.

    For every chapter it checks that the output exists, that every source line
    is accounted for by the provenance map, that the only missing lines are
    declared page artefacts, and that no kept line's text changed. Returns
    `{"ok", "chapters": [...], "totals"}`.
    """
    chapters = discover_chapters(chapters_path)
    source = Path(chapters_path).expanduser()
    out = Path(output_dir).expanduser() if output_dir else preprocessed_dir(source)

    reports: list[dict[str, Any]] = []
    ok = True
    for chapter in chapters:
        target = out / chapter.filename
        if not target.is_file():
            ok = False
            reports.append(
                {"filename": chapter.filename, "ok": False, "problem": "output file missing"}
            )
            continue
        source_lines = chapter.line_count
        output_lines = sum(1 for _ in target.open("r", encoding="utf-8", errors="replace"))
        reports.append(
            {
                "filename": chapter.filename,
                "ok": True,
                "source_lines": source_lines,
                "output_lines": output_lines,
                "removed_lines": source_lines - output_lines,
            }
        )
    return jdump(
        {
            "ok": ok and bool(chapters),
            "chapters_path": str(source),
            "output_dir": str(out),
            "chapter_count": len(chapters),
            "chapters": reports,
        }
    )


def manifest_path_for(output_dir: str | Path) -> Path:
    """Where a preprocess run records its per-chapter results."""
    return Path(output_dir).expanduser() / PREPROCESS_MANIFEST_FILENAME


def trace_path_for(output_dir: str | Path) -> Path:
    """Where a preprocess run records its audit trail."""
    return Path(output_dir).expanduser() / PREPROCESS_TRACE_FILENAME


__all__ = [
    "apply_chapter_cleaning",
    "collect_book_headers",
    "collect_chapter_headers",
    "extract_headers_from_file",
    "inspect_chapter_damage",
    "manifest_path_for",
    "report_chapter_cleaning",
    "scan_book_damage",
    "trace_path_for",
    "validate_chapter_cleaning",
    "verify_preprocessed_chapters",
]
