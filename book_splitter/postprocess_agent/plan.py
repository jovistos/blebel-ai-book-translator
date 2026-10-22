"""Book plan and input preparation for the postprocess stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.constants import (
    DEFAULT_POSTPROCESS_ATTEMPTS,
    DEFAULT_RECURSION_LIMIT,
    TRANSLATED_CHAPTERS_DIRNAME,
)
from ..core.errors import ToolError
from ..core.settings import LLMSettings
from ..documents.chapters import postprocessed_chapters_dir
from ..postprocess.inspect import chapter_files, scan_book_structure
from .artifacts import _selected_files, write_postprocess_trace
from .graph import build_postprocess_graph
from .models import PostprocessResult
from .result import build_postprocess_result
from .state import BookPlanState


@dataclass
class PostprocessPlan:
    source_dir: Path
    out_dir: Path
    files: list[Path]
    selected: list[Path]
    selected_names: set[str]
    book: dict[str, Any]
    plan: dict[str, Any]
    validation: dict[str, Any]
    web: dict[str, Any]
    canonical: dict[str, str]
    errors: list[str]
    usage_total: dict[str, int]
    settings: Any
    dry_run: bool
    write_manifest: bool


def prepare_postprocess(
    chapters_path: str | Path,
    output_dir: str | Path | None = None,
    settings: LLMSettings | None = None,
    max_attempts: int = DEFAULT_POSTPROCESS_ATTEMPTS,
    dry_run: bool = False,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    limit: int | None = None,
    require_web_search: bool = True,
    model: Any | None = None,
    book_model: Any | None = None,
    write_manifest: bool = True,
) -> PostprocessPlan | PostprocessResult:
    source_dir = Path(chapters_path).expanduser()
    if source_dir.is_file():
        source_dir = source_dir.parent
    source_dir = source_dir.resolve()
    nested = source_dir / TRANSLATED_CHAPTERS_DIRNAME
    if nested.is_dir():
        source_dir = nested
    files = chapter_files(source_dir)
    if not files:
        raise ToolError(
            f"no translated chapter files in {source_dir} — run `concatenate` first"
        )
    out_dir = (
        Path(output_dir).expanduser()
        if output_dir is not None
        else postprocessed_chapters_dir(source_dir)
    )
    book = scan_book_structure(source_dir)
    matters = {
        str(summary.get("filename")): str(summary.get("matter") or "body")
        for summary in book.get("chapters") or []
    }
    try:
        if out_dir.resolve() == source_dir.resolve():
            raise ToolError(
                "the postprocessed output folder must be different from the input "
                "translated-chapters folder"
            )
    except OSError:
        pass
    selected = _selected_files(files, limit, matters)
    selected_names = {path.name for path in selected}

    book_app = build_postprocess_graph(
        settings,
        model=book_model if book_model is not None else model,
    )
    book_initial: BookPlanState = {
        "chapters_path": str(source_dir),
        "max_attempts": max_attempts,
        "dry_run": dry_run,
        "require_web_search": require_web_search,
        "web_search_attempted": False,
        "web_searches": [],
        "attempts": 0,
        "usage": {},
        "log": [],
        "feedback": [],
        "messages": [],
    }
    web: dict[str, Any] = {
        "required": require_web_search,
        "attempted": False,
        "searches": [],
    }
    try:
        book_final = book_app.invoke(
            book_initial, config={"recursion_limit": recursion_limit}
        )
    except Exception as exc:
        result = build_postprocess_result(
            ok=False,
            source_dir=source_dir,
            out_dir=out_dir,
            chapter_count=len(selected),
            dry_run=dry_run,
            limited=False,
            first_chapters=None,
            web=web,
            errors=[f"book plan failed: {type(exc).__name__}: {exc}"],
        )
        if not dry_run and write_manifest:
            out_dir.mkdir(parents=True, exist_ok=True)
            result.trace_path = str(write_postprocess_trace(out_dir, result))
        return result
    validation = book_final.get("validation") or {}
    plan = book_final.get("book_plan") or {}
    web = {
        "required": require_web_search,
        "attempted": bool(book_final.get("web_search_attempted")),
        "searches": list(book_final.get("web_searches") or []),
    }
    if validation.get("naming"):
        # Validation normalises the plan (and may enable the optional chapter
        # names); record the effective policy rather than the submitted one.
        plan = {
            **plan,
            "naming": validation["naming"],
            "toc": validation.get("toc") or plan.get("toc") or {},
            "title_evidence": validation.get(
                "title_evidence", plan.get("title_evidence") or ""
            ),
            "title_source": validation.get(
                "title_source", plan.get("title_source") or ""
            ),
        }
    canonical = dict(validation.get("canonical_titles") or {})
    errors: list[str] = []
    usage_total: dict[str, int] = dict(book_final.get("usage") or {})

    if not validation.get("ok"):
        detail = validation.get("issues") or [(book_final.get("result") or {}).get("error")]
        errors.append(
            "book plan rejected: "
            + "; ".join(str(item) for item in detail if item)
        )
        result = build_postprocess_result(
            ok=False,
            source_dir=source_dir,
            out_dir=out_dir,
            chapter_count=len(selected),
            dry_run=dry_run,
            limited=False,
            first_chapters=None,
            book_plan=plan,
            validation=validation,
            web=web,
            usage=usage_total,
            errors=errors,
        )
        if not dry_run and write_manifest:
            out_dir.mkdir(parents=True, exist_ok=True)
            result.trace_path = str(write_postprocess_trace(out_dir, result))
        return result
    return PostprocessPlan(
        source_dir=source_dir,
        out_dir=out_dir,
        files=files,
        selected=selected,
        selected_names=selected_names,
        book=book,
        plan=plan,
        validation=validation,
        web=web,
        canonical=canonical,
        errors=errors,
        usage_total=usage_total,
        settings=settings,
        dry_run=dry_run,
        write_manifest=write_manifest,
    )


__all__ = ["PostprocessPlan", "prepare_postprocess"]
