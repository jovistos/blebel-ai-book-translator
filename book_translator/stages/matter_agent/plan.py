"""Book-level plan and input preparation for the matter stage."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from book_translator.core.constants import DEFAULT_MATTER_ATTEMPTS, DEFAULT_RECURSION_LIMIT
from book_translator.core.settings import LLMSettings
from book_translator.documents.chapters import matter_dir, matter_prompts_dir, translations_dir
from book_translator.documents.chunks import ChunkInput, discover_chunk_inputs
from book_translator.documents.manifest import load_matter_manifest
from book_translator.stages.matter.inspect import MatterBook, MatterSection, build_matter_book
from book_translator.stages.matter.plan import fallback_matter_plan, validate_matter_plan
from book_translator.stages.matter.synthesize import MatterSynthesisError, ensure_matter_sections

from .artifacts import (
    _chunk_sha256,
    _empty_result,
    _failed_result,
    _is_reusable,
    _maybe_write_trace,
    resolve_chapters_path,
    resolve_chunks_path,
    write_matter_plan_files,
)
from .graph import build_matter_graph
from .models import MatterResult
from .state import MatterPlanState


@dataclass
class MatterPlan:
    chapters: Path
    chunks: Path
    source_lang: str
    target_lang: str
    synthesis: Any
    book: MatterBook
    out_dir: Path
    prompt_dir: Path
    stage_dir: Path
    sections: list[Any]
    section_by_name: dict[str, MatterSection]
    inputs: list[ChunkInput]
    first_front: ChunkInput | None
    first_in_section: dict[str, str]
    previous: dict[str, Any]
    previous_files: dict[str, dict[str, Any]]
    reused: list[bool]
    pending: list[ChunkInput]
    plan: dict[str, Any]
    validation: dict[str, Any]
    book_final: dict[str, Any]
    settings: Any
    limited: bool
    resume_enabled: bool
    dry_run: bool
    save: bool
    max_attempts: int
    recursion_limit: int
    limit: int | None
    chapter_files: Any
    resume: bool
    write_prompts: bool


def prepare_matter(
    chapters_path: str | Path,
    chunks_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    prompts_dir: str | Path | None = None,
    matter_path: str | Path | None = None,
    source_language: str = "",
    target_language: str = "",
    book_title: str = "",
    book_author: str = "",
    settings: LLMSettings | None = None,
    settings_factory: Callable[[], Any] | None = None,
    model: Any | None = None,
    book_model: Any | None = None,
    chunk_model: Any | None = None,
    max_attempts: int = DEFAULT_MATTER_ATTEMPTS,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    dry_run: bool = False,
    save: bool = True,
    resume: bool = False,
    limit: int | None = None,
    chapter_files: Sequence[str] | None = None,
    write_manifest: bool = True,
    write_prompts: bool = True,
) -> MatterPlan | MatterResult:
    chapters = resolve_chapters_path(chapters_path)
    chunks = resolve_chunks_path(chunks_path, chapters)
    source_lang = str(source_language or "")
    target_lang = str(target_language or "")
    synthesis = None
    # A folder without a split manifest is still readable by the older
    # discovery path; creating matter there is not safe. A dry run must also
    # write nothing, so synthesis (which writes chapter and chunk records) is
    # skipped there; sections created by an earlier real run are still read.
    if not dry_run:
        with contextlib.suppress(MatterSynthesisError):
            synthesis = ensure_matter_sections(
                chapters,
                chunks,
                book_title=book_title,
                book_author=book_author,
                create_front=True,
                create_back=True,
            )
    book = build_matter_book(
        chapters,
        chunks,
        source_language=source_lang,
        target_language=target_lang,
        book_title=book_title,
        book_author=book_author,
    )
    out_dir = (
        Path(output_dir).expanduser()
        if output_dir is not None and str(output_dir).strip()
        else translations_dir(chunks)
    )
    prompt_dir = (
        Path(prompts_dir).expanduser()
        if prompts_dir is not None and str(prompts_dir).strip()
        else matter_prompts_dir(chapters)
    )
    stage_dir = (
        Path(matter_path).expanduser()
        if matter_path is not None and str(matter_path).strip()
        else matter_dir(chapters)
    )

    sections = list(book.sections)
    section_by_name = {section.filename: section for section in sections}
    if not sections:
        return _empty_result(
            book=book,
            chapters=chapters,
            chunks=chunks,
            out_dir=out_dir,
            prompt_dir=prompt_dir,
            stage_dir=stage_dir,
            source_lang=source_lang,
            target_lang=target_lang,
            dry_run=dry_run,
            summary="no front/back matter",
        )

    inputs, _source_dir = discover_chunk_inputs(chunks)
    allowed = set(section_by_name)
    inputs = [chunk for chunk in inputs if chunk.chapter_filename in allowed]
    if chapter_files is not None:
        wanted = {str(name) for name in chapter_files}
        inputs = [chunk for chunk in inputs if chunk.chapter_filename in wanted]
    total_before_limit = len(inputs)
    if limit is not None and int(limit) > 0:
        inputs = inputs[: int(limit)]
    limited = bool(limit is not None and int(limit) > 0 and len(inputs) < total_before_limit)
    if not inputs:
        return _empty_result(
            book=book,
            chapters=chapters,
            chunks=chunks,
            out_dir=out_dir,
            prompt_dir=prompt_dir,
            stage_dir=stage_dir,
            source_lang=source_lang,
            target_lang=target_lang,
            dry_run=dry_run,
            summary="no matter chunks",
        )

    for chunk in inputs:
        chunk.sha256 = _chunk_sha256(chunk)
    first_front = next(
        (
            chunk
            for chunk in inputs
            if section_by_name.get(chunk.chapter_filename)
            and section_by_name[chunk.chapter_filename].is_front
        ),
        None,
    )
    first_in_section: dict[str, str] = {}
    for chunk in inputs:
        first_in_section.setdefault(chunk.chapter_filename, chunk.filename)

    previous = load_matter_manifest(stage_dir) if resume and not dry_run else {}
    same_pair = bool(
        previous
        and str(previous.get("target_language") or "") == target_lang
        and str(previous.get("source_language") or "") == source_lang
    )
    resume_enabled = bool(resume and not dry_run and same_pair)
    previous_files = {
        str(entry.get("filename")): entry
        for entry in previous.get("files") or []
        if isinstance(entry, dict) and entry.get("filename")
    }
    reused = [
        _is_reusable(chunk, previous_files.get(chunk.filename), out_dir)
        if resume_enabled
        else False
        for chunk in inputs
    ]
    pending = [chunk for chunk, is_reused in zip(inputs, reused, strict=True) if not is_reused]

    if dry_run:
        # The localization plan and every localized chunk are model answers. A
        # dry run inventories the matter sections/chunks and the resumable
        # prefix, then returns before building either graph.
        return MatterResult(
            ok=True,
            chapters_path=str(book.chapters_path),
            chunks_path=str(chunks),
            output_dir=str(out_dir),
            prompts_dir=str(prompt_dir) if write_prompts else "",
            matter_path=str(stage_dir),
            source_language=source_lang,
            target_language=target_lang,
            book_title=book.book_title,
            book_author=book.book_author,
            source_file=book.source_file,
            synthetic_sections=list(synthesis.created) if synthesis else [],
            synthetic_chunks=list(synthesis.chunk_created) if synthesis else [],
            section_count=len(sections),
            chunk_count=0,
            ok_count=0,
            processed_count=len(pending),
            reused_count=sum(1 for value in reused if value),
            skipped_count=0,
            skipped=not pending,
            dry_run=True,
            save=save,
            resumed=resume_enabled,
            limited=limited,
            book_plan={},
            validation={},
            settings={
                "max_attempts": max(1, int(max_attempts)),
                "recursion_limit": int(recursion_limit),
                "limit": int(limit) if limit is not None else None,
                "chapter_files": list(chapter_files) if chapter_files is not None else None,
                "resume": bool(resume),
                "write_prompts": bool(write_prompts),
            },
            tokens={"chunks": len(pending)},
            totals={
                "sections": len(sections),
                "matter_chunks": len(pending),
                "ok": 0,
                "omitted": 0,
                "reused": sum(1 for value in reused if value),
                "localized_chars": 0,
            },
            timing={"run_seconds": 0.0, "seconds_per_chunk": 0.0},
            sections=[section.to_dict() for section in sections],
            chunks=[],
            errors=[],
        )

    if pending and (
        chunk_model is None
        and model is None
        and settings is None
        and settings_factory is not None
    ):
        settings = settings_factory()

    plan: dict[str, Any] = {}
    validation: dict[str, Any] = {}
    book_final: dict[str, Any] = {}
    if resume_enabled and isinstance(previous.get("book_plan"), dict):
        previous_plan = dict(previous["book_plan"])
        revalidated = validate_matter_plan(
            book, previous_plan, target_language=target_lang
        )
        if revalidated.get("ok"):
            plan = revalidated["plan"]
            validation = revalidated
        else:
            resume_enabled = False

    if not plan:
        if (
            book_model is None
            and model is None
            and settings is None
            and settings_factory is not None
        ):
            settings = settings_factory()
        book_app = build_matter_graph(
            settings,
            model=book_model if book_model is not None else model,
        )
        try:
            book_initial: MatterPlanState = {
                "chapters_path": book.chapters_path,
                "chunks_path": book.chunks_path,
                "book_title": book.book_title,
                "source_language": source_lang,
                "target_language": target_lang,
                "max_attempts": max(1, int(max_attempts)),
                "dry_run": dry_run,
                "book": book,
                "attempts": 0,
                "usage": {},
                "log": [],
                "feedback": [],
                "messages": [],
            }
            book_final = book_app.invoke(
                book_initial, config={"recursion_limit": recursion_limit}
            )
        except Exception as exc:
            # The plan model can fail at the provider boundary (network,
            # malformed multi-tool turn, rate limit). The filename fallback
            # below still gives the book a title page and a back page.
            book_final = {
                "result": {
                    "error": f"book plan failed: {type(exc).__name__}: {exc}"
                },
                "log": [f"book plan failed: {type(exc).__name__}: {exc}"],
                "usage": {},
            }

        validation = dict(book_final.get("validation") or {})
        plan = dict(book_final.get("matter_plan") or validation.get("plan") or {})
        if not validation.get("ok") or not plan:
            # A planning graph that cannot reach the network (or exhausts its
            # turns searching) must not leave the book without a title page.
            # The filename already carries the title and author; use them.
            fallback = fallback_matter_plan(book, target_language=target_lang)
            fallback_validation = validate_matter_plan(
                book, fallback, target_language=target_lang
            )
            if fallback_validation.get("ok"):
                plan = fallback_validation["plan"]
                warnings = list(fallback_validation.get("warnings") or [])
                warnings.append(
                    "the planning model did not produce a plan; used the filename fallback"
                )
                fallback_validation["warnings"] = warnings
                validation = fallback_validation
                book_final = {
                    **book_final,
                    "log": [
                        *list(book_final.get("log") or []),
                        "book_plan: filename fallback (no model plan)",
                    ],
                }
            else:
                error = "; ".join(
                    str(item) for item in fallback_validation.get("issues") or []
                ) or (
                    (book_final.get("result") or {}).get("error")
                    or "the localization plan was rejected"
                )
                result = _failed_result(
                    book=book,
                    chapters=chapters,
                    chunks=chunks,
                    out_dir=out_dir,
                    prompt_dir=prompt_dir,
                    stage_dir=stage_dir,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    dry_run=dry_run,
                    error=f"book plan rejected: {error}",
                    book_plan=plan,
                    validation=validation,
                )
                _maybe_write_trace(stage_dir, result, dry_run, write_manifest)
                return result

    write_matter_plan_files(
        out_dir=out_dir,
        prompt_dir=prompt_dir,
        stage_dir=stage_dir,
        book=book,
        plan=plan,
        validation=validation,
        book_final=book_final,
        write_prompts=write_prompts,
        dry_run=dry_run,
        save=save,
    )
    return MatterPlan(
        chapters=chapters,
        chunks=chunks,
        source_lang=source_lang,
        target_lang=target_lang,
        synthesis=synthesis,
        book=book,
        out_dir=out_dir,
        prompt_dir=prompt_dir,
        stage_dir=stage_dir,
        sections=sections,
        section_by_name=section_by_name,
        inputs=inputs,
        first_front=first_front,
        first_in_section=first_in_section,
        previous=previous,
        previous_files=previous_files,
        reused=reused,
        pending=pending,
        plan=plan,
        validation=validation,
        book_final=book_final,
        settings=settings,
        limited=limited,
        resume_enabled=resume_enabled,
        dry_run=dry_run,
        save=save,
        max_attempts=max_attempts,
        recursion_limit=recursion_limit,
        limit=limit,
        chapter_files=chapter_files,
        resume=resume,
        write_prompts=write_prompts,
    )


__all__ = ["MatterPlan", "prepare_matter"]
