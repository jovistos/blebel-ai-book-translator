"""Reuse, prompt records and artifacts for the matter stage."""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..core.artifacts import (
    sha256_text,
    timestamp,
    write_json,
    write_manifest,
    write_text,
    write_trace,
)
from ..core.constants import (
    CHUNKS_DIRNAME,
    MATTER_MANIFEST_FILENAME,
    MATTER_TRACE_FILENAME,
)
from ..core.errors import ToolError
from ..documents.chapters import (
    chapters_dir,
)
from ..documents.chunks import ChunkInput
from ..matter.inspect import MatterBook, MatterSection
from .models import MatterChunkResult, MatterResult


def resolve_chapters_path(chapters_path: str | Path) -> Path:
    """The chapters folder, whether handed it, the book folder, or a chunks folder."""
    candidate = Path(chapters_path).expanduser()
    if candidate.is_file():
        candidate = candidate.parent
    if candidate.is_dir() and candidate.name == CHUNKS_DIRNAME:
        candidate = candidate.parent
    try:
        return chapters_dir(candidate)
    except (OSError, ToolError):
        return candidate


def resolve_chunks_path(chunks_path: str | Path | None, chapters: Path) -> Path:
    if chunks_path is not None and str(chunks_path).strip():
        candidate = Path(chunks_path).expanduser()
        return candidate.parent if candidate.is_file() else candidate
    return chapters.parent / CHUNKS_DIRNAME


def _chunk_sha256(chunk: ChunkInput) -> str:
    return sha256_text(chunk.text)


def _is_reusable(
    chunk: ChunkInput,
    record: dict[str, Any] | None,
    out_dir: Path,
) -> bool:
    if not record or not record.get("ok"):
        return False
    recorded = str(record.get("source_sha256") or "")
    if recorded and recorded != chunk.sha256:
        return False
    return (out_dir / chunk.filename).is_file()


def _previous_usage(manifest: dict[str, Any]) -> dict[str, int]:
    tokens = manifest.get("tokens") or {}
    return {
        "input_tokens": int(tokens.get("input_tokens", 0) or 0),
        "output_tokens": int(tokens.get("output_tokens", 0) or 0),
        "total_tokens": int(tokens.get("total_tokens", 0) or 0),
        "llm_calls": int(tokens.get("llm_calls", tokens.get("calls", 0)) or 0),
    }


def _add_usage(total: dict[str, int], usage: dict[str, Any]) -> None:
    for key in ("input_tokens", "output_tokens", "total_tokens", "llm_calls"):
        total[key] = total.get(key, 0) + int(usage.get(key, 0) or 0)


# --------------------------------------------------------------------------- #
# debug records
# --------------------------------------------------------------------------- #
def _serialise_messages(messages: Sequence[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in messages:
        entry: dict[str, Any] = {
            "type": type(message).__name__,
            "content": str(getattr(message, "content", "") or ""),
        }
        calls = getattr(message, "tool_calls", None)
        if calls:
            entry["tool_calls"] = [
                {
                    "name": call.get("name"),
                    "args": call.get("args"),
                    "id": call.get("id"),
                }
                for call in calls
            ]
        name = getattr(message, "name", None)
        if name:
            entry["name"] = name
        status = getattr(message, "status", None)
        if status:
            entry["status"] = status
        out.append(entry)
    return out


def _write_prompt_files(
    prompts_dir: Path,
    *,
    chunk: ChunkInput,
    section: MatterSection,
    plan: dict[str, Any],
    final: dict[str, Any],
    result: MatterChunkResult,
) -> None:
    """Keep the whole agent input of one chunk, JSON and human-readable."""
    messages = _serialise_messages(final.get("messages") or [])
    payload = {
        "stage": "matter",
        "chunk": chunk.to_dict(),
        "section": section.to_dict(),
        "plan": plan,
        "source_text": chunk.text,
        "messages": messages,
        "validation": final.get("validation") or {},
        "verification": final.get("verification") or {},
        "omitted": result.omitted,
        "omit_reason": result.omit_reason,
        "text": result.translation,
        "log": list(final.get("log") or []),
    }
    json_path = prompts_dir / f"{chunk.filename}.json"
    md_path = prompts_dir / f"{chunk.filename}.prompt.md"
    write_json(json_path, payload, default=str)
    lines = [f"# {chunk.filename}", "", "## Messages", ""]
    for message in messages:
        lines.append(f"### {message.get('type')}")
        lines.append("")
        lines.append(str(message.get("content") or ""))
        lines.append("")
        lines.extend(
            f"- tool `{call.get('name')}` args: `{call.get('args')}`"
            for call in message.get("tool_calls") or []
        )
        lines.append("")
    write_text(md_path, "\n".join(lines))
    result.prompt_path = str(json_path)
    result.debug_path = str(md_path)


def _write_book_plan_files(
    matter_path: Path,
    prompts_dir: Path,
    *,
    book: MatterBook,
    plan: dict[str, Any],
    validation: dict[str, Any],
    final: dict[str, Any],
) -> None:
    matter_path.mkdir(parents=True, exist_ok=True)
    write_json(
        matter_path / "book-plan.json",
        {
            "stage": "matter",
            "book": book.to_dict(),
            "plan": plan,
            "validation": validation,
        },
        default=str,
    )
    messages = _serialise_messages(final.get("messages") or [])
    payload = {
        "stage": "matter",
        "phase": "book-plan",
        "book": book.to_dict(),
        "messages": messages,
        "plan": plan,
        "validation": validation,
    }
    prompts_dir.mkdir(parents=True, exist_ok=True)
    write_json(prompts_dir / "_book-plan.json", payload, default=str)
    lines = ["# Book localization plan", ""]
    for message in messages:
        lines.append(f"## {message.get('type')}")
        lines.append("")
        lines.append(str(message.get("content") or ""))
        lines.append("")
    write_text(prompts_dir / "_book-plan.prompt.md", "\n".join(lines))


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #

def _empty_result(
    *,
    book: MatterBook,
    chapters: Path,
    chunks: Path,
    out_dir: Path,
    prompt_dir: Path,
    stage_dir: Path,
    source_lang: str,
    target_lang: str,
    dry_run: bool,
    summary: str,
) -> MatterResult:
    return MatterResult(
        ok=True,
        chapters_path=str(chapters),
        chunks_path=str(chunks),
        output_dir=str(out_dir),
        prompts_dir=str(prompt_dir),
        matter_path=str(stage_dir),
        source_language=source_lang,
        target_language=target_lang,
        book_title=book.book_title,
        book_author=book.book_author,
        source_file=book.source_file,
        section_count=len(book.sections),
        skipped=True,
        dry_run=dry_run,
        totals={"sections": len(book.sections), "matter_chunks": 0},
        sections=[section.to_dict() for section in book.sections],
        errors=[],
        validation={"ok": True, "skipped": summary, "issues": [], "warnings": []},
    )


def _failed_result(
    *,
    book: MatterBook,
    chapters: Path,
    chunks: Path,
    out_dir: Path,
    prompt_dir: Path,
    stage_dir: Path,
    source_lang: str,
    target_lang: str,
    dry_run: bool,
    error: str,
    book_plan: dict[str, Any] | None = None,
    validation: dict[str, Any] | None = None,
) -> MatterResult:
    return MatterResult(
        ok=False,
        chapters_path=str(chapters),
        chunks_path=str(chunks),
        output_dir=str(out_dir),
        prompts_dir=str(prompt_dir),
        matter_path=str(stage_dir),
        source_language=source_lang,
        target_language=target_lang,
        book_title=book.book_title,
        book_author=book.book_author,
        source_file=book.source_file,
        section_count=len(book.sections),
        dry_run=dry_run,
        book_plan=book_plan or {},
        validation=validation or {},
        totals={"sections": len(book.sections), "matter_chunks": 0},
        sections=[section.to_dict() for section in book.sections],
        errors=[error],
    )


def _maybe_write_trace(
    stage_dir: Path, result: MatterResult, dry_run: bool, write_manifest: bool
) -> None:
    if dry_run or not write_manifest:
        return
    try:
        stage_dir.mkdir(parents=True, exist_ok=True)
        result.trace_path = str(write_matter_trace(stage_dir, result))
    except OSError:
        pass


def matter_manifest_payload(result: MatterResult) -> dict[str, Any]:
    return {
        "stage": "matter",
        "generated_at": timestamp(),
        "ok": result.ok,
        "skipped": result.skipped,
        "dry_run": result.dry_run,
        "resumed": result.resumed,
        "limited": result.limited,
        "chapters_path": result.chapters_path,
        "chunks_path": result.chunks_path,
        "output_dir": result.output_dir,
        "source_language": result.source_language,
        "target_language": result.target_language,
        "book_title": result.book_title,
        "source_file": result.source_file,
        "book_plan": result.book_plan,
        "validation": result.validation,
        "settings": result.settings,
        "tokens": result.tokens,
        "totals": result.totals,
        "files": [
            {**record, "verification_ok": bool(record.get("ok"))}
            for record in result.chunks
        ],
        "errors": result.errors,
    }


def write_matter_manifest(out_dir: str | Path, result: MatterResult) -> Path:
    return write_manifest(out_dir, MATTER_MANIFEST_FILENAME, matter_manifest_payload(result))


def write_matter_trace(out_dir: str | Path, result: MatterResult) -> Path:
    """Write ``_matter_trace.json``: the same run, for debugging."""
    return write_trace(out_dir, MATTER_TRACE_FILENAME, result.to_dict())



__all__ = [
    "_add_usage",
    "_chunk_sha256",
    "_empty_result",
    "_failed_result",
    "_is_reusable",
    "_maybe_write_trace",
    "_previous_usage",
    "_serialise_messages",
    "_write_book_plan_files",
    "_write_prompt_files",
    "resolve_chapters_path",
    "resolve_chunks_path",
    "matter_manifest_payload",
    "write_matter_manifest",
    "write_matter_trace",
]


def write_matter_plan_files(
    *,
    out_dir: Path,
    prompt_dir: Path,
    stage_dir: Path,
    book: MatterBook,
    plan: dict[str, Any],
    validation: dict[str, Any],
    book_final: dict[str, Any],
    write_prompts: bool,
    dry_run: bool,
    save: bool,
) -> None:
    """Create the stage folders and record the book plan."""
    if dry_run or not save:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_dir.mkdir(parents=True, exist_ok=True)
    if write_prompts:
        prompt_dir.mkdir(parents=True, exist_ok=True)
    if write_prompts and book_final:
        with contextlib.suppress(OSError):
            _write_book_plan_files(
                stage_dir,
                prompt_dir,
                book=book,
                plan=plan,
                validation=validation,
                final=book_final,
            )
    elif not (stage_dir / "book-plan.json").is_file():
        with contextlib.suppress(OSError):
            write_json(
                stage_dir / "book-plan.json",
                {
                    "stage": "matter",
                    "book": book.to_dict(),
                    "plan": plan,
                    "validation": validation,
                },
                default=str,
            )
