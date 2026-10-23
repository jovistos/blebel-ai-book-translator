"""The matter chunk loop: localize each chunk, or reuse its previous file."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from book_translator.core.errors import ConfigError
from book_translator.documents.chunks import ChunkInput
from book_translator.stages.matter.inspect import MatterBook, MatterSection

from .artifacts import _add_usage, _write_prompt_files
from .models import MatterChunkResult, ProgressHook
from .state import MatterChunkState


@dataclass
class MatterChunkRun:
    inputs: list[ChunkInput]
    reused: list[bool]
    section_by_name: dict[str, MatterSection]
    book: MatterBook
    out_dir: Path
    prompt_dir: Path
    plan: dict[str, Any]
    source_lang: str
    target_lang: str
    chunk_app: Any
    max_attempts: int
    recursion_limit: int
    dry_run: bool
    save: bool
    write_prompts: bool
    progress: ProgressHook | None
    clock: Any
    first_front: ChunkInput | None
    first_in_section: dict[str, str]
    previous_files: dict[str, dict[str, Any]]
    usage_tokens: dict[str, int]


def run_chunks(run: MatterChunkRun) -> tuple[list[MatterChunkResult], list[str], dict[str, int]]:
    records: list[MatterChunkResult] = []
    errors: list[str] = []
    completed = 0
    for position, chunk in enumerate(run.inputs):
        section = run.section_by_name.get(chunk.chapter_filename)
        if section is None:  # pragma: no cover - guarded by the filter above
            continue
        record = MatterChunkResult(
            filename=chunk.filename,
            section_filename=section.filename,
            section_kind=section.kind,
            chunk_index=int(chunk.chunk_index or 0),
            source_chars=len(chunk.text),
            source_sha256=chunk.sha256,
        )
        records.append(record)
        is_reused = run.reused[position]
        first_front_flag = bool(
            run.first_front is not None and chunk.filename == run.first_front.filename
        )
        first_section_flag = (
            run.first_in_section.get(chunk.chapter_filename) == chunk.filename
        )

        if is_reused:
            path = run.out_dir / chunk.filename
            try:
                record.translation = path.read_text(encoding="utf-8")
            except OSError as exc:
                record.ok = False
                record.error = f"{type(exc).__name__}: {exc}"
                errors.append(f"{chunk.filename}: {record.error}")
            record.reused = True
            record.translation_chars = len(record.translation)
            record.omitted = bool(
                (run.previous_files.get(chunk.filename) or {}).get("omitted")
            )
            record.omit_reason = str(
                (run.previous_files.get(chunk.filename) or {}).get("omit_reason") or ""
            )
            record.output_path = str(path)
            record.log.append("reused from the previous run (--resume)")
            completed += 1
            if run.progress is not None:
                with contextlib.suppress(Exception):
                    run.progress(record.to_dict(), completed, len(run.inputs))
            continue

        started = run.clock()
        try:
            if run.chunk_app is None:  # pragma: no cover - guarded by pending
                raise ConfigError("no matter chunk graph is available")
            initial: MatterChunkState = {
                "chapters_path": run.book.chapters_path,
                "chunks_path": run.book.chunks_path,
                "output_dir": str(run.out_dir),
                "prompts_dir": str(run.prompt_dir),
                "book": run.book,
                "section": section,
                "chunk": chunk,
                "plan": run.plan,
                "source_language": run.source_lang,
                "target_language": run.target_lang,
                "max_attempts": max(1, int(run.max_attempts)),
                "dry_run": run.dry_run,
                "save": run.save,
                "is_first_front_chunk": first_front_flag,
                "is_first_section_chunk": first_section_flag,
                "attempts": 0,
                "usage": {},
                "log": [],
                "feedback": [],
                "messages": [],
            }
            final = run.chunk_app.invoke(
                initial, config={"recursion_limit": run.recursion_limit}
            )
        except Exception as exc:
            record.ok = False
            record.error = f"{type(exc).__name__}: {exc}"
            record.log.append(f"localization failed: {record.error}")
            errors.append(f"{chunk.filename}: {record.error}")
            record.elapsed_seconds = run.clock() - started
            completed += 1
            if run.progress is not None:
                with contextlib.suppress(Exception):
                    run.progress(record.to_dict(), completed, len(run.inputs))
            continue

        record.elapsed_seconds = run.clock() - started
        record.attempts = int(final.get("attempts", 0) or 0)
        record.usage = dict(final.get("usage") or {})
        _add_usage(run.usage_tokens, record.usage)
        record.translation = str(final.get("translation") or "")
        record.translation_chars = len(record.translation)
        applied = dict(final.get("applied") or {})
        record.output_path = applied.get("output_path")
        submitted = dict(final.get("chunk_plan") or {})
        record.omitted = bool(submitted.get("omitted"))
        record.omit_reason = str(submitted.get("omit_reason") or "")
        verification = dict(final.get("verification") or {})
        record.ok = bool(verification.get("ok")) and not (
            final.get("result") or {}
        ).get("error")
        record.log.extend(str(item) for item in final.get("log") or [])
        if not record.ok:
            record.error = (
                "; ".join(str(item) for item in verification.get("problems") or [])
                or (final.get("result") or {}).get("error")
                or "the localization was not verified"
            )
            errors.append(f"{chunk.filename}: {record.error}")
        if not run.dry_run and run.save and run.write_prompts:
            with contextlib.suppress(OSError):
                _write_prompt_files(
                    run.prompt_dir,
                    chunk=chunk,
                    section=section,
                    plan=run.plan,
                    final=final,
                    result=record,
                )
        completed += 1
        if run.progress is not None:
            with contextlib.suppress(Exception):
                run.progress(record.to_dict(), completed, len(run.inputs))
    return records, errors, run.usage_tokens


__all__ = ["MatterChunkRun", "run_chunks"]
