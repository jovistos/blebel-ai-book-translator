"""Entry point for the front/back matter localization stage.

The stage sits between stage 6's per-chunk glossary and stage 7's body
translation. It is deliberately *before* translation and deliberately separate:

```
05-chunks/00000-00-front-matter-0000.md
05-chunks/00001-01-predgovor-0000.md
        │
        ▼  one book-level plan (web search for an existing title, BleBel credit,
        │  omissions, formatting), then one chunk graph per matter chunk
07-translations/00000-00-front-matter-0000.md   ← stage 8 reads this as usual
06b-matter/_matter_manifest.json                 ← plan, per-chunk decisions, tokens
06b-matter-prompts/<chunk>.json / .prompt.md     ← the whole agent input per chunk
```

The translated chunks keep the chunk names, so stage 8 needs no new join rule.
Stage 7 is told to skip these chapters (see ``include_matter=False`` in
:mod:`book_splitter.translation.runner`), which is what makes "front/back matter
has its own prompt and no dictionary" true end to end.

One book plan is produced first, then the chunks are localized in reading order.
The first front-matter chunk is special: the graph guarantees the localized title
and the ``BleBel`` credit are on it. An omitted chunk is written as an empty file
and recorded with its reason; the rest of the book is unaffected.

Nothing here uses the glossary. It is the one translation stage that is not
dictionary-driven.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..agent.tools.matter import reset_matter_scope, set_matter_scope
from ..chunk_glossary.runner import ChunkInput, discover_chunk_inputs
from ..core.constants import (
    CHUNKS_DIRNAME,
    DEFAULT_MATTER_ATTEMPTS,
    DEFAULT_RECURSION_LIMIT,
    MATTER_MANIFEST_FILENAME,
    MATTER_TRACE_FILENAME,
)
from ..core.errors import ConfigError, ToolError
from ..core.settings import LLMSettings
from ..documents.chapters import (
    chapters_dir,
    matter_dir,
    matter_prompts_dir,
    translations_dir,
)
from ..matter.inspect import MatterBook, MatterSection, build_matter_book
from ..matter.plan import fallback_matter_plan, validate_matter_plan
from ..matter.synthesize import MatterSynthesisError, ensure_matter_sections
from .graph import build_matter_chunk_graph, build_matter_graph
from .state import MatterChunkState, MatterPlanState

#: Monotonic clock used for timings; injectable so tests can be exact.
Clock = Callable[[], float]
#: Called once per finished matter chunk as ``(record, completed, total)``.
ProgressHook = Callable[[dict[str, Any], int, int], None]


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #
@dataclass
class MatterChunkResult:
    """One localized matter chunk: what it cost, where it went, whether it was omitted."""

    filename: str
    section_filename: str = ""
    section_kind: str = ""
    chunk_index: int = 0
    source_chars: int = 0
    translation_chars: int = 0
    source_sha256: str = ""
    omitted: bool = False
    omit_reason: str = ""
    attempts: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    output_path: str | None = None
    prompt_path: str | None = None
    debug_path: str | None = None
    ok: bool = True
    skipped: bool = False
    reused: bool = False
    error: str | None = None
    log: list[str] = field(default_factory=list)
    #: The localized text. Kept out of the manifest (the file is the artifact).
    translation: str = field(default="", repr=False, compare=False)

    def record(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "section_filename": self.section_filename,
            "section_kind": self.section_kind,
            "chunk_index": self.chunk_index,
            "source_chars": self.source_chars,
            "translation_chars": self.translation_chars,
            "source_sha256": self.source_sha256,
            "omitted": self.omitted,
            "omit_reason": self.omit_reason,
            "attempts": self.attempts,
            "usage": dict(self.usage),
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "output_path": self.output_path,
            "prompt_path": self.prompt_path,
            "debug_path": self.debug_path,
            "ok": self.ok,
            "skipped": self.skipped,
            "reused": self.reused,
            "error": self.error,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.record(), "log": list(self.log)}


@dataclass
class MatterResult:
    """Outcome of a whole front/back matter localization run."""

    ok: bool
    chapters_path: str
    chunks_path: str
    output_dir: str
    prompts_dir: str
    matter_path: str
    source_language: str = ""
    target_language: str = ""
    book_title: str = ""
    book_author: str = ""
    source_file: str = ""
    synthetic_sections: list[str] = field(default_factory=list)
    synthetic_chunks: list[str] = field(default_factory=list)
    section_count: int = 0
    chunk_count: int = 0
    ok_count: int = 0
    processed_count: int = 0
    reused_count: int = 0
    skipped_count: int = 0
    skipped: bool = False
    dry_run: bool = False
    save: bool = True
    resumed: bool = False
    limited: bool = False
    book_plan: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)
    totals: dict[str, int] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    sections: list[dict[str, Any]] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    manifest_path: str | None = None
    trace_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# paths and resume
# --------------------------------------------------------------------------- #
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


def load_matter_manifest(matter_path: str | Path) -> dict[str, Any]:
    """Read ``_matter_manifest.json`` from the stage folder."""
    path = Path(matter_path).expanduser() / MATTER_MANIFEST_FILENAME
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _chunk_sha256(chunk: ChunkInput) -> str:
    return hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()


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
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    lines = [f"# {chunk.filename}", "", "## Messages", ""]
    for message in messages:
        lines.append(f"### {message.get('type')}")
        lines.append("")
        lines.append(str(message.get("content") or ""))
        lines.append("")
        for call in message.get("tool_calls") or []:
            lines.append(f"- tool `{call.get('name')}` args: `{call.get('args')}`")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
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
    (matter_path / "book-plan.json").write_text(
        json.dumps(
            {
                "stage": "matter",
                "book": book.to_dict(),
                "plan": plan,
                "validation": validation,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
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
    (prompts_dir / "_book-plan.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    lines = ["# Book localization plan", ""]
    for message in messages:
        lines.append(f"## {message.get('type')}")
        lines.append("")
        lines.append(str(message.get("content") or ""))
        lines.append("")
    (prompts_dir / "_book-plan.prompt.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def run_matter(
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
    progress: ProgressHook | None = None,
    clock: Clock = time.perf_counter,
) -> MatterResult:
    """Localize every front/back matter chunk, after one book-level plan.

    ``book_model``/``chunk_model`` are injectable so the whole graph can be run
    offline against a scripted double. A real run builds them from ``settings``
    (or ``settings_factory``), resolved only when a model is actually needed, so
    a fully resumed run needs no API key.

    ``resume`` reuses a chunk whose manifest record is successful, whose source
    hash still matches, and whose translated file exists. It also reuses the
    previous book plan, which is what keeps the title/credit/omissions stable
    across a widened run.

    ``limit`` is a chunk limit inside the matter window, for a cheap test of the
    whole path. ``chapter_files`` restricts the run to named matter sections.
    ``dry_run`` still asks the model — the answer is the artifact — but writes no
    file, manifest or prompt record.
    """
    run_started = clock()
    chapters = resolve_chapters_path(chapters_path)
    chunks = resolve_chunks_path(chunks_path, chapters)
    source_lang = str(source_language or "")
    target_lang = str(target_language or "")
    synthesis = None
    try:
        synthesis = ensure_matter_sections(
            chapters,
            chunks,
            book_title=book_title,
            book_author=book_author,
            create_front=True,
            create_back=True,
        )
    except MatterSynthesisError:
        # A folder without a split manifest is still readable by the older
        # discovery path; creating matter there is not safe.
        pass
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
        scope = set_matter_scope(book.chapters_path, book.chunks_path, book=book)
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
        finally:
            reset_matter_scope(scope)

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

    if not dry_run and save:
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
                (stage_dir / "book-plan.json").write_text(
                    json.dumps(
                        {
                            "stage": "matter",
                            "book": book.to_dict(),
                            "plan": plan,
                            "validation": validation,
                        },
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    ),
                    encoding="utf-8",
                )

    if not resume_enabled:
        # A plan that could not be reused invalidates the earlier chunk reuse:
        # the title/credit/omissions may have changed, so localize again.
        reused = [False] * len(inputs)
        pending = list(inputs)

    usage_tokens = _previous_usage(previous) if resume_enabled else {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "llm_calls": 0,
    }
    if book_final:
        _add_usage(usage_tokens, book_final.get("usage") or {})

    chunk_app = (
        build_matter_chunk_graph(
            settings,
            model=chunk_model if chunk_model is not None else model,
        )
        if pending
        else None
    )

    records: list[MatterChunkResult] = []
    errors: list[str] = []
    completed = 0
    for position, chunk in enumerate(inputs):
        section = section_by_name.get(chunk.chapter_filename)
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
        is_reused = reused[position]
        first_front_flag = bool(first_front is not None and chunk.filename == first_front.filename)
        first_section_flag = first_in_section.get(chunk.chapter_filename) == chunk.filename

        if is_reused:
            path = out_dir / chunk.filename
            try:
                record.translation = path.read_text(encoding="utf-8")
            except OSError as exc:
                record.ok = False
                record.error = f"{type(exc).__name__}: {exc}"
                errors.append(f"{chunk.filename}: {record.error}")
            record.reused = True
            record.translation_chars = len(record.translation)
            record.omitted = bool((previous_files.get(chunk.filename) or {}).get("omitted"))
            record.omit_reason = str(
                (previous_files.get(chunk.filename) or {}).get("omit_reason") or ""
            )
            record.output_path = str(path)
            record.log.append("reused from the previous run (--resume)")
            completed += 1
            if progress is not None:
                with contextlib.suppress(Exception):
                    progress(record.to_dict(), completed, len(inputs))
            continue

        started = clock()
        scope = set_matter_scope(
            book.chapters_path,
            book.chunks_path,
            book=book,
            section=section,
            chunk=chunk,
            plan=plan,
            is_first_front_chunk=first_front_flag,
            is_first_section_chunk=first_section_flag,
        )
        try:
            if chunk_app is None:  # pragma: no cover - guarded by pending
                raise ConfigError("no matter chunk graph is available")
            initial: MatterChunkState = {
                "chapters_path": book.chapters_path,
                "chunks_path": book.chunks_path,
                "output_dir": str(out_dir),
                "prompts_dir": str(prompt_dir),
                "book": book,
                "section": section,
                "chunk": chunk,
                "plan": plan,
                "source_language": source_lang,
                "target_language": target_lang,
                "max_attempts": max(1, int(max_attempts)),
                "dry_run": dry_run,
                "save": save,
                "is_first_front_chunk": first_front_flag,
                "is_first_section_chunk": first_section_flag,
                "attempts": 0,
                "usage": {},
                "log": [],
                "feedback": [],
                "messages": [],
            }
            final = chunk_app.invoke(initial, config={"recursion_limit": recursion_limit})
        except Exception as exc:
            record.ok = False
            record.error = f"{type(exc).__name__}: {exc}"
            record.log.append(f"localization failed: {record.error}")
            errors.append(f"{chunk.filename}: {record.error}")
            record.elapsed_seconds = clock() - started
            completed += 1
            if progress is not None:
                with contextlib.suppress(Exception):
                    progress(record.to_dict(), completed, len(inputs))
            continue
        finally:
            reset_matter_scope(scope)

        record.elapsed_seconds = clock() - started
        record.attempts = int(final.get("attempts", 0) or 0)
        record.usage = dict(final.get("usage") or {})
        _add_usage(usage_tokens, record.usage)
        record.translation = str(final.get("translation") or "")
        record.translation_chars = len(record.translation)
        applied = dict(final.get("applied") or {})
        record.output_path = applied.get("output_path")
        submitted = dict(final.get("chunk_plan") or {})
        record.omitted = bool(submitted.get("omitted"))
        record.omit_reason = str(submitted.get("omit_reason") or "")
        verification = dict(final.get("verification") or {})
        record.ok = bool(verification.get("ok")) and not (final.get("result") or {}).get(
            "error"
        )
        record.log.extend(str(item) for item in final.get("log") or [])
        if not record.ok:
            record.error = (
                "; ".join(str(item) for item in verification.get("problems") or [])
                or (final.get("result") or {}).get("error")
                or "the localization was not verified"
            )
            errors.append(f"{chunk.filename}: {record.error}")
        if not dry_run and save and write_prompts:
            with contextlib.suppress(OSError):
                _write_prompt_files(
                    prompt_dir,
                    chunk=chunk,
                    section=section,
                    plan=plan,
                    final=final,
                    result=record,
                )
        completed += 1
        if progress is not None:
            with contextlib.suppress(Exception):
                progress(record.to_dict(), completed, len(inputs))

    totals = {
        "sections": len(sections),
        "matter_chunks": len(inputs),
        "ok": sum(1 for record in records if record.ok),
        "omitted": sum(1 for record in records if record.omitted),
        "reused": sum(1 for record in records if record.reused),
        "localized_chars": sum(record.translation_chars for record in records),
    }
    ok = bool(records) and all(record.ok for record in records) and not errors
    result = MatterResult(
        ok=ok,
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
        chunk_count=len(records),
        ok_count=sum(1 for record in records if record.ok),
        processed_count=len(pending),
        reused_count=sum(1 for record in records if record.reused),
        skipped_count=0,
        skipped=not records,
        dry_run=dry_run,
        save=save,
        resumed=resume_enabled,
        limited=limited,
        book_plan=plan,
        validation=validation,
        settings={
            "max_attempts": max(1, int(max_attempts)),
            "recursion_limit": int(recursion_limit),
            "limit": int(limit) if limit is not None else None,
            "chapter_files": list(chapter_files) if chapter_files is not None else None,
            "resume": bool(resume),
            "write_prompts": bool(write_prompts),
        },
        tokens={**usage_tokens, "chunks": len(pending)},
        totals=totals,
        timing={
            "run_seconds": round(clock() - run_started, 4),
            "seconds_per_chunk": round(
                (clock() - run_started) / len(records), 4
            )
            if records
            else 0.0,
        },
        sections=[section.to_dict() for section in sections],
        chunks=[record.record() for record in records],
        errors=errors,
    )
    if write_manifest and not dry_run and save:
        result.manifest_path = str(write_matter_manifest(stage_dir, result))
        result.trace_path = str(write_matter_trace(stage_dir, result))
    return result


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


def write_matter_manifest(out_dir: str | Path, result: MatterResult) -> Path:
    """Write ``_matter_manifest.json``: the plan and the per-chunk index."""
    path = Path(out_dir).expanduser() / MATTER_MANIFEST_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "matter",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
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
            {
                **record,
                "verification_ok": bool(record.get("ok")),
            }
            for record in result.chunks
        ],
        "errors": result.errors,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_matter_trace(out_dir: str | Path, result: MatterResult) -> Path:
    """Write ``_matter_trace.json``: the same run, for debugging."""
    path = Path(out_dir).expanduser() / MATTER_TRACE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


__all__ = [
    "Clock",
    "MatterChunkResult",
    "MatterResult",
    "ProgressHook",
    "load_matter_manifest",
    "resolve_chapters_path",
    "resolve_chunks_path",
    "run_matter",
    "write_matter_manifest",
    "write_matter_trace",
]
