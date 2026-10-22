"""Discovery, guards and resume for the translation stage."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.artifacts import sha256_text
from ..core.constants import (
    CHAPTERS_DIRNAME,
    DEFAULT_SOURCE_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
    DEFAULT_TRANSLATION_ATTEMPTS,
    DEFAULT_TRANSLATION_CONTEXT_CHARS,
    DEFAULT_TRANSLATION_MAX_TOKENS,
    TRANSLATION_GLOSSARY_SCOPES,
    TRANSLATION_METHODS,
)
from ..core.errors import ConfigError, ToolError
from ..core.run import RunContext
from ..documents.chapters import (
    translation_prompts_dir,
    translations_dir,
)
from ..documents.chunks import (
    ChunkInput,
    discover_chunk_inputs,
    resolve_glossary_path,
)
from ..documents.manifest import load_translation_manifest
from .prompts import build_system_prompt
from .records import ChunkStartHook, Clock, GlossaryRow, ProgressHook
from .support import (
    _book_title,
    _glossary_fingerprint,
    _int,
    _is_reusable,
    _model_identity,
    _source_file,
    _translation_resume_key,
    has_chunk_glossary_files,
    load_book_glossary_rows,
    resolve_chunk_glossary_dir,
)
from .translator import ChunkTranslator, make_translator


@dataclass
class TranslationPlan:
    """Everything the chunk loop and the result builder need."""

    inputs: list[ChunkInput]
    source_dir: Path
    out_dir: Path
    prompt_dir: Path
    title: str
    source_file: str
    scope: str
    method: str
    attempts: int
    max_tokens: int | None
    context_size: int
    source_language: str
    target_language: str
    save: bool
    dry_run: bool
    limited: bool
    include_matter: bool
    matter_skipped: int
    write_manifest: bool
    write_prompts: bool
    chunk_start: ChunkStartHook | None
    context: RunContext
    clock: Clock
    resolved_chunk_glossary: Path
    resolved_glossary: Path
    book_rows: list[GlossaryRow] = field(default_factory=list)
    system_prompt: str = ""
    previous_manifest: dict[str, Any] = field(default_factory=dict)
    previous_files: dict[str, dict[str, Any]] = field(default_factory=dict)
    carried_from: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_identity: str = ""
    resume_keys: dict[str, str] = field(default_factory=dict)
    reused: list[bool] = field(default_factory=list)
    resumed: bool = False
    pending: list[ChunkInput] = field(default_factory=list)
    translator: ChunkTranslator | None = None
    translator_name: str = ""
    settings: Any = None
    settings_payload: dict[str, Any] = field(default_factory=dict)


def prepare_translation(
    chunks_path: str,
    output_dir: str | None = None,
    *,
    prompts_dir: str | Path | None = None,
    chunk_glossary_path: str | Path | None = None,
    glossary_path: str | Path | None = None,
    glossary_scope: str = "chunk",
    source_language: str = DEFAULT_SOURCE_LANGUAGE,
    target_language: str = DEFAULT_TARGET_LANGUAGE,
    book_title: str = "",
    translator: ChunkTranslator | None = None,
    settings: Any = None,
    settings_factory: Callable[[], Any] | None = None,
    model_identity: str = "",
    max_attempts: int = DEFAULT_TRANSLATION_ATTEMPTS,
    method: str = "auto",
    max_tokens: int | None = DEFAULT_TRANSLATION_MAX_TOKENS,
    context_chars: int = DEFAULT_TRANSLATION_CONTEXT_CHARS,
    save: bool = True,
    dry_run: bool = False,
    limit: int | None = None,
    chapter_files: Sequence[str] | None = None,
    include_matter: bool = False,
    resume: bool = False,
    write_manifest: bool = True,
    write_prompts: bool = True,
    progress: ProgressHook | None = None,
    chunk_start: ChunkStartHook | None = None,
    clock: Clock,
    context: RunContext,
    translator_factory: Callable[..., ChunkTranslator] = make_translator,
) -> TranslationPlan:
    scope = str(glossary_scope or "chunk").strip().lower()
    if scope not in TRANSLATION_GLOSSARY_SCOPES:
        raise ToolError(
            f"unknown glossary scope {glossary_scope!r}; pick one of "
            f"{', '.join(TRANSLATION_GLOSSARY_SCOPES)}"
        )
    method = str(method or "auto").strip().lower()
    if method not in TRANSLATION_METHODS:
        raise ToolError(
            f"unknown translation method {method!r}; pick one of "
            f"{', '.join(TRANSLATION_METHODS)}"
        )
    attempts = max(1, _int(max_attempts, DEFAULT_TRANSLATION_ATTEMPTS))
    context_size = max(0, _int(context_chars, DEFAULT_TRANSLATION_CONTEXT_CHARS))

    all_inputs, source_dir = discover_chunk_inputs(chunks_path)
    if chapter_files is not None:
        allowed = {str(name) for name in chapter_files}
        all_inputs = [
            chunk for chunk in all_inputs if chunk.chapter_filename in allowed
        ]
        if not all_inputs:
            raise ToolError(
                "no chunks belong to the requested chapter window "
                f"({', '.join(sorted(allowed)[:5])}…)"
            )
    # Front and back matter belong to the localization stage that runs before
    # this one. It translated their chunks with a different prompt and without
    # the dictionary, so translating them again here would overwrite that work.
    matter_skipped = 0
    if not include_matter:
        from ..matter.inspect import matter_chapter_names

        matter_names = matter_chapter_names(
            source_dir, source_dir.parent / CHAPTERS_DIRNAME
        )
        if matter_names:
            kept = [
                chunk
                for chunk in all_inputs
                if chunk.chapter_filename not in matter_names
            ]
            matter_skipped = len(all_inputs) - len(kept)
            all_inputs = kept
    # A chunk's identity for resume is its *actual* current text, not the hash a
    # manifest remembered: a chunk file edited on disk must be translated again.
    for chunk in all_inputs:
        chunk.sha256 = sha256_text(chunk.text)
    limited = limit is not None and 0 < _int(limit) < len(all_inputs)
    inputs = (
        all_inputs[: _int(limit)]
        if limit is not None and _int(limit) > 0
        else all_inputs
    )

    out_dir = (
        Path(output_dir).expanduser() if output_dir else translations_dir(source_dir)
    )
    prompt_dir = (
        Path(prompts_dir).expanduser()
        if prompts_dir
        else translation_prompts_dir(out_dir)
    )
    title = _book_title(source_dir, book_title)
    source_file = _source_file(source_dir)

    plan = TranslationPlan(
        inputs=inputs,
        source_dir=source_dir,
        out_dir=out_dir,
        prompt_dir=prompt_dir,
        title=title,
        source_file=source_file,
        scope=scope,
        method=method,
        attempts=attempts,
        max_tokens=max_tokens,
        context_size=context_size,
        source_language=str(source_language),
        target_language=str(target_language),
        save=save,
        dry_run=dry_run,
        limited=limited,
        include_matter=bool(include_matter),
        matter_skipped=matter_skipped,
        write_manifest=write_manifest,
        write_prompts=write_prompts,
        chunk_start=chunk_start,
        context=context,
        clock=clock,
        resolved_chunk_glossary=resolve_chunk_glossary_dir(
            chunk_glossary_path or "", source_dir
        ),
        resolved_glossary=Path(),
    )
    if not inputs:
        return plan

    plan.resolved_glossary = resolve_glossary_path(glossary_path, source_dir)
    if scope == "chunk":
        if not has_chunk_glossary_files(plan.resolved_chunk_glossary):
            raise ToolError(
                f"no per-chunk glossary in {plan.resolved_chunk_glossary} — run "
                "`chunk-glossary` first, or pass --glossary-scope book to send the "
                "whole glossary.json with every chunk"
            )
        plan.book_rows = []
    else:
        plan.book_rows = load_book_glossary_rows(plan.resolved_glossary)

    plan.system_prompt = build_system_prompt(
        plan.source_language, plan.target_language, plan.title
    )

    # -- resume: which chunks does this run not have to ask about? ---------- #
    plan.previous_manifest = load_translation_manifest(out_dir) if not dry_run else {}
    same_pair = bool(
        plan.previous_manifest
        and str(plan.previous_manifest.get("target_language") or "")
        == plan.target_language
        and str(plan.previous_manifest.get("source_language") or "")
        == plan.source_language
    )
    resume_enabled = bool(
        resume and not dry_run and plan.previous_manifest and same_pair
    )
    plan.previous_files = {
        str(entry.get("filename")): entry
        for entry in plan.previous_manifest.get("files") or []
        if isinstance(entry, dict) and entry.get("filename")
    }
    plan.carried_from = plan.previous_files if same_pair else {}
    plan.model_identity = _model_identity(
        settings, translator, plan.previous_manifest, label=model_identity
    )
    plan.resume_keys = {
        chunk.filename: _translation_resume_key(
            chunk,
            scope=scope,
            glossary_sha256=_glossary_fingerprint(
                chunk,
                scope=scope,
                chunk_glossary_dir=plan.resolved_chunk_glossary,
                glossary_path=plan.resolved_glossary,
            ),
            context_chars=context_size,
            method=method,
            model_identity=plan.model_identity,
            max_tokens=max_tokens,
            source_language=plan.source_language,
            target_language=plan.target_language,
        )
        for chunk in inputs
    }
    plan.reused = [
        _is_reusable(
            chunk,
            plan.previous_files.get(chunk.filename),
            out_dir,
            plan.resume_keys[chunk.filename],
        )
        if resume_enabled
        else False
        for chunk in inputs
    ]
    plan.resumed = bool(resume_enabled)
    plan.pending = [
        chunk for chunk, is_reused in zip(inputs, plan.reused, strict=True) if not is_reused
    ]

    if translator is None and plan.pending and not dry_run:
        if settings is None and settings_factory is not None:
            settings = settings_factory()
        if settings is None:
            raise ConfigError(
                "the translation stage needs a model: pass settings (provider/model) "
                "or a translator"
            )
        translator = translator_factory(
            settings, max_attempts=attempts, method=method, max_tokens=max_tokens
        )
    plan.translator = translator
    plan.translator_name = (
        getattr(translator, "name", "injected") if translator is not None else ""
    )
    # ``settings_factory`` is deliberately not resolved for a fully resumed run,
    # so the model identity may have been taken from the previous manifest. Now
    # that a translator exists, confirm it.
    actual_identity = _model_identity(
        settings, translator, plan.previous_manifest, label=plan.model_identity
    )
    if resume_enabled and actual_identity and actual_identity != plan.model_identity:
        plan.model_identity = actual_identity
        plan.resume_keys = {
            chunk.filename: _translation_resume_key(
                chunk,
                scope=scope,
                glossary_sha256=_glossary_fingerprint(
                    chunk,
                    scope=scope,
                    chunk_glossary_dir=plan.resolved_chunk_glossary,
                    glossary_path=plan.resolved_glossary,
                ),
                context_chars=context_size,
                method=method,
                model_identity=plan.model_identity,
                max_tokens=max_tokens,
                source_language=plan.source_language,
                target_language=plan.target_language,
            )
            for chunk in inputs
        }
        plan.reused = [
            _is_reusable(
                chunk,
                plan.previous_files.get(chunk.filename),
                out_dir,
                plan.resume_keys[chunk.filename],
            )
            for chunk in inputs
        ]
        plan.pending = [
            chunk
            for chunk, is_reused in zip(inputs, plan.reused, strict=True)
            if not is_reused
        ]

    plan.settings = settings
    plan.settings_payload = {
        "glossary_scope": scope,
        "max_attempts": attempts,
        "method": method,
        "max_tokens": int(max_tokens) if max_tokens else None,
        "context_chars": context_size,
        "model_identity": plan.model_identity,
        "save": bool(save),
        "write_prompts": bool(write_prompts),
        "limit": int(limit) if limit is not None else None,
        "chapter_files": list(chapter_files) if chapter_files is not None else None,
        "include_matter": bool(include_matter),
        "matter_skipped": matter_skipped,
        "resume": bool(resume),
        "json_only": bool(getattr(translator, "json_only", False)),
    }
    return plan
