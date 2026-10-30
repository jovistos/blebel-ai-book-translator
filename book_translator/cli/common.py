"""Helpers shared by the command modules: paths, model settings and rendering."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from book_translator.pipeline.paths import StagePaths

from book_translator.core.constants import (
    CHAPTERS_DIRNAME,
    DEFAULT_KEEP_ENTITY_TYPES,
    POSTPROCESSED_CHAPTERS_DIRNAME,
    PREPROCESSED_DIRNAME,
    TRANSLATED_CHAPTERS_DIRNAME,
)
from book_translator.documents.chapters import (
    chapters_dir,
    has_chapter_files,
)
from book_translator.paths import (
    resolve_book_path,
    resolve_output_dir,
)
from book_translator.stages.conversion import is_markdown_source, run_conversion


def _print_json(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _resolve_book(args: argparse.Namespace) -> Path:
    return resolve_book_path(getattr(args, "book", None), input_dir=args.in_dir)


def _resolve_chapters_dir(
    book: str | None,
    in_dir: str | None,
    out: str | None,
    output_root: str | Path | None = None,
) -> Path:
    """Where stage 1's chapters are.

    Resolution order: an explicit ``--out``, a ``--book`` / positional argument
    that is already a folder, then the default output folder of the named book
    (or of the only book in ``data/input/``).
    """
    if out:
        return Path(out).expanduser()
    if book:
        candidate = Path(book).expanduser()
        if candidate.is_dir():
            return candidate
    source = resolve_book_path(book, input_dir=in_dir)
    return resolve_output_dir(source, None, output_root=output_root)


def _resolve_text_dir(
    book: str | None,
    in_dir: str | None,
    out: str | None,
    output_root: str | Path | None = None,
    text_in: str | Path | None = None,
    chapter_limit: int | None = None,
) -> Path:
    """The chapter folder the text stages read: cleaned by default.

    Stage 2 exists to repair conversion damage before the text stages see it, so
    the command-line default is the same as the pipeline's: ``02-preprocessed/``
    when a completed stage-2 manifest covers the chapters this run reads,
    ``01-chapters/`` otherwise. An explicit ``--out`` or a book argument that is
    already a folder still wins, which is how a caller asks for the raw text on
    purpose.
    """
    if text_in:
        return chapters_dir(Path(text_in).expanduser())
    if out:
        return Path(out).expanduser()
    if book:
        candidate = Path(book).expanduser()
        if candidate.is_dir():
            return candidate
    raw = chapters_dir(_resolve_chapters_dir(book, in_dir, None, output_root))
    return _covered_cleaned_dir(raw, chapter_limit, in_dir, output_root) or raw


def _covered_cleaned_dir(
    raw: Path,
    chapter_limit: int | None,
    in_dir: str | None,
    output_root: str | Path | None,
) -> Path | None:
    """``02-preprocessed/`` only when it covers every chapter in the window.

    The old check here asked only whether a manifest existed; a manifest from a
    ``--first-chapters`` preprocess then made every later command read the
    partial folder. The shared resolver applies the pipeline's coverage and
    hash rules instead.
    """
    from ..pipeline.paths import StagePaths
    from ..pipeline.sources import cleaned_covers_window

    paths = StagePaths.from_base(
        raw,
        input_dir=Path(in_dir).expanduser() if in_dir else None,
        output_root=Path(output_root).expanduser() if output_root else None,
    )
    return cleaned_covers_window(paths, chapter_limit)


def _command_paths(args: argparse.Namespace) -> StagePaths:
    """The shared stage layout for one stage command.

    Imported lazily so importing the CLI does not pull the pipeline package in
    until a command actually runs.
    """
    from ..pipeline.paths import StagePaths

    return StagePaths.for_command(args)


def _command_text_dir(
    args: argparse.Namespace, paths: StagePaths, chapter_limit: int | None = None
) -> Path:
    """The chapter folder a text stage reads, through the shared resolver.

    ``--out`` and ``--text-in`` stay explicit choices; otherwise
    :func:`book_translator.pipeline.sources.resolve_text_source` applies the same
    coverage/hash rules as the pipeline, so a partial ``02-preprocessed/`` can no
    longer silently shrink the book.
    """
    if getattr(args, "out", None) and not getattr(args, "text_in", None):
        return paths.chapters_dir
    from ..pipeline.sources import resolve_text_source

    return resolve_text_source(paths, args, chapter_limit=chapter_limit).path


def _ensure_markdown_source(
    source: Path, out: str | Path | None = None, output_root: str | Path | None = None
) -> Path:
    """Run stage 0 when the source is not Markdown and return what stage 1 reads.

    A Markdown source is returned untouched: stage 0 only exists to make the
    other formats readable by the line-based stages.
    """
    if is_markdown_source(source):
        return source
    result = run_conversion(source, out, output_root=output_root)
    for line in result.log:
        print(f"  · {line}", file=sys.stderr)
    if result.markdown_path:
        print(f"converted: {result.source} -> {result.markdown_path}", file=sys.stderr)
    return Path(result.markdown_path)


def _model_settings(
    args: argparse.Namespace,
    stage: str,
    *,
    translation: bool = False,
):
    """Resolve one command's model, honouring ``--llm-config`` and ``--llm``.

    The settings module is imported at call time so the offline tests keep
    patching ``book_translator.core.settings.resolve_settings`` (or
    ``resolve_translation_settings``) at the exact seam this command uses.
    """
    from ..core.settings import (
        parse_llm_overrides,
        resolve_settings,
        resolve_translation_settings,
        stage_overrides,
    )

    overrides = stage_overrides(
        parse_llm_overrides(getattr(args, "llm", None), default_stage=stage),
        stage,
    )
    config_path = getattr(args, "llm_config", None)
    require_key = not getattr(args, "dry_run", False)
    if translation:
        return resolve_translation_settings(
            provider=getattr(args, "translator", None),
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            temperature=args.temperature,
            config_path=config_path,
            overrides=overrides,
            require_key=require_key,
        )
    return resolve_settings(
        stage=stage,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        config_path=config_path,
        overrides=overrides,
        require_key=require_key,
    )


def _window_label(args: argparse.Namespace, limit: int | None) -> str | None:
    """Human description of the chapter window, without exposing the file count."""
    first = getattr(args, "first_chapters", None)
    if first is not None:
        return f"first {first} body chapter(s)"
    if limit is not None:
        return f"first {limit} chapter(s)"
    return None


def _parse_entity_types(value: str | None) -> tuple[str, ...]:
    """``"PER, loc ,ORG"`` -> ``("PER", "LOC", "ORG")``."""
    if not value:
        return DEFAULT_KEEP_ENTITY_TYPES
    wanted = tuple(part.strip().upper() for part in str(value).split(",") if part.strip())
    return wanted or DEFAULT_KEEP_ENTITY_TYPES


def _print_openings_line(openings: dict | None, *, detail: bool = True) -> None:
    """One line (and the chapters that still need a look) for the opening audit."""
    data = openings or {}
    if not data or data.get("skipped"):
        return
    dry = bool(data.get("dry_run"))
    removed = int(data.get("fixed_count") or 0)
    planned = sum(len(chapter.get("fixes") or []) for chapter in data.get("chapters") or [])
    print(
        f"openings        : {int(data.get('inspected_count') or 0)} checked · "
        f"{int(data.get('clean_count') or 0)} clean · "
        + (
            f"{planned} line(s) would be removed · "
            if dry
            else f"{removed} line(s) removed · "
        )
        + f"{int(data.get('remaining_count') or 0)} need attention"
    )
    if not detail:
        return
    verb = "would remove" if dry else "removed"
    for chapter in data.get("chapters") or []:
        fixes = chapter.get("fixes") or []
        if fixes:
            text = ", ".join(
                f"line {fix.get('line_number')} {str(fix.get('text') or '')!r}"
                for fix in fixes
            )
            print(f"  · {verb} {chapter.get('filename')}: {text}")
    for chapter in data.get("chapters") or []:
        remaining = chapter.get("remaining") or []
        for issue in remaining[:4]:
            print(
                f"  · {chapter.get('filename')} line {issue.get('line_number')}: "
                f"{issue.get('kind')} — {issue.get('detail')}"
            )


_CHAPTER_STAGE_FOLDERS = (
    POSTPROCESSED_CHAPTERS_DIRNAME,
    TRANSLATED_CHAPTERS_DIRNAME,
    PREPROCESSED_DIRNAME,
    CHAPTERS_DIRNAME,
)


def _resolve_openings_dir(args: argparse.Namespace) -> Path:
    """The folder to audit: an explicit one, or the book's most advanced stage."""
    from ..pipeline.paths import StagePaths
    from ..pipeline.sources import resolve_openings_input

    if getattr(args, "chapters_in", None):
        return resolve_openings_input(args)
    chapters = _resolve_chapters_dir(
        args.book, args.in_dir, None, getattr(args, "output_root", None)
    )
    resolved = resolve_openings_input(args, StagePaths.from_base(chapters))
    return resolved if has_chapter_files(resolved) else chapters
