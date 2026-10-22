"""Shared argparse parents for the stage commands.

Every stage command needs the same handful of options, and the old
``build_parser`` re-declared them one by one — which is how ``inspect`` ended up
without ``--json`` and ``scan`` without ``--quiet``. These factories return a
fresh ``ArgumentParser(add_help=False)`` so a command can be assembled with
``parents=[...]`` and the shared flags are declared exactly once.

The legacy parser in :mod:`book_splitter.cli.legacy` is being migrated to these
parents command by command; new commands (and the pipeline command) use them
directly.
"""

from __future__ import annotations

import argparse

from ..paths import default_input_dir, default_output_dir


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(add_help=False)


def book_parent(*, required: bool = False) -> argparse.ArgumentParser:
    """The positional ``book`` plus ``--in-dir``/``--output-root``."""
    parser = _parser()
    parser.add_argument(
        "book",
        nargs=None if required else "?",
        help="source document — a bare filename or a path "
        "(default: the only book found in data/input/)",
    )
    parser.add_argument(
        "--in-dir",
        default=None,
        help=f"folder holding the source documents (default: {default_input_dir()})",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=f"root of the generated artifacts (default: {default_output_dir()})",
    )
    return parser


def layout_parent() -> argparse.ArgumentParser:
    """``--output-root`` for commands that do not take a positional book."""
    parser = _parser()
    parser.add_argument(
        "--output-root",
        default=None,
        help=f"root of the generated artifacts (default: {default_output_dir()})",
    )
    return parser


def output_parent() -> argparse.ArgumentParser:
    """``--json``/``--quiet`` with the shared contract."""
    parser = _parser()
    parser.add_argument("--json", action="store_true", help="machine-readable result")
    parser.add_argument("--quiet", "-q", action="store_true")
    return parser


def dry_run_parent() -> argparse.ArgumentParser:
    parser = _parser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate every input, write no artifact and make no paid model call",
    )
    return parser


def resume_parent(dest: str = "skip_if_present") -> argparse.ArgumentParser:
    parser = _parser()
    parser.add_argument(
        "--skip-if-present",
        "--resume",
        dest=dest,
        action="store_true",
        help="reuse the work already on disk",
    )
    return parser


def window_parent() -> argparse.ArgumentParser:
    parser = _parser()
    parser.add_argument(
        "--first-chapters",
        "--chapters",
        dest="first_chapters",
        type=int,
        default=None,
        help="process only the first N body chapters",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process only the first N units (chapters or chunks)",
    )
    return parser


def manifest_parent() -> argparse.ArgumentParser:
    parser = _parser()
    parser.add_argument(
        "--no-manifest", action="store_true", help="do not write the manifest"
    )
    return parser


def prompts_parent() -> argparse.ArgumentParser:
    parser = _parser()
    parser.add_argument(
        "--no-prompts", action="store_true", help="do not keep the prompt records"
    )
    return parser


def text_parent() -> argparse.ArgumentParser:
    parser = _parser()
    parser.add_argument(
        "--text-in",
        default=None,
        help="chapter text folder to read (default: 02-preprocessed/ when it covers "
        "the window, else 01-chapters/)",
    )
    parser.add_argument(
        "--chunk-in",
        default=None,
        help="stage-5 input folder, overriding --text-in",
    )
    return parser


def llm_parent() -> argparse.ArgumentParser:
    parser = _parser()
    parser.add_argument(
        "--llm-config",
        default=None,
        metavar="FILE",
        help="per-stage model config (default: ./book_splitter.toml)",
    )
    parser.add_argument(
        "--llm",
        action="append",
        default=[],
        metavar="[STAGE.]FIELD=VALUE",
        help="override one model field for this stage, repeatable",
    )
    return parser


def model_parent() -> argparse.ArgumentParser:
    parser = _parser()
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--recursion-limit", type=int, default=60)
    # ``--json``/``--quiet`` are deliberately included: every model-backed
    # command needs them, and the old helper added them here.
    parser.add_argument("--json", action="store_true", help="machine-readable result")
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--llm-config", default=None, metavar="FILE")
    parser.add_argument("--llm", action="append", default=[], metavar="[STAGE.]FIELD=VALUE")
    return parser


def language_parent() -> argparse.ArgumentParser:
    parser = _parser()
    parser.add_argument("--language", default="auto")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--keep-types", default="PER,LOC,ORG,MISC")
    parser.add_argument("--drop-temporal-only", action="store_true")
    return parser


def epub_metadata_parent() -> argparse.ArgumentParser:
    parser = _parser()
    parser.add_argument("--title", default=None)
    parser.add_argument("--author", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--identifier", default=None)
    parser.add_argument("--description", default=None)
    parser.add_argument("--publisher", default=None)
    parser.add_argument("--cover", default=None)
    return parser


def path_override_parent() -> argparse.ArgumentParser:
    """The per-stage ``--*-out`` flags shared by the pipeline and the commands."""
    parser = _parser()
    for flag, dest in (
        ("--out", "out"),
        ("--preprocessed-out", "preprocessed_out"),
        ("--reduced-out", "reduced_out"),
        ("--unique-out", "unique_out"),
        ("--glossary-out", "glossary_out"),
        ("--chunks-out", "chunks_out"),
        ("--chunk-glossary-out", "chunk_glossary_out"),
        ("--translations-out", "translations_out"),
        ("--translation-prompts-out", "translation_prompts_out"),
        ("--matter-out", "matter_out"),
        ("--matter-prompts-out", "matter_prompts_out"),
        ("--translated-chapters-out", "translated_chapters_out"),
        ("--postprocess-out", "postprocess_out"),
        ("--book-out", "book_out"),
        ("--epub-out", "epub_out"),
    ):
        parser.add_argument(flag, dest=dest, default=None)
    return parser


def path_input_parent() -> argparse.ArgumentParser:
    """The explicit input-folder flags shared by the stage commands."""
    parser = _parser()
    for flag, dest in (
        ("--chapters-in", "chapters_in"),
        ("--chunks-in", "chunks_in"),
        ("--glossary-in", "glossary_in"),
        ("--chunk-glossary-in", "chunk_glossary_in"),
        ("--translations-in", "translations_in"),
        ("--translated-chapters-in", "translated_chapters_in"),
        ("--postprocessed-in", "postprocessed_in"),
    ):
        parser.add_argument(flag, dest=dest, default=None)
    return parser


__all__ = [
    "book_parent",
    "dry_run_parent",
    "epub_metadata_parent",
    "language_parent",
    "layout_parent",
    "llm_parent",
    "manifest_parent",
    "model_parent",
    "output_parent",
    "path_input_parent",
    "path_override_parent",
    "prompts_parent",
    "resume_parent",
    "text_parent",
    "window_parent",
]
