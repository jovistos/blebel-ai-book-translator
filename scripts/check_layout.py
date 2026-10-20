#!/usr/bin/env python
"""Verify the document layout and the installed dependencies.

Run it after cloning, or when a split writes to an unexpected place::

    python scripts/check_layout.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from book_splitter import paths  # noqa: E402


def _check(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def main() -> int:
    """Verify the local layout and the importable optional pieces."""
    problems: list[str] = []

    for required in ("langgraph", "langchain_core", "langchain_openai", "pydantic"):
        if not _check(required):
            problems.append(f"missing required dependency: {required}")

    # Stages 3, 5 and 6 are optional: report them, but never fail the check because of them.
    optional = {
        "docling": "stage 0 EPUB/PDF to Markdown (TXT/Markdown need no dependency)",
        "pypdfium2": "stage 0 PDF text-layer preflight (installed with docling)",
        "classla": "stage 3/6 NER for sr/hr/sl/bg",
        "stanza": "stage 3/6 NER for en",
        "lingua": "stage 3/6 language detection (lingua-language-detector)",
        "chonkie": "stage 5 chunking",
        "tiktoken": "stage 5 exact token counts (falls back to an estimate)",
        "rapidfuzz": "stage 6 fuzzy glossary retrieval",
    }
    absent = [f"{name} ({why})" for name, why in optional.items() if not _check(name)]

    input_dir = paths.default_input_dir()
    output_dir = paths.default_output_dir()
    print(f"repo root   : {paths.repo_root()}")
    print(f"input dir   : {input_dir}{'' if input_dir.is_dir() else '  (missing)'}")
    print(f"output dir  : {output_dir}{'' if output_dir.is_dir() else '  (missing)'}")

    if not input_dir.is_dir():
        problems.append(f"input folder does not exist: {input_dir}")
    if not output_dir.is_dir():
        problems.append(f"output folder does not exist: {output_dir}")

    books = sorted(p.name for p in input_dir.glob("*") if p.is_file() and p.name != ".gitkeep")
    print(f"source books: {', '.join(books) if books else '(none)'}")
    print(
        "optional deps: "
        + ("all present" if not absent else "not installed: " + ", ".join(absent))
    )
    if not books:
        problems.append(f"no source document in {input_dir}")

    for problem in problems:
        print(f"  ! {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
