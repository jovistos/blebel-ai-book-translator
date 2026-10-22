#!/usr/bin/env python
"""The whole book pipeline in one command.

Compatibility wrapper: this is the same parser and the same driver as
``python -m book_splitter pipeline``. The implementation lives in
``book_splitter/pipeline``; the script only keeps the historical path and
``prog`` name working.

    python scripts/run_pipeline.py BOOK.md
    python scripts/run_pipeline.py --start-at dictionary
    python -m book_splitter pipeline --with-translate
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from book_splitter.pipeline.parser import main as pipeline_main  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return pipeline_main(argv, prog="run_pipeline.py")


if __name__ == "__main__":
    raise SystemExit(main())
