#!/usr/bin/env python
"""Interactive wizard for the book translation pipeline.

    python scripts/run_wizard.py
    python -m book_translator wizard
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from book_translator.cli.commands.wizard import main as wizard_main  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return wizard_main(argv, prog="run_wizard.py")


if __name__ == "__main__":
    raise SystemExit(main())
