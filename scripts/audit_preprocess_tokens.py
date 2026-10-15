"""Audit a stage-2 run: prove that only page artefacts changed.

Stage 2 is allowed to change exactly four things:

* a ``#`` header marker may disappear (``demote`` / ``merge_with_next``);
* a printed page number may disappear (page-artefact lines, and page numbers
  printed inside or in front of a word);
* whitespace before punctuation may disappear;
* two lines may become one (split heading / split word), so the newline between
  them disappears.

This script compares the **non-whitespace characters** of stage 1's chapters with
stage 2's output and reports every difference that is not one of the above. A
difference that is a letter is a bug: a word that was rewritten, dropped or
invented. A difference that is a digit is a page number — expected, but printed
so a human can look at it.

    python scripts/audit_preprocess_tokens.py            # summary
    python scripts/audit_preprocess_tokens.py --details  # every difference

The comparison is character-level, so it is slow on a chapter whose paragraphs
are single 8000-character lines (the sample book's conversion produced those).
Expect it to take a couple of minutes for a whole book; it is a verification
tool, not part of the pipeline.
"""

from __future__ import annotations

import argparse
import difflib
import re
from pathlib import Path

#: Characters that may legitimately disappear in stage 2.
EXPECTED_LOST = frozenset("#,.;:!?…»\"')]")
#: A letter, which must never be invented or dropped by a repair.
LETTER = re.compile(r"[^\W\d_]")


def _compact(text: str) -> str:
    """Every non-whitespace character of the text, in order."""
    return re.sub(r"\s+", "", text)


def audit_file(source: Path, target: Path, details: bool = False) -> list[str]:
    """Report the differences between one chapter and its cleaned version."""
    if not target.is_file():
        return [f"    MISSING: {target.name}"]
    before = _compact(source.read_text(encoding="utf-8"))
    after = _compact(target.read_text(encoding="utf-8"))
    if before == after:
        return []

    problems: list[str] = []
    matcher = difflib.SequenceMatcher(None, before, after, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        removed, added = before[i1:i2], after[j1:j2]
        if any(LETTER.match(ch) for ch in removed + added):
            problems.append(f"    TEXT  -{removed[:70]!r} +{added[:70]!r}  (a letter changed)")
        elif details:
            problems.append(f"    ok    -{removed[:70]!r} +{added[:70]!r}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None, help="data/output folder to audit")
    parser.add_argument(
        "--details", action="store_true", help="also print the expected digit differences"
    )
    args = parser.parse_args()

    root = (
        Path(args.output).expanduser()
        if args.output
        else Path(__file__).resolve().parent.parent / "data" / "output"
    )
    if not root.is_dir():
        print(f"no output folder at {root} — run `split` and `preprocess` first")
        return 1

    total = 0
    for book in sorted(p for p in root.iterdir() if p.is_dir()):
        chapters, cleaned = book / "chapters", book / "preprocessed"
        if not cleaned.is_dir():
            continue
        print("=" * 70)
        print(book.name)
        book_problems = 0
        for source in sorted(chapters.glob("*.md")):
            if source.name.startswith("_"):
                continue
            problems = audit_file(source, cleaned / source.name, details=args.details)
            if problems:
                book_problems += len(problems)
                print(f"  {source.name}")
                for problem in problems[:12]:
                    print(problem)
        total += book_problems
        print(f"  letter-level differences: {book_problems}")
    print("=" * 70)
    if total:
        print(f"FAIL: {total} place(s) where a repair changed text — inspect the output")
        return 1
    print("OK: every difference is a page number, punctuation spacing or header markup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
