"""Re-apply stage 2's plans without calling the model again.

A preprocess run records, per chapter, the operations and text fixes the agent
submitted (`_preprocess_trace.json`). When a *repair rule* changes — not the
agent's decision — re-running the model would spend tokens to rediscover the same
plan. This script replays the recorded plan through the same validator, applier
and verifier the graph uses, so the output is byte-identical to what a fresh run
would write, and it says so.

    python scripts/replay_preprocess.py                 # every chapter in the trace
    python scripts/replay_preprocess.py --only 04- 07-  # by filename prefix
    python scripts/replay_preprocess.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from book_splitter.core.constants import (  # noqa: E402
    CHAPTERS_DIRNAME,
    PREPROCESS_TRACE_FILENAME,
)
from book_splitter.documents.chapters import (  # noqa: E402
    discover_chapters,
    preprocessed_dir,
)
from book_splitter.preprocess.transform import (  # noqa: E402
    apply_cleaning_plan,
    text_of,
    validate_cleaning_plan,
    validate_text_fixes,
    verify_cleaning_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chapters", default=None, help="stage-1 chapters folder")
    parser.add_argument("--output", default=None, help="stage-2 output folder")
    parser.add_argument("--only", nargs="*", default=None, help="filename prefixes to replay")
    parser.add_argument("--dry-run", action="store_true", help="validate, write nothing")
    args = parser.parse_args()

    base = Path(args.chapters).expanduser() if args.chapters else _default_chapters()
    out_dir = Path(args.output).expanduser() if args.output else preprocessed_dir(base)
    trace_path = out_dir / PREPROCESS_TRACE_FILENAME
    if not trace_path.is_file():
        print(f"no stage-2 trace at {trace_path}; run `preprocess` once first")
        return 1

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    chapters = {c.filename: c for c in discover_chapters(base)}
    written = failed = 0
    for record in trace.get("chapters", []):
        filename = record.get("filename", "")
        if args.only and not any(filename.startswith(p) for p in args.only):
            continue
        chapter = chapters.get(filename)
        if chapter is None:
            print(f"  {filename}: not in {base}")
            failed += 1
            continue
        operations = record.get("operations") or []
        text_fixes = record.get("text_fixes") or []

        validation = validate_cleaning_plan(chapter.path, operations)
        fixes, issues, _ = validate_text_fixes(chapter.path, text_fixes)
        problems = list(validation.get("issues") or []) + list(issues)
        if problems:
            print(f"  {filename}: plan no longer validates -> {problems[:2]}")
            failed += 1
            continue

        if args.dry_run:
            result = apply_cleaning_plan(chapter.path, validation["operations"], fixes)
            verification = verify_cleaning_plan(chapter.path, validation["operations"], fixes)
            print(
                f"  {filename}: {len(operations)} ops, {len(fixes)} fixes -> "
                f"{len(result.lines)} lines, verify={verification['ok']}"
            )
        else:
            result = apply_cleaning_plan(chapter.path, validation["operations"], fixes)
            (out_dir / filename).write_text(text_of(result), encoding="utf-8")
            verification = verify_cleaning_plan(chapter.path, validation["operations"], fixes)
            print(
                f"  {filename}: replayed {len(operations)} ops, {len(fixes)} fixes -> "
                f"verify={verification['ok']}"
            )
        if not verification["ok"]:
            print(f"      {verification['problems'][:3]}")
            failed += 1
        else:
            written += 1

    print(f"\n{'validated' if args.dry_run else 'rewrote'} {written} chapter(s), {failed} failed")
    return 1 if failed else 0


def _default_chapters() -> Path:
    output = ROOT / "data" / "output"
    folders = [
        p / CHAPTERS_DIRNAME
        for p in sorted(output.iterdir())
        if (p / CHAPTERS_DIRNAME).is_dir()
    ]
    if not folders:
        raise SystemExit("no chapters folder in data/output; run `split` first")
    return folders[0]


if __name__ == "__main__":
    raise SystemExit(main())
