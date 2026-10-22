"""The final phase of stages 2 and 9: audit every chapter opening, then fix it.

The agents decide per chapter; this pass is deterministic and runs over the
folder they just wrote. It walks the *whole book* — every chapter's first lines,
nothing else — and asks one question: does the opening read like a chapter
opening? A title followed by a stray copy of its number (``# Poglavlje 2`` then
``2``), a duplicated title line and a leftover page number are the shapes that
actually reach the final book, so those are the ones it proves and removes.

Nothing else is touched: a title that is only a number, a title that is only a
label and a chapter that opens on prose are *reported*, because a title is a
naming decision (stage 9's book plan), never a line deletion.

The pass is deliberately stage-shaped:

* ``keep_blank_slot=True`` is stage 2: a removed line leaves an empty line, so
  line numbers in the stage's log, manifest and trace keep pointing at the same
  text.
* ``keep_blank_slot=False`` is stage 9: the line is removed and the blank around
  it collapsed, so the finished chapter reads cleanly.
* ``dry_run=True`` reports the same findings and writes nothing.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.artifacts import write_text
from ..core.errors import ToolError
from ..documents.io import load_lines
from .inspect import OpeningAudit, chapter_files_in, inspect_opening_lines
from .repair import OpeningRepair, apply_opening_fixes, normalise_fixes


@dataclass
class OpeningAuditResult:
    """Outcome of one book-wide opening audit (and repair, when not a dry run)."""

    ok: bool = True
    dry_run: bool = False
    skipped: str = ""
    folder: str = ""
    chapter_count: int = 0
    inspected_count: int = 0
    clean_count: int = 0
    issue_count: int = 0
    fixable_count: int = 0
    fixed_count: int = 0
    remaining_count: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    chapters: list[dict[str, Any]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def changed_count(self) -> int:
        """How many chapters had at least one safe removal."""
        return sum(1 for chapter in self.chapters if chapter.get("fixes"))

    def summary(self) -> dict[str, Any]:
        """The manifest-sized view: issues and fixes, no opening lines."""
        return {
            "ok": self.ok,
            "dry_run": self.dry_run,
            "skipped": self.skipped,
            "folder": self.folder,
            "chapter_count": self.chapter_count,
            "inspected_count": self.inspected_count,
            "clean_count": self.clean_count,
            "issue_count": self.issue_count,
            "fixable_count": self.fixable_count,
            "fixed_count": self.fixed_count,
            "remaining_count": self.remaining_count,
            "counts": dict(self.counts),
            "chapters": [
                {key: value for key, value in chapter.items() if key != "lines"}
                for chapter in self.chapters
            ],
            "missing": list(self.missing),
            "errors": list(self.errors),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _chapter_record(
    audit: OpeningAudit,
    *,
    fixes: list[dict[str, Any]],
    remaining: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "filename": audit.filename,
        "title": audit.title,
        "title_line": audit.title_line,
        "ok": not remaining,
        "issues": [issue.to_dict() for issue in audit.issues],
        "fixes": fixes,
        "remaining": remaining,
        "lines": [dict(record) for record in audit.lines],
    }


def _line_endings(path: Path) -> tuple[str, bool]:
    """The file's newline convention and whether it ended with one.

    ``load_lines`` opens in universal-newline mode, so the line list alone cannot
    tell CRLF from LF or whether the last line had a terminator. Rewriting a
    repaired chapter must not change either: the rest of the pipeline compares
    source hashes and reconstructs books byte for byte.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return "\n", True
    if b"\r\n" in data:
        newline = "\r\n"
    elif b"\r" in data:
        newline = "\r"
    else:
        newline = "\n"
    return newline, data.endswith((b"\n", b"\r"))


def _write_chapter(
    path: Path,
    repair: OpeningRepair,
    *,
    newline: str = "\n",
    terminal_newline: bool = True,
) -> None:
    text = newline.join(repair.lines)
    if terminal_newline:
        text += newline
    write_text(path, text)


def audit_openings(
    folder: str | Path,
    filenames: Iterable[str] | None = None,
    *,
    dry_run: bool = False,
    keep_blank_slot: bool = False,
    keep_clean: bool = False,
) -> OpeningAuditResult:
    """Audit every chapter opening in ``folder``, repairing the safe leftovers.

    ``filenames`` restricts the pass to the run's chapter window (in reading
    order). When omitted, every chapter file of the folder is audited, TOC and
    bookkeeping files excluded. ``keep_clean`` also records the chapters with no
    finding (the tools use it to show every opening; the stage reports do not).
    """
    path = Path(folder).expanduser()
    names = (
        [Path(str(name)).name for name in filenames]
        if filenames is not None
        else [item.name for item in chapter_files_in(path)]
    )
    result = OpeningAuditResult(
        dry_run=dry_run,
        folder=str(path),
        chapter_count=len(names),
    )
    for name in names:
        chapter_path = path / name
        if not chapter_path.is_file():
            result.missing.append(name)
            continue
        try:
            lines = load_lines(chapter_path)
            audit = inspect_opening_lines(lines, filename=name)
        except (OSError, ToolError) as exc:
            result.errors.append(f"{name}: {type(exc).__name__}: {exc}")
            result.ok = False
            continue
        result.inspected_count += 1
        issues_before = len(audit.issues)
        for issue in audit.issues:
            result.counts[issue.kind] = result.counts.get(issue.kind, 0) + 1
        result.issue_count += issues_before
        fixes = normalise_fixes(audit.fixes(), len(lines))
        result.fixable_count += len(fixes)
        remaining = [issue.to_dict() for issue in audit.unfixable]
        if dry_run:
            # Nothing was written, so every finding is still open.
            remaining = [issue.to_dict() for issue in audit.issues]
        elif fixes:
            repair = apply_opening_fixes(lines, fixes, keep_blank_slot=keep_blank_slot)
            try:
                newline, terminal_newline = _line_endings(chapter_path)
                _write_chapter(
                    chapter_path,
                    repair,
                    newline=newline,
                    terminal_newline=terminal_newline,
                )
            except OSError as exc:
                result.errors.append(f"{name}: {type(exc).__name__}: {exc}")
                result.ok = False
                remaining = [issue.to_dict() for issue in audit.unfixable]
            else:
                result.fixed_count += len(repair.applied)
                after = inspect_opening_lines(repair.lines, filename=name)
                remaining = [issue.to_dict() for issue in after.issues]
                if after.fixable:
                    result.errors.append(
                        f"{name}: {len(after.fixable)} opening issue(s) survived the repair"
                    )
                    result.ok = False
        if issues_before:
            result.remaining_count += len(remaining)
        else:
            result.clean_count += 1
        if audit.issues or fixes or keep_clean:
            result.chapters.append(
                _chapter_record(audit, fixes=fixes, remaining=remaining)
            )
    return result


__all__ = ["OpeningAuditResult", "audit_openings"]
