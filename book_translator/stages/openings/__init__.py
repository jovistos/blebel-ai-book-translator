"""The final phase of stages 2 and 9: read every chapter opening, then clean it.

Both agent stages work one chapter at a time, and stage 2 is the only one that
looks at conversion damage while stage 9 is the only one that looks at titles.
Neither of them ever sees the *book*'s openings as a set, so a leftover that is
obvious in the finished file — a title followed by a stray copy of its number
(``# Poglavlje 2`` then ``2``), a duplicated title line, a printed page number
that survived next to the heading — can slip between them.

This package is the deterministic pass that closes that gap. It runs at the end
of both stages, over the folder the stage wrote:

``inspect``
    one chapter's first lines → a title candidate, the bounded opening records
    and provable issues. A chapter opening is a dozen clipped lines; the module
    never returns a paragraph.
``repair``
    the audit's safe removals, applied purely (no writes): a line that is
    nothing but a stray number or a copy of the title is dropped. Stage 2 keeps a
    blank slot so its line numbers stay valid; stage 9 collapses the blank.
``phase``
    the book-wide pass: audit every chapter, apply the safe removals, report what
    is clean, what was fixed and what needs a human (a title that is only a
    number, a chapter that opens on prose).

Public surface::

    inspect_opening(path)            one chapter's opening audit
    inspect_opening_lines(lines)     the same over lines already in memory
    scan_openings(paths)             a book-level report, no writes
    apply_opening_fixes(lines, ...)  the pure line rewrite
    audit_openings(folder, ...)      the phase both stage runners call
"""

from .inspect import (
    BARE_NUMBER_TITLE,
    DUPLICATE_TITLE,
    EMPTY_CHAPTER,
    FIX_REMOVE,
    LABEL_ONLY_TITLE,
    MISSING_TITLE,
    OPENING_FIXES,
    OPENING_ISSUES,
    STRAY_NUMBER_AFTER_TITLE,
    STRAY_NUMBER_BEFORE_TITLE,
    OpeningAudit,
    OpeningIssue,
    chapter_files_in,
    chapter_number_of_filename,
    inspect_opening,
    inspect_opening_lines,
    number_in_title,
    scan_openings,
)
from .phase import OpeningAuditResult, audit_openings
from .repair import OpeningRepair, apply_opening_fixes, normalise_fixes

__all__ = [
    "BARE_NUMBER_TITLE",
    "DUPLICATE_TITLE",
    "EMPTY_CHAPTER",
    "FIX_REMOVE",
    "LABEL_ONLY_TITLE",
    "MISSING_TITLE",
    "OPENING_FIXES",
    "OPENING_ISSUES",
    "OpeningAudit",
    "OpeningAuditResult",
    "OpeningIssue",
    "OpeningRepair",
    "STRAY_NUMBER_AFTER_TITLE",
    "STRAY_NUMBER_BEFORE_TITLE",
    "apply_opening_fixes",
    "audit_openings",
    "chapter_files_in",
    "chapter_number_of_filename",
    "inspect_opening",
    "inspect_opening_lines",
    "normalise_fixes",
    "number_in_title",
    "scan_openings",
]
