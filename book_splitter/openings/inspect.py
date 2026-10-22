"""What a chapter's opening says about the chapter (metadata only, never prose).

A chapter opening is the title and a handful of lines around it. Everything the
pipeline needs to know about it fits in a dozen clipped lines, so this module
never returns a paragraph:

* the title candidate (a Markdown heading, a chapter label, an ALL-CAPS line or
  a short standalone line),
* its line number, level and whether it already carries ``#`` markup,
* the chapter number read from the title or the filename,
* the bounded opening records, and
* every *issue* the shape of those lines proves: a title that is only a number,
  a stray number directly under the title (``# Poglavlje 2`` then ``2``), a
  duplicated title line, a chapter that opens on prose.

The issue vocabulary is fixed so the final phase and the agent tools agree::

    empty_chapter                the file holds no non-blank line
    missing_title                no title in the first lines (opens on prose)
    bare_number_title            the title is only a number (``## 2``, ``2``)
    label_only_title             the title is only ``Poglavlje``/``Chapter``
    stray_number_after_title     a line that is nothing but a number under the title
    stray_number_before_title    the same, above the title
    duplicate_title              a line that repeats the title verbatim

``stray_number_*`` and ``duplicate_title`` come with ``fix: "remove"``: the line
is provably junk next to the title, so the final phase may delete it. The others
are reported, not deleted — a title is a naming decision (stage 9's book plan),
never a line edit.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.constants import (
    FRONT_MATTER_FILENAME,
    MAX_OPENING_AUDIT_CHARS,
    MAX_OPENING_AUDIT_LINES,
    MAX_OPENING_AUDIT_SCAN_LINES,
    MAX_TITLE_LINE_CHARS,
)
from ..documents.chapters import is_toc_filename
from ..documents.io import load_lines
from ..documents.markdown import (
    header_of,
    is_all_caps,
    is_page_number_line,
    page_number_of,
)
from ..documents.text import excerpt, norm

#: Issue kinds (the tuple is the contract the reports and tools are built on).
EMPTY_CHAPTER = "empty_chapter"
MISSING_TITLE = "missing_title"
BARE_NUMBER_TITLE = "bare_number_title"
LABEL_ONLY_TITLE = "label_only_title"
STRAY_NUMBER_AFTER_TITLE = "stray_number_after_title"
STRAY_NUMBER_BEFORE_TITLE = "stray_number_before_title"
DUPLICATE_TITLE = "duplicate_title"

OPENING_ISSUES: tuple[str, ...] = (
    EMPTY_CHAPTER,
    MISSING_TITLE,
    BARE_NUMBER_TITLE,
    LABEL_ONLY_TITLE,
    STRAY_NUMBER_AFTER_TITLE,
    STRAY_NUMBER_BEFORE_TITLE,
    DUPLICATE_TITLE,
)
#: Issues whose line is provably junk: the whole line may be dropped.
FIX_REMOVE = "remove"
OPENING_FIXES: tuple[str, ...] = (FIX_REMOVE,)

#: Words that open a chapter/section label (multilingual, like the splitter).
_LABEL_WORDS = (
    "poglavlje",
    "glava",
    "giava",
    "chapter",
    "deo",
    "dio",
    "part",
    "uvod",
    "predgovor",
    "pogovor",
    "prolog",
    "epilog",
    "bele[sš]ke",
    "beleske",
    "notes",
    r"acknowledg\w*",
    "sadr[zž]aj",
    "contents",
)
_LABEL_RE = re.compile(
    r"^\s*(?:#+\s*)?(?:" + "|".join(_LABEL_WORDS) + r")\b",
    re.IGNORECASE,
)
#: ``Poglavlje`` alone: a label, but no number and no chapter name.
_LABEL_ONLY_RE = re.compile(
    r"^(?:poglavlje|glava|giava|chapter|deo|dio|part)\s*[.:\-\u2013\u2014]?$",
    re.IGNORECASE,
)
#: ``Poglavlje 4`` / ``CHAPTER IV`` / ``GLAVA DRUGA``: label plus its number.
_LABELLED_NUMBER_RE = re.compile(
    r"^\s*(?:#+\s*)?(?:poglavlje|glava|giava|chapter|deo|dio|part)\s+"
    r"([0-9]+|[IVXLCDM]+)\b",
    re.IGNORECASE,
)
#: ``2. Poglavlje`` / ``4. Title``: a numbered heading without a label word.
_NUMBERED_TITLE_RE = re.compile(r"^(?:#+\s*)?(\d{1,3})\s*[.)]\s+\S")
#: ``02-chapter-2.md`` / ``12-chapter-12-slug.md``.
_FILENAME_CHAPTER_RE = re.compile(r"^\d+[-_ ]*chapter[-_ ]*(\d+)", re.IGNORECASE)
#: ``05-chapter-2.md`` already matched above; ``07-uvod.md`` falls back to its prefix.
_FILENAME_NUMBER_RE = re.compile(r"^(\d{1,3})[-_ ]")
#: A plain line ending like a sentence is prose, never a title. Question and
#: exclamation marks are allowed (``Šta vam je na umu?`` is a real title).
_SENTENCE_END_RE = re.compile(r"[.\u2026]\s*$")
_ROMAN_RE = re.compile(r"^[IVXLCDM]+$", re.IGNORECASE)
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _roman_value(text: str) -> int | None:
    """``"IV"`` -> ``4``; ``None`` when the token is not a plausible numeral."""
    if not _ROMAN_RE.match(text):
        return None
    total = 0
    previous = 0
    for char in reversed(text.upper()):
        value = _ROMAN_VALUES[char]
        if value < previous:
            total -= value
        else:
            total += value
            previous = value
    return total if 0 < total < 4000 else None


def chapter_number_of_filename(filename: str) -> int | None:
    """The chapter number a stage-1/stage-9 filename encodes, or ``None``."""
    name = Path(str(filename or "")).name
    match = _FILENAME_CHAPTER_RE.search(name)
    if match:
        return int(match.group(1))
    match = _FILENAME_NUMBER_RE.match(name)
    return int(match.group(1)) if match else None


def is_front_matter_filename(filename: str) -> bool:
    """True for the front-matter file (``00-front-matter.md``, ``00-title.md``)."""
    name = Path(str(filename or "")).name.lower()
    return (
        name == FRONT_MATTER_FILENAME.lower()
        or "front-matter" in name
        or name.startswith("00-")
    )


def number_in_title(text: str) -> int | None:
    """The chapter number a title carries (``Poglavlje 4``, ``2. Poglavlje``, ``2``)."""
    value = (text or "").strip()
    if not value:
        return None
    match = _LABELLED_NUMBER_RE.match(value)
    if match:
        token = match.group(1)
        return int(token) if token.isdigit() else _roman_value(token)
    match = _NUMBERED_TITLE_RE.match(value)
    if match:
        return int(match.group(1))
    return page_number_of(value)


@dataclass
class OpeningIssue:
    """One provable problem in a chapter's opening, tied to an exact line."""

    kind: str
    detail: str
    line_number: int | None = None
    text: str = ""
    fix: str = ""

    @property
    def fixable(self) -> bool:
        return bool(self.fix)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OpeningAudit:
    """One chapter's opening: its title, bounded lines and issues."""

    filename: str = ""
    title: str = ""
    title_line: int | None = None
    title_is_heading: bool = False
    title_source: str = "missing"
    chapter_number: int | None = None
    title_number: int | None = None
    lines: list[dict[str, Any]] = field(default_factory=list)
    issues: list[OpeningIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def fixable(self) -> list[OpeningIssue]:
        return [issue for issue in self.issues if issue.fixable]

    @property
    def unfixable(self) -> list[OpeningIssue]:
        return [issue for issue in self.issues if not issue.fixable]

    def fixes(self) -> list[dict[str, Any]]:
        """The safe line removals this opening proves, in file order."""
        return [
            {
                "line_number": issue.line_number,
                "action": issue.fix,
                "kind": issue.kind,
                "text": issue.text,
            }
            for issue in self.issues
            if issue.fixable and issue.line_number is not None
        ]

    def summary(self) -> dict[str, Any]:
        """Compact record for a manifest or a book-level report."""
        return {
            "filename": self.filename,
            "title": self.title,
            "title_line": self.title_line,
            "ok": self.ok,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "title_is_heading": self.title_is_heading,
            "title_source": self.title_source,
            "chapter_number": self.chapter_number,
            "title_number": self.title_number,
            "lines": [dict(record) for record in self.lines],
        }


def _opening_records(lines: list[str]) -> list[dict[str, Any]]:
    """The first non-blank lines, clipped and labelled — never a paragraph."""
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(lines[:MAX_OPENING_AUDIT_SCAN_LINES], start=1):
        clean = raw.strip()
        if not clean:
            continue
        head = header_of(raw)
        body = head[1].strip() if head else clean
        number = page_number_of(body)
        records.append(
            {
                "line_number": index,
                "text": excerpt(body, MAX_OPENING_AUDIT_CHARS),
                "is_heading": head is not None,
                "is_number": number is not None,
                "number": number,
            }
        )
        if len(records) >= MAX_OPENING_AUDIT_LINES:
            break
    return records


def _is_plain_title(text: str, *, next_blank: bool) -> bool:
    """Conservative test for a title line that carries no ``#`` markup."""
    clean = (text or "").strip()
    if not clean or len(clean) > MAX_TITLE_LINE_CHARS:
        return False
    if is_page_number_line(clean):
        return False
    if clean.endswith(","):
        return False
    if _LABEL_RE.match(clean):
        return True
    if is_all_caps(clean):
        return True
    return bool(next_blank and not _SENTENCE_END_RE.search(clean))


def _next_blank(records: list[dict[str, Any]], index: int) -> bool:
    if index + 1 >= len(records):
        return True
    return int(records[index + 1]["line_number"]) > int(records[index]["line_number"]) + 1


def _find_title(
    records: list[dict[str, Any]], chapter_number: int | None
) -> int | None:
    """Index of the title record, or ``None`` when the chapter opens on prose."""
    for index, record in enumerate(records):
        if record["is_heading"]:
            # The first heading in the opening is the title, even when its text
            # is only a number: ``## 2`` is an untitled chapter whose translated
            # heading lost its label, not a page number at the top of the file.
            return index
        if record["is_number"]:
            # A plain number is only the title when the file has nothing better,
            # it matches the filename's chapter number, and prose follows: that
            # is ``2`` used as a heading, not a page number above the heading.
            follow = records[index + 1] if index + 1 < len(records) else None
            next_is_prose = bool(
                follow
                and not follow["is_number"]
                and not follow["is_heading"]
                and not _LABEL_RE.match(str(follow["text"]))
            )
            if (
                index == 0
                and chapter_number is not None
                and record["number"] == chapter_number
                and next_is_prose
            ):
                return index
            continue
        if _is_plain_title(record["text"], next_blank=_next_blank(records, index)):
            return index
        # The first line that is neither a number, a label nor a title is prose:
        # there is no title before it, and a later heading is a section heading.
        return None
    return None


def _is_duplicate(record: dict[str, Any], title_norm: str) -> bool:
    if not title_norm:
        return False
    return bool(record["text"]) and norm(record["text"]) == title_norm


def _junk_issues(
    records: list[dict[str, Any]],
    title_index: int,
    title_norm: str,
    title_number: int | None,
) -> list[OpeningIssue]:
    """Stray numbers / repeated title lines in the run directly around the title."""
    issues: list[OpeningIssue] = []

    def inspect(record: dict[str, Any], *, after: bool) -> OpeningIssue | None:
        if _is_duplicate(record, title_norm):
            return OpeningIssue(
                kind=DUPLICATE_TITLE,
                detail=(
                    "this line repeats the chapter title"
                    + (" below" if after else " above")
                    + " it; the title is already the heading"
                ),
                line_number=int(record["line_number"]),
                text=str(record["text"]),
                fix=FIX_REMOVE,
            )
        if not record["is_number"]:
            return None
        # A numbered *heading* is only junk when it repeats the chapter's own
        # number. ``## 1`` under ``# Poglavlje 2`` is a numbered section, not a
        # page artefact, and is left alone.
        if record["is_heading"] and record["number"] != title_number:
            return None
        return OpeningIssue(
            kind=(STRAY_NUMBER_AFTER_TITLE if after else STRAY_NUMBER_BEFORE_TITLE),
            detail=(
                "this line is nothing but a number "
                + ("below" if after else "above")
                + " the title — a duplicate of the chapter number or a printed "
                "page number that survived"
            ),
            line_number=int(record["line_number"]),
            text=str(record["text"]),
            fix=FIX_REMOVE,
        )

    for record in records[title_index + 1 :]:
        issue = inspect(record, after=True)
        if issue is None:
            break
        issues.append(issue)
    for record in reversed(records[:title_index]):
        issue = inspect(record, after=False)
        if issue is None:
            break
        issues.append(issue)
    # A heading lower in the opening that repeats the chapter title verbatim is
    # the same duplicate, just not adjacent: the title was set twice (an older
    # stage-9 run could leave its title above the prose and a second copy where
    # it used to be). Only a heading qualifies; a repeated prose line is the
    # chapter's own text.
    seen = {issue.line_number for issue in issues}
    for record in records[title_index + 1 :]:
        number = int(record["line_number"])
        if number in seen or not record["is_heading"]:
            continue
        if _is_duplicate(record, title_norm):
            issues.append(
                OpeningIssue(
                    kind=DUPLICATE_TITLE,
                    detail=(
                        "this heading repeats the chapter title lower in the "
                        "opening; the title is already the heading at the top"
                    ),
                    line_number=number,
                    text=str(record["text"]),
                    fix=FIX_REMOVE,
                )
            )
    issues.sort(key=lambda item: item.line_number or 0)
    return issues


def inspect_opening_lines(
    lines: Iterable[str],
    *,
    filename: str = "",
    chapter_number: int | None = None,
) -> OpeningAudit:
    """Read one chapter's opening from its lines and list every provable issue."""
    plain = [line.rstrip("\n") for line in lines]
    records = _opening_records(plain)
    filename_number = chapter_number_of_filename(filename) if filename else None
    expected = chapter_number if chapter_number is not None else filename_number
    audit = OpeningAudit(
        filename=Path(str(filename or "")).name,
        lines=records,
        chapter_number=expected,
    )
    if not records:
        audit.issues.append(
            OpeningIssue(
                kind=EMPTY_CHAPTER,
                detail="the chapter file has no non-blank line",
            )
        )
        return audit

    title_index = _find_title(records, expected)
    if title_index is None:
        if is_front_matter_filename(audit.filename):
            # A front-matter page (blurb, title page) legitimately opens on prose;
            # a chapter opening on prose is the finding this audit exists for.
            return audit
        audit.issues.append(
            OpeningIssue(
                kind=MISSING_TITLE,
                detail=(
                    "no title in the first "
                    f"{MAX_OPENING_AUDIT_LINES} non-blank lines; the chapter opens on "
                    "prose"
                ),
                line_number=int(records[0]["line_number"]),
                text=str(records[0]["text"]),
            )
        )
        return audit

    record = records[title_index]
    title = str(record["text"])
    audit.title = title
    audit.title_line = int(record["line_number"])
    audit.title_is_heading = bool(record["is_heading"])
    audit.title_source = (
        "heading"
        if record["is_heading"]
        else ("number" if record["is_number"] else "plain")
    )
    if not title:
        audit.issues.append(
            OpeningIssue(
                kind=MISSING_TITLE,
                detail="the first heading at the top of the chapter has no text",
                line_number=audit.title_line,
            )
        )
        return audit
    audit.title_number = number_in_title(title)
    if audit.title_number is None:
        audit.title_number = expected

    title_norm = norm(title)
    if record["is_number"]:
        audit.issues.append(
            OpeningIssue(
                kind=BARE_NUMBER_TITLE,
                detail=(
                    "the chapter title is only the number "
                    f"{record['number']}; the label that makes it a chapter heading "
                    "is missing"
                ),
                line_number=audit.title_line,
                text=title,
            )
        )
    elif _LABEL_ONLY_RE.match(title):
        audit.issues.append(
            OpeningIssue(
                kind=LABEL_ONLY_TITLE,
                detail=(
                    "the chapter title is only a label ("
                    f"{title!r}) with no number and no chapter name"
                ),
                line_number=audit.title_line,
                text=title,
            )
        )

    audit.issues.extend(
        _junk_issues(records, title_index, title_norm, audit.title_number)
    )
    return audit


def inspect_opening(chapter_path: str | Path, *, chapter_number: int | None = None) -> OpeningAudit:
    """Read one chapter file's opening (bounded) and audit it."""
    path = Path(chapter_path).expanduser()
    return inspect_opening_lines(
        load_lines(path),
        filename=path.name,
        chapter_number=chapter_number,
    )


def chapter_files_in(folder: str | Path) -> list[Path]:
    """The chapter ``*.md`` files of a folder in reading order (TOC and ``_`` skipped)."""
    path = Path(folder).expanduser()
    if path.is_file():
        path = path.parent
    files = [
        item
        for item in path.glob("*.md")
        if item.is_file() and not item.name.startswith("_") and not is_toc_filename(item.name)
    ]
    return sorted(files, key=_natural_key)


def _natural_key(path: Path) -> tuple[int, str]:
    """Sort ``2-x`` before ``10-x`` while keeping non-numeric names stable."""
    match = re.match(r"^(\d+)", path.name)
    return (int(match.group(1)) if match else 10**9, path.name.lower())


def scan_openings(
    paths: Iterable[str | Path],
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Audit a whole book's openings without writing anything.

    Returns the per-chapter audits (bounded opening lines included) plus the
    book-level counts, so a tool and the final phase report the same shape.
    """
    audits: list[OpeningAudit] = []
    for item in paths:
        if limit is not None and len(audits) >= int(limit):
            break
        audits.append(inspect_opening(item))
    counts: dict[str, int] = {}
    for audit in audits:
        for issue in audit.issues:
            counts[issue.kind] = counts.get(issue.kind, 0) + 1
    return {
        "chapter_count": len(audits),
        "ok_count": sum(1 for audit in audits if audit.ok),
        "issue_count": sum(len(audit.issues) for audit in audits),
        "fixable_count": sum(len(audit.fixable) for audit in audits),
        "counts": counts,
        "chapters": [audit.to_dict() for audit in audits],
    }


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
    "OpeningIssue",
    "STRAY_NUMBER_AFTER_TITLE",
    "STRAY_NUMBER_BEFORE_TITLE",
    "chapter_files_in",
    "chapter_number_of_filename",
    "inspect_opening",
    "inspect_opening_lines",
    "is_front_matter_filename",
    "number_in_title",
    "scan_openings",
]
