"""Shared state for the stage-9 validation rule groups.

Each rule group mutates one draft: it appends the exact message the agent sees
and writes back any normalised value a later rule (or the response) needs. The
draft is the only mutable state; the rule modules themselves are stateless.

The order in which :mod:`book_splitter.postprocess.rules` calls the groups is
part of the contract: the agent reads ``issues`` in order, and the trace stores
the response verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...core.constants import DEFAULT_CHAPTER_TITLE_LEVEL
from ...documents.text import excerpt
from ..inspect import ChapterStructure
from ..titles import clean_title, normalise_overrides


class Validation:
    """The two lists every rule group appends to."""

    issues: list[str]
    warnings: list[str]

    def fail(self, message: str) -> None:
        self.issues.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


@dataclass
class BookPlanDraft(Validation):
    """One book plan being validated, plus the derived/normalised values."""

    book: dict[str, Any]
    structures: dict[str, dict[str, Any]]
    raw: dict[str, Any]
    naming_raw: dict[str, Any] = field(default_factory=dict)
    #: ``None`` when the plan did not submit a usable ``toc`` object.
    toc_raw: dict[str, Any] | None = None

    mode: str = "keep"
    label: str = ""
    include_title: bool = False
    separator: str = ": "
    level: int = DEFAULT_CHAPTER_TITLE_LEVEL
    overrides: dict[str, str] = field(default_factory=dict)
    evidence: str = ""
    title_source: str = ""

    toc_action: str = ""
    toc_heading: str = "Sadržaj"
    toc_style: str = "numbered"
    toc_include_back_matter: bool = False

    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_plan(
        cls,
        plan: dict[str, Any],
        *,
        book: dict[str, Any],
        structures: dict[str, dict[str, Any]],
    ) -> BookPlanDraft:
        naming = plan.get("naming")
        if not isinstance(naming, dict):
            naming = {}
        toc = plan.get("toc")
        return cls(
            book=book,
            structures=structures,
            raw=plan,
            naming_raw=naming,
            toc_raw=toc if isinstance(toc, dict) else None,
            mode=str(naming.get("mode") or "keep").strip().lower(),
            label=clean_title(str(naming.get("label") or "")),
            include_title=bool(naming.get("include_title")),
            separator=str(naming.get("separator") or ": "),
            overrides=normalise_overrides(naming.get("overrides")),
            evidence=excerpt(str(plan.get("title_evidence") or "").strip(), 500),
            title_source=str(plan.get("title_source") or "").strip().lower(),
        )

    def naming(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "label": self.label,
            "include_title": self.include_title,
            "separator": self.separator,
            "level": self.level,
            "overrides": self.overrides,
        }

    def toc(self) -> dict[str, Any]:
        return {
            "action": self.toc_action,
            "heading": self.toc_heading,
            "style": self.toc_style,
            "include_back_matter": self.toc_include_back_matter,
        }

    def normalized(self) -> dict[str, Any]:
        return {
            "naming": self.naming(),
            "toc": self.toc(),
            "title_evidence": self.evidence,
            "title_source": self.title_source,
        }


@dataclass
class ChapterPlanDraft(Validation):
    """One chapter plan being validated, plus the normalised title values."""

    structure: ChapterStructure
    lines: list[str]
    canonical_title: str = ""
    plan: dict[str, Any] = field(default_factory=dict)
    title: dict[str, Any] = field(default_factory=dict)
    title_line: int | None = None
    title_action: str = "keep"
    title_text: str = ""
    title_level: int = DEFAULT_CHAPTER_TITLE_LEVEL
    claimed: set[int] = field(default_factory=set)

    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_plan(
        cls,
        plan: dict[str, Any],
        *,
        structure: ChapterStructure,
        lines: list[str],
        canonical_title: str,
    ) -> ChapterPlanDraft:
        title = plan.get("title")
        if not isinstance(title, dict):
            title = {}
        return cls(
            structure=structure,
            lines=lines,
            canonical_title=canonical_title,
            plan=plan,
            title=title,
            title_line=structure.title_line,
            title_action=str(title.get("action") or "keep"),
            title_text=str(title.get("text") or clean_title(canonical_title)),
        )


__all__ = ["BookPlanDraft", "ChapterPlanDraft", "Validation"]
