"""The book-wide dictionary being built, and the result of one merge.

The dictionary is the stage's whole point: one locked rendering per entity, kept
in insertion order, with the first rendering always winning. The mutable book
and the outcome of folding one chapter into it live here; the serialisable rows
it stores (:class:`~book_translator.glossary.records.GlossaryTerm`) and the run
records stay in :mod:`book_translator.glossary.records`.

Keeping the book out of the runner is what lets a test build one directly, and
keeping it out of ``records`` keeps "a record" and "the live dictionary" from
being the same module.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .models import GlossaryEntry
from .records import GlossaryTerm


@dataclass
class MergeOutcome:
    """What one chapter's answer did to the dictionary."""

    added: list[GlossaryTerm] = field(default_factory=list)
    repeats: list[str] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.added and not self.repeats and not self.conflicts


class GlossaryBook:
    """The dictionary, in insertion order, with the consistency rules applied.

    The first rendering of a term wins, always. That is the promise the prompt
    makes to the translator, so a later chapter that re-states a locked term
    differently is recorded as a **conflict** and the dictionary is left alone —
    the model gets another chance to conform in the next chapter, but the text it
    already translated can never disagree with itself.
    """

    def __init__(self, entries: Iterable[GlossaryTerm] = ()) -> None:
        self._entries: dict[str, GlossaryTerm] = {}
        self.conflicts: list[dict[str, Any]] = []
        for entry in entries:
            if entry.key:
                self._entries[entry.key] = entry

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)

    def __iter__(self):
        return iter(self._entries.values())

    def terms(self) -> list[GlossaryTerm]:
        return list(self._entries.values())

    def lock(self, entry: GlossaryTerm) -> None:
        """Install an already-locked entry, as ``--resume`` does from disk."""
        if entry.key:
            self._entries[entry.key] = entry

    def to_list(self) -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in self._entries.values()]

    def categories(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self._entries.values():
            counts[entry.category] = counts.get(entry.category, 0) + 1
        return dict(sorted(counts.items()))

    def merge(
        self,
        entries: Iterable[GlossaryEntry],
        *,
        chapter_index: int,
        chapter_filename: str,
    ) -> MergeOutcome:
        """Fold one chapter's answer into the dictionary.

        Runs in three cases, in this order: a term nobody has seen is added with
        its provenance; a term already locked in and returned with the *same*
        rendering is a repeat (deduplicated); a term already locked in and
        returned with a *different* rendering is a conflict, the existing entry
        stands, and both renderings are recorded so the run can report where the
        model tried to drift.
        """
        outcome = MergeOutcome()
        for entry in entries:
            key = entry.key
            if not key:
                continue
            existing = self._entries.get(key)
            if existing is None:
                term = GlossaryTerm.from_entry(
                    entry,
                    chapter_index=chapter_index,
                    chapter_filename=chapter_filename,
                )
                self._entries[key] = term
                outcome.added.append(term)
                continue
            if _same_rendering(existing.translation, entry.translation):
                if chapter_filename not in existing.repeats:
                    existing.repeats.append(chapter_filename)
                outcome.repeats.append(existing.term)
                continue
            conflict = {
                "term": existing.term,
                "chapter_index": chapter_index,
                "chapter_filename": chapter_filename,
                "locked": existing.translation,
                "proposed": entry.translation,
            }
            self.conflicts.append(conflict)
            outcome.conflicts.append(conflict)
        return outcome


def _same_rendering(left: str, right: str) -> bool:
    return " ".join(str(left or "").split()).casefold() == " ".join(str(right or "").split()).casefold()

__all__ = ["GlossaryBook", "MergeOutcome"]
