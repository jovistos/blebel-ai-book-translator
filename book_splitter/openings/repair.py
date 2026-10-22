"""Applying an opening audit's safe removals — pure, no writes.

The audit marks only a line that is *nothing but* a stray number or a copy of the
chapter title. This module deletes exactly those lines and nothing else, in two
flavours the two stages need:

``keep_blank_slot=False`` (stage 9)
    the line is removed and one adjacent blank line is collapsed, so the written
    chapter reads ``# Poglavlje 2`` / blank / prose.

``keep_blank_slot=True`` (stage 2)
    the line becomes empty and the file keeps its line count, exactly like the
    stage-2 cleaner: a line number in a log, a manifest or a trace still points
    at the same text.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OpeningRepair:
    """The repaired lines plus what was removed from which line."""

    lines: list[str] = field(default_factory=list)
    applied: list[dict[str, Any]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.applied)

    def summary(self) -> dict[str, Any]:
        return {"changed": self.changed, "applied": [dict(item) for item in self.applied]}


def normalise_fixes(
    fixes: Iterable[dict[str, Any]] | None,
    line_count: int,
) -> list[dict[str, Any]]:
    """Keep the valid ``remove`` instructions, one per line, in file order."""
    out: dict[int, dict[str, Any]] = {}
    for raw in fixes or []:
        if not isinstance(raw, dict):
            continue
        if str(raw.get("action") or "").strip().lower() != "remove":
            continue
        number_value = raw.get("line_number")
        if number_value is None:
            continue
        try:
            number = int(number_value)
        except (TypeError, ValueError):
            continue
        if number < 1 or number > line_count or number in out:
            continue
        out[number] = {
            "line_number": number,
            "action": "remove",
            "kind": str(raw.get("kind") or ""),
            "text": str(raw.get("text") or ""),
        }
    return [out[number] for number in sorted(out)]


def _drop_line(working: list[str], index: int) -> None:
    """Delete ``working[index]`` and the double blank that deletion can leave."""
    del working[index]
    if index == 0:
        if len(working) > 1 and not working[0].strip():
            del working[0]
        return
    if index < len(working) and not working[index].strip() and not working[index - 1].strip():
        del working[index]


def apply_opening_fixes(
    lines: Iterable[str],
    fixes: Iterable[dict[str, Any]] | None,
    *,
    keep_blank_slot: bool = False,
) -> OpeningRepair:
    """Delete the audited junk lines, keeping a blank slot when asked to."""
    plain = [line.rstrip("\n") for line in lines]
    result = OpeningRepair(lines=list(plain))
    targets = normalise_fixes(fixes, len(plain))
    for fix in reversed(targets):
        index = int(fix["line_number"]) - 1
        text = plain[index]
        result.applied.append(
            {
                "line_number": int(fix["line_number"]),
                "kind": fix["kind"],
                "text": text.strip(),
                "action": "remove",
            }
        )
        if keep_blank_slot:
            result.lines[index] = ""
        else:
            _drop_line(result.lines, index)
    result.applied.reverse()
    return result


__all__ = ["OpeningRepair", "apply_opening_fixes", "normalise_fixes"]
