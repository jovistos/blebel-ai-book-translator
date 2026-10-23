"""Rule group: per-chapter title overrides.

An override is the escape hatch for a name the naming policy cannot produce, so
every override must target a chapter that exists, be non-empty and stay within
the title limit.
"""

from __future__ import annotations

from book_translator.core.constants import MAX_POSTPROCESS_TITLE_CHARS

from .base import BookPlanDraft


def check_overrides(draft: BookPlanDraft) -> None:
    for filename, title in draft.overrides.items():
        if filename not in draft.structures:
            draft.fail(f"naming override targets an unknown chapter: {filename!r}")
            continue
        if not title:
            draft.fail(f"naming override for {filename!r} is empty")
        elif len(title) > MAX_POSTPROCESS_TITLE_CHARS:
            draft.fail(f"naming override for {filename!r} is too long")


__all__ = ["check_overrides"]
