"""Rule group: invariants of the *derived* canonical titles.

The derivation lives in :mod:`book_translator.stages.postprocess.titles`; this rule only
checks its output. It runs after every other book rule and stays silent when any
of them failed, so the agent is not told about titles computed from an invalid
policy.
"""

from __future__ import annotations

from book_translator.core.constants import MAX_POSTPROCESS_TITLE_CHARS
from book_translator.documents.text import norm

from .base import BookPlanDraft


def check_titles(draft: BookPlanDraft, titles: dict[str, str]) -> None:
    if draft.issues:
        return
    seen: dict[str, str] = {}
    for filename, title in titles.items():
        if not title:
            draft.fail(f"chapter {filename!r} would have an empty title")
            continue
        if len(title) > MAX_POSTPROCESS_TITLE_CHARS:
            draft.fail(f"chapter {filename!r} title is too long")
        key = norm(title)
        if (
            key
            and key in seen
            and draft.structures.get(filename, {}).get("matter") == "body"
        ):
            draft.warn(
                f"chapters {seen[key]!r} and {filename!r} would share the title {title!r}"
            )
        seen.setdefault(key, filename)


__all__ = ["check_titles"]
