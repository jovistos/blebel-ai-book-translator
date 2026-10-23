"""Rule group: where the title evidence came from.

The published-chapter-list lookup is mandatory in the graph, so a plan without
``title_evidence`` is rejected there. The ``title_source`` label is advisory: an
unknown value is reset with a warning rather than failing the plan.
"""

from __future__ import annotations

from .base import BookPlanDraft


def check_evidence(draft: BookPlanDraft, *, require_web_search: bool) -> None:
    if draft.title_source and draft.title_source not in {"web", "local", "mixed"}:
        draft.warn(
            "title_source should be 'web', 'local' or 'mixed'; it was ignored"
        )
        draft.title_source = ""
    if require_web_search and not draft.evidence:
        draft.fail(
            "title_evidence is required: record the URL(s) you read from the published "
            "chapter list, or the failure/no-result outcome of the mandatory web search"
        )


__all__ = ["check_evidence"]
