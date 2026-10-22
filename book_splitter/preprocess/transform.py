"""Compatibility facade for stage 2's plan / apply / verify modules.

The implementation moved out of this 948-line module in Phase 6:

``plan``
    the vocabulary's predicates and the two validators (no writes);
``apply``
    pure line rewriting and the per-line provenance map;
``verify``
    the content proof that only declared lines changed.

Nothing lives here any more. The facade stays for one phase so existing imports
(``scripts/replay_preprocess.py``, notebooks, out-of-tree callers) keep working;
new code should import from the module that owns the behaviour.
"""

from __future__ import annotations

from .apply import ApplyResult, LineEdit, apply_cleaning_plan, text_of
from .plan import (
    looks_like_caption,
    looks_like_page_artefact,
    next_content_line,
    strip_leading_page_number,
    validate_cleaning_plan,
    validate_text_fixes,
)
from .verify import verify_cleaning_plan, verify_text_fixes

__all__ = [
    "ApplyResult",
    "LineEdit",
    "apply_cleaning_plan",
    "looks_like_caption",
    "looks_like_page_artefact",
    "next_content_line",
    "strip_leading_page_number",
    "text_of",
    "validate_cleaning_plan",
    "validate_text_fixes",
    "verify_cleaning_plan",
    "verify_text_fixes",
]
