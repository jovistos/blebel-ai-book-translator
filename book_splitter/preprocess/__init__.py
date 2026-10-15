"""Stage 2 core: header inspection, damage detection, plan validation, repair.

The package is deliberately layered, and each layer is usable on its own:

``inspect``
    one chapter's headers → evidence records + deterministic recommendations
``damage``
    one chapter's prose → *shape* detections (never a paragraph in the output)
``corpus``
    the whole book → sampled text and a counted damage profile
``transform``
    a plan → validation, application and provenance (no writes)
"""

from .corpus import (
    DAMAGE_PATTERNS,
    SAMPLE_HEADERS,
    SAMPLE_STRATEGIES,
    SAMPLE_STRIDE,
    corpus_digest,
    sample_text,
    scan_corpus_damage,
)
from .damage import (
    FIXES,
    PAGE_NUMBER_IN_WORD,
    SPACE_BEFORE_PUNCTUATION,
    SPLIT_WORD,
    Finding,
    damage_digest,
    repair_page_number_in_word,
    repair_space_before_punctuation,
    repair_split_word,
    scan_chapter_damage,
)
from .inspect import (
    DECISIONS,
    DEMOTE,
    DESTRUCTIVE_DECISIONS,
    KEEP,
    MERGE_DEMOTE,
    MERGE_WITH_NEXT,
    REMOVE,
    HeaderRecord,
    chapter_digest,
    chapter_header_lines,
    classify_header,
    inspect_chapter,
)
from .transform import (
    ApplyResult,
    LineEdit,
    apply_cleaning_plan,
    looks_like_page_artefact,
    text_of,
    validate_cleaning_plan,
    validate_text_fixes,
    verify_cleaning_plan,
    verify_text_fixes,
)

__all__ = [
    "ApplyResult",
    "DAMAGE_PATTERNS",
    "DECISIONS",
    "DEMOTE",
    "DESTRUCTIVE_DECISIONS",
    "FIXES",
    "Finding",
    "HeaderRecord",
    "KEEP",
    "LineEdit",
    "MERGE_DEMOTE",
    "MERGE_WITH_NEXT",
    "PAGE_NUMBER_IN_WORD",
    "REMOVE",
    "SAMPLE_HEADERS",
    "SAMPLE_STRATEGIES",
    "SAMPLE_STRIDE",
    "SPACE_BEFORE_PUNCTUATION",
    "SPLIT_WORD",
    "apply_cleaning_plan",
    "chapter_digest",
    "chapter_header_lines",
    "classify_header",
    "corpus_digest",
    "damage_digest",
    "inspect_chapter",
    "looks_like_page_artefact",
    "repair_page_number_in_word",
    "repair_space_before_punctuation",
    "repair_split_word",
    "sample_text",
    "scan_chapter_damage",
    "scan_corpus_damage",
    "text_of",
    "validate_cleaning_plan",
    "validate_text_fixes",
    "verify_cleaning_plan",
    "verify_text_fixes",
]
