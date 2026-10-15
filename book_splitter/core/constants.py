"""Tunable constants shared by the analysis, slicing and agent layers.

Everything the pipeline treats as a hard guardrail lives here, so the reasoning
behind a limit can be read in one place instead of being scattered as magic
numbers across modules.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Context guardrails
# --------------------------------------------------------------------------- #

#: Hard cap on any single tool result handed back to the model.
MAX_TOOL_CHARS = 24_000
#: Hard cap on characters returned by the front-matter reader.
MAX_FRONT_MATTER_CHARS = 20_000
#: Hard cap on lines returned by the front-matter reader.
MAX_FRONT_MATTER_LINES = 1_200

# --------------------------------------------------------------------------- #
# Structural heuristics
# --------------------------------------------------------------------------- #

#: A line longer than this is prose, never a title.
MAX_TITLE_LINE_CHARS = 120
#: Excerpt length for a matched/validated line (titles only, never paragraphs).
EXCERPT_CHARS = 90
#: Lines scanned from the front of the book when hunting for a printed TOC.
TOC_SCAN_LINES = 400
#: Maximum length of the printed TOC block itself.
TOC_MAX_BLOCK_LINES = 120
#: A gap larger than this between split points suggests a missing chapter.
MAX_GAP_LINES = 400
#: A gap smaller than this between split points suggests a duplicate split.
MIN_GAP_LINES = 2

# --------------------------------------------------------------------------- #
# Tool defaults (surfaced in the tool schemas the model sees)
# --------------------------------------------------------------------------- #

DEFAULT_MAX_HEADERS = 300
DEFAULT_MAX_CANDIDATES = 400
DEFAULT_FRONT_MATTER_LINES = 200
DEFAULT_TOC_MATCH_SCORE = 0.72
DEFAULT_MAX_MATCHES_PER_TITLE = 3
DEFAULT_MIN_CHAPTERS = 3

# --------------------------------------------------------------------------- #
# Agent loop
# --------------------------------------------------------------------------- #

#: Maximum LLM turns before the graph gives up on the model.
MAX_FEEDBACK_LOOPS = 6
#: Headers included verbatim in the scout report (the rest are summarised).
MAX_HEADERS_IN_REPORT = 180
#: Default LangGraph recursion limit for a split run.
DEFAULT_RECURSION_LIMIT = 60

# --------------------------------------------------------------------------- #
# Output names
# --------------------------------------------------------------------------- #

FRONT_MATTER_FILENAME = "00-front-matter.md"
MANIFEST_FILENAME = "_manifest.json"
TRACE_FILENAME = "_agent_trace.json"
