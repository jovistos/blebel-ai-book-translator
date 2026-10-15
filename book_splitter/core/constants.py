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
# Preprocessing stage (stage 2)
# --------------------------------------------------------------------------- #

#: A page number printed in the margin never has more digits than this.
MAX_PAGE_NUMBER_DIGITS = 4
#: A header longer than this is body text that lost its paragraph markup.
MAX_PLAUSIBLE_HEADING_CHARS = 90
#: Two consecutive headers shorter than this are one split heading.
MAX_MERGE_HEADING_CHARS = 70
#: Lines of context an inspector call may return around one line.
DEFAULT_INSPECT_LINES = 2
#: Hard cap on lines a single inspector call may return, whatever was asked.
MAX_INSPECT_LINES = 6
#: Hard cap on characters a single inspector call may return.
MAX_INSPECT_CHARS = 600
#: Inspector calls allowed per chapter (anti-body-reading budget).
MAX_INSPECT_CALLS_PER_CHAPTER = 8
#: Every chapter is scanned; at most this many sample lines per chapter.
DEFAULT_SAMPLE_LINES_PER_CHAPTER = 40
#: Global cap on sample-load characters across a whole book.
MAX_SAMPLE_CHARS = 12_000
#: Sample text from any single line is clipped to this length.
MAX_SAMPLE_LINE_CHARS = 160
#: A body line must be at least this long before its tail is "broken".
MIN_BROKEN_TAIL_LINE_CHARS = 200
#: Longest word the closing-join detector will attempt to stitch together.
#: Generous on purpose: the page number that split the word is still part of the
#: string when the shape is measured (``i božan`` + ``199stvena``).
MAX_STITCHED_WORD_CHARS = 24
#: Depth of the printed page-number run used to detect the printed number.
PAGE_NUMBER_RUN_WINDOW = 6
#: Hard cap on a line that may be treated as a figure/table caption at all.
MAX_CAPTION_LINE_CHARS = 300
#: A caption this short is a label; a longer line that opens with "Slika n" is a
#: sentence describing the picture and belongs to the text.
MAX_PLAUSIBLE_CAPTION_CHARS = 200

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

#: Stage 2 (preprocessing) writes next to stage 1, never over it.
CHAPTERS_DIRNAME = "chapters"
PREPROCESSED_DIRNAME = "preprocessed"
PREPROCESS_MANIFEST_FILENAME = "_preprocess_manifest.json"
PREPROCESS_TRACE_FILENAME = "_preprocess_trace.json"
