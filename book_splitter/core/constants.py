"""Tunable constants shared by the analysis, slicing and agent layers.

Everything the pipeline treats as a hard guardrail lives here, so the reasoning
behind a limit can be read in one place instead of being scattered as magic
numbers across modules.
"""

from __future__ import annotations

#: The stage-folder names have one definition in :mod:`book_splitter.paths`;
#: importing them here keeps ``core.constants`` as the one stop for stage code
#: without maintaining a second copy of the strings.
from ..paths import (  # noqa: F401 - re-exported for stage modules
    CHAPTERS_DIRNAME,
    CHUNKS_DIRNAME,
    MARKDOWN_DIRNAME,
    REDUCED_DIRNAME,
    UNIQUE_DIRNAME,
)

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
DEFAULT_MAX_PATTERN_MATCHES = 200
DEFAULT_FRONT_MATTER_LINES = 200
DEFAULT_TOC_MATCH_SCORE = 0.72
DEFAULT_MAX_MATCHES_PER_TITLE = 3
DEFAULT_MIN_CHAPTERS = 3

# --------------------------------------------------------------------------- #
# Back matter at the tail of the book (`inspect_book_tail`)
#
# The tail reader is the second tool allowed to return prose (after
# `extract_front_matter`), so it is bounded on three sides at once: how many
# lines the agent may ask for, how long each line may be, and how many
# characters the whole window may hold. The agent starts at the short chunk and
# escalates only when the window cuts through the block.
# --------------------------------------------------------------------------- #

#: Window the agent is told to try first; escalate from here when needed.
DEFAULT_TAIL_CHUNK_LINES = 40
#: Asking for fewer lines than this is not worth a call.
MIN_TAIL_CHUNK_LINES = 10
#: Hard cap on the tail window, whatever the agent asks for.
MAX_TAIL_CHUNK_LINES = 300
#: Longest line returned by the tail reader (bio paragraphs are clipped).
MAX_TAIL_LINE_CHARS = 160
#: Hard cap on the characters of one tail window.
MAX_TAIL_WINDOW_CHARS = 12_000
#: Hard cap on the serialised tail report, so it can never be truncated into
#: invalid JSON by the global per-tool limit. The window is trimmed from the
#: front (oldest lines first) until the report fits, which keeps the true tail.
MAX_TAIL_REPORT_CHARS = 20_000
#: A repeated line must be at least this long and this many words to count as a
#: repeated title (``Chapter Ten: …``), which is what a re-printed TOC looks like.
MIN_REPEAT_TITLE_CHARS = 16
MIN_REPEAT_TITLE_WORDS = 3
#: Evidence lines listed in one tail report.
MAX_BACK_MATTER_CANDIDATES = 40

#: A trailing block longer than this is not plausible back matter.
MAX_BACK_MATTER_LINES = 1_000
#: …nor is one longer than this share of the whole book.
MAX_BACK_MATTER_FRACTION = 0.25
#: Floor for that share, so a very short book still allows a real back matter.
MIN_BACK_MATTER_LINES = 100

#: Default filename of the back-matter file the agent may propose.
BACK_MATTER_FILENAME = "back-matter.md"

# --------------------------------------------------------------------------- #
# `find_lines_matching`: the agent writes the pattern, so every dimension of the
# search is bounded — pattern length, how much of a line is examined, how many
# matches come back, and how long the scan may run.
# --------------------------------------------------------------------------- #

MAX_PATTERN_CHARS = 200
MAX_PATTERN_LINE_CHARS = 2_000
MAX_PATTERN_MATCHES = DEFAULT_MAX_PATTERN_MATCHES
PATTERN_TIME_BUDGET_SECONDS = 2.0

# --------------------------------------------------------------------------- #
# Web lookup (`web_search` / `fetch_web_page`)
#
# A book whose own table of contents was destroyed by the PDF conversion can
# still be identified from the outside: the published chapter list exists
# online. These tools are the only ones that talk to the network, so every
# dimension is capped — how many results come back, how long a request may run,
# how many bytes are downloaded, and how much text is shown to the model.
# --------------------------------------------------------------------------- #

#: Results the agent gets when it does not ask for a specific count.
DEFAULT_WEB_RESULTS = 6
#: Hard cap on results per search, whatever the agent asks for.
MAX_WEB_RESULTS = 12
#: Hard cap on characters returned by one page fetch.
MAX_WEB_FETCH_CHARS = 12_000
#: Page text the agent gets when it does not ask for a specific length.
DEFAULT_WEB_FETCH_CHARS = 6_000
#: Floor on a page fetch: fewer characters than this cannot hold a contents list.
MIN_WEB_FETCH_CHARS = 200
#: Bytes downloaded before the connection is cut (a page, not a video).
MAX_WEB_BYTES = 2_000_000
#: Wall clock for one search or one page fetch.
WEB_SEARCH_TIMEOUT_SECONDS = 12.0
#: Longest url the agent may ask the fetcher to open.
WEB_FETCH_MAX_URL_CHARS = 2_000
#: Sent with every request; some engines refuse the default urllib agent.
WEB_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)

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
# Dictionary stage (stage 3)
#
# Stage 3 reduces each chapter to the sentences that carry named entities, so a
# later LLM pass can mine the recurring nouns (characters, places, inventions)
# that a translation has to keep consistent.
#
# The NER gate is deliberately traditional (CLASSLA/stanza): it only *filters*.
# It is not asked to understand a fictional concept, because it cannot.
# --------------------------------------------------------------------------- #

#: A sentence with fewer words than this is not dictionary evidence.
DEFAULT_MIN_SENTENCE_WORDS = 4
#: Sentence case is not evidence of a proper noun, so a sentence whose only
#: capital letter is its first word is dropped by default.
DEFAULT_REQUIRE_UPPERCASE = True
#: The NER types that count as evidence. CLASSLA emits exactly these four.
DEFAULT_KEEP_ENTITY_TYPES: tuple[str, ...] = ("PER", "LOC", "ORG", "MISC")
#: Languages CLASSLA has its own models for. Everything else goes to stanza.
CLASSLA_LANGUAGES: frozenset[str] = frozenset({"sl", "hr", "sr", "bg", "mk"})
#: …but Macedonian ships without a NER model, so it cannot gate anything.
CLASSLA_NER_LANGUAGES: frozenset[str] = frozenset({"sl", "hr", "sr", "bg"})
#: Languages routed to stanza instead (stanza covers ~70, this pins the ones the
#: pipeline actually needs; add the ISO code here to route another one).
STANZA_LANGUAGES: frozenset[str] = frozenset({"en"})
#: Physical GPU the NER backends use unless ``--gpu`` / ``--cpu`` overrides it.
#: The machine running this pipeline keeps GPU 1 free for the taggers.
DEFAULT_GPU_INDEX = 1
#: Chapters sampled for language detection, and the characters read from each.
#: 5 x 4 000 characters is still only 20 KB, and it matters twice over: lingua
#: gets enough text to vote confidently, and the ekavian/ijekavian tie-breaker
#: (see ``dictionary/language.py``) gets enough forms to count. At 3 x 1 200 the
#: sample of a real Serbian novel held 4 ekavian forms — not enough to overrule
#: the ``hr`` vote lingua gives it.
LANGUAGE_SAMPLE_CHAPTERS = 5
LANGUAGE_SAMPLE_CHARS = 4_000
#: Fixed seed: the same book must always be routed to the same backend.
LANGUAGE_SAMPLE_SEED = 13
#: Languages offered to the detector (the union the two backends can serve).
DETECTABLE_LANGUAGES: tuple[str, ...] = ("sr", "hr", "sl", "bg", "mk", "en")

#: Stage 3 writes next to stages 1 and 2, never over them. Folder names carry
#: the stage number so a book folder sorts in pipeline order.
REDUCED_MANIFEST_FILENAME = "_reduced_manifest.json"
REDUCED_TRACE_FILENAME = "_reduced_trace.json"

# --------------------------------------------------------------------------- #
# Glossary stage (stage 4)
#
# Stage 4 is the LLM pass the funnel was built for: it reads the first-mention
# sentences of every body chapter, in reading order, and asks a model to lock in
# one rendering per entity, in the target language, before the book itself is
# translated. Every chapter after the first sees the dictionary built so far, so
# the renderings it invents are compatible with the ones already fixed.
# --------------------------------------------------------------------------- #

#: Default file names of the stage-4 artifacts, written next to ``03-reduced-unique/``.
GLOSSARY_DIRNAME = "04-glossary"
GLOSSARY_FILENAME = "glossary.json"
GLOSSARY_MARKDOWN_FILENAME = "glossary.md"
GLOSSARY_MANIFEST_FILENAME = "_glossary_manifest.json"
GLOSSARY_TRACE_FILENAME = "_glossary_trace.json"

#: The language pair the glossary is built for. A book is translated *from* the
#: language it is written in *into* the language the reader will hold.
DEFAULT_SOURCE_LANGUAGE = "en"
DEFAULT_TARGET_LANGUAGE = "sr"

#: Extraction attempts per chapter before the chapter is reported as failed. Two
#: extra attempts cost far less than a chapter whose names never make it into the
#: dictionary, and a retry is what turns "the model wrote prose around the JSON"
#: into a usable answer.
DEFAULT_GLOSSARY_ATTEMPTS = 3
#: Characters of one chapter's reduced file sent in a single model call. A chapter
#: longer than this is split on line boundaries and the dictionary is carried
#: across the parts, so a 100-page chapter still fits a small context window.
DEFAULT_GLOSSARY_CHUNK_CHARS = 24_000


# --------------------------------------------------------------------------- #
# Chunking stage (stage 5)
#
# Stage 5 cuts every chapter into translation-sized chunks. It is the only stage
# that looks at the body as *text* rather than as lines, so its contract is
# different from stage 1's: a chunk is an exact slice of the chapter it came from,
# cuts only at paragraph/sentence/line/word boundaries (in that order of
# preference), and carries enough metadata (chapter, global order, character
# offsets, hashes) to put the chapter back together by concatenation.
# --------------------------------------------------------------------------- #

#: Default chunk length, in tokens. Big enough to keep a scene together and small
#: enough to fit a translation prompt with room for the glossary and the answer.
DEFAULT_CHUNK_TOKENS = 1_000
#: Tokenizer used to *measure* a chunk. ``cl100k_base`` is the GPT-4 family
#: encoding and a good proxy for most translation models; it needs the optional
#: ``tiktoken`` package. Without it the stage falls back to a local approximation
#: (and says so in the manifest) rather than downloading a model mid-run.
DEFAULT_CHUNK_TOKENIZER = "cl100k_base"
#: The tokenizer every stage run can construct without a network dependency.
CHUNK_FALLBACK_TOKENIZER = "approx"
#: Chunk files are named ``<global>-<chapter>-<index>.md``. The wide global
#: prefix makes a plain lexicographic sort of the folder the book's reading order.
CHUNK_ORDER_DIGITS = 5
CHUNK_INDEX_DIGITS = 4
#: Chunking writes next to the chapter folders, never over them.
CHUNK_MANIFEST_FILENAME = "_chunk_manifest.json"
CHUNK_TRACE_FILENAME = "_chunk_trace.json"

# --------------------------------------------------------------------------- #
# Chunk glossary stage (stage 6)
#
# Stage 6 is the bridge between stage 4's book-wide dictionary and stage 5's
# translation-sized chunks. It runs in two phases, and the order is the design:
#
#   phase 1 — NER over *every* chunk first, collecting the entity terms per
#             chunk. No glossary lookup happens until the last chunk has been
#             tagged, so the run has one complete picture before it decides what
#             any chunk means.
#   phase 2 — for every term collected in phase 1, fuzzy-retrieve the matching
#             glossary entry (rapidfuzz) and write the relevant terms next to the
#             chunk, under the chunk's own name.
#
# The match is a *retrieval*, not a translation: stage 4 already locked the
# rendering, and stage 6 only attaches it to the chunks that need it.
# --------------------------------------------------------------------------- #

#: Stage 6 writes next to ``05-chunks/``, never over it.
CHUNK_GLOSSARY_DIRNAME = "06-chunk-glossary"
CHUNK_GLOSSARY_MANIFEST_FILENAME = "_chunk_glossary_manifest.json"
CHUNK_GLOSSARY_TRACE_FILENAME = "_chunk_glossary_trace.json"

#: Fuzzy-match score, on rapidfuzz's 0..100 scale. 95 means "the extracted term
#: and the glossary term differ by at most a typo or two"; lower is more
#: permissive, higher is closer to exact. The stage prints the value it used.
DEFAULT_CHUNK_GLOSSARY_THRESHOLD = 95.0
#: The rapidfuzz scorer used for retrieval. ``ratio`` is strict and predictable;
#: ``token_sort_ratio`` ignores word order, ``wratio`` tolerates partial names.
DEFAULT_CHUNK_GLOSSARY_SCORER = "ratio"
#: Scorers offered on the command line. Keys map to ``rapidfuzz.fuzz`` functions.
CHUNK_GLOSSARY_SCORERS: tuple[str, ...] = (
    "ratio",
    "token_sort_ratio",
    "token_set_ratio",
    "wratio",
    "partial_ratio",
    "qratio",
)
#: Candidates retrieved per extracted term. The best one is the match; the rest
#: are kept as alternatives in the trace, so a near miss is visible.
DEFAULT_CHUNK_GLOSSARY_CANDIDATES = 3
#: Terms shorter than this (after stripping punctuation) are NER noise, not
#: dictionary material: single letters, stray initials, page numbers.
DEFAULT_CHUNK_GLOSSARY_MIN_TERM_CHARS = 2

# --------------------------------------------------------------------------- #
# Translation stage (stage 7)
#
# Stage 7 is the end product the first six stages were built for. It walks the
# chunks in reading order and asks the model to translate exactly one chunk per
# call, handing it the glossary rows stage 6 attached to that chunk. Every call
# is structured output validated by Pydantic (the answer is one JSON object with
# a single ``translation`` field), retried with the failure named, and priced.
#
# Two folders come out of it: ``07-translations/`` holds the translated text of each
# chunk under the chunk's own name, and ``07-translation-prompts/`` holds the whole
# input of every call — system prompt, user prompt, the serialised messages, the
# glossary sent, and the source chunk — so a bad translation can be debugged
# without reconstructing the prompt by hand.
# --------------------------------------------------------------------------- #

#: Stage 7 writes next to ``05-chunks/``, never over it.
TRANSLATIONS_DIRNAME = "07-translations"
#: The debugging twin: one prompt/input record per chunk.
TRANSLATION_PROMPTS_DIRNAME = "07-translation-prompts"
TRANSLATION_MANIFEST_FILENAME = "_translation_manifest.json"
TRANSLATION_TRACE_FILENAME = "_translation_trace.json"

#: Model calls allowed per chunk before it is reported as failed. A failed
#: validation (prose around the JSON, an empty translation, a refusal) is a
#: retry, not a hole in the book.
DEFAULT_TRANSLATION_ATTEMPTS = 3
#: Response cap per call. A 1,000-token source chunk translates to well under
#: this, and the cap also stops a model that starts repeating itself.
DEFAULT_TRANSLATION_MAX_TOKENS = 8_192
#: Characters of the *previous* chunk's translation sent as context. 0 (the
#: default) sends none: the chunker already cuts on paragraph/sentence
#: boundaries, and the glossary is the consistency mechanism. Raise it when a
#: book's chunks break mid-sentence and the joins read badly.
DEFAULT_TRANSLATION_CONTEXT_CHARS = 0
#: Raw answer characters kept in the trace, enough to see a failure, not enough
#: to turn the trace into a second manuscript.
DEFAULT_TRANSLATION_RAW_CHARS = 2_000

#: Extraction strategies, mirroring stage 4's names. ``auto`` asks for native
#: structured output and moves to plain JSON mode only when the provider rejects
#: the schema; ``structured`` insists on the schema; ``json`` starts in JSON mode.
TRANSLATION_METHODS: tuple[str, ...] = ("auto", "structured", "json")
#: Where the glossary rows sent with a chunk come from: stage 6's per-chunk
#: ``matched_terms`` (default) or the whole of ``glossary.json``.
TRANSLATION_GLOSSARY_SCOPES: tuple[str, ...] = ("chunk", "book")

# --------------------------------------------------------------------------- #
# Front/back matter localization (the stage before translation)
#
# Front and back matter are not chapters and must not be translated as if they
# were. A title page, a copyright page and a table of contents are *localized*
# for the translated edition: the title may already have a published rendering,
# some of the original edition's legal/imprint material does not belong in the
# translation, and the new edition must say who translated it. That is a
# different task from translating a body chunk, so it runs as its own LangGraph
# agent before stage 7 and never consumes the dictionary.
#
# Its translated chunks still land in ``07-translations/`` (the chunk names are the
# join key stage 8 reads); this folder holds the book-level plan, the run's
# manifest/trace, and the prompt record of every agent call.
# --------------------------------------------------------------------------- #

#: The localization stage writes its book-level record next to ``07-translations/``.
MATTER_DIRNAME = "06b-matter"
#: The debugging twin: one whole agent input record per matter chunk.
MATTER_PROMPTS_DIRNAME = "06b-matter-prompts"
MATTER_MANIFEST_FILENAME = "_matter_manifest.json"
MATTER_TRACE_FILENAME = "_matter_trace.json"

#: The translator credit the localized front matter must state. The stage fixes
#: the name; only the surrounding sentence is localized.
MATTER_TRANSLATOR_NAME = "BleBel"
#: Default English credit line, used only when the model did not supply its own
#: localized sentence. The localized form is preferred.
DEFAULT_MATTER_CREDIT = "Translated by BleBel"
#: Model turns allowed for the book-level plan and for one matter chunk.
DEFAULT_MATTER_ATTEMPTS = 5
#: Cap on the source text a single read-tool call returns. Front/back matter is
#: usually short; a larger section is still read in one call up to this cap and
#: the tool says when it had to clip (the runner then refuses to localize a
#: clipped section rather than silently dropping text).
MAX_MATTER_READ_CHARS = 60_000
#: The raw answer kept in the trace for one matter chunk.
MAX_MATTER_TRANSLATION_CHARS = 40_000
#: The minimum share of the source's length a non-omitted matter translation
#: may keep. Lower than a body chunk's because a localized title page is
#: deliberately shorter than the original imprint page.
MIN_MATTER_LENGTH_RATIO = 0.08

# --------------------------------------------------------------------------- #
# Assembly stage (stage 8)
#
# Stage 8 is the inverse of stage 5's cut, applied to the *translated* chunks:
# it groups ``07-translations/<chunk>.md`` by the chapter the chunk manifest says
# each chunk belongs to, concatenates each group in chunk order, and writes one
# translated chapter file under the source chapter's own name. The output is a
# separate folder so ``01-chapters/`` (the source) and ``08-translated-chapters/`` (the
# product) are never confused, and so a partial or resumed run can be inspected
# without touching either the source or the per-chunk translations.
# --------------------------------------------------------------------------- #

#: Stage 8 writes next to ``07-translations/``, never over it.
TRANSLATED_CHAPTERS_DIRNAME = "08-translated-chapters"
ASSEMBLY_MANIFEST_FILENAME = "_translated_chapters_manifest.json"
ASSEMBLY_TRACE_FILENAME = "_translated_chapters_trace.json"

# --------------------------------------------------------------------------- #
# Postprocess stage (stage 9)
#
# Stage 9 is a book-level *heading* pass over the assembled translated chapters.
# It never reads a chapter's prose: the unit of work is the chapter's opening
# (a handful of clipped lines), its heading records with short surrounding
# context, and the book-level list of chapter titles. The agent decides a naming
# policy for the whole book first, then a per-chapter plan (the chapter heading,
# and keep/demote/promote for every other heading), and a table-of-contents plan.
#
# The output is a new ``09-postprocessed-chapters/`` folder: the translated chapter
# files are never edited in place, exactly as stages 1-8 never edit their input.
# --------------------------------------------------------------------------- #

#: Stage 9 writes next to ``08-translated-chapters/``, never over it.
POSTPROCESSED_CHAPTERS_DIRNAME = "09-postprocessed-chapters"
POSTPROCESS_MANIFEST_FILENAME = "_postprocess_manifest.json"
POSTPROCESS_TRACE_FILENAME = "_postprocess_trace.json"
#: The final, single-document concatenation of the postprocessed chapters.
#: Stage 10 writes one file, not a folder, but it carries its number too so it sorts
#: after every stage folder.
TRANSLATED_BOOK_FILENAME = "10-translated-book.md"
BIND_MANIFEST_FILENAME = "_bind_manifest.json"
BIND_TRACE_FILENAME = "_bind_trace.json"

#: Stage 11 packages the same postprocessed chapters as an EPUB 3 book: the
#: Markdown file is the reading copy, the EPUB is the one you hand to a reader.
TRANSLATED_EPUB_FILENAME = "11-translated-book.epub"
EPUB_MANIFEST_FILENAME = "_epub_manifest.json"
EPUB_TRACE_FILENAME = "_epub_trace.json"
#: Inside the container: the pictures, the text documents and the stylesheet.
EPUB_IMAGES_DIRNAME = "images"
EPUB_TEXT_DIRNAME = "text"
EPUB_STYLESHEET_FILENAME = "style.css"
#: The media type every EPUB names in its first, uncompressed zip entry.
EPUB_MIMETYPE = "application/epub+zip"

#: A table of contents created when the book has none. ``00-`` sorts it after
#: ``00-front-matter.md`` (``f`` < ``t``) and before ``01-…``.
TOC_FILENAME = "00-table-of-contents.md"
DEFAULT_TOC_HEADING = "Sadržaj"

#: Only this many lines at the start of a chapter may be inspected by the
#: postprocessing agent. The opening is where a chapter title lives.
MAX_POSTPROCESS_OPENING_LINES = 8
#: Characters of the opening a tool may return in one call.
MAX_POSTPROCESS_OPENING_CHARS = 2_400
#: A chapter title may be no longer than this.
MAX_POSTPROCESS_TITLE_CHARS = 160
#: The title candidate is searched in the first N lines of a chapter.
MAX_TITLE_SCAN_LINES = 24
#: Headings returned per chapter (the inventory is metadata, not prose).
MAX_POSTPROCESS_HEADINGS = 80
#: Lines of context (on each side) around a heading accepted by the inspector.
MAX_POSTPROCESS_CONTEXT_LINES = 2
#: Characters of the front/back matter a tool may return per block.
MAX_POSTPROCESS_MATTER_CHARS = 4_000
#: Lines of the front/back matter a tool may return per block.
MAX_POSTPROCESS_MATTER_LINES = 40
#: A TOC with more entries than this is truncated (a book does not have 300).
MAX_TOC_ENTRIES = 200
#: Default target-heading level for a normalized chapter title.
DEFAULT_CHAPTER_TITLE_LEVEL = 1
#: Default number of model turns per book plan / per chapter.
DEFAULT_POSTPROCESS_ATTEMPTS = 4

#: Title actions accepted by a chapter plan.
TITLE_ACTIONS: tuple[str, ...] = ("keep", "set", "insert")
#: Heading decisions accepted by a chapter plan. ``demote`` removes the ``#``
#: markup and keeps the text, which is exactly "remove the heading, leave the
#: sentence"; ``promote`` is its inverse: add the markup to a plain ALL-CAPS
#: heading whose markup was lost. There is deliberately no delete operation.
HEADING_DECISIONS: tuple[str, ...] = ("keep", "demote", "promote")
#: Naming policies accepted by a book plan.
NAMING_MODES: tuple[str, ...] = ("keep", "normalize")
#: TOC actions accepted by a book plan.
TOC_ACTIONS: tuple[str, ...] = ("create", "update", "keep")
#: TOC entry styles accepted by a book plan.
TOC_STYLES: tuple[str, ...] = ("numbered", "bullets")

# --------------------------------------------------------------------------- #
# Chapter-opening audit (the final phase of stages 2 and 9)
#
# The agent stages decide per chapter; this deterministic pass then walks the
# *book* and asks one narrow question of every chapter's first lines: does the
# opening read like a chapter opening? A title followed by a stray copy of its
# number (``# Poglavlje 2`` then ``2``), a duplicated title line and a leftover
# page number are the shapes it looks for. It reads only the top of a chapter and
# repairs only a line that is nothing but that duplicate/artefact.
# --------------------------------------------------------------------------- #

#: Non-blank opening lines the audit may read per chapter.
MAX_OPENING_AUDIT_LINES = 12
#: Physical lines scanned to find those non-blank lines.
MAX_OPENING_AUDIT_SCAN_LINES = 40
#: Characters kept per opening line in a report or tool result.
MAX_OPENING_AUDIT_CHARS = 160
#: Chapters a single opening-audit tool call may expand (the rest are counted).
MAX_OPENING_AUDIT_CHAPTERS = 60

# --------------------------------------------------------------------------- #
# Conversion stage (stage 0)
#
# Stage 0 turns a source book into the Markdown every later stage reads. EPUB and
# PDF go through docling; a plain-text file is copied (the content is unchanged,
# only the extension becomes ``.md``). An image-based PDF is *rejected* rather
# than OCR'd: this pipeline is built for sources that already carry a text layer,
# and OCR would invent a different book. The preflight below samples pages of the
# PDF and asks how many of them carry real characters, so that decision is made
# before docling loads a layout model.
# --------------------------------------------------------------------------- #

#: Stage 0 writes next to the stage folders, so it sorts before ``01-chapters/``.
CONVERT_MANIFEST_FILENAME = "_convert_manifest.json"
#: Images extracted from an EPUB/PDF live next to the converted Markdown. Every
#: later stage is one folder below the book folder, so one relative prefix works
#: from stage 0 through stage 9; stage 10 rewrites it once for the book root.
IMAGES_DIRNAME = "images"
CHAPTER_IMAGE_PREFIX = f"../{MARKDOWN_DIRNAME}/{IMAGES_DIRNAME}/"
ROOT_IMAGE_PREFIX = f"{MARKDOWN_DIRNAME}/{IMAGES_DIRNAME}/"
#: PDF pages sampled for the text-layer preflight (all of them in a short PDF).
PDF_TEXT_SAMPLE_PAGES = 12
#: Characters that make a sampled page "text-bearing" rather than a page number.
PDF_MIN_TEXT_CHARS_PER_PAGE = 24
#: Share of the sampled pages that must carry text for the PDF to be text-based.
PDF_MIN_TEXT_PAGE_RATIO = 0.5
#: A conversion that produces less than this is treated as a failure, not a book.
MIN_CONVERTED_MARKDOWN_CHARS = 32

# --------------------------------------------------------------------------- #
# Output names
# --------------------------------------------------------------------------- #

FRONT_MATTER_FILENAME = "00-front-matter.md"
MANIFEST_FILENAME = "_manifest.json"
TRACE_FILENAME = "_agent_trace.json"

#: Stage 1 (splitting) writes the source chapters; stage 2 (preprocessing) writes
#: next to it, never over it. Both names carry the stage number.
PREPROCESSED_DIRNAME = "02-preprocessed"
PREPROCESS_MANIFEST_FILENAME = "_preprocess_manifest.json"
PREPROCESS_TRACE_FILENAME = "_preprocess_trace.json"
