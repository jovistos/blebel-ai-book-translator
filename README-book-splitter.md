# Book pipeline — two LangGraph agents that turn a Markdown book into clean chapter files

The repository holds a two-stage pipeline. Each stage has its **own specialised
agent** with its **own toolset**, and the tools are grouped by the stage that
uses them (plus a generic `shared` group used by both):

| Stage | Agent | Question it answers | Writes |
| --- | --- | --- | --- |
| 1. **Split** | `book_splitter.agent` | *Where does each chapter start?* | `data/output/<book>/chapters/` |
| 2. **Preprocess** | `book_splitter.preprocess_agent` | *What in this chapter is not part of the book's text?* | `data/output/<book>/preprocessed/` |

Stage 1 turns one long Markdown book into a folder of chapter files
**without ever reading the book's body**. The LLM only ever *routes*: it looks at
line numbers and short title strings produced by deterministic tools, decides
where the chapters start, and submits a plan. A plain Python node then copies the
source lines verbatim into one file per chapter, and a verifier proves that the
chapter files concatenate back to the original **byte for byte**.

Stage 2 repairs each chapter file, **one chapter at a time, sequentially**. It
starts from the chapter's **headers**, never from its prose: page numbers that
became headings, figure captions marked up as headings, headings the PDF
conversion split across two lines, and words a page number was printed inside.
The agent describes its edits in a fixed vocabulary; a deterministic validator
checks every entry against the real file, and a per-line provenance map proves
afterwards that only the declared lines changed.

This is the implementation of the toolset described in
[gemini-splitting-markdown-books-without-llms.md](gemini-splitting-markdown-books-without-llms.md),
built for the sample book
[data/input/Aleksandar_Loven_-_Ljubav_seks_i_vase_srce.md](data/input/Aleksandar_Loven_-_Ljubav_seks_i_vase_srce.md)
whose PDF→Markdown conversion mangled the headings (78 `##` headers of which most are page
numbers and figure captions, `GIAVA 5` instead of `GLAVA 5`, chapter titles split onto the line
*below* `## GLAVA 2`, and an unfinished word like `ZDRAVO SRCE`). Fixing exactly that kind of
damage is what the LLM is for — and it does it by reading ~80 header lines, not 411 KB of prose.

## Documents: input and output are separate folders

Nothing that is a *document* lives next to the code any more:

```
data/
├── input/                                    # sources — read-only, gitignored
│   └── Aleksandar_Loven_-_Ljubav_seks_i_vase_srce.md
└── output/                                   # every generated artifact — gitignored
    └── Aleksandar_Loven_-_Ljubav_seks_i_vase_srce/
        ├── chapters/                         # stage 1 output
        │   ├── 00-front-matter.md
        │   ├── 01-predgovor.md
        │   ├── …
        │   ├── 15-beleske.md
        │   ├── _manifest.json                # filenames, line ranges, bytes, SHA-256
        │   └── _agent_trace.json             # the plan, validation report, token usage
        └── preprocessed/                     # stage 2 output — same names, cleaned
            ├── 00-front-matter.md
            ├── …
            ├── _preprocess_manifest.json     # per chapter: operations, fixes, provenance
            └── _preprocess_trace.json        # the full per-chapter audit trail
```

Stage 2 never overwrites stage 1's files: `chapters/` stays exactly as the book
came out of the splitter, so every repair can be re-derived and re-checked.

`data/input/` and `data/output/` are kept in the repository with a `.gitkeep`; their contents
are ignored, so a book and its output never end up in a commit. The layout is resolved in one
place — [book_splitter/paths.py](book_splitter/paths.py) — and can be overridden with
`--in-dir` and `--out` or from Python through `book_splitter.paths`.

## Result on the sample book

```
$ python -m book_splitter split Aleksandar_Loven_-_Ljubav_seks_i_vase_srce.md
strategy        : markdown_headers
llm calls       : 3  tokens: 33768
chapters        : 15
byte-for-byte   : True
lines           : 640/640
output dir      : …/data/output/Aleksandar_Loven_-_Ljubav_seks_i_vase_srce/chapters
```

The output folder contains `00-front-matter.md` plus 15 sections (`01-predgovor.md` …
`15-beleske.md`). Their concatenation hashes identically to the source file
(`2cfa45bd…aa29a`), so not a single character was rewritten — including the typos inherited
from the PDF conversion (`## GIAVA 5 …` is preserved verbatim inside
[08-glava-5-strah-od-ljubavi.md](data/output/Aleksandar_Loven_-_Ljubav_seks_i_vase_srce/chapters/08-glava-5-strah-od-ljubavi.md)).

Stage 2 then cleans those files, one at a time:

```
$ python -m book_splitter preprocess
chapters        : 16/16 clean
llm calls       : 59  tokens: 499894
damage found    : page_number_header=26, page_number_in_word=9, space_before_punctuation=29,
                  caption_as_header=10, long_prose_header=4, single_char_header=1
output dir      : …/data/output/Aleksandar_Loven_-_Ljubav_seks_i_vase_srce/preprocessed
files:
  ok  04-glava-1-ljubav-je-u-srcu-zivota.md   lines    58 -> 55   ops=7  fixes=3
  ok  05-glava-2-seks-i-srce.md               lines    34 -> 32   ops=4  fixes=3
  …
  ok  14-glava-10-zdravo-srce-osoba-koja-voli.md  lines 83 -> 76  ops=14 fixes=8
OK — every chapter cleaned and verified.
```

Across the book: **26 page-number headings deleted, 10 figure captions demoted, 38 body-text
fixes applied, 640 → 610 lines**, every chapter verified, and the character audit reports
**0 letter-level differences**. The visible result: every chapter now has exactly one real
heading (`## GLAVA 2 SEKS I SRCE` instead of `## GLAVA 2` + `## SEKS I SRCE`), the 26 page
numbers are gone, and `… psi182 ljutnje` became `… psiljutnje`.

## How it works

**Stage 1 — split** (`book_splitter.agent`):

```
START
  │
scout                     deterministic recon: headers, header quality, printed-TOC block   (no LLM)
  │
agent ◄────────────┐      LLM picks a strategy and calls tools
  │  ╲             │
  │   ▼            │
  │  tools ────────┘      executes tool calls + enforces the write policy + absorbs the plan
  │
  ▼
validate ───────► agent   deterministic checks; failures are fed back as corrections
  │
slice                     verbatim copy of the source into one file per chapter
  │
verify                    SHA-256 proof that nothing was lost or altered
  │
 END
```

Strategy fallback, in the order the agent is told to try:

1. **Native TOC** (`extract_native_toc`) — free and exact when the source is a PDF/EPUB.
2. **Markdown headers** (`extract_markdown_headers`) — used when the header set is *clean*
   enough; when it is merely *noisy* the agent filters it itself (this is what happened with
   the sample book).
3. **Printed TOC** (`detect_toc_block_tool` + `extract_front_matter` +
   `locate_toc_titles_in_file`) — the TOC gives the correct chapter order; fuzzy matching
   translates titles into body line numbers and survives conversion typos.
4. **Heuristics** (`extract_heuristic_candidates`) — short standalone lines, ALL CAPS lines and
   `CHAPTER/PART/GLAVA/DEO/POGLAVLJE` keywords, for books that lost all markup.

**Stage 2 — preprocess** (`book_splitter.preprocess_agent`), run once **per chapter**:

```
START
  │
collect                   deterministic recon: the chapter's headers + evidence + damage  (no LLM)
  │
agent ◄────────────┐      LLM decides on each header, in a fixed vocabulary
  │  ╲             │
  │   ▼            │
  │  tools ────────┘      executes tool calls + blocks writes for this chapter
  │
  ▼
validate ───────► agent   every operation checked against the real file; failures loop back
  │
apply                     deterministic write of the cleaned chapter (pages artefacts dropped)
  │
verify                    per-line provenance proof that only declared edits happened
  │
 END
```

The chapter loop itself lives in `run_preprocess` and is **sequential by
default** (`--workers 1`): chapter *n* is finished before chapter *n+1* starts,
which keeps the log readable and the failure mode obvious. Switching to a thread
pool is `--workers 4` and nothing else, because every chapter's work is a pure
function of `(chapter_path, output_dir)`, the graph is built once, and results
are collected before the book-level manifest is written.

### What stage 2 actually repairs

| Damage in the sample book | Detection | Repair |
| --- | --- | --- |
| `## 18`, `## 101`, `## 790` — the printed page number became a heading | header body is only a number | `remove` the line |
| `19`, `21`, `22` — a page number left as a plain body line | body line is only a number | `remove` the line |
| `## Slika 7. Protok uzbuđ enja u telu.`, `Slika 4. Blokada ljubavi.` — a figure caption, marked or not | caption marker + number, short or single-sentence body | `remove_caption` — the picture is not in the Markdown, so the label is not book text |
| `## GLAVA 2` / `## SEKS I SRCE` — one heading split over two lines | structural label with no title, next line a short header | `merge_with_next` |
| `## … i želja za smrć` / `## u` — the first half cut mid-word | unfinished tail continued by the next header | `merge_with_next` |
| `… psi182 ljutnje` — a page number printed *inside* a word | lowercase, digits, lowercase | drop the digits, glue the word (`psiljutnje`) |
| `1otkucaja u minuti` — a page number glued to the next word | digits at the start of a line, fused to a lowercase word | drop the digits (`otkucaja`) |
| `mišić ,` — whitespace before punctuation | space before `, . ; : ! ?` | delete the space |

The rules are deliberately narrow. `brzinu od 130 otkucaja`, `Tiraž 1000` and
`(TxA2)`/`(PG12)` are never touched, because an inline number is separated from
its word by a space and is therefore not the shape of a page break. A wrong
`remove` can only ever delete a line that is *nothing but digits*; a wrong
`remove_caption` can only ever delete a line that opens with a caption marker and
a number, is at most 200 characters or a single sentence, and does not read as
prose about the figures (`Slike 12a i 12b ilustruju …` is left alone). The
verifier re-checks both shapes after the fact.

Removed lines leave a **blank slot** rather than shortening the file, so a line
number in the log, the manifest or the trace still points at the same text in
`chapters/`.

## Tools, grouped by stage

`python -m book_splitter tools` prints exactly this grouping, and each stage
binds only its own tools plus the `shared` ones — so the splitting agent cannot
reach the cleaning tools and the preprocessing agent cannot slice the book.

### `shared` — generic, used by both stages

| Tool | Purpose |
| --- | --- |
| `extract_markdown_headers` | `#` headers + exact line numbers (stage 1 picks cuts from it, stage 2 inspects them) |
| `inspect_line_window` | at most 6 clipped lines around one line number — the only way to look "at the text" |
| `list_chapter_files` | stage 1's manifest, read back as an ordered chapter list |
| `sample_corpus_text` | a bounded, strided sample of one chapter (labelled SAMPLES ONLY) |
| `scan_corpus_damage_tool` | the counted damage profile of a whole book |

### `split` — stage 1 only

| Tool | Purpose | Added |
| --- | --- | --- |
| `extract_native_toc` | Embedded PDF/EPUB bookmark outline | |
| `extract_front_matter` | First *N* lines only (the only prose the LLM may see) | |
| `locate_toc_titles_in_file` | Fuzzy title → line number, skips the TOC block, snaps sub-headings | |
| `extract_heuristic_candidates` | Title-looking lines when the markup is gone | |
| `slice_book_by_line_numbers` | Writes one file per chapter, verbatim | |
| `detect_toc_block_tool` | Locates the printed TOC block and its line range | ✔ |
| `propose_split_map` | The tool the agent *submits its answer* through | ✔ |
| `validate_split_map_tool` | Dry-run scoring of a plan before anything is written | ✔ |
| `verify_sliced_output` | Re-reads the output and checks it against the source | ✔ |

### `preprocess` — stage 2 only

| Tool | Purpose |
| --- | --- |
| `collect_chapter_headers` | **the first step**: every header of one chapter with its evidence and a deterministic recommendation |
| `collect_book_headers` | the same inventory across the whole book, in one call |
| `extract_headers_from_file` | a bare header list for any single file |
| `inspect_chapter_damage` | body-text damage by shape (line number + fragments + reconstructed word) |
| `scan_book_damage` | the book-wide damage profile plus a text digest |
| `validate_chapter_cleaning` | dry-run a chapter's plan against the real file |
| `report_chapter_cleaning` | **the tool the agent submits its plan through** |
| `apply_chapter_cleaning` | writes the cleaned chapter (blocked until its plan is validated) |
| `verify_preprocessed_chapters` | re-checks a whole `preprocessed/` folder against `chapters/` |

## Guardrails

- **No body text in the prompt.** Tools return line numbers, headers, candidate titles and
  ≤90-character excerpts. Stage 2 adds evidence flags, damage fragments and `inspect_line_window`
  (≤6 lines, ≤600 characters per call); every tool result is capped at 24 000 characters and none
  of the stage-2 tools can return a chapter's prose.
- **No hallucinated line numbers.** Every proposed split point is checked against the real file:
  in range, strictly increasing, unique, actually a heading or a standalone title line, and
  *not* inside the printed TOC block. A bad map is rejected with concrete fixes and sent back to
  the agent (bounded by `--max-attempts`).
- **No writes before validation.** `slice_book_by_line_numbers` is blocked at the graph level
  until validation passes, even if the model calls it. Stage 2 does the same per chapter:
  `apply_chapter_cleaning` is blocked until *that* chapter's plan has been validated.
- **No content loss.** The verifier recomputes SHA-256 over the files on disk and compares it
  with the source slice the manifest claims to cover. Stage 2 goes further: every source line is
  accounted for in a provenance map, and a line may only disappear when it was a header
  consisting of nothing but digits *and* the plan declared it.
- **No arbitrary rewriting.** Stage 2 has no free-text edit. A plan entry can only choose one of
  four decisions (`keep` / `demote` / `remove` / `merge_with_next`) plus three named text fixes,
  each with a machine-checkable rule; `apply` re-derives the output with that rule and the
  verifier re-runs it.
- **No path escapes.** Filenames are normalised to `NN-slug.md`; directory components are
  stripped and duplicates are disambiguated instead of overwritten. A filename whose slug is a
  near-miss of its own short heading (`15-belestke.md` for `## BELEŠKE`) is repaired from the
  heading, while descriptive names such as `05-glava-2-seks-i-srce.md` are left untouched.
  Stage 2 keeps each chapter's filename, so `chapters/` and `preprocessed/` line up one to one.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-book-splitter.txt -r requirements-dev.txt
```

Optional, only for `extract_native_toc` with PDF/EPUB sources:
`pip install PyMuPDF ebooklib beautifulsoup4`.

## Use

```bash
# 1. What does the book actually contain? (no LLM, no API key, no writes)
#    With a single book in data/input/ the argument can be omitted entirely.
python -m book_splitter inspect --candidates

# 2. Stage 1 — split it (writes to data/output/<book>/chapters/)
python -m book_splitter split BOOK.md

# Plans the chapters and validates them, but writes nothing
python -m book_splitter split BOOK.md --dry-run

# 3. What does stage 2 have to fix? (no LLM, no writes)
python -m book_splitter scan                  # the book-wide damage profile
python -m book_splitter scan --digest         # per chapter: headers + evidence + recommendation

# 4. Stage 2 — clean every chapter, one at a time
#    (writes to data/output/<book>/preprocessed/)
python -m book_splitter preprocess
python -m book_splitter preprocess --dry-run          # validate every plan, write nothing
python -m book_splitter preprocess --workers 4        # same work, thread pool

# 5. Optional: verify a preprocessed folder against the chapters it came from
python -m book_splitter tools --stage preprocess

# Explicit folders / fully local model
python -m book_splitter split BOOK.md --out /tmp/chapters/
python -m book_splitter preprocess --out /tmp/chapters/ --preprocessed-out /tmp/clean/
python -m book_splitter preprocess \
    --provider lmstudio --model qwen/qwen3.8-27b --base-url http://127.0.0.1:1234/v1
```

Useful flags: `--in-dir`, `--out`, `--preprocessed-out`, `--max-attempts`, `--workers`,
`--front-matter-lines`, `--no-front-matter`, `--dry-run`, `--json`, `--quiet`,
`--temperature`, `--api-key`. `python -m book_splitter tools` lists the toolset **grouped by
stage**, and [scripts/check_layout.py](scripts/check_layout.py) verifies the folder layout and the
installed dependencies.

### Configuration

Copy [.env.example](.env.example) to `.env`, or pass flags. Credential lookup order:
`--api-key` → `SPLITTER_API_KEY` / the provider's own variable → `.env` → `refs:` in
`~/.dsh/.credentials.yaml`. Local endpoints (`localhost`, `127.0.0.1`) may run keyless.
Both stages use the same provider settings.

### Output

```
data/output/<book-stem>/chapters/          # stage 1
├── 00-front-matter.md                     # cover/copyright/TOC, everything before chapter 1
├── 01-predgovor.md
├── 02-uvod.md
├── …
├── 15-beleske.md
├── _manifest.json                         # filenames, line ranges, byte counts, SHA-256
└── _agent_trace.json                      # the plan, the validation report, the token usage

data/output/<book-stem>/preprocessed/      # stage 2
├── 00-front-matter.md … 15-beleske.md     # same names, cleaned
├── _preprocess_manifest.json              # per chapter: operations, text fixes, provenance
└── _preprocess_trace.json                 # the full per-chapter audit trail
```

`_agent_trace.json` is stage 1's audit trail: which strategy won, what the agent proposed, and
what the validators said about it. `_preprocess_trace.json` is stage 2's: for every chapter, the
operations the agent chose, the fixes it requested, the validator's verdict, and the provenance
map of the file that was written.

## Tests

```bash
python -m pytest          # 86 tests, no network, no API key
```

They pin the behaviour that matters: the canonical split set of the sample book is rediscovered,
the typo'd and two-line headings resolve to the right lines, malformed plans (TOC entries, prose
lines, duplicates, out-of-range line numbers) are rejected, slicing is byte-for-byte reversible,
the graph blocks premature writes, a rejected plan loops back with corrections, the dry run
writes nothing, and — asserted directly — **no chapter prose ever appears in the messages sent
to the model**.

Stage 2's tests use the exact damaged strings the sample book contains
(`… psi182 ljutnje`, `1otkucaja`, `disfunkcional38 nost`, `75 lič nosti`), and assert the two
halves of the safety story: a repair rule may only remove the page number it was written for
(`brzinu od 130 otkucaja`, `Tiraž 1000`, `(TxA2)`/`(PG12)` survive), and a plan that
demotes body text, removes a real heading, merges into a page number, or claims damage that is
not there is rejected before anything is written.

### Replaying a plan without the model

Changing a *repair rule* (not an agent decision) does not need the model at all. The trace holds
every plan, so it can be replayed through the same validator, applier and verifier:

```bash
python scripts/replay_preprocess.py --dry-run      # validate every recorded plan
python scripts/replay_preprocess.py --only 04- 07- # rewrite just those chapters
```

That is how the caption and page-number rules were finalised here: the decisions stayed, the
transformation changed, and all 16 chapters verified again for the price of a few seconds.

### Auditing a real run

Stage 2 is the only place in the pipeline that may change a character, so a real run is followed
by a character-level audit:

```bash
python scripts/audit_preprocess_tokens.py            # summary
python scripts/audit_preprocess_tokens.py --details  # every difference
```

It compares every chapter with its cleaned version and fails if any *letter* differs. Only page
numbers, punctuation spacing and `#` markup may change. On the sample book it reports
**0 letter-level differences** — the 250-odd edits are all page artefacts. The script is
deliberately slow on chapters whose paragraphs are single 8000-character lines.

## Layout

The package is split by responsibility; each folder can be read on its own.

| Path | Role |
| --- | --- |
| [book_splitter/paths.py](book_splitter/paths.py) | where documents live: `data/input/` in, `data/output/` out |
| [book_splitter/cli.py](book_splitter/cli.py) | `inspect` / `split` / `scan` / `preprocess` / `tools` commands |
| [book_splitter/core/constants.py](book_splitter/core/constants.py) | every guardrail and default in one place |
| [book_splitter/core/settings.py](book_splitter/core/settings.py) | provider and credential resolution |
| [book_splitter/core/errors.py](book_splitter/core/errors.py) | `ToolError`, `ConfigError` |
| [book_splitter/documents/io.py](book_splitter/documents/io.py) | cached source reading |
| [book_splitter/documents/markdown.py](book_splitter/documents/markdown.py) | header/standalone/prose line shapes (both stages) |
| [book_splitter/documents/bodyline.py](book_splitter/documents/bodyline.py) | body-line *damage* shapes: split words, mid-word page numbers |
| [book_splitter/documents/chapters.py](book_splitter/documents/chapters.py) | reads stage 1's `chapters/` folder back as chapter records |
| [book_splitter/documents/text.py](book_splitter/documents/text.py) | folding, similarity, slugs, tool-result caps |
| [book_splitter/analysis/toc.py](book_splitter/analysis/toc.py) | printed-TOC block detection |
| [book_splitter/analysis/headers.py](book_splitter/analysis/headers.py) | `strong` / `noisy` / `absent` verdict |
| [book_splitter/analysis/candidates.py](book_splitter/analysis/candidates.py) | heuristic title candidates |
| [book_splitter/analysis/locate.py](book_splitter/analysis/locate.py) | fuzzy title → line number, heading snapping |
| [book_splitter/slicing/plan.py](book_splitter/slicing/plan.py) | safe filenames and line ranges |
| [book_splitter/slicing/slice.py](book_splitter/slicing/slice.py) | verbatim copy, manifest, dry-run preview |
| [book_splitter/slicing/validation.py](book_splitter/slicing/validation.py) | deterministic map scoring |
| [book_splitter/slicing/verify.py](book_splitter/slicing/verify.py) | SHA-256 proof that nothing was lost |
| [book_splitter/preprocess/inspect.py](book_splitter/preprocess/inspect.py) | one chapter's headers + evidence + deterministic recommendation |
| [book_splitter/preprocess/damage.py](book_splitter/preprocess/damage.py) | damage detection and the three repair rules |
| [book_splitter/preprocess/corpus.py](book_splitter/preprocess/corpus.py) | sampling and the book-wide damage profile |
| [book_splitter/preprocess/transform.py](book_splitter/preprocess/transform.py) | plan validation, application, provenance, verification |
| [book_splitter/agent/prompts.py](book_splitter/agent/prompts.py) | stage 1 system prompt and the fallback protocol |
| [book_splitter/agent/llm.py](book_splitter/agent/llm.py) | the chat model with the toolset bound |
| [book_splitter/agent/tools/registry.py](book_splitter/agent/tools/registry.py) | tool registry with the `shared` / `split` / `preprocess` stage tags |
| [book_splitter/agent/tools/shared.py](book_splitter/agent/tools/shared.py) | the generic tools both stages bind |
| [book_splitter/agent/tools/](book_splitter/agent/tools/__init__.py) | stage 1 tools: `extract` / `toc` / `proposal` / `execution` |
| [book_splitter/agent/tools/preprocess.py](book_splitter/agent/tools/preprocess.py) | stage 2 tools: header collection, damage, plan submission, apply |
| [book_splitter/agent/nodes/](book_splitter/agent/nodes/__init__.py) | `scout` / `agent` / `tools` / `validate` / `slice` / `verify` |
| [book_splitter/agent/routing.py](book_splitter/agent/routing.py) | the conditional edges of the splitting graph |
| [book_splitter/agent/graph.py](book_splitter/agent/graph.py) | splitting graph assembly |
| [book_splitter/agent/runner.py](book_splitter/agent/runner.py) | `run_split` and the trace writer |
| [book_splitter/agent/state.py](book_splitter/agent/state.py) | stage 1 graph state (metadata only, never prose) |
| [book_splitter/agent/result.py](book_splitter/agent/result.py) | the `SplitterResult` dataclass |
| [book_splitter/preprocess_agent/prompts.py](book_splitter/preprocess_agent/prompts.py) | stage 2 system prompt and the decision vocabulary |
| [book_splitter/preprocess_agent/llm.py](book_splitter/preprocess_agent/llm.py) | the model with *only* the preprocessing tools bound |
| [book_splitter/preprocess_agent/nodes.py](book_splitter/preprocess_agent/nodes.py) | `collect` / `agent` / `tools` / `validate` / `apply` / `verify` |
| [book_splitter/preprocess_agent/routing.py](book_splitter/preprocess_agent/routing.py) | the conditional edges of the preprocessing graph |
| [book_splitter/preprocess_agent/graph.py](book_splitter/preprocess_agent/graph.py) | preprocessing graph for **one chapter** |
| [book_splitter/preprocess_agent/runner.py](book_splitter/preprocess_agent/runner.py) | `run_preprocess`: the sequential (or pooled) chapter loop |
| [book_splitter/preprocess_agent/state.py](book_splitter/preprocess_agent/state.py) | stage 2 graph state (metadata only, never prose) |
| [tests/](tests/conftest.py) | the test suite, split the same way as the code |

The old flat module paths (`book_splitter.tools`, `book_splitter.graph`,
`book_splitter.config`, `book_splitter.prompts`, `book_splitter.state`) are kept as thin
re-export shims, so existing imports and notebooks keep working.

## Related but external

[external/](external/README.md) is **not part of this repository**. It holds third-party
checkouts — currently `TranslateBooksWithLLMs`, itself a separate git repository — for local
side-by-side use. The whole folder is gitignored (only its README is tracked), and
`book_splitter` never imports from it.

## Limitations

- Splitting is **line-based**: a chapter must start at the beginning of a line. Two-column PDF
  conversions that interleave text *within* a line still split correctly at headings, but the
  interleaving inside a chapter is not repaired (that is a translation/cleanup concern).
- The printed-TOC route needs the TOC to be in the front matter and its titles to resemble the
  body headings; titles that are entirely absent from the body fall back to heuristics.
- Front matter is kept in `00-front-matter.md` rather than discarded.
- Stage 2 repairs **shapes**, not meanings. It cannot fix a mis-OCR'd word (`ielja` for `želja`),
  a paragraph whose sentences were reordered by a two-column page break, or a page number that is
  indistinguishable from a real inline number — those stay as they are, and `inspect_chapter_damage`
  reports the shapes it found so a human can decide. The tool is designed to be extended: adding a
  repair means adding one detector in `preprocess/damage.py` and one branch in `verify_text_fixes`.
- **A page number that split a word at a line start is only half repaired.** `30 spreč avaju` has
  the digits removed, but the fragments stay apart, so the sentence reads `spreč avaju` rather than
  `sprečavaju`. Joining them needs the one piece of context a shape rule does not have: the same
  pattern (`40 godina, patila je …`) opens legitimate paragraphs with a real number. `130 otkucaja`,
  `Tiraž 1000`, `(TxA2)` and `(PG12)` are never touched for the same reason. A future version could
  let the agent decide from a two-line window — the decision would be validated and verified like
  every other one — but that is not implemented, and guessing it in a rule would risk gluing real
  prose together.
- **A caption that takes more than one sentence to describe its picture is kept as text.**
  `Slika 10. Rascep jedinstva na ego-centar i srce … Svi impuls … Kada postoji rascep …` is a
  250-character, five-sentence label; the shape `Slika <n> <lowercase word>` is also how prose that
  refers to a figure begins, so the rule refuses to delete it. It stays as a paragraph. One such
  line exists in the sample book; the other fifteen captions are removed.
- Stage 2's `merge_with_next` joins two source lines into one, and `remove` / `remove_caption` leave
  a **blank slot** where the artefact was. Neither output line is byte-equal to its source by
  design, which is why the provenance map records the source lines and the verifier re-derives the
  transformation instead of comparing bytes.
- Stage 2 reads only the *headers* of a chapter, plus the page numbers and captions it recognises in
  the body, unless the agent explicitly asks for a window or a damage scan — so a defect that has no
  header and no recognised shape is invisible to it. A blank slot also means the cleaned chapter has
  the same line numbering as `chapters/`, at the cost of a few consecutive blank lines.
