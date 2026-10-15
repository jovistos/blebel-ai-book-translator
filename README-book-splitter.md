# Book Splitter — a LangGraph agent that splits a Markdown book into chapter files

A LangGraph agent that turns one long Markdown book into a folder of chapter files
**without ever reading the book's body**.

The LLM only ever *routes*: it looks at line numbers and short title strings produced by
deterministic tools, decides where the chapters start, and submits a plan. A plain Python node
then copies the source lines verbatim into one file per chapter, and a verifier proves the
chapter files concatenate back to the original **byte for byte**.

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
        └── chapters/
            ├── 00-front-matter.md
            ├── 01-predgovor.md
            ├── …
            ├── 15-beleske.md
            ├── _manifest.json                # filenames, line ranges, bytes, SHA-256
            └── _agent_trace.json             # the plan, validation report, token usage
```

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

## How it works

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

## Tools

The six tools from the reference document, plus four additions that make the agent reliable:

| Tool | Purpose | Added |
| --- | --- | --- |
| `extract_native_toc` | Embedded PDF/EPUB bookmark outline | |
| `extract_markdown_headers` | `#` headers + exact line numbers | |
| `extract_front_matter` | First *N* lines only (the only prose the LLM may see) | |
| `locate_toc_titles_in_file` | Fuzzy title → line number, skips the TOC block, snaps sub-headings | |
| `extract_heuristic_candidates` | Title-looking lines when the markup is gone | |
| `slice_book_by_line_numbers` | Writes one file per chapter, verbatim | |
| `detect_toc_block_tool` | Locates the printed TOC block and its line range | ✔ |
| `propose_split_map` | The tool the agent *submits its answer* through | ✔ |
| `validate_split_map_tool` | Dry-run scoring of a plan before anything is written | ✔ |
| `verify_sliced_output` | Re-reads the output and checks it against the source | ✔ |

## Guardrails

- **No body text in the prompt.** Tools return line numbers, headers, candidate titles and
  ≤90-character excerpts. The single prose-returning tool (`extract_front_matter`) is capped in
  lines *and* characters; every tool result is additionally capped at 24 000 characters.
- **No hallucinated line numbers.** Every proposed split point is checked against the real file:
  in range, strictly increasing, unique, actually a heading or a standalone title line, and
  *not* inside the printed TOC block. A bad map is rejected with concrete fixes and sent back to
  the agent (bounded by `--max-attempts`).
- **No writes before validation.** `slice_book_by_line_numbers` is blocked at the graph level
  until validation passes, even if the model calls it.
- **No content loss.** The verifier recomputes SHA-256 over the files on disk and compares it
  with the source slice the manifest claims to cover.
- **No path escapes.** Filenames are normalised to `NN-slug.md`; directory components are
  stripped and duplicates are disambiguated instead of overwritten. A filename whose slug is a
  near-miss of its own short heading (`15-belestke.md` for `## BELEŠKE`) is repaired from the
  heading, while descriptive names such as `05-glava-2-seks-i-srce.md` are left untouched.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-book-splitter.txt -r requirements-dev.txt
```

Optional, only for `extract_native_toc` with PDF/EPUB sources:
`pip install PyMuPDF ebooklib beautifulsoup4`.

## Use

```bash
# 1. Look at what the book actually contains (no LLM, no API key, no writes).
#    With a single book in data/input/ the argument can be omitted entirely.
python -m book_splitter inspect --candidates

# 2. Split it (writes to data/output/<book>/chapters/)
python -m book_splitter split BOOK.md

# Plans the chapters and validates them, but writes nothing
python -m book_splitter split BOOK.md --dry-run

# Explicit folders / fully local model
python -m book_splitter split BOOK.md --out /tmp/chapters/
python -m book_splitter split BOOK.md \
    --provider lmstudio --model qwen/qwen3.8-27b --base-url http://127.0.0.1:1234/v1
```

Useful flags: `--in-dir`, `--out`, `--max-attempts`, `--front-matter-lines`,
`--no-front-matter`, `--json`, `--quiet`, `--temperature`, `--api-key`.
`python -m book_splitter tools` lists the toolset, and
[scripts/check_layout.py](scripts/check_layout.py) verifies the folder layout and the
installed dependencies.

### Configuration

Copy [.env.example](.env.example) to `.env`, or pass flags. Credential lookup order:
`--api-key` → `SPLITTER_API_KEY` / the provider's own variable → `.env` → `refs:` in
`~/.dsh/.credentials.yaml`. Local endpoints (`localhost`, `127.0.0.1`) may run keyless.

### Output

```
data/output/<book-stem>/chapters/
├── 00-front-matter.md                 # cover/copyright/TOC, everything before chapter 1
├── 01-predgovor.md
├── 02-uvod.md
├── …
├── 15-beleske.md
├── _manifest.json                     # filenames, line ranges, byte counts, SHA-256 per file
└── _agent_trace.json                  # the plan, the validation report, the token usage
```

`_agent_trace.json` is the audit trail: which strategy won, what the agent proposed, and what
the validators said about it.

## Tests

```bash
python -m pytest          # 43 tests, no network, no API key
```

They pin the behaviour that matters: the canonical split set of the sample book is rediscovered,
the typo'd and two-line headings resolve to the right lines, malformed plans (TOC entries, prose
lines, duplicates, out-of-range line numbers) are rejected, slicing is byte-for-byte reversible,
the graph blocks premature writes, a rejected plan loops back with corrections, the dry run
writes nothing, and — asserted directly — **no chapter prose ever appears in the messages sent
to the model**.

## Layout

The package is split by responsibility; each folder can be read on its own.

| Path | Role |
| --- | --- |
| [book_splitter/paths.py](book_splitter/paths.py) | where documents live: `data/input/` in, `data/output/` out |
| [book_splitter/cli.py](book_splitter/cli.py) | `inspect` / `split` / `tools` commands |
| [book_splitter/core/constants.py](book_splitter/core/constants.py) | every guardrail and default in one place |
| [book_splitter/core/settings.py](book_splitter/core/settings.py) | provider and credential resolution |
| [book_splitter/core/errors.py](book_splitter/core/errors.py) | `ToolError`, `ConfigError` |
| [book_splitter/documents/io.py](book_splitter/documents/io.py) | cached source reading |
| [book_splitter/documents/markdown.py](book_splitter/documents/markdown.py) | header/standalone/prose line shapes |
| [book_splitter/documents/text.py](book_splitter/documents/text.py) | folding, similarity, slugs, tool-result caps |
| [book_splitter/analysis/toc.py](book_splitter/analysis/toc.py) | printed-TOC block detection |
| [book_splitter/analysis/headers.py](book_splitter/analysis/headers.py) | `strong` / `noisy` / `absent` verdict |
| [book_splitter/analysis/candidates.py](book_splitter/analysis/candidates.py) | heuristic title candidates |
| [book_splitter/analysis/locate.py](book_splitter/analysis/locate.py) | fuzzy title → line number, heading snapping |
| [book_splitter/slicing/plan.py](book_splitter/slicing/plan.py) | safe filenames and line ranges |
| [book_splitter/slicing/slice.py](book_splitter/slicing/slice.py) | verbatim copy, manifest, dry-run preview |
| [book_splitter/slicing/validation.py](book_splitter/slicing/validation.py) | deterministic map scoring |
| [book_splitter/slicing/verify.py](book_splitter/slicing/verify.py) | SHA-256 proof that nothing was lost |
| [book_splitter/agent/prompts.py](book_splitter/agent/prompts.py) | system prompt and the fallback protocol |
| [book_splitter/agent/llm.py](book_splitter/agent/llm.py) | the chat model with the toolset bound |
| [book_splitter/agent/tools/](book_splitter/agent/tools/__init__.py) | the ten tools, one module per concern, plus the registry |
| [book_splitter/agent/nodes/](book_splitter/agent/nodes/__init__.py) | `scout` / `agent` / `tools` / `validate` / `slice` / `verify` |
| [book_splitter/agent/routing.py](book_splitter/agent/routing.py) | the conditional edges of the graph |
| [book_splitter/agent/graph.py](book_splitter/agent/graph.py) | graph assembly |
| [book_splitter/agent/runner.py](book_splitter/agent/runner.py) | `run_split` and the trace writer |
| [book_splitter/agent/state.py](book_splitter/agent/state.py) | graph state (metadata only, never prose) |
| [book_splitter/agent/result.py](book_splitter/agent/result.py) | the `SplitterResult` dataclass |
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
