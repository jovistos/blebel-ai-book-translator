# Book pipeline — two LangGraph agents, a dictionary funnel, a glossary, reversible chunks and a per-chunk glossary over a Markdown book

The repository holds a six-stage pipeline that turns one long Markdown book into
the raw material of a consistent translation. Stages 1 and 2 each have their **own
specialised agent** with its **own toolset**, and the tools are grouped by the
stage that uses them (plus a generic `shared` group used by both). Stage 3 has no
agent at all: it is a deterministic funnel with an optional NER tagger. Stage 4
has no agent and no tools either — it is a straight, ordered sequence of
LLM calls with a Pydantic-validated answer. Stage 5 has no model at all: it cuts
each chapter into translation-sized chunks with chonkie and proves they
concatenate back byte for byte. Stage 6 uses no LLM either: it tags every chunk
with the same NER backends, then fuzzy-matches each extracted term against the
glossary and writes the terms a chunk mentions next to that chunk.

| Stage | Module | Question it answers | Writes |
| --- | --- | --- | --- |
| 1. **Split** | `book_splitter.agent` | *Where does each chapter start?* | `data/output/<book>/chapters/` |
| 2. **Preprocess** | `book_splitter.preprocess_agent` | *What in this chapter is not part of the book's text?* | `data/output/<book>/preprocessed/` |
| 3. **Dictionary** | `book_splitter.dictionary` | *Which sentences carry a proper noun worth keeping consistent?* | `data/output/<book>/reduced/` + `reduced-unique/` |
| 4. **Glossary** | `book_splitter.glossary` | *How must each of those nouns be written in the target language, once and for all?* | `data/output/<book>/glossary/` |
| 5. **Chunk** | `book_splitter.chunking` | *Where can the chapter be cut without breaking a paragraph or a sentence — and how do we put it back?* | `data/output/<book>/chunks/` |
| 6. **Chunk glossary** | `book_splitter.chunk_glossary` | *Which locked renderings does this chunk actually mention?* | `data/output/<book>/chunk-glossary/` |

Stage 1 turns one long Markdown book into a folder of chapter files
**without ever reading the book's body**. The LLM only ever *routes*: it looks at
line numbers and short title strings produced by deterministic tools, decides
where the chapters start, and submits a plan. Two bounded windows are the only
prose it may see — the front matter (to read a printed table of contents) and the
end of the book (to tell whether the tail is back matter, and split it off) — plus
the clipped results of `web_search` / `fetch_web_page` when the book has no usable
table of contents of its own. A plain Python node then copies the source lines
verbatim into one file per chapter, and a verifier proves that the chapter files
concatenate back to the original **byte for byte**.

Stage 2 repairs each chapter file, **one chapter at a time, sequentially**. It
starts from the chapter's **headers**, never from its prose: page numbers that
became headings, figure captions marked up as headings, headings the PDF
conversion split across two lines, and words a page number was printed inside.
The agent describes its edits in a fixed vocabulary; a deterministic validator
checks every entry against the real file, and a per-line provenance map proves
afterwards that only the declared lines changed.

Stage 3 reduces each chapter, **one chapter at a time, first to last**, to the raw
sentences that are worth mining for a translation glossary: split into sentences,
drop the ones with no capital past the first word, drop the ones under four words,
then keep only the sentences a tagger found a `PER`/`LOC`/`ORG`/`MISC` mention in.
Those survivors go to `reduced/`. A second, free step then keeps only the
sentences that **introduce a noun the book has not used yet** and writes them to
`reduced-unique/`: the same pass, the same tagger, a much shorter list. Both
folders use the chapter's own filename, so all three folders line up one to one
and the two reductions can be compared sentence by sentence. The tagger is chosen
by detected language (CLASSLA for Serbian/Croatian/Slovenian/Bulgarian, stanza for
English) and is used **only as a gatekeeper** — it cannot recognise an invented
term like *Mithril*, so it never builds the dictionary itself.

Stage 4 is the LLM pass stage 3 was built for. It walks the body chapters
**first to last**, skips the front matter and the back matter, and grows one
dictionary for the whole book: for every person, place, organisation, artifact and
recurring term of art it reads out of a chapter, it locks in a single rendering in
the target language, with the context a translator needs to decline or place it.
Every chapter after the first is handed the whole dictionary so far and told those
decisions are final, which is what makes chapter 30 use the name chapter 2 chose.
The dictionary, its markdown twin, a per-chapter manifest and a trace land in
`glossary/`, and the run ends by printing what it cost in tokens.

Stage 5 cuts every chapter, **first to last**, into translation-sized chunks
(1000 tokens by default). It is the only stage that looks at the body as text
instead of as lines: chonkie's recursive chunker prefers paragraph boundaries,
then sentence boundaries, then line boundaries, then whitespace, so a chunk ends
where a reader would pause whenever the requested length allows it. The text
written for a chunk is an exact slice of the chapter — nothing is trimmed or
re-wrapped — so concatenating a chapter's chunks in order gives the chapter back
byte for byte. `chunks/_chunk_manifest.json` records, for every chunk, the chapter
it belongs to, its global order, its character and line spans, its token count and
the boundary kind on each side; `reconstruct_chapter()` and `reconstruct_book()`
are the read side of that record.

Stage 6 joins stages 4 and 5, and its order is the contract. It runs NER over
**every** chunk first — the same CLASSLA/stanza backends stage 3 uses, but here
they *find* terms instead of gating sentences — and collects, per chunk, the
distinct `PER`/`LOC`/`ORG`/`MISC` terms it contains. Only when the last chunk has
been tagged does the second phase begin: every collected term is scored against
`glossary.json` with `rapidfuzz`, and the best candidate is accepted when it
reaches the threshold (95 % by default, and a flag). The accepted terms are
written next to the chunk, under the chunk's own name:
`chunks/00001-01-predgovor-0000.md` → `chunk-glossary/00001-01-predgovor-0000.md.json`.
The file carries the chunk's identity (global order, chapter, character and line
spans) and a `matched_terms` list of `term` / `translation` / `category` rows, so a
translation call can be sent with exactly the renderings that chunk must obey.

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
        ├── preprocessed/                     # stage 2 output — same names, cleaned
        │   ├── 00-front-matter.md
        │   ├── …
        │   ├── _preprocess_manifest.json     # per chapter: operations, fixes, provenance
        │   └── _preprocess_trace.json        # the full per-chapter audit trail
        ├── reduced/                          # stage 3 output — sentences worth mining
        │   ├── 00-front-matter.md
        │   ├── …
        │   ├── _reduced_manifest.json        # per chapter: kept/dropped, entities, nouns
        │   └── _reduced_trace.json           # + examples of what each filter dropped
        ├── reduced-unique/                   # stage 3, second step — first mentions only
        │   ├── 00-front-matter.md
        │   ├── …
        │   └── (same filenames; no manifest of its own — see reduced/)
        ├── glossary/                         # stage 4 output — one book-wide dictionary
        │   ├── glossary.json                 # term / category / locked translation
        │   ├── glossary.md                   # the same, as a review table
        │   ├── _glossary_manifest.json       # per chapter: tokens, attempts, conflicts
        │   └── _glossary_trace.json          # + every entry and raw answer
        ├── chunks/                           # stage 5 output — flat, globally ordered
        │   ├── 00000-00-front-matter-0000.md # <global>-<chapter>-<index>.md
        │   ├── 00001-01-predgovor-0000.md
        │   ├── …
        │   ├── _chunk_manifest.json          # per chunk: chapter, order, spans, tokens, boundary
        │   └── _chunk_trace.json             # + per-chapter logs and the verification report
        └── chunk-glossary/                   # stage 6 output — one file per chunk
            ├── 00000-00-front-matter-0000.md.json  # named after the chunk it describes
            ├── 00001-01-predgovor-0000.md.json
            ├── …
            ├── _chunk_glossary_manifest.json # per chunk: NER counts, matched terms
            └── _chunk_glossary_trace.json    # + every term, candidate and log
```

Stage 2 never overwrites stage 1's files, and stage 3 never overwrites either:
`chapters/` stays exactly as the book came out of the splitter, so every repair
and every reduction can be re-derived and re-checked. `reduced-unique/` is a
subset of `reduced/`, and both are described by `reduced/_reduced_manifest.json`,
so the smaller folder can always be traced back to the sentences it dropped.

Stage 5 writes to its own sibling folder and never edits a chapter. Chunk files are
plain text slices, so a `cat` in filename order rebuilds the book and a `cat` of
one chapter's range rebuilds that chapter — the manifest is the index that tells
you which range that is. The global order is the *first* field of every filename
precisely so a directory listing is already in reading order.

Stage 6 also writes a sibling folder (`chunk-glossary/`) and never edits a chunk.
Its file names repeat the chunk's full name plus `.json`, so the mapping is
unambiguous and a consumer strips `.json` to find the text it describes. The
book-level index is `_chunk_glossary_manifest.json`; the per-chunk files are the
thing a translation prompt reads.

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

Stage 3 then reduces each chapter to the sentences worth mining, reduces those again
to first mentions, and reports what each step removed and what the tagger cost:

```
$ python -m book_splitter dictionary           # on the sample book, 16 chapters
language        : sr (lingua+reflex, confidence 1.00)  backend: classla on cpu
  note          : hr vote overruled: 22 ekavian vs 1 ijekavian form(s) -> sr
chapters        : 16/16 ok
sentences       : 4271 -> 473 kept
dropped         : short=12 no-uppercase=3619 no-entity=167
first mentions  : 473 -> 282 kept (duplicate-noun=191, nouns=526)
entities        : LOC=70, MISC=167, ORG=215, PER=400
ner time        : 3.55 s for 640 sentence(s)  (5.6 ms each, 180/s)
model load      : 2.11 s (classla, cpu)
total time      : 5.75 s  (chapters 3.61 s)
output dir      : …/data/output/Aleksandar_Loven_-_Ljubav_seks_i_vase_srce/reduced
unique dir      : …/data/output/Aleksandar_Loven_-_Ljubav_seks_i_vase_srce/reduced-unique
OK — reduced chapters written.
OK — first-mention chapters written.
```

Per chapter both numbers are printed as the run goes:

```
  [5/16] 04-glava-1-ljubav-je-u-srcu-zivota.md   394 sentences -> 24    kept -> 22    unique  ner  181 ms
```

The two filters that need no tagger do most of the work — of 4 271 sentences only
**640 (15 %) reach the tagger**, and the gate then keeps **473**. `ner time` counts
only the seconds spent *inside* the tagger, so the split, the filters, the gate, the
first-mention pass and the file writes are visibly free by comparison; every timing is
also in `_reduced_manifest.json` (`timing`, plus `ner_seconds`/`elapsed_seconds` per
chapter).

The second reduction is the whole point of writing two folders: on the sample book it
drops **191 of the 473** sentences (**40 %**) and 39 % of the bytes, because the book
mentions `Džordž` in 26 of the 473 kept sentences, `Toni` in 13 and `Meri` in 9 — the
first of each is the one the dictionary needs. Chapter by chapter the effect is very
uneven — `15-beleske.md` loses only 5 of 57 (a bibliography introduces a new name in
almost every entry) while `07-glava-4-gubitak-ljubavi-i-gubitak-nade.md` loses 13 of 24
(one patient's story, told over and over). That difference is exactly what the two
folders let you see before the LLM pass runs.

What survives is dictionary material: over the sample book the top `PER` mentions are
`Džordž`, `Pol`, `Džim`, `Dženi`, `Fridman`, `Toni`, `Moris`, `Alexandar Lowen`, and
the `LOC`/`ORG` lists carry `Seli`, `Beograd`, `London`, `New York`,
`Psychosomatic Medicine`, `The Healing Heart`. Two kinds of noise get through and are
worth knowing about before the LLM pass reads these files: **Markdown headings**
(`GLAVA 1 LJUBAV JE U SRCU ŽIVOTA` is all caps, so CLASSLA tags it `ORG` and it
survives as a line) and **fragments** (single letters like `A` reaching the gate as
`MISC`, mostly from citations). The first-mention filter leaves both alone — it can
only drop a sentence whose nouns were *all* met before.

Stage 4 then turns that list into the dictionary — one call per body chapter, the
front matter skipped, the dictionary carried forward. Because the default pair is
`en → sr` and the sample book above is Serbian, the run below is the English novel in
`data/input/` (`--limit 2` stops it after chapter 1, which is what a first, priced run
looks like):

```
$ python -m book_splitter glossary --out data/output/Dont-You-Forget-About-Me---Alexandra-Potter/chapters --limit 2
language=en -> sr  model=deepseek/deepseek-flash  method=auto  attempts<=3
skip        : front matter, back matter
  [1/1] 01-chapter-1.md                                33 new    0 locked   0 conflict     8,784 tok  1 call(s)

glossary        : en -> sr  deepseek/deepseek-flash
  note          : the provider rejected native structured output; the run used JSON mode
chapters        : 1/1 body chapter(s) ok  skipped: 1
entries         : 33  (artifact=6, character=15, location=8, organization=3, terminology=1)
tokens used     : 8,784 total (2,541 in, 6,243 out) over 1 chapter(s), 1 call(s)
time            : 34.6 s  (model 34.5 s)  34.5 s/chapter
OK — 33 entries locked in.
```

What the model wrote for one of them, verbatim:

```json
{
  "term": "Jenny Hamilton-Proctor",
  "category": "character",
  "translation": "Dženi Hamilton-Proktor"
}
```

That is the whole entry: the source name, its category, and the one Serbian
rendering the translator must reuse. Nothing else — an entry is a lookup, and every
field in it is paid for on every chapter of the book. The DeepSeek
endpoint advertises a schema-capable response format and then rejects it with a 400,
so `method=auto` stops asking for native structured output after that refusal — which
is what the `note` line reports — and the run continues in JSON mode, still validated
by Pydantic.

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
   `CHAPTER/PART/GLAVA/DEO/POGLAVLJE` keywords, for books that lost all markup. That list is
   capped and, in a book with no markup at all, mostly prose noise; `find_lines_matching` lets
   the agent write its own regex (`^Chapter \d+$`) and get every marker with its line number
   in one call, instead of the first few hundred title-looking lines.
5. **Web lookup** (`web_search` + `fetch_web_page`) — for the one question no amount of local
   evidence can answer: *how many chapters does this book have, and what are they called?*
   When the printed TOC is gone **and** the headers are a mix of chapter titles and junk, the
   file only offers candidates. The published contents list is online, so the agent looks it
   up (searching the *original* title of a translation), reads it from a publisher, bookseller,
   library or Open Library record, and uses it as ground truth — both to reject the junk headers
   and to recover a chapter whose heading lost its number (`1 …`, `2 …`, an unnumbered title,
   `4 …`, `7 …`). See [When the book cannot identify its own chapters](#when-the-book-cannot-identify-its-own-chapters).
6. **Prose chapter starts** (`role: "prose-chapter"`) — for a chapter whose heading the
   conversion dropped *entirely*, so there is no line to point at. The agent proposes the
   chapter's first paragraph instead, with the evidence that makes it a chapter opening. See
   [When the heading itself is gone](#when-the-heading-itself-is-gone).

### When the book cannot identify its own chapters

The first book this route was written for,
[data/input/Irvin_D._Jalom_-_Gledanje_u_sunce.md](data/input/Irvin_D._Jalom_-_Gledanje_u_sunce.md)
(Irvin D. Yalom, *Staring at the Sun*, Serbian translation) has **no table of contents at all**
and 115 `##` headers of which nine are the real chapter/section openings. The others are a
second title page, the translator's blurb, running heads and scanner watermarks (`## zdls/bojana888`,
`## CW &amp; BU`, `## .`). Worse, the conversion dropped the *number* from three of the seven
chapter headings, so the file reads `## 1 SMRTNA RANA`, `## 2 PREPOZNAVANJE STRAHA OD SMRTI`,
`## ISKUSTVO KOJE OTVARA OČI`, `## 4 MOĆ IDEJA`, … — chapters 3, 5 and 6 are there, but nothing
inside the file says so.

The deterministic scout cannot tell either: it scores those headers `strong`, because they *do*
look like titles. `web_search("Staring at the Sun Irvin Yalom table of contents")` followed by
`fetch_web_page` on the publisher's or Google Books' contents page returns the seven English
titles (`The Mortal Wound`, `Recognizing Death Anxiety`, `The Awakening Experience`, `The Power
of Ideas`, `Overcoming Death Terror Through Connection`, `Death Awareness: A Memoir`,
`Addressing Death Anxiety: Advice for Therapists`), which is exactly the spine — the unnumbered
Serbian titles are the missing chapters 3, 5 and 6.

```
$ python -m book_splitter split Irvin_D._Jalom_-_Gledanje_u_sunce.md
strategy        : toc_titles
llm calls       : 6  tokens: 136624
chapters        : 9
byte-for-byte   : True
lines           : 1950/1950
files:
  00-front-matter.md                                       lines     1-22
  01-predgovor-i-zahvalnost.md                             lines    23-86
  02-1-smrtna-rana.md                                      lines    87-142
  03-2-prepoznavanje-straha-od-smrti.md                    lines   143-266
  04-iskustvo-koje-otvara-oci.md                           lines   267-630     <- chapter 3, number lost
  05-4-moc-ideja.md                                        lines   631-927
  06-prevazilazenje-straha-od-smrti-kroz-povezanost.md     lines   928-1147    <- chapter 5, number lost
  07-svest-o-smrti-memoari.md                              lines  1148-1489    <- chapter 6, number lost
  08-7-pristup-strahu-od-smrti-saveti-terapeutima.md       lines  1490-1927
  09-pogovor.md                                            lines  1928-1947
  10-back-matter.md            (back-matter)               lines  1948-1950    <- ## O AUTORU
OK — every source line is accounted for.
```

The agent ran `web_search` on turn 1, read the Google Books contents page on turn 2, and the
whole 400 KB book was cut into 11 byte-exact files in 6 LLM turns. The only prose it ever saw
was the bounded front-matter window, the tail window and a handful of `inspect_line_window`
excerpts — never a chapter's text.

### When the heading itself is gone

The Yalom case still had a line to point at: an unnumbered title. A worse conversion drops the
chapter heading *entirely* — no `##`, no empty header, no double blank line, no page number. The
chapter then starts on a plain paragraph, and a header-only map silently merges it into the
chapter before it. That is the case the second role exists for:

```json
{"line_number": 227, "filename": "03-2.md", "title": "2",
 "role": "prose-chapter",
 "evidence": "published contents list: chapter 2 opens with Louisa waking before six, which is
              this line; chapter 1 ends at the body in the alley. No `## 2.` survives."}
```

The role is deliberately narrow, and validation enforces every rule:

- the line must **open a paragraph** — the line above it is blank. A line in the middle of a
  paragraph is rejected, so the role can never cut a sentence in half;
- it must be text: not blank, not a lone page number, at least two words and a dozen characters;
- **`evidence` is required** (`reason`, `justification`, `source` and `provenance` are accepted
  spellings). It is quoted into the validation result and the trace, so a map always says *why*
  a paragraph was called a chapter opening;
- it may not sit inside the printed TOC block, and it counts as a chapter like any other.

Validation cannot check the *content* of that claim — a misplaced prose cut moves text between
two files without losing a byte. Both the validation result and the run log therefore carry a
warning whenever a map rests on prose starts, and the agent is told to say in its `notes` how
many chapters do. `find_lines_matching` reports the flags that make a candidate usable
(`standalone` + `title_like` for the old shape, `prose_start_issue: null` for this one), so the
model can filter instead of guess.

[data/input/Kenneth_M._Cameron_-_Zima_u_hotelu_smrti.md](data/input/Kenneth_M._Cameron_-_Zima_u_hotelu_smrti.md)
(Kenneth M. Cameron, *Winter at Death's Hotel*) is the book this was built for: **15 chapters, of
which 8 headings were dropped** — only `## 1.`, `## 3.`, `## 4.` and `## 10.`–`## 13.` survive.
The agent could not do it (it exhausted its turns and wrote nothing: every candidate it could
point at was a heading, and the headings it needed are not in the file), so the recovered map is
kept as a first-class input and replayed with
[`--map`](#replaying-a-map-without-the-model):

```bash
python -m book_splitter split Kenneth_M._Cameron_-_Zima_u_hotelu_smrti.md \
    --map data/output/Kenneth_M._Cameron_-_Zima_u_hotelu_smrti/chapters/_recovered_map.json
strategy        : preset_map
llm calls       : 0  tokens: 0
chapters        : 16
byte-for-byte   : True
lines           : 7178/7178
```

### Back matter is split off, not left inside the last chapter

A flat splitter cuts at chapter starts, so everything after the last one — the author
biography, the publisher's catalogue of the author's other books, the acknowledgements, a
teaser excerpt, the chapter list printed a second time — would land inside the last chapter
file and be translated as if it were the book.

`inspect_book_tail` looks at the very end of the book and answers *is there back matter, and
where does it start*. It is deliberately read through a **window the agent widens**:

1. the agent calls it with the default short window (`chunk_lines=40`);
2. if the result says `escalate: true`, the window was not decisive — it either cut through
   the block or showed no back matter while lines remained above — so the agent calls again
   with `chunk_lines=next_chunk_lines` (each step doubles the window, capped at 300 lines);
3. once `back_matter.boundary_visible` is true, the tool returns a ready plan entry with
   `role: "back-matter"`, which the agent appends as the **last** entry of its map.

Evidence is shape-based and language-agnostic where it can be: publisher keywords (`Also
available`, `About the author`, `ISBN`, `All rights reserved`, `Visit:`/`Follow:`), series
headings (`In the … Series:`), ISBN bullets, author-bio sentences, and **repeated titles** —
lines that occur more than once in the file, which is exactly what a re-printed table of
contents looks like. A trailing run counts as back matter when it holds at least two evidence
lines, or one that is structurally unambiguous. A prose line stops the run, so a stray keyword
inside the story cannot drag the boundary down into the plot.

`role: "back-matter"` is the one split point that does **not** have to look like a chapter
title, because an author biography is a paragraph and a catalogue opens with a book list.
Validation accepts it only as the last entry, only with that role, and only while the block it
leaves behind is plausible: at most 1 000 lines and at most a quarter of the book. The slicer
then plans it with `kind: "back-matter"`, so it gets its own `NN-back-matter.md`, stays out of
the chapter count, and still takes part in the byte-for-byte verification like every other file.

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

### Stage 3 — dictionary (`book_splitter.dictionary`), run once **per chapter**

No LLM, no agent, no toolset: a fixed funnel, cheapest test first, so the
expensive step only ever sees a fraction of the text.

```
chapter file
  │
split                      deterministic sentence split (abbreviations, initials,
  │                        ellipses, `5. maja 1990.`, dialogue + attribution)
  │
uppercase filter           drop a sentence whose only capital is its first word —
  │                        sentence case is not evidence of a proper noun
  │
4-word floor               drop a sentence too short for context
  │
language routing           `--language auto` samples five chapters with lingua-py
  │                        and picks CLASSLA (sr/hr/sl/bg) or stanza (en)
  │
NER gate                   keep the sentence only if it carries PER / LOC / ORG / MISC
  │
reduced/<same-filename>.md one raw sentence per line, no NER markup
  │
noun index                 the nouns of every kept sentence (case-folded surface
  │                        form, entity type ignored), carried across the whole book
  │
first-mention filter       keep the sentence only if at least one of its nouns has
  │                        not been seen in an earlier sentence — `reduced/` says
  │                        "Marko" nine hundred times, a dictionary needs it once
  │
reduced-unique/<same-filename>.md   the same shape, the first mentions only
```

On the sample book the two free filters alone leave **15 %** of the sentences for the
tagger — 640 of 4 271 — which is what makes the NER pass cheap (3.3 s on CPU for the
whole book). Earlier measurements on the two bigger Serbian novels:
870 of 4 002 for *Gledanje u sunce*, 2 938 of 14 180 for *Zima u hotelu smrti*.

The last step costs nothing on top of that: it reuses the entity mentions the gate
already found, so `reduced-unique/` comes out of the same pass and the same tagger
run. It is **book-wide and ordered** — chapters are visited first to last and
chapter 3's `Marko` suppresses chapter 9's — and its rule is *"at least one new
noun"*, not *"no repeated noun"*, so a first mention that happens to share a
sentence with a known name is not thrown away. Both folders are written by default
and switched independently: `--no-save-unique` keeps only `reduced/`,
`--unique-only` keeps only `reduced-unique/`.

**Serbian and Croatian are not reliably separable by a language detector**, and that
matters here because the choice picks the CLASSLA model: `lingua-py` labels an ekavian
Serbian novel `hr` with confidence 1.0. So after the vote the sampled text is checked
for the reflex forms that actually differ between the two standards — `vreme`/`dete`/
`reč` against `vrijeme`/`dijete`/`riječ` — and an `hr`/`bs` vote that is overwhelmingly
ekavian is read as `sr` (and the reverse). The tie-breaker needs at least 5 forms and a
2:1 margin, so a genuinely Croatian or Bosnian book keeps its own language; the run
prints `lingua+reflex` and the counts when it fires. `--language sr` bypasses all of
this.

The tagger is a **gatekeeper, never the dictionary builder**. CLASSLA's NER is a
traditional four-type model: it has no way to know that `Mithril` is a substance
or that `Hyperdrive` is a device, and it will tag them `MISC` or not at all. What
it is good at is throwing away the sentences that cannot teach a dictionary
anything — generic dialogue, pure action, generic description — and everything it
keeps is a whole raw sentence, which is what a later LLM pass needs to decide what
the recurring nouns actually are.

Two shapes survive the gate that are not dictionary evidence, visible in the sample
book's own output: Markdown **headings** (all caps, so CLASSLA reads `GLAVA 1 LJUBAV JE
U SRCU ŽIVOTA` as an organisation) and citation **fragments** (single letters such as
`A` tagged `MISC`). Both are cheap to filter as a next step, and both are visible in
the reduced files rather than hidden, which is the point of writing them out. The
first-mention filter does not remove them — a heading is a "new noun" the first time
it appears — and it cannot: it only ever drops a sentence whose nouns were all met
before.

The stage stops there, on purpose: two reductions first, so the funnel can be
compared folder by folder and inspected sentence by sentence before stage 4 — the
dictionary step that consumes it — runs over the result.

### Stage 4 — glossary (`book_splitter.glossary`), run once **per chapter, in order**

This is the LLM pass stage 3 was built for. One call per chapter, first to last,
each call handed the dictionary the earlier chapters produced:

```
reduced-unique/01-chapter-1.md  call 1: dictionary empty
   │                                   -> extract every entity, lock a rendering
   │                                      for each: term, category, translation
   │                                                             (Pydantic-validated)
   ▼
dictionary so far (chapters 1..n)
   │
reduced-unique/02-chapter-2.md  call 2: prompt = dictionary so far + this chapter's
   │                                   sentences; the model is told the locked
   │                                   renderings are final and to return only what
   │                                   is new
   ▼
… every body chapter, in reading order …
   │
glossary/glossary.json            the dictionary: one entry per entity, with the
glossary/glossary.md              chapter it came from
glossary/_glossary_manifest.json  per chapter: tokens, attempts, entries added
glossary/_glossary_trace.json     + the entries added and the conflicts recorded
```

**Front matter and back matter are skipped by default.** A copyright page, a
printed table of contents and an "about the author" note are full of names — a
publisher, a city, an acknowledgement — that must not be locked into the book's
terminology. Stage 1's manifest already records each file's `kind`, so the
decision is *read* (`front-matter` / `chapter` / `back-matter`) rather than
guessed; a filename fallback (`00-front-matter.md`, `*-back-matter.md`) covers a
folder whose manifest is missing. `--include-front-matter` /
`--include-back-matter` put either block back, and the skipped files still appear
in the manifest with the reason, so a dictionary built without them is auditable.

**What one entry holds** — three fields, validated with Pydantic
(`book_splitter/glossary/models.py`):

| Field | Meaning |
| --- | --- |
| `term` | the entity as the book names it, in the source language, in its canonical form |
| `category` | `character`, `location`, `organization`, `artifact` or `terminology` |
| `translation` | the **locked** rendering in the target language, in its citation form |

The five categories are the ones that need *different treatment* in translation: a
person keeps a name shape, a real place takes the established target-language
exonym, a descriptive organisation name is translated, a unique object is
transliterated, and a domain term is defined once and then reused verbatim.

The prompt asks for **the citation form** and says so explicitly for the inflected
targets it knows (`TARGET_NOTES` in `prompts.py`): Serbian gets the nominative
singular, so the translator can decline `Poter` to `Potera`, `Poteru`, `Poterom`
itself. There is deliberately no description field — an entry is the pair the
translator looks up, and a per-entry commentary costs output tokens on every
chapter while telling the translator little it cannot read from the sentence it is
translating. (An older dictionary that still carries `brief_context` loads fine;
the extra field is ignored.)

**The prompts know they are translating a book.** The system prompt is the same
for every chapter and opens with it: the model is not summarising, it is the
terminology curator for a book *being translated* from one named language into
another, its answer is what the translator is bound to, and the same name must not
appear two ways. The user prompt is the variable half — the dictionary so far,
then the chapter's sentences — so a provider that caches a prefix pays for the
instructions once and the growing dictionary never invalidates them.

**What the next chapter is re-sent.** Only the locked pair — one line per entry,
`Term → Translation`. The model only has to know *that* a term is taken and *which*
rendering it owns, and because this block is re-sent with every chapter, every
extra field would be paid for once per chapter of the book. `glossary.json` keeps
the category and the chapter provenance for the translator and the reviewer;
`glossary.md` shows the same three fields as a table.

**Structured output with retries.** The answer must be the JSON object
`{"entries": [ … ]}`. The extractor asks for it through the provider's native
structured-output mode and validates it as a Pydantic model; a bare array or an
invented wrapper is unwrapped rather than punished, but a missing field is a
failure. On failure the call is retried up to `--max-attempts` (default 3), each
retry carrying a note that quotes the previous failure, and the **last attempt
falls back to plain JSON mode** (code fences and "here is the JSON" prose are
stripped). Every attempt's tokens are counted, including the ones that were
rejected — the run's token report is the bill, not the successes.

**The dictionary's own rules.** The first rendering of a term wins, always. A
later chapter that returns a term already locked in with the *same* rendering is
deduplicated and recorded as a repeat; returning it with a *different* rendering
is recorded as a **conflict**, the locked rendering stands, and the disagreement
lands in the manifest — the text already translated can never be made to disagree
with itself. Terms are keyed case-folded and space-collapsed (`Potter` =
`potter `), so a capitalisation difference is not a second entry, while an
inflection (`Potter` / `Potera`) is not an entity of its own: the entry stores the
citation form, and the translator declines it.

**Cost control.** The dictionary is re-sent with every chapter, so on a long book
its length dominates the input bill. Two flags exist for that:
`--max-context-entries` caps how many entries are rendered in full (the rest are
still named in a single "already locked" line so the model does not re-invent
them), and `--max-chapter-chars` splits an unusually long chapter into parts,
carrying the dictionary from part to part exactly as it is carried between
chapters. `--limit N` prices the first N chapters before committing to the book,
and `--dry-run` asks the model about everything and writes nothing.

**Resuming.** `--resume` reloads `glossary.json` and the manifest, keeps the
locked renderings and the tokens already spent, and starts at the first body
chapter that has no successful record. A book that died on chapter 30 costs
chapters 30..N, not the book — and a fully processed book exits without resolving
a model or an API key at all. `run_pipeline.py --skip-if-present` passes the same
flag through, so a run that stopped in the pipeline finishes the same way.

**In the pipeline it is the second half of the `dictionary` phase.** Running
`python scripts/run_pipeline.py BOOK --skip-if-present` reduces the chapters and
then builds the glossary in the same phase — no separate flag. `--no-glossary`
keeps the phase deterministic (reduction only, no key needed), and `--dry-run`
skips the glossary half. While it runs, a live bar shows the chapter in flight,
elapsed time and an ETA; a terminal redraws it in place, a piped or logged run
gets a heartbeat line every ten seconds, and each finished chapter prints a
durable line with its new terms, tokens and seconds::

      ████████░░░░░░░░░░░░░░  8/43 · 6m 12s elapsed · ETA ~27m · now 09-chapter-9.md (14s)
    [8/43] 08-chapter-8.md                        2 new ·  1 locked ·  0 conflict ·   9,214 tok ·  68.4s

A resumed run also **refuses to write an empty dictionary over a non-empty one**:
if the manifest records entries and `glossary.json` holds none (deleted, truncated
or hand-edited), the run stops and says so instead of replacing the renderings a
translation already depends on. That is the one file in this pipeline whose loss
cannot be reconstructed from the book.

### Stage 5 — chunking (`book_splitter.chunking`), no model at all

A chapter file is fine for a reader and far too big for a translation call: a
100-page chapter blows through a context window, and cutting it at a fixed
character count breaks words, sentences and paragraphs. Stage 5 turns each
chapter into files a translator can consume one at a time:

```
chapters/01-predgovor.md
   │  chonkie RecursiveChunker, preference order:
   │      paragraph  →  sentence  →  line  →  whitespace  →  hard cut
   ▼
chunks/00000-01-predgovor-0000.md   exact slice [start_char:end_char)
chunks/00001-01-predgovor-0001.md
…
chunks/_chunk_manifest.json         chapter, global order, spans, tokens, boundaries
```

The length is measured in **tokens**, not characters: `--chunk-size` defaults to
`1000` and is a hard maximum (a chunk that a single unsplittable token pushed over
the limit is split by the stage itself, at a source-slice boundary). The default
tokenizer is `cl100k_base`, built by `tiktoken`; when `tiktoken` (or the requested
tokenizer) cannot be loaded — no package, no vocabulary cache, an offline machine
— the stage falls back to a local four-characters-per-token estimate and records
the fallback and its reason in the manifest. It never downloads a model in the
middle of a run and never fails a book because a vocabulary did not load.

**Reversibility is the contract.** The text written for a chunk is
`chapter_text[start_char:end_char]`, with no trimming, no re-wrapping and no
normalisation. The run verifies that every file is exactly that slice and that
concatenating a chapter's chunks in `chunk_index` order reproduces the chapter
byte for byte; `verification.chapters_reconstruct_byte_for_byte` and
`verification.book_reconstructs_byte_for_byte` in the manifest are that proof.
The read side is part of the package:

```python
from book_splitter.chunking import reconstruct_book, reconstruct_chapter

reconstruct_book("data/output/<book>/chunks")                    # == the whole book
reconstruct_chapter("data/output/<book>/chunks", "01-predgovor.md")
```

**What the manifest keeps.** One record per chunk (`files`) and one per chapter
(`chapters`):

| Field | Meaning |
| --- | --- |
| `global_index` | 0-based position in the book; also the first part of the filename |
| `chapter_filename` / `chapter_id` / `chapter_index` | which chapter the chunk belongs to |
| `chunk_index` | 0-based position inside that chapter |
| `start_char` / `end_char` | the exact slice offsets in the chapter text |
| `start_line` / `end_line` | the same span in chapter line numbers, for a human |
| `token_count` | measured by the resolved tokenizer, the number the limit was enforced with |
| `boundary_before` / `boundary_after` | `chapter-start`, `chapter-end`, `paragraph`, `sentence`, `line`, `word` or `hard` |
| `sha256` / `bytes` / `first_line` / `last_line` | integrity and a quick human preview |

`boundary_counts` and `word_breaking_chunks` are reported at the top level: with
the default 1000 tokens a normal prose book cuts at paragraph boundaries and the
word-breaking count is zero; a very small `--chunk-size` will necessarily reach
sentence or word boundaries, and the manifest says so instead of pretending.

**Resuming and pricing.** `--skip-if-present` reuses a chapter whose recorded
chunk files still concatenate to its *current* text and whose manifest was written
with the same chunk length and an exact (not fallback) tokenizer. When every
chapter can be reused, the stage returns before loading chonkie or a tokenizer at
all. `--dry-run` computes and verifies every cut but writes no chunk, manifest or
trace — the cheap way to price a chunk length before a full run. A previous run
with a different `--chunk-size` is deliberately not reused: the old files are valid
chunks, but not chunks of the length that was just requested.

### Stage 6 — chunk glossary (`book_splitter.chunk_glossary`), two phases and no LLM

Stage 4 locked one rendering per entity for the whole book. Stage 5 cut the book
into pieces a model can translate. The missing question is the one a translator
asks before sending chunk 137: *which of those locked renderings does this chunk
actually mention?* Stage 6 answers it, and it refuses to answer it one chunk at a
time.

```
chunks/00000-01-predgovor-0000.md ─┐
chunks/00001-01-predgovor-0001.md ─┤  PHASE 1 — NER over every chunk
…                                  ├─ stage 3's CLASSLA/stanza tagger
chunks/00042-08-glav…-0013.md ─────┘  → per-chunk distinct PER/LOC/ORG/MISC terms
                                      (no glossary lookup yet)
                                              │
                                              ▼  only after the last chunk is tagged
                                      PHASE 2 — fuzzy retrieval
                                      rapidfuzz scores every term against
                                      glossary.json; ≥ threshold wins
                                              │
                                              ▼
chunk-glossary/00000-01-predgovor-0000.md.json
chunk-glossary/00001-01-predgovor-0001.md.json
…
chunk-glossary/_chunk_glossary_manifest.json
chunk-glossary/_chunk_glossary_trace.json
```

**The order is the contract.** The NER pass finishes for **all** chunks before the
first glossary lookup. That is not an optimisation; it is what makes the stage one
complete picture instead of a chunk-at-a-time guess, and the tests pin it by
recording the order of the injected tagger and matcher. The run reports the two
phases separately (`phases[].name` is `ner` then `match`), with their own seconds
and counts.

**The tagger is stage 3's.** `book_splitter.dictionary.ner` builds the backend for
the detected language — CLASSLA for Serbian/Croatian/Slovenian/Bulgarian, stanza
for English — and stage 6 calls the same `entities()` method over each chunk's
full text. The language is detected from a sample of the chunks (the same
`lingua-py` sampler and ekavian/ijekavian tie-breaker as stage 3), or pinned with
`--language sr`; `--gpu N` / `--cpu` select the device exactly as they do for the
dictionary stage. The tagger is used to **find** terms here, not to gate
sentences, so dates and other noise can be dropped with `--drop-temporal-only` and
single-character fragments with `--min-term-chars` (default 2).

**The retrieval is rapidfuzz.** Each distinct term is scored against every entry
of `glossary.json` on rapidfuzz's 0..100 scale. The best candidate is the match
when its score reaches `--match-threshold` (default **95 %**). The scorer is a
flag because the right one depends on the book:

| `--scorer` | When it helps |
| --- | --- |
| `ratio` (default) | near-exact names: the tagger and the glossary agree except for a typo or punctuation |
| `token_sort_ratio` | word order moves (`York New` vs `New York`) |
| `token_set_ratio` | one side has extra words (`Džordž Fridman` vs `Džordž`) |
| `wratio` | a mention is a fragment of a longer canonical name |
| `partial_ratio` | the term is a substring of the glossary entry |
| `qratio` | rapidfuzz's faster, slightly looser ratio |

`--match-candidates N` keeps the top N candidates per term: the best is the match,
the rest are recorded as `alternatives`; a term below the threshold is kept with
its `closest` candidate and score, so a translator can see *why* nothing was
attached. A term that two NER mentions resolve to the same glossary entry is
merged in the chunk's `matched_terms` (one row per dictionary entry, mentions
summed), because a repeated entry in a prompt is paid for and says nothing new.

**What a per-chunk file keeps.** The chunk's identity — `filename`, `path`,
`global_index`, `chunk_index`, `chapter_filename`, character and line spans,
`token_count`, `sha256` — plus:

| Field | Meaning |
| --- | --- |
| `terms` | every distinct NER term in the chunk, with `types`, `mentions`, `matched`, `score`, the matched `glossary` entry, the `closest` near miss, and `alternatives` |
| `matched_terms` | the accepted retrieval: `term`, `glossary_term`, `category`, `translation`, `score`, `mentions`, and the chapter that introduced the entry |
| `counts` | `entities`, `terms`, `matched`, `unmatched` |
| `threshold` / `scorer` / `language` / `backend` | the settings that produced the file |

`matched_terms` is the list a translation prompt consumes; the rest is provenance.
The book-level `_chunk_glossary_manifest.json` repeats the counts and accepted
terms per chunk, with the two phases and their timings, so a consumer can find the
chunks that need a term without opening every file. The trace adds every term and
near miss, and each chunk's log.

**Running it.** The command derives both inputs from the book folder:

```bash
python -m book_splitter chunk-glossary mybook
python -m book_splitter chunk-glossary mybook --match-threshold 90
python -m book_splitter chunk-glossary mybook --scorer token_sort_ratio
python -m book_splitter chunk-glossary mybook --limit 3        # a quick look
python -m book_splitter chunk-glossary mybook --dry-run        # both phases, no writes
python -m book_splitter chunk-glossary --chunks-in /tmp/chunks/ \
    --glossary-in /tmp/glossary/ --chunk-glossary-out /tmp/chunk-glossary/
```

`--chunks-in` accepts the chunks folder, the book folder, or
`_chunk_manifest.json`; `--glossary-in` accepts the glossary folder or
`glossary.json`. Without them the stage reads `chunks/` and
`glossary/glossary.json` beside the book's `chapters/` folder. `rapidfuzz` is the
only new dependency, and it is imported lazily — a missing wheel is one
`pip install` line, not a traceback.

On the English novel in `data/input/` (`--limit 2` tags the first two chunks):

```
$ python -m book_splitter chunk-glossary --out …/chapters --language en --cpu --limit 2
language        : en (explicit, confidence 1.00)  backend: stanza on cpu
chunks          : 2/2 ok
entities        : 38 mention(s) -> 31 distinct term(s)
glossary        : 412 entries loaded
matched         : 20 term(s) in 2 chunk(s)  (threshold 95.0%, scorer ratio,
                  unique dictionary entries 18)
unmatched       : 11 term(s)
ner             : 1.45 s for 2 chunk(s)
match           : 2 ms for 2 chunk(s)
total time      : 6.22 s
output dir      : …/chunk-glossary
OK — 20 relevant term(s) saved, one file per chunk under the chunk's own name.
```

`Facebook` matched its locked `Fejsbuk` at 100; a heading like `Chapter 1` stayed
unmatched with `Catherine` as its closest candidate (55.56), which is exactly the
noise `--min-term-chars` and `--drop-temporal-only` exist to trim if a book needs
it.

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
| `extract_front_matter` | First *N* lines only (one of the two prose tools) | |
| `locate_toc_titles_in_file` | Fuzzy title → line number, skips the TOC block, snaps sub-headings | |
| `extract_heuristic_candidates` | Title-looking lines when the markup is gone | |
| `find_lines_matching` | Every line matching a regex the agent writes, flagged as a usable split point or not: `standalone`+`title_like` for a heading/title line, `prose_start_issue: null` for a chapter whose heading is gone | ✔ |
| `inspect_book_tail` | The end of the book: is there back matter, and where does it start — read through a window the agent widens | ✔ |
| `web_search` | The published contents list of a book that lost its own TOC: one bounded query against a public search engine | ✔ |
| `fetch_web_page` | Reads one public `http(s)` page as block-separated text, so a contents list stays one title per line | ✔ |
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

### Stage 3 — no tools at all

`dictionary` binds no tools because it calls no model. Its whole vocabulary is
command-line flags, and each one maps to exactly one step of the funnel:

| Flag | Effect |
| --- | --- |
| `--language auto\|sr\|hr\|sl\|bg\|en` | route the book to CLASSLA or stanza (default: sample a few chapters and detect) |
| `--gpu N` / `--cpu` | which physical GPU the tagger may use (default: GPU 1, via `CUDA_VISIBLE_DEVICES`); the summary reports the device actually used |
| `--min-words N` | the length floor (default: 4) |
| `--no-uppercase-filter` | keep sentences whose only capital is the first word |
| `--keep-types PER,LOC,…` | which entity types save a sentence (default: all four) |
| `--drop-temporal-only` | also drop a sentence whose only entity is a date/time |
| `--skip-if-present` | reuse reduced chapters that already exist; loads no tagger when nothing is left |
| `--no-save-reduced` | run the funnel and write no reduced chapters (the manifest still lands) |
| `--no-save-unique` | write no first-mention chapters, only the reduced ones |
| `--unique-only` | write only the first-mention chapters (implies `--no-save-reduced`) |
| `--unique-out DIR` | where the first-mention chapters go (default: a sibling of `reduced/`) |
| `--limit N` | reduce only the first N chapters, while tuning the filters |
| `--dry-run` | reduce everything, write nothing |

### Stage 4 — no tools either

`glossary` calls a model, but not as an agent: the sequence of calls is fixed
(one per body chapter, first to last), the answer is a schema rather than a
decision about which tool to use next, and there is nothing for the model to look
up. Its whole vocabulary is command-line flags:

| Flag | Effect |
| --- | --- |
| `--source-language/-s CODE` | the language the book is written in (default: `en`) |
| `--target-language/-t CODE` | the language being translated into (default: `sr`); both are named in the prompt |
| `--sentences unique\|reduced` | mine the first mentions (default) or every entity-bearing sentence |
| `--sentences-in DIR` | read the sentences from this folder instead of deriving it from `--out` |
| `--include-front-matter` / `--include-back-matter` | mine a block that is skipped by default |
| `--max-attempts N` | model calls allowed per chapter before it is reported as failed (default: 3) |
| `--extraction-method auto\|structured\|json` | `auto` (default) asks for native structured output and falls back to JSON mode on the last attempt |
| `--max-tokens N` | response cap per call (default: 8192); `0` leaves it to the provider |
| `--max-chapter-chars N` | split a chapter larger than this into parts, carrying the dictionary across them (default: 24000) |
| `--max-context-entries N` | cap the dictionary sent with each chapter (`0` = send all; a cap trades a little consistency for input tokens) |
| `--resume` (alias `--skip-if-present`) | reuse the glossary and the recorded chapters, and start at the first one missing |
| `--book-title TITLE` | the title the prompt uses (default: the source filename from stage 1) |
| `--limit N` | process only the first N chapters, to price the run |
| `--dry-run` | ask about every chapter, write nothing |
| `--provider` / `--model` / `--base-url` / `--api-key` / `--temperature` | the same model flags as stages 1 and 2; the default provider is DeepSeek |

### Stages 5 and 6 — no tools either

`chunk` binds no tools because it calls no model at all: its input is the chapter
text and its vocabulary is `--chunk-size`, `--chunk-tokenizer`, `--limit`,
`--dry-run` and `--skip-if-present`. `chunk-glossary` also binds no tools: it runs
a fixed two-phase pipeline (NER over every chunk, then rapidfuzz retrieval) whose
whole vocabulary is command-line flags — `--chunks-in`, `--glossary-in`,
`--chunk-glossary-out`, `--match-threshold`, `--scorer`, `--candidates`,
`--min-term-chars`, `--drop-temporal-only`, `--language`, `--gpu`/`--cpu`,
`--limit`, `--no-manifest` and `--dry-run`.

## Guardrails

- **No body text in the prompt.** Tools return line numbers, headers, candidate titles and
  ≤90-character excerpts. Stage 2 adds evidence flags, damage fragments and `inspect_line_window`
  (≤6 lines, ≤600 characters per call); every tool result is capped at 24 000 characters and none
  of the stage-2 tools can return a chapter's prose. Stage 1 has exactly two bounded prose
  windows: `extract_front_matter` (the front of the book, ≤1 200 lines / 20 000 characters) and
  `inspect_book_tail` (the end of the book, ≤300 lines / 12 000 characters of window text). The
  tail report is trimmed from the oldest line until it fits the tool cap, so it is never cut
  mid-JSON, and the line it proposes is always kept in view.
- **The network is bounded, and outside the book.** `web_search` / `fetch_web_page` are the only
  tools that leave the machine, and only stage 1 binds them (stage 2 cannot reach them). A search
  returns at most 12 results (`6` by default), a fetch downloads at most 2 MB, clips the page to
  ≤12 000 characters (`6 000` by default) with a 200-character floor, and every request has a
  12-second timeout. URLs must be public `http(s)`: `file:`, `localhost` and private/reserved
  addresses are refused before a socket is opened. Both tools are provider-fallbacks
  (Brave → DuckDuckGo) and return `{"error", "attempts", "hint"}` rather than raising, so a
  blocked or rate-limited engine degrades instead of failing the run. Every payload is labelled
  **external, untrusted data** and the prompt tells the model to use it only to identify chapter
  titles and never to follow instructions found in it.
- **No hallucinated line numbers.** Every proposed split point is checked against the real file:
  in range, strictly increasing, unique, actually a heading or a standalone title line, and
  *not* inside the printed TOC block. Two roles are exempt from "must look like a title", and
  only two: a `role: "back-matter"` entry, which is allowed to be prose but must be last and must
  leave a plausible-size block, and a `role: "prose-chapter"` entry, which must **open a
  paragraph** (the line above it is blank) and carry `evidence` naming the contents list that
  makes it a chapter opening. A bad map is rejected with concrete fixes and sent back to the
  agent (bounded by `--max-attempts`).
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
  Stage 2 keeps each chapter's filename, so `chapters/` and `preprocessed/` line up one to one,
  and stage 3 does the same with `reduced/` and `reduced-unique/`.
- **Stage 3 drops, never edits.** A reduced chapter contains only whole sentences copied
  verbatim from the chapter, one per line. No word is changed, no NER markup is written, and the
  manifest records exactly how many sentences each filter removed and why (with clipped examples
  in the trace).

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-book-splitter.txt -r requirements-dev.txt
```

Stage 5 (`chunk`) needs `chonkie`; `tiktoken` is what makes the default
`cl100k_base` token count exact:

```bash
pip install chonkie tiktoken        # already in requirements-book-splitter.txt
```

Both are imported lazily. Without `tiktoken` the chunk length is still enforced,
but measured with the documented local approximation (and the manifest says so);
without `chonkie` the whole-pipeline script reports the phase as skipped and
`python -m book_splitter chunk` exits with the exact `pip install` line.

Stage 6 (`chunk-glossary`) needs `rapidfuzz` for the fuzzy retrieval, and the same
tagger as stage 3 for the NER pass:

```bash
pip install rapidfuzz                # already in requirements-book-splitter.txt
pip install -r requirements-dictionary.txt   # the tagger, if stage 3 already runs
```

`rapidfuzz` is a small wheel with no torch dependency and is imported lazily; the
tagger is only needed for phase 1, and `--language` / `--gpu` / `--cpu` behave
exactly as they do for `dictionary`. A missing tagger or missing `rapidfuzz` is one
`pip install` line, not a traceback.

Optional, only for `extract_native_toc` with PDF/EPUB sources:
`pip install PyMuPDF ebooklib beautifulsoup4`.

Optional, only for stage 3 (`dictionary`) — a tagger plus a language detector:

```bash
pip install -r requirements-dictionary.txt     # classla + stanza + lingua-py (pulls in torch)
```

⚠️ **`classla` pins `torch<=2.6` on Python < 3.14**, so a plain `pip install classla` next to a
newer torch will downgrade it and pull ~2 GB of CUDA wheels. If torch is already installed to your
liking, keep it:

```bash
pip install numpy requests tqdm "protobuf==4.21.2" obeliks reldi-tokeniser lxml
pip install --no-deps classla==2.2.3
```

That combination (torch 2.14 + classla 2.2.3) is verified working here — CLASSLA loads its Serbian
models and tags correctly, so the pin is stricter than it needs to be.

The three packages are imported lazily, so the splitter and the preprocessor keep working without
them; a `dictionary` run without them fails with the exact `pip install` line, and the whole-pipeline
script reports the phase as skipped instead of failing. Model weights download on first use — for
Serbian that is ~450 MB (the `standard` package: tokenize, pos, lemma, depparse, ner, pretrain), not
the ~150 MB a partial pipeline would need, so the stage asks for `tokenize,ner` at download time too.
`CLASSLA_RESOURCES_DIR` moves CLASSLA's copy; stanza keeps its own in the user cache dir as of 1.15
(`~/.cache/stanza/<version>/resources`, not `~/stanza_resources`) and honours
`STANZA_RESOURCES_DIR`. The `model load` line of the first run therefore includes the download
(measured: 1m 08s for English) and drops to a second or two once the weights are on disk.

## Use

```bash
# 1. What does the book actually contain? (no LLM, no API key, no writes)
#    With a single book in data/input/ the argument can be omitted entirely.
python -m book_splitter inspect --candidates

# 2. Stage 1 — split it (writes to data/output/<book>/chapters/)
python -m book_splitter split BOOK.md

# Plans the chapters and validates them, but writes nothing
python -m book_splitter split BOOK.md --dry-run

# Replay a chapter map from a file: no LLM, no API key, no network
python -m book_splitter split BOOK.md --map my-map.json

# Reuse a chapters folder that is already there: no model, no key, no rewrite
python -m book_splitter split BOOK.md --skip-if-present

# 3. What does stage 2 have to fix? (no LLM, no writes)
python -m book_splitter scan                  # the book-wide damage profile
python -m book_splitter scan --digest         # per chapter: headers + evidence + recommendation

# 4. Stage 2 — clean every chapter, one at a time
#    (writes to data/output/<book>/preprocessed/)
python -m book_splitter preprocess
python -m book_splitter preprocess --dry-run          # validate every plan, write nothing
python -m book_splitter preprocess --workers 4        # same work, thread pool

# 5. Stage 3 — reduce each chapter to its entity-bearing sentences, then to
#    first mentions (writes to data/output/<book>/reduced/ and reduced-unique/)
python -m book_splitter dictionary                    # auto-detect the language
python -m book_splitter dictionary --language sr --gpu 1
python -m book_splitter dictionary --limit 3          # a quick look while tuning
python -m book_splitter dictionary --skip-if-present  # reuse what is already reduced
python -m book_splitter dictionary --no-save-reduced  # count only, write no reduced chapters
python -m book_splitter dictionary --no-save-unique   # write reduced/ only
python -m book_splitter dictionary --unique-only      # write reduced-unique/ only
python -m book_splitter dictionary --out data/output/BOOK/preprocessed/   # reduce the cleaned text

# 6. Stage 4 — build the glossary, one chapter at a time, first to last
#    (writes to data/output/<book>/glossary/; calls the model, so it costs tokens)
python -m book_splitter glossary                      # English -> Serbian, the default pair
python -m book_splitter glossary -s sr -t en          # a Serbian book into English
python -m book_splitter glossary --limit 2 --dry-run  # price the first two chapters, write nothing
python -m book_splitter glossary --max-attempts 5     # be more patient with a flaky model
python -m book_splitter glossary --resume             # continue a run that stopped, reusing its tokens
python -m book_splitter glossary --include-front-matter   # the blocks around the body are skipped by default
python -m book_splitter glossary --max-context-entries 150  # cap the re-sent dictionary on a long book

# 7. Stage 5 — cut each chapter into 1000-token chunks that concatenate back
#    (writes to data/output/<book>/chunks/; no model, no key)
python -m book_splitter chunk                         # default 1000 tokens
python -m book_splitter chunk --chunk-size 700        # smaller translation units
python -m book_splitter chunk --dry-run               # verify every cut, write nothing
python -m book_splitter chunk --skip-if-present       # reuse existing chunks
python -m book_splitter chunk --limit 3               # a quick look at three chapters
python -m book_splitter chunk --out data/output/BOOK/preprocessed/   # chunk the cleaned text

# 8. Stage 6 — tag every chunk first, then attach the glossary terms it mentions
#    (writes to data/output/<book>/chunk-glossary/; no LLM, no API key;
#     the NER tagger is the same one stage 3 uses)
python -m book_splitter chunk-glossary                # default 95% rapidfuzz match
python -m book_splitter chunk-glossary --match-threshold 90   # more permissive
python -m book_splitter chunk-glossary --scorer token_sort_ratio
python -m book_splitter chunk-glossary --limit 3      # a quick look at three chunks
python -m book_splitter chunk-glossary --dry-run      # both phases, no writes

# Optional: list the toolset per stage / verify a folder layout
python -m book_splitter tools --stage preprocess

# Explicit folders / fully local model
python -m book_splitter split BOOK.md --out /tmp/chapters/
python -m book_splitter preprocess --out /tmp/chapters/ --preprocessed-out /tmp/clean/
python -m book_splitter dictionary --out /tmp/chapters/ --unique-out /tmp/mentions/
python -m book_splitter glossary --out /tmp/chapters/ --glossary-out /tmp/glossary/
python -m book_splitter chunk --out /tmp/chapters/ --chunks-out /tmp/chunks/
python -m book_splitter chunk-glossary --chunks-in /tmp/chunks/ \
    --glossary-in /tmp/glossary/ --chunk-glossary-out /tmp/chunk-glossary/
python -m book_splitter preprocess \
    --provider lmstudio --model qwen/qwen3.8-27b --base-url http://127.0.0.1:1234/v1
```

Useful flags: `--in-dir`, `--out`, `--map`, `--preprocessed-out`, `--reduced-out`, `--unique-out`,
`--chunks-out`, `--chunk-size`, `--chunk-tokenizer`,
`--chunks-in`, `--glossary-in`, `--chunk-glossary-out`, `--match-threshold`, `--scorer`,
`--candidates`, `--min-term-chars`, `--drop-temporal-only`,
`--glossary-out`, `--sentences`, `--sentences-in`, `--source-language`, `--target-language`,
`--include-front-matter`, `--include-back-matter`, `--extraction-method`, `--max-tokens`,
`--max-chapter-chars`, `--max-context-entries`, `--resume`,
`--max-attempts`, `--workers`, `--front-matter-lines`, `--no-front-matter`, `--skip-if-present`,
`--language`, `--gpu`, `--cpu`, `--min-words`, `--keep-types`, `--no-save-reduced`,
`--no-save-unique`, `--unique-only`, `--limit`,
`--dry-run`, `--json`, `--quiet`, `--temperature`, `--api-key`.
`python -m book_splitter tools` lists the toolset **grouped by
stage**, and [scripts/check_layout.py](scripts/check_layout.py) verifies the folder layout and the
installed dependencies.

### One command for the whole pipeline

[scripts/run_pipeline.py](scripts/run_pipeline.py) runs the phases in the only order that works —
recon → split → scan → dictionary (→ preprocess, opt-in) → chunk → chunk-glossary — and narrates
each one: book size and header verdict, the strategy the agent chose, the byte-for-byte proof, the
damage profile, how many sentences survived the dictionary funnel and how many of those are first
mentions, how many entities were locked into the glossary — with a live progress bar while that
runs — what the LLM pass cost in tokens, how many reversible chunks the chapter text was cut into,
and finally how many glossary terms each chunk mentions.
`--debug` adds headers, planned files and damage examples; the same transcript always
lands in `data/output/<book>/pipeline.log`, debug included.

```bash
python scripts/run_pipeline.py                      # every free phase, the only book in data/input/
python scripts/run_pipeline.py BOOK.md --debug      # + the per-phase detail
python scripts/run_pipeline.py --dry-run            # validate every plan, write no artifact
python scripts/run_pipeline.py --stop-after split   # stop once the chapters exist
python scripts/run_pipeline.py --start-at dictionary  # stage 3 only, over existing chapters
python scripts/run_pipeline.py --skip-if-present    # reuse chapters, reduced chapters, glossary
python scripts/run_pipeline.py --no-dictionary      # split + scan + chunk only
python scripts/run_pipeline.py --unique-only        # stage 3 first mentions only
python scripts/run_pipeline.py --start-at chunk     # stage 5 only, over existing chapters
python scripts/run_pipeline.py --start-at chunk_glossary   # stage 6 only, over existing chunks
python scripts/run_pipeline.py --chunk-size 700     # smaller translation chunks
python scripts/run_pipeline.py --match-threshold 90 # more permissive chunk glossary retrieval
python scripts/run_pipeline.py --no-chunk-glossary  # stop after the chunks
python scripts/run_pipeline.py --no-chunk           # stop before the chunking phase
python scripts/run_pipeline.py BOOK.md --map map.json --stop-after split   # no LLM at all

# the dictionary phase builds the glossary too; --no-glossary keeps it deterministic
python scripts/run_pipeline.py --start-at dictionary --skip-if-present   # resume the glossary
python scripts/run_pipeline.py --no-glossary                             # reduce only, no key needed
python scripts/run_pipeline.py --limit 2 --dry-run                      # never spends
```

Stage 3 stays selected but **reports itself as skipped** when its optional tagger is not
installed, so a split-only workflow still finishes; `--with-dictionary` / `--no-dictionary`
override that for one run. Stage 5 does the same for a missing `chonkie`, and `--no-chunk`
drops it from the run. `--chunk-in` points the chunker at the cleaned
`preprocessed/` folder instead of `chapters/`; by default it cuts the splitter's output.

Stage 6 is the last phase and the bridge to translation: it tags every chunk first, then
fuzzy-matches the extracted terms against the glossary. It needs both stage 5's chunks and stage
4's `glossary.json`; when either is missing it reports a skip with the reason (a missing tagger or
`rapidfuzz` is reported the same way), so the earlier phases keep their artifacts. `--no-chunk-glossary`
drops it, `--start-at chunk_glossary` runs only it, `--match-threshold` and `--match-scorer` tune the
retrieval, and `--no-manifest` keeps only the per-chunk files. The closing report lists the stage-6
manifest and the `chunk-glossary/` folder.

The glossary is not a phase of its own: it is the second half of `dictionary`, because it reads
exactly what the reduction just wrote. It refuses to run under `--dry-run` (there is nothing to
validate without asking the model, and a dry run must not spend), and `--no-glossary` drops it so
the phase stays deterministic. `--skip-if-present` resumes it: a run that stopped after chapter 30
pays for chapters 30..N and keeps the tokens already spent. The closing report ends with the
dictionary's path and the token bill. `python -m book_splitter glossary` remains the shorter way to
run stage 4 on its own.

It exits `0` only when every phase it ran succeeded, `1` when a phase failed (the phases before it
stay on disk), and `2` on a setup error such as a missing book or an unresolvable provider.
`--json` appends the whole run — per phase: status, seconds, summary and the phase's own result
object — as one JSON document.

### Configuration

Copy [.env.example](.env.example) to `.env`, or pass flags. Credential lookup order:
`--api-key` → `SPLITTER_API_KEY` / the provider's own variable → `.env` → `refs:` in
`~/.dsh/.credentials.yaml`. Local endpoints (`localhost`, `127.0.0.1`) may run keyless.
Stages 1, 2 and 4 use the same provider settings (DeepSeek by default); stages 3, 5 and 6 call no
model at all, so they need no key. Stage 4 needs one only when there is a chapter left to mine: a
`--resume` run over a finished book exits without resolving a model.

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

data/output/<book-stem>/reduced/           # stage 3
├── 00-front-matter.md … 15-beleske.md     # same names: one raw kept sentence per line
├── _reduced_manifest.json                 # kept/dropped, entity + noun totals, timings,
│                                          #   and the per-chapter noun index a rerun reads
└── _reduced_trace.json                    # + per-chapter seconds + examples of each drop

data/output/<book-stem>/reduced-unique/    # stage 3, second step
└── 00-front-matter.md … 15-beleske.md     # same names: first mentions only (a subset)

data/output/<book-stem>/glossary/          # stage 4
├── glossary.json                          # the dictionary: entries + languages + token total
├── glossary.md                            # the same, as a table per category, for review
├── _glossary_manifest.json                # per chapter: tokens, attempts, added/repeats/conflicts
└── _glossary_trace.json                   # + every entry a chapter added and every raw answer

data/output/<book-stem>/chunks/            # stage 5 — flat, globally ordered
├── 00000-00-front-matter-0000.md          # <global>-<chapter>-<index>.md, an exact slice
├── 00001-01-predgovor-0000.md
├── …
├── _chunk_manifest.json                   # per chunk: chapter, order, char/line spans,
│                                          #   tokens, boundary kind, hashes + the proof
└── _chunk_trace.json                      # the same run, with each chapter's own log

data/output/<book-stem>/chunk-glossary/   # stage 6 — one file per chunk
├── 00000-00-front-matter-0000.md.json     # named after the chunk it describes
├── 00001-01-predgovor-0000.md.json
├── …
├── _chunk_glossary_manifest.json          # per chunk: NER counts + matched terms,
│                                          #   the two phases and their timings
└── _chunk_glossary_trace.json             # + every term, closest candidate and log
```

The two stage-3 folders are two views of one pass: `reduced/` holds every sentence the NER gate
kept, `reduced-unique/` holds the subset that introduces at least one noun no earlier sentence
used, decided book-wide in chapter order. `reduced/_reduced_manifest.json` describes both
(`unique_dir`, `unique_totals`, and per file `unique_lines`, `unique_kept`,
`dropped_duplicate_noun`, `noun_keys`), so `reduced-unique/` needs no bookkeeping of its own —
and a rerun can resume either folder without re-tagging a chapter it reuses.

`glossary.json` is stage 4's artifact and the only file the translation stage needs to read: one
entry per entity, `term` / `category` / `translation`, plus the chapter that introduced it
(`chapter_index`, `chapter_filename`, `also_returned_in`). It is written in
insertion order — reading order of the book. `glossary.md` is the same dictionary as a review
table, and the token total — the number the run prints last — is stored under `tokens` alongside
the input/output split and the number of model calls.

`_agent_trace.json` is stage 1's audit trail: which strategy won, what the agent proposed, and
what the validators said about it. `_preprocess_trace.json` is stage 2's: for every chapter, the
operations the agent chose, the fixes it requested, the validator's verdict, and the provenance
map of the file that was written. `_reduced_trace.json` is stage 3's: the detected language, its
samples and any reflex overrule, which backend ran and on which device, the timings, and per
chapter how many sentences each filter removed with clipped examples of the casualties — the two
cheap filters, the NER gate, and the first-mention pass (`detail[].unique`, which also lists the
nouns each kept sentence introduced). `_glossary_trace.json` is stage 4's: for every chapter, the
entries it added, the terms it returned again, the conflicts it tried to introduce, the raw answers
it gave, and the tokens each of them cost. `_chunk_manifest.json` is stage 5's index — the chunk
files are text, so the manifest is what makes them a *book* again: it records every chunk's chapter,
global order, character and line spans, token count, boundary kinds and hashes, plus the
`verification` block that proves the folder concatenates back to the chapter text.
`_chunk_glossary_manifest.json` is stage 6's index: per chunk, how many entity mentions the tagger
found, how many distinct terms, which glossary entries matched, and the two phases with their own
seconds. The per-chunk `*.md.json` files are the payload a translation prompt reads — the chunk's
name is the first field of the filename and the chunk's spans are inside, so the file can be found
from the text or the text from the file. `_chunk_glossary_trace.json` adds every term, its closest
candidate and score even when it stayed below the threshold, which is how a missed name is
diagnosed without re-running the tagger.

To rebuild without any code, a `cat` of the `*.md` files in filename order is the whole book,
because the global index is the first field of every name. Per chapter, the manifest's
`chapter_filename` groups the files and `chunk_index` orders them. In Python,
`book_splitter.chunking.reconstruct_chapter()` and `reconstruct_book()` do the same
without reading a filename.

## Tests

```bash
python -m pytest          # offline: no network, no API key, no tagger, no GPU
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

Stage 3's tests inject a fake tagger, so they pin the funnel without a model, a weight download or
a GPU: the sentence splitter on abbreviations, initials, ellipses, `5. maja 1990.` and dialogue
attributions; the uppercase and four-word filters and which reason each drop is reported under;
one reduced file per chapter with one raw sentence per line; the manifest shape the later stages
read; the `skip_if_present` short-circuit (a second run must not call the tagger at all); language
routing to CLASSLA or stanza; and that `--gpu` / `--cpu` set `CUDA_VISIBLE_DEVICES` before any
backend is built. The first-mention step is tested on its own (the noun key, the "at least one new
noun" rule, the book-wide `seen` set) and through the runner: both folders written, the second one
a strict subset, a chapter re-run picking up the nouns of the chapters it did not re-tag, and the
CLI switches that add, move or drop the second folder.

Stage 4's tests inject a scripted extractor and a scripted chat model, so the whole stage runs
offline: the Pydantic contract (the five categories, a normalised label the model invented, an
empty field rejected, a bare array and an invented wrapper unwrapped); the retry loop (a rejected
answer is retried with the failure named in the prompt, the last `auto` attempt falls back to JSON
mode, a transport failure is not charged tokens and a rejected answer is); the ordered run (front
and back matter skipped and *reported* as skipped, every chapter after the first handed the
dictionary so far, the first rendering winning over a later disagreement, a long chapter split and
the parts sharing the dictionary); the token total (every call of every chapter, retries included);
and the four artifacts, including a resumed run that reuses the locked entries and the tokens
already spent. The CLI is exercised through `main()` with those two seams patched, so the flags,
the exit codes and the printed token line are pinned without a key.

Stage 5's tests run offline with chonkie's built-in `word` tokenizer, so no vocabulary is
downloaded: every chunk is an exact source slice, concatenating a chapter's chunks in
`chunk_index` order (and the whole folder in `global_index` order) reproduces the text byte for
byte, the global order and chapter membership in the manifest match the files on disk, the chunk
length is a hard limit, paragraph boundaries are kept when they fit, an oversized paragraph is
cut without losing a character, a dry run writes nothing, `--skip-if-present` reuses files and
re-chunks a chapter whose text changed, a different chunk size forces a fresh cut, an empty
chapter is handled as a zero-chunk chapter, and the CLI (`main()`) writes the manifest and prints
the reconstruction proof.

Stage 6's tests inject a scripted tagger and a recorder around the real rapidfuzz matcher, so no
model is loaded and the whole stage runs offline. They pin the contract that matters: **every chunk
is tagged before the first glossary lookup** (the recorder proves the order), each chunk produces
exactly one file named after it, an exact term retrieves its locked translation, a near miss (the
inflected `Džordža` against `Džordž`) is recorded as `closest` at the default 95 % and becomes a
match at 90 %, an unknown term (`Mithril`) stays unmatched with its nearest score, the per-chunk
`matched_terms` merge two mentions of the same dictionary entry, a dry run runs both phases and
writes nothing, `--limit` tags only the first chunks, a missing glossary/chunks folder is one
actionable error, and the CLI resolves the folders from the book, prints the threshold and phases,
and returns the expected exit codes. The matcher's own tests pin the normalisation (punctuation
trimmed, hyphens and diacritics kept), the glossary loader (dedupe, bare list, missing file), the
threshold/scorer validation and the candidate ordering.

### Replaying a map without the model

A chapter map is data, and it can be supplied instead of discovered. `--map FILE` skips the agent
entirely — no provider, no API key, no network — and puts the file through the *same* validator,
slicer and byte-for-byte verifier the graph uses:

```bash
python -m book_splitter split BOOK.md --map my-map.json              # -> chapters/
python -m book_splitter split BOOK.md --map my-map.json --dry-run    # validate only
python scripts/run_pipeline.py BOOK.md --map my-map.json --stop-after split
```

The format is documented in [book_splitter/slicing/mapfile.py](book_splitter/slicing/mapfile.py):
a list of `{"line_number", "filename", "title"?, "role"?, "evidence"?}` entries, or an object
with that list under `splits` / `split_mapping` / `map` / `chapters` / `entries`. Aliases are
accepted (`line`/`start_line`, `file`/`name`, `kind` for `role`, `provenance` for `evidence`), so
a hand-recovered, richly documented map can be replayed unchanged — for example the one written
for *Zima u hotelu smrti* at
[data/output/Kenneth_M._Cameron_-_Zima_u_hotelu_smrti/chapters/_recovered_map.json](data/output/Kenneth_M._Cameron_-_Zima_u_hotelu_smrti/chapters/_recovered_map.json),
which states its evidence and confidence per chapter. A map that fails validation writes nothing
but its trace, and the trace says which entry was refused and why.

This is the escape hatch for the one thing no deterministic check can supply: *which paragraph*
opens a chapter whose heading the conversion deleted. Write that judgement down once, and from
then on it is as reproducible as an agent plan — one line of the trace, one `preset_map` run, zero
tokens.

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
| [book_splitter/cli.py](book_splitter/cli.py) | `inspect` / `split` / `scan` / `preprocess` / `dictionary` / `glossary` / `chunk` / `chunk-glossary` / `tools` commands |
| [book_splitter/core/constants.py](book_splitter/core/constants.py) | every guardrail and default in one place |
| [book_splitter/core/settings.py](book_splitter/core/settings.py) | provider and credential resolution |
| [book_splitter/core/errors.py](book_splitter/core/errors.py) | `ToolError`, `ConfigError` |
| [book_splitter/documents/io.py](book_splitter/documents/io.py) | cached source reading |
| [book_splitter/documents/markdown.py](book_splitter/documents/markdown.py) | header/standalone/prose line shapes (both stages) |
| [book_splitter/documents/bodyline.py](book_splitter/documents/bodyline.py) | body-line *damage* shapes: split words, mid-word page numbers |
| [book_splitter/documents/chapters.py](book_splitter/documents/chapters.py) | reads stage 1's `chapters/` folder back as chapter records; resolves the `preprocessed/`, `reduced/`, `reduced-unique/`, `glossary/`, `chunks/` and `chunk-glossary/` siblings |
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
| [book_splitter/dictionary/sentences.py](book_splitter/dictionary/sentences.py) | stage 3: sentence splitting and the two dependency-free filters |
| [book_splitter/dictionary/language.py](book_splitter/dictionary/language.py) | stage 3: sampled language detection (`lingua-py`), the ekavian/ijekavian tie-breaker, backend routing |
| [book_splitter/dictionary/ner.py](book_splitter/dictionary/ner.py) | stage 3: the CLASSLA/stanza gate, entity rules, GPU selection and the device actually used |
| [book_splitter/dictionary/unique.py](book_splitter/dictionary/unique.py) | stage 3: the noun key and the first-mention filter (book-wide, in chapter order) |
| [book_splitter/dictionary/runner.py](book_splitter/dictionary/runner.py) | stage 3: `run_dictionary`, the chapter loop, timings, `reduced/` + `reduced-unique/` and their manifest |
| [book_splitter/glossary/models.py](book_splitter/glossary/models.py) | stage 4: the Pydantic answer (`term` / `category` / `translation`) and category normalisation |
| [book_splitter/glossary/prompts.py](book_splitter/glossary/prompts.py) | stage 4: the book-translation system prompt, the incremental chapter prompt, the dictionary renderer |
| [book_splitter/glossary/extract.py](book_splitter/glossary/extract.py) | stage 4: one model call, structured output, the retry loop and token accounting |
| [book_splitter/glossary/runner.py](book_splitter/glossary/runner.py) | stage 4: `run_glossary`, the ordered chapter walk, the front/back-matter skip, merge rules and the four artifacts |
| [book_splitter/chunking/boundaries.py](book_splitter/chunking/boundaries.py) | stage 5: names the boundary kind of a cut (paragraph / sentence / line / word / hard) |
| [book_splitter/chunking/tokenizer.py](book_splitter/chunking/tokenizer.py) | stage 5: resolves the tokenizer and the offline approximation, with the reason recorded |
| [book_splitter/chunking/runner.py](book_splitter/chunking/runner.py) | stage 5: `run_chunking`, the reversible cut, the manifest, the skip/reuse rules and the reconstruction helpers |
| [book_splitter/chunking/__init__.py](book_splitter/chunking/__init__.py) | stage 5: public write side (`run_chunking`) and read side (`reconstruct_chapter` / `reconstruct_book`) |
| [book_splitter/chunk_glossary/matching.py](book_splitter/chunk_glossary/matching.py) | stage 6: glossary loading, term normalisation and the rapidfuzz matcher (threshold and scorer are options) |
| [book_splitter/chunk_glossary/runner.py](book_splitter/chunk_glossary/runner.py) | stage 6: `run_chunk_glossary`, the two phases (NER over every chunk, then retrieval), the per-chunk files and the manifest/trace |
| [book_splitter/chunk_glossary/__init__.py](book_splitter/chunk_glossary/__init__.py) | stage 6: public surface (`run_chunk_glossary`, the matcher and the per-chunk records) |
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
- A split point is a heading, a standalone title line, or a paragraph opening explicitly marked
  `role: "prose-chapter"`. That last shape is checked for *form* (it must open a paragraph) and
  for the *presence* of evidence, never for the evidence's truth: which paragraph opens a
  heading-less chapter is a reading judgement, so a wrong one moves text between two files
  without losing a byte. Validation warns when a map rests on prose starts, and `--map` exists so
  such a judgement can be reviewed, written down and replayed.
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
- Stage 3's no-uppercase filter is deliberately aggressive: on the sample book it removes 3 619 of
  4 271 sentences on its own (85 %), including any sentence whose only capital is a name the author
  left uncapitalised or that opens with a proper noun in the first position (`Beograd je veliki
  grad.` goes; `U Beogradu je veliki grad.` stays). That is the price of not running the tagger on
  everything; `--no-uppercase-filter` turns it off when a book needs it.
- Stage 3 has no NER model for Macedonian and none at all for a language outside CLASSLA's four plus
  stanza's `en`. Adding one means adding the ISO code to `STANZA_LANGUAGES` and installing stanza's
  models for it — the routing is one line, the tagger does the rest.
- **Automatic language detection is only trustworthy for the sr/en decision**, which is why the
  ekavian/ijekavian tie-breaker exists. Slovenian and Bulgarian books are detected by lingua-py
  alone, and a book that mixes standards (or quotes heavily from another language) can still be
  routed to the wrong CLASSLA model. `--language sr` is the escape hatch, and the run records both
  the detector's vote and the overrule so the decision is auditable.
- The gate keeps whole sentences, so a sentence can be kept for one entity and still be mostly
  filler — and two shapes that are not dictionary evidence get through: all-caps Markdown headings
  (CLASSLA reads them as `ORG`) and one-letter citation fragments (`MISC`). They are a small share
  of the output (the sample book's own run: 16 heading lines, plus `A`×29 among the `MISC`
  mentions) and they are visible in the reduced files rather than hidden, which is deliberate while
  the filter chain is still being tuned.
- Stage 3's temporal-only filter (`--drop-temporal-only`) is a surface-form heuristic, because
  neither CLASSLA nor stanza has a TIME/DATE entity type: it recognises years, clock times,
  `d.m.yyyy`, month and weekday names, and a short list of temporal nouns. A date written in a shape
  outside that list survives. Note that CLASSLA rarely tags a bare year at all (`Sreli smo se 1996.
  godine.` yields no entity), so this flag matters less than it looks.
- Stage 3 timings measure wall clock on the machine that ran them: the sample book's 3.3 s of NER
  is **CPU** time, because no GPU was reachable from the shell that produced the run. The summary
  reports the device actually used, so a silent CPU fallback cannot be mistaken for a GPU run.
- Stage 3 does **not** build the dictionary. It only reduces the text; stage 4 is the step that
  reads the reduced chapters and decides which nouns need a fixed translation, and
  `reduced-unique/` is the list it reads by default — the whole first-mention set, one sentence per
  noun, in book order. Both folders are saved (and switchable with `--no-save-unique` /
  `--unique-only`) so stage 4 can be pointed at either (`--sentences reduced`), and so the funnel's
  output can be compared while the rule is adjusted.
- **Stage 4's consistency is only as good as the model's answer.** The dictionary locks what the
  first chapter that mentioned an entity decided, and the run cannot tell a good rendering from a
  bad one — it can only tell a *consistent* one. What it does give a reviewer is the evidence:
  `glossary.md` is short enough to read, every entry says which chapter introduced it, and a later
  chapter that disagrees is recorded as a conflict instead of quietly overriding the lock. Read the
  markdown before translating; that is what it is for.
- **The dictionary is re-sent with every chapter**, so the input cost of a long book grows with the
  square of its entity count (each call pays for every entry locked so far). On the sample-sized
  books that is negligible; on a 600-page novel with a thousand entities it is the dominant cost.
  `--max-context-entries` caps it, at the price of the model no longer seeing the full rendering of
  every locked term (it still sees that the term is taken, so it will not create a second entry),
  and `--limit N` prices the first chapters before the decision is made for the book.
- Stage 4 calls the model **once per chapter with no tools and no retrieval**, so it can only know
  what the given sentences and the dictionary so far say. An entity whose first mention is a bare
  surname is locked as that surname; a book that only ever refers to a place by an abbreviation
  gets an abbreviation. The prompt asks for the full name behind an alias when the chapter shows
  one, but a later chapter cannot amend an entry — that is the price of "the first rendering wins",
  and it is deliberate: an amendable dictionary is one a translated chapter can disagree with.
- Stage 4 skips the front and back matter by default, and stage 1's `kind` is what it trusts. A book
  whose manifest is missing or whose blocks were never labelled falls back to filenames, which means
  an unusually named back-matter file (`15-beleske.md`) is **mined** rather than skipped. The
  manifest records exactly which files were skipped and why, so the mistake is visible; the fix is
  `--include-back-matter` on a rerun (or a chapter map that names the block `back-matter`).
- Stage 4's `--resume` trusts the manifest. If the reduced chapters changed after the first run
  (stage 3 re-run with different filters), the entries from the reused chapters are the old
  chapter's; delete `glossary/` or run without `--resume` to rebuild.
- The first-mention filter keys on the **surface form** of a mention, case-folded and trimmed, and
  ignores its entity type. Two inflections of the same name (`Marko` / `Marku`) are therefore two
  nouns, and one name tagged `PER` in one place and `MISC` in another is one noun. It also cannot
  remove the heading/fragment noise above: a heading is a new noun the first time it appears. What
  it does guarantee is the shape of the output — every surviving sentence introduces at least one
  noun, and the union of the kept sentences' nouns is the book's whole noun set.
