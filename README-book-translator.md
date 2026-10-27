# Book pipeline — source conversion, four LangGraph agent stages, a dictionary funnel, a glossary, reversible chunks, a per-chunk glossary, front/back matter localization, the translation, its heading postprocessing and the final book

The repository holds the pipeline that turns one long book into a consistent
translation. **Stage 0** turns the source document — EPUB, a text-based PDF or a
TXT — into the Markdown every later stage reads: docling does the EPUB/PDF
conversion with OCR **off**, so an image-based PDF is rejected instead of OCR'd,
and a TXT is copied under a `.md` name. Stage 0 also extracts the source's
pictures into `00-markdown/images/` and writes Markdown image references, so the
illustrations travel through every later stage into the final translated book;
`--no-images` keeps the old text-only conversion. A Markdown source skips stage 0
entirely.
Stages 1, 2, 6b and 9 each have their **own
specialised agent** with its **own toolset**, and the tools are grouped by the
stage that uses them (plus a generic `shared` group used by all of them). Stage 3 has no
agent at all: it is a deterministic funnel with an optional NER tagger. Stage 4
has no agent and no tools either — it is a straight, ordered sequence of
LLM calls with a Pydantic-validated answer. Stage 5 has no model at all: it cuts
each chapter into translation-sized chunks with chonkie and proves they
concatenate back byte for byte. Stage 6 uses no LLM either: it tags every chunk
with the same NER backends, then fuzzy-matches each extracted term against the
glossary and writes the terms a chunk mentions next to that chunk. **Stage 6b is
a separate agent that localizes the front and back matter before translation**:
it uses **no dictionary**, searches the web for an existing published translation
of the title, omits the original edition's imprint/ISBN/CIP material, credits
**BleBel**, and formats the result as published front/back matter. Stage 7 is the
body translation itself: it translates the body chunks one at a time, sends each
one its glossary terms, validates a Pydantic translation answer with retries, and
keeps the whole prompt of every call for debugging. Stage 8 is the inverse of
stage 5 applied to those translated chunks: it concatenates them back into one
file per chapter, in a folder separate from the source. Stage 9 is the heading
pass over those translated chapters: one book-level naming policy, one chapter at
a time, a table of contents made consistent with the final titles, and a separate
`09-postprocessed-chapters/` folder. Stage 10 is the deterministic
concatenation of that folder into one `10-translated-book.md`, and stage 11
packages the same folder as an EPUB 3 book, `11-translated-book.epub` — the file
you actually open in a reader.

Every artifact folder is prefixed with the number of the stage that writes it
(`00-markdown/`, `01-chapters/`, `02-preprocessed/`, …, `09-postprocessed-chapters/`,
and the final `10-translated-book.md` and `11-translated-book.epub`), so a plain
sort of a book's output folder follows the pipeline order. Folders written by earlier runs keep their old
names; only new writes use the numbered layout.

| Stage | Module | Question it answers | Writes |
| --- | --- | --- | --- |
| 0. **Convert** | `book_translator.stages.conversion` | *What does this EPUB/PDF/TXT say as plain Markdown — and what do its pictures look like?* | `data/output/<book>/00-markdown/` + `00-markdown/images/` |
| 1. **Split** | `book_translator.stages.agent` | *Where does each chapter start?* | `data/output/<book>/01-chapters/` |
| 2. **Preprocess** | `book_translator.stages.preprocess_agent` | *What in this chapter is not part of the book's text?* | `data/output/<book>/02-preprocessed/` (the text stages 3–5 read by default) |
| 3. **Dictionary** | `book_translator.stages.dictionary` | *Which sentences carry a proper noun worth keeping consistent?* | `data/output/<book>/03-reduced/` + `03-reduced-unique/` |
| 4. **Glossary** | `book_translator.stages.glossary` | *How must each of those nouns be written in the target language, once and for all?* | `data/output/<book>/04-glossary/` |
| 5. **Chunk** | `book_translator.stages.chunking` | *Where can the chapter be cut without breaking a paragraph or a sentence — and how do we put it back?* | `data/output/<book>/05-chunks/` |
| 6. **Chunk glossary** | `book_translator.stages.chunk_glossary` | *Which locked renderings does this chunk actually mention?* | `data/output/<book>/06-chunk-glossary/` |
| 6b. **Matter** | `book_translator.stages.matter_agent` | *How is the front/back matter of the translated edition titled, credited, pruned, and its images/covers recreated? (Requires multimodal model)* | translated chunks in `07-translations/` + `06b-matter/` + `06b-matter-prompts/` |
| 7. **Translate** | `book_translator.stages.translation` | *What does this body chunk say in the target language, using those renderings?* | `data/output/<book>/07-translations/` + `07-translation-prompts/` |
| 8. **Assemble** | `book_translator.stages.assembly` | *What does each whole chapter say, now that every chunk is translated?* | `data/output/<book>/08-translated-chapters/` |
| 9. **Postprocess** | `book_translator.stages.postprocess_agent` | *Are the translated chapter titles, the other headings and the table of contents consistent?* | `data/output/<book>/09-postprocessed-chapters/` |
| 10. **Bind** | `book_translator.stages.binding` | *What does the whole translated book look like as one document?* | `data/output/<book>/10-translated-book.md` |
| 11. **EPUB** | `book_translator.stages.epub` | *What does the book look like in a reader — images, navigation, metadata?* | `data/output/<book>/11-translated-book.epub` |

Stage 0 reads the source from `data/input/` and writes `00-markdown/<book>.md`.
Three formats are accepted. A `.txt` is copied **byte for byte** and only its name
changes to `.md`. An `.epub` is converted by docling's EPUB backend. A `.pdf` is
checked for a text layer first: stage 0 samples up to twelve pages and counts the
pages that carry real characters (not just a printed page number). If fewer than
half of the sampled pages have text, the PDF is reported as **image-based** and
rejected — the conversion pipeline runs with `do_ocr = False` and never falls
back to OCR, because OCR would invent a different book. Run OCR on such a PDF
outside the pipeline (or supply the EPUB/TXT edition) and re-run stage 0. Every
conversion writes `_convert_manifest.json` next to the Markdown with the source
hash, the method and the PDF page evidence; a second run reuses the output while
that hash still matches, and `--force` converts again. `inspect` and `split` call
stage 0 automatically when the source is not Markdown; the one-command pipeline
does the same as its first phase.

**Pictures.** Unless `--no-images` is passed, docling also extracts every picture
of an EPUB/PDF into `00-markdown/images/` and the Markdown references it as
`![Image](../00-markdown/images/<name>).` The `../00-markdown/images/` prefix is
deliberate: it resolves from `00-markdown/` itself and from every later stage
folder (`01-chapters/`, `05-chunks/`, `07-translations/`, …), all of which are one
level below the book folder. Duplicate exports of the same bytes (a repeated
section break) are merged. Stage 10 rewrites the prefix once to
`00-markdown/images/` for the final book at the book root, so the links resolve
there too, and stage 11 copies the pictures it references into the EPUB itself.
The manifest records `images` and `image_refs`; a conversion whose `images/`
folder has been deleted is not reused.

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
conversion split across two lines, words a page number was printed inside, and
page numbers the conversion merged into a text line (listed as candidates with
the words on both sides). The agent describes its edits in a fixed vocabulary; a
deterministic validator checks every entry against the real file, and a per-line
provenance map proves afterwards that only the declared lines changed.

Stage 3 reduces each chapter, **one chapter at a time, first to last**, to the raw
sentences that are worth mining for a translation glossary: split into sentences,
drop the ones with no capital past the first word, drop the ones under four words,
then keep only the sentences a tagger found a `PER`/`LOC`/`ORG`/`MISC` mention in.
Those survivors go to `03-reduced/`. A second, free step then keeps only the
sentences that **introduce a noun the book has not used yet** and writes them to
`03-reduced-unique/`: the same pass, the same tagger, a much shorter list. Both
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
`04-glossary/`, and the run ends by printing what it cost in tokens.

Stage 5 cuts every chapter, **first to last**, into translation-sized chunks
(1000 tokens by default). It is the only stage that looks at the body as text
instead of as lines: chonkie's recursive chunker prefers paragraph boundaries,
then sentence boundaries, then line boundaries, then whitespace, so a chunk ends
where a reader would pause whenever the requested length allows it. The text
written for a chunk is an exact slice of the chapter — nothing is trimmed or
re-wrapped — so concatenating a chapter's chunks in order gives the chapter back
byte for byte. `05-chunks/_chunk_manifest.json` records, for every chunk, the chapter
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
`05-chunks/00001-01-predgovor-0000.md` → `06-chunk-glossary/00001-01-predgovor-0000.md.json`.
The file carries the chunk's identity (global order, chapter, character and line
spans) and a `matched_terms` list of `term` / `translation` / `category` rows, so a
translation call can be sent with exactly the renderings that chunk must obey.

Stage 6b localizes the **front and back matter** before stage 7, and it is the one
translation stage that deliberately uses **no dictionary**. Front matter is not a
chapter: its title may already have a published rendering in the target language,
its imprint/copyright/ISBN page must not be carried into the translated edition
unchanged, and the new edition has to say who translated it. A book-level
LangGraph agent first searches the web (`web_search` / `fetch_web_page`, the same
tools the splitter uses) for an existing published translation and reuses its
exact title when one is found; it then fixes the **BleBel** credit, lists the
source phrases that must be omitted, and gives every front/back matter file its
localized section title. That plan is binding for one agent run per matter chunk:
each chunk is localized with a front/back-matter prompt (not a chapter prompt),
formatted as published Markdown, and validated against the plan — the first
front-matter chunk is guaranteed to open with the localized title and the
translator credit, and an omitted chunk is written empty with the reason in the
manifest. The localized chunks land in `07-translations/` under the usual chunk
names, so stage 8 joins them with the body without a new rule, while the plan,
manifest, trace and the whole prompt of every agent call are kept in `06b-matter/`
and `06b-matter-prompts/`.

Stage 6b additionally executes an end-to-end **multimodal matter image translation**
phase: images referenced in matter chunks (such as original book covers, title
plates, or frontispieces) are sent with book context (localized title, author,
BleBel credit, and chapter text) to a multimodal vision + image generation model
(`gemini-nano-banana-2.1`, requiring a multimodal model and `GEMINI_API_KEY`).
The model recreates the artwork seamlessly from scratch with translated lettering
natively embedded in the typography and artwork (avoiding patch overlays or bounding
boxes), while leaving pure text-free art untouched. Recreated images land in
`07-translations/images/<stem>_translated.<ext>` and Markdown references are rewritten.

Stage 7 skips those chapters by default; `--include-matter` restores the old
behaviour for a caller that wants the body prompt to translate everything.

Stage 7 is the call that file exists for. It walks the **body** chunks in reading
order, one model call at a time, and sends each chunk with its `matched_terms` as
a binding glossary block. The answer is a Pydantic-validated JSON object with one
field (`translation`); a refusal, an empty answer, an answer that merely repeats
the source or a summary instead of a translation is retried with the failure named,
and every attempt is priced. Translated chunks land in `07-translations/` under the
chunk's own name, so concatenating them in filename order rebuilds the translated
book. The **whole input** of every call — system prompt, user prompt, the
serialised messages, the glossary rows that were sent and the source chunk — is
written to `07-translation-prompts/`, once as JSON and once as a readable Markdown
prompt, so a bad paragraph can be debugged without reconstructing the request.
English into Serbian is the default pair, `--limit N` translates a prefix, and
`--resume` skips chunks an earlier run already finished.

This is the implementation of the toolset described in the original design notes for
splitting Markdown books without LLMs, built for the sample book
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
        ├── 00-markdown/                         # stage 0 output — only for EPUB/PDF/TXT
        │   ├── <book>.md                       # the converted source stage 1 reads
        │   ├── images/                         # pictures extracted from the EPUB/PDF
        │   │   └── image_000001_<hash>.png     # referenced as ../00-markdown/images/…
        │   └── _convert_manifest.json          # source hash, method, PDF text-layer evidence,
        │                                       #   image and image-reference counts
        ├── 01-chapters/                         # stage 1 output
        │   ├── 00-front-matter.md
        │   ├── 01-predgovor.md
        │   ├── …
        │   ├── 15-beleske.md
        │   ├── _manifest.json                # filenames, line ranges, bytes, SHA-256
        │   └── _agent_trace.json             # the plan, validation report, token usage
        ├── 02-preprocessed/                     # stage 2 output — same names, cleaned
        │   ├── 00-front-matter.md               # (the text stages 3-5 read by default)
        │   ├── …
        │   ├── _preprocess_manifest.json     # per chapter: operations, fixes, provenance
        │   └── _preprocess_trace.json        # the full per-chapter audit trail
        ├── 03-reduced/                          # stage 3 output — sentences worth mining
        │   ├── 00-front-matter.md
        │   ├── …
        │   ├── _reduced_manifest.json        # per chapter: kept/dropped, entities, nouns
        │   └── _reduced_trace.json           # + examples of what each filter dropped
        ├── 03-reduced-unique/                   # stage 3, second step — first mentions only
        │   ├── 00-front-matter.md
        │   ├── …
        │   └── (same filenames; no manifest of its own — see 03-reduced/)
        ├── 04-glossary/                         # stage 4 output — one book-wide dictionary
        │   ├── glossary.json                 # term / category / locked translation
        │   ├── glossary.md                   # the same, as a review table
        │   ├── _glossary_manifest.json       # per chapter: tokens, attempts, conflicts
        │   └── _glossary_trace.json          # + every entry and raw answer
        ├── 05-chunks/                           # stage 5 output — flat, globally ordered
        │   ├── 00000-00-front-matter-0000.md # <global>-<chapter>-<index>.md
        │   ├── 00001-01-predgovor-0000.md
        │   ├── …
        │   ├── _chunk_manifest.json          # per chunk: chapter, order, spans, tokens, boundary
        │   └── _chunk_trace.json             # + per-chapter logs and the verification report
        ├── 06-chunk-glossary/                   # stage 6 output — one file per chunk
        │   ├── 00000-00-front-matter-0000.md.json  # named after the chunk it describes
        │   ├── 00001-01-predgovor-0000.md.json
        │   ├── …
        │   ├── _chunk_glossary_manifest.json # per chunk: NER counts, matched terms
        │   └── _chunk_glossary_trace.json    # + every term, candidate and log
        ├── 06b-matter/                           # stage 6b — the plan and its audit trail
        │   ├── book-plan.json                # localized title, credit, omissions, sections
        │   ├── _matter_manifest.json         # per matter chunk: chars, omitted, tokens, hash
        │   └── _matter_trace.json            # + every chunk's log and verification
        ├── 06b-matter-prompts/                   # stage 6b debug — the whole agent input
        │   ├── _book-plan.json               # the plan agent's full conversation
        │   ├── _book-plan.prompt.md
        │   └── 00000-…-0000.md.json          # messages, plan, source, answer per chunk
        ├── 07-translations/                     # stages 6b + 7 output — the translated chunks
        │   ├── 00000-00-front-matter-0000.md # stage 6b (localized matter; same names)
        │   ├── 00001-01-predgovor-0000.md
        │   ├── …
        │   ├── _translation_manifest.json    # per chunk: chars, tokens, attempts, hash
        │   └── _translation_trace.json       # + terms, logs and the run's totals
        ├── 07-translation-prompts/              # stage 7 debug — the whole input per call
        │   ├── 00001-01-predgovor-0000.md.json        # system+user prompt, messages,
        │   │                                          # glossary, source chunk, answer
        │   └── 00001-01-predgovor-0000.md.prompt.md   # the same prompt, human-readable
        ├── 08-translated-chapters/              # stage 8 output — one file per chapter
        │   ├── 00-front-matter.md            # same names as 01-chapters/, never over them
        │   ├── 01-predgovor.md
        │   ├── …
        │   ├── _translated_chapters_manifest.json  # per chapter: chunks, chars, hash, state
        │   └── _translated_chapters_trace.json     # + every chapter's record and log
        ├── 09-postprocessed-chapters/           # stage 9 output — never edits its input
        │   ├── 00-front-matter.md            # same names; title/heading fixes applied
        │   ├── 00-table-of-contents.md       # only when a new TOC had to be created
        │   ├── 01-predgovor.md
        │   ├── …
        │   ├── _postprocess_manifest.json    # plan, per-chapter decisions, TOC, hashes
        │   └── _postprocess_trace.json       # + every chapter's log and verification
        ├── 10-translated-book.md                # stage 10 — the whole book as one file
        │   # _bind_manifest.json / _bind_trace.json sit next to it
        └── 11-translated-book.epub               # stage 11 — the same book, as EPUB 3
            # _epub_manifest.json / _epub_trace.json sit next to it
```

Stage 2 never overwrites stage 1's files, and stage 3 never overwrites either:
`01-chapters/` stays exactly as the book came out of the splitter, so every repair
and every reduction can be re-derived and re-checked. `03-reduced-unique/` is a
subset of `03-reduced/`, and both are described by `03-reduced/_reduced_manifest.json`,
so the smaller folder can always be traced back to the sentences it dropped.

Stage 5 writes to its own sibling folder and never edits a chapter. Chunk files are
plain text slices, so a `cat` in filename order rebuilds the book and a `cat` of
one chapter's range rebuilds that chapter — the manifest is the index that tells
you which range that is. The global order is the *first* field of every filename
precisely so a directory listing is already in reading order.

Stage 6 also writes a sibling folder (`06-chunk-glossary/`) and never edits a chunk.
Its file names repeat the chunk's full name plus `.json`, so the mapping is
unambiguous and a consumer strips `.json` to find the text it describes. The
book-level index is `_chunk_glossary_manifest.json`; the per-chunk files are the
thing a translation prompt reads.

Stage 6b writes its localized matter chunks into `07-translations/` as well, so
`05-chunks/` and `07-translations/` still line up one to one and stage 8 needs no new
join rule. Its own two siblings are the plan and the debug trail: `06b-matter/` holds
`book-plan.json`, the per-chunk manifest and the trace, while `06b-matter-prompts/`
holds the whole conversation of the book-plan agent and one JSON/prompt record per
matter chunk. The source `01-chapters/` and the body translation prompts are never
touched.

Stage 7 writes two more siblings and edits nothing. `07-translations/` mirrors
`05-chunks/` file for file, so the translated book is a `cat` in filename order or a
per-chapter concatenation exactly like the source. `07-translation-prompts/` is the
debug twin: for each chunk, `<chunk>.md.json` holds the system prompt, the user
prompt, the serialised messages, the glossary rows that were sent, the source chunk
and the answer (or the failure), and `<chunk>.md.prompt.md` holds the same two
prompts in a form a human can read in an editor.

Stage 8 writes the last sibling, `08-translated-chapters/`, and edits nothing. It
groups `07-translations/<chunk>.md` by the chapter stage 5's manifest assigns each
chunk to, concatenates each group in `chunk_index` order, and writes one file per
chapter under the source chapter's own name. The source `01-chapters/` folder is
never touched, and a chapter is written only when *every* one of its chunks was
translated: a partial translation is reported as an incomplete chapter instead of
being written as a plausible-looking truncation. `_translated_chapters_manifest.json`
records each chapter's chunk count, character counts, SHA-256, output path and
whether it was written or reused.

`data/input/` and `data/output/` are kept in the repository with a `.gitkeep`; their contents
are ignored, so a book and its output never end up in a commit. The layout is resolved in one
place — [book_translator/paths.py](book_translator/paths.py) — and can be overridden with
`--in-dir` and `--out` or from Python through `book_translator.paths`.

## Result on the sample book

```
$ python -m book_translator split Aleksandar_Loven_-_Ljubav_seks_i_vase_srce.md
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
[08-glava-5-strah-od-ljubavi.md](data/output/Aleksandar_Loven_-_Ljubav_seks_i_vase_srce/01-chapters/08-glava-5-strah-od-ljubavi.md)).

Stage 2 then cleans those files, one at a time:

```
$ python -m book_translator preprocess
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
$ python -m book_translator dictionary           # on the sample book, 16 chapters
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
$ python -m book_translator glossary --out data/output/Dont-You-Forget-About-Me---Alexandra-Potter/chapters --limit 2
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

**Stage 1 — split** (`book_translator.stages.agent`):

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
$ python -m book_translator split Irvin_D._Jalom_-_Gledanje_u_sunce.md
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
python -m book_translator split Kenneth_M._Cameron_-_Zima_u_hotelu_smrti.md \
    --map data/output/Kenneth_M._Cameron_-_Zima_u_hotelu_smrti/01-chapters/_recovered_map.json
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

**Stage 2 — preprocess** (`book_translator.stages.preprocess_agent`), run once **per chapter**:

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

The folder it writes is not a dead end: the pipeline treats `02-preprocessed/` as
the chapter text for stages 3–5, so the dictionary, the glossary and the chunks
all read the repaired lines. The folder keeps the stage-1 filenames and order, and
`_preprocess_manifest.json` carries the same per-chapter fields as `_manifest.json`
(title, source line range, lines, bytes, hashes), which is why
`load_split_manifest`/`discover_chapters` can read either folder. `--no-preprocess`
keeps both the cleaning and that switch off; `--text-in 01-chapters/` keeps the
cleaning but sends the later stages back to the raw text.

### What stage 2 actually repairs

| Damage in the sample book | Detection | Repair |
| --- | --- | --- |
| `## 18`, `## 101`, `## 790` — the printed page number became a heading | header body is only a number | `remove` the line |
| `19`, `21`, `22` — a page number left as a plain body line | body line is only a number | `remove` the line |
| `- 2 -`, `- 208 -`, `— 208 —` — the dashes printed around the page number survived, as a header (`## - 18 -`) or as a plain body line | the line is only a number framed by matching leading and trailing dashes | `remove` the line |
| `- 1 3 -`, `- 1 0 4 -`, `- 3 3` — the conversion put a space between every digit | the line is only digits, with whitespace allowed between them | `remove` the line |
| `- \_ 1 1 \_`, `\_ 1 3 \_` — the underscores printed around the page number survived, escaped or not | the line is only a number framed by a pair of underscores (optionally with dashes outside them) | `remove` the line |
| `- 2 0 "Well`, `… to be mean to him the - 6 8 -`, `… a year. - 117 "Obviously,"`, `… Valerie 265` — a page number merged into a text line | a dashed token inside a line, or a bare number at a line edge, listed with the words on both sides and the nearby printed page markers | `inline_page_number` — delete only the token, keep every word around it |
| `## Slika 7. Protok uzbuđ enja u telu.`, `Slika 4. Blokada ljubavi.` — a figure caption, marked or not | caption marker + number, short or single-sentence body | keep the caption (`keep` in body text, `demote` when it was marked up as a heading); the picture itself is preserved by stage 0 |
| `## GLAVA 2` / `## SEKS I SRCE` — one heading split over two lines | structural label with no title, next line a short header | `merge_with_next` |
| `## … i želja za smrć` / `## u` — the first half cut mid-word | unfinished tail continued by the next header | `merge_with_next` |
| `JUTRO`, `POPOLE`, `ČETVRTAK, 16. AVGUST 2012.` — a real ALL-CAPS section heading the conversion left as plain body text | standalone ALL-CAPS line, short and title-shaped, inside a body chapter (never a dedication/title page) | `promote` — add the `##` markup, keep every word |
| `… psi182 ljutnje` — a page number printed *inside* a word | lowercase, digits, lowercase | drop the digits, glue the word (`psiljutnje`) |
| `1otkucaja u minuti` — a page number glued to the next word | digits at the start of a line, fused to a lowercase word | drop the digits (`otkucaja`) |
| `mišić ,` — whitespace before punctuation | space before `, . ; : ! ?` | delete the space |

The rules are deliberately narrow. `brzinu od 130 otkucaja`, `Tiraž 1000` and
`(TxA2)`/`(PG12)` are never touched, because an inline number is separated from
its word by a space and is therefore not the shape of a page break. A wrong
`remove` can only ever delete a line that is *nothing but a page number* — bare
digits, digits spaced apart by the conversion (`- 1 0 4 -`), or digits framed by
matching dashes or underscores (`- 18 -`, `— 208 —`, `- \_ 1 1 \_`). A one-sided
unspaced dash (`- 2`) stays a Markdown list item, not an artefact; a one-sided
dash around spaced digits (`- 3 3`) is accepted, because that is a page number
shape rather than list content. A wrong
`remove_caption` can only ever delete a line that opens with a caption marker and
a number, is at most 200 characters or a single sentence, and does not read as
prose about the figures (`Slike 12a i 12b ilustruju …` is left alone). It is kept
in the vocabulary for old plans, but the agent prompt no longer uses it: with
images preserved, captions are part of the book and stay with their picture.
`promote` can only add `##` to a standalone ALL-CAPS line in a body chapter and
changes no word. The verifier re-checks all these shapes after the fact.

**Pictures are never a cleaning decision.** A line that carries a Markdown image
(`![…](…)`) or an HTML `<img …>` tag is not prose: the validator rejects any
`remove`, `demote`, `merge_with_next` or text fix aimed at it, and the verifier
compares the before/after image-reference list, in order, so a plan cannot lose
or reorder a picture even by accident. Caption lines are kept (see above), which
is what puts the illustration and its label next to each other in the final
book.

A page number that was merged into a text line is the one case the rule does not
decide alone. The damage scan (`inspect_chapter_damage`) lists every candidate
with the token, its position (`prefix` / `suffix` / `middle`), a few words before
and after it on the line, the nearest prose line on either side, and whether the
number falls between the printed page markers around it (`fits_page_sequence`).
The agent then chooses: a dashed token that fits the page sequence is a page
number (`suggested: remove`), while a bare number that does not fit
(`... daily 100`, `... do it 844`) is prose and stays. The `inline_page_number`
fix deletes only that token and normalises the whitespace around it; every word
before and after remains.

Removed lines leave a **blank slot** rather than shortening the file, so a line
number in the log, the manifest or the trace still points at the same text in
`01-chapters/`.

#### Final phase — the chapter-opening audit (`book_translator.stages.openings`)

Every per-chapter decision is local, so after the folder is written the stage runs
one deterministic pass over the **whole book's openings** — the first lines of
every chapter, and nothing else. It asks one question: does the opening read like
a chapter opening? A title followed by a stray copy of its number
(`# Poglavlje 2` then `2`), a line that repeats the title and a printed page
number that survived next to the heading are provable junk, so they are removed
immediately. A title that is only a number, a title that is only `Poglavlje` and a
chapter that opens on prose are **reported, never deleted** — a title is a naming
decision (stage 9's book plan), not a line edit.

The pass keeps stage 2's semantics: a removed line leaves an empty line, so the
stage's line numbers stay valid. `PreprocessResult.openings` (also in
`_preprocess_manifest.json` and `_preprocess_trace.json`) records what was checked,
what was clean, what was removed and what still needs a human. `--dry-run` reports
the same findings and writes nothing.

### Stage 3 — dictionary (`book_translator.stages.dictionary`), run once **per chapter**

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
03-reduced/<same-filename>.md one raw sentence per line, no NER markup
  │
noun index                 the nouns of every kept sentence (case-folded surface
  │                        form, entity type ignored), carried across the whole book
  │
first-mention filter       keep the sentence only if at least one of its nouns has
  │                        not been seen in an earlier sentence — `03-reduced/` says
  │                        "Marko" nine hundred times, a dictionary needs it once
  │
03-reduced-unique/<same-filename>.md   the same shape, the first mentions only
```

On the sample book the two free filters alone leave **15 %** of the sentences for the
tagger — 640 of 4 271 — which is what makes the NER pass cheap (3.3 s on CPU for the
whole book). Earlier measurements on the two bigger Serbian novels:
870 of 4 002 for *Gledanje u sunce*, 2 938 of 14 180 for *Zima u hotelu smrti*.

The last step costs nothing on top of that: it reuses the entity mentions the gate
already found, so `03-reduced-unique/` comes out of the same pass and the same tagger
run. It is **book-wide and ordered** — chapters are visited first to last and
chapter 3's `Marko` suppresses chapter 9's — and its rule is *"at least one new
noun"*, not *"no repeated noun"*, so a first mention that happens to share a
sentence with a known name is not thrown away. Both folders are written by default
and switched independently: `--no-save-unique` keeps only `03-reduced/`,
`--unique-only` keeps only `03-reduced-unique/`.

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

### Stage 4 — glossary (`book_translator.stages.glossary`), run once **per chapter, in order**

This is the LLM pass stage 3 was built for. One call per chapter, first to last,
each call handed the dictionary the earlier chapters produced:

```
03-reduced-unique/01-chapter-1.md  call 1: dictionary empty
   │                                   -> extract every entity, lock a rendering
   │                                      for each: term, category, translation
   │                                                             (Pydantic-validated)
   ▼
dictionary so far (chapters 1..n)
   │
03-reduced-unique/02-chapter-2.md  call 2: prompt = dictionary so far + this chapter's
   │                                   sentences; the model is told the locked
   │                                   renderings are final and to return only what
   │                                   is new
   ▼
… every body chapter, in reading order …
   │
04-glossary/glossary.json            the dictionary: one entry per entity, with the
04-glossary/glossary.md              chapter it came from
04-glossary/_glossary_manifest.json  per chapter: tokens, attempts, entries added
04-glossary/_glossary_trace.json     + the entries added and the conflicts recorded
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
(`book_translator/stages/glossary/models.py`):

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

### Stage 5 — chunking (`book_translator.stages.chunking`), no model at all

A chapter file is fine for a reader and far too big for a translation call: a
100-page chapter blows through a context window, and cutting it at a fixed
character count breaks words, sentences and paragraphs. Stage 5 turns each
chapter into files a translator can consume one at a time:

```
01-chapters/01-predgovor.md
   │  chonkie RecursiveChunker, preference order:
   │      paragraph  →  sentence  →  line  →  whitespace  →  hard cut
   ▼
05-chunks/00000-01-predgovor-0000.md   exact slice [start_char:end_char)
05-chunks/00001-01-predgovor-0001.md
…
05-chunks/_chunk_manifest.json         chapter, global order, spans, tokens, boundaries
```

The length is measured in **tokens**, not characters: `--chunk-size` defaults to
`1000` and is a hard maximum (a chunk that a single unsplittable token pushed over
the limit is split by the stage itself, at a source-slice boundary). The default
tokenizer is `cl100k_base`, built by `tiktoken`; when `tiktoken` (or the requested
tokenizer) cannot be loaded — no package, no vocabulary cache, an offline machine
— the stage falls back to a local four-characters-per-token estimate and records
the fallback and its reason in the manifest. It never downloads a model in the
middle of a run and never fails a book because a vocabulary did not load.

**A picture is never cut in half.** If chonkie's boundary lands inside a
Markdown image reference, the stage moves that one cut to the image's left or
right edge (whichever keeps both chunks within the token limit). The chunk files
are still exact slices of the chapter, but a translator now always sees a whole
`![…](…)` reference rather than a path fragment split across two calls.

**Reversibility is the contract.** The text written for a chunk is
`chapter_text[start_char:end_char]`, with no trimming, no re-wrapping and no
normalisation. The run verifies that every file is exactly that slice and that
concatenating a chapter's chunks in `chunk_index` order reproduces the chapter
byte for byte; `verification.chapters_reconstruct_byte_for_byte` and
`verification.book_reconstructs_byte_for_byte` in the manifest are that proof.
The read side is part of the package:

```python
from book_translator.stages.chunking import reconstruct_book, reconstruct_chapter

reconstruct_book("data/output/<book>/05-chunks")                    # == the whole book
reconstruct_chapter("data/output/<book>/05-chunks", "01-predgovor.md")
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

### Stage 6 — chunk glossary (`book_translator.stages.chunk_glossary`), two phases and no LLM

Stage 4 locked one rendering per entity for the whole book. Stage 5 cut the book
into pieces a model can translate. The missing question is the one a translator
asks before sending chunk 137: *which of those locked renderings does this chunk
actually mention?* Stage 6 answers it, and it refuses to answer it one chunk at a
time.

```
05-chunks/00000-01-predgovor-0000.md ─┐
05-chunks/00001-01-predgovor-0001.md ─┤  PHASE 1 — NER over every chunk
…                                  ├─ stage 3's CLASSLA/stanza tagger
05-chunks/00042-08-glav…-0013.md ─────┘  → per-chunk distinct PER/LOC/ORG/MISC terms
                                      (no glossary lookup yet)
                                              │
                                              ▼  only after the last chunk is tagged
                                      PHASE 2 — fuzzy retrieval
                                      rapidfuzz scores every term against
                                      glossary.json; ≥ threshold wins
                                              │
                                              ▼
06-chunk-glossary/00000-01-predgovor-0000.md.json
06-chunk-glossary/00001-01-predgovor-0001.md.json
…
06-chunk-glossary/_chunk_glossary_manifest.json
06-chunk-glossary/_chunk_glossary_trace.json
```

**The order is the contract.** The NER pass finishes for **all** chunks before the
first glossary lookup. That is not an optimisation; it is what makes the stage one
complete picture instead of a chunk-at-a-time guess, and the tests pin it by
recording the order of the injected tagger and matcher. The run reports the two
phases separately (`phases[].name` is `ner` then `match`), with their own seconds
and counts.

**The tagger is stage 3's.** `book_translator.stages.dictionary.ner` builds the backend for
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
python -m book_translator chunk-glossary mybook
python -m book_translator chunk-glossary mybook --match-threshold 90
python -m book_translator chunk-glossary mybook --scorer token_sort_ratio
python -m book_translator chunk-glossary mybook --limit 3        # a quick look
python -m book_translator chunk-glossary mybook --dry-run        # both phases, no writes
python -m book_translator chunk-glossary --chunks-in /tmp/05-chunks/ \
    --glossary-in /tmp/04-glossary/ --chunk-glossary-out /tmp/06-chunk-glossary/
```

`--chunks-in` accepts the chunks folder, the book folder, or
`_chunk_manifest.json`; `--glossary-in` accepts the glossary folder or
`glossary.json`. Without them the stage reads `05-chunks/` and
`04-glossary/glossary.json` beside the book's `01-chapters/` folder. `rapidfuzz` is the
only new dependency, and it is imported lazily — a missing wheel is one
`pip install` line, not a traceback.

On the English novel in `data/input/` (`--limit 2` tags the first two chunks):

```
$ python -m book_translator chunk-glossary --out …/chapters --language en --cpu --limit 2
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

### Stage 6b — front/back matter localization (`book_translator.stages.matter_agent`), no dictionary

Front matter is not a chapter, and back matter is not an epilogue. They are the
parts of a *published edition*: the title page, the translator credit, the
copyright/imprint page, the table of contents, the author's preface, the notes and
the "about the author". Translating them with the body prompt is wrong in three
ways — the dictionary may contain publisher/place names that do not belong in the
translated edition, the title may have a published target-language rendering that
must be reused, and much of the original imprint (ISBN, CIP/legal deposit,
printing history, original publisher's permissions) should be omitted, not
translated.

Stage 6b is a separate LangGraph agent that runs **after chunking and before
translation**. Before it inventories the matter, it makes sure both ends of the
book exist: the input filename is read as `Title - Author` (or `Title by Author`),
and `00-front-matter.md` / `99-back-matter.md` are created with matching chapter
and chunk records when the source book carried neither. Those generated sections
are localized like any other matter section; the title page carries the filename
title and the BleBel credit, and the back page opens an "about the author" section.
The same title/author hints are sent to the planning agent.

It reuses the splitter's `web_search` / `fetch_web_page` tools and the shared
structural tools, and adds the readers and submission tools it needs:

```
START
  │
survey              deterministic matter inventory: front/back files, titles, chunks   (no LLM)
  │
book_agent ◄──────┐  plan: search the published title, fix the BleBel credit,
  │  ╲            │        list omitted source phrases, set section titles
  │   ▼           │
  │  book_tools ──┘  list_matter_sections / read_matter_section / list_book_chapters /
  │                 web_search / fetch_web_page / validate_matter_plan_tool
  │
book_validate ───► book_agent   deterministic: title required, credit names BleBel,
  │                              omitted phrases exist and are enforceable
  ▼
for every matter chunk, in reading order:
  collect           exact chunk + the binding plan                            (no LLM)
    │
  chunk_agent ◄──┐  localize this chunk as front/back matter, in Markdown
    │  ╲         │
    │   ▼        │
    │  chunk_tools ──┐  read_matter_section / read_matter_chunk / web_search /
    │                │  validate_matter_localization
    │  chunk_validate │  non-empty (unless explicitly omitted), title/credit on the
    │                │  first front-matter chunk, omitted phrases really gone
    │  apply         │  write 07-translations/<chunk>.md (empty for an omitted chunk)
    │  verify        ┘  prove the file holds exactly the validated text
    ▼
07-translations/<chunk>.md + 06b-matter/_matter_manifest.json + 06b-matter-prompts/<chunk>.json
```

The **book plan** is the contract, and the prompt is explicit about it:

* **the title** — search the web for an existing published translation first; reuse
  its exact title and record the URL in `title_evidence`. Only when no published
  translation exists, translate the title appropriately. If the search provider is
  unavailable, the agent stops after one attempt and plans from the local title and
  author hints; a deterministic filename plan is the last fallback, so a missing
  network never leaves the book without a title page.
* **the credit** — the front matter must say, in the target language, that the
  edition was translated by **BleBel**. The deterministic formatter puts
  `# <translated title>` and the credit line on the first front-matter chunk if
  the model did not.
* **the omissions** — original imprint, ISBN/barcode, CIP/legal deposit, printing
  history, original copyright/permissions, prices and print-edition references are
  listed as exact source phrases in `omit`; the validator rejects a translation
  that still contains one.
* **the formatting** — the chunk prompt asks for published Markdown: one `#` title,
  an italic credit line, blank lines between blocks, `##` section headings, lists
  for an epigraph/contents, and no code fence around the whole answer.

`discover_matter_sections` uses stage 9's classifier, so `01-predgovor.md`,
`02-uvod.md`, `03-beleske.md`, `NN-pogovor.md` and the explicit `00-front-matter.md`
/ `NN-back-matter.md` files are recognized even when the filename looks numbered.
The stage translates **no body chunk**: stage 7 filters the matter chapters out by
default and reports how many it left to this stage. `--no-matter` drops the stage,
and `--include-matter` makes stage 7 translate matter chunks with the body prompt
(the old behaviour).

```bash
python -m book_translator matter mybook                 # -> 06b-matter/ + localized chunks
python -m book_translator matter mybook -s sr -t en     # another direction
python -m book_translator matter mybook --limit 3       # first three matter chunks
python -m book_translator matter mybook --resume        # reuse finished matter chunks
```

A resume is safe: a chunk is reused only when the manifest records it as
successful, its source SHA-256 still matches, the language pair is unchanged, and
the translated file exists (an omitted chunk is a legitimate empty file).

### Stage 7 — translation (`book_translator.stages.translation`), one chunk at a time

Stage 5 cut the book into pieces a model can translate; stage 4 locked one
rendering per entity; stage 6 attached exactly the renderings each piece mentions.
Stage 7 is the pass those three artifacts exist for. It translates the **body**
chunks: the front/back matter chunks are skipped by default because stage 6b
already localized them with a different prompt and no dictionary
(`--include-matter` restores the old all-chunks behaviour):

```
05-chunks/00042-08-glav…-0013.md ───┐
06-chunk-glossary/00042-….md.json ──┤  one prompt:  system (constant)
                                 ├─              + glossary rows (binding)
                                 │              + previous translation tail (optional)
                                 │              + the chunk itself
                                 ▼
                       structured-output call  →  {"translation": "..."}
                                 │        Pydantic-validated, retried on failure
                                 ▼
07-translations/00042-08-glav…-0013.md           the translated chunk
07-translation-prompts/00042-….md.json           the whole input of the call
07-translation-prompts/00042-….md.prompt.md      the same prompt, readable
07-translations/_translation_manifest.json       per chunk: chars, tokens, attempts, sha
07-translations/_translation_trace.json          + terms, logs and the run's totals
```

**The order is the book's order.** Chunks are translated first to last, one call at
a time. A chunk's translation does not depend on the next chunk, but the run's
output is a *prefix* of the book at every moment: kill it after chunk 90 and
`07-translations/` holds chapters 1–90 translated, and `--resume` continues at 91.
When `--context-chars N` is set, the last N characters of the previous translation
are sent as continuity context (clearly marked "do not translate it again"); the
default is 0, because the chunker already cuts on paragraph/sentence boundaries and
the glossary is the consistency mechanism.

**The glossary is per chunk and binding.** The prompt opens with stage 6's
`matched_terms` — `Džordž → Džordž`, `London → London`, `Facebook → Fejsbuk` — and
the system prompt says the renderings are final: a name may be declined for
grammar, its stem may not change. `--glossary-scope book` sends the whole
`glossary.json` with every chunk instead (more consistent, much more input tokens).

**Pictures are opaque markers to the model.** A chunk never sends a real image
path to the LLM. Every `![…](…)` (and every raw `<img …>`) is replaced by a
unique `[[IMG_n]]` marker before the prompt is built; the system prompt says to
copy each marker exactly once in its original place. When the answer comes back,
the marker is restored to the original reference and the chunk is written. The
translator rejects a dropped, duplicated or invented marker exactly like an
empty answer — it is retried, and the chunk fails rather than publishing a book
with a missing illustration. A chunk that contains nothing but pictures is
passed through unchanged with no model call. The marker map is kept in
`07-translation-prompts/<chunk>.md.json` under `images.markers`, next to the
protected prompt and the original source chunk.

**Every call is structured and validated.** The model is asked for a JSON object
with one field:

```json
{"translation": "the complete translated chunk"}
```

`ChunkTranslationAnswer` is Pydantic-validated before anything is written, and a
call is retried (default `--max-attempts 3`) when the answer is wrapped in prose, an
empty string, the source unchanged, or a fraction of the source's length (a summary
instead of a translation). Each attempt after the first carries a repair note naming
the failure. `--method auto` uses native structured output and moves to JSON mode
when a provider rejects the schema; `--translation-method json` starts in JSON mode.
Every attempt's tokens are counted, so a chunk that needed three tries is *priced*
as three tries, and a chunk that still fails is reported — the run continues and
`07-translations/` simply has one hole, visible in the manifest.

**The whole input is kept.** For every chunk, the debug folder holds:

| File / field | What it is |
| --- | --- |
| `07-translation-prompts/<chunk>.md.json` | one structured record per call |
| `.prompt.system` | the constant system prompt (translator role, rules, JSON shape) |
| `.prompt.user` | the per-chunk prompt: book, chunk identity, glossary, context, source, task |
| `.prompt.messages` | the same two prompts as the `[{role, content}]` list sent to the model |
| `.prompt.chars` / `.prompt.sha256` | prompt size and a hash, so a prompt can be diffed between runs |
| `.glossary.entries` | every binding row that was actually in the prompt |
| `.source_chunk.text` | the exact chunk the call was made about |
| `.answer.text` / `.answer.usage` | the accepted translation and what it cost (or `null` + `.error`) |
| `07-translation-prompts/<chunk>.md.prompt.md` | `# System prompt` and `# User prompt`, readable |

The prompt markdown is written *before* the call, so even a chunk whose request
hangs or dies has its exact input on disk.

**Running it.** The command derives every input and output from the book folder:

```bash
python -m book_translator matter mybook                    # first: localize front/back matter
python -m book_translator translate mybook                 # en -> sr, every body chunk
python -m book_translator translate mybook -s sr -t en     # the other direction
python -m book_translator translate mybook --limit 3       # translate three chunks first
python -m book_translator translate mybook --first-chapters 3  # first three chapters' chunks
python -m book_translator translate mybook --resume        # skip finished chunks, continue
python -m book_translator translate mybook --glossary-scope book
python -m book_translator translate mybook --context-chars 400
python -m book_translator translate --chunks-in /tmp/05-chunks/ \
    --chunk-glossary-in /tmp/06-chunk-glossary/ --translations-out /tmp/07-translations/ \
    --prompts-out /tmp/07-translation-prompts/
```

**Choosing the model: DeepSeek or a local LM Studio.** Stage 7 is the only stage
with its own two-way provider choice. DeepSeek is the default; `--translator
lmstudio` sends the body chunks to the OpenAI-compatible server you host at
`http://127.0.0.1:1234/v1` instead:

```bash
python -m book_translator translate mybook                        # deepseek/deepseek-flash
python -m book_translator translate mybook --translator lmstudio  # auto-detected local model
```

With no `--model`, the loaded id is read from the server's `GET /v1/models`
(a translation-tuned `en2sr` id is preferred when several models are loaded), and
no API key is needed. `--base-url` points at another port. The choice is
**translation-body only**: stage 6b's front/back-matter localization and stages 1,
2, 4 and 9 keep the rest of the configuration, so switching stage 7 does not move
the rest of the pipeline to the local model. The same choice lives in
[book_translator.toml](book_translator.toml) as the `[llm.translate]` table, or on one
command line as `--llm translate.provider=lmstudio` (see
[Configuration](#configuration)). `TRANSLATION_PROVIDER`, `TRANSLATION_MODEL` and
`TRANSLATION_BASE_URL` still work as stage-specific environment overrides.

`--chunks-in` accepts the chunks folder, the book folder or
`_chunk_manifest.json`; `--chunk-glossary-in` accepts stage 6's folder (or the book
folder, whose `06-chunk-glossary/` is then used). A run with no per-chunk glossary
fails with the command that produces it, unless `--glossary-scope book` is chosen.
The command prints a live progress bar with the completed-chunk count, elapsed time,
an ETA, the tokens spent so far and the chunk now in flight, then one durable line
per chunk and a summary with the **total tokens used** (input, output, calls) and
the characters translated. `--dry-run` asks the model for every chunk and writes
nothing; `--quiet` drops the header, the bar and the per-chunk lines.

`--resume` (`--skip-if-present`) is safe: a chunk is reused only when the manifest
records it as successful, its translation file is non-empty, the source chunk's
SHA-256 still matches, and the source/target language pair is unchanged. The
manifest is rewritten after **every** finished chunk, so an interrupted run resumes
without paying for the prefix again. `--limit N` applies first, so `--limit 3
--resume` is the natural "finish the first three, then stop" command.

`--first-chapters N` is the chapter-aligned variant: it reads stage 5's manifest,
takes the first N **body** chapters in reading order (front matter and a marked
back-matter block do not consume one of the N), and translates exactly the chunks
that belong to them. Unlike `--limit`, it never stops in the middle of a chapter,
so the result is always a clean chapter prefix. Translated that way, stage 8 can
assemble those chapters without any holes. Combining it with `--resume` is how a
run grows: `--first-chapters 5 --resume`, then `--first-chapters 10 --resume`,
reuses the first five chapters and pays only for the new five.

An illustrative two-chunk run (`--limit 2`) prints:

```
$ python -m book_translator translate --out …/chapters --limit 2
language=en -> sr  model=deepseek/deepseek-flash  method=auto  attempts<=3  scope=chunk
  [  1/2] ok    00001-01-chapter-1-0000.md      4,135 -> 4,290 chars ·   2,804 tok · 1 attempt(s) ·   9.8s
  [  2/2] ok    00002-01-chapter-1-0001.md      4,056 -> 4,200 chars ·   2,690 tok · 1 attempt(s) ·   9.4s

chunks          : 2/2 ok
glossary        : per-chunk matched terms · 20 row(s) sent
text            : 8,191 -> 8,490 characters
tokens used     : 5,494 total (4,180 in, 1,314 out) over 2 call(s), 2 chunk(s) asked
time            : 19.2s  (model 19.2s)  9.6 s/chunk
output dir      : …/translations
prompts         : …/translation-prompts
OK — 2 chunk(s) translated; concatenate 07-translations/ in filename order to rebuild the translated book.
```

Because the translation keeps the source chunk's trailing newlines, a
`cat 07-translations/*.md` in filename order preserves the book's paragraph breaks.
The stage adds no dependency: it reuses `langchain-openai` and Pydantic, exactly
like stage 4.

### Stage 8 — assembly (`book_translator.stages.assembly`), the translated chunks become chapters

Stage 5 proved that a chapter is its chunks concatenated; stage 8 applies the same
concatenation to the *translated* chunks, using stage 5's own index instead of
guessing from filenames:

```
05-chunks/_chunk_manifest.json        "chunk 00042 belongs to 08-glav…, index 13"
        │
        ├── 07-translations/00042-08-glav…-0013.md   ┐
        ├── 07-translations/00043-08-glav…-0014.md   ├─ concatenate in chunk_index order
        └── …                                     ┘
        ▼
08-translated-chapters/08-glav…md     one file per chapter, source names kept
```

**All or nothing per chapter.** A chapter file is written only when every chunk
that belongs to it has a translation. A partial translation (for example
`translate --limit` stopping in the middle of a chapter) is reported as
`incomplete` and skipped, never written as a truncated chapter that looks
complete. `--first-chapters N` avoids the problem entirely by translating a
chapter-aligned prefix; `08-translated-chapters/` then contains exactly those
chapters.

**A separate folder, by design.** `01-chapters/` is the source text and is never
overwritten; `08-translated-chapters/` is the product and can be deleted and rebuilt
at any time. It is a sibling of `07-translations/`, so `--start-at concatenate` on an
earlier translation needs no other argument.

**Resumable.** `--skip-if-present` (`--resume`) compares each assembled chapter with
the bytes already on disk and reuses an identical file. Growing the window from N
to M therefore writes only the new chapters. The manifest records, per chapter,
its chunk count, translated/all chunk counts, character counts, SHA-256, output
path and whether it was written or reused.

Running it:

```bash
python -m book_translator concatenate mybook                  # every fully translated chapter
python -m book_translator concatenate mybook --first-chapters 5
python -m book_translator concatenate mybook --first-chapters 10 --resume
python -m book_translator concatenate --chunks-in /tmp/book/chunks \
    --translations-in /tmp/book/translations --chapters-out /tmp/translated-chapters
```

`--chunks-in` accepts the chunks folder, the book folder or
`_chunk_manifest.json`; `--translations-in` defaults to the chunks folder's
sibling `07-translations/`; `--chapters-out` defaults to `08-translated-chapters/`.
`--dry-run` assembles every chapter in memory and writes nothing. `--json` returns
the same records machine-readably.

### Stage 9 — postprocess (`book_translator.stages.postprocess_agent`), headings first, prose never

Stage 8 produces one translated file per chapter; stage 9 makes their **structure**
consistent. It is the third agent stage, and it is deliberately not a prose editor:

1. **Book pass.** `list_chapter_titles` inventories every chapter title, its line,
   whether it is already a Markdown heading, its number, front/back matter and the
   current TOC. The model reads the whole list first, then chooses **one naming
   policy** (`keep` or `normalize` with a label such as `Poglavlje`, an optional
   separator and the chapter's own title words) plus per-chapter overrides.
   A chapter's own words are recognised both in the label line
   (`Poglavlje 4: The Storm`) and on the line after a bare label (`Poglavlje 4`
   or `GLAVA DRUGA` followed by `THE STORM`, whether plain or a level-1 heading).
   When any body chapter carries such a name the stage enables the policy's
   `include_title` part itself, so a plan cannot silently replace the names with
   bare numbers in the headings and the TOC; `keep` likewise carries a detected
   subtitle into the title. Before choosing a policy the model **must** search the
   web for the published chapter list (`web_search`, then `fetch_web_page` on a
   promising publisher/library/bookstore page); the graph refuses
   `report_book_plan` until that call has happened. A published target-language
   contents list becomes exact `overrides`; a source-language one is only evidence
   for which local line is the name. The outcome is recorded in the plan's
   `title_evidence`, and a failed lookup still counts as the mandatory attempt.
   `inspect_matter_blocks` shows the bounded front/back matter and the TOC state.
   The plan is submitted through `report_book_plan`.
2. **Chapter pass, one chapter at a time.** `inspect_chapter_opening` returns at
   most **8 clipped lines** — the only place a chapter title can be.
   `inspect_chapter_headings` returns one record per heading with a deterministic
   suggestion (`keep`/`demote`/`promote`) and one line of evidence; ambiguous
   headings get `inspect_heading_context`. A real ALL-CAPS section heading whose
   `#` markup the translation lost (`JUTRO`, `POPOLE`,
   `ČETVRTAK, 16. AVGUST 2012.`) is also in the candidate list, with `level: 0`
   and `suggested: promote`; front/back matter is excluded. The agent then submits
   `report_chapter_postprocess`:
   - `title`: `keep`, `set` (rewrite the existing title line) or `insert` (the
     chapter has no title at all);
   - `headings`: `keep`, `demote` or `promote` for every other heading.
     **`demote` removes only the `#` markup and leaves the words as a paragraph;
     `promote` is its inverse and only adds the markup back** — there is no
     delete operation, so a mistaken heading can never delete text.
   When the book policy folds a subtitle into the title, the now-duplicate
   subtitle line is recorded as `subtitle_merged` and removed; its words are in
   the heading.
3. **TOC pass.** Entries are generated from the **final** titles. If the book has
   no TOC, a new `00-table-of-contents.md` is written, which sorts right after
   `00-front-matter.md` and before `01-…`. If a TOC exists, the detected block (or
   separate file) is rewritten in place; a `keep` plan is upgraded to an update if
   the stored entries no longer match.
4. **Verify.** The validator checks every title action and heading line against the
   real file; `verify_chapter_plan` proves that undeclared lines are verbatim and
   only declared headings were demoted or promoted; `verify_toc` proves every
   expected title appears once, in order.
5. **Opening audit (final phase).** After every chapter is written — reused ones
   included — the stage walks the first lines of all of them, the same deterministic
   pass stage 2 runs (`book_translator.stages.openings`). It removes a stray copy of the
   chapter number under the title (`# Poglavlje 2` then `2`), a duplicated title
   line and a leftover page number, collapsing the blank around the removed line,
   and reports what still needs a human (a title that is only a number, a chapter
   that opens on prose). It runs **before** the TOC, which is built from the final
   titles, and `PostprocessResult.openings` / `_postprocess_manifest.json` record
   the counts and the per-chapter findings.

   This is also why a translated untitled chapter is handled correctly. A
   translated `CHAPTER 2` can arrive as `## 2`: that first-line bare number is the
   chapter's **title** (the filename says chapter 2), not a printed page number, so
   the book plan `set`s it to `Poglavlje 2` instead of treating the chapter as
   title-less, inserting a second heading and demoting the number into a stray
   paragraph — the failure the audit now repairs even in folders written by an
   older run. `--resume` is title-aware for the same reason: a chapter whose stored
   title is no longer the title the stage reads today is reprocessed instead of
   reused, so a resume run heals a folder written before the fix.

The output is a **separate folder**, `09-postprocessed-chapters/`. The translated
chapters are input and are never edited in place. A chapter plan that fails
validation loops back with concrete issues; a chapter that never produces a valid
plan fails alone, not the book. In the one-command pipeline the stage is opt-in
(`--with-postprocess`, or `--start-at postprocess`), just like translation; the
standalone `postprocess` command is always available.

Running it:

```bash
python -m book_translator postprocess mybook                 # every translated chapter
python -m book_translator postprocess mybook --first-chapters 5
python -m book_translator postprocess mybook --resume        # reuse unchanged chapters
python -m book_translator postprocess mybook --dry-run       # validate every plan, write nothing
python -m book_translator postprocess --translated-chapters-in /tmp/book/translated-chapters \
    --postprocessed-out /tmp/book/postprocessed-chapters
```

`--first-chapters N` means the front matter plus the first N **body** chapters.
`--limit N` is the same chapter-aligned window. The chapter plan receives the
canonical title from the book plan, so a chapter cannot silently invent a name
that the rest of the book does not use.

The opening audit can also be run on its own, with no model and no key, over any
chapter folder — for a finished book, that is how an already-written
`09-postprocessed-chapters/` is repaired in place:

```bash
python -m book_translator openings mybook                    # the most advanced folder: 09 → 08 → 02 → 01
python -m book_translator openings mybook --dry-run          # report every finding, write nothing
python -m book_translator openings mybook --chapters-in /tmp/book/02-preprocessed --keep-blank-slot
python -m book_translator openings --chapters-in /tmp/book/09-postprocessed-chapters --json
```

`--keep-blank-slot` is stage 2's semantics (a removed line leaves an empty line,
so line numbers stay valid); the default is stage 9's (remove the line and collapse
the blank). Only a line that is nothing but a stray number or a copy of the title
is ever deleted.

### Stage 10 — bind (`book_translator.stages.binding`), one Markdown book

The final step is deterministic. `run_bind` reads
`09-postprocessed-chapters/*.md` in reading order — front matter, the separate TOC
(when there is one), the chapters and back matter — joins them with exactly one
blank line, and writes `10-translated-book.md` next to the folder. When the optional
postprocess pass was not selected, it binds `08-translated-chapters/*.md` directly,
so `--with-translate` always leaves one readable book; stage 9 only adds the
heading policy and table of contents. It re-reads the output and verifies every
input file appears once, in order. No model, no key.

```bash
python -m book_translator bind mybook                    # -> data/output/<book>/10-translated-book.md
python -m book_translator bind mybook --dry-run          # assemble and verify, write nothing
python -m book_translator bind --postprocessed-in /tmp/book/postprocessed-chapters \
    --book-out /tmp/book/10-translated-book.md
```

`--postprocessed-in` defaults to `data/output/<book>/09-postprocessed-chapters/`;
`--book-out` defaults to `10-translated-book.md` next to it. `_bind_manifest.json`
records every file, its hash and the final size.

**Image links are rebased once.** Chapter files live one folder below the book
folder and reference pictures as `../00-markdown/images/…`; the bound book lives
at the book root, so stage 10 rewrites that prefix to `00-markdown/images/…` as
it joins the files. The verification step applies the same rewrite to each input
before looking for it in the output, and `image_refs` in the result/manifest is
the number of pictures the final book points at.

### Stage 11 — EPUB (`book_translator.stages.epub`), the book a reader opens

`10-translated-book.md` is the reading copy for a human and the input for nothing;
a reader wants an EPUB. Stage 11 packages the **same** chapter folder as an EPUB 3
book, `11-translated-book.epub`, and it reads `09-postprocessed-chapters/*.md` —
not the bound Markdown — because one XHTML document per chapter is exactly what an
EPUB's spine wants. Re-splitting the single file on `#` headings would have to
guess those boundaries back.

```bash
python -m book_translator epub mybook                   # -> data/output/<book>/11-translated-book.epub
python -m book_translator epub mybook --cover cover.jpg # with a cover page
python -m book_translator epub mybook --title "Naslov" --author "Autor" --language sr
python -m book_translator epub mybook --dry-run         # build and verify, write nothing
```

`--postprocessed-in` defaults to `data/output/<book>/09-postprocessed-chapters/`,
`--epub-out` to `11-translated-book.epub` next to it. What the stage does:

- **One document per chapter.** The chapter's first `#` heading is the document
  `<title>` and the navigation label. `#` becomes `<h1>`, `##` an `<h2>` with an
  anchor, and the navigation nests the sections under the chapter that contains
  them.
- **No printed table of contents.** `00-table-of-contents.md` is skipped: every
  reader builds its own from the navigation document, and shipping the numbered
  list as well would give the reader two tables of contents, one of them dead.
  `toc_dropped` in the result says whether one was there.
- **Pictures are embedded once.** Each referenced picture is copied into
  `OEBPS/images/` and its `src` rewritten; ten references to the same picture
  become one file and ten links. A reference that cannot be resolved is an error
  in the result and a visible `[path]` placeholder in the book — the package is
  still written, so nothing is silently lost.
- **Metadata is re-read, not invented.** The title, author and target language
  stage 6b recorded in `06b-matter/_matter_manifest.json` become `dc:title`,
  `dc:creator` and `dc:language`; `--title/--author/--language/--identifier/
  --description/--publisher` override any of them. The identifier is a UUID
  derived from the book folder, so a rebuild is still the same publication.
- **Deterministic.** Zip timestamps come from the newest chapter file, entries are
  written in a fixed order, and item ids are positional: rebuilding unchanged
  chapters produces the **same bytes and the same SHA-256**.
- **Verified after writing.** The stage re-opens its own package and checks the
  mandatory first (uncompressed) `mimetype` entry, `META-INF/container.xml`, the
  package document (EPUB 3, metadata, manifest, spine), the navigation document
  (every entry points at a real document), every spine item's XHTML, and every
  `<img src>`. `_epub_manifest.json` records the verdict, the files, the picture
  hashes and the final size; `_epub_trace.json` records the whole result.

The package uses only the standard library for the container and `markdown-it-py`
for the Markdown-to-XHTML render, imported lazily: no pandoc, no calibre, no
ebooklib. The stylesheet is a small file shipped with the package
(`book_translator/stages/epub/assets/style.css`): justified prose with a first-line indent,
hyphenation driven by the document's `lang`, an `<hr>` scene divider that survives
every reader, and `max-width: 100%` on pictures.

The one-command pipeline runs stage 11 after stage 10 (`--no-epub` drops it,
`--start-at epub` runs it alone over an existing chapter folder, `--epub-cover`
adds a cover).

## Tools, grouped by stage

`python -m book_translator tools` prints exactly this grouping, and each stage
binds only its own tools plus the `shared` ones — so the splitting agent cannot
reach the cleaning tools, and it has no write tool at all: its chapter files are
written by the deterministic `slice` node after validation. The preprocessing
agent likewise cannot slice the book.
Stage 0 (`convert`) binds no tools and calls no model: it is one deterministic
docling pass plus the PDF text-layer preflight described above.

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
| `detect_toc_block_tool` | Locates the printed TOC block and its line range | ✔ |
| `propose_split_map` | The tool the agent *submits its answer* through | ✔ |
| `validate_split_map_tool` | Dry-run scoring of a plan before anything is written | ✔ |

### `matter` — stage 6b, its own agent and no dictionary

| Tool | Purpose |
| --- | --- |
| `list_matter_sections` | **the first step**: every front/back matter file, its kind, title, sizes, chunks and the body chapter titles (metadata only) |
| `read_matter_section` | the full text of one matter file (this stage is allowed to read prose — it localizes it) |
| `read_matter_chunk` | the exact current chunk, or an adjacent chunk by filename, for continuity |
| `list_book_chapters` | body chapter titles for a localized contents block; never body prose |
| `validate_matter_plan_tool` | dry-run the title/credit/omission/section plan |
| `report_matter_plan` | **the book-level answer**: localized title, BleBel credit, exact omitted phrases, section titles |
| `validate_matter_localization` | dry-run one chunk's localized Markdown against the binding plan |
| `report_matter_localization` | **the chunk-level answer**: the complete localized chunk, or an explicit omission with a reason |
| `web_search`, `fetch_web_page` | reused from the splitter: find a published translation and reuse its exact title |
| `list_chapter_files`, `extract_markdown_headers`, `inspect_line_window` | the shared structural tools, reused rather than duplicated |

The book graph binds the plan/reader/web tools; the chunk graph binds the chunk
readers, the plan readers, the two report/validate pairs and the same web tools.
No write tool is exposed: the graph writes the validated text.

### `preprocess` — stage 2 only

| Tool | Purpose |
| --- | --- |
| `collect_chapter_headers` | **the first step**: every header of one chapter with its evidence and a deterministic recommendation, plus plain page-number/caption/ALL-CAPS-heading body candidates |
| `collect_book_headers` | the same inventory across the whole book, in one call |
| `extract_headers_from_file` | a bare header list for any single file |
| `inspect_chapter_damage` | body-text damage by shape (line number + fragments + reconstructed word) |
| `scan_book_damage` | the book-wide damage profile plus a text digest |
| `validate_chapter_cleaning` | dry-run a chapter's plan against the real file |
| `report_chapter_cleaning` | **the tool the agent submits its plan through** |
| `audit_chapter_opening` | the opening check: ≤12 clipped opening lines plus every provable opening issue (stray number under the title, duplicated title, leftover page number, bare-number title) |
| `apply_chapter_cleaning` | writes the cleaned chapter (blocked until its plan is validated) |
| `verify_preprocessed_chapters` | re-checks a whole `02-preprocessed/` folder against `01-chapters/` |

### `postprocess` — stage 9 only

| Tool | Purpose |
| --- | --- |
| `list_chapter_titles` | **the first step**: every chapter title, its line/shape, numbering, front/back matter and TOC state — the whole list before any naming decision |
| `inspect_matter_blocks` | bounded front/back matter (≤40 clipped lines per block) plus the detected TOC |
| `web_search` | **mandatory** published-contents lookup: original title + author → chapter list/table of contents; a failed call still counts if its outcome is recorded in `title_evidence` |
| `fetch_web_page` | reads a publisher/library/bookstore page returned by `web_search` as text, so the contents list can be read |
| `inspect_chapter_opening` | at most 8 clipped opening lines; the chapter-title candidate and subtitle |
| `audit_book_openings` | every chapter's opening in one bounded call: the book-level counts and the chapters that need a look (a title that is only a number, a stray number under the title, a duplicated title line) |
| `audit_chapter_opening` | one chapter's opening audit (≤12 clipped lines), shared with stage 2 |
| `inspect_chapter_headings` | every heading with its line, suggested `keep`/`demote`/`promote`, risk and one-line evidence; plain ALL-CAPS heading candidates are included with `level: 0` |
| `inspect_heading_context` | one heading plus at most two clipped non-blank lines on each side |
| `validate_book_plan_tool` | dry-run the naming policy and TOC action against the book |
| `report_book_plan` | **the book-level answer**: naming policy + TOC action + `title_evidence` from the mandatory web search |
| `validate_chapter_postprocess` | dry-run one chapter's title/heading plan |
| `report_chapter_postprocess` | **the chapter-level answer**: title action + `keep`/`demote`/`promote` decisions |
| `apply_chapter_postprocess` | writes one postprocessed chapter (blocked until validated) |
| `apply_book_toc` | creates or updates the TOC from the final titles (blocked until validated) |
| `verify_postprocessed_chapters` | checks the output folder and its TOC |
| `list_chapter_files`, `extract_markdown_headers`, `inspect_line_window` | reuses the splitter's shared tools inside the stage's toolset |

The book graph binds the book-plan tools plus the shared inventory/header tools;
the chapter graph binds the chapter tools plus `extract_markdown_headers` and
`inspect_line_window`. The book planner cannot apply a chapter, and the chapter
editor cannot change the book's naming policy.

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
| `--unique-out DIR` | where the first-mention chapters go (default: a sibling of `03-reduced/`) |
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

### Stages 5, 6, 7, 8, 10 and 11 — no tools either

`chunk` binds no tools because it calls no model at all: its input is the chapter
text and its vocabulary is `--chunk-size`, `--chunk-tokenizer`, `--limit`,
`--dry-run` and `--skip-if-present`. `chunk-glossary` also binds no tools: it runs
a fixed two-phase pipeline (NER over every chunk, then rapidfuzz retrieval) whose
whole vocabulary is command-line flags — `--chunks-in`, `--glossary-in`,
`--chunk-glossary-out`, `--match-threshold`, `--scorer`, `--candidates`,
`--min-term-chars`, `--drop-temporal-only`, `--language`, `--gpu`/`--cpu`,
`--limit`, `--no-manifest` and `--dry-run`. `translate` calls a model, but it too
binds no tools: the model receives a directory-free task (one prompt, one JSON
answer) and its vocabulary is `--chunks-in`, `--chunk-glossary-in`,
`--glossary-scope`, `--translations-out`, `--prompts-out`, `--source-language`,
`--target-language`, `--limit`, `--resume`, `--method`, `--max-attempts`,
`--max-tokens`, `--context-chars`, `--translator` (`deepseek` or `lmstudio`,
its own provider choice), `--no-prompts`, `--no-manifest` and `--dry-run`.
`bind` binds no tools either: it is a deterministic file concatenation with
`--postprocessed-in`, `--book-out`, `--no-manifest` and `--dry-run`. `epub` is
the same kind of stage: a deterministic package build whose whole vocabulary is
`--postprocessed-in`, `--epub-out`, the metadata overrides (`--title`,
`--author`, `--language`, `--identifier`, `--description`, `--publisher`),
`--cover`, `--no-manifest` and `--dry-run`.

## Guardrails

- **No body text in the prompt.** Tools return line numbers, headers, candidate titles and
  ≤90-character excerpts. Stage 2 adds evidence flags, damage fragments and `inspect_line_window`
  (≤6 lines, ≤600 characters per call); every tool result is capped at 24 000 characters and none
  of the stage-2 tools can return a chapter's prose. Stage 9 is stricter still: its dedicated
  chapter-text tool, `inspect_chapter_opening`, returns **at most 8 clipped lines**, its
  `inspect_heading_context` returns at most two clipped lines on each side, and everything else is
  a heading record with one line of evidence; it also reuses the shared `inspect_line_window`
  (≤6 lines / ≤600 characters). `inspect_matter_blocks` caps the front and back matter at 40
  clipped lines per block. Stage 1 has exactly two bounded prose
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
  agent (bounded by `--max-attempts`). The budget is printed in the scout report, and on the
  last allowed turn the agent is told that a submission is the only useful call left — a run
  that spends its final turn inspecting writes nothing, which is how *We Need to Talk About
  Kevin* failed before its [recovered map](#replaying-a-map-without-the-model) was written.
- **No writes before validation.** Stage 1's `slice` is a deterministic tail node reachable
  only from a validated plan; there is no model-reachable write tool in that graph at all.
  Stage 2 does the same per chapter: `apply_chapter_cleaning` is blocked until *that*
  chapter's plan has been validated.
- **No content loss.** The verifier recomputes SHA-256 over the files on disk and compares it
  with the source slice the manifest claims to cover. Stage 2 goes further: every source line is
  accounted for in a provenance map, and a line may only disappear when it was a header
  consisting of nothing but a printed page number (bare digits or a dash-framed number)
  *and* the plan declared it.
- **No arbitrary rewriting.** Stage 2 has no free-text edit. A plan entry can only choose one of
  seven decisions (`keep` / `promote` / `demote` / `remove` / `remove_caption` /
  `merge_with_next` / `merge_demote`) plus four named text fixes, each with a
  machine-checkable rule; `apply` re-derives the output with that rule and the verifier
  re-runs it.
- **No path escapes.** Filenames are normalised to `NN-slug.md`; directory components are
  stripped and duplicates are disambiguated instead of overwritten. A filename whose slug is a
  near-miss of its own short heading (`15-belestke.md` for `## BELEŠKE`) is repaired from the
  heading, while descriptive names such as `05-glava-2-seks-i-srce.md` are left untouched.
  Stage 2 keeps each chapter's filename, so `01-chapters/` and `02-preprocessed/` line up one to one,
  and stage 3 does the same with `03-reduced/` and `03-reduced-unique/`.
- **The opening audit deletes only provable junk.** The final phase of stages 2 and 9 can remove
  a line only when it is *nothing but* a stray number next to the title, a copy of the title, or a
  printed page number in the same run — and it re-audits the written file to prove that no fixable
  issue survives. A title that is only a number, a title that is only a label and a chapter that
  opens on prose are reported, never deleted: a title is a naming decision for stage 9's book plan.
- **Stage 3 drops, never edits.** A reduced chapter contains only whole sentences copied
  verbatim from the chapter, one per line. No word is changed, no NER markup is written, and the
  manifest records exactly how many sentences each filter removed and why (with clipped examples
  in the trace).

## Install

Python 3.11+ is required. The package metadata lives in
[pyproject.toml](pyproject.toml); the optional stages are installed as extras:

```bash
python -m venv .venv && . .venv/bin/activate

pip install -e ".[dev]"                 # development checkout: tests, lint, types
pip install "book-translator[convert]"    # EPUB / text-based PDF input
pip install "book-translator[dictionary]" # the NER tagger for stages 3 and 6
pip install "book-translator[chunk]"      # chonkie + tiktoken + rapidfuzz
pip install "book-translator[epub]"       # markdown-it-py for stage 11
```

| Extra | Unlocks |
|---|---|
| `convert` | Stage 0: EPUB / text-based PDF → Markdown through docling |
| `dictionary` | Stage 3 and the NER half of stage 6: CLASSLA, stanza, lingua |
| `chunk` | Stage 5 chunking and stage 6 fuzzy retrieval: chonkie, tiktoken, rapidfuzz |
| `epub` | Stage 11: markdown-it-py |
| `dev` | The offline test suite, ruff, mypy and the build tooling |

Every optional import is lazy, so a split-only install keeps working without any
of them. `dev` includes the chunk and EPUB dependencies so
`pip install -e ".[dev]"` runs the whole suite; `convert` and `dictionary` stay
out because docling/torch and the taggers are heavy and their tests use stubs.

The installed console scripts and the module form are equivalent:

```bash
book-translator --help
book-translator pipeline BOOK.md
book-translator-pipeline BOOK.md        # same driver, no `pipeline` subcommand
python -m book_translator --help
```

The historical `requirements-*.txt` files still work; they are thin shims that
install the matching extra (`requirements-book-translator.txt` → `.[chunk]`,
`requirements-dev.txt` → `.[dev]`, `requirements-dictionary.txt` →
`.[dictionary]`).

Then copy [.env.example](.env.example) to `.env` and fill in the key(s); which
model each stage uses is set in [book_translator.toml](book_translator.toml) (see
[Configuration](#configuration)).

Stage 0 (`convert`) needs docling only when the source is an EPUB or a PDF; a
TXT or Markdown source needs no dependency at all. docling pulls its own layout
models (and torch), so it is deliberately optional and imported lazily:

```bash
pip install -e ".[convert]"           # or: pip install "book-translator[convert]"
```

The PDF text-layer preflight uses `pypdfium2`, which docling already installs;
without docling it can also fall back to PyMuPDF. An image-based PDF is rejected
with an actionable message — OCR is never attempted.

Stages 5 and 6 need the chunk extra: `chonkie` cuts the chapters, `tiktoken`
makes the default `cl100k_base` token count exact, and `rapidfuzz` retrieves the
glossary terms:

```bash
pip install -e ".[chunk]"
```

Stage 6's NER half and stage 3 need the dictionary extra:

```bash
pip install -e ".[dictionary]"
```

Both are imported lazily. Without `tiktoken` the chunk length is still enforced,
but measured with the documented local approximation (and the manifest says so);
without `chonkie` the whole-pipeline script reports the phase as skipped and
`book-translator chunk` exits with the exact `pip install` line. The tagger is only
needed for stage 6 phase 1, and `--language` / `--gpu` / `--cpu` behave exactly
as they do for `dictionary`.

Stage 11 (`epub`) renders the Markdown to XHTML with `markdown-it-py` and writes
the EPUB 3 container with the standard library — no pandoc, calibre or ebooklib:

```bash
pip install -e ".[epub]"
```

A run that stops at `10-translated-book.md` needs none of it.

Optional, only for `extract_native_toc` with PDF/EPUB sources:
`pip install PyMuPDF ebooklib beautifulsoup4`.

⚠️ **`classla` pins `torch<=2.6` on Python < 3.14**, so a plain
`pip install -e ".[dictionary]"` next to a newer torch will downgrade it and pull
~2 GB of CUDA wheels. If torch is already installed to your liking, keep it:

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

### Building a wheel

```bash
python -m build
```

The wheel installs the `book_translator` package (including
`book_translator/stages/epub/assets/style.css`), both console scripts and all five extras.
The sdist includes the README, the `book_translator.toml` example, the tests and the
scripts; `MANIFEST.in` is the list. CI builds both and smoke-tests the installed
wheel, so packaging regressions fail before a release.

## Use

```bash
# 0. Stage 0 — convert an EPUB, a text-based PDF or a TXT to Markdown
#    (writes to data/output/<book>/00-markdown/; a .md source skips this stage)
python -m book_translator convert BOOK.epub
python -m book_translator convert BOOK.pdf              # rejected if it has no text layer
python -m book_translator convert BOOK.txt              # just copies the bytes under a .md name
python -m book_translator convert BOOK.pdf --dry-run    # text-layer check only, no writes
python -m book_translator convert BOOK.pdf --force      # convert again despite a matching hash
python -m book_translator convert BOOK.epub --no-images # text only; write no images/ folder

# 1. What does the book actually contain? (no LLM, no API key, no writes;
#    stage 0 may write 00-markdown/ first for a non-Markdown source)
#    With a single book in data/input/ the argument can be omitted entirely.
#    inspect and split run stage 0 automatically for a non-Markdown source.
python -m book_translator inspect --candidates

# 2. Stage 1 — split it (writes to data/output/<book>/01-chapters/)
python -m book_translator split BOOK.md

# Plans the chapters and validates them, but writes nothing
python -m book_translator split BOOK.md --dry-run

# Replay a chapter map from a file: no LLM, no API key, no network
python -m book_translator split BOOK.md --map my-map.json

# Reuse a chapters folder that is already there: no model, no key, no rewrite
python -m book_translator split BOOK.md --skip-if-present

# 3. What does stage 2 have to fix? (no LLM, no writes)
python -m book_translator scan                  # the book-wide damage profile
python -m book_translator scan --digest         # per chapter: headers + evidence + recommendation

# 4. Stage 2 — clean every chapter, one at a time
#    (writes to data/output/<book>/02-preprocessed/)
python -m book_translator preprocess
python -m book_translator preprocess --dry-run          # validate every plan, write nothing
python -m book_translator preprocess --workers 4        # same work, thread pool

# 5. Stage 3 — reduce each chapter to its entity-bearing sentences, then to
#    first mentions (writes to data/output/<book>/03-reduced/ and 03-reduced-unique/)
python -m book_translator dictionary                    # auto-detect the language
python -m book_translator dictionary --language sr --gpu 1
python -m book_translator dictionary --limit 3          # a quick look while tuning
python -m book_translator dictionary --skip-if-present  # reuse what is already reduced
python -m book_translator dictionary --no-save-reduced  # count only, write no reduced chapters
python -m book_translator dictionary --no-save-unique   # write 03-reduced/ only
python -m book_translator dictionary --unique-only      # write 03-reduced-unique/ only
python -m book_translator dictionary --out data/output/BOOK/02-preprocessed/   # reduce the cleaned text

# 6. Stage 4 — build the glossary, one chapter at a time, first to last
#    (writes to data/output/<book>/04-glossary/; calls the model, so it costs tokens)
python -m book_translator glossary                      # English -> Serbian, the default pair
python -m book_translator glossary -s sr -t en          # a Serbian book into English
python -m book_translator glossary --limit 2 --dry-run  # price the first two chapters, write nothing
python -m book_translator glossary --max-attempts 5     # be more patient with a flaky model
python -m book_translator glossary --resume             # continue a run that stopped, reusing its tokens
python -m book_translator glossary --include-front-matter   # the blocks around the body are skipped by default
python -m book_translator glossary --max-context-entries 150  # cap the re-sent dictionary on a long book

# 7. Stage 5 — cut each chapter into 1000-token chunks that concatenate back
#    (writes to data/output/<book>/05-chunks/; no model, no key)
python -m book_translator chunk                         # default 1000 tokens
python -m book_translator chunk --chunk-size 700        # smaller translation units
python -m book_translator chunk --dry-run               # verify every cut, write nothing
python -m book_translator chunk --skip-if-present       # reuse existing chunks
python -m book_translator chunk --limit 3               # a quick look at three chapters
python -m book_translator chunk --out data/output/BOOK/02-preprocessed/   # chunk the cleaned text

# 8. Stage 6 — tag every chunk first, then attach the glossary terms it mentions
#    (writes to data/output/<book>/06-chunk-glossary/; no LLM, no API key;
#     the NER tagger is the same one stage 3 uses)
python -m book_translator chunk-glossary                # default 95% rapidfuzz match
python -m book_translator chunk-glossary --match-threshold 90   # more permissive
python -m book_translator chunk-glossary --scorer token_sort_ratio
python -m book_translator chunk-glossary --limit 3      # a quick look at three chunks
python -m book_translator chunk-glossary --first-chapters 3   # the first three chapters' chunks
python -m book_translator chunk-glossary --resume       # reuse finished per-chunk files
python -m book_translator chunk-glossary --dry-run      # both phases, no writes

# 9. Stage 6b — localize the front/back matter before translation
#    (writes the localized chunks to data/output/<book>/07-translations/, the plan to
#     06b-matter/ and every agent input to 06b-matter-prompts/; no dictionary is used)
python -m book_translator matter                        # web-search the published title
python -m book_translator matter -s sr -t en            # another direction
python -m book_translator matter --limit 3              # first three matter chunks
python -m book_translator matter --resume               # reuse finished matter chunks

# 10. Stage 7 — translate the body chunks, one call at a time, with their glossary
#     (writes to data/output/<book>/07-translations/, and the whole prompt of every
#      call to data/output/<book>/07-translation-prompts/; calls the model, so it costs)
python -m book_translator translate                     # English -> Serbian, every body chunk
python -m book_translator translate -s sr -t en         # a Serbian book into English
python -m book_translator translate --limit 3           # translate three chunks first
python -m book_translator translate --first-chapters 3  # translate the first three chapters
python -m book_translator translate --resume            # skip finished chunks, continue
python -m book_translator translate --glossary-scope book   # send the whole glossary
python -m book_translator translate --context-chars 400 # carry the previous translation's tail
python -m book_translator translate --no-prompts        # skip the debug prompt records
python -m book_translator translate --include-matter    # old behaviour: body prompt on matter too
python -m book_translator translate --translator lmstudio   # local model (auto-detected, no key needed)

# 11. Stage 8 — glue the translated chunks back into chapters
#     (writes to data/output/<book>/08-translated-chapters/; no LLM, no API key)
python -m book_translator concatenate                   # every fully translated chapter
python -m book_translator concatenate --first-chapters 3
python -m book_translator concatenate --first-chapters 10 --resume
python -m book_translator concatenate --dry-run         # assemble in memory, write nothing

# 12. Stage 9 — make the translated headings and TOC consistent
#     (writes to data/output/<book>/09-postprocessed-chapters/; calls the model)
python -m book_translator postprocess                   # whole translated book
python -m book_translator postprocess --first-chapters 3
python -m book_translator postprocess --resume          # reuse unchanged chapters
python -m book_translator postprocess --dry-run         # validate every plan, write nothing

# 12b. The chapter-opening audit on its own (the final phase of stages 2 and 9;
#      no LLM, no API key; the default folder is the book's most advanced stage)
python -m book_translator openings                      # audit and repair every opening
python -m book_translator openings --dry-run            # report every finding, write nothing
python -m book_translator openings --chapters-in data/output/BOOK/02-preprocessed --keep-blank-slot

# 13. Stage 10 — concatenate the postprocessed chapters into one book
#     (writes data/output/<book>/10-translated-book.md; no LLM, no API key)
python -m book_translator bind                          # every postprocessed file, in order
python -m book_translator bind --dry-run                # assemble in memory, write nothing

# 14. Stage 11 — package the same chapters as an EPUB 3 book
#     (writes data/output/<book>/11-translated-book.epub; no LLM, no API key)
python -m book_translator epub                          # images embedded, reader navigation
python -m book_translator epub --cover cover.jpg         # with a cover page
python -m book_translator epub --dry-run                 # build and verify, write nothing

# Optional: list the toolset per stage / verify a folder layout
python -m book_translator tools --stage postprocess

# Explicit folders / fully local model
python -m book_translator split BOOK.md --out /tmp/01-chapters/
python -m book_translator preprocess --out /tmp/01-chapters/ --preprocessed-out /tmp/clean/
python -m book_translator dictionary --out /tmp/01-chapters/ --unique-out /tmp/mentions/
python -m book_translator glossary --out /tmp/01-chapters/ --glossary-out /tmp/04-glossary/
python -m book_translator chunk --out /tmp/01-chapters/ --chunks-out /tmp/05-chunks/
python -m book_translator chunk-glossary --chunks-in /tmp/05-chunks/ \
    --glossary-in /tmp/04-glossary/ --chunk-glossary-out /tmp/06-chunk-glossary/
python -m book_translator translate --chunks-in /tmp/05-chunks/ \
    --chunk-glossary-in /tmp/06-chunk-glossary/ \
    --translations-out /tmp/07-translations/ --prompts-out /tmp/07-translation-prompts/
python -m book_translator concatenate --chunks-in /tmp/05-chunks/ \
    --translations-in /tmp/07-translations/ --chapters-out /tmp/08-translated-chapters/
python -m book_translator postprocess --translated-chapters-in /tmp/08-translated-chapters/ \
    --postprocessed-out /tmp/09-postprocessed-chapters/
python -m book_translator bind --postprocessed-in /tmp/09-postprocessed-chapters/ \
    --book-out /tmp/10-translated-book.md
python -m book_translator preprocess \
    --provider lmstudio --model qwen/qwen3.8-27b --base-url http://127.0.0.1:1234/v1
```

Useful flags: `--in-dir`, `--out`, `--map`, `--preprocessed-out`, `--reduced-out`, `--unique-out`,
`--chunks-out`, `--chunk-size`, `--chunk-tokenizer`,
`--chunks-in`, `--glossary-in`, `--chunk-glossary-out`, `--match-threshold`, `--scorer`,
`--candidates`, `--min-term-chars`, `--drop-temporal-only`,
`--chunk-glossary-in`, `--glossary-scope`, `--translations-out`, `--prompts-out`,
`--translations-in`, `--chapters-out`, `--first-chapters`, `--translated-chapters-out`,
`--translated-chapters-in`, `--postprocessed-out`, `--postprocessed-in`, `--book-out`,
`--with-postprocess`, `--no-postprocess`, `--no-bind`, `--no-epub`,
`--epub-out`, `--epub-cover`, `--epub-title`, `--epub-author`, `--epub-language`,
`--context-chars`, `--method`, `--no-prompts`, `--translator`,
`--translate-provider`, `--translate-model`, `--translate-base-url`,
`--translate-api-key`, `--translate-temperature`,
`--glossary-out`, `--sentences`, `--sentences-in`, `--source-language`, `--target-language`,
`--include-front-matter`, `--include-back-matter`, `--extraction-method`, `--max-tokens`,
`--max-chapter-chars`, `--max-context-entries`, `--resume`,
`--max-attempts`, `--workers`, `--front-matter-lines`, `--no-front-matter`, `--skip-if-present`,
`--language`, `--gpu`, `--cpu`, `--min-words`, `--keep-types`, `--no-save-reduced`,
`--no-save-unique`, `--unique-only`, `--limit`,
`--dry-run`, `--json`, `--quiet`, `--temperature`, `--api-key`.
`python -m book_translator tools` lists the toolset **grouped by
stage**, and [scripts/check_layout.py](scripts/check_layout.py) verifies the folder layout and the
installed dependencies.

### Default input and dry runs

Two rules apply to both the stage commands and `run_pipeline.py`:

* **The text stages read the cleaned text by default.** `dictionary`, `glossary`, `chunk`,
  `chunk-glossary`, `translate`, `concatenate`, `postprocess`, `bind` and `epub` use
  `02-preprocessed/` when stage 2 wrote a completed manifest for it, and fall back to
  `01-chapters/` otherwise. An explicit `--out` (or a book argument that is already a folder) is
  used as given, so `--out .../01-chapters` asks for the raw text on purpose. `scan` and
  `preprocess` always read `01-chapters/`, because they run before (or produce) the cleaned copy.
  `matter` also keeps `01-chapters/`, because it creates synthetic front/back-matter files in that
  folder and appends them to stage 1's manifest; its chunk text still comes from `05-chunks/`.
* **`--dry-run` validates but never spends.** Deterministic stages still do their real work
  (the chunker verifies every cut, the binder assembles in memory, the EPUB packager builds and
  verifies the package) and write nothing. Model-backed stages stop at their deterministic
  preview: they resolve the inputs, discover the chapters/chunks/matter sections, check the
  resumable prefix and report what would run, but they do **not** construct a chat model or call
  a provider. No chapter, chunk, glossary, translation, manifest or trace is written. A dry run
  is a validation mode, not a way to see a model's plan; use a real run with `--limit` /
  `--first-chapters` when the goal is a cheaper sample.

### One command for the whole pipeline

The pipeline runs through one parser and one driver: `python -m book_translator pipeline` and
[scripts/run_pipeline.py](scripts/run_pipeline.py) (a thin compatibility wrapper) are the same
command. It runs the phases in the only order that works —
convert → recon → split → scan → preprocess → dictionary → chunk → chunk-glossary → matter
(→ translate and concatenate, opt-in → postprocess → bind → epub) — and narrates each one: book size and header verdict, the strategy the agent
chose, the byte-for-byte proof, the damage profile, how many sentences survived the dictionary funnel
and how many of those are first mentions, how many entities were locked into the glossary — with a
live progress bar while that runs — what the LLM pass cost in tokens, how many reversible chunks the
chapter text was cut into, how many glossary terms each chunk mentions, which front/back matter
sections were localized and which title the agent reused, and, when translation is switched on, how
many body chunks were translated, what they cost in tokens, and how many chapters stage 8 assembled
from them.
`--debug` adds headers, planned files and damage examples; the same transcript always
lands in `data/output/<book>/pipeline.log`, debug included.

```bash
python -m book_translator pipeline                     # every default phase, the only book in data/input/
python -m book_translator pipeline --json              # one machine-readable summary, no narration
python -m book_translator pipeline --quiet             # no narration; errors still go to stderr
python scripts/run_pipeline.py                      # the same driver through the old script path
python scripts/run_pipeline.py BOOK.epub            # stage 0 converts it, then the usual run
python scripts/run_pipeline.py BOOK.pdf --stop-after convert   # just the text-layer check + Markdown
python scripts/run_pipeline.py BOOK.pdf --force-convert       # convert again despite a matching hash
python scripts/run_pipeline.py BOOK.md --debug      # + the per-phase detail
python scripts/run_pipeline.py --dry-run            # validate every plan, write no artifact
                                                    # (stage 0 converts into a temporary folder)
python scripts/run_pipeline.py --no-preprocess      # skip stage 2 (cleaning) and read raw text
python scripts/run_pipeline.py --stop-after split   # stop once the chapters exist
python scripts/run_pipeline.py --start-at dictionary  # stage 3 only, over existing chapters
python scripts/run_pipeline.py --text-in 01-chapters   # put stages 3-5 back on the raw text
python scripts/run_pipeline.py --start-at matter      # stage 6b only, over existing chunks
python scripts/run_pipeline.py --with-matter          # localize matter without translating the body
python scripts/run_pipeline.py --skip-if-present    # reuse chapters, reduced chapters, glossary
python scripts/run_pipeline.py --no-dictionary      # split + scan + chunk only
python scripts/run_pipeline.py --unique-only        # stage 3 first mentions only
python scripts/run_pipeline.py --start-at chunk     # stage 5 only, over existing chapters
python scripts/run_pipeline.py --start-at chunk_glossary   # stage 6 only, over existing chunks
python scripts/run_pipeline.py --chunk-size 700     # smaller translation chunks
python scripts/run_pipeline.py --match-threshold 90 # more permissive chunk glossary retrieval
python scripts/run_pipeline.py --no-chunk-glossary  # stop after the chunks
python scripts/run_pipeline.py --no-chunk           # stop before the chunking phase
python scripts/run_pipeline.py --with-translate --translate-limit 3
                                                    # translate three chunks first
python scripts/run_pipeline.py --with-translate --skip-if-present
                                                    # resume a translation run
python scripts/run_pipeline.py --start-at translate --with-translate
                                                    # stage 7 only, over existing chunks
python scripts/run_pipeline.py --start-at concatenate
                                                    # stage 8 only, over existing translations
python scripts/run_pipeline.py --first-chapters 5 --with-translate
                                                    # split in full; the first five chapters elsewhere
python scripts/run_pipeline.py --first-chapters 10 --skip-if-present --with-translate
                                                    # widen 5 -> 10: reuse 1-5, process 6-10 only
python scripts/run_pipeline.py --with-translate --translate-first-chapters 3
                                                    # translate + assemble only three chapters
python scripts/run_pipeline.py BOOK.md --map map.json --stop-after split   # no LLM at all

# the dictionary phase builds the glossary too; --no-glossary keeps it deterministic
python scripts/run_pipeline.py --start-at dictionary --skip-if-present   # resume the glossary
python scripts/run_pipeline.py --no-glossary                             # reduce only, no key needed
python scripts/run_pipeline.py --limit 2 --dry-run                      # validate, write nothing
```

Stage 3 stays selected but **reports itself as skipped** when its optional tagger is not
installed, so a split-only workflow still finishes; `--with-dictionary` / `--no-dictionary`
override that for one run. Stage 5 does the same for a missing `chonkie`, and `--no-chunk`
drops it from the run. Stages 3 and 5 read the **cleaned** `02-preprocessed/` folder by
default, so the page numbers and split headings stage 2 repaired never reach the dictionary,
the glossary or the chunks; `--text-in 01-chapters/` forces the raw text for every stage, and
`--chunk-in` overrides the folder for the chunker alone. Resume flags do not cross that
source change: a reduction or glossary built from `01-chapters/` is rebuilt when the run
now reads `02-preprocessed/`, because the filenames match but the text does not.

Stage 6 is the last free phase and the bridge to translation: it tags every chunk first, then
fuzzy-matches the extracted terms against the glossary. It needs both stage 5's chunks and stage
4's `glossary.json`; when either is missing it reports a skip with the reason (a missing tagger or
`rapidfuzz` is reported the same way), so the earlier phases keep their artifacts. `--no-chunk-glossary`
drops it, `--start-at chunk_glossary` runs only it, `--match-threshold` and `--match-scorer` tune the
retrieval, and `--no-manifest` keeps only the per-chunk files. The closing report lists the stage-6
manifest and the `06-chunk-glossary/` folder.

Stage 6b (`matter`) runs automatically with stage 7, and can be run on its own with
`--with-matter` or `--start-at matter`. It localizes the front/back matter chunks with its own
book-level plan and chunk graphs, **without** the dictionary, and writes them to the same
`07-translations/` folder so stage 8 needs no special case. The plan searches the web for a published
translated title, fixes the BleBel credit, lists original-edition material to omit and gives each
section its localized title. `--matter-limit N` prices the first N matter chunks, `--skip-if-present`
resumes, `--matter-out` / `--matter-prompts-out` move the plan/debug folders, and `--no-matter` drops
the phase. The closing report lists `06b-matter/`, `06b-matter-prompts/` and the stage-6b manifest and trace.
Stage 7 then translates only the body chunks; pass `--include-matter` to put the matter chunks back in
front of the body prompt (they will overwrite the localized chunks).

Stage 7 (`translate`) is **opt-in**, because it is one paid model call per chunk: add
`--with-translate`. It reads stage 5's chunks and stage 6's per-chunk glossary, sends each body chunk
with its binding terms, and writes `07-translations/` plus the debug folder `07-translation-prompts/`. It
needs the per-chunk glossary; when stage 6 did not run (or produced no per-chunk files) the phase
reports a skip instead of failing the run. `--translate-limit N` translates a prefix — the cheap way
to test the whole path — and `--skip-if-present` resumes: chunks already translated and unchanged are
reused, and the manifest is rewritten after every chunk, so an interrupted run loses only the chunk in
flight. `--translate-glossary-scope book` sends the whole glossary instead of the per-chunk terms;
`--translate-attempts`, `--translate-method`, `--translate-max-tokens` and
`--translate-context-chars` tune the calls; `--no-translate` drops the phase for an explicit command
line. `--translate-provider deepseek|lmstudio` chooses stage 7's model without
moving any other phase: `lmstudio` talks to `http://127.0.0.1:1234/v1`,
auto-detects the loaded model (or takes it from `--translate-model`), and needs no
key; front/back matter and every other stage keep their own `[llm.<stage>]`
section from [book_translator.toml](book_translator.toml). A `--dry-run` reports
stage 7 as skipped instead of spending, exactly
like the glossary half of `dictionary`. The closing report lists
`07-translations/`, `07-translation-prompts/` and the stage-7 manifest and trace.

Stage 8 (`concatenate`) runs automatically with stage 7 and can be run on its own with
`--start-at concatenate`. It groups the translated chunks by stage 5's recorded chapter, joins each
group in chunk order, and writes `08-translated-chapters/` without touching the source `01-chapters/`. A
chapter with an untranslated chunk is reported as incomplete and skipped, never written truncated.
`--no-concat` drops it from a run, and `--translated-chapters-out` moves the folder. The closing
report lists `08-translated-chapters/` and the stage-8 manifest and trace.

`--first-chapters N` is the whole-pipeline window: splitting still runs over the **whole** book (its
boundaries are the coordinate system every later stage reads), while dictionary, the glossary,
preprocessing, chunking, the chunk glossary, translation and assembly all see only the first N
**body** chapters. The matter stage is the exception: it always localizes the whole front/back
matter set, because those files are not chapters and its job is the edition's title/credit/imprint
page, not a chapter prefix. `00-front-matter.md` (and a marked back-matter block) does not consume one of the
N: the stages that take a file-count limit still walk past it, but `--first-chapters 5` means
chapters 1–5. For the chunk-based stages the window is mapped through stage 5's manifest, so it never
stops mid-chapter. `--translate-first-chapters N` narrows translation and assembly alone. Because
every stage's manifest is written as it goes and every stage honours `--skip-if-present`, widening
the window (`5`, then `10`) reuses the finished chapters, reductions, glossary entries, chunks,
per-chunk glossaries, translations and assembled chapters, and pays only for the new prefix. Without
`--skip-if-present` the run starts that stage fresh, as the flag's name implies. `--limit` is still
there as the raw unit limit for a quick look; when both are given, the smaller one wins.

The glossary is not a phase of its own: it is the second half of `dictionary`, because it reads
exactly what the reduction just wrote. It runs under `--dry-run` like every other selected stage:
the real plan is validated (which means asking the model), and no glossary, manifest or trace is
written. `--no-glossary` drops it so the phase stays deterministic. `--skip-if-present` resumes it:
a run that stopped after chapter 30 pays for chapters 30..N and keeps the tokens already spent. The
closing report ends with the dictionary's path and the token bill.
`python -m book_translator glossary` remains the shorter way to run stage 4 on its own.

It exits `0` only when every phase it ran succeeded, `1` when a phase failed (the phases before it
stay on disk), and `2` on a setup error such as a missing book or an unresolvable provider.
`--json` appends the whole run — per phase: status, seconds, summary and the phase's own result
object — as one JSON document.

### Configuration

**One file holds every model.** [book_translator.toml](book_translator.toml) has an
`[llm]` table with the shared defaults and one optional table per stage that calls
a model:

| table | stage |
|---|---|
| `[llm.split]` | stage 1 — the routing agent |
| `[llm.preprocess]` | stage 2 — the per-chapter cleaning agent |
| `[llm.dictionary]` | stages 3–4 — the glossary extractor |
| `[llm.matter]` | stage 6b — the front/back matter localizer |
| `[llm.translate]` | stage 7 — the chunk translator |
| `[llm.postprocess]` | stage 9 — the headings/TOC agent |

`[llm.glossary]` is accepted as an alias for `[llm.dictionary]` (the standalone
`glossary` command's stage).

Each field is optional: `provider`, `model`, `base_url`, `temperature`, `timeout`,
`max_retries`, `api_key_env`. `model` / `base_url` never cross a provider boundary:
`[llm]` values, the shared flags and a stage's own model are used only for the
provider they were written for, so switching translation to a local server never
inherits a cloud model id. Temperature and timeouts are generic and always apply:

```toml
[llm]
provider = "deepseek"
model = "deepseek-flash"

[llm.split]
model = "deepseek-flash"   # the default; use deepseek-v4-pro for a stronger pass

[llm.translate]
provider = "lmstudio"
base_url = "http://127.0.0.1:1234/v1"
# model omitted: the loaded id is discovered from GET /v1/models
```

`deepseek-flash` is the default in `[llm]` and is set explicitly in every stage
table, so every stage uses it unless you change that table. DeepSeek's current ids
are `deepseek-flash` (V4.1 Flash, the default) and `deepseek-v4-pro` (stronger,
thinking); the older `deepseek-chat` and `deepseek-reasoner` aliases are retired.

Override one field for one run without editing the file — the pipeline and every
module command that calls a model accept the same switches:

```bash
python scripts/run_pipeline.py mybook.md \
    --llm split.model=deepseek-flash \
    --llm translate.provider=lmstudio
python -m book_translator split mybook.md --llm model=deepseek-flash
BOOK_TRANSLATOR_CONFIG=/tmp/cheap.toml python scripts/run_pipeline.py mybook.md
```

The pipeline also takes `--llm-config FILE`. A module command may omit the stage
prefix (`--llm model=...` applies to that command's stage); the pipeline requires
it (`--llm translate.model=...`), because one run touches several stages.

Precedence for one stage and one field, highest first: `--llm` → the stage's own
flag (`--translate-model`, or `--model` on a single command) → stage-specific
environment (`SPLITTER_<STAGE>_<FIELD>`, `TRANSLATION_*`, `LMSTUDIO_*`) → the
shared pipeline flags (`--provider`/`--model`/`--base-url`/`--api-key`) →
`[llm.<stage>]` → `[llm]` → the legacy `SPLITTER_*` variables → the provider
default. The provider itself follows the same idea: `--llm <stage>.provider` → the
stage flag → `SPLITTER_<STAGE>_PROVIDER` → `TRANSLATION_PROVIDER` → `--provider` →
`[llm.<stage>]` → `[llm]` → `SPLITTER_PROVIDER` → `deepseek` (or stage 7's
two-backend default). Both orders are executable tables in
[book_translator/core/settings.py](book_translator/core/settings.py) and are exercised
pair by pair in `tests/test_settings_precedence.py`.

**Keys stay in `.env`.** Copy [.env.example](.env.example) to `.env` and set the
key for each provider you configured: `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`,
`OPENROUTER_API_KEY`, `LM_STUDIO_API_KEY`, `OLLAMA_API_KEY`. A stage may name a
different variable with `api_key_env = "MY_KEY"`. Credential lookup order:
the explicit `--api-key` → `--llm <stage>.api_key` / `api_key_env` → the shared
`--api-key` (when its provider matches) → `SPLITTER_<STAGE>_API_KEY` /
`TRANSLATION_API_KEY` / the provider's own variable → `[llm.<stage>].api_key_env`
→ `[llm].api_key_env` → the provider's built-in key variable →
`SPLITTER_API_KEY` → `refs:` in `~/.dsh/.credentials.yaml`. Local endpoints
(`localhost`, `127.0.0.1`) may run keyless.

Stages 3, 5 and 6 (reduction, chunking, NER matching) call no model at all. Stage 4
needs a key only when there is a chapter left to mine, and stage 7 only when there
is a chunk left to translate: a `--resume` run over a finished book exits without
resolving a model (and without a key).

### Output

```
data/output/<book-stem>/00-markdown/         # stage 0 — only for a non-Markdown source
├── <book>.md                             # the converted source stage 1 reads
└── _convert_manifest.json                # source hash, method, PDF text-layer evidence

data/output/<book-stem>/01-chapters/          # stage 1
├── 00-front-matter.md                     # cover/copyright/TOC, everything before chapter 1
├── 01-predgovor.md
├── 02-uvod.md
├── …
├── 15-beleske.md
├── _manifest.json                         # filenames, line ranges, byte counts, SHA-256
└── _agent_trace.json                      # the plan, the validation report, the token usage

data/output/<book-stem>/02-preprocessed/      # stage 2
├── 00-front-matter.md … 15-beleske.md     # same names, cleaned
├── _preprocess_manifest.json              # per chapter: operations, text fixes, provenance
└── _preprocess_trace.json                 # the full per-chapter audit trail

data/output/<book-stem>/03-reduced/           # stage 3
├── 00-front-matter.md … 15-beleske.md     # same names: one raw kept sentence per line
├── _reduced_manifest.json                 # kept/dropped, entity + noun totals, timings,
│                                          #   and the per-chapter noun index a rerun reads
└── _reduced_trace.json                    # + per-chapter seconds + examples of each drop

data/output/<book-stem>/03-reduced-unique/    # stage 3, second step
└── 00-front-matter.md … 15-beleske.md     # same names: first mentions only (a subset)

data/output/<book-stem>/04-glossary/          # stage 4
├── glossary.json                          # the dictionary: entries + languages + token total
├── glossary.md                            # the same, as a table per category, for review
├── _glossary_manifest.json                # per chapter: tokens, attempts, added/repeats/conflicts
└── _glossary_trace.json                   # + every entry a chapter added and every raw answer

data/output/<book-stem>/05-chunks/            # stage 5 — flat, globally ordered
├── 00000-00-front-matter-0000.md          # <global>-<chapter>-<index>.md, an exact slice
├── 00001-01-predgovor-0000.md
├── …
├── _chunk_manifest.json                   # per chunk: chapter, order, char/line spans,
│                                          #   tokens, boundary kind, hashes + the proof
└── _chunk_trace.json                      # the same run, with each chapter's own log

data/output/<book-stem>/06-chunk-glossary/   # stage 6 — one file per chunk
├── 00000-00-front-matter-0000.md.json     # named after the chunk it describes
├── 00001-01-predgovor-0000.md.json
├── …
├── _chunk_glossary_manifest.json          # per chunk: NER counts + matched terms,
│                                          #   the two phases and their timings
└── _chunk_glossary_trace.json             # + every term, closest candidate and log

data/output/<book-stem>/06b-matter/           # stage 6b — plan and audit trail
├── book-plan.json                         # localized title, credit, omissions, sections
├── _matter_manifest.json                  # per matter chunk: chars, omitted, tokens, hash
└── _matter_trace.json                     # + every chunk's log and verification

data/output/<book-stem>/06b-matter-prompts/   # stage 6b — the whole agent input per call
├── _book-plan.json                        # the plan agent's full conversation
├── _book-plan.prompt.md
└── 00000-00-front-matter-0000.md.json     # messages, plan, source, answer per chunk

data/output/<book-stem>/07-translations/     # stages 6b + 7 — the translated chunks
├── 00000-00-front-matter-0000.md         # same names as 05-chunks/; concatenate to rebuild
├── 00001-01-predgovor-0000.md
├── …
├── _translation_manifest.json             # per chunk: chars, sha, tokens, attempts, paths
└── _translation_trace.json                # + terms, logs, raw answers and the run's totals

data/output/<book-stem>/07-translation-prompts/   # stage 7 — the whole input of every call
├── 00000-00-front-matter-0000.md.json     # system+user prompt, messages, glossary,
│                                          #   source chunk text, answer (or error)
└── 00000-00-front-matter-0000.md.prompt.md   # the same prompt, readable

data/output/<book-stem>/08-translated-chapters/   # stage 8 — one file per translated chapter
├── 00-front-matter.md                    # same names as 01-chapters/, never over them
├── 01-predgovor.md
├── …
├── _translated_chapters_manifest.json    # per chapter: chunks, chars, sha, written/reused
└── _translated_chapters_trace.json       # + every chapter's record and error

data/output/<book-stem>/09-postprocessed-chapters/  # stage 9 — headings/TOC fixed
├── 00-front-matter.md
├── 00-table-of-contents.md               # created only when the book had no TOC
├── 01-predgovor.md
├── …
├── _postprocess_manifest.json            # book plan, per-chapter decisions, TOC state
└── _postprocess_trace.json               # + every chapter's log and verification

data/output/<book-stem>/10-translated-book.md  # stage 10 — all postprocessed files, in order

data/output/<book-stem>/11-translated-book.epub   # stage 11 — the same chapters, as EPUB 3
├── mimetype, META-INF/container.xml      # the container: stored, uncompressed, first
├── OEBPS/content.opf                     # EPUB 3 package: dc metadata, manifest, spine
├── OEBPS/nav.xhtml                       # the reader's table of contents (no printed TOC)
├── OEBPS/text/<chapter>.xhtml            # one spine document per postprocessed file
├── OEBPS/css/style.css                   # the typography the book ships with
└── OEBPS/images/<picture>                # every referenced picture, embedded once
  plus _epub_manifest.json                # metadata, spine, picture hashes, verification
  plus _epub_trace.json                   # the whole result, for replay and debugging
```

The two stage-3 folders are two views of one pass: `03-reduced/` holds every sentence the NER gate
kept, `03-reduced-unique/` holds the subset that introduces at least one noun no earlier sentence
used, decided book-wide in chapter order. `03-reduced/_reduced_manifest.json` describes both
(`unique_dir`, `unique_totals`, and per file `unique_lines`, `unique_kept`,
`dropped_duplicate_noun`, `noun_keys`), so `03-reduced-unique/` needs no bookkeeping of its own —
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

`07-translations/` is stage 7's text — the translated chunks, one per source chunk, same filename — and
`_translation_manifest.json` is its index: for every chunk the source and translation character
counts, the source chunk's SHA-256, the translation's SHA-256, attempts, method, tokens, and the paths
of the translation and its prompt record. `--resume` reads exactly that record to decide whether a
file can stand in for a chunk. `_translation_trace.json` adds the per-chunk terms, logs and the raw
answer clipped to the trace budget. `07-translation-prompts/` is not an index but the evidence: the full
system prompt, user prompt, serialised messages, glossary rows and source chunk that produced one
translated file, with the answer or the exact failure. That folder is why a bad paragraph can be
debugged line by line — the prompt is not reconstructed after the fact, it is the one that was sent.

`08-translated-chapters/` is stage 8's text — one file per chapter, same filename as the source chapter
— and `_translated_chapters_manifest.json` is its index: per chapter the chunk count, how many were
translated, the character counts, the assembled SHA-256, the output path, and whether the run wrote
or reused the file. `--skip-if-present` compares the assembled bytes with the file already there, so
widening `--first-chapters` does not rewrite the earlier chapters. `_translated_chapters_trace.json`
is the same record with each chapter's missing chunks and error, if any.

To rebuild without any code, a `cat` of the `05-chunks/*.md` files in filename order is the whole book,
and a `cat` of `07-translations/*.md` in the same order is the whole translated book; the global index
is the first field of every name. Per chapter, the manifest's `chapter_filename` groups the files and
`chunk_index` orders them; the translated chunks keep those names, so the same grouping works for the
translation — and that grouping is exactly what stage 8 writes out as
`08-translated-chapters/<chapter>.md`. In Python, `book_translator.stages.chunking.reconstruct_chapter()` and
`reconstruct_book()` do the same without reading a filename, and
`book_translator.stages.assembly.run_assembly()` does it for the translated side.

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
writes nothing, `--limit` tags only the first chunks, a missing 04-glossary/chunks folder is one
actionable error, and the CLI resolves the folders from the book, prints the threshold and phases,
and returns the expected exit codes. The matcher's own tests pin the normalisation (punctuation
trimmed, hyphens and diacritics kept), the glossary loader (dedupe, bare list, missing file), the
threshold/scorer validation and the candidate ordering.

Stage 7's tests inject the translator, so no provider is called. They pin the stage's contract:
chunks are translated once and in reading order, every translated file keeps the chunk's name, the
prompt carries exactly that chunk's glossary rows (and the whole dictionary under
`--glossary-scope book`), the debug folder holds the whole input (system prompt, user prompt,
messages, glossary, source text, answer) plus a readable prompt file, `--limit` translates a prefix,
a failing chunk is reported without stopping the book and still leaves its prompt, a dry run writes
nothing, and `--resume` reuses finished and unchanged chunks (and refuses a different language pair
or a changed chunk hash). The translator's own tests script the structured-output endpoint: a valid
answer is accepted, a schema failure is retried with the failure named and both attempts are priced,
an empty/unchanged/summary answer is rejected, JSON mode and the provider-unsupported fallback work,
and every attempt failing raises with the usage attached. The CLI tests patch the same two seams and
pin the English → Serbian default, the folder defaults, the progress lines, the token total, the JSON
result and the exit codes.

Stage 8's tests run the real chunker and a real assembly over hand-written translations: each chapter
file is exactly its translated chunks concatenated in `chunk_index` order, `--first-chapters` writes
only that prefix, `--skip-if-present` reuses the old chapters and writes only the new ones, an
untranslated chunk marks its chapter incomplete (and no truncated file is written), a dry run writes
nothing, and the source `01-chapters/` folder is untouched. The pipeline tests also pin the whole-pipeline
window: splitting still runs over the full book, every later stage receives the first-N limit, the
chunk-based stages receive the chapter-aligned chunk count, and widening N with `--skip-if-present`
forwards the resume flags to every stage.

The opening audit (`tests/test_openings.py`) is pinned end to end, with no model: the exact
failure that motivated it — `# Poglavlje 2` followed by a stray `2` — is found and removed, with the
blank collapsed for stage 9 and kept as a slot for stage 2; a duplicated title, a leftover page
number and a heading that repeats the chapter number are removed; a numbered section heading
(`## 1`), a title that is only a number, a label-only title and a front matter page that opens on
prose are not; a first-line `## 2` is the *title* for stage 9 (so the plan sets it instead of
inserting a second heading), while a `## 18` page number above a real `## Poglavlje 1` is still
skipped; both stage runners apply the pass as their final phase (stage 2 keeping the line count,
its manifest/trace recording the counts); a dry run writes nothing; `--resume` reprocesses a
chapter whose detected title no longer matches the canonical title; and the tools return bounded
openings, never a paragraph.

Stage 11's tests (`tests/test_epub.py`) build a tiny book on disk and then read the zip the stage
produced, the way a reader would: the `mimetype` entry is first and stored, `META-INF/container.xml`
points at a package document that says EPUB 3 and carries `dc:title`/`dc:creator`/`dc:language`/
`dc:identifier`/`dcterms:modified`, the navigation document lists one entry per chapter with the
sections nested underneath and **no printed `Sadržaj` page**, and every spine document is
well-formed XHTML with its own `<title>`. One picture referenced twice is embedded once and both
`src` attributes are rewritten; a picture that cannot be resolved is an error *and* a visible
placeholder while the book is still written; a cover becomes the first page and the
`cover-image` manifest item. The package is byte-identical on a rebuild, a dry run writes nothing,
explicit metadata beats the recorded metadata, and a hand-built broken package (wrong mimetype,
compressed mimetype, truncated zip) is reported as a problem instead of a traceback.

### Replaying a map without the model

A chapter map is data, and it can be supplied instead of discovered. `--map FILE` skips the agent
entirely — no provider, no API key, no network — and puts the file through the *same* validator,
slicer and byte-for-byte verifier the graph uses:

```bash
python -m book_translator split BOOK.md --map my-map.json              # -> 01-chapters/
python -m book_translator split BOOK.md --map my-map.json --dry-run    # validate only
python scripts/run_pipeline.py BOOK.md --map my-map.json --stop-after split
```

The format is documented in [book_translator/stages/slicing/mapfile.py](book_translator/stages/slicing/mapfile.py):
a list of `{"line_number", "filename", "title"?, "role"?, "evidence"?}` entries, or an object
with that list under `splits` / `split_mapping` / `map` / `chapters` / `entries`. Aliases are
accepted (`line`/`start_line`, `file`/`name`, `kind` for `role`, `provenance` for `evidence`), so
a hand-recovered, richly documented map can be replayed unchanged — for example the one written
for *Zima u hotelu smrti* at
[data/output/Kenneth_M._Cameron_-_Zima_u_hotelu_smrti/01-chapters/_recovered_map.json](data/output/Kenneth_M._Cameron_-_Zima_u_hotelu_smrti/01-chapters/_recovered_map.json),
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
| [book_translator/paths.py](book_translator/paths.py) | where documents live: `data/input/` in, `data/output/` out |
| [book_translator/stages/conversion/runner.py](book_translator/stages/conversion/runner.py) | stage 0: EPUB/PDF/TXT → Markdown (`00-markdown/`), picture extraction to `00-markdown/images/`, the PDF text-layer gate (no OCR) and the conversion manifest |
| [book_translator/cli/](book_translator/cli/__init__.py) | the command-line package: `main`/`build_parser` compatibility surface, one module per command under `cli/commands/`, shared helpers (`cli/common.py`), argparse parents (`cli/parents.py`), the root parser (`cli/parser.py`) and the `pipeline` dispatch; `cli/legacy.py` is a re-export shim |
| [book_translator/pipeline/](book_translator/pipeline/__init__.py) | the one orchestrator: `options.py` (normalized options + exact runner kwargs), `paths.py` (`StagePaths`), `sources.py` (text/chunk fallbacks), `driver.py` (phase order, stop/start, records) and `parser.py` (the shared whole-pipeline CLI) |
| [book_translator/core/report.py](book_translator/core/report.py) | `human_seconds`, `human_bytes`, `human_count` and the small `recon`/`scan` payload summary shared by both front ends (stage narration stays with the front end that owns it) |
| [book_translator/core/constants.py](book_translator/core/constants.py) | every guardrail and default in one place |
| [book_translator/core/settings.py](book_translator/core/settings.py) | provider and credential resolution |
| [book_translator/core/errors.py](book_translator/core/errors.py) | `ToolError`, `ConfigError` |
| [book_translator/core/llm.py](book_translator/core/llm.py) | the one structured-call loop shared by stages 4 and 7: `build_chat_model`, `StructuredCaller`, `TokenUsage`, answer parsing and the provider-refusal latch |
| [book_translator/core/artifacts.py](book_translator/core/artifacts.py) | atomic `write_json`/`write_text`, manifest/trace readers and writers, the UTC timestamp convention and the one SHA-256 family |
| [book_translator/core/run.py](book_translator/core/run.py) | `RunContext` (clock, dry-run, limit, progress) and `run_items` with per-item error isolation |
| [book_translator/documents/io.py](book_translator/documents/io.py) | cached source reading |
| [book_translator/documents/images.py](book_translator/documents/images.py) | find/protect/restore Markdown and HTML image references; the `[[IMG_n]]` round-trip every stage shares |
| [book_translator/documents/markdown.py](book_translator/documents/markdown.py) | header/standalone/prose line shapes (both stages) |
| [book_translator/documents/bodyline.py](book_translator/documents/bodyline.py) | body-line *damage* shapes: split words, mid-word page numbers |
| [book_translator/documents/book.py](book_translator/documents/book.py) | the one `BookMetadata` record: title/author/stem plus the publication fields both the matter and EPUB stages share |
| [book_translator/documents/chapters.py](book_translator/documents/chapters.py) | reads `01-chapters/` (or the cleaned `02-preprocessed/`) back as chapter records, from `_manifest.json` or `_preprocess_manifest.json`, and exposes the stage-folder helpers |
| [book_translator/documents/layout.py](book_translator/documents/layout.py) | the one `STAGE_DIRS` registry and `resolve_stage_dir`: a book folder, a sibling stage folder or a file all resolve to the stage being written |
| [book_translator/documents/chunks.py](book_translator/documents/chunks.py) | the read side of stage 5's folder: `ChunkInput`, `discover_chunk_inputs`, `chunk_files`, chunk order and per-chunk file naming |
| [book_translator/documents/manifest.py](book_translator/documents/manifest.py) | every stage manifest reader (`load_split_manifest`, `load_chunk_manifest`, `load_translation_manifest`, …), so a stage never imports the stage that wrote one |
| [book_translator/documents/window.py](book_translator/documents/window.py) | the `--first-chapters N` window: body-chapter selection, file-count arithmetic for chapter-based stages, and chunk-membership filtering for chunk-based stages |
| [book_translator/documents/text.py](book_translator/documents/text.py) | folding, similarity, slugs, tool-result caps |
| [book_translator/stages/analysis/toc.py](book_translator/stages/analysis/toc.py) | printed-TOC block detection |
| [book_translator/stages/analysis/headers.py](book_translator/stages/analysis/headers.py) | `strong` / `noisy` / `absent` verdict |
| [book_translator/stages/analysis/candidates.py](book_translator/stages/analysis/candidates.py) | heuristic title candidates |
| [book_translator/stages/analysis/locate.py](book_translator/stages/analysis/locate.py) | fuzzy title → line number, heading snapping |
| [book_translator/stages/slicing/plan.py](book_translator/stages/slicing/plan.py) | safe filenames and line ranges |
| [book_translator/stages/slicing/slice.py](book_translator/stages/slicing/slice.py) | verbatim copy, manifest, dry-run preview |
| [book_translator/stages/slicing/validation.py](book_translator/stages/slicing/validation.py) | deterministic map scoring |
| [book_translator/stages/slicing/verify.py](book_translator/stages/slicing/verify.py) | SHA-256 proof that nothing was lost |
| [book_translator/stages/preprocess/inspect.py](book_translator/stages/preprocess/inspect.py) | one chapter's headers + evidence + deterministic recommendation |
| [book_translator/stages/preprocess/damage.py](book_translator/stages/preprocess/damage.py) | damage detection and the three repair rules |
| [book_translator/stages/preprocess/corpus.py](book_translator/stages/preprocess/corpus.py) | sampling and the book-wide damage profile |
| [book_translator/stages/preprocess/plan.py](book_translator/stages/preprocess/plan.py) | the legal-edit vocabulary, its predicates and the two validators (no writes) |
| [book_translator/stages/preprocess/apply.py](book_translator/stages/preprocess/apply.py) | pure line rewriting, the provenance map and the `LineProvenance` index |
| [book_translator/stages/preprocess/verify.py](book_translator/stages/preprocess/verify.py) | the content proof that only declared lines changed |
| [book_translator/stages/openings/inspect.py](book_translator/stages/openings/inspect.py) | the final phase of stages 2/9: one chapter's opening — title candidate, ≤12 clipped lines and the provable issue vocabulary |
| [book_translator/stages/openings/repair.py](book_translator/stages/openings/repair.py) | the final phase: pure removal of the audited junk lines (blank slot for stage 2, collapsed blank for stage 9) |
| [book_translator/stages/openings/phase.py](book_translator/stages/openings/phase.py) | the final phase: the book-wide pass, counts/findings and the per-chapter report both stage runners store |
| [book_translator/stages/dictionary/sentences.py](book_translator/stages/dictionary/sentences.py) | stage 3: sentence splitting and the two dependency-free filters |
| [book_translator/stages/dictionary/language.py](book_translator/stages/dictionary/language.py) | stage 3: sampled language detection (`lingua-py`), the ekavian/ijekavian tie-breaker, backend routing |
| [book_translator/stages/dictionary/ner.py](book_translator/stages/dictionary/ner.py) | stage 3: the CLASSLA/stanza gate, entity rules, GPU selection and the device actually used |
| [book_translator/stages/dictionary/unique.py](book_translator/stages/dictionary/unique.py) | stage 3: the noun key and the first-mention filter (book-wide, in chapter order) |
| [book_translator/stages/dictionary/runner.py](book_translator/stages/dictionary/runner.py) | stage 3: `run_dictionary`, the chapter loop, timings, `03-reduced/` + `03-reduced-unique/` and their manifest |
| [book_translator/stages/glossary/models.py](book_translator/stages/glossary/models.py) | stage 4: the Pydantic answer (`term` / `category` / `translation`) and category normalisation |
| [book_translator/stages/glossary/prompts.py](book_translator/stages/glossary/prompts.py) | stage 4: the book-translation system prompt, the incremental chapter prompt, the dictionary renderer |
| [book_translator/stages/glossary/extract.py](book_translator/stages/glossary/extract.py) | stage 4: the chapter schema and outcome over `core.llm.StructuredCaller` |
| [book_translator/stages/glossary/book.py](book_translator/stages/glossary/book.py) | stage 4: the live dictionary — first rendering wins, repeats deduplicated, drift recorded as a conflict |
| [book_translator/stages/glossary/records.py](book_translator/stages/glossary/records.py) | stage 4: the four records (`GlossaryTerm`, `ChapterPlan`, `ChapterGlossary`, `GlossaryResult`) and the chapter selection |
| [book_translator/stages/glossary/runner.py](book_translator/stages/glossary/runner.py) | stage 4: `run_glossary`, the ordered chapter walk, the front/back-matter skip and the four artifacts |
| [book_translator/stages/chunking/boundaries.py](book_translator/stages/chunking/boundaries.py) | stage 5: names the boundary kind of a cut (paragraph / sentence / line / word / hard) |
| [book_translator/stages/chunking/tokenizer.py](book_translator/stages/chunking/tokenizer.py) | stage 5: resolves the tokenizer and the offline approximation, with the reason recorded |
| [book_translator/stages/chunking/runner.py](book_translator/stages/chunking/runner.py) | stage 5: `run_chunking`, the reversible cut, the manifest, the skip/reuse rules and the reconstruction helpers |
| [book_translator/stages/chunking/__init__.py](book_translator/stages/chunking/__init__.py) | stage 5: public write side (`run_chunking`) and read side (`reconstruct_chapter` / `reconstruct_book`) |
| [book_translator/stages/chunk_glossary/matching.py](book_translator/stages/chunk_glossary/matching.py) | stage 6: glossary loading, term normalisation and the rapidfuzz matcher (threshold and scorer are options) |
| [book_translator/stages/chunk_glossary/runner.py](book_translator/stages/chunk_glossary/runner.py) | stage 6: `run_chunk_glossary`, the two phases (NER over every chunk, then retrieval), the per-chunk files and the manifest/trace |
| [book_translator/stages/chunk_glossary/__init__.py](book_translator/stages/chunk_glossary/__init__.py) | stage 6: public surface (`run_chunk_glossary`, the matcher and the per-chunk records) |
| [book_translator/stages/matter/metadata.py](book_translator/stages/matter/metadata.py) | stage 6b: `Title - Author` parsing from the input filename (re-exports the shared `BookMetadata`) |
| [book_translator/stages/matter/generate.py](book_translator/stages/matter/generate.py) | stage 6b: the English text of a synthetic front/back matter section |
| [book_translator/stages/matter/records.py](book_translator/stages/matter/records.py) | stage 6b: split/chunk manifest records, totals and the two manifest updates |
| [book_translator/stages/matter/synthesize.py](book_translator/stages/matter/synthesize.py) | stage 6b: `ensure_matter_sections` — decide what is missing and write it |
| [book_translator/stages/translation/models.py](book_translator/stages/translation/models.py) | stage 7: the Pydantic answer — one `translation` field, non-blank |
| [book_translator/stages/translation/prompts.py](book_translator/stages/translation/prompts.py) | stage 7: the constant translator system prompt and the per-chunk prompt (glossary, optional context, source, task) |
| [book_translator/stages/translation/translator.py](book_translator/stages/translation/translator.py) | stage 7: the translation schema, quality validation and JSON recovery over `core.llm.StructuredCaller` |
| [book_translator/stages/translation/runner.py](book_translator/stages/translation/runner.py) | stage 7: `run_translation`, the ordered chunk walk, `--limit`/resume rules, the translation files and the whole-input debug records |
| [book_translator/stages/translation/__init__.py](book_translator/stages/translation/__init__.py) | stage 7: public surface (`run_translation`, the prompt builders, the translator and the records) |
| [book_translator/stages/assembly/runner.py](book_translator/stages/assembly/runner.py) | stage 8: `run_assembly`, the chapter grouping, all-or-nothing assembly, the resume comparison and the manifest/trace |
| [book_translator/stages/assembly/__init__.py](book_translator/stages/assembly/__init__.py) | stage 8: public surface (`run_assembly`, the result record and the folder resolvers) |
| [book_translator/stages/postprocess/inspect.py](book_translator/stages/postprocess/inspect.py) | stage 9: the chapter-title inventory, bounded openings, heading records and matter inspection |
| [book_translator/stages/postprocess/titles.py](book_translator/stages/postprocess/titles.py) | stage 9: title text primitives and the canonical chapter-title policy |
| [book_translator/stages/postprocess/rules/](book_translator/stages/postprocess/rules/__init__.py) | stage 9: validation split by rule group (evidence, naming, overrides, TOC, titles, chapter title, headings) |
| [book_translator/stages/postprocess/plan.py](book_translator/stages/postprocess/plan.py) | stage 9: the public validators (thin orchestrators), chapter apply and provenance verification |
| [book_translator/stages/postprocess/toc.py](book_translator/stages/postprocess/toc.py) | stage 9: TOC detection, generation, in-place update and consistency verification |
| [book_translator/stages/postprocess_agent/graph.py](book_translator/stages/postprocess_agent/graph.py) | stage 9: the book-level and per-chapter `StageSpec`s, compiled by the scaffold |
| [book_translator/stages/postprocess_agent/runner.py](book_translator/stages/postprocess_agent/runner.py) | stage 9: `run_postprocess`, the book pass, the ordered chapter walk and the TOC write |
| [book_translator/stages/binding/runner.py](book_translator/stages/binding/runner.py) | stage 10: `run_bind`, the ordered single-file concatenation and its verification |
| [book_translator/stages/epub/runner.py](book_translator/stages/epub/runner.py) | stage 11: `run_epub`, the chapter render, picture embedding, metadata resolution and the package write |
| [book_translator/stages/epub/verify.py](book_translator/stages/epub/verify.py) | stage 11: post-write verification, split by rule (mimetype, container, package document, navigation, documents, cover) |
| [book_translator/stages/epub/render.py](book_translator/stages/epub/render.py) | stage 11: Markdown → XHTML (`markdown_it`, imported lazily), heading anchors, picture figures |
| [book_translator/stages/epub/package.py](book_translator/stages/epub/package.py) | stage 11: the EPUB 3 container — zip layout, package document, navigation document, cover page |
| [book_translator/stages/epub/metadata.py](book_translator/stages/epub/metadata.py) | stage 11: `dc:` metadata from stage 6b's records, the stable UUID, the package timestamp and the ZIP date encoding |
| [book_translator/stages/epub/assets/style.css](book_translator/stages/epub/assets/style.css) | stage 11: the stylesheet every package ships |
| [book_translator/stages/agent/prompts.py](book_translator/stages/agent/prompts.py) | stage 1 system prompt and the fallback protocol |
| [book_translator/stages/agent/llm.py](book_translator/stages/agent/llm.py) | the chat model with the toolset bound |
| [book_translator/stages/agent/tools/registry.py](book_translator/stages/agent/tools/registry.py) | tool registry with the `shared` / `split` / `preprocess` stage tags |
| [book_translator/stages/agent/tools/shared.py](book_translator/stages/agent/tools/shared.py) | the generic tools both stages bind |
| [book_translator/stages/agent/tools/](book_translator/stages/agent/tools/__init__.py) | stage 1 tools: `extract` / `toc` / `proposal` / `execution` |
| [book_translator/toolkit/preprocess/tools.py](book_translator/toolkit/preprocess/tools.py) | stage 2 tools: header collection, damage, plan submission, apply |
| [book_translator/toolkit/postprocess/tools.py](book_translator/toolkit/postprocess/tools.py) | stage 9 tools: title inventory, heading records, plan submission, apply/verify |
| [book_translator/toolkit/matter/tools.py](book_translator/toolkit/matter/tools.py) | the front/back matter readers and localization tools |
| [book_translator/toolkit/context.py](book_translator/toolkit/context.py) | `ToolContext`: the run's folder/book/section/chunk, passed to tools explicitly instead of through a `ContextVar` |
| [book_translator/stages/agent/tools/openings.py](book_translator/stages/agent/tools/openings.py) | the chapter-opening audit tools shared by stages 2 and 9 (`audit_chapter_opening`, `audit_book_openings`) |
| [book_translator/stages/agent/nodes/](book_translator/stages/agent/nodes/__init__.py) | `scout` / `validate` / `slice` / `verify`, plus the splitter's `give_up` override |
| [book_translator/stage_graph/](book_translator/stage_graph/__init__.py) | the shared scaffold every stage is compiled from: `AgentStateBase`, `StageSpec`, `build_stage_graph` and the three routers |
| [book_translator/stage_graph/nodes.py](book_translator/stage_graph/nodes.py) | the shared `agent` / `tools` / `give_up` node bodies, `absorb_plan` and `usage` |
| [book_translator/stages/agent/graph.py](book_translator/stages/agent/graph.py) | stage 1 `StageSpec` (`scout` → `agent` ↔ `tools` → `validate` → `slice` → `verify`) and the dry-run router, compiled by the scaffold |
| [book_translator/stages/agent/runner.py](book_translator/stages/agent/runner.py) | `run_split` and the trace writer |
| [book_translator/stages/agent/state.py](book_translator/stages/agent/state.py) | stage 1 graph state (metadata only, never prose); the shared keys come from `AgentStateBase` |
| [book_translator/stages/agent/result.py](book_translator/stages/agent/result.py) | the `SplitterResult` dataclass |
| [book_translator/stages/preprocess_agent/prompts.py](book_translator/stages/preprocess_agent/prompts.py) | stage 2 system prompt and the decision vocabulary |
| [book_translator/stages/preprocess_agent/llm.py](book_translator/stages/preprocess_agent/llm.py) | the model with *only* the preprocessing tools bound |
| [book_translator/stages/preprocess_agent/nodes.py](book_translator/stages/preprocess_agent/nodes.py) | `collect` / `validate` / `apply` / `verify` |
| [book_translator/stages/preprocess_agent/graph.py](book_translator/stages/preprocess_agent/graph.py) | stage 2 `StageSpec` for **one chapter**, compiled by the scaffold |
| [book_translator/stages/preprocess_agent/runner.py](book_translator/stages/preprocess_agent/runner.py) | `run_preprocess`: the sequential (or pooled) chapter loop |
| [book_translator/stages/preprocess_agent/state.py](book_translator/stages/preprocess_agent/state.py) | stage 2 graph state (metadata only, never prose); the shared keys come from `AgentStateBase` |
| [tests/](tests/conftest.py) | the test suite, split the same way as the code |

## Related but external

[external/](external/README.md) is **not part of this repository**. It holds third-party
checkouts — currently `TranslateBooksWithLLMs`, itself a separate git repository — for local
side-by-side use. The whole folder is gitignored (only its README is tracked), and
`book_translator` never imports from it.

## Limitations

- **Stage 0 never OCRs.** A PDF with no embedded text layer is rejected with a report of how many
  sampled pages carried text; run OCR outside the pipeline (or supply the EPUB/TXT edition) and
  re-run. A scanned PDF that *does* carry an invisible OCR layer is accepted, because that layer is
  text the source already contains. EPUB/PDF conversion is docling's layout interpretation: it may
  produce different heading levels than the printed book, which is exactly the damage stages 1-2
  are designed to repair. A TXT is copied verbatim, so it must already use Markdown headings (or be
  split by the heuristic/prose-chapter strategies).
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
- **A caption now stays with its picture.** Before stage 0 extracted images, a
  figure caption was a label for something the Markdown did not contain, so
  stage 2 deleted it. Since images are preserved, captions are kept: a body
  caption is left alone (`keep`) and a caption that the converter marked up as a
  heading is `demote`d to a paragraph. The old `remove_caption` decision remains
  accepted for compatibility, but the prompt no longer recommends it.
- Stage 2's `merge_with_next` joins two source lines into one, `remove` (and the
  legacy `remove_caption`) leave a **blank slot** where the artefact was, and
  `promote` adds `##` to a plain ALL-CAPS heading. Those output lines are not
  byte-equal to their source by design, which is why the provenance map records
  the source lines and the verifier re-derives the transformation instead of
  comparing bytes.
- Stage 2 reads only the *headers* of a chapter, plus the page numbers and captions it recognises in
  the body, unless the agent explicitly asks for a window or a damage scan — so a defect that has no
  header and no recognised shape is invisible to it. A blank slot also means the cleaned chapter has
  the same line numbering as `01-chapters/`, at the cost of a few consecutive blank lines.
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
  `03-reduced-unique/` is the list it reads by default — the whole first-mention set, one sentence per
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
  chapter's; delete `04-glossary/` or run without `--resume` to rebuild.
- The first-mention filter keys on the **surface form** of a mention, case-folded and trimmed, and
  ignores its entity type. Two inflections of the same name (`Marko` / `Marku`) are therefore two
  nouns, and one name tagged `PER` in one place and `MISC` in another is one noun. It also cannot
  remove the heading/fragment noise above: a heading is a new noun the first time it appears. What
  it does guarantee is the shape of the output — every surviving sentence introduces at least one
  noun, and the union of the kept sentences' nouns is the book's whole noun set.
- **Stage 7 translates chunks, not chapters.** A chunk is an exact slice, so a
  chunk can begin or end mid-sentence; the prompt says so, but the model still sees
  only that slice. `--context-chars N` sends the tail of the previous translation
  to bridge the seam, and it is off by default because the chunker already prefers
  paragraph and sentence boundaries. If a book's chunks break mid-sentence often,
  raise it and compare a few joins.
- **The per-chunk glossary is only as complete as stage 6's NER.** A name the
  tagger missed (or one below the fuzzy threshold) is not in the prompt, so the
  model renders it on its own. `--glossary-scope book` sends the whole
  `glossary.json` instead and cannot miss an entry, at the price of a much larger
  prompt for every chunk. The debug record says exactly which rows were sent
  (`.glossary.entries`), so a name rendered inconsistently can be traced back to
  whether it was in the prompt at all.
- **The stage validates the answer's shape, not its fidelity.** It catches an empty
  answer, an unchanged source, a summary that is a fraction of the source's length,
  and a provider refusal — but a fluent translation that changes a fact is beyond
  what a schema can see. Read `07-translations/` (or diff a chapter against its source)
  before publishing; the per-chunk prompts and answers in `07-translation-prompts/`
  are the evidence for that review.
- **Structured output depends on the provider.** `--method auto` uses native
  structured output and moves to JSON mode when a provider rejects the schema; the
  run records `json_only` in its settings when that happened. A model that cannot
  produce the `{"translation": ...}` object at all will fail its chunks after
  `--max-attempts` calls, and those chunks are reported (the rest of the book is
  still translated).
- **Stage 7's `--resume` is safer than stage 4's but still trusts the manifest.** A
  chunk is reused only when the recorded source SHA-256 matches the file now, the
  translation file is non-empty, and the language pair is unchanged — but a
  translation edited by hand after the run is kept, because the manifest records
  the run's output hash and not a reason to distrust the file. Delete the
  translation (or run without `--resume`) to re-translate a chunk.
- **Stage 8 is all-or-nothing per chapter.** If one chunk of a chapter is missing
  (a failed translation, or `translate --limit` stopping mid-chapter), that chapter
  is reported `incomplete` and not written, because a truncated chapter would look
  complete to a reader. The manifest and the command's summary name the missing
  chunks; run `translate` for them (or use a chapter-aligned `--first-chapters`)
  and re-run `concatenate`. `--skip-if-present` trusts the bytes on disk: a chapter
  file whose bytes already equal the current concatenation is reused, while a
  hand-edited file differs and is therefore rewritten; run without the flag (or
  edit the translations) if the assembly is not authoritative for that chapter.
- **Trailing newlines are preserved, leading and internal whitespace are the
  model's.** A translated chunk ends with the same number of newlines as its source
  chunk, so concatenation keeps the paragraph breaks, but the model may re-indent a
  list or losslessly re-wrap a line. `07-translations/*.md` is translated text with the
  source's Markdown structure, not a byte-level diff of the source.
