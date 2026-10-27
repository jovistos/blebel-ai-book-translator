# Pipeline Stages 0 to 5: Ingestion, Decomposition, Terminology & Chunking

This document provides complete, systematic specifications for the first six stages of the pipeline: from converting the raw source document up to cutting chapters into byte-reversible translation chunks.

---

## Stage 0: Conversion (`book_translator.stages.conversion`)

> **Question Answered:** *What does this source document say as plain Markdown, and where are its pictures?*

### Purpose & Invariants
* Accepts `.epub`, `.pdf`, `.txt`, and `.md` formats.
* If the source is already Markdown (`.md` or `.markdown`), Stage 0 is a no-op; Stage 1 reads the file directly.
* Plain text (`.txt`) is copied byte-for-byte under a `.md` filename.
* EPUB and PDF files are converted using **Docling**.
* **Strict OCR Refusal Policy:** For PDF files, embedded text layers are checked before conversion. Scanned or image-only PDFs without a real text layer are rejected immediately with an actionable error. The pipeline will not guess or invent text via OCR.
* **Image Extraction:** Embedded images are extracted to `00-markdown/images/` and referenced in Markdown using standard image tags (`![caption](images/<file>)`). Use `--no-images` to disable picture extraction.

### Artifacts Written
* `data/output/<book>/00-markdown/<book>.md`
* `data/output/<book>/00-markdown/images/*` (illustrations, maps, photos)
* `data/output/<book>/00-markdown/_convert_manifest.json` (records source file SHA-256 and conversion settings)

---

## Stage 1: Chapter Split (`book_translator.stages.agent` & `slicing`)

> **Question Answered:** *Where does each chapter start, without reading the body prose?*

### Purpose & Invariants
* Analyzes the source Markdown structure using the **Split Agent** (LangGraph).
* Operates strictly on line numbers, headers, and pattern matches. The agent never reads prose paragraphs.
* Slices the source file **verbatim**. No lines are reflowed, re-indented, or re-encoded.
* **Lossless Concatenation Guarantee:** Concatenating all files in `01-chapters/` in natural numerical order reproduces the source Markdown file byte-for-byte.

### Chapter Classification
Each slice is categorized into one of three kinds:
1. `front_matter`: Title page, copyright page, dedication, preface, foreword (`00-front-matter.md`).
2. `body`: Numbered or titled main book chapters (`01-chapter-1.md`, `02-chapter-2.md`, ...).
3. `back_matter`: Epilogue, afterword, acknowledgments, index, notes (`99-back-matter.md`).

### Artifacts Written
* `data/output/<book>/01-chapters/00-front-matter.md`
* `data/output/<book>/01-chapters/01-*.md`, `02-*.md`, ...
* `data/output/<book>/01-chapters/99-back-matter.md`
* `data/output/<book>/01-chapters/_split_manifest.json`
* `data/output/<book>/01-chapters/_agent_trace.json`

---

## Stage 2: Preprocess (`book_translator.stages.preprocess_agent`)

> **Question Answered:** *What in this chapter is layout damage, page numbers, or misplaced headers?*

### Purpose & Invariants
* Cleans conversion defects chapter by chapter using the **Preprocess Agent**.
* Detects running headers, running footers, printed page numbers embedded in text, and headings broken across lines.
* Evaluates proposed changes against a strict 5-action vocabulary (`keep`, `promote`, `demote`, `remove`, `merge_with_next`).
* Operates as a pure function of `(chapter_path, output_dir)` to guarantee deterministic, isolated results.
* Runs the deterministic **Chapter Openings Audit** at completion to clean any lingering stray numbers before prose begins.

### Provenance Tracking
Generates a complete per-line provenance record:
```json
{
  "source_line": 142,
  "output_line": 139,
  "operation": "remove",
  "reason": "isolated printed page number '42'"
}
```

### Artifacts Written
* `data/output/<book>/02-preprocessed/00-front-matter.md`
* `data/output/<book>/02-preprocessed/01-*.md`, ...
* `data/output/<book>/02-preprocessed/_preprocess_manifest.json`
* `data/output/<book>/02-preprocessed/_preprocess_trace.json`

---

## Stage 3: Dictionary Funnel (`book_translator.stages.dictionary`)

> **Question Answered:** *Which sentences carry a proper noun worth keeping consistent across the entire book?*

### Purpose & Invariants
* A **deterministic sentence extractor** combined with a high-speed Named Entity Recognition (NER) tagger.
* **No LLM is called in Stage 3.** It operates purely on CPU/GPU language models:
  * South Slavic / Balkan languages: **CLASSLA** pipeline.
  * English and other Western languages: **Stanza** pipeline.
* Sentence Funnel rules:
  1. Splits chapter prose into individual sentences.
  2. Discards sentences shorter than `DEFAULT_MIN_SENTENCE_WORDS` (default: 4 words).
  3. Tags Named Entities (`PER` for Person, `LOC` for Location, `ORG` for Organization).
  4. Keeps only sentences that carry genuine named entities.
* Generates two outputs:
  * Full reduced sentences (`03-reduced/`).
  * **First Mentions Only** (`03-reduced-unique/`): Keeps only the first occurrence of each entity across the book, dramatically reducing token costs for Stage 4.

### Artifacts Written
* `data/output/<book>/03-reduced/*.md`
* `data/output/<book>/03-reduced-unique/*.md`
* `data/output/<book>/03-reduced/_reduced_manifest.json`

---

## Stage 4: Glossary Builder (`book_translator.stages.glossary`)

> **Question Answered:** *How must each named entity be rendered in the target language, once and for all?*

### Purpose & Invariants
* Iterates through the first-mention sentences in `03-reduced-unique/` **in reading order**.
* Calls a structured LLM (via `StructuredCaller`) to lock in target language translations for characters, place names, organizations, and recurring terms of art.
* **Accumulator Memory Architecture:**
  * When translating Chapter 1 entities, the glossary is empty.
  * When translating Chapter 2 entities, the prompt includes all terms locked in Chapter 1.
  * By Chapter 40, the model receives the complete book-wide glossary so far, preventing conflicting translations of recurring terms.
* **Front & Back Matter Skipped by Default:** Prevents locking original edition publisher names, copyright dates, or translator names into the translation glossary.

### Pydantic Output Schema
```json
{
  "terms": [
    {
      "source": "Hercule Poirot",
      "target": "Erkil Poaro",
      "category": "character",
      "notes": "Belgian detective, phonetic Serbian Cyrillic/Latin"
    }
  ]
}
```

### Artifacts Written
* `data/output/<book>/04-glossary/glossary.json` (Machine-readable term dictionary)
* `data/output/<book>/04-glossary/glossary.md` (Human-readable markdown table)
* `data/output/<book>/04-glossary/_glossary_manifest.json`

---

## Stage 5: Reversible Chunking (`book_translator.stages.chunking`)

> **Question Answered:** *Where can the chapter be cut into translation-sized chunks without breaking sentences or paragraphs?*

### Purpose & Invariants
* Cuts each preprocessed chapter into translation chunks (default target: ~1000 tokens) using `chonkie`'s recursive chunker.
* Respects natural textual boundaries in strict priority order:
  1. Markdown paragraph breaks (`\n\n`)
  2. Sentence terminators (`. `, `! `, `? `)
  3. Soft line breaks (`\n`)
  4. Word spaces
* **Zero Text Loss / Absolute Invariant:** Concatenating a chapter's chunks restores the original chapter file byte-for-byte:
  $$\bigoplus_{k=0}^{M} \text{chunk}_k \equiv \text{chapter}$$
* Chunks are numbered globally (`00001-...`, `00002-...`) so directory sorting reflects the exact reading order of the entire book.

### Artifacts Written
* `data/output/<book>/05-chunks/00000-00-front-matter-0000.md`
* `data/output/<book>/05-chunks/00001-01-chapter-1-0000.md`
* `data/output/<book>/05-chunks/00002-01-chapter-1-0001.md`
* `data/output/<book>/05-chunks/_chunk_manifest.json`
* `data/output/<book>/05-chunks/_chunk_trace.json`
