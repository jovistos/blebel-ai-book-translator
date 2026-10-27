# Data Layout, Manifests & Storage Contracts

`book-translator` organizes all filesystem artifacts strictly according to stage number and purpose. This prevents accidental overwrites, guarantees byte-level traceability, and ensures any interrupted run can be diagnosed or resumed.

---

## 1. Directory Structure

All source files reside in `data/input/`. All intermediate and final artifacts are written under `data/output/<book-slug>/`:

```
data/
├── input/
│   ├── Book Title - Author.epub
│   └── Another Book.md
└── output/
    └── Book-Title---Author/
        ├── 00-markdown/                      <- Stage 0: Converted Markdown & illustrations
        │   ├── Book-Title---Author.md
        │   ├── images/
        │   │   ├── img-001.png
        │   │   └── img-002.jpg
        │   └── _convert_manifest.json
        ├── 01-chapters/                      <- Stage 1: Verbatim chapter slices
        │   ├── 00-front-matter.md
        │   ├── 01-chapter-1.md
        │   ├── ...
        │   ├── 99-back-matter.md
        │   ├── _split_manifest.json
        │   └── _agent_trace.json
        ├── 02-preprocessed/                  <- Stage 2: Cleaned chapters
        │   ├── 01-chapter-1.md
        │   ├── _preprocess_manifest.json
        │   └── _preprocess_trace.json
        ├── 03-reduced/                       <- Stage 3: Entity sentences
        │   └── 01-chapter-1.md
        ├── 03-reduced-unique/                <- Stage 3: First mentions only
        │   └── 01-chapter-1.md
        ├── 04-glossary/                      <- Stage 4: Book terminology
        │   ├── glossary.json
        │   ├── glossary.md
        │   └── _glossary_manifest.json
        ├── 05-chunks/                        <- Stage 5: Reversible chunks
        │   ├── 00001-01-chapter-1-0000.md
        │   ├── 00002-01-chapter-1-0001.md
        │   └── _chunk_manifest.json
        ├── 06-chunk-glossary/                <- Stage 6: Per-chunk terms
        │   ├── 00001-01-chapter-1-0000.json
        │   └── _chunk_glossary_manifest.json
        ├── 06b-matter/                       <- Stage 6b: Matter plan & metadata
        │   ├── matter_plan.json
        │   ├── _matter_manifest.json         <- Chunk metadata & image recreation records
        │   └── _matter_trace.json            <- Full logs including multimodal image operations
        ├── 06b-matter-prompts/               <- Stage 6b: Agent prompts & logs
        ├── 07-translations/                  <- Stage 6b & 7: Translated chunks & images
        │   ├── images/                       <- Stage 6b: Multimodal recreated images (*_translated.*)
        │   │   └── cover_translated.jpg
        │   ├── 00000-00-front-matter-0000.md
        │   ├── 00001-01-chapter-1-0000.md
        │   └── _translation_manifest.json
        ├── 07-translation-prompts/           <- Stage 7: Exact prompt and response audits
        │   ├── 00001-01-chapter-1-0000.json
        │   └── 00001-01-chapter-1-0000.prompt.md
        ├── 08-translated-chapters/           <- Stage 8: Reassembled translated chapters
        │   ├── 01-chapter-1.md
        │   └── _assembly_manifest.json
        ├── 09-postprocessed-chapters/        <- Stage 9: Harmonized titles & TOC
        │   ├── 00-table-of-contents.md
        │   ├── 01-chapter-1.md
        │   └── _postprocess_manifest.json
        ├── 10-translated-book.md             <- Stage 10: Complete Markdown translation
        └── 11-translated-book.epub           <- Stage 11: Reader-ready EPUB 3 publication
```

---

## 2. Manifest Specifications

Every stage records a machine-readable manifest (`_<stage>_manifest.json`) capturing the input parameters, execution timestamps, source SHA-256 hashes, and output file manifests.

### A. Stage 1 Split Manifest (`_split_manifest.json`)
Records how the source file was divided:
```json
{
  "source_path": "data/input/The-Great-Gatsby.md",
  "source_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "total_lines": 4210,
  "chapter_count": 9,
  "strategy": "langgraph_split_agent",
  "files": [
    {
      "name": "01-chapter-1.md",
      "start_line": 45,
      "end_line": 420,
      "line_count": 375,
      "sha256": "4b227777d4dd1fc61c6f884f48641d02b4d121d3fd328cb08b5531fcacdabf8a"
    }
  ],
  "concatenation_verified": true
}
```

### B. Stage 5 Chunk Manifest (`_chunk_manifest.json`)
Guarantees byte-for-byte reconstruction:
```json
{
  "chapter": "01-chapter-1.md",
  "chunk_count": 4,
  "chunks": [
    {
      "filename": "00001-01-chapter-1-0000.md",
      "global_index": 1,
      "chunk_index": 0,
      "char_start": 0,
      "char_end": 4120,
      "token_count": 980,
      "left_boundary": "start",
      "right_boundary": "paragraph"
    }
  ],
  "reconstruction_byte_exact": true
}
```

### C. Stage 6 Chunk Glossary (`06-chunk-glossary/<chunk>.json`)
Supplies matched entities directly to translation prompts:
```json
{
  "chunk": "00001-01-chapter-1-0000.md",
  "chapter": "01-chapter-1.md",
  "matched_terms": [
    {
      "source": "Daisy Buchanan",
      "target": "Dejzi Bjukenen",
      "category": "character",
      "similarity_score": 100.0
    },
    {
      "source": "East Egg",
      "target": "Ist Eg",
      "category": "location",
      "similarity_score": 100.0
    }
  ]
}
```

---

## 3. Auditing & Prompt Tracing

`book-translator` guarantees that no translation output is an inscrutable "black box":
* **`07-translation-prompts/<chunk>.json`**: Contains the full payload sent to the LLM client, including system instructions, formatted terminology tables, preceding translation context, source markdown chunk, and raw model output.
* **`07-translation-prompts/<chunk>.prompt.md`**: Human-readable markdown rendition of the prompt for rapid manual inspection.
* **`_agent_trace.json`**: Records every LangGraph tool invocation, argument payload, and validation rejection during agent execution.
