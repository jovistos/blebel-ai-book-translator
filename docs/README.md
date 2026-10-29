# Documentation Portal: BleBelTranslateBook (`book-translator`)

Welcome to the comprehensive, systematic documentation for **BleBelTranslateBook** (package name: `book-translator`).

This documentation covers every facet of the system: from architectural principles and multi-agent state machines to stage-by-stage data contracts, CLI commands, model provider setup, and developer guidelines.

---

## Documentation Map

```
docs/
├── README.md                                  <- You are here (Master Index & Navigation)
├── architecture/
│   ├── overview.md                            <- Core philosophy, design invariants & execution flow
│   └── langgraph-agents.md                    <- The 4 LangGraph agents (Split, Preprocess, Matter, Postprocess)
├── pipeline/
│   ├── stages-1-to-6.md                       <- Ingestion, splitting, preprocessing, dictionary, glossary, chunking
│   └── stages-7-to-12.md                      <- Matter, translation, assembly, postprocessing, bind, EPUB
├── configuration/
│   └── models-and-providers.md                <- TOML config, providers (DeepSeek, LM Studio, OpenAI, Gemini, Claude), env vars
├── cli/
│   └── reference.md                           <- book-translator & pipeline CLI reference, subcommands, wizard, scripts
├── data-layout-and-contracts.md               <- Directory conventions (01 to 12), manifests, traces, schemas
└── development/
    └── contributing-and-testing.md            <- Setup, 800+ test suite, snapshot testing, ruff, mypy, packaging
```

---

## Reading Paths by Role

### 1. I Want to Translate a Book (Operator / User)
1. Read the root [README.md](../README.md) for quickstart and dependencies.
2. Read [CLI Reference](cli/reference.md) to understand `book-translator-pipeline` and subcommands.
3. Read [Models & Providers Configuration](configuration/models-and-providers.md) to configure cloud API keys or run completely offline with local LM Studio.
4. Try running the interactive terminal wizard: `book-translator wizard` or `python scripts/run_wizard.py`.

### 2. I Want to Understand How the System Works (Architect / Engineer)
1. Read [Architecture Overview](architecture/overview.md) for the core philosophy (verifiable agents, deterministic funnels, byte-for-byte reversibility).
2. Read [LangGraph Agents Deep Dive](architecture/langgraph-agents.md) to inspect how the 4 specialized agents make decisions without reading book prose.
3. Read [Pipeline Stages 1 to 6](pipeline/stages-1-to-6.md) and [Pipeline Stages 7 to 12](pipeline/stages-7-to-12.md) for stage-by-stage technical specifications.
4. Read [Data Layout & Contracts](data-layout-and-contracts.md) for directory conventions and manifest schemas.

### 3. I Want to Modify Code or Add Features (Developer / Contributor)
1. Read [Contributing & Testing](development/contributing-and-testing.md) for environment setup, offline test harness, snapshot testing, and linters.
2. Review the public API surface in [book_translator/__init__.py](../book_translator/__init__.py).
3. Review the pipeline orchestrator in [book_translator/pipeline/driver.py](../book_translator/pipeline/driver.py).

---

## Quick Reference: Pipeline Stages at a Glance

| Stage | Name | Technology / Engine | Question Answered | Output Directory / File |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Convert** | `docling` (text-only, OCR off) | *What does the EPUB/PDF/TXT say as Markdown, and where are its pictures?* | `01-markdown/` + `images/` |
| **2** | **Split** | LangGraph Agent + Verbatim Slicer | *Where does each chapter start? (Never reads prose)* | `02-chapters/` |
| **3** | **Preprocess** | LangGraph Agent + Provenance Validator | *What in each chapter is conversion damage or page numbers?* | `03-preprocessed/` |
| **4** | **Dictionary & Glossary** | CLASSLA/Stanza NER + Structured LLM | *Which proper nouns need consistent translation and how must they be rendered?* | `04-reduced/`, `04-reduced-unique/` & `04-glossary/` |
| **5** | **Chunk** | `chonkie` Recursive Chunker | *Where can chapters be cut reversibly without splitting paragraphs?* | `05-chunks/` |
| **6** | **Chunk Glossary** | NER Tagger + `rapidfuzz` (No LLM) | *Which locked glossary terms does this chunk actually mention?* | `06-chunk-glossary/` |
| **7** | **Matter** | LangGraph Agent + Multimodal Model | *How is front/back matter localized, credited to BleBel, pruned, and its images/covers recreated? (Requires multimodal model)* | `07-matter/` + `08-translations/` |
| **8** | **Translate** | Structured LLM (DeepSeek / LM Studio) | *What does this chunk say in the target language using those terms?* | `08-translations/` + prompts |
| **9** | **Assemble** | Deterministic Reassembler | *What does each whole chapter say once all its chunks are translated?* | `09-translated-chapters/` |
| **10**| **Postprocess** | LangGraph Agent + Openings Audit | *Are translated chapter titles, headings, and TOC aligned and clean?* | `10-postprocessed-chapters/` |
| **11**| **Bind** | Deterministic Concatenator | *What does the whole translated book look like as one Markdown file?* | `11-translated-book.md` |
| **12**| **EPUB** | `markdown-it-py` + EPUB3 Builder | *What does the book look like in an e-reader (images, styles, nav)?* | `12-translated-book.epub` |
