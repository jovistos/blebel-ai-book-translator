# Command-Line Interface (CLI) Reference

The package registers two primary console script entry points:
* **`book-translator-pipeline`**: Orchestrates the multi-stage end-to-end pipeline (or partial slices of it).
* **`book-translator`**: Provides direct access to individual stage subcommands, inspections, audits, and interactive wizards.

---

## 1. Orchestrated Pipeline CLI: `book-translator-pipeline`

Run the entire pipeline or any subset with a single command. Equivalent to `python scripts/run_pipeline.py`.

```bash
book-translator-pipeline [BOOK] [OPTIONS]
```

### Essential Options

| Option | Description |
| :--- | :--- |
| `BOOK` | Path or filename of the book. If only one file exists in `data/input/`, it is selected automatically. |
| `--stop-after <stage>` | Stop pipeline execution after the named stage (e.g. `--stop-after chunk`). |
| `--start-at <stage>` | Skip earlier stages and begin execution at the named stage (e.g. `--start-at matter`). |
| `--first-chapters <N>` | Restrict downstream stages (preprocessing, dictionary, translation) to the first $N$ chapters. |
| `--with-translate` | Include Stage 8 (body translation) in the run. (By default, pipeline stops after preparation). |
| `--with-postprocess` | Include Stage 10 (heading postprocessing) in the run. |
| `--skip-if-present` | Skip recomputing chapters, chunks, or translations that already exist and match on disk. |
| `--dry-run` | Validate and plan all operations without writing files or spending LLM tokens. |
| `--translate-provider <name>` | Shortcut to override the translation backend (`deepseek` or `lmstudio`). |
| `--cpu` | Force CPU-only mode for NER backends (CLASSLA/Stanza), bypassing GPU checks. |
| `--json` | Emit execution report as a single structured JSON object on stdout (no progress narration). |
| `--quiet` | Suppress stage banner narration and print only warnings and errors. |
| `--llm <stage>.<field>=<val>` | Override any model parameter for a specific stage (e.g. `--llm split.model=gpt-4o`). |

---

## 2. Common Pipeline Workflows

### A. Initial Structural Preparation (Stages 0 to 6)
Prepare the book, clean headers, extract terminology, and cut into translation chunks:
```bash
book-translator-pipeline "MyBook.epub" --cpu
```

### B. Translate First 3 Chapters for Quality Verification
Test translation quality on a small sample before running the entire book:
```bash
book-translator-pipeline "MyBook.md" \
  --first-chapters 3 \
  --with-translate \
  --with-postprocess \
  --skip-if-present \
  --cpu
```

### C. Run Translation Completely Offline with Local LM Studio
Perform body translation using whatever model is currently loaded in LM Studio:
```bash
book-translator-pipeline "MyBook.md" \
  --with-translate \
  --translate-provider lmstudio \
  --skip-if-present \
  --cpu
```

### D. Compile Finished Chapters to EPUB
If chapters are already translated and postprocessed, package the book directly:
```bash
book-translator-pipeline "MyBook.md" --start-at epub
```

---

## 3. Modular Stage Subcommands: `book-translator`

Each stage can be executed independently using `book-translator <command> [OPTIONS]`:

| Command | Stage | Description |
| :--- | :---: | :--- |
| **`convert`** | 1 | Convert EPUB, PDF, or TXT into Markdown in `01-markdown/` and extract pictures. |
| **`inspect`** | Pre | Inspect source Markdown structure, line count, candidate headers (no LLM, no writes). |
| **`split`** | 2 | Execute the Split Agent to cut source Markdown into `02-chapters/`. |
| **`scan`** | 3 (Pre) | Run pre-flight damage scan across chapters (page numbers, split headings). |
| **`preprocess`**| 3 | Clean conversion damage and normalize headers in `03-preprocessed/`. |
| **`dictionary`**| 4 | Reduce chapters to entity-bearing sentences (`04-reduced/` and `04-reduced-unique/`). |
| **`glossary`** | 4 | Build the canonical book dictionary in `04-glossary/` using structured LLM calls. |
| **`chunk`** | 5 | Cut chapters into reversible, paragraph-preserving chunks in `05-chunks/`. |
| **`chunk-glossary`** | 6 | Tag chunks with NER and fuzzy-match terms against the glossary in `06-chunk-glossary/`. |
| **`matter`** | 7 | Localize front/back matter (web-search title, BleBel credit) and recreate matter images multimodally (requires multimodal model) in `07-matter/` and `08-translations/`. |
| **`translate`** | 8 | Translate body chunks using their specific glossary in `08-translations/`. |
| **`concatenate`**| 9 | Reassemble translated chunks back into whole chapters in `09-translated-chapters/`. |
| **`postprocess`**| 10 | Harmonize translated chapter titles and table of contents in `10-postprocessed-chapters/`. |
| **`openings`** | Gate | Audit and repair chapter opening lines (strip stray numbers, duplicate titles). Final phase of Stages 3 and 10. |
| **`bind`** | 11 | Concatenate postprocessed chapters into `11-translated-book.md`. |
| **`epub`** | 12 | Package postprocessed chapters and images into `12-translated-book.epub`. |
| **`wizard`** | Interactive | Launch the interactive terminal setup and execution wizard. |
| **`tools`** | Debug | Print JSON schemas of tools registered for each stage agent. |

---

## 4. Helper Scripts (`scripts/`)

* **`scripts/run_pipeline.py`**: Direct executable wrapper for `book-translator-pipeline`.
* **`scripts/run_wizard.py`**: Interactive wizard prompting the user to select books, stages, providers, and options.
* **`scripts/check_layout.py`**: Verifies that output directory structure and file naming conform to repository contracts.
* **`scripts/audit_preprocess_tokens.py`**: Analyzes token usage and provenance modifications across Stage 3 preprocessed files.
* **`scripts/replay_preprocess.py`**: Replays an existing `_preprocess_manifest.json` against source files without calling any LLM.
