# Development, Testing & Contribution Guide

This guide covers setting up a local development environment, running the comprehensive test suite, executing static analysis, and contributing code to `book-translator`.

---

## 1. Development Environment Setup

`book-translator` requires Python 3.11, 3.12, or 3.13 (**Python 3.14+ is not supported** due to PyTorch/Stanza/protobuf C-API constraints).

```bash
# Clone the repository
git clone git@github.com:your-org/BleBelTranslateBook.git
cd BleBelTranslateBook

# Create a virtual environment (named .venv_book_translator by convention)
# Explicitly use Python 3.13, 3.12, or 3.11
python3.13 -m venv .venv_book_translator
source .venv_book_translator/bin/activate

# Install all dependencies and local package in one command
pip install --no-deps -r requirements.txt -e .
```

All required runtime and development dependencies (testing, linting, typing, conversion, dictionary NER, chunking, and EPUB packaging) are pinned and included in `requirements.txt`.

---

## 2. Test Suite Architecture

The repository contains over 800 automated tests covering unit logic, integration pipelines, and contract invariants.

The test suite is designed to run **completely offline without calling any paid or cloud LLM APIs**:
* LLM calls are mocked or run via deterministic test harnesses.
* NER operations use lightweight fixtures or CPU test mocks.
* Slicing, chunking, fuzzy matching, and EPUB packaging are tested with real files.

### Running Tests

```bash
# Run the entire offline test suite
pytest

# Run tests with short progress output
pytest -q

# Run a specific test module
pytest tests/test_stage_graph.py
pytest tests/test_chunking.py
pytest tests/test_epub.py
```

### Snapshot Testing (`tests/test_snapshots.py`)
CLI help outputs and the public package surface (`book_translator.__all__`) are frozen as snapshots in `tests/snapshots/`. If you add a new CLI command, option, or exported symbol, tests will alert you to the contract change.

To deliberately regenerate snapshots after an intentional change:
```bash
UPDATE_SNAPSHOTS=1 pytest tests/test_snapshots.py
```

### Path & Documentation Contract Testing (`tests/test_docs_paths.py`)
All backticked source paths and relative links in documentation are asserted against real filesystem paths. Any broken link or renamed module mentioned in docs will fail test validation.

---

## 3. Code Quality Standards

### Linting & Formatting with Ruff
The repository uses **Ruff** for linting and code formatting:
* Line length limit: 100 characters.
* Target Python version: 3.11.

```bash
# Run Ruff lint checks
ruff check .

# Run Ruff with automatic fixes
ruff check . --fix

# Format code
ruff format .
```

### Static Type Checking with Mypy
Type annotations are enforced across `book_translator`:

```bash
mypy book_translator
```

---

## 4. Packaging and Wheel Verification

Packaging configuration is defined in [pyproject.toml](../../pyproject.toml) and checked by `tests/test_packaging.py`.

To build the source distribution and wheel:
```bash
python -m build
```
Verify that the generated wheel contains:
1. Console script entry points (`book-translator` and `book-translator-pipeline`).
2. Package data: `book_translator/stages/epub/assets/style.css`.
3. Proper dependencies and metadata.
