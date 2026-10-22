"""Shared fixtures and ground truth for the book-splitter tests.

Kept separate from ``conftest.py`` so test modules can import the constants
explicitly. No network, no API key, no real LLM.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from book_splitter import paths

ROOT = Path(__file__).resolve().parent.parent
#: The real sample book (PDF-converted, mangled headings). Skipped when absent.
SAMPLE = paths.default_input_dir(ROOT) / "Aleksandar_Loven_-_Ljubav_seks_i_vase_srce.md"
#: A small hand-written fixture with no Markdown markup at all.
PLAIN = Path(__file__).resolve().parent / "fixtures" / "plain_book_no_headers.md"

#: The canonical chapter starts of the sample book: the PDF conversion mangled
#: the headings (`GIAVA 5`, titles on the line below `GLAVA 2`, page numbers as
#: `## 18`), so these line numbers are the ground truth the agent rediscovers.
SAMPLE_SPLITS = [46, 50, 74, 78, 136, 170, 215, 245, 259, 263, 303, 367, 383, 434, 517]

SAMPLE_TOC_TITLES = [
    "Predgovor", "Uvod", "PRVI DEO: ISPUNJENJE LJUBAVI", "1. Ljubav je u srcu života",
    "2. Seks i srce", "3. U srcu smo još uvek deca",
    '4. Gubitak ljubavi i gubitak nade "Ne mogu da živim bez tebe"',
    "5. Strah od ljubavi", "DRUGI DEO: SLOMLJENO SRCE I BOLEST SRCA", "6. Ljubav, stres i srce",
    "7. Sr čani udar", "8. Naprasna smrt", "9. Volja za životom i ielja za smrću",
    "10. Zdravo srce - Osoba koja voli", "Beleške",
]

requires_sample = pytest.mark.skipif(not SAMPLE.is_file(), reason="sample book not present")


@pytest.fixture
def sample() -> Path:
    """The full sample book, or a skip when it is not checked out."""
    if not SAMPLE.is_file():
        pytest.skip("sample book not present")
    return SAMPLE


@pytest.fixture
def plain() -> Path:
    """The small markup-free fixture book."""
    return PLAIN


@pytest.fixture
def plan():
    """Build a chapter map from ``(line_number, filename)`` pairs."""

    def _plan(pairs):
        return [{"line_number": ln, "filename": fn} for ln, fn in pairs]

    return _plan
