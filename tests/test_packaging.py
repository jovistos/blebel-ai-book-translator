"""The packaged project's metadata is a contract, like the CLI help snapshot.

These tests do not build a wheel (CI does that); they make the source metadata
consistent so a wheel build cannot silently lose the extras, the console scripts
or the one non-Python asset the EPUB stage needs.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import book_translator
import pytest

from _fixtures import ROOT

PYPROJECT = ROOT / "pyproject.toml"
REQUIREMENTS_TXT = ROOT / "requirements.txt"


def _metadata() -> dict[str, Any]:
    import tomllib

    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _requirement_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_the_expected_extras_exist() -> None:
    extras = _metadata()["project"]["optional-dependencies"]

    assert set(extras) == {"convert", "dictionary", "chunk", "epub", "dev"}
    # The dev extra must be enough to run the offline suite, which exercises the
    # chunking and EPUB stages for real.
    dev_names = {item.split(">=")[0].split("==")[0] for item in extras["dev"]}
    assert {"chonkie", "tiktoken", "rapidfuzz", "markdown-it-py"} <= dev_names


def test_console_scripts_resolve_to_callables() -> None:
    scripts = _metadata()["project"]["scripts"]

    assert scripts == {
        "book-translator": "book_translator.cli:main",
        "book-translator-pipeline": "book_translator.pipeline.parser:main",
    }
    for target in scripts.values():
        module_name, attribute = target.split(":", 1)
        module = importlib.import_module(module_name)
        assert callable(getattr(module, attribute))


def test_the_version_has_one_source() -> None:
    project = _metadata()["project"]
    dynamic = _metadata()["tool"]["setuptools"]["dynamic"]

    assert project["dynamic"] == ["version"]
    assert dynamic["version"] == {"attr": "book_translator.__version__"}
    assert isinstance(book_translator.__version__, str)
    assert book_translator.__version__.count(".") >= 1


def test_the_epub_stylesheet_is_declared_as_package_data() -> None:
    data = _metadata()["tool"]["setuptools"]["package-data"]

    assert data["book_translator.stages.epub"] == ["assets/*.css"]
    assert (ROOT / "book_translator" / "stages" / "epub" / "assets" / "style.css").is_file()


def test_requirements_file_exists_and_contains_pinned_packages() -> None:
    assert REQUIREMENTS_TXT.is_file(), "requirements.txt must exist"
    lines = _requirement_lines(REQUIREMENTS_TXT)
    packages = [line for line in lines if not line.startswith("--")]
    assert len(packages) > 50
    package_names = {item.split("==")[0].lower() for item in packages}
    assert {"langchain-core", "langgraph", "pydantic", "torch"} <= package_names


def test_the_sdist_manifest_keeps_the_documented_files() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    for name in ("README-book-translator.md", "book_translator.toml", "tests", "scripts"):
        assert name in manifest
