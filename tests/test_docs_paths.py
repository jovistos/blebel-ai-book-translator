"""Documentation paths are part of the contract too.

The README is the manual for a package whose source layout changed several times.
A link or a backticked source path that no longer resolves sends a reader to a
file that does not exist, so it is checked like any other artifact.
"""

from __future__ import annotations

import re

from _fixtures import ROOT

README = ROOT / "README-book-splitter.md"
_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
_BACKTICKED_PATH = re.compile(r"`([^`\n]*book_splitter/[^`\n]+)`")

#: Generated examples and placeholders that cannot exist in a clean checkout.
_SKIP_PREFIXES = (
    "data/output/",
    "../00-markdown/images/",
)


def _local_targets() -> list[str]:
    targets: list[str] = []
    for raw in _LINK.findall(README.read_text(encoding="utf-8")):
        target = raw.split("#", 1)[0].strip()
        if not target or target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        if target.startswith(_SKIP_PREFIXES):
            continue
        if "<" in target or ">" in target:
            continue
        if "…" in target:
            continue
        targets.append(target)
    return targets


def test_every_relative_readme_link_resolves() -> None:
    missing = [target for target in _local_targets() if not (ROOT / target).exists()]

    assert not missing, "README links point at missing files: " + ", ".join(missing)


def test_every_backticked_source_path_exists() -> None:
    text = README.read_text(encoding="utf-8")
    missing = [
        token
        for token in set(_BACKTICKED_PATH.findall(text))
        if not (ROOT / token).exists()
    ]

    assert not missing, "README mentions missing source paths: " + ", ".join(missing)


def test_the_package_layout_lists_the_real_module_directories() -> None:
    text = (ROOT / "book_splitter" / "__init__.py").read_text(encoding="utf-8")
    layout = text.split("Package layout::", 1)[1]

    # Output folders may be numbered; source packages are not. `04-glossary/` and
    # `06b-matter/` were the two stale entries.
    assert "04-glossary/" not in layout
    assert "06b-matter/" not in layout
    assert "glossary/" in layout
    assert "matter/" in layout
