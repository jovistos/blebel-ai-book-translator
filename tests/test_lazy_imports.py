"""The package surface is lazy: a small import must stay small.

The probe runs in a fresh interpreter because pytest itself imports most of the
dependency tree. It reports wall time, ``tracemalloc`` peak and the loaded module
count so the test fails on the two regressions that matter: importing every
LangGraph stage, or importing the OpenAI client.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, cast

import book_splitter

from _fixtures import ROOT

_PROBE = """
import json
import sys
import time
import tracemalloc

tracemalloc.start()
start = time.perf_counter()
{imports}
elapsed = time.perf_counter() - start
_, peak = tracemalloc.get_traced_memory()
heavy = sorted(
    name
    for name in sys.modules
    if name == "openai" or name.startswith(("langgraph", "langchain"))
)
print(json.dumps({{
    "elapsed": elapsed,
    "peak_mb": peak / 1_000_000,
    "module_count": len(sys.modules),
    "heavy": heavy,
}}))
"""


def _probe(imports: str) -> dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE.format(imports=imports)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return cast(dict[str, Any], json.loads(completed.stdout))


def test_importing_a_document_helper_does_not_import_the_agent_runtime() -> None:
    result = _probe("import book_splitter.documents.text")

    assert result["heavy"] == []
    assert result["module_count"] < 500
    assert result["peak_mb"] < 30.0
    assert result["elapsed"] < 0.9


def test_importing_the_package_alone_is_lazy() -> None:
    result = _probe("import book_splitter")

    assert result["heavy"] == []
    assert result["module_count"] < 200
    assert result["peak_mb"] < 20.0


def test_importing_the_cli_does_not_import_the_agent_runtime() -> None:
    result = _probe("import book_splitter.cli")

    assert result["heavy"] == []
    assert result["module_count"] < 500
    assert result["peak_mb"] < 30.0
    assert result["elapsed"] < 0.9


def test_reading_an_agent_export_loads_its_module_on_demand() -> None:
    result = _probe("from book_splitter import run_split")

    assert "langgraph" in result["heavy"]


def test_the_lazy_map_covers_every_export_exactly() -> None:
    assert set(book_splitter._LAZY_EXPORTS) == set(book_splitter.__all__)
    assert len(book_splitter.__all__) == len(set(book_splitter.__all__))


def test_dir_still_lists_the_public_names_without_importing_them() -> None:
    assert set(book_splitter.__all__) <= set(dir(book_splitter))
