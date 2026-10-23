"""End-to-end filesystem contract: input is read, output is written elsewhere.

These tests exercise the real default layout (``data/input`` -> ``data/output``)
against the shipped sample book, then restore the output folder exactly as they
found it. The real provider is never contacted: the graph runs with a scripted
model and a chapter map produced by the deterministic locator.
"""

from __future__ import annotations

import json

import pytest
from book_translator import paths
from book_translator.agent import build_graph
from book_translator.agent.runner import summarise_run
from book_translator.analysis.locate import locate_toc_titles
from book_translator.documents.io import load_lines
from book_translator.slicing.verify import verify_output_dir
from langchain_core.messages import AIMessage

from _fixtures import SAMPLE, SAMPLE_TOC_TITLES, requires_sample

#: Canonical output filenames, in split order.
CHAPTER_NAMES = [
    "01-predgovor.md", "02-uvod.md", "03-prvi-deo-ispunjenje-ljubavi.md",
    "04-glava-1-ljubav-je-u-srcu-zivota.md", "05-glava-2-seks-i-srce.md",
    "06-glava-3-u-srcu-smo-jos-uvek-deca.md", "07-glava-4-gubitak-ljubavi-i-gubitak-nade.md",
    "08-glava-5-strah-od-ljubavi.md", "09-drugi-deo-slomljeno-srce-i-bolesti-srca.md",
    "10-glava-6-ljubav-stres-i-srce.md", "11-glava-7-srcani-udar.md",
    "12-glava-8-naprasna-smrt.md", "13-glava-9-volja-za-zivotom-i-zelja-za-smrcu.md",
    "14-glava-10-zdravo-srce-osoba-koja-voli.md", "15-beleske.md",
]


class _ScriptedModel:
    """Replays canned messages; never touches a provider."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, messages, *_, **__):
        index = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[index]


def _tool_call(name, args, call_id="c1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def _canonical_plan(book) -> list[dict]:
    """The chapter map a good run produces, from the deterministic locator."""
    lines = load_lines(str(book))
    matches = locate_toc_titles(lines, SAMPLE_TOC_TITLES)["matches"]
    assert all(m["found"] for m in matches), "the locator must find every sample chapter"
    return [
        {"line_number": m["line_number"], "filename": name, "title": m["title"]}
        for m, name in zip(matches, CHAPTER_NAMES, strict=True)
    ]


@pytest.fixture
def pristine_output():
    """Snapshot the default output folder and restore it after the test."""
    out_dir = paths.resolve_output_dir(SAMPLE, None)
    saved: dict = {}
    if out_dir.is_dir():
        for path in out_dir.iterdir():
            if path.is_file():
                saved[path.name] = path.read_bytes()
    yield out_dir
    if out_dir.is_dir():
        for path in out_dir.iterdir():
            if path.is_file() and path.name not in saved:
                path.unlink()
    for name, blob in saved.items():
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / name).write_bytes(blob)


@requires_sample
def test_default_output_dir_is_derived_from_the_book_stem():
    expected = paths.default_output_dir() / SAMPLE.stem / "01-chapters"
    assert paths.resolve_output_dir(SAMPLE, None) == expected
    assert expected.parts[-3:] == ("output", SAMPLE.stem, "01-chapters")
    # The output folder is never the input folder.
    assert SAMPLE.parent != expected.parent
    assert paths.default_input_dir() not in expected.parents


@requires_sample
def test_graph_writes_only_into_the_default_output_folder(pristine_output, tmp_path):
    out_dir = pristine_output
    plan = _canonical_plan(SAMPLE)
    model = _ScriptedModel(
        [
            _tool_call("validate_split_map_tool", {"file_path": str(SAMPLE), "split_mapping": plan}, "c1"),
            _tool_call(
                "propose_split_map",
                {"split_mapping": plan, "strategy": "toc_titles", "notes": "test run"},
                "c2",
            ),
        ]
    )
    final = build_graph(model=model).invoke(
        {
            "source_path": str(SAMPLE),
            "output_dir": str(out_dir),
            "front_matter_lines": 200,
            "max_attempts": 6,
            "min_chapters": 3,
            "include_front_matter": True,
            "dry_run": False,
            "attempts": 0,
            "usage": {},
            "log": [],
            "feedback": [],
            "messages": [],
        },
        config={"recursion_limit": 40},
    )

    assert final["validation"]["ok"], final["validation"]["issues"]
    assert final["manifest"]["chapter_count"] == len(CHAPTER_NAMES)

    # Every artifact landed in the output folder...
    produced = {p.name for p in out_dir.iterdir() if p.is_file()}
    assert set(CHAPTER_NAMES) | {"00-front-matter.md", "_manifest.json"} <= produced
    # ...and nothing leaked into the input folder.
    assert SAMPLE.is_file()
    assert not list(SAMPLE.parent.glob("_manifest.json"))
    assert not list(SAMPLE.parent.glob("*.md.bak"))
    assert not list(tmp_path.glob("*.md"))


@requires_sample
def test_default_output_verifies_byte_for_byte(pristine_output):
    # Existing book folders from before the stage-numbered layout keep their old
    # ``chapters/`` folder; new runs write ``01-chapters/``.
    out_dir = pristine_output
    if not (out_dir / "_manifest.json").is_file():
        legacy = out_dir.parent / "chapters"
        if (legacy / "_manifest.json").is_file():
            out_dir = legacy
    if not (out_dir / "_manifest.json").is_file():
        pytest.skip("no split output for the sample book — run the split first")

    report = verify_output_dir(out_dir)
    assert report["ok"], report
    assert report["reassembles_byte_for_byte"]
    assert report["output_lines"] == report["source_lines"] == 640
    assert report["files_checked"] == len(CHAPTER_NAMES) + 1  # + front matter

    manifest = json.loads((out_dir / "_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_path"] == str(SAMPLE.resolve())
    assert manifest["chapter_count"] == len(CHAPTER_NAMES)
    # The manifest must point at the output folder, not at the input folder.
    assert manifest["output_dir"] == str(out_dir.resolve())
    assert all("/data/output/" in f["path"] for f in manifest["files"])


@requires_sample
def test_trace_is_written_for_a_completed_run(tmp_path):
    """`summarise_run` records the audit trail next to the chapters."""
    plan = _canonical_plan(SAMPLE)
    model = _ScriptedModel(
        [_tool_call("propose_split_map", {"split_mapping": plan, "strategy": "toc_titles"}, "c1")]
    )
    out_dir = tmp_path / "01-chapters"
    final = build_graph(model=model).invoke(
        {
            "source_path": str(SAMPLE),
            "output_dir": str(out_dir),
            "max_attempts": 4,
            "min_chapters": 3,
            "include_front_matter": True,
            "dry_run": False,
            "attempts": 0,
            "usage": {},
            "log": [],
            "feedback": [],
            "messages": [],
        },
        config={"recursion_limit": 40},
    )
    result = summarise_run(final, str(SAMPLE), out_dir, dry_run=False)

    assert result.ok
    trace = json.loads((out_dir / "_agent_trace.json").read_text(encoding="utf-8"))
    assert trace["result"]["ok"] is True
    assert trace["result"]["strategy"] == "toc_titles"
    assert trace["verification"]["reassembles_byte_for_byte"] is True
    assert trace["proposed_splits"] == final["proposed_splits"]
