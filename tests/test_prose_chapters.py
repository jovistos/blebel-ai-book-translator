"""Chapter starts whose heading the conversion dropped, and ``--map`` replay.

Two behaviours are pinned here:

* a ``role: "prose-chapter"`` split point is accepted only when it opens a
  paragraph and carries a stated justification — the guardrail that keeps the
  role from becoming a way to cut prose anywhere;
* a map supplied from a file (the ``--map`` path) goes through exactly the same
  validation as an agent's plan, slices verbatim, verifies byte for byte and
  needs no provider, no key and no network.
"""

from __future__ import annotations

import json

import pytest
from book_translator.cli import main as cli_main
from book_translator.core.errors import ToolError
from book_translator.documents.io import load_lines
from book_translator.stages.agent import run_split_from_map
from book_translator.stages.analysis.prose import (
    PROSE_CHAPTER_ROLE,
    is_paragraph_start,
    is_prose_chapter_role,
    prose_start_problem,
)
from book_translator.stages.analysis.toc import detect_toc_block
from book_translator.stages.slicing.mapfile import load_split_map
from book_translator.stages.slicing.plan import plan_files
from book_translator.stages.slicing.validation import validate_split_map

#: A five-chapter map of the plain fixture; the third start is prose (line 20 is
#: the first line of chapter 1's body, sitting two lines under its own title).
PLAIN_MAP = [
    {"line_number": 12, "filename": "01-predgovor.md", "title": "Predgovor"},
    {"line_number": 18, "filename": "02-dolazak.md", "title": "1. Dolazak"},
    {
        "line_number": 20,
        "filename": "03-dolazak-nastavak.md",
        "title": "1. Dolazak (nastavak)",
        "role": PROSE_CHAPTER_ROLE,
        "evidence": "printed TOC lists a section starting here; no heading survives in the file",
    },
    {"line_number": 28, "filename": "04-oluja.md", "title": "2. Oluja"},
    {"line_number": 44, "filename": "05-biljeske.md", "title": "Bilješke"},
]


def _without_evidence(entry: dict) -> dict:
    return {k: v for k, v in entry.items() if k != "evidence"}


# --------------------------------------------------------------------------- #
# shape predicates
# --------------------------------------------------------------------------- #
def test_role_spellings_are_recognised():
    assert is_prose_chapter_role("prose-chapter")
    assert is_prose_chapter_role(" prose_chapter ")
    assert is_prose_chapter_role("Prose Chapter")
    assert not is_prose_chapter_role("back-matter")
    assert not is_prose_chapter_role(None)


def test_paragraph_start_means_nothing_above_it(plain):
    lines = load_lines(str(plain))
    assert is_paragraph_start(lines, 11)  # line 12, blank line 11 above it
    assert is_paragraph_start(lines, 0)  # line 1 opens the file
    assert not is_paragraph_start(lines, 1)  # line 2 sits under line 1
    assert not is_paragraph_start(lines, 2)  # line 3 is blank
    assert prose_start_problem(lines, 2)  # blank line cannot open a chapter


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def test_validate_accepts_a_prose_chapter_start_with_evidence(plain):
    lines = load_lines(str(plain))
    result = validate_split_map(lines, PLAIN_MAP, toc_block=detect_toc_block(lines))
    assert result["ok"], result["issues"]
    assert result["prose_chapter_count"] == 1
    assert any("prose-chapter" in w for w in result["warnings"])
    point = next(p for p in result["points"] if p["line_number"] == 20)
    assert point["role"] == PROSE_CHAPTER_ROLE
    assert "TOC" in point["evidence"]


def test_validate_rejects_a_prose_chapter_start_without_evidence(plain):
    lines = load_lines(str(plain))
    mapping = [
        _without_evidence(entry) if entry["line_number"] == 20 else entry for entry in PLAIN_MAP
    ]
    result = validate_split_map(lines, mapping, toc_block=detect_toc_block(lines))
    assert not result["ok"]
    assert any("no usable `evidence`" in issue for issue in result["issues"])


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ({"line_number": 3}, "it is blank"),  # blank line
        ({"line_number": 2}, "does not start a paragraph"),  # inside a paragraph
        ({"line_number": 16}, "only a page number"),  # a lone page number
    ],
)
def test_validate_rejects_prose_starts_that_do_not_open_a_chapter(plain, entry, expected):
    lines = load_lines(str(plain))
    mapping = [
        {"line_number": 12, "filename": "01-predgovor.md"},
        {"line_number": 18, "filename": "02-dolazak.md"},
        {**entry, "filename": "03-x.md", "role": PROSE_CHAPTER_ROLE,
         "evidence": "a claim with enough characters to count as evidence"},
        {"line_number": 28, "filename": "04-oluja.md"},
        {"line_number": 44, "filename": "05-biljeske.md"},
    ]
    result = validate_split_map(lines, mapping, toc_block=detect_toc_block(lines))
    assert not result["ok"]
    assert any(expected in issue for issue in result["issues"]), result["issues"]


def test_validate_rejects_a_prose_start_that_is_too_short(tmp_path):
    source = tmp_path / "short.md"
    source.write_text(
        "Naslov knjige\n\nDa.\n\nOvo je prava rečenica koja može da otvori poglavlje.\n",
        encoding="utf-8",
    )
    lines = load_lines(str(source))
    mapping = [
        {"line_number": 1, "filename": "01-naslov.md"},
        {"line_number": 3, "filename": "02-da.md", "role": PROSE_CHAPTER_ROLE,
         "evidence": "the published contents list has a chapter here"},
        {"line_number": 5, "filename": "03-ostatak.md"},
    ]
    result = validate_split_map(lines, mapping)
    assert not result["ok"]
    assert any("too short to open a chapter" in issue for issue in result["issues"])


def test_validate_rejects_a_prose_start_inside_the_printed_toc(plain):
    lines = load_lines(str(plain))
    mapping = [
        {"line_number": 12, "filename": "01-predgovor.md"},
        {"line_number": 18, "filename": "02-dolazak.md"},
        {"line_number": 9, "filename": "03-toc.md", "role": PROSE_CHAPTER_ROLE,
         "evidence": "this line is inside the printed contents list on purpose"},
        {"line_number": 44, "filename": "04-biljeske.md"},
    ]
    result = validate_split_map(lines, mapping, toc_block=detect_toc_block(lines))
    assert not result["ok"]
    assert any("printed table of contents" in issue for issue in result["issues"])


def test_validate_still_rejects_an_unmarked_prose_line(plain):
    """Without the role, a prose line is refused exactly as before."""
    lines = load_lines(str(plain))
    mapping = [
        {"line_number": 12, "filename": "01-predgovor.md"},
        {"line_number": 18, "filename": "02-dolazak.md"},
        {"line_number": 20, "filename": "03-prose.md"},
        {"line_number": 28, "filename": "04-oluja.md"},
        {"line_number": 44, "filename": "05-biljeske.md"},
    ]
    result = validate_split_map(lines, mapping, toc_block=detect_toc_block(lines))
    assert not result["ok"]
    assert any("does not look like a title" in issue for issue in result["issues"])


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #
def test_plan_files_treats_a_prose_chapter_as_a_chapter(plain):
    planned = plan_files(
        load_lines(str(plain)),
        [
            {"line_number": 12, "filename": "01-predgovor.md"},
            {"line_number": 20, "filename": "02-dolazak.md", "role": PROSE_CHAPTER_ROLE},
            {"line_number": 44, "filename": "03-biljeske.md", "role": "back-matter"},
        ],
    )
    kinds = {item["filename"]: item["kind"] for item in planned}
    assert kinds["02-dolazak.md"] == "chapter"
    assert kinds["03-biljeske.md"] == "back-matter"
    assert next(item["filename"] for item in planned) == "00-front-matter.md"


# --------------------------------------------------------------------------- #
# the --map path
# --------------------------------------------------------------------------- #
def test_run_split_from_map_slices_verifies_and_writes_a_trace(plain, tmp_path):
    result = run_split_from_map(str(plain), str(tmp_path), PLAIN_MAP)

    assert result.ok, result.error
    assert result.strategy == "preset_map"
    assert result.attempts == 0 and result.usage == {}
    assert result.chapter_count == 5
    assert result.verification["reassembles_byte_for_byte"]
    assert result.verification["output_lines"] == result.verification["source_lines"]

    trace = json.loads((tmp_path / "_agent_trace.json").read_text(encoding="utf-8"))
    assert trace["result"]["strategy"] == "preset_map"
    assert trace["proposed_splits"][2]["role"] == PROSE_CHAPTER_ROLE
    assert trace["validation"]["prose_chapter_count"] == 1

    blob = b"".join((tmp_path / f["filename"]).read_bytes() for f in result.files)
    assert blob == plain.read_bytes()


def test_run_split_from_map_writes_nothing_when_the_map_is_rejected(plain, tmp_path):
    result = run_split_from_map(
        str(plain),
        str(tmp_path),
        [{"line_number": 20, "filename": "01-x.md", "role": PROSE_CHAPTER_ROLE}],
    )

    assert not result.ok
    assert "did not pass validation" in (result.error or "")
    assert list(tmp_path.glob("*.md")) == []
    assert (tmp_path / "_agent_trace.json").is_file()  # the trace explains the refusal


def test_run_split_from_map_dry_run_validates_without_writing(plain, tmp_path):
    result = run_split_from_map(str(plain), str(tmp_path), PLAIN_MAP, dry_run=True)

    assert result.ok, result.error
    assert list(tmp_path.glob("*.md")) == []
    assert result.files


# --------------------------------------------------------------------------- #
# loading a map file
# --------------------------------------------------------------------------- #
def test_load_split_map_accepts_a_documented_recovered_map(tmp_path):
    payload = {
        "note": "recovered by hand",
        "map": [
            {"file": "00-front-matter.md", "start_line": 1, "kind": "front-matter"},
            {"file": "01-predgovor.md", "start_line": 12, "title": "Predgovor"},
            {
                "file": "02-dolazak.md",
                "start_line": 20,
                "kind": "prose-chapter",
                "provenance": "chapter 1 begins here; its heading was dropped",
                "confidence": "high",
            },
        ],
    }
    path = tmp_path / "map.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    mapping, notes = load_split_map(path)

    assert [entry["line_number"] for entry in mapping] == [12, 20]
    assert mapping[1]["role"] == PROSE_CHAPTER_ROLE
    assert mapping[1]["evidence"].startswith("chapter 1 begins here")
    assert any("front-matter" in note for note in notes)


def test_load_split_map_accepts_a_bare_list(tmp_path):
    path = tmp_path / "list.json"
    path.write_text(json.dumps([{"line_number": 12, "filename": "01-a.md"}]), encoding="utf-8")
    mapping, _ = load_split_map(path)
    assert mapping == [{"line_number": 12, "filename": "01-a.md"}]


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("not json at all", "not valid JSON"),
        ("{}", "no list of split points"),
        ('"a string"', "must hold a list of split points"),
        ("[]", "holds no split points"),
        ('[{"filename": "01-a.md"}]', "no integer line number"),
        ('[{"line_number": 12}]', "has no filename"),
        ('[{"line_number": 12, "filename": "01-a.md"}, "nope"]', "is not an object"),
    ],
)
def test_load_split_map_reports_unusable_files(tmp_path, content, expected):
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ToolError) as excinfo:
        load_split_map(path)
    assert expected in str(excinfo.value)


def test_load_split_map_reports_a_missing_file(tmp_path):
    with pytest.raises(ToolError) as excinfo:
        load_split_map(tmp_path / "nope.json")
    assert "not found" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# the CLI
# --------------------------------------------------------------------------- #
def test_cli_split_with_a_map_needs_no_provider(plain, tmp_path, monkeypatch, capsys):
    import book_translator.core.settings as settings_module

    def explode(*_, **__):  # pragma: no cover - only called on the agent path
        raise AssertionError("--map must not resolve provider settings")

    monkeypatch.setattr(settings_module, "resolve_settings", explode)

    source = tmp_path / "book.md"
    source.write_bytes(plain.read_bytes())
    map_path = tmp_path / "map.json"
    map_path.write_text(json.dumps(PLAIN_MAP, ensure_ascii=False), encoding="utf-8")
    out_dir = tmp_path / "01-chapters"

    code = cli_main(["split", str(source), "--map", str(map_path), "--out", str(out_dir)])
    captured = capsys.readouterr()

    assert code == 0
    assert "map=" in captured.out
    assert "byte-for-byte   : True" in captured.out
    manifest = json.loads((out_dir / "_manifest.json").read_text(encoding="utf-8"))
    assert manifest["chapter_count"] == 5
    blob = b"".join((out_dir / f["filename"]).read_bytes() for f in manifest["files"])
    assert blob == source.read_bytes()


def test_cli_split_with_a_broken_map_file_fails_cleanly(plain, tmp_path, capsys):
    map_path = tmp_path / "bad.json"
    map_path.write_text("{}", encoding="utf-8")
    code = cli_main(["split", str(plain), "--map", str(map_path), "--out", str(tmp_path / "out")])
    captured = capsys.readouterr()
    assert code == 2
    assert "map error" in captured.err
