"""Offline tests for the chapter-opening audit (the final phase of stages 2 and 9).

The contract the finished book depends on:

* a stray number under the title (``# Poglavlje 2`` then ``2``) is found and
  removed, with the blank line collapsed in stage 9 and kept as a slot in stage 2;
* a duplicated title line and a leftover page number are removed; a numbered
  section heading (``## 1``) is not;
* a title that is only a number, a title that is only a label and a chapter that
  opens on prose are reported, never deleted;
* a first-line bare-number heading (``## 2``) is the chapter's *title* for stage
  9, not a page number, so the plan sets it instead of inserting a second heading;
* both stage runners run the audit as their final phase, over the folder they
  just wrote;
* the tools return bounded openings, never a paragraph.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from book_translator.stages.agent.tools.openings import audit_book_openings, audit_chapter_opening
from book_translator.stages.openings import (
    BARE_NUMBER_TITLE,
    DUPLICATE_TITLE,
    EMPTY_CHAPTER,
    MISSING_TITLE,
    STRAY_NUMBER_AFTER_TITLE,
    apply_opening_fixes,
    audit_openings,
    inspect_opening,
    inspect_opening_lines,
)
from book_translator.stages.postprocess.inspect import find_title_candidate, scan_chapter_structure
from book_translator.stages.postprocess.plan import apply_chapter_plan, text_of
from book_translator.stages.postprocess_agent import run_postprocess
from book_translator.stages.preprocess_agent import runner as preprocess_runner
from langchain_core.messages import AIMessage


def _chapter(folder: Path, name: str, body: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(body, encoding="utf-8")
    return path


def _kinds(audit) -> list[str]:
    return [issue.kind for issue in audit.issues]


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
def test_a_stray_number_under_the_title_is_found_and_fixable():
    audit = inspect_opening_lines(
        ["# Poglavlje 2", "", "2", "", "Mila laktova dalje."],
        filename="05-chapter-2.md",
    )

    assert audit.title == "Poglavlje 2"
    assert audit.title_line == 1
    assert audit.title_number == 2
    assert _kinds(audit) == [STRAY_NUMBER_AFTER_TITLE]
    issue = audit.issues[0]
    assert issue.line_number == 3
    assert issue.fix == "remove"
    assert audit.fixes() == [
        {
            "line_number": 3,
            "action": "remove",
            "kind": STRAY_NUMBER_AFTER_TITLE,
            "text": "2",
        }
    ]


def test_the_chapter_number_comes_from_the_filename_when_the_title_has_none():
    audit = inspect_opening_lines(
        ["# CHAPTER TWO", "", "2", "", "Text."],
        filename="02-chapter-2.md",
    )

    assert audit.chapter_number == 2
    assert _kinds(audit) == [STRAY_NUMBER_AFTER_TITLE]


def test_a_duplicated_title_line_is_removed():
    audit = inspect_opening_lines(
        ["# Poglavlje 2", "", "Poglavlje 2", "", "Text."],
        filename="05-chapter-2.md",
    )

    assert _kinds(audit) == [DUPLICATE_TITLE]
    assert audit.fixes()[0]["line_number"] == 3


def test_a_duplicate_title_heading_lower_in_the_opening_is_removed():
    audit = inspect_opening_lines(
        ["# Poglavlje 66", "", "Proza.", "", "# Poglavlje 66", "", "Više proze."],
        filename="69-chapter-66.md",
    )

    assert _kinds(audit) == [DUPLICATE_TITLE]
    assert audit.fixes()[0]["line_number"] == 5


def test_a_bare_number_title_is_reported_and_not_deleted():
    audit = inspect_opening_lines(["## 2", "", "Mila laktova dalje."], filename="05-chapter-2.md")

    assert audit.title == "2"
    assert audit.title_is_heading is True
    assert audit.title_source == "heading"
    assert _kinds(audit) == [BARE_NUMBER_TITLE]
    assert audit.fixes() == []


def test_a_plain_number_that_is_the_chapter_title_is_reported():
    audit = inspect_opening_lines(["2", "", "Mila laktova dalje."], filename="05-chapter-2.md")

    assert audit.title == "2"
    assert audit.title_line == 1
    assert _kinds(audit) == [BARE_NUMBER_TITLE]


def test_a_label_only_title_is_reported():
    audit = inspect_opening_lines(["Poglavlje", "", "Mila laktova dalje."], filename="05-x.md")

    assert _kinds(audit) == ["label_only_title"]
    assert audit.fixes() == []


def test_a_numbered_section_heading_is_not_a_stray_number():
    audit = inspect_opening_lines(
        ["# Poglavlje 2", "", "## 1", "", "Mila laktova dalje."],
        filename="05-chapter-2.md",
    )

    assert audit.issues == []


def test_a_heading_that_repeats_the_chapter_number_is_a_stray_number():
    audit = inspect_opening_lines(
        ["# Poglavlje 2", "", "## 2", "", "Mila laktova dalje."],
        filename="05-chapter-2.md",
    )

    assert _kinds(audit) == [STRAY_NUMBER_AFTER_TITLE]
    assert audit.fixes()[0]["line_number"] == 3


def test_a_chapter_that_opens_on_prose_is_reported():
    audit = inspect_opening_lines(["Mila laktova dalje, rečenica."], filename="05-chapter-2.md")

    assert _kinds(audit) == [MISSING_TITLE]


def test_an_empty_chapter_is_reported():
    audit = inspect_opening_lines(["", "", ""], filename="05-chapter-2.md")

    assert _kinds(audit) == [EMPTY_CHAPTER]


def test_the_front_matter_may_open_on_prose():
    audit = inspect_opening_lines(
        ["In this latest installment of the series..."], filename="00-front-matter.md"
    )

    assert audit.issues == []


def test_the_opening_is_bounded():
    lines = ["# Poglavlje 2", ""] + [f"line {index}" for index in range(30)]

    audit = inspect_opening_lines(lines, filename="02-chapter-2.md")

    assert len(audit.lines) <= 12
    assert all(len(record["text"]) <= 160 for record in audit.lines)


# --------------------------------------------------------------------------- #
# repair
# --------------------------------------------------------------------------- #
def test_the_repair_collapses_the_blank_around_the_removed_line():
    repair = apply_opening_fixes(
        ["# Poglavlje 2", "", "2", "", "Mila."],
        [{"line_number": 3, "action": "remove", "kind": STRAY_NUMBER_AFTER_TITLE}],
    )

    assert repair.lines == ["# Poglavlje 2", "", "Mila."]
    assert repair.applied[0]["text"] == "2"


def test_the_repair_keeps_a_blank_slot_for_stage_two():
    repair = apply_opening_fixes(
        ["# Poglavlje 2", "", "2", "", "Mila."],
        [{"line_number": 3, "action": "remove", "kind": STRAY_NUMBER_AFTER_TITLE}],
        keep_blank_slot=True,
    )

    assert repair.lines == ["# Poglavlje 2", "", "", "", "Mila."]
    assert len(repair.lines) == 5


def test_the_repair_ignores_a_non_remove_action():
    repair = apply_opening_fixes(
        ["# Poglavlje 2", "", "2"],
        [{"line_number": 3, "action": "keep"}],
    )

    assert repair.lines == ["# Poglavlje 2", "", "2"]
    assert repair.applied == []


# --------------------------------------------------------------------------- #
# the book-wide phase
# --------------------------------------------------------------------------- #
def test_the_phase_repairs_the_folder_and_a_second_pass_is_clean(tmp_path: Path):
    folder = tmp_path / "09-postprocessed-chapters"
    _chapter(folder, "05-chapter-2.md", "# Poglavlje 2\n\n2\n\nMila laktova dalje.\n")
    _chapter(folder, "06-chapter-3.md", "# Poglavlje 3\n\nNastavak.\n")

    result = audit_openings(folder)

    assert result.ok
    assert result.inspected_count == 2
    assert result.issue_count == 1
    assert result.fixed_count == 1
    assert result.remaining_count == 0
    assert result.changed_count == 1
    assert (folder / "05-chapter-2.md").read_text(encoding="utf-8") == (
        "# Poglavlje 2\n\nMila laktova dalje.\n"
    )
    assert audit_openings(folder).issue_count == 0


def test_the_phase_in_a_dry_run_writes_nothing(tmp_path: Path):
    folder = tmp_path / "09-postprocessed-chapters"
    path = _chapter(folder, "05-chapter-2.md", "# Poglavlje 2\n\n2\n\nMila.\n")
    before = path.read_bytes()

    result = audit_openings(folder, dry_run=True)

    assert result.fixed_count == 0
    assert result.fixable_count == 1
    assert result.chapters[0]["fixes"]
    assert path.read_bytes() == before


def test_the_phase_reports_a_chapter_that_cannot_be_repaired(tmp_path: Path):
    folder = tmp_path / "09-postprocessed-chapters"
    _chapter(folder, "05-chapter-2.md", "## 2\n\nMila.\n")

    result = audit_openings(folder)

    assert result.ok
    assert result.remaining_count == 1
    assert result.counts == {BARE_NUMBER_TITLE: 1}
    assert result.chapters[0]["ok"] is False


def test_the_phase_reports_a_missing_chapter_file(tmp_path: Path):
    folder = tmp_path / "clean"
    folder.mkdir()

    result = audit_openings(folder, ["05-chapter-2.md"])

    assert result.chapter_count == 1
    assert result.inspected_count == 0
    assert result.missing == ["05-chapter-2.md"]


# --------------------------------------------------------------------------- #
# tools
# --------------------------------------------------------------------------- #
def test_the_chapter_tool_returns_a_bounded_opening(tmp_path: Path):
    folder = tmp_path / "chapters"
    path = _chapter(folder, "05-chapter-2.md", "# Poglavlje 2\n\n2\n\nMila.\n")

    payload = json.loads(audit_chapter_opening.invoke({"chapter_path": str(path)}))

    assert payload["ok"] is False
    assert payload["safe_fixes"][0]["line_number"] == 3
    assert len(payload["opening"]) <= 12
    assert "OPENING ONLY" in payload["note"]


def test_the_book_tool_counts_clean_chapters_and_expands_findings(tmp_path: Path):
    folder = tmp_path / "chapters"
    _chapter(folder, "05-chapter-2.md", "# Poglavlje 2\n\n2\n\nMila.\n")
    _chapter(folder, "06-chapter-3.md", "# Poglavlje 3\n\nNastavak.\n")

    payload = json.loads(audit_book_openings.invoke({"chapters_path": str(folder)}))

    assert payload["chapter_count"] == 2
    assert payload["clean_count"] == 1
    assert payload["counts"] == {STRAY_NUMBER_AFTER_TITLE: 1}
    assert [chapter["filename"] for chapter in payload["chapters"]] == ["05-chapter-2.md"]

    full = json.loads(
        audit_book_openings.invoke({"chapters_path": str(folder), "include_clean": True})
    )
    assert len(full["chapters"]) == 2


def test_a_first_line_number_heading_is_the_title_for_stage_nine(tmp_path: Path):
    folder = tmp_path / "08-translated-chapters"
    path = _chapter(folder, "05-chapter-2.md", "## 2\n\nMila laktova dalje.\n")

    candidate = find_title_candidate(path.read_text(encoding="utf-8").splitlines(), chapter_number=2)
    structure = scan_chapter_structure(path)

    assert candidate["line_number"] == 1
    assert candidate["text"] == "2"
    assert structure.title == "2"
    assert structure.title_line == 1
    assert structure.title_is_heading is True


def test_a_page_number_heading_above_the_real_title_is_still_skipped(tmp_path: Path):
    lines = ["## 18", "", "## Poglavlje 1", "", "Mila."]

    candidate = find_title_candidate(lines, chapter_number=1)

    assert candidate["line_number"] == 3
    assert candidate["text"] == "Poglavlje 1"


def test_setting_the_bare_number_title_leaves_no_stray_number(tmp_path: Path):
    folder = tmp_path / "08-translated-chapters"
    path = _chapter(folder, "05-chapter-2.md", "## 2\n\n2\n\nMila laktova dalje.\n")
    structure = scan_chapter_structure(path)

    repaired = apply_chapter_plan(
        path,
        {"title": {"action": "set", "line_number": 1, "text": "Poglavlje 2", "level": 1}},
        structure=structure,
    )
    path.write_text(text_of(repaired), encoding="utf-8")

    # The plan replaces the bare-number heading; the audit then removes the copy.
    assert text_of(repaired).startswith("# Poglavlje 2\n\n2\n")
    assert audit_openings(folder).fixed_count == 1
    assert (folder / "05-chapter-2.md").read_text(encoding="utf-8").startswith(
        "# Poglavlje 2\n\nMila"
    )


# --------------------------------------------------------------------------- #
# the stage runners run it as their final phase
# --------------------------------------------------------------------------- #
class _ScriptedModel:
    """Minimal chat-model double: replays canned AIMessages."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.seen = 0

    def invoke(self, messages, *_, **__):
        index = min(self.seen, len(self.responses) - 1)
        self.seen += 1
        return self.responses[index]


def _tool_call(name, args, call_id="call_1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def test_a_repair_preserves_crlf_and_a_missing_terminal_newline(tmp_path: Path):
    folder = tmp_path / "chapters"
    path = _chapter(folder, "01-chapter.md", "## CHAPTER 1\r\n\r\n2\r\n\r\nText.")
    before = path.read_bytes()
    assert before.endswith(b"Text.")  # no terminator to begin with

    result = audit_openings(folder, keep_blank_slot=True)

    assert result.fixed_count == 1
    after = path.read_bytes()
    assert b"\r\n" in after
    assert b"\n" not in after.replace(b"\r\n", b"")  # no bare LF was introduced
    assert after.endswith(b"Text.")  # still no terminal newline
    assert after == b"## CHAPTER 1\r\n\r\n\r\n\r\nText."


def test_stage_nine_removes_the_stray_number_under_the_title(tmp_path: Path):
    source = tmp_path / "08-translated-chapters"
    _chapter(source, "05-chapter-2.md", "## 2\n\n2\n\nMila laktova dalje.\n")
    out = tmp_path / "09-postprocessed-chapters"
    book_model = _ScriptedModel(
        [
            _tool_call(
                "report_book_plan",
                {
                    "plan": {
                        "naming": {
                            "mode": "normalize",
                            "label": "Poglavlje",
                            "include_title": False,
                            "level": 1,
                        },
                        "toc": {"action": "create", "heading": "Sadržaj"},
                    }
                },
            )
        ]
    )
    chapter_model = _ScriptedModel(
        [
            _tool_call(
                "report_chapter_postprocess",
                {
                    "chapter_path": str(source / "05-chapter-2.md"),
                    "title": {
                        "action": "set",
                        "line_number": 1,
                        "text": "Poglavlje 2",
                        "level": 1,
                    },
                    "headings": [],
                },
            )
        ]
    )

    result = run_postprocess(
        source,
        out,
        book_model=book_model,
        chapter_model=chapter_model,
        require_web_search=False,
    )

    assert result.ok, result.errors
    written = (out / "05-chapter-2.md").read_text(encoding="utf-8")
    assert written == "# Poglavlje 2\n\nMila laktova dalje.\n"
    assert result.openings["fixed_count"] == 1
    assert result.totals["opening_fixes"] == 1
    assert result.totals["opening_remaining"] == 0
    assert result.manifest_path is not None
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["openings"]["fixed_count"] == 1


def _book_plan_call():
    return _tool_call(
        "report_book_plan",
        {
            "plan": {
                "naming": {
                    "mode": "normalize",
                    "label": "Poglavlje",
                    "include_title": False,
                    "level": 1,
                },
                "toc": {"action": "create", "heading": "Sadržaj"},
            }
        },
    )


def test_stage_nine_audits_a_reused_chapter_too(tmp_path: Path):
    source = tmp_path / "08-translated-chapters"
    _chapter(source, "05-chapter-2.md", "## 2\n\nMila laktova dalje.\n")
    out = tmp_path / "09-postprocessed-chapters"
    chapter_model = _ScriptedModel(
        [
            _tool_call(
                "report_chapter_postprocess",
                {
                    "chapter_path": str(source / "05-chapter-2.md"),
                    "title": {
                        "action": "set",
                        "line_number": 1,
                        "text": "Poglavlje 2",
                        "level": 1,
                    },
                    "headings": [],
                },
            )
        ]
    )
    first = run_postprocess(
        source,
        out,
        book_model=_ScriptedModel([_book_plan_call()]),
        chapter_model=chapter_model,
        require_web_search=False,
    )
    assert first.ok, first.errors
    # Simulate a folder written by an older run: the stray number is back.
    (out / "05-chapter-2.md").write_text(
        "# Poglavlje 2\n\n2\n\nMila laktova dalje.\n", encoding="utf-8"
    )

    second = run_postprocess(
        source,
        out,
        book_model=_ScriptedModel([_book_plan_call()]),
        chapter_model=None,
        require_web_search=False,
        skip_if_present=True,
    )

    assert second.ok, second.errors
    assert second.reused_count == 1
    assert (out / "05-chapter-2.md").read_text(encoding="utf-8") == (
        "# Poglavlje 2\n\nMila laktova dalje.\n"
    )
    assert second.openings["fixed_count"] == 1


def test_a_reused_chapter_whose_title_moved_is_reprocessed(tmp_path: Path):
    source = tmp_path / "08-translated-chapters"
    _chapter(source, "05-chapter-2.md", "## 2\n\nMila laktova dalje.\n")
    out = tmp_path / "09-postprocessed-chapters"
    first = run_postprocess(
        source,
        out,
        book_model=_ScriptedModel([_book_plan_call()]),
        chapter_model=_ScriptedModel(
            [
                _tool_call(
                    "report_chapter_postprocess",
                    {
                        "chapter_path": str(source / "05-chapter-2.md"),
                        "title": {
                            "action": "set",
                            "line_number": 1,
                            "text": "Poglavlje 2",
                            "level": 1,
                        },
                        "headings": [],
                    },
                )
            ]
        ),
        require_web_search=False,
    )
    assert first.ok, first.errors
    # A folder written before the title-detection fix: a bare number at the top
    # (read as the title) with the real title further down, behind prose.
    (out / "05-chapter-2.md").write_text(
        "2\n\nMila laktova dalje.\n\n# Poglavlje 2\n\nNastavak.\n", encoding="utf-8"
    )
    moved = scan_chapter_structure(out / "05-chapter-2.md")
    assert moved.title == "2"
    assert moved.title_line == 1

    second = run_postprocess(
        source,
        out,
        book_model=_ScriptedModel([_book_plan_call()]),
        chapter_model=_ScriptedModel(
            [
                _tool_call(
                    "report_chapter_postprocess",
                    {
                        "chapter_path": str(source / "05-chapter-2.md"),
                        "title": {
                            "action": "set",
                            "line_number": 1,
                            "text": "Poglavlje 2",
                            "level": 1,
                        },
                        "headings": [],
                    },
                )
            ]
        ),
        require_web_search=False,
        skip_if_present=True,
    )

    assert second.ok, second.errors
    assert second.reused_count == 0
    assert (out / "05-chapter-2.md").read_text(encoding="utf-8") == (
        "# Poglavlje 2\n\nMila laktova dalje.\n"
    )


class _WritingPreprocessApp:
    """A graph double whose ``apply`` step writes the chapter unchanged."""

    def invoke(self, state, config=None):
        source = Path(state["chapter_path"])
        target = Path(state["output_dir"]) / source.name
        text = source.read_text(encoding="utf-8")
        target.write_text(text, encoding="utf-8")
        lines = text.splitlines()
        return {
            "applied": {
                "ok": True,
                "output_path": str(target),
                "source_lines": len(lines),
                "output_lines": len(lines),
                "removed_artefacts": [],
            },
            "verification": {"ok": True},
            "validation": {"ok": True, "text_fixes": []},
            "operations": [],
            "attempts": 1,
            "usage": {},
            "log": [],
        }


def test_stage_two_keeps_the_line_count_when_it_removes_an_opening_line(
    tmp_path: Path, monkeypatch
):
    chapters = tmp_path / "01-chapters"
    _chapter(chapters, "02-chapter-1.md", "## CHAPTER 1\n\n2\n\nText.\n")
    out = tmp_path / "02-preprocessed"
    monkeypatch.setattr(
        preprocess_runner,
        "build_preprocess_graph",
        lambda *args, **kwargs: _WritingPreprocessApp(),
    )

    result = preprocess_runner.run_preprocess(
        str(chapters),
        str(out),
        write_manifest=False,
    )

    assert result.ok, result.errors
    written = (out / "02-chapter-1.md").read_text(encoding="utf-8")
    assert written == "## CHAPTER 1\n\n\n\nText.\n"
    assert result.openings["fixed_count"] == 1
    assert result.chapters[0]["output_lines"] == 5
    assert any("openings: removed" in line for line in result.chapters[0]["log"])


def test_both_stage_runners_skip_the_phase_on_a_dry_run(tmp_path: Path):
    chapters = tmp_path / "01-chapters"
    _chapter(chapters, "02-chapter-1.md", "## CHAPTER 1\n\nText.\n")

    result = preprocess_runner.run_preprocess(
        str(chapters),
        str(tmp_path / "clean"),
        dry_run=True,
        write_manifest=False,
    )

    assert result.dry_run is True
    # A dry run never builds the graph, so the opening audit has nothing to read.
    assert result.openings == {}
    assert not (tmp_path / "clean").exists()


def test_the_phase_audit_is_clean_on_a_normal_opening(tmp_path: Path):
    folder = tmp_path / "chapters"
    _chapter(folder, "01-chapter-1.md", "## CHAPTER 1\n\nFirst sentence.\n")
    _chapter(folder, "02-chapter-2.md", "# Poglavlje 2\n\nSecond sentence.\n")

    result = audit_openings(folder)

    assert result.ok
    assert result.issue_count == 0
    assert result.clean_count == 2
    assert result.chapters == []


@pytest.mark.parametrize(
    "body",
    [
        "# Poglavlje 2\n\n2\n\nMila.\n",
        "# Poglavlje 2\n\nPoglavlje 2\n\nMila.\n",
    ],
)
def test_inspect_opening_reads_the_file(tmp_path: Path, body: str):
    folder = tmp_path / "chapters"
    path = _chapter(folder, "05-chapter-2.md", body)

    audit = inspect_opening(path)

    assert audit.filename == "05-chapter-2.md"
    assert audit.issues and audit.issues[0].fixable
