"""Offline tests for stage 9 (heading postprocessing) and stage 10 (bind).

No provider and no network: the graph models are scripted doubles. The tests pin
the contract the book depends on:

* the book inventory sees every title before a naming policy is chosen;
* a title is set, inserted or kept without touching the chapter body;
* an accidental heading is demoted (markup removed, text kept);
* the TOC is created after the front matter, or updated in place;
* the model never sees chapter prose;
* the final single-file book contains the postprocessed files in order.
"""

from __future__ import annotations

import json
from pathlib import Path

from book_splitter.binding import run_bind
from book_splitter.postprocess.inspect import (
    inspect_matter_blocks,
    scan_book_structure,
    scan_chapter_structure,
)
from book_splitter.postprocess.plan import (
    apply_chapter_plan,
    text_of,
    validate_book_plan,
    validate_chapter_plan,
    verify_chapter_plan,
)
from book_splitter.postprocess.toc import (
    apply_toc,
    build_toc_entries,
    verify_toc,
)
from book_splitter.postprocess_agent import run_postprocess
from langchain_core.messages import AIMessage, HumanMessage


class ScriptedModel:
    """Minimal chat-model double: replays canned AIMessages."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.seen_messages = []

    def invoke(self, messages, *_, **__):
        self.seen_messages.append(list(messages))
        index = min(len(self.seen_messages) - 1, len(self.responses) - 1)
        return self.responses[index]


def _tool_call(name, args, call_id="call_1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def _book(title: str, body: str) -> Path:
    folder = Path(title)
    folder.mkdir()
    (folder / "01-chapter-1.md").write_text(body, encoding="utf-8")
    return folder


def test_book_inventory_lists_every_chapter_title(tmp_path: Path):
    for name, title in (
        ("01-chapter-1.md", "Poglavlje 1"),
        ("02-chapter-2.md", "Poglavlje 2"),
    ):
        (tmp_path / name).write_text(f"{title}\n\nPrva rečenica.\n", encoding="utf-8")

    book = scan_book_structure(tmp_path)

    assert book["chapter_count"] == 2
    assert book["body_chapter_count"] == 2
    assert book["consistent_titles"] is True
    assert [chapter["title"] for chapter in book["chapters"]] == [
        "Poglavlje 1",
        "Poglavlje 2",
    ]
    assert book["toc"]["exists"] is False


def test_book_inventory_reads_the_original_identity_from_the_matter_plan(tmp_path: Path):
    source = tmp_path / "08-translated-chapters"
    source.mkdir()
    (source / "01-chapter-1.md").write_text(
        "Poglavlje 1\n\nPrva rečenica.\n", encoding="utf-8"
    )
    matter = tmp_path / "06b-matter"
    matter.mkdir()
    (matter / "_matter_manifest.json").write_text(
        json.dumps(
            {
                "source_language": "en",
                "target_language": "sr",
                "book_plan": {
                    "original_title": "The Book",
                    "author": "An Author",
                    "translated_title": "Knjiga",
                },
            }
        ),
        encoding="utf-8",
    )

    book = scan_book_structure(source)

    assert book["original_title"] == "The Book"
    assert book["book_author"] == "An Author"
    assert book["translated_title"] == "Knjiga"
    assert book["source_language"] == "en"
    assert book["target_language"] == "sr"


def test_naming_policy_normalizes_mixed_titles_and_builds_toc(tmp_path: Path):
    (tmp_path / "01-chapter-1.md").write_text(
        "Poglavlje 1\n\nŠta vam je na umu?\n\nPrva rečenica.\n", encoding="utf-8"
    )
    (tmp_path / "02-chapter-2.md").write_text(
        "2. Poglavlje\n\nDruga rečenica.\n", encoding="utf-8"
    )
    book = scan_book_structure(tmp_path)
    plan = {
        "naming": {
            "mode": "normalize",
            "label": "Poglavlje",
            "include_title": True,
            "separator": ": ",
            "level": 1,
        },
        "toc": {"action": "create", "heading": "Sadržaj", "style": "numbered"},
    }

    validation = validate_book_plan(tmp_path, plan, book=book)

    assert validation["ok"], validation["issues"]
    titles = validation["canonical_titles"]
    assert titles["01-chapter-1.md"] == "Poglavlje 1: Šta vam je na umu?"
    assert titles["02-chapter-2.md"] == "Poglavlje 2"
    assert [entry["title"] for entry in validation["toc_entries"]] == [
        titles["01-chapter-1.md"],
        titles["02-chapter-2.md"],
    ]


def test_a_chapter_name_after_a_number_word_label_is_detected(tmp_path: Path):
    chapter = tmp_path / "02-chapter-2.md"
    chapter.write_text(
        "GLAVA DRUGA\n\nŠETNJA JE DOBRA ZA VAS I ZA BOCVANU\n\nPrva rečenica.\n",
        encoding="utf-8",
    )

    structure = scan_chapter_structure(chapter)

    assert structure.title == "GLAVA DRUGA"
    assert structure.subtitle == "ŠETNJA JE DOBRA ZA VAS I ZA BOCVANU"
    assert structure.subtitle_line == 3


def test_a_level_one_heading_under_a_label_is_a_subtitle_not_a_heading(tmp_path: Path):
    chapter = tmp_path / "02-chapter-2.md"
    chapter.write_text(
        "# CHAPTER TWO\n\n# WALKING IS GOOD FOR YOU\n\nProse.\n", encoding="utf-8"
    )

    structure = scan_chapter_structure(chapter)

    assert structure.title == "CHAPTER TWO"
    assert structure.subtitle == "WALKING IS GOOD FOR YOU"
    assert [heading["line_number"] for heading in structure.headings] == [1]


def test_a_plan_that_would_drop_existing_chapter_names_is_upgraded(tmp_path: Path):
    (tmp_path / "01-chapter-1.md").write_text(
        "Poglavlje 1\n\nŠta vam je na umu?\n\nPrva rečenica.\n", encoding="utf-8"
    )
    book = scan_book_structure(tmp_path)
    plan = {
        "naming": {
            "mode": "normalize",
            "label": "Poglavlje",
            "include_title": False,
            "level": 1,
        },
        "toc": {"action": "create", "heading": "Sadržaj", "style": "numbered"},
    }

    validation = validate_book_plan(tmp_path, plan, book=book)

    assert validation["ok"], validation["issues"]
    assert validation["naming"]["include_title"] is True
    assert validation["canonical_titles"]["01-chapter-1.md"] == (
        "Poglavlje 1: Šta vam je na umu?"
    )
    assert any("include_title" in warning for warning in validation["warnings"])
    assert [entry["title"] for entry in validation["toc_entries"]] == [
        "Poglavlje 1: Šta vam je na umu?"
    ]


def test_a_keep_policy_carries_a_subtitle_into_the_toc(tmp_path: Path):
    (tmp_path / "01-chapter-1.md").write_text(
        "Poglavlje 1\n\nŠta vam je na umu?\n\nPrva rečenica.\n", encoding="utf-8"
    )
    book = scan_book_structure(tmp_path)
    plan = {
        "naming": {"mode": "keep", "label": "Poglavlje", "level": 1},
        "toc": {"action": "create", "heading": "Sadržaj", "style": "numbered"},
    }

    validation = validate_book_plan(tmp_path, plan, book=book)

    assert validation["ok"], validation["issues"]
    assert validation["canonical_titles"]["01-chapter-1.md"] == (
        "Poglavlje 1: Šta vam je na umu?"
    )
    assert [entry["title"] for entry in validation["toc_entries"]] == [
        "Poglavlje 1: Šta vam je na umu?"
    ]


def test_title_is_set_and_missing_title_is_inserted(tmp_path: Path):
    existing = tmp_path / "existing.md"
    existing.write_text(
        "Poglavlje 1\n\nŠta vam je na umu?\n\nPrva rečenica.\n", encoding="utf-8"
    )
    structure = scan_chapter_structure(existing)
    plan = {
        "title": {
            "action": "set",
            "line_number": 1,
            "text": "Poglavlje 1: Šta vam je na umu?",
            "level": 1,
        },
        "headings": [],
    }
    validation = validate_chapter_plan(
        existing,
        plan,
        canonical_title="Poglavlje 1: Šta vam je na umu?",
        structure=structure,
    )
    assert validation["ok"], validation["issues"]
    applied = apply_chapter_plan(
        existing, plan, canonical_title="Poglavlje 1: Šta vam je na umu?", structure=structure
    )
    target = tmp_path / "out.md"
    target.write_text(text_of(applied), encoding="utf-8")
    verification = verify_chapter_plan(
        existing,
        plan,
        canonical_title="Poglavlje 1: Šta vam je na umu?",
        output_path=target,
        structure=structure,
    )
    assert verification["ok"], verification["problems"]
    assert target.read_text(encoding="utf-8").startswith("# Poglavlje 1: Šta vam je na umu?\n\n")
    assert "Šta vam je na umu?\n" not in target.read_text(encoding="utf-8").split("\n", 1)[1]

    missing = tmp_path / "missing.md"
    missing.write_text("Prva rečenica priče.\n\nDruga.\n", encoding="utf-8")
    missing_structure = scan_chapter_structure(missing)
    insert_plan = {
        "title": {"action": "insert", "text": "Poglavlje 2", "level": 1},
        "headings": [],
    }
    assert validate_chapter_plan(
        missing,
        insert_plan,
        canonical_title="Poglavlje 2",
        structure=missing_structure,
    )["ok"]
    inserted = apply_chapter_plan(
        missing, insert_plan, canonical_title="Poglavlje 2", structure=missing_structure
    )
    assert inserted.lines[0] == "# Poglavlje 2"
    assert inserted.lines[1] == ""


def test_accidental_heading_is_demoted_and_text_is_kept(tmp_path: Path):
    chapter = tmp_path / "01-chapter-1.md"
    chapter.write_text(
        "# Poglavlje 1\n\nUvod.\n\n"
        "## Ovo je zapravo rečenica koja je postala naslov.\n\nNastavak.\n",
        encoding="utf-8",
    )
    structure = scan_chapter_structure(chapter)
    plan = {
        "title": {"action": "keep", "line_number": 1, "text": "Poglavlje 1", "level": 1},
        "headings": [{"line_number": 5, "decision": "demote", "note": "a sentence"}],
    }
    validation = validate_chapter_plan(
        chapter, plan, canonical_title="Poglavlje 1", structure=structure
    )
    assert validation["ok"], validation["issues"]
    applied = apply_chapter_plan(
        chapter, plan, canonical_title="Poglavlje 1", structure=structure
    )
    assert "## Ovo je" not in "\n".join(applied.lines)
    assert "Ovo je zapravo rečenica koja je postala naslov." in applied.lines
    assert applied.demoted_lines == [5]


def test_toc_is_created_right_after_front_matter(tmp_path: Path):
    (tmp_path / "00-front-matter.md").write_text("# Naslov knjige\n\nAutor\n", encoding="utf-8")
    (tmp_path / "01-chapter-1.md").write_text(
        "# Poglavlje 1\n\nPrva.\n", encoding="utf-8"
    )
    (tmp_path / "02-chapter-2.md").write_text(
        "# Poglavlje 2\n\nDruga.\n", encoding="utf-8"
    )
    book = scan_book_structure(tmp_path)
    plan = {
        "naming": {"mode": "keep", "label": "Poglavlje"},
        "toc": {"action": "create", "heading": "Sadržaj", "style": "numbered"},
    }
    validation = validate_book_plan(tmp_path, plan, book=book)
    entries = build_toc_entries(book, plan, titles=validation["canonical_titles"])
    applied = apply_toc(tmp_path, plan, entries)
    verification = verify_toc(tmp_path, plan, entries)

    assert applied["ok"]
    assert verification["ok"], verification["problems"]
    assert (tmp_path / "00-table-of-contents.md").is_file()
    assert "Poglavlje 1" in (tmp_path / "00-table-of-contents.md").read_text(encoding="utf-8")


def test_existing_toc_is_updated_in_place(tmp_path: Path):
    (tmp_path / "00-front-matter.md").write_text(
        "# Naslov knjige\n\n## Sadržaj\n\n1. Stari naslov\n\nAutor\n", encoding="utf-8"
    )
    (tmp_path / "01-chapter-1.md").write_text(
        "# Poglavlje 1\n\nPrva.\n", encoding="utf-8"
    )
    book = scan_book_structure(tmp_path)
    assert book["toc"]["exists"] is True
    plan = {
        "naming": {"mode": "keep", "label": "Poglavlje"},
        "toc": {"action": "update", "heading": "Sadržaj", "style": "numbered"},
    }
    validation = validate_book_plan(tmp_path, plan, book=book)
    entries = build_toc_entries(book, plan, titles=validation["canonical_titles"])
    applied = apply_toc(tmp_path, plan, entries)

    text = (tmp_path / "00-front-matter.md").read_text(encoding="utf-8")
    assert applied["changed"]
    assert "Poglavlje 1" in text
    assert "Stari naslov" not in text
    assert verify_toc(tmp_path, plan, entries)["ok"]


def test_matter_inspection_is_bounded_and_reports_toc(tmp_path: Path):
    lines = ["# Naslov", ""] + [f"red {number}" for number in range(80)]
    (tmp_path / "00-front-matter.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (tmp_path / "01-chapter-1.md").write_text("# Poglavlje 1\n\nPrva.\n", encoding="utf-8")

    report = inspect_matter_blocks(tmp_path)

    assert report["front_matter"] is not None
    assert len(report["front_matter"]["lines"]) <= 40
    assert report["toc"]["exists"] is False


def test_postprocess_graph_writes_a_separate_folder_and_toc(tmp_path: Path):
    source = tmp_path / "08-translated-chapters"
    source.mkdir()
    (source / "01-chapter-1.md").write_text(
        "Poglavlje 1\n\nŠta vam je na umu?\n\nPrva rečenica.\n", encoding="utf-8"
    )
    out = tmp_path / "09-postprocessed-chapters"
    book_model = ScriptedModel(
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
    chapter_model = ScriptedModel(
        [
            _tool_call(
                "report_chapter_postprocess",
                {
                    "chapter_path": str(source / "01-chapter-1.md"),
                    "title": {
                        "action": "set",
                        "line_number": 1,
                        "text": "Poglavlje 1: Šta vam je na umu?",
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
    assert (out / "01-chapter-1.md").is_file()
    assert (out / "00-table-of-contents.md").is_file()
    assert not (source / "00-table-of-contents.md").exists()
    assert (source / "01-chapter-1.md").read_text(encoding="utf-8").startswith(
        "Poglavlje 1\n"
    )
    assert (out / "01-chapter-1.md").read_text(encoding="utf-8").startswith(
        "# Poglavlje 1: Šta vam je na umu?\n"
    )
    toc_text = (out / "00-table-of-contents.md").read_text(encoding="utf-8")
    assert "1. Poglavlje 1: Šta vam je na umu?" in toc_text
    assert result.toc["ok"]
    assert result.manifest_path and Path(result.manifest_path).is_file()


def test_chapter_agent_is_told_to_resubmit_on_its_last_turn(tmp_path: Path):
    """The failure seen on *We Need to Talk About Kevin*: the last turn is a dry run.

    The model had submitted a plan, the graph rejected it, and the final allowed turn
    was spent on `validate_chapter_postprocess` — which passed, but never replaced the
    plan in the state, so the chapter was written nowhere.
    """
    from book_splitter.postprocess_agent.nodes import make_chapter_agent_node

    source = tmp_path / "08-translated-chapters"
    source.mkdir()
    chapter = source / "02-chapter-2.md"
    chapter.write_text("Poglavlje 2\n\nDruga rečenica.\n", encoding="utf-8")

    model = ScriptedModel([AIMessage(content="done")])
    node = make_chapter_agent_node(model)
    state = {
        "chapter_path": str(chapter),
        "messages": [HumanMessage(content="report the chapter plan")],
        "feedback": ["Your chapter plan was rejected."],
        "attempts": 3,
        "max_attempts": 4,
        "validation": {"ok": False},
    }
    node(state)

    last_turn = "\n".join(str(m.content) for m in model.seen_messages[-1])
    assert "FINAL TURN" in last_turn
    assert "report_chapter_postprocess" in last_turn

    # An earlier turn gets the correction alone, without the final-turn pressure.
    state["attempts"] = 0
    node(state)
    earlier = "\n".join(str(m.content) for m in model.seen_messages[-1])
    assert "FINAL TURN" not in earlier
    assert "rejected" in earlier


def test_book_plan_requires_a_web_search_first(tmp_path: Path, monkeypatch):
    source = tmp_path / "08-translated-chapters"
    source.mkdir()
    (source / "01-chapter-1.md").write_text(
        "Poglavlje 1\n\nPrva rečenica.\n", encoding="utf-8"
    )
    out = tmp_path / "09-postprocessed-chapters"
    queries: list[str] = []

    def fake_search(query, max_results=5):
        queries.append(query)
        return {
            "query": query,
            "provider": "test",
            "result_count": 1,
            "results": [
                {
                    "title": "Contents",
                    "url": "https://example.com/toc",
                    "snippet": "Chapter 1",
                }
            ],
        }

    monkeypatch.setattr(
        "book_splitter.agent.tools.websearch.search_web", fake_search
    )
    plan_without_evidence = {
        "naming": {"mode": "keep", "label": "Poglavlje", "level": 1},
        "toc": {"action": "create", "heading": "Sadržaj"},
    }
    plan_with_evidence = {
        **plan_without_evidence,
        "title_evidence": "https://example.com/toc",
        "title_source": "web",
    }
    book_model = ScriptedModel(
        [
            _tool_call("report_book_plan", {"plan": plan_without_evidence}),
            _tool_call(
                "web_search",
                {"query": "Tea for the Traditionally Built table of contents"},
            ),
            _tool_call("report_book_plan", {"plan": plan_with_evidence}),
        ]
    )
    chapter_model = ScriptedModel(
        [
            _tool_call(
                "report_chapter_postprocess",
                {
                    "chapter_path": str(source / "01-chapter-1.md"),
                    "title": {
                        "action": "set",
                        "line_number": 1,
                        "text": "Poglavlje 1",
                        "level": 1,
                    },
                    "headings": [],
                },
            )
        ]
    )

    result = run_postprocess(
        source, out, book_model=book_model, chapter_model=chapter_model
    )

    assert result.ok, result.errors
    assert queries == ["Tea for the Traditionally Built table of contents"]
    assert result.web["required"] is True
    assert result.web["attempted"] is True
    assert result.web["searches"][0]["query"] == queries[0]
    assert result.web["searches"][0]["result_count"] == 1
    assert result.book_plan["title_evidence"] == "https://example.com/toc"
    # The premature submission was refused with an actionable message.
    assert any(
        "web search is mandatory" in str(message.content)
        for message in book_model.seen_messages[1]
    )


def test_book_plan_requires_web_evidence(tmp_path: Path):
    (tmp_path / "01-chapter-1.md").write_text(
        "Poglavlje 1\n\nPrva rečenica.\n", encoding="utf-8"
    )
    book = scan_book_structure(tmp_path)
    plan = {
        "naming": {"mode": "keep", "label": "Poglavlje", "level": 1},
        "toc": {"action": "create", "heading": "Sadržaj"},
    }

    missing = validate_book_plan(
        tmp_path, plan, book=book, require_web_search=True
    )

    assert missing["ok"] is False
    assert any("title_evidence" in issue for issue in missing["issues"])

    plan["title_evidence"] = "web search unavailable; used the local chapter names"
    plan["title_source"] = "local"
    accepted = validate_book_plan(
        tmp_path, plan, book=book, require_web_search=True
    )
    assert accepted["ok"], accepted["issues"]
    assert accepted["title_evidence"].startswith("web search unavailable")


def test_model_never_sees_chapter_prose(tmp_path: Path):
    source = tmp_path / "08-translated-chapters"
    source.mkdir()
    marker = "Ovo je jedinstvena rečenica tela koju model ne sme da vidi."
    opening = "\n\n".join(f"Uvodna rečenica {number}." for number in range(10))
    (source / "01-chapter-1.md").write_text(
        f"Poglavlje 1\n\n{opening}\n\n{marker}\n\nNastavak.\n", encoding="utf-8"
    )
    out = tmp_path / "post"
    book_model = ScriptedModel(
        [
            _tool_call(
                "report_book_plan",
                {
                    "plan": {
                        "naming": {"mode": "keep", "label": "Poglavlje"},
                        "toc": {"action": "create", "heading": "Sadržaj"},
                    }
                },
            )
        ]
    )
    chapter_model = ScriptedModel(
        [
            _tool_call(
                "report_chapter_postprocess",
                {
                    "chapter_path": str(source / "01-chapter-1.md"),
                    "title": {
                        "action": "set",
                        "line_number": 1,
                        "text": "Poglavlje 1",
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

    assert result.ok
    seen = "\n".join(
        str(message.content)
        for batch in chapter_model.seen_messages
        for message in batch
    )
    # The marker is well outside the 8-line opening window, so the capped
    # inspector must never surface it.
    assert marker not in seen


def test_bind_concatenates_postprocessed_files_in_reading_order(tmp_path: Path):
    source = tmp_path / "09-postprocessed-chapters"
    source.mkdir()
    (source / "00-front-matter.md").write_text("# Naslov\n\n", encoding="utf-8")
    (source / "00-table-of-contents.md").write_text(
        "# Sadržaj\n\n1. Poglavlje 1\n", encoding="utf-8"
    )
    (source / "01-chapter-1.md").write_text("# Poglavlje 1\n\nPrva.\n", encoding="utf-8")
    target = tmp_path / "10-translated-book.md"

    result = run_bind(source, target)

    assert result.ok, result.errors
    assert result.chapter_count == 3
    assert result.toc_included is True
    text = target.read_text(encoding="utf-8")
    assert text.index("# Naslov") < text.index("# Sadržaj") < text.index("# Poglavlje 1")
    assert result.manifest_path and Path(result.manifest_path).is_file()


def test_a_plain_chapter_title_must_be_promoted_to_a_heading(tmp_path: Path):
    chapter = tmp_path / "01-chapter-1.md"
    chapter.write_text("Poglavlje 1\n\nPrva rečenica.\n", encoding="utf-8")
    structure = scan_chapter_structure(chapter)
    plan = {
        "title": {"action": "keep", "line_number": 1, "text": "Poglavlje 1", "level": 1},
        "headings": [],
    }
    validation = validate_chapter_plan(
        chapter, plan, canonical_title="Poglavlje 1", structure=structure
    )
    assert validation["ok"] is False
    assert any("not a Markdown heading" in issue for issue in validation["issues"])

    plan["title"]["action"] = "set"
    assert validate_chapter_plan(
        chapter, plan, canonical_title="Poglavlje 1", structure=structure
    )["ok"]
    applied = apply_chapter_plan(
        chapter, plan, canonical_title="Poglavlje 1", structure=structure
    )
    assert applied.lines[0] == "# Poglavlje 1"


def test_an_all_caps_date_line_is_the_chapter_title(tmp_path: Path):
    chapter = tmp_path / "01-november-8-2000.md"
    chapter.write_text(
        "8. NOVEMBAR 2000.\n\nDragi Frenkline,\n\nPrva rečenica.\n", encoding="utf-8"
    )
    structure = scan_chapter_structure(chapter)

    # A date is a title even though it ends with a period; the salutation below
    # it is body text, not the chapter's subtitle.
    assert structure.title == "8. NOVEMBAR 2000."
    assert structure.title_line == 1
    assert structure.subtitle_line is None

    plan = {"title": {"action": "set", "line_number": 1, "level": 1}, "headings": []}
    applied = apply_chapter_plan(
        chapter, plan, canonical_title="8. novembar 2000.", structure=structure
    )
    assert applied.lines[0] == "# 8. novembar 2000."
    assert "8. NOVEMBAR 2000." not in applied.lines
    assert "Dragi Frenkline," in applied.lines


def test_a_salutation_is_never_promoted_to_a_chapter_title(tmp_path: Path):
    chapter = tmp_path / "01-letter.md"
    chapter.write_text("Dragi Frenkline,\n\nPrva rečenica.\n", encoding="utf-8")
    structure = scan_chapter_structure(chapter)

    assert structure.title_line is None
    plan = {"title": {"action": "insert", "level": 1}, "headings": []}
    applied = apply_chapter_plan(
        chapter, plan, canonical_title="Prvo pismo", structure=structure
    )
    assert applied.lines[0] == "# Prvo pismo"
    assert "Dragi Frenkline," in applied.lines


def test_a_preface_is_front_matter_even_when_its_filename_is_numbered(tmp_path: Path):
    (tmp_path / "00-front-matter.md").write_text("# Title\n", encoding="utf-8")
    (tmp_path / "01-predgovor.md").write_text("# PREDGOVOR\n\nText.\n", encoding="utf-8")
    (tmp_path / "02-1-prvo-poglavlje.md").write_text(
        "# 1 PRVO POGLAVLJE\n\nText.\n", encoding="utf-8"
    )
    book = scan_book_structure(tmp_path)

    assert book["front_matter"]["filename"] == "00-front-matter.md"
    by_name = {chapter["filename"]: chapter for chapter in book["chapters"]}
    assert by_name["01-predgovor.md"]["matter"] == "front"
    assert by_name["02-1-prvo-poglavlje.md"]["body_index"] == 1
    plan = {
        "naming": {
            "mode": "normalize",
            "label": "Poglavlje",
            "include_title": False,
            "level": 1,
        },
        "toc": {"action": "create", "heading": "Sadržaj"},
    }
    validation = validate_book_plan(tmp_path, plan, book=book)
    assert validation["canonical_titles"]["02-1-prvo-poglavlje.md"] == "Poglavlje 1"
    assert validation["canonical_titles"]["01-predgovor.md"] == "PREDGOVOR"


def test_first_plain_title_wins_over_a_later_section_header(tmp_path: Path):
    chapter = tmp_path / "01-chapter-1.md"
    chapter.write_text(
        "Poglavlje 1\n\n## Ovo je podnaslov\n\nPrva rečenica.\n", encoding="utf-8"
    )
    structure = scan_chapter_structure(chapter)

    assert structure.title == "Poglavlje 1"
    assert structure.title_line == 1
    assert structure.title_is_heading is False
    assert structure.headings[0]["text"] == "Ovo je podnaslov"
