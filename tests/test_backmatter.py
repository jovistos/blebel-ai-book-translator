"""Tests for the tail reader and the back-matter split (no LLM, no network).

Everything here is deterministic: the detector, the tool wrapper, the validation
rule and the slice/manifest bookkeeping for the one split point that does not
have to look like a chapter title.
"""

from __future__ import annotations

import json

from book_splitter.agent.tools import (
    SPLIT_TOOLS,
    inspect_book_tail,
    propose_split_map,
    stage_of,
    tool_names,
)
from book_splitter.agent.tools.registry import SPLIT
from book_splitter.analysis.backmatter import (
    back_matter_line_limit,
    clamp_chunk,
    detect_back_matter,
    is_back_matter_role,
)
from book_splitter.core.constants import (
    DEFAULT_TAIL_CHUNK_LINES,
    MAX_TAIL_CHUNK_LINES,
    MIN_TAIL_CHUNK_LINES,
)
from book_splitter.slicing.slice import slice_book
from book_splitter.slicing.validation import validate_split_map
from book_splitter.slicing.verify import verify_output

# --------------------------------------------------------------------------- #
# A synthetic book: front TOC, four chapters, and (optionally) a tail of
# publisher furniture — an author biography, a long catalogue and the chapter
# list printed a second time.
# --------------------------------------------------------------------------- #

TITLES = [
    "Chapter One: The Long Road",
    "Chapter Two: A Quiet House",
    "Chapter Three: The Return",
    "Chapter Four: What Was Said",
]

BIO_LINE = "Alexander Test is the author of the Long Road series and lives in Scotland."


def _prose(prefix: str) -> str:
    return f"{prefix}: " + "the story continues along a dusty road for a while " * 4


def _book(with_back_matter: bool = True, paragraphs: int = 12, catalogue: int = 24) -> list[str]:
    lines: list[str] = ["THE TEST BOOK\n", "\n", "CONTENTS\n", "\n"]
    for title in TITLES:
        lines += [f"{title}\n", "\n"]
    for i, title in enumerate(TITLES, start=1):
        lines += [f"{title.upper()}\n", "\n"]
        for p in range(paragraphs):
            lines += [_prose(f"Paragraph {p} of chapter {i}") + "\n", "\n"]
    if with_back_matter:
        lines += [f"{BIO_LINE}\n", "\n", "Also available as an ebook:\n"]
        for i in range(1, catalogue + 1):
            lines.append(f"Long Road Volume {i} • 978-0-000-00{i:03d}-{i % 10}\n")
        lines += ["Visit: Follow: Friend:\n", "\n"]
        for title in TITLES:
            lines += [f"{title}\n", "\n"]
    return lines


def _first_line(lines: list[str], needle: str) -> int:
    """1-indexed line number of the first line containing ``needle``."""
    return next(i for i, line in enumerate(lines, start=1) if needle in line)


def _write(tmp_path, lines: list[str]) -> str:
    path = tmp_path / "book.md"
    path.write_text("".join(lines), encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


def test_short_window_escalates_and_the_loop_converges_on_the_boundary():
    # A catalogue long enough that the default 40-line window sits entirely
    # inside it: the block's first line can only be seen from further up.
    lines = _book(paragraphs=70, catalogue=140)
    expected = _first_line(lines, BIO_LINE)

    first = detect_back_matter(lines, chunk_lines=DEFAULT_TAIL_CHUNK_LINES, next_index=5)
    assert first["back_matter"]["detected"] is True
    assert first["escalate"] is True
    assert first["back_matter"]["boundary_visible"] is False
    assert first["back_matter"]["start_line"] is None
    assert first["proposed_split"] is None
    assert first["next_chunk_lines"] == DEFAULT_TAIL_CHUNK_LINES * 2

    # Follow the tool's own instruction: ask again with the larger chunk.
    chunk = first["next_chunk_lines"]
    seen = [first]
    for _ in range(6):
        result = detect_back_matter(lines, chunk_lines=chunk, next_index=5)
        seen.append(result)
        if not result["escalate"]:
            break
        chunk = result["next_chunk_lines"]

    last = seen[-1]
    assert last["escalate"] is False
    assert last["back_matter"]["boundary_visible"] is True
    assert last["back_matter"]["start_line"] == expected
    assert last["proposed_split"]["line_number"] == expected
    assert last["proposed_split"]["filename"] == "05-back-matter.md"
    assert last["proposed_split"]["role"] == "back-matter"
    assert len(seen) >= 3  # the short window genuinely was not enough


def test_the_boundary_is_a_line_of_text_not_the_blank_line_above_it():
    lines = _book()
    expected = _first_line(lines, BIO_LINE)
    assert lines[expected - 2].strip() == ""  # a blank separator precedes the block
    result = detect_back_matter(lines, chunk_lines=MAX_TAIL_CHUNK_LINES, next_index=5)
    assert result["proposed_split"]["line_number"] == expected

    mapping = [{"line_number": expected, "filename": "05-back-matter.md", "role": "back-matter"}]
    assert validate_split_map(lines, mapping)["ok"] is False  # too few chapters, but not blank
    assert not any("is blank" in i for i in validate_split_map(lines, mapping)["issues"])

    blank_mapping = [
        {"line_number": expected - 1, "filename": "05-back-matter.md", "role": "back-matter"}
    ]
    assert any("is blank" in i for i in validate_split_map(lines, blank_mapping)["issues"])


def test_a_book_without_back_matter_says_so_and_proposes_nothing():
    lines = _book(with_back_matter=False)

    short = detect_back_matter(lines, chunk_lines=DEFAULT_TAIL_CHUNK_LINES)
    assert short["back_matter"]["detected"] is False
    assert short["proposed_split"] is None
    assert short["escalate"] is True  # "not enough" - look at a bigger chunk
    assert "escalate" in short["recommended_action"] or "again" in short["recommended_action"]

    full = detect_back_matter(lines, chunk_lines=MAX_TAIL_CHUNK_LINES)
    assert full["escalate"] is False
    assert full["back_matter"]["detected"] is False
    assert full["proposed_split"] is None
    assert "do not add a back-matter split" in full["recommended_action"]


def test_a_stray_keyword_in_prose_does_not_create_back_matter():
    lines = ["CHAPTER ONE\n", "\n", "He had sold the copyright years ago.\n"]
    result = detect_back_matter(lines, chunk_lines=MAX_TAIL_CHUNK_LINES)
    assert result["back_matter"]["detected"] is False
    assert result["proposed_split"] is None


def test_an_about_the_author_heading_is_structural_evidence_whatever_the_language():
    """``## O AUTORU`` heads a biography, not a chapter — the vocabulary is not English-only."""
    lines = [
        "CHAPTER ONE\n",
        "\n",
        "The last paragraph of the book.\n",
        "\n",
        "## O AUTORU\n",
        "\n",
        "Irvin D. Jalom je počasni profesor psihijatrije i živi u Kaliforniji.\n",
    ]
    result = detect_back_matter(lines, chunk_lines=MAX_TAIL_CHUNK_LINES, next_index=2)
    assert result["back_matter"]["detected"] is True
    assert result["back_matter"]["boundary_visible"] is True
    assert "about-author-heading" in result["back_matter"]["signals"]
    assert result["proposed_split"]["line_number"] == 5
    assert result["proposed_split"]["filename"] == "02-back-matter.md"
    assert result["escalate"] is False


def test_only_a_whole_line_counts_as_an_about_the_author_heading():
    from book_splitter.analysis.backmatter import line_signals

    assert line_signals("## O AUTORU", set()) == ["about-author-heading"]
    assert line_signals("Über den Autor", set()) == ["about-author-heading"]
    # …a sentence that merely mentions the author does not.
    assert line_signals("Razgovarali smo o autoru te knjige.", set()) == []


def test_chunk_length_is_clamped_and_the_window_is_capped_in_characters():
    assert clamp_chunk(1, 5000) == MIN_TAIL_CHUNK_LINES
    assert clamp_chunk(10_000, 5000) == MAX_TAIL_CHUNK_LINES
    assert clamp_chunk("nonsense", 5000) == DEFAULT_TAIL_CHUNK_LINES
    assert clamp_chunk(400, 12) == 12  # never longer than the book

    lines = [_prose(f"Line {i}") + "\n" for i in range(600)]
    result = detect_back_matter(lines, chunk_lines=MAX_TAIL_CHUNK_LINES)
    window = result["window"]
    assert result["chunk_lines"] == MAX_TAIL_CHUNK_LINES
    assert window["truncated_at_chars"] is True
    assert window["end_line"] == len(lines)  # the true tail is never dropped
    assert window["lines"][-1]["line_number"] == len(lines)


def test_tool_is_registered_for_stage_one_only():
    assert stage_of("inspect_book_tail") == "split"
    assert "inspect_book_tail" in tool_names(SPLIT)
    assert "inspect_book_tail" in [t.name for t in SPLIT_TOOLS]
    first = inspect_book_tail.description.strip().splitlines()[0]
    assert "back matter" in first.lower()


def test_tool_reports_the_tail_of_a_real_file(tmp_path):
    lines = _book()
    path = _write(tmp_path, lines)
    payload = json.loads(
        inspect_book_tail.invoke({"file_path": path, "chunk_lines": MAX_TAIL_CHUNK_LINES, "next_index": 5})
    )
    assert payload["file"] == path
    assert payload["back_matter"]["boundary_visible"] is True
    assert payload["proposed_split"]["line_number"] == _first_line(lines, BIO_LINE)
    assert payload["candidates"]["count"] > 20
    assert "TAIL WINDOW ONLY" in payload["note"]


# --------------------------------------------------------------------------- #
# Validation: the back-matter role is the only relaxed split point
# --------------------------------------------------------------------------- #


def _book_plan(lines: list[str], role: bool = True, back_matter_line: int | None = None) -> list[dict]:
    starts = [_first_line(lines, f"{t.upper()}") for t in TITLES]
    mapping = [
        {"line_number": ln, "filename": f"{i:02d}-chapter.md"}
        for i, ln in enumerate(starts, start=1)
    ]
    entry = {
        "line_number": back_matter_line or _first_line(lines, BIO_LINE),
        "filename": "05-back-matter.md",
    }
    if role:
        entry["role"] = "back-matter"
    mapping.append(entry)
    return mapping


def test_validation_accepts_back_matter_as_the_last_entry():
    lines = _book()
    result = validate_split_map(lines, _book_plan(lines))
    assert result["ok"], result["issues"]
    assert result["back_matter_count"] == 1
    assert result["chapter_count"] == len(TITLES)  # the block is not a chapter
    assert result["split_count"] == len(TITLES) + 1
    assert result["points"][-1]["role"] == "back-matter"


def test_validation_accepts_a_back_matter_split_that_is_not_title_shaped():
    """An author biography is a 200-character paragraph glued between two lines."""
    lines: list[str] = []
    for i in range(1, 4):
        lines += [f"CHAPTER {i}\n", "\n"]
        for p in range(30):
            lines += [f"Prose paragraph {p} in chapter {i}.\n", "\n"]
    lines += ["CATALOG START\n", "Y " * 100 + "\n", "MORE CATALOG\n"]
    long_line = len(lines) - 1

    mapping = [
        {"line_number": 1, "filename": "01-c.md"},
        {"line_number": 63, "filename": "02-c.md"},
        {"line_number": 125, "filename": "03-c.md"},
        {"line_number": long_line, "filename": "04-back-matter.md", "role": "back-matter"},
    ]
    result = validate_split_map(lines, mapping)
    assert result["ok"], result["issues"]
    point = result["points"][-1]
    assert point["is_standalone"] is False and point["similarity_to_title"] is None
    assert any("back matter need not" in w for w in result["warnings"])


def test_the_same_entry_without_the_role_is_rejected_as_prose():
    """The catalogue line is not standalone, so only the role makes it legal."""
    lines = _book()
    line = _first_line(lines, "Also available as an ebook")
    with_role = validate_split_map(lines, _book_plan(lines, back_matter_line=line))
    assert with_role["ok"], with_role["issues"]

    without_role = validate_split_map(
        lines, _book_plan(lines, role=False, back_matter_line=line)
    )
    assert not without_role["ok"]
    assert any("does not look like a title" in issue for issue in without_role["issues"])


def test_validation_rejects_a_back_matter_split_that_is_not_last():
    lines = _book()
    mapping = _book_plan(lines)
    mapping.insert(0, mapping.pop())  # submitted out of order; sorted, so still last...
    result = validate_split_map(lines, mapping)
    assert result["ok"], result["issues"]  # order in the list does not matter

    # Put a real chapter AFTER the back-matter entry: that is what must fail.
    mapping = _book_plan(lines)
    mapping.append({"line_number": len(lines) - 1, "filename": "06-late.md"})
    result = validate_split_map(lines, mapping)
    assert not result["ok"]
    assert any("must be the LAST entry" in issue for issue in result["issues"])


def test_validation_rejects_a_back_matter_block_that_is_too_long():
    lines = _book(with_back_matter=False)
    limit = back_matter_line_limit(len(lines))
    mapping = [{"line_number": 5, "filename": "05-back-matter.md", "role": "back-matter"}]
    result = validate_split_map(lines, mapping)
    tail = len(lines) - mapping[0]["line_number"] + 1
    assert tail > limit
    assert any("trailing lines" in issue for issue in result["issues"])


def test_role_survives_the_proposal_tool():
    lines = _book()
    plan = _book_plan(lines)
    payload = json.loads(propose_split_map.invoke({"split_mapping": plan, "strategy": "heuristics"}))
    assert payload["accepted"]
    assert payload["split_mapping"][-1]["role"] == "back-matter"

    without = [dict(e) for e in plan]
    without[-1].pop("role")
    payload = json.loads(propose_split_map.invoke({"split_mapping": without}))
    assert "role" not in payload["split_mapping"][-1]


def test_is_back_matter_role_accepts_the_documented_spellings():
    assert is_back_matter_role("back-matter")
    assert is_back_matter_role(" Back_Matter ")
    assert not is_back_matter_role("chapter")
    assert not is_back_matter_role(None)


# --------------------------------------------------------------------------- #
# Slicing: the block becomes its own file, the book stays byte-for-byte
# --------------------------------------------------------------------------- #


def test_slice_writes_the_back_matter_file_and_keeps_the_book_intact(tmp_path):
    lines = _book()
    path = _write(tmp_path, lines)
    mapping = _book_plan(lines)
    manifest = slice_book(path, mapping, str(tmp_path / "out"))

    kinds = {f["filename"]: f["kind"] for f in manifest["files"]}
    assert kinds["05-back-matter.md"] == "back-matter"
    assert sum(1 for k in kinds.values() if k == "chapter") == len(TITLES)
    assert manifest["chapter_count"] == len(TITLES)
    assert manifest["file_count"] == len(TITLES) + 2  # + front matter + back matter

    back = tmp_path / "out" / "05-back-matter.md"
    assert back.read_text(encoding="utf-8").startswith("Alexander Test is the author")

    report = verify_output(manifest)
    assert report["ok"], report["problems"]
    assert report["reassembles_byte_for_byte"] is True
    assert report["output_lines"] == len(lines)
