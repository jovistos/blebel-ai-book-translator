"""Tests for stage 2's core: header evidence, damage shapes, repair rules.

Everything here is deterministic: no LLM, no network, no API key. The repairs are
tested against the *real* damaged strings found in the sample book, because those
are the cases the stage exists for.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from book_splitter.documents.bodyline import (
    has_midword_number,
    is_word_continuation,
    starts_with_lowercase,
)
from book_splitter.documents.markdown import (
    is_all_caps,
    is_page_number_line,
    page_number_of,
    strip_emphasis,
)
from book_splitter.preprocess.damage import (
    INLINE_PAGE_NUMBER,
    PAGE_NUMBER_IN_WORD,
    SPACE_BEFORE_PUNCTUATION,
    SPLIT_WORD,
    find_inline_page_numbers,
    inline_page_number_candidate,
    repair_inline_page_number,
    repair_page_number_in_word,
    repair_space_before_punctuation,
    repair_split_word,
    scan_chapter_damage,
)
from book_splitter.preprocess.inspect import (
    DEMOTE,
    KEEP,
    MERGE_DEMOTE,
    MERGE_WITH_NEXT,
    REMOVE,
    REMOVE_CAPTION,
    inspect_chapter,
)
from book_splitter.preprocess.transform import (
    apply_cleaning_plan,
    validate_cleaning_plan,
    validate_text_fixes,
    verify_cleaning_plan,
)

# --------------------------------------------------------------------------- #
# line shapes
# --------------------------------------------------------------------------- #


def test_page_number_lines_are_recognised_with_and_without_markup():
    assert is_page_number_line("## 18")
    assert is_page_number_line("18")
    assert is_page_number_line("## 790")
    assert page_number_of("## 101") == 101
    assert not is_page_number_line("## 2. POGLAVLJE")
    assert not is_page_number_line("## GLAVA 1 LJUBAV JE U SRCU ŽIVOTA")
    assert not is_page_number_line("## 130 otkucaja")


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("- 2 -", 2),
        ("- 208 -", 208),
        ("-208-", 208),
        ("– 208 –", 208),  # en dash
        ("— 208 —", 208),  # em dash
        ("## - 2 -", 2),  # a framed page number that became a header
        ("## — 208 —", 208),
    ],
)
def test_framed_page_numbers_are_recognised(line, expected):
    """The converter sometimes keeps the dashes printed around a page number."""
    assert is_page_number_line(line)
    assert page_number_of(line) == expected


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("- 1 3 -", 13),  # the conversion put a space between the digits
        ("- 1 0 4 -", 104),
        ("## - 1 0 4 -", 104),  # the same, as a header
        ("- 2 2 5 - -", 225),  # an extra printed dash survived
        ("- \\_ 1 1 \\_", 11),  # escaped underscores around the number
        ("\\_ 1 3 \\_", 13),
        ("_11_", 11),  # the same, without the backslashes
        ("- 3 3", 33),  # a one-sided dash, but spaced digits
        ("- 3 2 2", 322),
        ("1 3", 13),  # bare digits that the conversion spaced apart
    ],
)
def test_spaced_and_underscored_page_numbers_are_recognised(line, expected):
    assert is_page_number_line(line)
    assert page_number_of(line) == expected


@pytest.mark.parametrize(
    "line",
    [
        "- 2",  # a Markdown list item, not a framed page number
        "- 2020",  # a one-sided unspaced dash is still a list item
        "2 -",
        "- 2 - text",
        "- 2. POGLAVLJE",
        "- 2 - 3 -",
        "--",
        "_11",  # a lone underscore is emphasis, not a frame
        "A \\_ 5 3 \\_",  # real text before the page number
        "- 1 2 3 4 5 -",  # more digits than a page number can have
    ],
)
def test_a_frame_must_surround_only_the_number(line):
    assert not is_page_number_line(line)
    assert page_number_of(line) is None


def test_all_caps_detection_ignores_punctuation_and_digits():
    assert is_all_caps("GLAVA 1 LJUBAV JE U SRCU ŽIVOTA")
    assert is_all_caps("SEKS I SRCE:")
    assert is_all_caps("PRVI DEO ISPUNJENJE LJUBAVI")
    assert not is_all_caps("Slika 7. Protok uzbuđ enja")
    assert not is_all_caps("A")


def test_emphasis_markers_are_stripped_only_when_paired():
    assert strip_emphasis("radionica * , počeo") == "radionica * , počeo"
    assert strip_emphasis("tekst **važan** deo") == "tekst važan deo"


# --------------------------------------------------------------------------- #
# damage shapes
# --------------------------------------------------------------------------- #


def test_midword_number_detects_a_page_number_inside_a_word():
    hits = has_midword_number("analitička terapija us12 merena na telo")
    assert [h["digits"] for h in hits] == ["12"]
    assert hits[0]["position"] == "midword"


def test_midword_number_detects_a_page_number_glued_to_the_next_word():
    hits = has_midword_number("1otkucaja u minuti")
    assert [h["digits"] for h in hits] == ["1"]
    assert hits[0]["position"] == "leading"


@pytest.mark.parametrize(
    "line",
    [
        "Masters i Džonson 1 zabeležili su brzinu od 130 otkucaja u minuti.",
        "Treće izdanje Beograd, 2005. Tiraž 1000 ISBN 86-7348-167-8",
        "To su tromboksan (TxA2) i prostaciklin (PG12), oba derivati.",
        "Godine 2011 bila je teška.",
    ],
)
def test_inline_numbers_are_not_reported_as_damage(line):
    assert has_midword_number(line) == []


def test_a_broken_word_tail_continuing_on_the_next_line_is_a_continuation():
    left = "x" * 200 + " pažljivu psi"
    assert is_word_continuation(left, "182 ljutnje. Za zdravo funkcionisanje")
    assert is_word_continuation(left, "1otkucaja u minuti")
    assert not is_word_continuation(left, "Za zdravo funkcionisanje")
    assert not is_word_continuation("kratko", "nastavak")


def test_starts_with_lowercase_is_a_continuation_not_a_title():
    assert starts_with_lowercase("verovatno je tako")
    assert not starts_with_lowercase("Verovatno je tako")
    assert not starts_with_lowercase("## GLAVA 2")


# --------------------------------------------------------------------------- #
# repair rules — the exact strings the sample book contains
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("... terapija us12 merena na telo", "... terapija usmerena na telo"),
        ("diskusija o disfunkcional38 nost smanjuje", "diskusija o disfunkcionalnost smanjuje"),
        ("Kao adoles90 cent, krao je", "Kao adolescent, krao je"),
        ('Drža117 vama", pisao je', 'Državama", pisao je'),
        ("koji pot122 padaju pod kontrolu", "koji potpadaju pod kontrolu"),
        ("pažljivu psi182 ljutnje.", "pažljivu psiljutnje."),
        ("Ako se koncen192 trišemo", "Ako se koncentrišemo"),
        ("ljubavi. Na796 j češ ći dokaz", "ljubavi. Naj češ ći dokaz"),
        ("i božan199 stvena.", "i božanstvena."),
    ],
)
def test_page_number_inside_a_word_is_repaired(source, expected):
    assert repair_page_number_in_word(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("brzinu od 130 otkucaja", "brzinu od 130 otkucaja"),
        ("Tiraž 1000 ISBN 86-7348-167-8", "Tiraž 1000 ISBN 86-7348-167-8"),
        ("(TxA2) i (PG12)", "(TxA2) i (PG12)"),
    ],
)
def test_legitimate_numbers_survive_the_repair(source, expected):
    assert repair_page_number_in_word(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # a page number printed in front of the first word of a line
        ("30 spreč avaju da nešto učini", "spreč avaju da nešto učini"),
        ("40 godina, patila je od depresije", "godina, patila je od depresije"),
    ],
)
def test_a_leading_page_number_is_removed_with_its_space(source, expected):
    assert repair_page_number_in_word(source) == expected


def test_space_before_punctuation_is_repaired():
    assert (
        repair_space_before_punctuation("Sve je mišić , kao i svaki") == "Sve je mišić, kao i svaki"
    )
    assert repair_space_before_punctuation("pomoć , i to") == "pomoć, i to"
    assert repair_space_before_punctuation("nema šta da se menja") == "nema šta da se menja"


@pytest.mark.parametrize(
    ("line", "position", "token", "page"),
    [
        ('- 2 0 "Well," you said.', "prefix", "- 2 0", 20),
        ("... to be mean to him the - 6 8 -", "suffix", "- 6 8 -", 68),
        ('... a year. - 117 "Obviously," I said.', "middle", "- 117", 117),
        ("... our sister Valerie 265", "suffix", "265", 265),
    ],
)
def test_inline_page_number_candidate_finds_the_token_and_position(
    line, position, token, page
):
    candidate = inline_page_number_candidate(line)
    assert candidate is not None
    assert candidate["position"] == position
    assert candidate["token"] == token
    assert candidate["page_number"] == page


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('- 2 0 "Well," you said.', '"Well," you said.'),
        ("... to be mean to him the - 6 8 -", "... to be mean to him the"),
        (
            '... a year. - 117 "Obviously," I said.',
            '... a year. "Obviously," I said.',
        ),
        ("... our sister Valerie 265", "... our sister Valerie"),
    ],
)
def test_inline_page_number_repair_keeps_every_word(source, expected):
    assert repair_inline_page_number(source) == expected


@pytest.mark.parametrize(
    "line",
    [
        "- 2 -",  # a standalone page number, not an inline one
        "- 2 2 5 - -",  # standalone, with an extra printed dash
        "mid-1980s music",  # a hyphen inside a word
        "## GLAVA 2",  # a heading, not page furniture
        "NOVEMBER 1 5, 2 0 0 0",  # a four-digit bare year
        "1982 was a good year.",  # a bare four-digit year
    ],
)
def test_inline_page_number_ignores_standalone_decorated_and_real_numbers(line):
    assert inline_page_number_candidate(line) is None
    assert repair_inline_page_number(line) == line


def test_split_word_join_keeps_the_word_intact():
    left = "x" * 200 + " i božan"
    joined, rest = repair_split_word(left, "stvena. Po istom pravilu")
    assert joined.endswith("i božan")
    assert joined.rstrip().endswith("božan")
    assert rest == "stvena. Po istom pravilu"


# --------------------------------------------------------------------------- #
# header classification, on the sample book's damage
# --------------------------------------------------------------------------- #


def _chapter(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "chapter.md"
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return path


def _inline_chapter(tmp_path: Path) -> Path:
    """A chapter with one page number merged into a line, between two markers."""
    return _chapter(
        tmp_path,
        [
            "## GLAVA 1",  # 1
            "",  # 2
            "Prose before. " + "x" * 120,  # 3
            "",  # 4
            "- 19 -",  # 5
            "",  # 6
            "Prose just before the page break.",  # 7
            "",  # 8
            '- 2 0 "Well," you said.',  # 9
            "",  # 10
            "Prose just after the page break.",  # 11
            "",  # 12
            "- 21 -",  # 13
            "",  # 14
            "Prose after.",  # 15
        ],
    )


def test_inline_page_number_candidate_carries_context_and_page_sequence(tmp_path):
    path = _inline_chapter(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    findings = find_inline_page_numbers(lines)
    assert len(findings) == 1
    finding = findings[0]
    assert finding.line_number == 9
    assert finding.risk == "low"
    extra = finding.extra
    assert extra["token"] == "- 2 0"
    assert extra["page_number"] == 20
    assert extra["position"] == "prefix"
    assert extra["fits_page_sequence"] is True
    assert extra["suggested"] == "remove"
    assert extra["page_before"] == 19
    assert extra["page_after"] == 21
    assert extra["previous_line_number"] == 7
    assert extra["next_line_number"] == 11
    assert "Prose just before" in extra["previous_text"]
    assert "Prose just after" in extra["next_text"]
    assert extra["after_same_line"].startswith('"Well,')


def test_a_bare_number_that_does_not_fit_the_page_sequence_is_flagged_for_review(tmp_path):
    path = _chapter(
        tmp_path,
        [
            "## GLAVA 1",
            "",
            "- 19 -",
            "",
            "I did a daily 100",
            "",
            "- 21 -",
        ],
    )
    finding = find_inline_page_numbers(path.read_text(encoding="utf-8").splitlines())[0]
    assert finding.extra["token"] == "100"
    assert finding.extra["fits_page_sequence"] is False
    assert finding.extra["suggested"] == "review"
    assert finding.risk == "high"


def test_inline_page_number_fix_is_applied_and_verified(tmp_path):
    path = _inline_chapter(tmp_path)
    fixes = [{"line_number": 9, "fix": INLINE_PAGE_NUMBER}]
    accepted, issues, _ = validate_text_fixes(path, fixes)
    assert issues == [] and len(accepted) == 1

    result = apply_cleaning_plan(path, [], accepted)
    assert result.lines[8] == '"Well," you said.'
    assert result.touched[9]["fix"] == INLINE_PAGE_NUMBER
    verification = verify_cleaning_plan(path, [], accepted)
    assert verification["ok"], verification["problems"]


def test_an_inline_page_number_fix_without_a_candidate_is_rejected(tmp_path):
    path = _chapter(tmp_path, ["## GLAVA 1", "", "Običan pasus teksta."])
    accepted, issues, _ = validate_text_fixes(
        path, [{"line_number": 3, "fix": INLINE_PAGE_NUMBER}]
    )
    assert accepted == []
    assert issues and "no inline_page_number damage" in issues[0]


def test_a_page_number_header_is_recommended_for_removal(tmp_path):
    path = _chapter(tmp_path, ["## GLAVA 1", "", "## 18", "", "Tekst pasusa. " + "x" * 200])
    report = inspect_chapter(path)
    review = {item["line_number"]: item for item in report["review"]}
    assert review[3]["suggested"] == REMOVE
    assert review[3]["risk"] == "very_low"
    # the real heading is left alone
    assert 1 not in review or review[1]["suggested"] == KEEP


def test_a_framed_page_number_header_is_recommended_for_removal(tmp_path):
    path = _chapter(tmp_path, ["## GLAVA 1", "", "## - 18 -", "", "Tekst pasusa. " + "x" * 200])
    review = {item["line_number"]: item for item in inspect_chapter(path)["review"]}
    assert review[3]["suggested"] == REMOVE
    assert review[3]["risk"] == "very_low"


def test_a_caption_header_is_removed(tmp_path):
    path = _chapter(
        tmp_path,
        ["## GLAVA 1", "", "## Slika 7. Protok uzbuđ enja u telu.", "", "Tekst. " + "x" * 200],
    )
    review = {item["line_number"]: item for item in inspect_chapter(path)["review"]}
    assert review[3]["suggested"] == REMOVE_CAPTION


def test_two_halves_of_one_heading_are_merged(tmp_path):
    path = _chapter(tmp_path, ["## GLAVA 2", "", "## SEKS I SRCE", "", "Tekst. " + "x" * 200])
    review = {item["line_number"]: item for item in inspect_chapter(path)["review"]}
    assert review[1]["suggested"] == MERGE_WITH_NEXT
    assert review[3]["suggested"] == MERGE_DEMOTE


def test_body_text_promoted_to_a_header_is_demoted(tmp_path):
    long_line = "## " + "reč " * 60
    path = _chapter(tmp_path, ["## GLAVA 1", "", long_line, "", "Tekst."])
    review = {item["line_number"]: item for item in inspect_chapter(path)["review"]}
    assert review[3]["suggested"] == DEMOTE


def test_a_real_heading_is_kept(tmp_path):
    path = _chapter(
        tmp_path,
        ["## BELEŠKE", "", "## 2. POGLAVLJE", "", "1. Neki citat. " + "x" * 200],
    )
    report = inspect_chapter(path, include_all=True)
    by_line = {item["line_number"]: item for item in report["review"]}
    assert by_line[1]["suggested"] == KEEP  # a real heading
    assert by_line[3]["suggested"] == KEEP  # a numbered section heading
    # ... and by default neither reaches the agent, because nothing needs changing
    default = inspect_chapter(path)
    assert default["review"] == []
    assert default["body_candidates"] == []


def test_header_record_carries_the_evidence(tmp_path):
    path = _chapter(tmp_path, ["## GLAVA 2", "", "## SEKS I SRCE", "", "Tekst. " + "x" * 200])
    record = inspect_chapter(path)["headers"][0]
    assert record["is_all_caps"] is True
    assert record["next_is_header"] is True
    assert record["evidence"]


# --------------------------------------------------------------------------- #
# the repair engine: validate → apply → verify
# --------------------------------------------------------------------------- #


def test_a_plan_that_demotes_body_text_is_rejected(tmp_path):
    path = _chapter(tmp_path, ["## GLAVA 1", "", "Običan pasus teksta."])
    result = validate_cleaning_plan(path, [{"line_number": 3, "decision": "demote"}])
    assert not result["ok"]
    assert "not a header" in result["issues"][0]


def test_removing_a_real_heading_is_rejected(tmp_path):
    path = _chapter(tmp_path, ["## GLAVA 1", "", "Tekst."])
    result = validate_cleaning_plan(path, [{"line_number": 1, "decision": "remove"}])
    assert not result["ok"]
    assert "only allowed for a line that is only a printed page number" in result["issues"][0]


def test_a_valid_plan_removes_only_declared_artefacts(tmp_path):
    path = _chapter(tmp_path, ["## GLAVA 5 STRAH OD LJUBAVI", "", "## 101", "", "Tekst pasusa."])
    operations = [{"line_number": 3, "decision": "remove", "note": "page 101"}]
    assert validate_cleaning_plan(path, operations)["ok"]

    result = apply_cleaning_plan(path, operations)
    # The artefact's slot stays, blank: every other line keeps its line number.
    assert result.lines == ["## GLAVA 5 STRAH OD LJUBAVI", "", "", "", "Tekst pasusa."]
    assert result.removed_artefacts == [3]

    verification = verify_cleaning_plan(path, operations)
    assert verification["ok"], verification["problems"]
    assert verification["provenance"]["removed_artefacts"] == [3]


def test_merging_across_a_blank_line_keeps_the_heading(tmp_path):
    path = _chapter(
        tmp_path, ["## GLAVA 2", "", "## SEKS I SRCE", "", "Tekst pasusa. " + "x" * 200]
    )
    operations = [
        {"line_number": 1, "decision": "merge_with_next"},
        {"line_number": 3, "decision": "demote"},
    ]
    assert validate_cleaning_plan(path, operations)["ok"]
    result = apply_cleaning_plan(path, operations)
    assert result.lines[0] == "## GLAVA 2 SEKS I SRCE"
    assert result.lines[1] == ""
    assert verify_cleaning_plan(path, operations)["ok"]


def test_merging_is_rejected_when_the_next_header_is_a_page_number(tmp_path):
    path = _chapter(tmp_path, ["## GLAVA 2", "", "## 56", "", "Tekst."])
    result = validate_cleaning_plan(path, [{"line_number": 1, "decision": "merge_with_next"}])
    assert not result["ok"]
    assert "page number" in result["issues"][0]


def test_a_fix_for_damage_that_is_not_there_is_rejected(tmp_path):
    path = _chapter(tmp_path, ["## GLAVA 1", "", "Bez ikakvog oštećenja ovde."])
    accepted, issues, _ = validate_text_fixes(path, [{"line_number": 3, "fix": SPLIT_WORD}])
    assert accepted == []
    assert issues and "no split_word damage" in issues[0]


def test_a_text_fix_is_applied_and_verified(tmp_path):
    damaged = "x" * 200 + " us12 merena na telo, i mišić , kasnije."
    path = _chapter(tmp_path, [damaged])
    fixes = [
        {"line_number": 1, "fix": PAGE_NUMBER_IN_WORD},
        {"line_number": 1, "fix": SPACE_BEFORE_PUNCTUATION},
    ]
    accepted, issues, _ = validate_text_fixes(path, fixes)
    assert issues == [] and len(accepted) == 2

    result = apply_cleaning_plan(path, [], accepted)
    assert "usmerena" in result.lines[0]
    assert "mišić," in result.lines[0]
    verification = verify_cleaning_plan(path, [], accepted)
    assert verification["ok"], verification["problems"]


def test_a_split_word_fix_joins_the_two_lines(tmp_path):
    left = "x" * 200 + " i božan"
    path = _chapter(tmp_path, [left, "stvena. Po istom pravilu.", ""])
    accepted, issues, _ = validate_text_fixes(path, [{"line_number": 1, "fix": SPLIT_WORD}])
    assert issues == [] and len(accepted) == 1
    result = apply_cleaning_plan(path, [], accepted)
    assert "božanstvena" in result.lines[0]
    assert len(result.lines) == 2  # two source lines became one
    assert result.provenance[0].origin == "joined_split_word"
    assert verify_cleaning_plan(path, [], accepted)["ok"]


def test_a_demoted_heading_loses_a_leading_page_number(tmp_path):
    path = _chapter(
        tmp_path,
        ["## GLAVA 3", "", "## 75 lič nosti. Impuls posezanja je sputan.", "", "Tekst."],
    )
    operations = [{"line_number": 3, "decision": "demote"}]
    assert validate_cleaning_plan(path, operations)["ok"]
    result = apply_cleaning_plan(path, operations)
    assert result.lines[2] == "lič nosti. Impuls posezanja je sputan."
    assert verify_cleaning_plan(path, operations)["ok"]


def test_a_numbered_heading_keeps_its_number_when_demoted(tmp_path):
    path = _chapter(tmp_path, ["## 2. POGLAVLJE", "", "1. Neki citat. " + "x" * 200])
    operations = [{"line_number": 1, "decision": "demote"}]
    result = apply_cleaning_plan(path, operations)
    assert result.lines[0] == "2. POGLAVLJE"
    assert verify_cleaning_plan(path, operations)["ok"]


def test_merging_drops_the_page_number_from_the_second_half(tmp_path):
    path = _chapter(
        tmp_path, ["## GLAVA 2", "", "## 56 Slika 7. Protok", "", "Tekst. " + "x" * 200]
    )
    operations = [
        {"line_number": 1, "decision": "merge_with_next"},
        {"line_number": 3, "decision": "demote"},
    ]
    result = apply_cleaning_plan(path, operations)
    assert result.lines[0] == "## GLAVA 2 Slika 7. Protok"
    assert verify_cleaning_plan(path, operations)["ok"]


def test_plain_page_number_lines_and_captions_are_reported(tmp_path):
    path = _chapter(
        tmp_path,
        [
            "## GLAVA 1",
            "",
            "Tekst pasusa. " + "x" * 200,
            "",
            "19",
            "",
            "Slika 4. Blokada ljubavi. Impuls je blokiran.",
            "",
            "Slike 12a i 12b ilustruju različstruktura lič nosti.",
        ],
    )
    report = inspect_chapter(path)
    candidates = {c["line_number"]: c for c in report["body_candidates"]}
    assert candidates[5]["suggested"] == REMOVE
    assert candidates[5]["kind"] == "page_number"
    assert candidates[7]["suggested"] == REMOVE_CAPTION
    assert candidates[7]["kind"] == "caption"
    # the sentence that merely mentions the figures is not a candidate
    assert 9 not in candidates


def test_framed_page_number_body_lines_are_reported(tmp_path):
    """``- 2 -`` left as a plain body line is a page artefact, not a list item."""
    path = _chapter(
        tmp_path,
        [
            "## GLAVA 1",
            "",
            "Tekst pasusa. " + "x" * 200,
            "",
            "- 2 -",
            "",
            "Nastavak pasusa.",
            "",
            "— 4 —",
            "",
            "Nastavak drugog pasusa.",
        ],
    )
    report = inspect_chapter(path)
    candidates = {c["line_number"]: c for c in report["body_candidates"]}
    assert candidates[5]["kind"] == "page_number"
    assert candidates[5]["suggested"] == REMOVE
    assert candidates[9]["kind"] == "page_number"
    assert candidates[9]["suggested"] == REMOVE
    # both are also in the review list the agent is shown
    review = {item["line_number"]: item for item in report["review"]}
    assert review[5]["suggested"] == REMOVE
    assert review[9]["suggested"] == REMOVE


def test_spaced_and_underscored_page_lines_are_reported_and_removable(tmp_path):
    """The shapes real PDF conversions leave: spaced digits and escaped underscores."""
    path = _chapter(
        tmp_path,
        [
            "## GLAVA 1",
            "",
            "Tekst pasusa. " + "x" * 200,
            "",
            "- 1 3 -",
            "",
            "Nastavak pasusa.",
            "",
            "- \\_ 1 1 \\_",
            "",
            "Nastavak drugog pasusa.",
        ],
    )
    report = inspect_chapter(path)
    candidates = {c["line_number"]: c for c in report["body_candidates"]}
    assert candidates[5]["kind"] == "page_number"
    assert candidates[5]["suggested"] == REMOVE
    assert candidates[9]["kind"] == "page_number"
    assert candidates[9]["suggested"] == REMOVE

    operations = [
        {"line_number": 5, "decision": "remove", "note": "printed page 13"},
        {"line_number": 9, "decision": "remove", "note": "printed page 11"},
    ]
    assert validate_cleaning_plan(path, operations)["ok"]
    result = apply_cleaning_plan(path, operations)
    assert result.removed_artefacts == [5, 9]
    assert result.touched[5]["page_number"] == 13
    assert result.touched[9]["page_number"] == 11
    assert verify_cleaning_plan(path, operations)["ok"]


def test_a_caption_body_line_can_be_removed_and_is_verified(tmp_path):
    path = _chapter(
        tmp_path,
        [
            "## GLAVA 1",
            "",
            "Tekst pasusa. " + "x" * 200,
            "",
            "Slika 4. Blokada ljubavi. Impuls je blokiran.",
            "",
            "19",
        ],
    )
    operations = [
        {"line_number": 5, "decision": "remove_caption", "note": "figure label"},
        {"line_number": 7, "decision": "remove", "note": "page number"},
    ]
    assert validate_cleaning_plan(path, operations)["ok"]
    result = apply_cleaning_plan(path, operations)
    assert result.lines == ["## GLAVA 1", "", "Tekst pasusa. " + "x" * 200, "", "", "", ""]
    assert result.removed_captions == [5]
    assert result.removed_artefacts == [7]
    verification = verify_cleaning_plan(path, operations)
    assert verification["ok"], verification["problems"]
    assert verification["removed_captions"] == [5]


def test_a_framed_page_number_line_can_be_removed_and_is_verified(tmp_path):
    path = _chapter(
        tmp_path,
        ["## GLAVA 1", "", "Tekst pasusa. " + "x" * 200, "", "- 2 -", "", "Nastavak."],
    )
    operations = [{"line_number": 5, "decision": "remove", "note": "printed page 2"}]
    assert validate_cleaning_plan(path, operations)["ok"]
    result = apply_cleaning_plan(path, operations)
    assert result.lines[4] == ""
    assert result.removed_artefacts == [5]
    assert result.touched[5]["page_number"] == 2
    verification = verify_cleaning_plan(path, operations)
    assert verification["ok"], verification["problems"]
    assert verification["provenance"]["removed_artefacts"] == [5]


def test_a_framed_page_number_header_can_be_removed_and_is_verified(tmp_path):
    path = _chapter(tmp_path, ["## GLAVA 1", "", "## - 18 -", "", "Tekst."])
    operations = [{"line_number": 3, "decision": "remove", "note": "printed page 18"}]
    assert validate_cleaning_plan(path, operations)["ok"]
    result = apply_cleaning_plan(path, operations)
    assert result.lines[2] == ""
    assert result.removed_artefacts == [3]
    assert result.touched[3]["page_number"] == 18
    assert verify_cleaning_plan(path, operations)["ok"]


def test_removing_a_sentence_that_mentions_a_figure_is_rejected(tmp_path):
    sentence = "Slike 12a i 12b ilustruju različstruktura lič nosti. Slika 12a je rigidna."
    path = _chapter(tmp_path, ["## GLAVA 1", "", sentence])
    result = validate_cleaning_plan(path, [{"line_number": 3, "decision": "remove_caption"}])
    assert not result["ok"]
    assert "does not look like one" in result["issues"][0]

    # ... and neither `remove` (page numbers only) nor `demote` (headers only) can touch it
    assert not validate_cleaning_plan(path, [{"line_number": 3, "decision": "remove"}])["ok"]
    assert not validate_cleaning_plan(path, [{"line_number": 3, "decision": "demote"}])["ok"]


def test_the_corpus_scan_counts_plain_page_numbers_and_captions(sample):
    from book_splitter.documents.chapters import discover_chapters
    from book_splitter.paths import book_chapters_dir
    from book_splitter.preprocess.corpus import scan_corpus_damage

    # The *sample* book's folder, named explicitly: data/output may hold other
    # books (each with its own damage profile), and globbing picks one at random.
    chapters = book_chapters_dir(sample)
    if not (chapters / "_manifest.json").is_file():
        pytest.skip("the sample book has not been split into data/output yet")
    report = scan_corpus_damage(discover_chapters(chapters))
    # the sample book has both kinds of artefact in measurable numbers
    assert report["counts"]["page_number_header"] > 0
    assert report["counts"]["plain_page_number"] > 0
    assert report["counts"]["caption_as_header"] > 0


def test_scan_chapter_damage_reports_the_shapes_without_prose(tmp_path):
    path = _chapter(
        tmp_path,
        [
            "x" * 200 + " i božan",
            "stvena. Po istom pravilu.",
            "",
            "analitička terapija us12 merena, i mišić , ovde.",
        ],
    )
    report = scan_chapter_damage(path)
    kinds = {f["kind"] for f in report["findings"]}
    assert kinds == {SPLIT_WORD, PAGE_NUMBER_IN_WORD, SPACE_BEFORE_PUNCTUATION}
    # the findings carry fragments and line numbers, not paragraphs
    for finding in report["findings"]:
        assert finding["line_number"] >= 1
        assert len(finding["before"]) <= 60
