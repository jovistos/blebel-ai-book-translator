"""The provenance index that makes stage-2 verification linear.

``verify_cleaning_plan`` used to answer "where in the output did source line n
go?" and "is this text still in the output?" by scanning lists once per source
line. :class:`LineProvenance` answers both in O(1) after one build pass; these
tests pin its mapping, its missing-line behaviour and its use by the verifier.
"""

from __future__ import annotations

from pathlib import Path

from book_splitter.preprocess.apply import (
    ApplyResult,
    LineProvenance,
    apply_cleaning_plan,
)
from book_splitter.preprocess.verify import verify_cleaning_plan

LINES = [
    "## GLAVA 1",  # 1
    "",  # 2
    "## 12",  # 3
    "",  # 4
    "Slika 4. Blokada ljubavi.",  # 5
    "",  # 6
    "TEKST KOJI JE NASLOV",  # 7
    "",  # 8
    "Kraj.",  # 9
]
OPERATIONS = [
    {"line_number": 3, "decision": "remove"},
    {"line_number": 5, "decision": "remove_caption"},
    {"line_number": 7, "decision": "promote"},
]


def _chapter(tmp_path: Path) -> Path:
    path = tmp_path / "chapter.md"
    path.write_text("".join(line + "\n" for line in LINES), encoding="utf-8")
    return path


def test_the_index_maps_every_source_line_once(tmp_path: Path) -> None:
    result = apply_cleaning_plan(_chapter(tmp_path), OPERATIONS)
    provenance = LineProvenance.from_result(result)

    assert provenance.unaccounted(len(LINES)) == []
    for source_line in range(1, len(LINES) + 1):
        assert provenance.entry(source_line) is not None
        assert provenance.output_index(source_line) >= 0

    assert provenance.entry(1).origin == "kept"
    assert provenance.entry(3).origin == "removed_page_number"
    assert provenance.entry(5).origin == "removed_caption"
    assert provenance.entry(7).origin == "promoted_heading"


def test_output_index_points_at_the_slot_the_applier_wrote(tmp_path: Path) -> None:
    result = apply_cleaning_plan(_chapter(tmp_path), OPERATIONS)
    provenance = LineProvenance.from_result(result)

    # A removed line keeps its slot, blank, so line numbers stay stable.
    assert provenance.output_index(3) == 2
    assert result.lines[2] == ""
    assert provenance.output_index(7) == 6
    assert result.lines[6] == "## TEKST KOJI JE NASLOV"


def test_has_text_answers_the_verbatim_question(tmp_path: Path) -> None:
    result = apply_cleaning_plan(_chapter(tmp_path), OPERATIONS)
    provenance = LineProvenance.from_result(result)

    assert provenance.has_text("## GLAVA 1")
    assert provenance.has_text("## TEKST KOJI JE NASLOV")
    assert not provenance.has_text("## 12")
    assert not provenance.has_text("Slika 4. Blokada ljubavi.")
    assert provenance.has_text("")


def test_a_missing_entry_is_reported_not_silently_misindexed() -> None:
    # The old entry_line() helper returned -1 for an unaccounted line and the
    # caller indexed output[-1], quietly checking the wrong line.
    result = ApplyResult(lines=["a", ""], removed_artefacts=[3])
    provenance = LineProvenance.from_result(result)

    assert provenance.entry(3) is None
    assert provenance.output_index(3) == -1
    assert provenance.unaccounted(4) == [1, 2, 3, 4]


def test_apply_result_can_build_its_own_index(tmp_path: Path) -> None:
    result = apply_cleaning_plan(_chapter(tmp_path), OPERATIONS)

    assert result.index().output_index(1) == 0
    assert result.index().has_text("Kraj.")


def test_the_verifier_uses_the_index_and_still_passes(tmp_path: Path) -> None:
    path = _chapter(tmp_path)

    verification = verify_cleaning_plan(path, OPERATIONS)

    assert verification["ok"], verification["problems"]
    assert verification["provenance"]["removed_artefacts"] == [3]
