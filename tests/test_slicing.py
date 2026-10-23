"""Tests for planning, verbatim slicing and verification."""

from __future__ import annotations

import pytest
from book_translator.documents.io import load_lines
from book_translator.documents.text import normalize_filename
from book_translator.stages.slicing.plan import plan_files
from book_translator.stages.slicing.slice import preview_split, slice_book
from book_translator.stages.slicing.verify import verify_output

from _fixtures import PLAIN, SAMPLE, SAMPLE_SPLITS, requires_sample

# --------------------------------------------------------------------------- #
# Filenames
# --------------------------------------------------------------------------- #


def test_normalize_filename_is_path_safe():
    assert normalize_filename("../../etc/passwd") == "passwd.md"
    assert "/" not in normalize_filename("a/b/c.md")
    assert normalize_filename("") == "section.md"
    assert normalize_filename("07-GLAVA 3: U SRCU SMO") == "07-glava-3-u-srcu-smo.md"


@requires_sample
def test_filename_typo_is_reconciled_but_descriptive_names_survive(plan):
    mapping = plan([(517, "15-belestke.md"), (136, "05-glava-2-seks-i-srce.md"),
                    (245, "08-giava-5-strah-od-ljubavi.md")])
    names = [f["filename"] for f in plan_files(load_lines(str(SAMPLE)), mapping)]
    assert "15-beleske.md" in names  # model typo repaired from the heading
    assert "05-glava-2-seks-i-srce.md" in names  # two-word slug: left alone
    assert "08-giava-5-strah-od-ljubavi.md" in names  # descriptive: left alone


# --------------------------------------------------------------------------- #
# Slicing / verification
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("source", [PLAIN, SAMPLE])
def test_slice_and_verify_is_byte_for_byte(source, tmp_path, plan):
    if source == SAMPLE and not SAMPLE.is_file():
        pytest.skip("sample book not present")
    lines = load_lines(str(source))
    splits = [12, 18, 28, 38, 44] if source == PLAIN else SAMPLE_SPLITS
    mapping = plan([(ln, f"{i:02d}-part.md") for i, ln in enumerate(splits, start=1)])
    manifest = slice_book(str(source), mapping, str(tmp_path), include_front_matter=True)

    assert manifest["total_lines"] == len(lines)
    report = verify_output(manifest)
    assert report["ok"] and report["reassembles_byte_for_byte"]
    assert report["output_lines"] == report["source_lines"] == len(lines)
    assert report["line_ranges_contiguous"]
    assert not report["missing_files"] and not report["problems"]

    # Reassembled bytes must equal the source file exactly.
    blob = b"".join((tmp_path / f["filename"]).read_bytes() for f in manifest["files"])
    assert blob == source.read_bytes()


def test_preview_matches_what_slicing_writes(plain, tmp_path, plan):
    mapping = plan([(12, "01-a.md"), (18, "02-b.md"), (28, "03-c.md"), (38, "04-d.md")])
    preview = preview_split(str(plain), mapping, str(tmp_path))
    manifest = slice_book(str(plain), mapping, str(tmp_path), include_front_matter=True)
    assert [(f["filename"], f["start_line"], f["end_line"]) for f in preview["files"]] == [
        (f["filename"], f["start_line"], f["end_line"]) for f in manifest["files"]
    ]
    assert preview["file_count"] == manifest["file_count"]
    assert preview["total_lines"] == manifest["total_lines"]


def test_slice_excludes_front_matter_on_request(plain, tmp_path, plan):
    mapping = plan([(12, "01-a.md"), (18, "02-b.md"), (28, "03-c.md"), (38, "04-d.md")])
    manifest = slice_book(str(plain), mapping, str(tmp_path), include_front_matter=False)
    assert all(f["filename"] != "00-front-matter.md" for f in manifest["files"])
    assert manifest["files"][0]["start_line"] == 12


def test_slice_rejects_duplicate_filenames(plain, tmp_path, plan):
    mapping = plan([(12, "same.md"), (18, "same.md"), (28, "same.md"), (38, "same.md")])
    manifest = slice_book(str(plain), mapping, str(tmp_path))
    names = [f["filename"] for f in manifest["files"]]
    chapters = [n for n in names if n != "00-front-matter.md"]
    assert len(names) == len(set(names)) == 5  # 4 chapters + front matter
    assert chapters == ["same.md", "same-18.md", "same-28.md", "same-38.md"]


def test_slice_writes_manifest_next_to_chapters(plain, tmp_path, plan):
    mapping = plan([(12, "01-a.md"), (18, "02-b.md"), (28, "03-c.md")])
    slice_book(str(plain), mapping, str(tmp_path))
    assert (tmp_path / "_manifest.json").is_file()

    from book_translator.stages.slicing.verify import verify_output_dir

    assert verify_output_dir(tmp_path)["ok"]
