"""Stage 2's manifest is a chapter manifest, not a private progress log.

``02-preprocessed/`` has no stage-1 ``_manifest.json``, so the rest of the
pipeline reads it through the same ``load_split_manifest``/``discover_chapters``
pair. That only works while the stage-2 record keeps the fields those readers
need: filename and order, title, source line range, and both hashes.
"""

from __future__ import annotations

import json
from pathlib import Path

from book_translator.documents.chapters import discover_chapters, load_split_manifest
from book_translator.stages.preprocess_agent.runner import (
    ChapterResult,
    PreprocessResult,
    write_preprocess_manifest,
)


def test_the_cleaned_manifest_reads_back_like_a_stage_one_manifest(tmp_path: Path):
    raw = tmp_path / "01-chapters"
    raw.mkdir()
    source = raw / "01-chapter.md"
    source.write_text("## 18\n\nMarko je ušao.\n", encoding="utf-8")
    (raw / "_manifest.json").write_text(
        json.dumps(
            {
                "source_file": "book.md",
                "strategy": "markdown_headers",
                "files": [
                    {
                        "filename": "01-chapter.md",
                        "title": "Prva glava",
                        "start_line": 3,
                        "end_line": 5,
                        "lines": 3,
                        "sha256": "source-hash",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "02-preprocessed"
    out.mkdir()
    (out / "01-chapter.md").write_text("Marko je ušao.\n", encoding="utf-8")

    chapters = discover_chapters(raw)
    record = ChapterResult(
        chapter_id="01-chapter",
        filename="01-chapter.md",
        source_path=str(source),
        output_path=str(out / "01-chapter.md"),
        ok=True,
        source_lines=3,
        output_lines=1,
        verification={"ok": True},
    )
    result = PreprocessResult(
        ok=True,
        chapters_path=str(raw),
        output_dir=str(out),
        chapter_count=1,
        ok_count=1,
        chapters=[record.to_dict()],
    )
    write_preprocess_manifest(out, result, chapters, [record])

    manifest = load_split_manifest(out)
    assert manifest["stage"] == "preprocess"
    assert manifest["source_file"] == "book.md"
    assert manifest["source_path"] == "book.md"
    [entry] = manifest["files"]
    assert entry["title"] == "Prva glava"
    assert entry["source_sha256"] == "source-hash"
    assert entry["sha256"]  # the cleaned file's own hash
    assert entry["bytes"] > 0

    cleaned = discover_chapters(out)
    assert [chapter.filename for chapter in cleaned] == ["01-chapter.md"]
    assert cleaned[0].title == "Prva glava"
    assert cleaned[0].start_line == 3
    assert cleaned[0].line_count == 1
