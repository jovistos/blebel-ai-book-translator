"""Tests for the shared reporting primitives and per-stage summaries."""

from __future__ import annotations

import io

import pytest
from book_splitter.agent.result import SplitterResult
from book_splitter.assembly.models import AssemblyResult
from book_splitter.binding.runner import BindResult
from book_splitter.chunk_glossary.models import ChunkGlossaryResult
from book_splitter.chunking.models import ChunkingResult
from book_splitter.conversion.runner import ConversionResult
from book_splitter.core.report import (
    Bar,
    StageSummary,
    human_bytes,
    human_seconds,
    json_text,
    summarize,
    summarize_payload,
)
from book_splitter.dictionary.models import DictionaryResult
from book_splitter.epub.result import EpubResult
from book_splitter.glossary.records import GlossaryResult
from book_splitter.matter_agent.models import MatterResult
from book_splitter.openings.phase import OpeningAuditResult
from book_splitter.postprocess_agent.models import PostprocessResult
from book_splitter.preprocess_agent.models import PreprocessResult
from book_splitter.translation.records import TranslationResult


def test_human_seconds_covers_the_whole_range():
    assert human_seconds(0.0213) == "21 ms"
    assert human_seconds(1.5) == "1.50 s"
    assert human_seconds(389.42) == "6m 29.4s"
    assert human_seconds(3723.4) == "1h 02m 03.4s"
    assert human_seconds(-1) == "0 ms"


def test_human_bytes_and_counts():
    assert human_bytes(411144) == "401.5 KiB"
    assert human_bytes(512) == "512 B"
    assert human_bytes(2 * 1024 * 1024) == "2.0 MiB"


def test_json_text_is_stable_and_unicode_friendly():
    assert json_text({"a": "Ж"}) == '{\n  "a": "Ж"\n}'


def test_generic_summary_uses_to_dict_for_unknown_results():
    class Unknown:
        def to_dict(self):
            return {"ok": False, "summary": "nope", "count": 3}

    summary = summarize(Unknown())
    assert summary.headline == "nope"
    assert summary.failed is True
    assert ("count", "3") in summary.rows


def test_conversion_summary():
    result = ConversionResult(
        source="/tmp/book.epub",
        source_kind="epub",
        markdown_path="/tmp/00-markdown/book.md",
        method="docling",
        pages=10,
        sampled_pages=10,
        text_pages=10,
        images=3,
        image_refs=3,
        dry_run=True,
    )
    summary = summarize(result)
    assert summary.headline == "epub -> Markdown · docling"
    assert summary.status == "dry-run"
    assert ("images", "3 file(s) · 3 reference(s)") in summary.rows
    assert summary.artifacts == (("markdown", "/tmp/00-markdown/book.md"),)


def test_split_summary():
    result = SplitterResult(
        ok=True,
        source_path="/tmp/book.md",
        output_dir="/tmp/01-chapters",
        strategy="markdown_headers",
        chapter_count=2,
        attempts=2,
        usage={"total_tokens": 1234},
        verification={"reassembles_byte_for_byte": True, "output_lines": 6, "source_lines": 6},
        files=[{"filename": "01-a.md", "start_line": 1, "end_line": 3, "bytes": 10}],
    )
    summary = summarize(result)
    assert "2 chapters" in summary.headline
    assert "1,234 tokens" in summary.headline
    assert ("byte-for-byte", "True (6/6 lines)") in summary.rows
    assert summary.detail == ("01-a.md                                              lines 1-3  10 B",)


def test_preprocess_summary():
    result = PreprocessResult(
        ok=True,
        chapters_path="/tmp/01-chapters",
        output_dir="/tmp/02-preprocessed",
        chapter_count=1,
        ok_count=1,
        corpus_scan={"counts": {"page_number_header": 2}},
        openings={"inspected_count": 1, "fixed_count": 0, "remaining_count": 0},
        chapters=[{"filename": "01-a.md", "source_lines": 4, "output_lines": 3, "attempts": 1}],
    )
    summary = summarize(result)
    assert summary.headline == "1/1 chapters clean"
    assert ("damage", "page_number_header=2") in summary.rows
    assert ("openings", "1 checked · 0 fixed · 0 need attention") in summary.rows


def test_dictionary_summary():
    result = DictionaryResult(
        ok=True,
        chapters_path="/tmp/01-chapters",
        output_dir="/tmp/03-reduced",
        unique_dir="/tmp/03-reduced-unique",
        language={"language": "sr", "method": "sample", "confidence": 0.9},
        backend="classla",
        chapter_count=2,
        ok_count=2,
        totals={"sentences": 10, "kept": 4},
        unique_totals={"kept": 3, "nouns": 5, "dropped_duplicate_noun": 1},
        entities={"PER": 2},
        timing={"ner_seconds": 1.5, "run_seconds": 2.0},
    )
    summary = summarize(result)
    assert "2/2 chapters" in summary.headline
    assert ("entities", "PER=2") in summary.rows
    assert ("ner time", "1.50 s") in summary.rows


def test_glossary_summary():
    result = GlossaryResult(
        ok=True,
        chapters_path="/tmp/01-chapters",
        sentences_dir="/tmp/03-reduced-unique",
        output_dir="/tmp/04-glossary",
        chapter_count=2,
        ok_count=2,
        entry_count=7,
        tokens={"total_tokens": 100, "calls": 2},
        provider="deepseek",
        model="deepseek-flash",
    )
    summary = summarize(result)
    assert summary.headline == "7 terms · 2/2 chapters"
    assert ("llm", "2 call(s) · 100 tokens") in summary.rows


def test_chunking_summary():
    result = ChunkingResult(
        ok=True,
        chapters_path="/tmp/01-chapters",
        output_dir="/tmp/05-chunks",
        chapter_count=1,
        ok_count=1,
        chunk_count=3,
        settings={"tokenizer": {"name": "tiktoken"}},
        verification={
            "chunks_are_source_slices": True,
            "book_reconstructs_byte_for_byte": True,
            "chunk_size_violations": 0,
        },
    )
    summary = summarize(result)
    assert summary.headline == "3 chunks · 1 chapters"
    assert ("tokenizer", "tiktoken") in summary.rows


def test_chunk_glossary_summary():
    result = ChunkGlossaryResult(
        ok=True,
        chunks_path="/tmp/05-chunks",
        glossary_path="/tmp/04-glossary/glossary.json",
        output_dir="/tmp/06-chunk-glossary",
        chunk_count=2,
        entry_count=7,
        entity_count=5,
        term_count=4,
        matched_term_count=3,
        unmatched_term_count=1,
        matched_chunk_count=2,
        language={"language": "sr", "method": "sample"},
        backend="classla",
    )
    summary = summarize(result)
    assert summary.headline == "3/4 terms matched · 2 chunks"


def test_matter_summary():
    result = MatterResult(
        ok=True,
        chapters_path="/tmp/01-chapters",
        chunks_path="/tmp/05-chunks",
        output_dir="/tmp/07-translations",
        prompts_dir="/tmp/06b-matter-prompts",
        matter_path="/tmp/06b-matter",
        section_count=2,
        chunk_count=3,
        ok_count=3,
        book_title="Naslov",
        book_author="Autor",
        book_plan={"title": "Naslov"},
        totals={"omitted": 1},
        tokens={"total_tokens": 50},
    )
    summary = summarize(result)
    assert summary.headline == "3/3 matter chunks localized"
    assert ("localized title", "Naslov") in summary.rows


def test_translation_summary():
    result = TranslationResult(
        ok=True,
        chunks_path="/tmp/05-chunks",
        chunk_glossary_path="/tmp/06-chunk-glossary",
        output_dir="/tmp/07-translations",
        prompts_dir="/tmp/07-translation-prompts",
        chunk_count=2,
        ok_count=2,
        translator="deepseek",
        provider="deepseek",
        model="deepseek-flash",
        tokens={"total_tokens": 100},
    )
    summary = summarize(result)
    assert summary.headline == "2/2 chunks translated"
    assert ("model", "deepseek · deepseek/deepseek-flash") in summary.rows


def test_assembly_summary():
    result = AssemblyResult(
        ok=True,
        chunks_path="/tmp/05-chunks",
        translations_path="/tmp/07-translations",
        output_dir="/tmp/08-translated-chapters",
        chapter_count=2,
        ok_count=2,
        totals={"translated_chunks": 4, "chunks": 4},
    )
    summary = summarize(result)
    assert summary.headline == "2/2 chapters written"
    assert ("chunks", "4 translated of 4 in the window") in summary.rows


def test_postprocess_summary():
    result = PostprocessResult(
        ok=True,
        chapters_path="/tmp/08-translated-chapters",
        output_dir="/tmp/09-postprocessed-chapters",
        chapter_count=2,
        ok_count=2,
        book_plan={"title": "Naslov"},
        validation={"ok": True},
        toc={"ok": True},
        usage={"total_tokens": 30},
    )
    summary = summarize(result)
    assert summary.headline == "2/2 chapters postprocessed"
    assert ("validation", "ok") in summary.rows


def test_bind_summary():
    result = BindResult(
        ok=True,
        chapters_path="/tmp/09-postprocessed-chapters",
        output_path="/tmp/10-translated-book.md",
        chapter_count=3,
        toc_included=True,
        chars=1000,
        lines=50,
    )
    summary = summarize(result)
    assert summary.headline == "3 files · 1,000 characters"


def test_epub_summary():
    result = EpubResult(
        ok=True,
        chapters_path="/tmp/09-postprocessed-chapters",
        output_path="/tmp/11-translated-book.epub",
        title="Naslov",
        author="Autor",
        language="sr",
        identifier="urn:uuid:1",
        document_count=3,
        image_count=2,
        chars=1000,
        bytes=2048,
        verification={
            "mimetype_first": True,
            "container_ok": True,
            "opf_ok": True,
            "nav_ok": True,
            "xhtml_valid": 3,
            "documents": 3,
        },
    )
    summary = summarize(result)
    assert summary.headline == "3 documents · 2,048 bytes"
    assert ("verification", "mimetype ok · container ok · package ok · navigation ok · 3/3 XHTML") in summary.rows


def test_openings_summary():
    result = OpeningAuditResult(
        folder="/tmp/01-chapters",
        chapter_count=3,
        inspected_count=3,
        clean_count=2,
        issue_count=1,
        fixable_count=1,
        fixed_count=1,
        remaining_count=0,
    )
    summary = summarize(result)
    assert summary.headline == "3 opening(s) · 1 fixed · 0 need attention"
    assert ("folder", "/tmp/01-chapters") in summary.rows


def test_payload_summaries_for_recon_and_scan():
    recon = summarize_payload(
        "recon",
        {
            "source": "/tmp/book.md",
            "line_count": 100,
            "header_count": 8,
            "header_assessment": {"verdict": "noisy"},
            "candidate_count": 4,
            "printed_toc_block": {"found": True},
        },
    )
    assert recon.headline == "100 lines · 8 headers (noisy) · 4 candidates"
    scan = summarize_payload(
        "scan",
        {"chapter_count": 2, "total_findings": 3, "counts": {"page_number_header": 3}},
    )
    assert scan.headline == "3 findings in 1 shape(s)"


def test_render_puts_rows_detail_and_artifacts_in_order():
    summary = StageSummary(
        headline="x",
        rows=(("chapters", "1/1"),),
        detail=("01-a.md",),
        warnings=("careful",),
        errors=("boom",),
        artifacts=(("manifest", "/tmp/m.json"),),
        next_steps=("next: run chunk",),
    )
    assert summary.render(debug=False) == [
        "  chapters        : 1/1",
        "  warning         : careful",
        "  error           : boom",
        "  manifest        : /tmp/m.json",
        "  next: run chunk",
    ]
    assert summary.render(debug=True)[1] == "    01-a.md"


def test_bar_piped_prints_a_durable_line_and_never_carriages_returns():
    stream = io.StringIO()
    bar = Bar(total=2, stream=stream, tty=False)
    bar.start({"filename": "01-a.md"})
    bar.finish({"ok": True, "filename": "01-a.md", "tokens": {"total_tokens": 10}}, 1, 2)
    bar.finish({"ok": True, "filename": "02-b.md", "tokens": {"total_tokens": 5}}, 2, 2)
    bar.close()
    text = stream.getvalue()
    assert "\r" not in text
    assert "[  1/2] ok    01-a.md" in text
    assert "10 tok" in text
    assert "[  2/2] ok    02-b.md" in text


def test_bar_tty_redraws_and_clears():
    stream = io.StringIO()
    bar = Bar(total=1, stream=stream, tty=True)
    bar.start({"filename": "01-a.md"})
    bar.finish({"ok": True, "filename": "01-a.md"}, 1, 1)
    bar.close()
    text = stream.getvalue()
    assert "\r" in text
    assert "01-a.md" in text


def test_bar_accepts_custom_formatters():
    stream = io.StringIO()
    bar = Bar(
        total=1,
        stream=stream,
        tty=False,
        format_record=lambda record, done, total: f"custom {done}/{total} {record['x']}",
        format_line=lambda state: "custom line",
    )
    bar.start({"x": 1})
    assert bar.finish({"x": 7}, 1, 1) == "custom 1/1 7"
    assert bar.line() == "custom line"
    bar.close()


def test_a_broken_custom_formatter_is_not_hidden():
    bar = Bar(total=1, stream=io.StringIO(), tty=False, format_record=lambda *_: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        bar.finish({}, 1, 1)
