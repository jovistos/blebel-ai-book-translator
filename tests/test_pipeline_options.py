"""Tests for phase selection and the exact runner call each stage makes."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pytest
from book_splitter.pipeline.options import (
    PHASES,
    PhaseSelection,
    PipelineOptions,
    StageOptions,
    StageRuntime,
)
from book_splitter.pipeline.paths import StagePaths
from book_splitter.pipeline.sources import StageSources


def base_ns(**overrides) -> argparse.Namespace:
    values = dict(  # noqa: C408 - keeping the keyword list readable
        book="book.md",
        in_dir=None,
        output_root=None,
        start_at="recon",
        stop_after="epub",
        dry_run=False,
        json=False,
        quiet=False,
        debug=False,
        no_log=False,
        log_file=None,
        out=None,
        chunk_size=1000,
        chunk_tokenizer="cl100k_base",
        first_chapters=None,
        translate_first_chapters=None,
        limit=None,
        max_attempts=4,
        min_chapters=3,
        front_matter_lines=200,
        recursion_limit=60,
        no_front_matter=False,
        workers=1,
        language="auto",
        gpu=None,
        cpu=False,
        min_words=4,
        no_uppercase_filter=False,
        keep_types="PER, LOC",
        drop_temporal_only=False,
        no_save_reduced=False,
        no_save_unique=False,
        unique_only=False,
        examples=5,
        threshold=None,
        match_threshold=95.0,
        scorer="ratio",
        candidates=3,
        min_term_chars=2,
        sentences="unique",
        sentences_in=None,
        source_language="en",
        target_language="sr",
        book_title="Book",
        max_context_entries=0,
        include_front_matter=False,
        include_back_matter=False,
        extraction_method="auto",
        max_tokens=None,
        max_chapter_chars=24000,
        matter_limit=None,
        translate_limit=None,
        glossary_scope="chunk",
        method="auto",
        context_chars=0,
        include_matter=False,
        require_web_search=True,
        epub_title=None,
        epub_author=None,
        epub_language=None,
        identifier=None,
        description=None,
        publisher=None,
        cover=None,
        no_manifest=False,
        no_prompts=False,
        skip_if_present=False,
        with_translate=False,
        with_matter=False,
        with_postprocess=False,
        no_preprocess=False,
        no_dictionary=False,
        no_glossary=False,
        no_chunk=False,
        no_chunk_glossary=False,
        no_translate=False,
        no_matter=False,
        no_concat=False,
        no_postprocess=False,
        no_bind=False,
        no_epub=False,
        map=None,
        images=True,
        force_convert=False,
        llm=[],
        llm_config=None,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.fixture
def options() -> PipelineOptions:
    return PipelineOptions.from_namespace(base_ns())


@pytest.fixture
def runtime(tmp_path: Path) -> StageRuntime:
    source = tmp_path / "book.md"
    source.write_text("# Book\n", encoding="utf-8")
    paths = StagePaths.for_pipeline(
        base_ns(), source=source, output_root=tmp_path / "out"
    )
    sources = StageSources(
        chapters=paths.chapters_dir,
        text=paths.chapters_dir,
        text_label="raw chapters",
        chunk_input=paths.chapters_dir,
        chunks=paths.chunks_dir,
        chunk_glossary=paths.chunk_glossary_dir,
        glossary=paths.glossary_dir,
        translations=paths.translations_dir,
        translated_chapters=paths.translated_chapters_dir,
        postprocess=paths.postprocess_dir,
    )
    return StageRuntime(
        paths=paths,
        sources=sources,
        source=source,
        original_source=source,
        clock=time.perf_counter,
    )


# --------------------------------------------------------------------------- #
# phase selection
# --------------------------------------------------------------------------- #
def test_default_selection_for_a_markdown_book():
    selection = PhaseSelection.from_namespace(base_ns())
    assert selection.selected(convert_needed=False) == (
        "recon",
        "split",
        "scan",
        "preprocess",
        "dictionary",
        "chunk",
        "chunk_glossary",
    )
    assert "matter" in selection.switched_off(convert_needed=False)
    assert "translate" in selection.switched_off(convert_needed=False)


def test_with_translate_turns_on_matter_translate_assembly_and_book_outputs():
    selection = PhaseSelection.from_namespace(base_ns(with_translate=True))
    selected = selection.selected(convert_needed=False)
    assert selected[-5:] == (
        "matter",
        "translate",
        "concatenate",
        "bind",
        "epub",
    )
    assert "postprocess" not in selected


def test_with_postprocess_is_added_after_concatenate():
    selection = PhaseSelection.from_namespace(
        base_ns(with_translate=True, with_postprocess=True)
    )
    selected = selection.selected(convert_needed=False)
    assert selected[-3:] == ("postprocess", "bind", "epub")


def test_no_chunk_drops_everything_that_needs_chunks():
    selection = PhaseSelection.from_namespace(
        base_ns(with_translate=True, no_chunk=True)
    )
    selected = selection.selected(convert_needed=False)
    assert "chunk" not in selected
    assert "chunk_glossary" not in selected
    assert "matter" not in selected
    assert "translate" not in selected
    assert "concatenate" not in selected


def test_no_preprocess_and_no_dictionary_drop_their_phases():
    selection = PhaseSelection.from_namespace(
        base_ns(no_preprocess=True, no_dictionary=True)
    )
    selected = selection.selected(convert_needed=False)
    assert "preprocess" not in selected
    assert "dictionary" not in selected


def test_start_and_stop_narrow_the_window():
    selection = PhaseSelection.from_namespace(
        base_ns(start_at="chunk_glossary", stop_after="chunk_glossary")
    )
    assert selection.selected(convert_needed=False) == ("chunk_glossary",)


def test_start_at_matter_selects_only_matter():
    selection = PhaseSelection.from_namespace(
        base_ns(start_at="matter", with_matter=True)
    )
    assert selection.selected(convert_needed=False) == ("matter",)


def test_start_at_concatenate_also_binds_and_packages():
    selection = PhaseSelection.from_namespace(base_ns(start_at="concatenate"))
    assert selection.selected(convert_needed=False) == (
        "concatenate",
        "bind",
        "epub",
    )


def test_no_bind_also_drops_epub_and_no_epub_drops_only_epub():
    no_bind = PhaseSelection.from_namespace(base_ns(with_translate=True, no_bind=True))
    assert "bind" not in no_bind.selected(convert_needed=False)
    assert "epub" not in no_bind.selected(convert_needed=False)
    no_epub = PhaseSelection.from_namespace(base_ns(with_translate=True, no_epub=True))
    assert "bind" in no_epub.selected(convert_needed=False)
    assert "epub" not in no_epub.selected(convert_needed=False)


def test_convert_is_kept_only_for_a_non_markdown_source():
    selection = PhaseSelection.from_namespace(base_ns(start_at="convert"))
    assert "convert" in selection.selected(convert_needed=True)
    assert "convert" not in selection.selected(convert_needed=False)


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def test_validation_reports_usage_errors():
    options = PipelineOptions.from_namespace(
        base_ns(
            start_at="epub",
            stop_after="recon",
            chunk_size=0,
            first_chapters=-1,
            llm=["nope.model=x"],
        )
    )
    errors = options.validate()
    assert any("comes after" in error for error in errors)
    assert any("chunk-size" in error for error in errors)
    assert any("first-chapters" in error for error in errors)
    assert any("nope" in error for error in errors)


def test_validation_accepts_a_normal_run(options):
    assert options.validate() == []


# --------------------------------------------------------------------------- #
# exact runner calls
# --------------------------------------------------------------------------- #
def test_split_call(options, runtime):
    call = options.stage.call("split", runtime)
    assert call.positional == (runtime.source, runtime.paths.chapters_dir)
    assert call.kwargs == {
        "settings": None,
        "front_matter_lines": 200,
        "max_attempts": 4,
        "min_chapters": 3,
        "include_front_matter": True,
        "dry_run": False,
        "recursion_limit": 60,
        "skip_if_present": False,
    }


def test_split_map_call(options, runtime):
    mapping = [{"line_number": 1, "filename": "01-a.md", "title": "A"}]
    call = options.stage.call("split_map", runtime, mapping=mapping)
    assert call.positional == (runtime.source, runtime.paths.chapters_dir, mapping)
    assert call.kwargs == {
        "min_chapters": 3,
        "include_front_matter": True,
        "dry_run": False,
        "skip_if_present": False,
    }


def test_dictionary_call_uses_the_text_source_and_normalized_keep_types(
    options, runtime
):
    call = options.stage.call("dictionary", runtime)
    assert call.positional == (runtime.sources.text, runtime.paths.reduced_dir)
    assert call.kwargs["unique_output_dir"] == str(runtime.paths.unique_dir)
    assert call.kwargs["keep_types"] == ("PER", "LOC")
    assert call.kwargs["write_manifest"] is True
    assert call.kwargs["dry_run"] is False


def test_preprocess_call_drops_manifest_and_prompts_on_a_dry_run(runtime):
    options = PipelineOptions.from_namespace(base_ns(dry_run=True))
    call = options.stage.call("preprocess", runtime)
    assert call.kwargs["write_manifest"] is False
    assert call.kwargs["dry_run"] is True
    assert call.kwargs["workers"] == 1


def test_chunk_call_uses_the_chunk_input(options, runtime):
    call = options.stage.call("chunk", runtime)
    assert call.positional == (runtime.sources.chunk_input, runtime.paths.chunks_dir)
    assert call.kwargs["chunk_size"] == 1000
    assert call.kwargs["tokenizer"] == "cl100k_base"
    assert call.kwargs["limit"] is None


def test_chunk_glossary_call_carries_glossary_and_window(options, runtime):
    call = options.stage.call("chunk_glossary", runtime)
    assert call.positional == (runtime.sources.chunks, runtime.paths.chunk_glossary_dir)
    assert call.kwargs["glossary_path"] == str(runtime.sources.glossary)
    assert call.kwargs["threshold"] == 95.0
    assert call.kwargs["scorer"] == "ratio"
    assert call.kwargs["chapter_files"] is None


def test_matter_call_passes_three_positional_paths(options, runtime):
    call = options.stage.call("matter", runtime)
    assert call.positional == (
        runtime.sources.chapters,
        runtime.sources.chunks,
        runtime.paths.translations_dir,
    )
    assert call.kwargs["prompts_dir"] == str(runtime.paths.matter_prompts_dir)
    assert call.kwargs["matter_path"] == str(runtime.paths.matter_dir)
    assert call.kwargs["resume"] is False


def test_translate_call_uses_the_pipeline_window(options, runtime):
    options = PipelineOptions.from_namespace(
        base_ns(translate_limit=3, first_chapters=2, translate_first_chapters=5)
    )
    call = options.stage.call("translate", runtime)
    assert call.kwargs["limit"] == 3
    assert call.kwargs["chapter_files"] is None
    assert call.kwargs["glossary_scope"] == "chunk"
    assert call.kwargs["resume"] is False


def test_postprocess_call_uses_first_chapters_not_the_file_count(options, runtime):
    options = PipelineOptions.from_namespace(
        base_ns(first_chapters=4, limit=99)
    )
    call = options.stage.call("postprocess", runtime)
    assert call.kwargs["limit"] == 4


def test_concatenate_call_uses_translate_first_chapters(options, runtime):
    options = PipelineOptions.from_namespace(
        base_ns(first_chapters=2, translate_first_chapters=5)
    )
    call = options.stage.call("concatenate", runtime)
    assert call.kwargs["first_chapters"] == 5


def test_bind_and_epub_calls_use_the_book_paths(options, runtime):
    bind = options.stage.call("bind", runtime)
    assert bind.positional == (
        runtime.sources.postprocess,
        runtime.paths.translated_book_path,
    )
    epub = options.stage.call("epub", runtime)
    assert epub.positional == (runtime.sources.postprocess, runtime.paths.epub_path)
    assert epub.kwargs["title"] is None
    assert epub.kwargs["cover"] is None


def test_epub_aliases_are_normalized_from_the_stage_flags(runtime):
    ns = base_ns(
        title="Naslov",
        author="Autor",
        language="sr",
        identifier="urn:uuid:1",
        description="Opis",
        publisher="Izdavač",
        cover="cover.jpg",
    )
    options = PipelineOptions.from_namespace(ns)
    call = options.stage.call("epub", runtime)
    assert call.kwargs["title"] == "Naslov"
    assert call.kwargs["author"] == "Autor"
    assert call.kwargs["language"] == "sr"
    assert call.kwargs["identifier"] == "urn:uuid:1"
    assert call.kwargs["description"] == "Opis"
    assert call.kwargs["publisher"] == "Izdavač"
    assert call.kwargs["cover"] == "cover.jpg"


def test_threshold_and_method_aliases(runtime):
    ns = base_ns(threshold=80.0, method="json")
    options = PipelineOptions.from_namespace(ns)
    assert options.stage.call("chunk_glossary", runtime).kwargs["threshold"] == 80.0
    assert options.stage.call("translate", runtime).kwargs["method"] == "json"


def test_every_phase_builds_a_call(options, runtime):
    for stage in (
        "convert",
        "split",
        "dictionary",
        "glossary",
        "preprocess",
        "chunk",
        "chunk_glossary",
        "matter",
        "translate",
        "concatenate",
        "postprocess",
        "bind",
        "epub",
    ):
        call = options.stage.call(stage, runtime)
        assert call.stage == stage
        assert callable(call.runner)
        assert isinstance(call.kwargs, dict)
    assert len(PHASES) == 14


def test_stage_mode_translation_limit_uses_limit_not_translate_limit(runtime):
    ns = base_ns(limit=2, translate_limit=9)
    options = PipelineOptions.from_namespace(ns)
    stage = StageOptions.from_namespace(ns, mode="stage")
    assert stage.translation_limit() == 2
    assert options.stage.translation_limit() == 9

def test_pipeline_translate_flags_are_normalized(runtime):
    ns = base_ns(
        translate_method="json",
        translate_max_tokens=123,
        translate_context_chars=45,
        translate_glossary_scope="book",
        translate_attempts=2,
    )
    options = PipelineOptions.from_namespace(ns)
    call = options.stage.call("translate", runtime)
    assert call.kwargs["method"] == "json"
    assert call.kwargs["max_tokens"] == 123
    assert call.kwargs["context_chars"] == 45
    assert call.kwargs["glossary_scope"] == "book"
    assert call.kwargs["max_attempts"] == 2
