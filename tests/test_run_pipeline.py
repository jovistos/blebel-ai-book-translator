"""Offline tests for ``scripts/run_pipeline.py`` and the progress hook it drives.

No network, no API key: both agent runs are replaced by stubs, so these tests pin
the *narration and the control flow* — phase order, what stops on failure, what
the log keeps that the console does not — not the agents themselves.
"""

from __future__ import annotations

import json
from pathlib import Path

import book_translator.stages.agent
import book_translator.stages.dictionary
import book_translator.stages.preprocess_agent
import pytest
from book_translator.core.errors import MissingDependencyError
from book_translator.pipeline.driver import PipelineDriver
from book_translator.pipeline.options import PipelineOptions
from book_translator.stages.agent.result import SplitterResult
from book_translator.stages.dictionary.runner import DictionaryResult
from book_translator.stages.preprocess_agent import runner as preprocess_runner
from book_translator.stages.preprocess_agent.runner import ChapterResult, PreprocessResult

from _fixtures import PLAIN, SAMPLE, requires_sample

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def script():
    """The pipeline parser module shared by the CLI and the compatibility script."""
    from book_translator.pipeline import parser

    return parser


def install_chunk_glossary(monkeypatch, *, ok: bool = True, files: bool = False):
    """Replace stage 6, so no test loads a tagger or reads a real glossary."""
    from book_translator.stages.chunk_glossary import ChunkGlossaryResult

    calls: list[Path] = []

    def fake_run_chunk_glossary(chunks_path, output_dir=None, **kwargs):
        calls.append(Path(chunks_path))
        out = Path(output_dir or chunks_path)
        if output_dir:
            out.mkdir(parents=True, exist_ok=True)
            for name in ("_chunk_glossary_manifest.json", "_chunk_glossary_trace.json"):
                (out / name).write_text("{}", encoding="utf-8")
            if files:
                (out / "00000-01-a-0000.md.json").write_text(
                    '{"matched_terms": []}', encoding="utf-8"
                )
        return ChunkGlossaryResult(
            ok=ok,
            chunks_path=str(chunks_path),
            glossary_path=str(out.parent / "04-glossary" / "glossary.json"),
            output_dir=str(out),
            chunk_count=2,
            ok_count=2 if ok else 0,
            processed_count=2,
            entry_count=7,
            entity_count=5,
            term_count=4,
            matched_term_count=3 if ok else 0,
            unmatched_term_count=1 if ok else 0,
            matched_chunk_count=2 if ok else 0,
            settings={"threshold": 95.0, "scorer": "ratio"},
            totals={
                "entity_mentions": 5,
                "terms": 4,
                "matched_terms": 3 if ok else 0,
                "unmatched_terms": 1 if ok else 0,
                "matched_chunks": 2 if ok else 0,
                "unique_matched_glossary_terms": 2 if ok else 0,
            },
            phases=[
                {"name": "ner", "chunks": 2, "seconds": 0.1},
                {"name": "match", "chunks": 2, "seconds": 0.1},
            ],
            timing={"run_seconds": 0.2, "ner_phase_seconds": 0.1, "match_phase_seconds": 0.1},
            chunks=[],
            errors=[] if ok else ["chunk: boom"],
        )

    monkeypatch.setattr("book_translator.stages.chunk_glossary.run_chunk_glossary", fake_run_chunk_glossary)
    return calls


def install_matter(monkeypatch, *, ok: bool = True):
    """Replace the front/back matter agent, so no test calls a paid model."""
    from book_translator.stages.matter.inspect import discover_matter_sections
    from book_translator.stages.matter_agent import MatterResult

    calls: list[dict] = []

    def fake_run_matter(chapters_path, chunks_path=None, output_dir=None, **kwargs):
        calls.append(
            {
                "chapters_path": str(chapters_path),
                "chunks_path": str(chunks_path or ""),
                "output_dir": output_dir,
                **kwargs,
            }
        )
        out = Path(output_dir or chunks_path or chapters_path)
        out.mkdir(parents=True, exist_ok=True)
        sections = discover_matter_sections(chapters_path, chunks_path)
        written = 0
        for section in sections:
            for filename in section.chunk_filenames:
                (out / filename).write_text(
                    "# Naslov\n\n*Translated by BleBel*\n\nPREVEDENO\n",
                    encoding="utf-8",
                )
                written += 1
        prompts = Path(kwargs.get("prompts_dir") or out.parent / "06b-matter-prompts")
        prompts.mkdir(parents=True, exist_ok=True)
        (prompts / "one.json").write_text("{}", encoding="utf-8")
        stage = Path(kwargs.get("matter_path") or out.parent / "06b-matter")
        stage.mkdir(parents=True, exist_ok=True)
        (stage / "book-plan.json").write_text("{}", encoding="utf-8")
        manifest = stage / "_matter_manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        trace = stage / "_matter_trace.json"
        trace.write_text("{}", encoding="utf-8")
        return MatterResult(
            ok=ok,
            chapters_path=str(chapters_path),
            chunks_path=str(chunks_path or ""),
            output_dir=str(out),
            prompts_dir=str(prompts),
            matter_path=str(stage),
            source_language=str(kwargs.get("source_language") or "en"),
            target_language=str(kwargs.get("target_language") or "sr"),
            section_count=len(sections),
            chunk_count=written,
            ok_count=written if ok else 0,
            processed_count=written,
            skipped=not sections,
            book_plan={
                "translated_title": "Naslov",
                "translation_credit": "Translated by BleBel",
                "omit": [],
            },
            validation={"ok": True, "issues": [], "warnings": []},
            tokens={"total_tokens": 100, "llm_calls": 1},
            totals={"matter_chunks": written, "ok": written, "omitted": 0},
            timing={"run_seconds": 0.1, "seconds_per_chunk": 0.1},
            sections=[section.to_dict() for section in sections],
            chunks=[],
            errors=[] if ok else ["00-front-matter-0000.md: boom"],
            manifest_path=str(manifest),
            trace_path=str(trace),
        )

    monkeypatch.setattr("book_translator.stages.matter_agent.run_matter", fake_run_matter)
    return calls


def install_translate(monkeypatch, *, ok: bool = True):
    """Replace stage 7, so no test ever calls a paid model."""
    from book_translator.stages.translation import TranslationResult

    calls: list[dict] = []

    def fake_run_translation(chunks_path, output_dir=None, **kwargs):
        calls.append({"chunks_path": str(chunks_path), "output_dir": output_dir, **kwargs})
        out = Path(output_dir or chunks_path)
        prompts = Path(kwargs.get("prompts_dir") or out.parent / "07-translation-prompts")
        manifest = out / "_translation_manifest.json"
        trace = out / "_translation_trace.json"
        dry_run = bool(kwargs.get("dry_run"))
        if not dry_run:
            out.mkdir(parents=True, exist_ok=True)
            prompts.mkdir(parents=True, exist_ok=True)
            (out / "00000-01-a-0000.md").write_text("PREVEDENO\n", encoding="utf-8")
            (prompts / "00000-01-a-0000.md.json").write_text("{}", encoding="utf-8")
            manifest.write_text("{}", encoding="utf-8")
            trace.write_text("{}", encoding="utf-8")
        return TranslationResult(
            ok=ok,
            chunks_path=str(chunks_path),
            chunk_glossary_path=str(kwargs.get("chunk_glossary_path") or ""),
            output_dir=str(out),
            prompts_dir=str(prompts),
            source_language=str(kwargs.get("source_language") or "en"),
            target_language=str(kwargs.get("target_language") or "sr"),
            translator="fake/translator",
            provider="fake",
            model="fake-1",
            glossary_scope=str(kwargs.get("glossary_scope") or "chunk"),
            chunk_count=1,
            ok_count=1 if ok else 0,
            processed_count=1,
            dry_run=dry_run,
            tokens={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "calls": 1,
                "chunks": 1,
            },
            totals={
                "translated_chunks": 1 if ok else 0,
                "source_chars": 100,
                "translation_chars": 90,
                "failed_chunks": 0 if ok else 1,
            },
            timing={"run_seconds": 0.1, "model_seconds": 0.1, "seconds_per_chunk": 0.1},
            chunks=[],
            errors=[] if ok else ["00000-01-a-0000.md: boom"],
            manifest_path=None if dry_run else str(manifest),
            trace_path=None if dry_run else str(trace),
        )

    monkeypatch.setattr("book_translator.stages.translation.run_translation", fake_run_translation)
    return calls


@pytest.fixture
def run(script, monkeypatch):
    """Run the pipeline for real, with whatever stubs the test installed."""
    install_chunk_glossary(monkeypatch)
    install_matter(monkeypatch)

    def _run(*argv):
        # Stage 5 is on by default. Keep every test offline: a built-in chonkie
        # tokenizer needs no vocabulary download, and the stub chapters are tiny.
        args = list(argv)
        if "--chunk-tokenizer" not in args:
            args += ["--chunk-tokenizer", "word", "--chunk-size", "50"]
        parsed = script.build_parser().parse_args(args)
        pipeline = PipelineDriver(PipelineOptions.from_namespace(parsed), ns=parsed)
        try:
            return pipeline.run()
        finally:
            pipeline.close()

    return _run


def install_split(monkeypatch, *, ok: bool = True, writes: dict[str, str] | None = None):
    """Replace stage 1: optionally write chapter files, like a real run would."""
    calls: list[Path] = []

    def fake_split(source_path, output_dir, **kwargs):
        calls.append(Path(output_dir))
        if writes:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            for name, text in writes.items():
                (out / name).write_text(text, encoding="utf-8")
        return SplitterResult(
            ok=ok,
            source_path=str(source_path),
            output_dir=str(output_dir),
            strategy="markdown_headers" if ok else None,
            chapter_count=len(writes or {}),
            attempts=2,
            usage={"total_tokens": 1234},
            validation={"ok": ok, "issues": [] if ok else ["line 5 sits inside the printed TOC"]},
            verification={
                "reassembles_byte_for_byte": True,
                "output_lines": 6,
                "source_lines": 6,
            },
            error=None if ok else "the plan was rejected",
            log=["scout: 78 headers, verdict noisy"],
        )

    monkeypatch.setattr(book_translator.stages.agent, "run_split", fake_split)
    return calls


def install_preprocess(monkeypatch, *, ok: bool = True, writes: dict[str, str] | None = None):
    """Replace stage 2: report each chapter through the progress hook.

    ``writes`` makes the stub behave like the real phase: it leaves the cleaned
    chapters and ``_preprocess_manifest.json`` on disk, which is what the later
    stages read.
    """
    calls: list[Path] = []

    def fake_preprocess(chapters_path, output_dir=None, *, progress=None, **kwargs):
        calls.append(Path(chapters_path))
        folder = Path(chapters_path)
        files = sorted(folder.glob("*.md")) if folder.is_dir() else []
        records = []
        for index, path in enumerate(files, start=1):
            record = ChapterResult(
                chapter_id=path.stem,
                filename=path.name,
                source_path=str(path),
                output_path=str(Path(output_dir or chapters_path) / path.name),
                ok=ok,
                source_lines=3,
                output_lines=2,
                operations=[{"decision": "remove"}],
                text_fixes=[{"kind": "drop_page_number"}],
                removed_artefacts=[2],
                verification={"ok": ok},
                attempts=1,
                error=None if ok else "not verified",
                log=["kept 1 header, dropped 1 artefact"],
            )
            records.append(record)
            if progress is not None:
                progress(record.to_dict(), index, len(files))
        if writes is not None:
            out = Path(output_dir or chapters_path)
            out.mkdir(parents=True, exist_ok=True)
            manifest_files = []
            for record in records:
                target = out / record.filename
                text = writes.get(record.filename)
                if text is not None:
                    target.write_text(text, encoding="utf-8")
                manifest_files.append(
                    {
                        "filename": record.filename,
                        "path": str(target),
                        "source_lines": record.source_lines,
                        "lines": len(text.splitlines()) if text is not None else 0,
                        "verification_ok": record.ok,
                    }
                )
            (out / "_preprocess_manifest.json").write_text(
                json.dumps(
                    {
                        "stage": "preprocess",
                        "ok": ok and bool(records),
                        "dry_run": False,
                        "files": manifest_files,
                    }
                ),
                encoding="utf-8",
            )
        return PreprocessResult(
            ok=ok and bool(records),
            chapters_path=str(chapters_path),
            output_dir=str(output_dir or chapters_path),
            chapter_count=len(records),
            ok_count=sum(1 for record in records if record.ok),
            corpus_scan={"counts": {}, "total_findings": 0},
            chapters=[record.to_dict() for record in records],
            usage={"total_tokens": 5000},
            errors=[],
        )

    monkeypatch.setattr(book_translator.stages.preprocess_agent, "run_preprocess", fake_preprocess)
    return calls


#: Two damaged chapter files, so the scan phase has something to report.
DAMAGED = {
    "01-predgovor.md": "## 18\n\nPrvi deo teksta.\n",
    "02-uvod.md": "## UVOD\n\nmišić ,\n",
}


def install_dictionary(
    monkeypatch,
    *,
    ok: bool = True,
    missing: bool = False,
    writes: dict[str, str] | None = None,
    glossary: bool = True,
):
    """Replace stage 3 (both halves), so no test loads a tagger or calls a model."""
    calls: list[Path] = []

    def fake_run_dictionary(chapters_path, output_dir=None, **kwargs):
        calls.append(Path(chapters_path))
        if missing:
            raise MissingDependencyError("the tagger is not installed: pip install classla")
        # A real reduction always leaves at least one sentence behind, and the
        # glossary half reads that folder. With no explicit writes, leave one
        # file so the phase can reach the (stubbed) glossary.
        records = dict(writes if writes is not None else {"01-a.md": "Marko je ušao.\n"})
        if records and output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            # The real phase writes both reductions; stage 4 reads the second, so
            # the stub has to leave it on disk too.
            unique = out.parent / "03-reduced-unique"
            unique.mkdir(parents=True, exist_ok=True)
            for name, kept in records.items():
                (out / name).write_text(kept, encoding="utf-8")
                (unique / name).write_text(kept, encoding="utf-8")
        return DictionaryResult(
            ok=ok,
            chapters_path=str(chapters_path),
            output_dir=str(output_dir or chapters_path),
            language={"language": "sr", "method": "lingua", "confidence": 0.9, "samples": []},
            backend="classla",
            chapter_count=len(records),
            ok_count=len(records) if ok else 0,
            processed_count=len(records),
            totals={"sentences": 12 * len(records), "kept": 5 * len(records),
                    "sentences_tagged": 4 * len(records)},
            timing={
                "backend_load_seconds": 8.0,
                "ner_seconds": 2.5,
                "chapter_seconds": 3.0,
                "run_seconds": 11.5,
                "milliseconds_per_sentence": 156.25,
                "sentences_per_second": 6.4,
            },
            entities={"PER": 3, "LOC": 2},
            errors=[] if ok else ["01-a.md: the tagger crashed"],
        )

    monkeypatch.setattr(book_translator.stages.dictionary, "run_dictionary", fake_run_dictionary)
    if glossary:
        # The dictionary phase now builds the glossary too. Stub it by default so
        # every test stays offline; a test that wants to observe or fail it calls
        # install_glossary() itself, which patches over this one.
        install_glossary(monkeypatch)
    return calls


# --------------------------------------------------------------------------- #
# the script: phase order, failure, dry run, log, json
# --------------------------------------------------------------------------- #
@requires_sample
def test_every_phase_runs_in_order_and_the_log_keeps_the_detail(
    script, tmp_path, monkeypatch, capsys, run
):
    install_split(monkeypatch, writes=DAMAGED)
    install_dictionary(monkeypatch, writes={"01-predgovor.md": "Prvi deo teksta.\n"})
    preprocess_calls = install_preprocess(monkeypatch)

    # Stage 2 is on by default; the explicit flag keeps this test pinned to the
    # phase order regardless of how the module-level switch is set.
    code = run(str(SAMPLE), "--output-root", str(tmp_path / "out"), "--with-preprocess")
    captured = capsys.readouterr()

    assert code == 0
    markers = [
        "[1/7] RECON",
        "[2/7] SPLIT",
        "[3/7] SCAN",
        "[4/7] PREPROCESS",
        "[5/7] DICTIONARY",
        "[6/7] CHUNK",
        "[7/7] CHUNK_GLOSSARY",
    ]
    positions = [captured.out.index(marker) for marker in markers]
    assert positions == sorted(positions)
    assert "page_number_header" in captured.out  # the scan reported real damage
    assert "ok   01-predgovor.md" in captured.out  # live per-chapter progress
    assert "DONE" in captured.out
    # Debug detail stays off the console…
    assert "heuristic candidates:" not in captured.out
    # …but the log always keeps it, plus the provenance header.
    log = (tmp_path / "out" / SAMPLE.stem / "pipeline.log").read_text(encoding="utf-8")
    assert "heuristic candidates:" in log
    assert "# argv:" in log
    assert "SCAN" in log and "DICTIONARY" in log and "PREPROCESS" in log and "CHUNK" in log
    assert "CHUNK_GLOSSARY" in log
    assert preprocess_calls == [tmp_path / "out" / SAMPLE.stem / "01-chapters"]
    assert (tmp_path / "out" / SAMPLE.stem / "05-chunks" / "_chunk_manifest.json").is_file()


def test_stage_zero_converts_a_txt_source_before_recon(
    script, tmp_path, monkeypatch, capsys, run
):
    """A non-Markdown source gets its own first phase; Markdown sources keep seven."""
    source = tmp_path / "book.txt"
    source.write_text("# Chapter One\n\nSome text.\n", encoding="utf-8")
    install_split(monkeypatch)

    code = run(str(source), "--output-root", str(tmp_path / "out"), "--stop-after", "split")
    captured = capsys.readouterr()

    assert code == 0
    assert "[1/3] CONVERT" in captured.out
    assert "[2/3] RECON" in captured.out
    assert "[3/3] SPLIT" in captured.out
    assert "text-copy" in captured.out

    markdown = tmp_path / "out" / "book" / "00-markdown" / "book.md"
    assert markdown.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")


def test_stage_zero_dry_run_keeps_the_converted_text_out_of_the_output_folder(
    script, tmp_path, monkeypatch, capsys, run
):
    source = tmp_path / "book.txt"
    source.write_text("# Chapter One\n\nSome text.\n", encoding="utf-8")
    install_split(monkeypatch)  # writes no chapters, exactly like a dry run

    code = run(
        str(source),
        "--output-root",
        str(tmp_path / "out"),
        "--dry-run",
        "--stop-after",
        "split",
    )
    captured = capsys.readouterr()

    assert code == 0
    assert "converted into a temporary folder" in captured.out
    assert not (tmp_path / "out" / "book" / "00-markdown").exists()


@requires_sample
def test_skip_if_present_reuses_chapters_without_a_provider(
    script, tmp_path, monkeypatch, capsys, run
):
    """A reused split must not need an API key, or even resolve provider settings."""
    from book_translator.paths import resolve_output_dir

    # The chapters are already on disk before the run — that is the whole premise
    # of the flag, and it is what makes the phase decide without a provider.
    chapters = resolve_output_dir(SAMPLE, None, output_root=tmp_path / "out")
    chapters.mkdir(parents=True)
    (chapters / "01-predgovor.md").write_text("## 18\n\nPrvi deo teksta.\n", encoding="utf-8")
    (chapters / "02-uvod.md").write_text("## UVOD\n\nmišić ,\n", encoding="utf-8")
    (chapters / "_manifest.json").write_text(
        json.dumps(
            {
                "source_file": str(SAMPLE),
                "strategy": "preset_map",
                "files": [
                    {"filename": "01-predgovor.md", "start_line": 1, "end_line": 3},
                    {"filename": "02-uvod.md", "start_line": 5, "end_line": 7},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    split_calls = install_split(monkeypatch, writes=DAMAGED)
    install_dictionary(monkeypatch)

    def explode(*_, **__):  # pragma: no cover - must never be reached
        raise AssertionError("a reused split must not resolve provider settings")

    monkeypatch.setattr("book_translator.pipeline.driver.resolve_settings", explode)

    code = run(
        str(SAMPLE),
        "--output-root",
        str(tmp_path / "out"),
        "--skip-if-present",
        "--no-preprocess",
    )
    captured = capsys.readouterr()

    assert code == 0
    assert split_calls == []  # the splitter was never invoked
    assert "reused_existing_chapters" in captured.out
    assert "not used — the chapters were reused" in captured.out
    assert "2 reused" in captured.out
    assert "DONE" in captured.out


@requires_sample
def test_a_rejected_split_stops_before_stage_two(script, tmp_path, monkeypatch, capsys, run):
    install_split(monkeypatch, ok=False)
    preprocess_calls = install_preprocess(monkeypatch)

    code = run(str(SAMPLE), "--output-root", str(tmp_path / "out"))
    captured = capsys.readouterr()

    assert code == 1
    assert "STOPPED" in captured.out
    assert "PREPROCESS" not in captured.out
    assert "✗ split" in captured.err
    assert "line 5 sits inside the printed TOC" in captured.err
    assert preprocess_calls == []


def test_map_replays_a_supplied_map_without_a_model(script, tmp_path, monkeypatch, capsys, run):
    """``--map`` runs stage 1 from a file: no provider, no key, byte-for-byte output."""
    import book_translator.core.settings as settings_module

    def explode(*_, **__):  # pragma: no cover - the agent path would call this
        raise AssertionError("--map must not resolve provider settings")

    monkeypatch.setattr(settings_module, "resolve_settings", explode)

    source = tmp_path / "book.md"
    source.write_bytes(PLAIN.read_bytes())
    mapping = [
        {"line_number": 12, "filename": "01-predgovor.md"},
        {"line_number": 18, "filename": "02-dolazak.md"},
        {
            "line_number": 20,
            "filename": "03-dolazak-nastavak.md",
            "role": "prose-chapter",
            "evidence": "printed TOC lists a section here; no heading survives in the file",
        },
        {"line_number": 28, "filename": "04-oluja.md"},
        {"line_number": 44, "filename": "05-biljeske.md"},
    ]
    map_path = tmp_path / "map.json"
    map_path.write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")

    code = run(
        str(source),
        "--map",
        str(map_path),
        "--output-root",
        str(tmp_path / "out"),
        "--stop-after",
        "split",
    )
    captured = capsys.readouterr()

    assert code == 0
    assert "a supplied map routes" in captured.out
    assert "no agent turns" in captured.out
    chapters = tmp_path / "out" / source.stem / "01-chapters"
    manifest = json.loads((chapters / "_manifest.json").read_text(encoding="utf-8"))
    assert manifest["chapter_count"] == 5
    blob = b"".join((chapters / f["filename"]).read_bytes() for f in manifest["files"])
    assert blob == source.read_bytes()


@requires_sample
def test_a_dry_run_without_chapters_reports_stage_two_as_skipped(
    script, tmp_path, monkeypatch, capsys, run
):
    install_split(monkeypatch)  # writes nothing, exactly like the real --dry-run
    install_dictionary(monkeypatch)
    preprocess_calls = install_preprocess(monkeypatch)

    code = run(
        str(SAMPLE), "--output-root", str(tmp_path / "out"), "--dry-run", "--with-preprocess"
    )
    captured = capsys.readouterr()

    assert code == 0
    assert captured.out.count("skipped") >= 2
    assert "SCAN" in captured.out and "DICTIONARY" in captured.out
    assert "PREPROCESS" in captured.out
    assert "DONE" in captured.out
    assert preprocess_calls == []


@requires_sample
def test_json_prints_one_machine_readable_summary(script, tmp_path, monkeypatch, capsys, run):
    install_split(monkeypatch, writes={"01-a.md": "## 18\n"})
    dictionary_calls = install_dictionary(monkeypatch, writes={"01-a.md": "## 18\n"})
    install_preprocess(monkeypatch)

    code = run(str(SAMPLE), "--output-root", str(tmp_path / "out"), "--json", "--with-preprocess")
    out = capsys.readouterr().out

    assert code == 0
    # One JSON object on stdout, no human narration around it.
    assert out.lstrip().startswith("{")
    assert "DONE" not in out
    payload = json.loads(out)
    assert payload["ok"] is True
    assert [phase["name"] for phase in payload["phases"]] == [
        "recon",
        "split",
        "scan",
        "preprocess",
        "dictionary",
        "chunk",
        "chunk_glossary",
    ]
    assert payload["phases"][1]["result"]["chapter_count"] == 1
    assert payload["reduced_dir"].endswith("03-reduced")
    assert payload["unique_dir"].endswith("03-reduced-unique")
    assert payload["chunks_dir"].endswith("05-chunks")
    assert payload["chunk_glossary_dir"].endswith("06-chunk-glossary")
    assert payload["log"].endswith("pipeline.log")
    assert dictionary_calls == [tmp_path / "out" / SAMPLE.stem / "01-chapters"]


@requires_sample
def test_a_failed_chapter_stops_stage_two(script, tmp_path, monkeypatch, capsys, run):
    install_split(monkeypatch, writes=DAMAGED)
    install_dictionary(monkeypatch)
    install_preprocess(monkeypatch, ok=False)

    code = run(str(SAMPLE), "--output-root", str(tmp_path / "out"), "--with-preprocess")
    captured = capsys.readouterr()

    assert code == 1
    assert "FAIL" in captured.err
    assert "STOPPED" in captured.out
    assert "chapter(s) failed" in captured.err


@requires_sample
def test_stage_two_is_on_by_default_and_can_be_dropped(
    script, tmp_path, monkeypatch, capsys, run
):
    """Stage 2 repairs the chapters before anything reads them; --no-preprocess drops it."""
    install_split(monkeypatch, writes=DAMAGED)
    install_dictionary(monkeypatch)
    preprocess_calls = install_preprocess(monkeypatch)

    code = run(str(SAMPLE), "--output-root", str(tmp_path / "out"))
    captured = capsys.readouterr()

    assert code == 0
    assert "[4/7] PREPROCESS" in captured.out
    assert preprocess_calls == [tmp_path / "out" / SAMPLE.stem / "01-chapters"]
    assert "DONE" in captured.out

    dropped = tmp_path / "dropped"
    dropped_calls = install_preprocess(monkeypatch)
    code = run(str(SAMPLE), "--output-root", str(dropped), "--no-preprocess")
    captured = capsys.readouterr()

    assert code == 0
    assert "PREPROCESS" not in captured.out
    assert "preprocess switched off" in captured.out
    # Stage 2 is dropped, so six phases are selected: scan is third, chunk glossary last.
    assert "[3/6] SCAN" in captured.out
    assert "[4/6] DICTIONARY" in captured.out
    assert "[5/6] CHUNK" in captured.out
    assert "[6/6] CHUNK_GLOSSARY" in captured.out
    assert "DONE" in captured.out
    assert dropped_calls == []


@requires_sample
def test_the_cleaned_text_is_what_stages_three_to_five_read(
    script, tmp_path, monkeypatch, capsys, run
):
    """Stage 2 is not a dead end: the dictionary, glossary and chunker read it."""
    install_split(monkeypatch, writes=DAMAGED)
    dictionary_calls = install_dictionary(
        monkeypatch, writes={"01-predgovor.md": "Prvi deo teksta.\n"}
    )
    glossary_calls = install_glossary(monkeypatch)
    cleaned = {
        "01-predgovor.md": "Prvi deo teksta.\n",
        "02-uvod.md": "mišić.\n",
    }
    install_preprocess(monkeypatch, writes=cleaned)

    out = tmp_path / "out"
    code = run(str(SAMPLE), "--output-root", str(out))
    captured = capsys.readouterr()

    cleaned_dir = out / SAMPLE.stem / "02-preprocessed"
    assert code == 0
    assert dictionary_calls == [cleaned_dir]
    assert glossary_calls == [cleaned_dir]
    assert "cleaned by stage 2" in captured.out
    manifest = json.loads(
        (out / SAMPLE.stem / "05-chunks" / "_chunk_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["chapters_path"] == str(cleaned_dir)
    chunk_text = "".join(
        path.read_text(encoding="utf-8")
        for path in sorted((out / SAMPLE.stem / "05-chunks").glob("*.md"))
    )
    assert "Prvi deo teksta." in chunk_text
    assert "## 18" not in chunk_text

    # --text-in pins every text stage back to the raw chapters.
    raw_out = tmp_path / "raw"
    raw_calls = install_dictionary(
        monkeypatch, writes={"01-predgovor.md": "Prvi deo teksta.\n"}
    )
    install_preprocess(monkeypatch, writes=cleaned)
    code = run(
        str(SAMPLE),
        "--output-root",
        str(raw_out),
        "--text-in",
        str(raw_out / SAMPLE.stem / "01-chapters"),
    )
    captured = capsys.readouterr()
    assert code == 0
    assert raw_calls == [raw_out / SAMPLE.stem / "01-chapters"]
    manifest = json.loads(
        (raw_out / SAMPLE.stem / "05-chunks" / "_chunk_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["chapters_path"] == str(raw_out / SAMPLE.stem / "01-chapters")

    # A window the cleaned folder does not cover falls back to the raw text.
    partial = tmp_path / "partial"
    partial_calls = install_dictionary(
        monkeypatch, writes={"01-predgovor.md": "Prvi deo teksta.\n"}
    )
    install_preprocess(monkeypatch, writes={"01-predgovor.md": "Prvi deo teksta.\n"})
    code = run(
        str(SAMPLE),
        "--output-root",
        str(partial),
        "--first-chapters",
        "2",
    )
    captured = capsys.readouterr()
    assert code == 0
    assert partial_calls == [partial / SAMPLE.stem / "01-chapters"]
    assert "cleaned text does not cover this window" in captured.out


@requires_sample
def test_stage_three_is_on_by_default_and_can_be_dropped(
    script, tmp_path, monkeypatch, capsys, run
):
    install_split(monkeypatch, writes=DAMAGED)
    dictionary_calls = install_dictionary(monkeypatch)

    code = run(
        str(SAMPLE), "--output-root", str(tmp_path / "out"), "--no-preprocess"
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "DICTIONARY" in captured.out
    assert "language" in captured.out
    assert "ner time" in captured.out
    assert dictionary_calls == [tmp_path / "out" / SAMPLE.stem / "01-chapters"]

    dropped = tmp_path / "dropped"
    install_dictionary(monkeypatch)  # a fresh call list, patched over the old one
    code = run(
        str(SAMPLE), "--output-root", str(dropped), "--no-dictionary", "--no-preprocess"
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "DICTIONARY" not in captured.out
    assert "dictionary switched off" in captured.out


@requires_sample
def test_chunk_stage_is_on_by_default_and_can_be_dropped(
    script, tmp_path, monkeypatch, capsys, run
):
    install_split(monkeypatch, writes=DAMAGED)
    install_dictionary(monkeypatch)

    code = run(
        str(SAMPLE), "--output-root", str(tmp_path / "out"), "--no-preprocess"
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "CHUNK" in captured.out
    assert (tmp_path / "out" / SAMPLE.stem / "05-chunks" / "_chunk_manifest.json").is_file()

    dropped = tmp_path / "dropped"
    code = run(
        str(SAMPLE), "--output-root", str(dropped), "--no-chunk", "--no-preprocess"
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "CHUNK" not in captured.out
    assert "chunking switched off" in captured.out
    assert not (dropped / SAMPLE.stem / "05-chunks").exists()


@requires_sample
def test_chunk_glossary_stage_is_on_by_default_and_can_be_dropped(
    script, tmp_path, monkeypatch, capsys, run
):
    install_split(monkeypatch, writes=DAMAGED)
    install_dictionary(monkeypatch)
    calls = install_chunk_glossary(monkeypatch)

    out = tmp_path / "out"
    code = run(
        str(SAMPLE), "--output-root", str(out), "--no-preprocess"
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "[6/6] CHUNK_GLOSSARY" in captured.out
    assert calls == [out / SAMPLE.stem / "05-chunks"]
    assert (out / SAMPLE.stem / "06-chunk-glossary" / "_chunk_glossary_manifest.json").is_file()

    dropped = tmp_path / "dropped"
    dropped_calls = install_chunk_glossary(monkeypatch)
    code = run(
        str(SAMPLE), "--output-root", str(dropped), "--no-chunk-glossary", "--no-preprocess"
    )
    captured = capsys.readouterr()
    assert code == 0
    assert dropped_calls == []
    assert "[6/6] CHUNK_GLOSSARY" not in captured.out
    assert "chunk glossary switched off" in captured.out
    assert not (dropped / SAMPLE.stem / "06-chunk-glossary").exists()


@requires_sample
def test_a_missing_tagger_skips_stage_three_without_failing_the_run(
    script, tmp_path, monkeypatch, capsys, run
):
    install_split(monkeypatch, writes=DAMAGED)
    install_dictionary(monkeypatch, missing=True)

    code = run(
        str(SAMPLE), "--output-root", str(tmp_path / "out"), "--no-preprocess"
    )
    captured = capsys.readouterr()

    assert code == 0
    assert "DONE" in captured.out
    assert "skipped — the optional tagger is not installed" in captured.out
    assert "pip install classla" in captured.out
    # The closing report points at the install command instead of claiming output.
    assert "install stage 3's tagger" in captured.out
    assert "holds the reduced chapters" not in captured.out


# --------------------------------------------------------------------------- #
# stage 4: opt-in, and it does not spend on a dry run
# --------------------------------------------------------------------------- #
def install_glossary(monkeypatch, *, ok: bool = True):
    """Replace stage 4, so no test ever calls a paid model."""
    from book_translator.stages.glossary import GlossaryResult

    calls: list[Path] = []

    def fake_run_glossary(chapters_path, output_dir=None, **kwargs):
        calls.append(Path(chapters_path))
        dry_run = bool(kwargs.get("dry_run"))
        if output_dir and not dry_run:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            for name in (
                "glossary.json",
                "glossary.md",
                "_glossary_manifest.json",
                "_glossary_trace.json",
            ):
                (out / name).write_text("{}", encoding="utf-8")
        return GlossaryResult(
            ok=ok,
            chapters_path=str(chapters_path),
            sentences_dir=str(Path(chapters_path).parent / "03-reduced-unique"),
            output_dir=str(output_dir or chapters_path),
            source_language="en",
            target_language="sr",
            provider="deepseek",
            model="deepseek-flash",
            body_chapter_count=len(DAMAGED),
            chapter_count=len(DAMAGED),
            ok_count=len(DAMAGED) if ok else 0,
            processed_count=len(DAMAGED),
            entry_count=7,
            dry_run=dry_run,
            categories={"character": 5, "location": 2},
            tokens={
                "input_tokens": 900,
                "output_tokens": 100,
                "total_tokens": 1000,
                "calls": 2,
                "chapters": 2,
            },
            timing={"run_seconds": 4.0, "model_seconds": 3.5, "model_calls": 2},
            errors=[] if ok else ["01-a.md: boom"],
        )

    monkeypatch.setattr("book_translator.stages.glossary.run_glossary", fake_run_glossary)
    return calls


@requires_sample
def test_the_dictionary_phase_builds_the_glossary_by_default(
    script, tmp_path, monkeypatch, capsys, run
):
    install_split(monkeypatch, writes=DAMAGED)
    install_dictionary(
        monkeypatch, writes={"01-predgovor.md": "Prvi deo teksta.\n"}, glossary=False
    )
    glossary_calls = install_glossary(monkeypatch)

    code = run(
        str(SAMPLE),
        "--output-root",
        str(tmp_path / "out"),
        "--limit",
        "2",
        "--no-preprocess",
    )
    captured = capsys.readouterr()
    assert code == 0
    # The glossary is inside the dictionary phase; stage 6 is the only later
    # phase that mentions the glossary, and it runs after the chunks.
    assert "[4/6] DICTIONARY" in captured.out
    assert "[6/6] CHUNK_GLOSSARY" in captured.out
    assert "glossary — one chapter at a time" in captured.out
    assert "entries" in captured.out
    assert glossary_calls == [tmp_path / "out" / SAMPLE.stem / "01-chapters"]
    assert "stage-4 glossary" in captured.out

    # ...and --no-glossary keeps the phase deterministic (no model, no key).
    install_dictionary(
        monkeypatch,
        writes={"01-predgovor.md": "Prvi deo teksta.\n"},
        glossary=False,
    )
    no_gloss_calls = install_glossary(monkeypatch)
    code = run(
        str(SAMPLE),
        "--output-root",
        str(tmp_path / "no-glossary"),
        "--limit",
        "2",
        "--no-glossary",
        "--no-preprocess",
    )
    captured = capsys.readouterr()
    assert code == 0
    assert no_gloss_calls == []
    assert "glossary — one chapter at a time" not in captured.out
    assert "--no-glossary is set" in captured.out


@requires_sample
def test_a_glossary_dry_run_runs_without_writing(
    script, tmp_path, monkeypatch, capsys, run
):
    install_split(monkeypatch, writes=DAMAGED)
    install_dictionary(monkeypatch, writes={"01-predgovor.md": "Prvi deo teksta.\n"})
    glossary_calls = install_glossary(monkeypatch)

    code = run(
        str(SAMPLE),
        "--output-root",
        str(tmp_path / "out"),
        "--with-glossary",
        "--dry-run",
        "--no-preprocess",
    )
    captured = capsys.readouterr()
    assert code == 0
    # The glossary plan is the model's answer, so validating it means asking;
    # the dry run only suppresses the artifacts.
    assert len(glossary_calls) == 1
    assert "dry run" in captured.out
    glossary_dir = tmp_path / "out" / SAMPLE.stem / "04-glossary"
    assert not (glossary_dir / "glossary.json").exists()


@requires_sample
def test_a_failed_glossary_chapter_stops_the_run(script, tmp_path, monkeypatch, capsys, run):
    install_split(monkeypatch, writes=DAMAGED)
    install_dictionary(monkeypatch, writes={"01-predgovor.md": "Prvi deo teksta.\n"})
    install_glossary(monkeypatch, ok=False)

    code = run(
        str(SAMPLE), "--output-root", str(tmp_path / "out"), "--with-glossary", "--no-preprocess"
    )
    captured = capsys.readouterr()
    assert code == 1
    assert "STOPPED" in captured.out
    assert "01-a.md: boom" in captured.err


# --------------------------------------------------------------------------- #
# the hook the script drives: run_preprocess(progress=…)
# --------------------------------------------------------------------------- #
class _FakePreprocessApp:
    """Minimal stand-in for the compiled per-chapter graph."""

    def __init__(self, final: dict) -> None:
        self.final = final

    def invoke(self, state, config=None):
        return dict(self.final)


def _chapter_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "01-chapters"
    folder.mkdir()
    (folder / "01-a.md").write_text("## A\n\ntext\n", encoding="utf-8")
    (folder / "02-b.md").write_text("## B\n\ntext\n", encoding="utf-8")
    return folder


@requires_sample
def test_a_preprocess_dry_run_writes_no_manifest_or_trace(tmp_path, monkeypatch):
    final = {
        "applied": {"ok": True, "source_lines": 3, "output_lines": 3},
        "verification": {"ok": True},
        "validation": {"ok": True, "text_fixes": []},
        "operations": [],
        "attempts": 1,
        "usage": {"total_tokens": 7},
        "log": [],
    }
    monkeypatch.setattr(
        preprocess_runner,
        "build_preprocess_graph",
        lambda *args, **kwargs: _FakePreprocessApp(final),
    )

    result = preprocess_runner.run_preprocess(
        str(_chapter_folder(tmp_path)),
        str(tmp_path / "clean"),
        dry_run=True,
        # The default is True, as the CLI passes it; the runner must still not
        # write a manifest or trace for a run that wrote no chapter.
    )

    assert result.ok and result.dry_run
    assert result.manifest_path is None
    assert result.trace_path is None
    assert not (tmp_path / "clean" / "_preprocess_manifest.json").exists()
    assert not (tmp_path / "clean" / "_preprocess_trace.json").exists()


@requires_sample
def test_run_preprocess_reports_every_chapter(tmp_path, monkeypatch):
    final = {
        "applied": {"ok": True, "source_lines": 3, "output_lines": 3, "removed_artefacts": []},
        "verification": {"ok": True},
        "validation": {"ok": True, "text_fixes": []},
        "operations": [],
        "attempts": 1,
        "usage": {"total_tokens": 7},
        "log": ["nothing to do"],
    }
    monkeypatch.setattr(
        preprocess_runner,
        "build_preprocess_graph",
        lambda *args, **kwargs: _FakePreprocessApp(final),
    )
    seen: list[tuple[str, int, int]] = []

    result = preprocess_runner.run_preprocess(
        str(_chapter_folder(tmp_path)),
        str(tmp_path / "clean"),
        dry_run=False,
        write_manifest=False,
        progress=lambda chapter, done, total: seen.append((chapter["filename"], done, total)),
    )

    assert result.ok and result.chapter_count == 2
    assert [done for _, done, _ in seen] == [1, 2]
    assert {total for _, _, total in seen} == {2}
    assert result.usage.get("total_tokens") == 14  # 2 chapters x 7


@requires_sample
def test_a_raising_progress_hook_never_breaks_the_run(tmp_path, monkeypatch):
    final = {
        "applied": {"ok": True, "source_lines": 3, "output_lines": 3, "removed_artefacts": []},
        "verification": {"ok": True},
        "validation": {"ok": True, "text_fixes": []},
        "operations": [],
        "attempts": 1,
        "usage": {"total_tokens": 7},
        "log": [],
    }
    monkeypatch.setattr(
        preprocess_runner,
        "build_preprocess_graph",
        lambda *args, **kwargs: _FakePreprocessApp(final),
    )

    def boom(*args):
        raise RuntimeError("the reporter is broken, the run is not")

    result = preprocess_runner.run_preprocess(
        str(_chapter_folder(tmp_path)),
        str(tmp_path / "clean"),
        dry_run=True,
        write_manifest=False,
        progress=boom,
    )

    assert result.ok and result.chapter_count == 2


# --------------------------------------------------------------------------- #
# stage 7: opt-in translation, with its own limit and debug folder
# --------------------------------------------------------------------------- #
@requires_sample
def test_translate_is_opt_in_and_runs_only_with_the_flag(
    script, tmp_path, monkeypatch, capsys, run
):
    install_split(monkeypatch, writes=DAMAGED)
    install_dictionary(monkeypatch)
    install_chunk_glossary(monkeypatch, files=True)
    calls = install_translate(monkeypatch)

    out = tmp_path / "out"
    code = run(
        str(SAMPLE), "--output-root", str(out), "--no-preprocess"
    )
    captured = capsys.readouterr()

    assert code == 0
    assert calls == []
    assert "stage 7 (translation) is switched off" in captured.out
    assert "TRANSLATE" not in captured.out
    assert not (out / SAMPLE.stem / "07-translations").exists()

    calls = install_translate(monkeypatch)
    code = run(
        str(SAMPLE), "--output-root", str(out), "--with-translate", "--no-preprocess"
    )
    captured = capsys.readouterr()

    assert code == 0
    # Eleven phases now: the EPUB package follows the bound Markdown book.
    assert "[8/11] TRANSLATE" in captured.out
    assert len(calls) == 1
    assert calls[0]["chunks_path"] == str(out / SAMPLE.stem / "05-chunks")
    assert calls[0]["glossary_scope"] == "chunk"
    assert calls[0]["source_language"] == "en"
    assert calls[0]["target_language"] == "sr"
    assert calls[0]["limit"] is None
    assert (out / SAMPLE.stem / "07-translations" / "_translation_manifest.json").is_file()
    assert (out / SAMPLE.stem / "07-translation-prompts").is_dir()


@requires_sample
def test_the_matter_stage_runs_before_translation(
    script, tmp_path, monkeypatch, capsys, run
):
    install_split(monkeypatch, writes=DAMAGED)
    install_dictionary(monkeypatch)
    install_chunk_glossary(monkeypatch, files=True)
    matter_calls = install_matter(monkeypatch)
    translate_calls = install_translate(monkeypatch)

    out = tmp_path / "out"
    code = run(
        str(SAMPLE), "--output-root", str(out), "--with-translate", "--no-preprocess"
    )
    captured = capsys.readouterr()

    assert code == 0
    assert len(matter_calls) == 1
    assert len(translate_calls) == 1
    assert translate_calls[0]["include_matter"] is False
    assert captured.out.index(" MATTER") < captured.out.index(" TRANSLATE")
    assert matter_calls[0]["source_language"] == "en"
    assert matter_calls[0]["target_language"] == "sr"
    assert (out / SAMPLE.stem / "06b-matter" / "_matter_manifest.json").is_file()
    assert (out / SAMPLE.stem / "06b-matter-prompts").is_dir()

    matter_calls = install_matter(monkeypatch)
    code = run(
        str(SAMPLE),
        "--output-root",
        str(tmp_path / "no-matter"),
        "--with-translate",
        "--no-matter",
        "--no-preprocess",
    )
    captured = capsys.readouterr()
    assert code == 0
    assert matter_calls == []
    assert "MATTER" not in captured.out


@requires_sample
def test_the_translate_limit_is_forwarded(script, tmp_path, monkeypatch, capsys, run):
    install_split(monkeypatch, writes=DAMAGED)
    install_dictionary(monkeypatch)
    install_chunk_glossary(monkeypatch, files=True)
    calls = install_translate(monkeypatch)

    code = run(
        str(SAMPLE),
        "--output-root",
        str(tmp_path / "out"),
        "--with-translate",
        "--translate-limit",
        "3",
        "--no-preprocess",
    )
    captured = capsys.readouterr()

    assert code == 0
    assert len(calls) == 1
    assert calls[0]["limit"] == 3
    assert "limit" in captured.out


@requires_sample
def test_the_translate_provider_is_a_stage_7_only_choice(
    script, tmp_path, monkeypatch, capsys, run
):
    install_split(monkeypatch, writes=DAMAGED)
    install_dictionary(monkeypatch)
    install_chunk_glossary(monkeypatch, files=True)
    calls = install_translate(monkeypatch)

    code = run(
        str(SAMPLE),
        "--output-root",
        str(tmp_path / "out"),
        "--with-translate",
        "--translate-provider",
        "lmstudio",
        "--translate-model",
        "my-local-model",
        "--no-preprocess",
    )
    captured = capsys.readouterr()

    assert code == 0
    assert len(calls) == 1
    resolved = calls[0]["settings_factory"]()
    assert resolved.provider == "lmstudio"
    assert resolved.model == "my-local-model"
    assert resolved.base_url == "http://127.0.0.1:1234/v1"
    assert "lmstudio/my-local-model" in captured.out


@requires_sample
def test_translate_skips_when_the_chunk_glossary_is_missing(
    script, tmp_path, monkeypatch, capsys, run
):
    install_split(monkeypatch, writes=DAMAGED)
    install_dictionary(monkeypatch)
    install_chunk_glossary(monkeypatch)  # manifests only, no per-chunk files
    calls = install_translate(monkeypatch)

    code = run(
        str(SAMPLE),
        "--output-root",
        str(tmp_path / "out"),
        "--with-translate",
        "--no-preprocess",
    )
    captured = capsys.readouterr()

    assert code == 0
    assert calls == []
    assert "no per-chunk glossary" in captured.out


@requires_sample
def test_translate_is_dropped_by_no_translate(script, tmp_path, monkeypatch, capsys, run):
    install_split(monkeypatch, writes=DAMAGED)
    install_dictionary(monkeypatch)
    install_chunk_glossary(monkeypatch, files=True)
    calls = install_translate(monkeypatch)

    code = run(
        str(SAMPLE),
        "--output-root",
        str(tmp_path / "out"),
        "--with-translate",
        "--no-translate",
        "--no-preprocess",
    )

    assert code == 0
    assert calls == []
    assert "TRANSLATE" not in capsys.readouterr().out


@requires_sample
def test_a_dry_run_without_chunks_skips_translate_without_calling_it(
    script, tmp_path, monkeypatch, capsys, run
):
    install_split(monkeypatch, writes=DAMAGED)
    install_dictionary(monkeypatch)
    install_chunk_glossary(monkeypatch, files=True)
    calls = install_translate(monkeypatch)

    code = run(
        str(SAMPLE),
        "--output-root",
        str(tmp_path / "out"),
        "--with-translate",
        "--dry-run",
        "--no-preprocess",
    )
    captured = capsys.readouterr()

    assert code == 0
    # A dry-run chunk phase writes no chunks, so there is nothing to translate.
    # When chunks do exist, a dry run calls the stage (which writes nothing).
    assert calls == []
    assert "no chunks on disk to translate" in captured.out
    assert "dry run" in captured.out


# --------------------------------------------------------------------------- #
# the chapter window: first N chapters, split still the whole book
# --------------------------------------------------------------------------- #
def _wrap_calls(monkeypatch, module, name: str, seen: list[dict]) -> None:
    """Record every keyword argument the pipeline hands to a stage function."""
    current = getattr(module, name)

    def wrapper(*args, **kwargs):
        seen.append(dict(kwargs))
        return current(*args, **kwargs)

    monkeypatch.setattr(module, name, wrapper)


def install_translate_for_chunks(monkeypatch):
    """A stage-7 double that writes a real translation file for every chunk."""
    from book_translator.stages.chunking import chunk_files
    from book_translator.stages.translation import TranslationResult

    calls: list[dict] = []

    def fake_run_translation(chunks_path, output_dir=None, **kwargs):
        calls.append({"chunks_path": str(chunks_path), **kwargs})
        entries = chunk_files(chunks_path)
        limit = kwargs.get("limit")
        selected = entries[: int(limit)] if limit else entries
        out = Path(output_dir or chunks_path)
        out.mkdir(parents=True, exist_ok=True)
        for entry in selected:
            name = str(entry["filename"])
            (out / name).write_text(f"PREVEDENO:{name}\n", encoding="utf-8")
        prompts = Path(kwargs.get("prompts_dir") or out.parent / "07-translation-prompts")
        prompts.mkdir(parents=True, exist_ok=True)
        (prompts / "one.json").write_text("{}", encoding="utf-8")
        manifest = out / "_translation_manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        trace = out / "_translation_trace.json"
        trace.write_text("{}", encoding="utf-8")
        return TranslationResult(
            ok=True,
            chunks_path=str(chunks_path),
            chunk_glossary_path=str(kwargs.get("chunk_glossary_path") or ""),
            output_dir=str(out),
            prompts_dir=str(prompts),
            glossary_scope=str(kwargs.get("glossary_scope") or "chunk"),
            chunk_count=len(selected),
            ok_count=len(selected),
            processed_count=len(selected),
            tokens={"total_tokens": 10 * len(selected), "calls": len(selected)},
            totals={
                "translated_chunks": len(selected),
                "source_chars": 100 * len(selected),
                "translation_chars": 90 * len(selected),
            },
            timing={"run_seconds": 0.1, "seconds_per_chunk": 0.1},
            chunks=[],
            errors=[],
            manifest_path=str(manifest),
            trace_path=str(trace),
        )

    monkeypatch.setattr("book_translator.stages.translation.run_translation", fake_run_translation)
    return calls


@requires_sample
def test_first_chapters_limits_every_stage_after_split(
    script, tmp_path, monkeypatch, capsys, run
):
    import book_translator.stages.assembly
    import book_translator.stages.chunk_glossary
    import book_translator.stages.chunking
    import book_translator.stages.dictionary
    import book_translator.stages.glossary

    install_split(monkeypatch, writes=DAMAGED)
    install_dictionary(monkeypatch)
    install_glossary(monkeypatch)
    install_chunk_glossary(monkeypatch, files=True)
    install_translate_for_chunks(monkeypatch)

    seen: dict[str, list[dict]] = {
        "split": [],
        "dictionary": [],
        "glossary": [],
        "chunk": [],
        "chunk_glossary": [],
        "translate": [],
        "assembly": [],
    }
    _wrap_calls(monkeypatch, book_translator.stages.agent, "run_split", seen["split"])
    _wrap_calls(monkeypatch, book_translator.stages.dictionary, "run_dictionary", seen["dictionary"])
    _wrap_calls(monkeypatch, book_translator.stages.glossary, "run_glossary", seen["glossary"])
    _wrap_calls(monkeypatch, book_translator.stages.chunking, "run_chunking", seen["chunk"])
    _wrap_calls(
        monkeypatch,
        book_translator.stages.chunk_glossary,
        "run_chunk_glossary",
        seen["chunk_glossary"],
    )
    _wrap_calls(monkeypatch, book_translator.stages.translation, "run_translation", seen["translate"])
    _wrap_calls(monkeypatch, book_translator.stages.assembly, "run_assembly", seen["assembly"])

    out = tmp_path / "out"
    code = run(
        str(SAMPLE),
        "--output-root",
        str(out),
        "--first-chapters",
        "1",
        "--with-translate",
        "--no-preprocess",
    )
    captured = capsys.readouterr()

    assert code == 0
    assert "chapter window" in captured.out
    # Splitting ignores the window: it is the coordinate system of every later stage.
    assert "limit" not in seen["split"][0]
    assert seen["dictionary"][0]["limit"] == 1
    assert seen["glossary"][0]["limit"] == 1
    assert seen["chunk"][0]["limit"] == 1
    assert seen["chunk_glossary"][0]["chapter_files"] == ["01-predgovor.md"]
    assert seen["translate"][0]["chapter_files"] == ["01-predgovor.md"]
    assert seen["translate"][0]["limit"] is None
    assert seen["assembly"][0]["first_chapters"] == 1

    translated = out / SAMPLE.stem / "08-translated-chapters"
    assert (translated / "01-predgovor.md").is_file()
    assert not (translated / "02-uvod.md").exists()


@requires_sample
def test_increasing_the_chapter_window_resumes_with_skip_if_present(
    script, tmp_path, monkeypatch, capsys, run
):
    import book_translator.stages.assembly
    import book_translator.stages.chunk_glossary
    import book_translator.stages.chunking
    import book_translator.stages.dictionary
    import book_translator.stages.glossary

    install_split(monkeypatch, writes=DAMAGED)
    install_dictionary(monkeypatch)
    install_glossary(monkeypatch)
    install_chunk_glossary(monkeypatch, files=True)
    install_translate_for_chunks(monkeypatch)

    seen: dict[str, list[dict]] = {
        "dictionary": [],
        "chunk": [],
        "chunk_glossary": [],
        "translate": [],
        "assembly": [],
    }
    _wrap_calls(monkeypatch, book_translator.stages.dictionary, "run_dictionary", seen["dictionary"])
    _wrap_calls(monkeypatch, book_translator.stages.glossary, "run_glossary", [])
    _wrap_calls(monkeypatch, book_translator.stages.chunking, "run_chunking", seen["chunk"])
    _wrap_calls(
        monkeypatch,
        book_translator.stages.chunk_glossary,
        "run_chunk_glossary",
        seen["chunk_glossary"],
    )
    _wrap_calls(monkeypatch, book_translator.stages.translation, "run_translation", seen["translate"])
    _wrap_calls(monkeypatch, book_translator.stages.assembly, "run_assembly", seen["assembly"])

    out = tmp_path / "out"
    first = run(
        str(SAMPLE),
        "--output-root",
        str(out),
        "--first-chapters",
        "1",
        "--with-translate",
        "--skip-if-present",
        "--no-preprocess",
    )
    assert first == 0
    assert seen["dictionary"][-1]["limit"] == 1
    assert seen["chunk_glossary"][-1]["chapter_files"] == ["01-predgovor.md"]
    assert seen["translate"][-1]["chapter_files"] == ["01-predgovor.md"]
    assert seen["translate"][-1]["limit"] is None
    assert seen["assembly"][-1]["first_chapters"] == 1

    second = run(
        str(SAMPLE),
        "--output-root",
        str(out),
        "--first-chapters",
        "2",
        "--with-translate",
        "--skip-if-present",
        "--no-preprocess",
    )
    assert second == 0
    # The window widens; every stage is asked for two body chapters and is told
    # to resume, so the work done in the first run is reused.
    assert seen["dictionary"][-1]["limit"] == 2
    assert seen["dictionary"][-1]["skip_if_present"] is True
    assert seen["chunk"][-1]["limit"] == 2
    assert seen["chunk"][-1]["skip_if_present"] is True
    assert seen["chunk_glossary"][-1]["chapter_files"] == [
        "01-predgovor.md",
        "02-uvod.md",
    ]
    assert seen["chunk_glossary"][-1]["skip_if_present"] is True
    assert seen["translate"][-1]["chapter_files"] == [
        "01-predgovor.md",
        "02-uvod.md",
    ]
    assert seen["translate"][-1]["limit"] is None
    assert seen["translate"][-1]["resume"] is True
    assert seen["assembly"][-1]["first_chapters"] == 2
    assert seen["assembly"][-1]["skip_if_present"] is True
    assert (out / SAMPLE.stem / "08-translated-chapters" / "02-uvod.md").is_file()
