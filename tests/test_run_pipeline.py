"""Offline tests for ``scripts/run_pipeline.py`` and the progress hook it drives.

No network, no API key: both agent runs are replaced by stubs, so these tests pin
the *narration and the control flow* — phase order, what stops on failure, what
the log keeps that the console does not — not the agents themselves.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import book_splitter.agent
import book_splitter.dictionary
import book_splitter.preprocess_agent
import pytest
from book_splitter.agent.result import SplitterResult
from book_splitter.core.errors import MissingDependencyError
from book_splitter.dictionary.runner import DictionaryResult
from book_splitter.preprocess_agent import runner as preprocess_runner
from book_splitter.preprocess_agent.runner import ChapterResult, PreprocessResult

from _fixtures import PLAIN, SAMPLE, requires_sample

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def script():
    """``scripts/run_pipeline.py`` loaded as a module (it is not a package)."""
    name = "run_pipeline_script"
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "run_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: ``from __future__ import annotations`` makes
    # dataclass field types strings, which dataclasses resolves through sys.modules.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def install_chunk_glossary(monkeypatch, *, ok: bool = True):
    """Replace stage 6, so no test loads a tagger or reads a real glossary."""
    from book_splitter.chunk_glossary import ChunkGlossaryResult

    calls: list[Path] = []

    def fake_run_chunk_glossary(chunks_path, output_dir=None, **kwargs):  # noqa: ARG001
        calls.append(Path(chunks_path))
        out = Path(output_dir or chunks_path)
        if output_dir:
            out.mkdir(parents=True, exist_ok=True)
            for name in ("_chunk_glossary_manifest.json", "_chunk_glossary_trace.json"):
                (out / name).write_text("{}", encoding="utf-8")
        return ChunkGlossaryResult(
            ok=ok,
            chunks_path=str(chunks_path),
            glossary_path=str(out.parent / "glossary" / "glossary.json"),
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

    monkeypatch.setattr("book_splitter.chunk_glossary.run_chunk_glossary", fake_run_chunk_glossary)
    return calls


@pytest.fixture
def run(script, monkeypatch):
    """Run the pipeline for real, with whatever stubs the test installed."""
    install_chunk_glossary(monkeypatch)

    def _run(*argv):
        # Stage 5 is on by default. Keep every test offline: a built-in chonkie
        # tokenizer needs no vocabulary download, and the stub chapters are tiny.
        args = list(argv)
        if "--chunk-tokenizer" not in args:
            args += ["--chunk-tokenizer", "word", "--chunk-size", "50"]
        pipeline = script.Pipeline(script.build_parser().parse_args(args))
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

    monkeypatch.setattr(book_splitter.agent, "run_split", fake_split)
    return calls


def install_preprocess(monkeypatch, *, ok: bool = True):
    """Replace stage 2: report each chapter through the progress hook."""
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

    monkeypatch.setattr(book_splitter.preprocess_agent, "run_preprocess", fake_preprocess)
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

    def fake_run_dictionary(chapters_path, output_dir=None, **kwargs):  # noqa: ARG001
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
            unique = out.parent / "reduced-unique"
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

    monkeypatch.setattr(book_splitter.dictionary, "run_dictionary", fake_run_dictionary)
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

    # Stage 2 is switched off by default for now, so ask for it explicitly.
    code = run(str(SAMPLE), "--output-root", str(tmp_path / "out"), "--with-preprocess")
    captured = capsys.readouterr()

    assert code == 0
    markers = [
        "[1/7] RECON",
        "[2/7] SPLIT",
        "[3/7] SCAN",
        "[4/7] DICTIONARY",
        "[5/7] PREPROCESS",
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
    assert preprocess_calls == [tmp_path / "out" / SAMPLE.stem / "chapters"]
    assert (tmp_path / "out" / SAMPLE.stem / "chunks" / "_chunk_manifest.json").is_file()


@requires_sample
def test_skip_if_present_reuses_chapters_without_a_provider(
    script, tmp_path, monkeypatch, capsys, run
):
    """A reused split must not need an API key, or even resolve provider settings."""
    from book_splitter.paths import resolve_output_dir

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

    monkeypatch.setattr(script, "resolve_settings", explode)

    code = run(str(SAMPLE), "--output-root", str(tmp_path / "out"), "--skip-if-present")
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
    import book_splitter.core.settings as settings_module

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
    chapters = tmp_path / "out" / source.stem / "chapters"
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
def test_json_flag_appends_a_machine_readable_summary(script, tmp_path, monkeypatch, capsys, run):
    install_split(monkeypatch, writes={"01-a.md": "## 18\n"})
    dictionary_calls = install_dictionary(monkeypatch, writes={"01-a.md": "## 18\n"})
    install_preprocess(monkeypatch)

    code = run(str(SAMPLE), "--output-root", str(tmp_path / "out"), "--json", "--with-preprocess")
    out = capsys.readouterr().out

    assert code == 0
    payload = json.loads(out[out.rindex("\n{\n") :])
    assert payload["ok"] is True
    assert [phase["name"] for phase in payload["phases"]] == [
        "recon",
        "split",
        "scan",
        "dictionary",
        "preprocess",
        "chunk",
        "chunk_glossary",
    ]
    assert payload["phases"][1]["result"]["chapter_count"] == 1
    assert payload["reduced_dir"].endswith("reduced")
    assert payload["unique_dir"].endswith("reduced-unique")
    assert payload["chunks_dir"].endswith("chunks")
    assert payload["chunk_glossary_dir"].endswith("chunk-glossary")
    assert payload["log"].endswith("pipeline.log")
    assert dictionary_calls == [tmp_path / "out" / SAMPLE.stem / "chapters"]


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
def test_stage_two_is_switched_off_by_default(script, tmp_path, monkeypatch, capsys, run):
    """The temporary switch: a plain run stops after ``dictionary``."""
    if script.PREPROCESS_ENABLED:  # pragma: no cover - the switch was turned back on
        pytest.skip("stage 2 is enabled again; the temporary switch is gone")

    install_split(monkeypatch, writes=DAMAGED)
    install_dictionary(monkeypatch)
    preprocess_calls = install_preprocess(monkeypatch)

    code = run(str(SAMPLE), "--output-root", str(tmp_path / "out"))
    captured = capsys.readouterr()

    assert code == 0
    assert "PREPROCESS" not in captured.out
    assert "switched off" in captured.out
    # Stage 2 is dropped, so six phases are selected: scan is third, chunk glossary last.
    assert "[3/6] SCAN" in captured.out
    assert "[4/6] DICTIONARY" in captured.out
    assert "[5/6] CHUNK" in captured.out
    assert "[6/6] CHUNK_GLOSSARY" in captured.out
    assert "DONE" in captured.out
    assert preprocess_calls == []


@requires_sample
def test_stage_three_is_on_by_default_and_can_be_dropped(
    script, tmp_path, monkeypatch, capsys, run
):
    install_split(monkeypatch, writes=DAMAGED)
    dictionary_calls = install_dictionary(monkeypatch)

    code = run(str(SAMPLE), "--output-root", str(tmp_path / "out"))
    captured = capsys.readouterr()
    assert code == 0
    assert "DICTIONARY" in captured.out
    assert "language" in captured.out
    assert "ner time" in captured.out
    assert dictionary_calls == [tmp_path / "out" / SAMPLE.stem / "chapters"]

    dropped = tmp_path / "dropped"
    install_dictionary(monkeypatch)  # a fresh call list, patched over the old one
    code = run(str(SAMPLE), "--output-root", str(dropped), "--no-dictionary")
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

    code = run(str(SAMPLE), "--output-root", str(tmp_path / "out"))
    captured = capsys.readouterr()
    assert code == 0
    assert "CHUNK" in captured.out
    assert (tmp_path / "out" / SAMPLE.stem / "chunks" / "_chunk_manifest.json").is_file()

    dropped = tmp_path / "dropped"
    code = run(str(SAMPLE), "--output-root", str(dropped), "--no-chunk")
    captured = capsys.readouterr()
    assert code == 0
    assert "CHUNK" not in captured.out
    assert "chunking switched off" in captured.out
    assert not (dropped / SAMPLE.stem / "chunks").exists()


@requires_sample
def test_chunk_glossary_stage_is_on_by_default_and_can_be_dropped(
    script, tmp_path, monkeypatch, capsys, run
):
    install_split(monkeypatch, writes=DAMAGED)
    install_dictionary(monkeypatch)
    calls = install_chunk_glossary(monkeypatch)

    out = tmp_path / "out"
    code = run(str(SAMPLE), "--output-root", str(out))
    captured = capsys.readouterr()
    assert code == 0
    assert "[6/6] CHUNK_GLOSSARY" in captured.out
    assert calls == [out / SAMPLE.stem / "chunks"]
    assert (out / SAMPLE.stem / "chunk-glossary" / "_chunk_glossary_manifest.json").is_file()

    dropped = tmp_path / "dropped"
    dropped_calls = install_chunk_glossary(monkeypatch)
    code = run(str(SAMPLE), "--output-root", str(dropped), "--no-chunk-glossary")
    captured = capsys.readouterr()
    assert code == 0
    assert dropped_calls == []
    assert "[6/6] CHUNK_GLOSSARY" not in captured.out
    assert "chunk glossary switched off" in captured.out
    assert not (dropped / SAMPLE.stem / "chunk-glossary").exists()


@requires_sample
def test_a_missing_tagger_skips_stage_three_without_failing_the_run(
    script, tmp_path, monkeypatch, capsys, run
):
    install_split(monkeypatch, writes=DAMAGED)
    install_dictionary(monkeypatch, missing=True)

    code = run(str(SAMPLE), "--output-root", str(tmp_path / "out"))
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
    from book_splitter.glossary import GlossaryResult

    calls: list[Path] = []

    def fake_run_glossary(chapters_path, output_dir=None, **kwargs):
        calls.append(Path(chapters_path))
        if output_dir:
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
            sentences_dir=str(Path(chapters_path).parent / "reduced-unique"),
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

    monkeypatch.setattr("book_splitter.glossary.run_glossary", fake_run_glossary)
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

    code = run(str(SAMPLE), "--output-root", str(tmp_path / "out"), "--limit", "2")
    captured = capsys.readouterr()
    assert code == 0
    # The glossary is inside the dictionary phase; stage 6 is the only later
    # phase that mentions the glossary, and it runs after the chunks.
    assert "[4/6] DICTIONARY" in captured.out
    assert "[6/6] CHUNK_GLOSSARY" in captured.out
    assert "glossary — one chapter at a time" in captured.out
    assert "entries" in captured.out
    assert glossary_calls == [tmp_path / "out" / SAMPLE.stem / "chapters"]
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
    )
    captured = capsys.readouterr()
    assert code == 0
    assert no_gloss_calls == []
    assert "glossary — one chapter at a time" not in captured.out
    assert "--no-glossary is set" in captured.out


@requires_sample
def test_a_glossary_dry_run_skips_the_phase_without_spending(
    script, tmp_path, monkeypatch, capsys, run
):
    install_split(monkeypatch, writes=DAMAGED)
    install_dictionary(monkeypatch, writes={"01-predgovor.md": "Prvi deo teksta.\n"})
    glossary_calls = install_glossary(monkeypatch)

    code = run(
        str(SAMPLE), "--output-root", str(tmp_path / "out"), "--with-glossary", "--dry-run"
    )
    captured = capsys.readouterr()
    assert code == 0
    assert glossary_calls == []
    assert "glossary skipped: a --dry-run does not spend tokens" in captured.out


@requires_sample
def test_a_failed_glossary_chapter_stops_the_run(script, tmp_path, monkeypatch, capsys, run):
    install_split(monkeypatch, writes=DAMAGED)
    install_dictionary(monkeypatch, writes={"01-predgovor.md": "Prvi deo teksta.\n"})
    install_glossary(monkeypatch, ok=False)

    code = run(str(SAMPLE), "--output-root", str(tmp_path / "out"), "--with-glossary")
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

    def invoke(self, state, config=None):  # noqa: ARG002 - mirrors the graph API
        return dict(self.final)


def _chapter_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "chapters"
    folder.mkdir()
    (folder / "01-a.md").write_text("## A\n\ntext\n", encoding="utf-8")
    (folder / "02-b.md").write_text("## B\n\ntext\n", encoding="utf-8")
    return folder


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
        dry_run=True,
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
