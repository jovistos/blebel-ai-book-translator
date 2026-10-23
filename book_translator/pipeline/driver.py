"""The pipeline driver: phase order, stop/start, records and reporting.

This module is the one orchestrator. ``scripts/run_pipeline.py``, the
``book_translator pipeline`` subcommand and any library caller all reach the same
driver; the phase bodies, the console, the run log and the JSON report live here
and nowhere else.

Every runner call is built by
:meth:`book_translator.pipeline.options.StageOptions.call`, so the CLI, the
pipeline and the tests share one kwargs contract. The parsed namespace is kept
only for the narration and the few diagnostics that are not runner options.
"""

from __future__ import annotations

import shutil
import sys
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

from book_translator.conversion import (
    classify_source,
)
from book_translator.core.errors import ConfigError, ToolError
from book_translator.core.report import human_seconds
from book_translator.core.settings import (
    STAGES,
    load_dotenv_files,
    load_llm_config,
    normalize_stage,
    parse_llm_overrides,
    resolve_settings,
    resolve_translation_settings,
)
from book_translator.documents.chapters import (
    chapters_dir,
    discover_chapters,
    file_sha256,
    has_chapter_files,
    has_chunk_files,
    has_postprocessed_chapter_files,
    has_translated_chapter_files,
    has_translation_files,
    load_split_manifest,
)
from book_translator.documents.window import (
    chapter_window_limit_for_path,
    normalize_limit,
)
from book_translator.paths import (
    repo_root,
    resolve_book_path,
)

from .console import Console
from .execution import PhaseExecutionMixin, PhaseFailed
from .options import StageRuntime
from .paths import StagePaths
from .phases import NEEDS_CHAPTERS, TITLES
from .report import PhaseRecord, PhaseState
from .reporter import (
    PipelineReporter,
    build_summary_json,
    derive_phase_state,
    find_artifact_files,
    print_artifacts,
    print_finish_header,
    print_next_steps,
    print_phase_table,
)
from .sources import resolve_sources

ROOT = repo_root()

def rel(path: str | Path) -> str:
    """A path relative to the repository root when it lives inside it."""
    resolved = Path(path).expanduser()
    try:
        return str(resolved.resolve().relative_to(ROOT))
    except ValueError:
        return str(resolved)

def has_chapters(path: Path) -> bool:
    """True when ``path`` holds at least one chapter file."""
    return has_chapter_files(path)

class PipelineDriver(PhaseExecutionMixin):
    """Runs the selected phases in order and reports on each one."""

    def __init__(self, options: Any, *, ns: Any = None) -> None:
        self.options = options
        self.args: Any = ns if ns is not None else getattr(options, "ns", None)
        assert self.args is not None
        args = self.args
        # One source of truth for the switches and the window: the shared
        # PhaseSelection and StageOptions, not a second copy of the flag logic.
        self.preprocess_enabled = options.selection.preprocess
        self.dictionary_enabled = options.selection.dictionary
        self.glossary_enabled = options.selection.glossary
        self.chunk_enabled = options.selection.chunk
        self.chunk_glossary_enabled = options.selection.chunk_glossary
        self.translate_enabled = options.selection.translate
        self.matter_enabled = options.selection.matter
        self.concat_enabled = options.selection.concatenate
        self.postprocess_enabled = options.selection.postprocess
        self.bind_enabled = options.selection.bind
        self.epub_enabled = options.selection.epub
        self.first_chapters = normalize_limit(options.stage.first_chapters)
        self.translate_first_chapters = (
            normalize_limit(options.stage.translate_first_chapters)
            or self.first_chapters
        )
        self.translate_limit = (
            options.stage.translate_limit
            if options.stage.translate_limit is not None
            else options.stage.limit
        )
        self.chapter_limit: int | None = None
        self.records: list[PhaseRecord] = []
        self.exit_code = 0
        self.started = time.monotonic()
        self._settings: Any = None
        self._translation_settings: Any = None
        self._stage_settings: dict[str, Any] = {}
        self._llm_config: Any = None
        # ``--llm <stage>.<field>=VALUE``: parsed once, applied per phase.
        self._llm_overrides = parse_llm_overrides(getattr(args, "llm", None))
        self._progress_lock = threading.Lock()
        self._bar_stop = threading.Event()
        self._bar_thread: threading.Thread | None = None
        self._bar_state: dict[str, Any] = {"active": False, "total": 0, "done": 0}
        self._last_tick = self.started
        self._count_width = 1
        self._name_width = 24

        load_dotenv_files()  # so the banner's model label matches the real resolution
        args = self.args
        self.source = resolve_book_path(args.book, input_dir=args.in_dir)
        # Stage 0 converts a non-Markdown source before any phase reads its lines.
        # ``self.source`` is replaced by the converted Markdown once it has run;
        # the original path is kept for the banner and the JSON summary.
        self.original_source = self.source
        # Validation up front: an unsupported format is one clear error from
        # ``main`` instead of a traceback inside the banner.
        self.source_kind = classify_source(self.source)
        self.convert_needed = self.source_kind != "markdown"
        self._convert_temp_dir: Path | None = None
        self.paths = StagePaths.for_pipeline(
            args,
            source=self.source,
            original_source=self.original_source,
            input_dir=Path(args.in_dir).expanduser() if getattr(args, "in_dir", None) else None,
            output_root=Path(args.output_root).expanduser()
            if getattr(args, "output_root", None)
            else None,
        )
        self._refresh_chapter_limit()

        log_path = None
        if not args.no_log:
            log_path = (
                Path(args.log_file).expanduser()
                if args.log_file
                else self.book_dir / "pipeline.log"
            )
        self.console = Console(
            log_path,
            debug=args.debug,
            quiet=bool(getattr(args, "quiet", False) or getattr(args, "json", False)),
        )
        self.reporter = PipelineReporter(self)

    # -- path aliases (kept for the phase bodies while they migrate) -------- #
    @property
    def markdown_out(self) -> Path:
        return self.paths.markdown_dir

    @property
    def book_dir(self) -> Path:
        return self.paths.book_dir

    @property
    def chapters_dir(self) -> Path:
        return self.paths.chapters_dir

    @property
    def preprocessed_out(self) -> Path:
        return self.paths.preprocessed_dir

    @property
    def reduced_out(self) -> Path:
        return self.paths.reduced_dir

    @property
    def unique_out(self) -> Path:
        return self.paths.unique_dir

    @property
    def glossary_out(self) -> Path:
        return self.paths.glossary_dir

    @property
    def text_in(self) -> Path | None:
        value = getattr(self.args, "text_in", None)
        return Path(value).expanduser() if value else None

    @property
    def chunk_in(self) -> Path | None:
        value = getattr(self.args, "chunk_in", None)
        return Path(value).expanduser() if value else None

    @property
    def chunks_out(self) -> Path:
        return self.paths.chunks_dir

    @property
    def chunk_glossary_out(self) -> Path:
        return self.paths.chunk_glossary_dir

    @property
    def translations_out(self) -> Path:
        return self.paths.translations_dir

    @property
    def translation_prompts_out(self) -> Path:
        return self.paths.translation_prompts_dir

    @property
    def matter_out(self) -> Path:
        return self.paths.matter_dir

    @property
    def matter_prompts_out(self) -> Path:
        return self.paths.matter_prompts_dir

    @property
    def translated_chapters_out(self) -> Path:
        return self.paths.translated_chapters_dir

    @property
    def postprocess_out(self) -> Path:
        return self.paths.postprocess_dir

    @property
    def translated_book_out(self) -> Path:
        return self.paths.translated_book_path

    @property
    def epub_out(self) -> Path:
        return self.paths.epub_path

    def _stage_runtime(
        self,
        *,
        settings: Any = None,
        settings_factory: Any = None,
        progress: Any = None,
        chapter_start: Any = None,
        chunk_start: Any = None,
        chapter_limit: int | None = None,
        chapter_files: Any = None,
        sentences_dir: str | None = None,
        model_identity: str = "",
        postprocess: Path | None = None,
        source: Path | None = None,
        original_source: Path | None = None,
        conversion_target: Path | None = None,
        conversion_live: bool = False,
    ) -> StageRuntime:
        """Resolve the inputs and the runtime seams one phase's runner needs."""
        sources = resolve_sources(
            self.args,
            self.paths,
            chapter_limit=chapter_limit,
            preprocess_enabled=self.preprocess_enabled,
        )
        if postprocess is not None:
            sources = replace(sources, postprocess=postprocess)
        return StageRuntime(
            paths=self.paths,
            sources=sources,
            settings=settings,
            settings_factory=settings_factory,
            progress=progress,
            chapter_start=chapter_start,
            chunk_start=chunk_start,
            chapter_limit=chapter_limit,
            chapter_files=chapter_files,
            sentences_dir=sentences_dir,
            model_identity=model_identity,
            source=source or self.source,
            original_source=original_source or self.original_source,
            conversion_target=conversion_target,
            conversion_live=conversion_live,
        )

    # -- setup ------------------------------------------------------------- #
    @property
    def llm_config(self):
        """The per-stage model file, read once and only when a phase needs it."""
        if self._llm_config is None:
            self._llm_config = load_llm_config(self.args.llm_config)
        return self._llm_config

    def _shared_model_args(self) -> dict[str, Any]:
        """The flags shared by every stage.

        The resolver applies them only to a stage whose provider matches the one
        they were written for, so ``--model`` never leaks a cloud model id into a
        stage that a config section pinned to a local server.
        """
        return {
            "provider": self.args.provider,
            "model": self.args.model,
            "base_url": self.args.base_url,
            "api_key": self.args.api_key,
            "temperature": self.args.temperature,
        }

    @property
    def settings(self):
        """Stage 1's model connection (the shared default stage)."""
        return self.settings_for("split")

    def settings_for(self, stage: str):
        """One stage's connection, resolved lazily, cached, and keyless until needed.

        ``stage`` is one of :data:`STAGES` (``split``, ``preprocess``,
        ``dictionary``, ``matter``, ``translate``, ``postprocess``). Each stage
        reads its own ``[llm.<stage>]`` section from ``book_translator.toml``.
        """
        canonical = normalize_stage(stage)
        if canonical is None:
            raise ConfigError(
                f"unknown pipeline stage {stage!r}; choose one of {', '.join(STAGES)}"
            )
        if canonical == "translate":
            return self.translation_settings
        if canonical == "split" and self._settings is not None:
            return self._settings
        if canonical in self._stage_settings:
            return self._stage_settings[canonical]
        resolved = resolve_settings(
            stage=canonical,
            shared=self._shared_model_args(),
            config=self.llm_config,
            overrides=self._llm_overrides.get(canonical),
        )
        if canonical == "split":
            self._settings = resolved
        self._stage_settings[canonical] = resolved
        return resolved

    @property
    def translation_settings(self):
        """Stage 7's model: DeepSeek, or the local LM Studio server.

        Its ``[llm.translate]`` section is independent of every other stage, so
        switching translation to LM Studio never moves the split, dictionary,
        matter or postprocess stages onto the local model. Resolution stays lazy
        so a fully resumed translation needs no key and a run without translation
        never touches it.
        """
        if self._translation_settings is None:
            self._translation_settings = resolve_translation_settings(
                provider=self.args.translate_provider,
                model=self.args.translate_model,
                base_url=self.args.translate_base_url,
                api_key=self.args.translate_api_key,
                temperature=self.args.translate_temperature,
                shared=self._shared_model_args(),
                config=self.llm_config,
                overrides=self._llm_overrides.get("translate"),
            )
        return self._translation_settings

    def _refresh_chapter_limit(self) -> None:
        """Turn the body-chapter window into the file count each stage takes.

        Called once at construction (for existing chapters) and again after the
        split phase, because a fresh run does not know whether the split will
        write ``00-front-matter.md`` until the phase has run. The effective limit
        deliberately includes any front matter *before* chapter 1 while not
        counting it as one of the N body chapters.
        """
        self.chapter_limit = chapter_window_limit_for_path(
            self.chapters_dir,
            self.first_chapters,
            getattr(self.args, "limit", None),
        )

    def _window_text(self) -> str | None:
        """Human label for the active window; the file count stays internal."""
        if self.first_chapters is not None:
            return f"first {self.first_chapters} body chapter(s)"
        if self.chapter_limit is not None:
            return f"first {self.chapter_limit} chapter(s)"
        return None

    def _cleaned_window(self) -> Path | None:
        """Stage 2's folder when it covers exactly the chapters this run reads.

        ``--first-chapters`` makes stage 2 clean only the window, so
        ``02-preprocessed/`` is a prefix of ``01-chapters/``, not a copy. It is
        usable only when every chapter in the window has a cleaned file, the
        stage-2 manifest was written for the current chapter set (same source
        line counts) and neither the source chapter nor the cleaned file changed
        since it was verified. Anything else falls back to the raw chapters
        rather than silently cleaning some chapters and not others.
        """
        folder = self.preprocessed_out
        if not folder.is_dir() or not has_chapter_files(folder):
            return None
        manifest = load_split_manifest(folder)
        if not manifest or not manifest.get("ok") or manifest.get("dry_run"):
            return None
        entries = {
            str(entry.get("filename")): entry
            for entry in manifest.get("files") or []
            if isinstance(entry, dict) and entry.get("filename")
        }
        if not entries:
            return None
        try:
            chapters = discover_chapters(self.chapters_dir)
        except (ToolError, OSError):
            return None
        window = chapters[: self.chapter_limit] if self.chapter_limit else chapters
        if not window:
            return None
        for chapter in window:
            entry = entries.get(chapter.filename)
            if entry is None or not (folder / chapter.filename).is_file():
                return None
            recorded_lines = entry.get("source_lines")
            if recorded_lines is not None and int(recorded_lines) != chapter.line_count:
                return None
            recorded_source = str(entry.get("source_sha256") or "")
            if recorded_source and chapter.sha256 and recorded_source != chapter.sha256:
                return None
            recorded_clean = str(entry.get("sha256") or "")
            if recorded_clean and file_sha256(folder / chapter.filename) != recorded_clean:
                return None
        return folder

    def _text_choice(self) -> tuple[Path, str]:
        """The folder stages 3-5 read, plus why it was chosen."""
        if self.text_in is not None:
            return chapters_dir(self.text_in), "explicit --text-in"
        if not self.preprocess_enabled:
            return self.chapters_dir, "raw chapters (--no-preprocess)"
        cleaned = self._cleaned_window()
        if cleaned is not None:
            return cleaned, "cleaned by stage 2"
        if has_chapter_files(self.preprocessed_out):
            return self.chapters_dir, "cleaned text does not cover this window"
        return self.chapters_dir, "raw chapters"

    def text_input(self) -> Path:
        """The chapter folder stages 3-5 read: the cleaned text when usable."""
        return self._text_choice()[0]

    def chunk_input(self) -> Path:
        """Stage 5's input: ``--chunk-in`` wins, else the resolved text folder."""
        if self.chunk_in is not None:
            return chapters_dir(self.chunk_in)
        return self.text_input()

    def _kv_text_input(self, console: Any, input_dir: Path) -> None:
        """Report the text folder and, when relevant, why it is not the cleaned one."""
        label = rel(input_dir)
        if input_dir == self.preprocessed_out:
            label += "  (cleaned by stage 2)"
        elif (
            self.chunk_in is None
            and self.text_in is None
            and self.preprocess_enabled
            and has_chapter_files(self.preprocessed_out)
        ):
            label += "  (cleaned text does not cover this window)"
        console.kv("input", label)

    def _selected_phases(self) -> list[str]:
        return list(
            self.options.selection.selected(convert_needed=self.convert_needed)
        )

    def _switched_off_phases(self) -> list[str]:
        """Phases in the requested window that a switch dropped."""
        return list(
            self.options.selection.switched_off(convert_needed=self.convert_needed)
        )

    def close(self) -> None:
        self.console.close()
        if self._convert_temp_dir is not None:
            shutil.rmtree(self._convert_temp_dir, ignore_errors=True)
            self._convert_temp_dir = None

    # -- the run ----------------------------------------------------------- #
    def run(self) -> int:
        phases = self._selected_phases()
        self._banner(phases)
        for name in self._switched_off_phases():
            if name == "concatenate" and not self.translate_enabled:
                self.console.info(
                    "  · concatenate: switched off — add --with-translate to assemble "
                    "the translation, or --start-at concatenate"
                )
                continue
            flag = {
                "preprocess": "--no-preprocess",
                "dictionary": "--with-dictionary",
                "chunk": "--no-chunk",
                "chunk_glossary": "--no-chunk-glossary",
                "matter": "--with-matter / --no-matter",
                "translate": "--with-translate",
                "concatenate": "--no-concat",
                "postprocess": "--with-postprocess",
                "bind": "--no-bind",
                "epub": "--no-epub",
            }.get(name, "--with-" + name)
            self.console.info(f"  · {name}: switched off — skipped (see {flag})")
        for index, name in enumerate(phases, start=1):
            record = PhaseRecord(name=name)
            self.records.append(record)
            title = TITLES[name]
            if name == "split" and getattr(self.args, "map", None):
                title = "stage 1 — a supplied map routes, the lines are copied verbatim"
            self.console.phase(index, len(phases), name, title)

            skip_for_dry_run = (
                name in NEEDS_CHAPTERS and self.args.dry_run and not has_chapters(self.chapters_dir)
            )
            if skip_for_dry_run:
                record.ok = True
                record.summary = "skipped — a dry run writes no chapters"
                self.console.info("  · skipped: a --dry-run split writes no chapter files to scan")
                continue

            if name == "chunk_glossary" and not has_chunk_files(self.chunks_out):
                record.ok = True
                record.summary = "skipped — no chunks on disk to annotate"
                self.console.info(
                    "  · skipped: no chunk files — run the chunk phase (or a real run) first"
                )
                continue

            if name == "matter" and not has_chunk_files(self.chunks_out):
                record.ok = True
                record.summary = "skipped — no chunks on disk to localize"
                self.console.info(
                    "  · skipped: no chunk files — run the chunk phase (or a real run) first"
                )
                continue

            if name == "translate":
                from book_translator.translation.runner import has_chunk_glossary_files

                if not has_chunk_files(self.chunks_out):
                    record.ok = True
                    record.summary = "skipped — no chunks on disk to translate"
                    self.console.info(
                        "  · skipped: no chunk files — run the chunk phase first"
                    )
                    continue
                if not has_chunk_glossary_files(self.chunk_glossary_out):
                    record.ok = True
                    record.summary = "skipped — no per-chunk glossary to send"
                    self.console.info(
                        "  · skipped: no per-chunk glossary — run the chunk_glossary phase first"
                    )
                    continue

            if name == "concatenate" and not has_translation_files(self.translations_out):
                record.ok = True
                record.summary = "skipped — no translated chunks to assemble"
                self.console.info(
                    "  · skipped: no translated chunks — run the translate phase first "
                    "(or --start-at concatenate after a translated run)"
                )
                continue

            if name == "postprocess" and not has_translated_chapter_files(
                self.translated_chapters_out
            ):
                record.ok = True
                record.summary = "skipped — no translated chapters to postprocess"
                self.console.info(
                    "  · skipped: no translated chapters — run the translate/concatenate "
                    "phases first (or --start-at postprocess after a translated run)"
                )
                continue

            if name == "bind" and not has_postprocessed_chapter_files(self.postprocess_out):
                record.ok = True
                record.summary = "skipped — no postprocessed chapters to bind"
                self.console.info(
                    "  · skipped: no postprocessed chapters — run the postprocess phase first"
                )
                continue

            tick = time.monotonic()
            try:
                summary, payload = getattr(self, f"phase_{name}")()
            except PhaseFailed as exc:
                record.summary, record.error = exc.summary, str(exc)
            except ConfigError as exc:
                record.error, self.exit_code = str(exc), 2
            except (ToolError, FileNotFoundError) as exc:
                record.error = str(exc)
            except KeyboardInterrupt:  # pragma: no cover - interactive only
                self.console.error("\n  interrupted")
                self.exit_code = 130
                break
            except Exception as exc:  # a crash must still be reported and logged
                record.error = f"{type(exc).__name__}: {exc}"
                self.console.debug(traceback.format_exc())
                if self.args.debug:
                    traceback.print_exc()
            finally:
                record.seconds = time.monotonic() - tick

            if record.error:
                self.console.error(f"  ✗ {name}: {record.error}")
                break
            record.ok, record.summary, record.payload = True, summary, payload
            if name == "split":
                # The split may have just written the front matter; recompute the
                # chapter window now that the real file list exists.
                self._refresh_chapter_limit()
            self.console.info(f"  ✓ {name} finished in {human_seconds(record.seconds)}")

            if (
                getattr(self.args, "interactive", False)
                and not getattr(self.args, "dry_run", False)
                and sys.stdin.isatty()
                and name in {"glossary", "matter", "postprocess"}
            ):
                proceed = self._checkpoint_pause(name, record)
                if not proceed:
                    self.console.info(f"  · aborted at {name} checkpoint by user")
                    self.exit_code = 130
                    break

        return self._finish()

    def _checkpoint_pause(self, name: str, record: PhaseRecord) -> bool:
        """Interactive checkpoint pause allowing human review and intervention."""
        import json

        self.console.emit("")
        rich_c = getattr(self.console, "_rich_console", None)
        if getattr(self.console, "is_rich", False) and rich_c is not None:
            from rich.panel import Panel
            from rich.table import Table

            if name == "glossary":
                glossary_file = self.glossary_out / "glossary.json"
                terms_count = 0
                sample_rows: list[dict[str, Any]] = []
                if glossary_file.exists():
                    try:
                        data = json.loads(glossary_file.read_text(encoding="utf-8"))
                        entries = data if isinstance(data, list) else data.get("entries", [])
                        terms_count = len(entries)
                        sample_rows = entries[:15]
                    except Exception:
                        pass

                tbl = Table(title=f"Checkpoint: Stage 4 Glossary ({terms_count} terms generated)", border_style="cyan")
                tbl.add_column("Term", style="bold")
                tbl.add_column("Category", style="dim")
                tbl.add_column("Translation", style="bold green")
                tbl.add_column("Context", style="dim")
                for e in sample_rows:
                    tbl.add_row(
                        str(e.get("term", "")),
                        str(e.get("category", "")),
                        str(e.get("translation", "")),
                        str(e.get("context", ""))[:50],
                    )
                rich_c.print(tbl)
                rich_c.print(
                    Panel(
                        f"[bold yellow]Checkpoint Pause[/bold yellow]\n"
                        f"Glossary saved to: [cyan]{rel(glossary_file)}[/cyan]\n\n"
                        f"[bold][c][/bold] Continue to chunking / translation\n"
                        f"[bold][e][/bold] Pause to edit glossary in your editor\n"
                        f"[bold][a][/bold] Abort pipeline",
                        border_style="yellow",
                    )
                )
            elif name == "matter":
                title_file = self.matter_out / "title.json"
                fm_file = self.matter_out / "00-front-matter.md"
                content_preview = ""
                if title_file.exists():
                    content_preview += f"Title metadata: {title_file.read_text(encoding='utf-8')[:200]}\n"
                if fm_file.exists():
                    content_preview += f"Front matter preview:\n{fm_file.read_text(encoding='utf-8')[:300]}\n..."

                rich_c.print(
                    Panel(
                        f"[bold yellow]Checkpoint: Stage 6b Matter Localized[/bold yellow]\n\n"
                        f"{content_preview}\n\n"
                        f"[bold][c][/bold] Continue to chunk translation\n"
                        f"[bold][a][/bold] Abort pipeline",
                        border_style="yellow",
                    )
                )
            elif name == "postprocess":
                rich_c.print(
                    Panel(
                        f"[bold yellow]Checkpoint: Stage 9 Headings Postprocessed[/bold yellow]\n"
                        f"Postprocessed folder: [cyan]{rel(self.postprocess_out)}[/cyan]\n\n"
                        f"[bold][c][/bold] Continue to bind into final book & EPUB\n"
                        f"[bold][a][/bold] Abort pipeline",
                        border_style="yellow",
                    )
                )
        else:
            print(f"\n--- Checkpoint: {name} completed ---")
            print("[c] Continue / [e] Edit file / [a] Abort")

        while True:
            try:
                choice = input("Select [c/e/a] (default 'c'): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return False
            if choice in {"", "c", "continue", "y", "yes"}:
                return True
            elif choice in {"a", "abort", "n", "no", "q", "quit"}:
                return False
            elif choice in {"e", "edit"}:
                target_path = self.glossary_out / "glossary.json" if name == "glossary" else self.book_dir
                print(f"You can now edit: {target_path}")
                try:
                    input("Press [Enter] when finished editing to resume...")
                except (EOFError, KeyboardInterrupt):
                    return False
                return True
            else:
                print("Invalid choice. Please enter 'c', 'e', or 'a'.")



    def _banner(self, phases: list[str]) -> None:
        self.reporter.banner(phases)

    def _finish(self) -> int:
        total = time.monotonic() - self.started
        aborted = any(not record.ok for record in self.records)
        return self.reporter.finish(total, aborted)

    def _print_finish_header(self, console: Any, total: float, aborted: bool) -> None:
        print_finish_header(console, total, aborted, len(self.records), len(self._selected_phases()))

    def _print_phase_table(self, console: Any) -> None:
        print_phase_table(console, self.records)

    def _print_artifacts(self, console: Any) -> None:
        print_artifacts(self, console)

    def _phase_state(self) -> PhaseState:
        return derive_phase_state(self.records)

    def _print_next_steps(self, console: Any, state: PhaseState, aborted: bool) -> None:
        print_next_steps(self, console, state, aborted)

    def _artifact_files(self) -> list[tuple[str, str | Path]]:
        return find_artifact_files(self)

    def _summary_json(self, total: float) -> dict[str, Any]:
        return build_summary_json(self, total)
