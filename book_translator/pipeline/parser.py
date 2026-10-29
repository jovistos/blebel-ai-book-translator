"""The pipeline command's argument surface.

The whole-pipeline parser used to live in ``scripts/run_pipeline.py``. It is
here so ``python -m book_translator pipeline`` and the compatibility script share
one parser, one option normalization and one driver.
"""

from __future__ import annotations

import argparse
import sys

from ..core.constants import (
    CHUNK_GLOSSARY_SCORERS,
    DEFAULT_CHUNK_GLOSSARY_CANDIDATES,
    DEFAULT_CHUNK_GLOSSARY_MIN_TERM_CHARS,
    DEFAULT_CHUNK_GLOSSARY_SCORER,
    DEFAULT_CHUNK_GLOSSARY_THRESHOLD,
    DEFAULT_CHUNK_TOKENIZER,
    DEFAULT_CHUNK_TOKENS,
    DEFAULT_FRONT_MATTER_LINES,
    DEFAULT_GPU_INDEX,
    DEFAULT_KEEP_ENTITY_TYPES,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MIN_CHAPTERS,
    DEFAULT_MIN_SENTENCE_WORDS,
    DEFAULT_RECURSION_LIMIT,
    DEFAULT_SOURCE_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
    DEFAULT_TRANSLATION_ATTEMPTS,
    DEFAULT_TRANSLATION_CONTEXT_CHARS,
    DEFAULT_TRANSLATION_MAX_TOKENS,
    MAX_FEEDBACK_LOOPS,
    TOC_SCAN_LINES,
    TRANSLATION_GLOSSARY_SCOPES,
    TRANSLATION_METHODS,
)
from ..core.errors import ConfigError, ToolError
from ..core.settings import (
    LLM_CONFIG_FILENAME,
    PROVIDERS,
    STAGES,
    TRANSLATION_CHOICES,
)
from ..paths import default_input_dir, default_output_dir
from .driver import PipelineDriver
from .options import PHASES, PipelineOptions


def build_parser(prog: str = "book_translator pipeline") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "book",
        nargs="?",
        default=None,
        help="source document — filename or path (default: the only book in data/input/)",
    )
    parser.add_argument(
        "--in-dir", default=None, help=f"folder holding the source (default: {default_input_dir()})"
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=f"root of the generated artifacts (default: {default_output_dir()})",
    )
    parser.add_argument(
        "--out",
        "-o",
        default=None,
        help="stage-1 chapters folder (default: <output-root>/<book>/01-chapters)",
    )
    parser.add_argument(
        "--map",
        default=None,
        metavar="FILE",
        help="stage 2: replay a chapter map from a JSON file instead of running the agent "
        "(no provider, no key, no network)",
    )
    parser.add_argument(
        "--preprocessed-out", default=None, help="stage-3 folder (default: a sibling of 02-chapters/)"
    )
    parser.add_argument(
        "--reduced-out", default=None, help="stage-4 folder (default: a sibling of 02-chapters/)"
    )
    parser.add_argument(
        "--unique-out",
        default=None,
        help="stage-4 first-mention folder (default: a sibling of the reduced folder, "
        "04-reduced-unique/)",
    )
    parser.add_argument(
        "--text-in",
        default=None,
        help="chapters folder stages 4-5 read as text (default: 03-preprocessed/ when "
        "stage 3 covers the window, else 02-chapters/); point it at 02-chapters/ to "
        "force the raw text",
    )
    parser.add_argument(
        "--chunk-in",
        default=None,
        help="stage-5 input folder, overriding --text-in (default: the text folder "
        "above, i.e. the cleaned 03-preprocessed/ when stage 3 covered the window)",
    )
    parser.add_argument(
        "--chunks-out",
        default=None,
        help="stage-5 output folder (default: <book>/05-chunks/)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_TOKENS,
        help=f"stage-5 maximum chunk length in tokens (default: {DEFAULT_CHUNK_TOKENS})",
    )
    parser.add_argument(
        "--chunk-tokenizer",
        default=DEFAULT_CHUNK_TOKENIZER,
        help="stage-5 tokenizer used to measure a chunk: a tiktoken encoding "
        f"(default: {DEFAULT_CHUNK_TOKENIZER}), 'word', 'character', 'byte', or a "
        "Hugging Face tokenizer name; falls back to a local approximation offline",
    )
    parser.add_argument(
        "--chunk-glossary-out",
        default=None,
        help="stage 6: where the per-chunk glossary files go "
        "(default: <book>/06-chunk-glossary)",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=DEFAULT_CHUNK_GLOSSARY_THRESHOLD,
        help="stage 6: minimum rapidfuzz score as a percentage "
        f"(default: {DEFAULT_CHUNK_GLOSSARY_THRESHOLD}); lower is more permissive",
    )
    parser.add_argument(
        "--match-scorer",
        choices=CHUNK_GLOSSARY_SCORERS,
        default=DEFAULT_CHUNK_GLOSSARY_SCORER,
        help=f"stage 6: rapidfuzz scorer (default: {DEFAULT_CHUNK_GLOSSARY_SCORER})",
    )
    parser.add_argument(
        "--match-candidates",
        type=int,
        default=DEFAULT_CHUNK_GLOSSARY_CANDIDATES,
        help="stage 6: glossary candidates retrieved per term "
        f"(default: {DEFAULT_CHUNK_GLOSSARY_CANDIDATES})",
    )
    parser.add_argument(
        "--min-term-chars",
        type=int,
        default=DEFAULT_CHUNK_GLOSSARY_MIN_TERM_CHARS,
        help="stage 6: drop extracted terms shorter than this "
        f"(default: {DEFAULT_CHUNK_GLOSSARY_MIN_TERM_CHARS})",
    )
    parser.add_argument(
        "--no-chunk-glossary",
        action="store_true",
        help="drop stage 6 (per-chunk NER + glossary retrieval) from this run",
    )
    parser.add_argument(
        "--translations-out",
        default=None,
        help="stage 8: where the translated chunks go (default: <book>/08-translations)",
    )
    parser.add_argument(
        "--translation-prompts-out",
        default=None,
        help="stage 8: where the per-chunk prompt/input records go "
        "(default: <book>/08-translation-prompts)",
    )
    parser.add_argument(
        "--matter-out",
        default=None,
        help="stage 7: where the front/back matter plan, manifest and trace go "
        "(default: <book>/07-matter)",
    )
    parser.add_argument(
        "--matter-prompts-out",
        default=None,
        help="stage 7: where the matter agent's whole prompt/input records go "
        "(default: <book>/07-matter-prompts)",
    )
    parser.add_argument(
        "--with-matter",
        action="store_true",
        help="run the front/back matter localization agent even when translation is "
        "not selected (it is selected with --with-translate by default)",
    )
    parser.add_argument(
        "--no-matter",
        action="store_true",
        help="drop the front/back matter localization stage; stage 8 then skips the "
        "matter chunks and stages 9-11 do not assemble them",
    )
    parser.add_argument(
        "--matter-limit",
        type=int,
        default=None,
        help="stage 7: localize only the first N matter chunks (a cheap test of the "
        "plan and the localization loop)",
    )
    parser.add_argument(
        "--translated-chapters-out",
        default=None,
        help="stage 9: where the translated chapter files go "
        "(default: <book>/09-translated-chapters)",
    )
    parser.add_argument(
        "--postprocess-out",
        default=None,
        help="stage 10: where the postprocessed chapter files go "
        "(default: <book>/10-postprocessed-chapters)",
    )
    parser.add_argument(
        "--book-out",
        default=None,
        help="stage 11: where the single Markdown book goes "
        "(default: <book>/11-translated-book.md)",
    )
    parser.add_argument(
        "--with-postprocess",
        action="store_true",
        help="run stage 10 even when neither translation nor assembly is selected "
        "(over an existing translated-chapters folder)",
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="drop stage 10 (heading/TOC postprocessing) from this run",
    )
    parser.add_argument(
        "--no-bind",
        action="store_true",
        help="drop stage 11 (single Markdown book concatenation) from this run",
    )
    parser.add_argument(
        "--epub-out",
        default=None,
        help="stage 12: where the EPUB 3 book goes "
        "(default: <book>/12-translated-book.epub)",
    )
    parser.add_argument(
        "--no-epub",
        action="store_true",
        help="drop stage 12 (EPUB 3 packaging) from this run",
    )
    parser.add_argument(
        "--epub-cover",
        default=None,
        help="stage 12: cover picture — a path, or a file name inside the book's "
        "01-markdown/images/ folder",
    )
    parser.add_argument("--epub-title", default=None, help="stage 12: override the title")
    parser.add_argument("--epub-author", default=None, help="stage 12: override the author")
    parser.add_argument(
        "--epub-language",
        default=None,
        help="stage 12: override the language, an ISO 639 code",
    )
    parser.add_argument(
        "--first-chapters",
        "--chapters",
        dest="first_chapters",
        type=int,
        default=None,
        help="run every body stage after splitting over only the first N body chapters; "
        "front/back matter does not count as a chapter, and splitting itself "
        "always runs over the whole book (the matter stage is never windowed)",
    )
    parser.add_argument(
        "--translate-first-chapters",
        type=int,
        default=None,
        help="stage 8/9: translate and assemble only the first N body chapters "
        "(default: --first-chapters)",
    )
    parser.add_argument(
        "--no-concat",
        action="store_true",
        help="drop stage 9 (concatenating the translated chunks into chapters) "
        "from this run",
    )
    parser.add_argument(
        "--with-translate",
        action="store_true",
        help="run stage 8 (translate every chunk) — off by default because it is one "
        "paid model call per chunk",
    )
    parser.add_argument(
        "--no-translate",
        action="store_true",
        help="drop stage 8 from this run (it is off unless --with-translate is given)",
    )
    parser.add_argument(
        "--translate-limit",
        type=int,
        default=None,
        help="stage 8: translate only the first N chunks (defaults to --limit); the "
        "cheap way to test the whole translation path",
    )
    parser.add_argument(
        "--translate-glossary-scope",
        choices=TRANSLATION_GLOSSARY_SCOPES,
        default="chunk",
        help="stage 8: send the matched terms stage 6 attached to each chunk (default) "
        "or the whole glossary.json with every chunk",
    )
    parser.add_argument(
        "--translate-attempts",
        type=int,
        default=DEFAULT_TRANSLATION_ATTEMPTS,
        help="stage 8: model calls allowed per chunk before it is reported as failed "
        f"(default: {DEFAULT_TRANSLATION_ATTEMPTS})",
    )
    parser.add_argument(
        "--translate-method",
        choices=TRANSLATION_METHODS,
        default="auto",
        help="stage 8: 'auto' asks for native structured output and falls back to JSON "
        "mode when the provider rejects the schema (default)",
    )
    parser.add_argument(
        "--translate-max-tokens",
        type=int,
        default=DEFAULT_TRANSLATION_MAX_TOKENS,
        help="stage 8: response cap per model call "
        f"(default: {DEFAULT_TRANSLATION_MAX_TOKENS})",
    )
    parser.add_argument(
        "--translate-context-chars",
        type=int,
        default=DEFAULT_TRANSLATION_CONTEXT_CHARS,
        help="stage 8: send the last N characters of the previous translation as "
        f"continuity context (default: {DEFAULT_TRANSLATION_CONTEXT_CHARS} = off)",
    )
    parser.add_argument(
        "--translate-provider",
        choices=sorted(TRANSLATION_CHOICES),
        default=None,
        help="stage 8 only: 'deepseek' (the cloud default) or 'lmstudio' (the local "
        "OpenAI-compatible server at http://127.0.0.1:1234/v1, whose loaded model "
        "is auto-detected). Front/back matter and every other stage keep the "
        "shared provider",
    )
    parser.add_argument(
        "--translate-model",
        default=None,
        help="stage 8 only: the model id; for LM Studio the loaded model is auto-detected by default",
    )
    parser.add_argument(
        "--translate-base-url",
        default=None,
        help="stage 8 only: OpenAI-compatible base URL (default: the provider's; "
        "LM Studio uses http://127.0.0.1:1234/v1)",
    )
    parser.add_argument(
        "--translate-api-key",
        default=None,
        help="stage 8 only: API key; a local LM Studio endpoint needs none",
    )
    parser.add_argument(
        "--translate-temperature",
        type=float,
        default=None,
        help="stage 8 only: sampling temperature (default: --temperature, 0.0)",
    )
    parser.add_argument(
        "--include-matter",
        action="store_true",
        help="stage 8: translate the front/back matter chunks too, with the body "
        "prompt (the old behaviour; overwrites stage 7's localized chunks)",
    )

    parser.add_argument(
        "--force-convert",
        action="store_true",
        help="stage 1: convert again even when the Markdown in 01-markdown/ still "
        "matches the source hash",
    )
    parser.add_argument(
        "--no-images",
        dest="images",
        action="store_false",
        help="stage 1: do not extract pictures from the EPUB/PDF (images are "
        "extracted into 01-markdown/images/ by default)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate every plan (model-backed stages still run their real work and "
        "may spend tokens); write no artifact except this log (stage 1 converts into "
        "a temporary folder so the later plans still have text to read)",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="stage 3: chapters at once (default 1)"
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=MAX_FEEDBACK_LOOPS,
        help="max agent turns per unit of work (stage 4: extraction attempts per chapter)",
    )
    parser.add_argument(
        "--front-matter-lines", type=int, default=DEFAULT_FRONT_MATTER_LINES, help="stage 2 budget"
    )
    parser.add_argument(
        "--min-chapters", type=int, default=DEFAULT_MIN_CHAPTERS, help="stage 2 floor"
    )
    parser.add_argument(
        "--no-front-matter", action="store_true", help="stage 2: do not write 00-front-matter.md"
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="stages 3/4/6: do not write their manifest files",
    )
    parser.add_argument(
        "--language",
        default="auto",
        help="stages 4/6: 'auto' detects the language by sampling 02-chapters/chunks; otherwise "
        "an ISO code (sr, hr, sl, bg via CLASSLA or en via stanza)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help=f"stages 4/6: physical GPU for the tagger (default: GPU {DEFAULT_GPU_INDEX}, or "
        "$CUDA_VISIBLE_DEVICES when that is already set)",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="stages 4/6: run the tagger on the CPU"
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=DEFAULT_MIN_SENTENCE_WORDS,
        help=f"stage 4: drop sentences shorter than this (default: {DEFAULT_MIN_SENTENCE_WORDS})",
    )
    parser.add_argument(
        "--no-uppercase-filter",
        action="store_true",
        help="stage 4: keep sentences whose only capital letter is the first word",
    )
    parser.add_argument(
        "--keep-types",
        default=",".join(DEFAULT_KEEP_ENTITY_TYPES),
        help=f"stages 4/6: entity types that count "
        f"(default: {','.join(DEFAULT_KEEP_ENTITY_TYPES)})",
    )
    parser.add_argument(
        "--drop-temporal-only",
        action="store_true",
        help="stages 4/6: drop a sentence/term when every entity in it is a date or a time",
    )
    parser.add_argument(
        "--skip-if-present",
        "--resume",
        dest="skip_if_present",
        action="store_true",
        help="reuse work already on disk: existing chapters for stage 2, existing "
        "reduced chapters for stage 4 (no model is loaded for the latter), "
        "existing chunk glossaries for stage 6, existing matter localizations for "
        "stage 7, existing translations for stage 8, "
        "and existing translated chapters for stage 9",
    )
    parser.add_argument(
        "--no-save-reduced",
        action="store_true",
        help="stage 4: run the filters but write no reduced chapters",
    )
    parser.add_argument(
        "--no-save-unique",
        action="store_true",
        help="stage 4: write no first-mention chapters, only the reduced ones",
    )
    parser.add_argument(
        "--unique-only",
        action="store_true",
        help="stage 4: write only the first-mention chapters (implies --no-save-reduced)",
    )
    parser.add_argument(
        "--no-dictionary",
        action="store_true",
        help="drop stage 4 from this run (it is on by default)",
    )
    parser.add_argument(
        "--no-chunk",
        action="store_true",
        help="drop stage 5 (chunking) from this run (it is on by default)",
    )
    parser.add_argument(
        "--with-glossary",
        action="store_true",
        help="build the glossary (the default; kept so an older command line still works)",
    )
    parser.add_argument(
        "--no-glossary",
        action="store_true",
        help="stage 4: reduce only, do not build the glossary (no model call, no key needed)",
    )
    parser.add_argument(
        "--glossary-out",
        default=None,
        help="stage 4: where glossary.json / glossary.md go (default: <book>/04-glossary)",
    )
    parser.add_argument(
        "--source-language",
        default=DEFAULT_SOURCE_LANGUAGE,
        help=f"stage 4: the language the book is written in (default: {DEFAULT_SOURCE_LANGUAGE})",
    )
    parser.add_argument(
        "--target-language",
        default=DEFAULT_TARGET_LANGUAGE,
        help=f"stage 4: the language being translated into (default: {DEFAULT_TARGET_LANGUAGE})",
    )
    parser.add_argument(
        "--sentences",
        choices=("unique", "reduced"),
        default="unique",
        help="stage 4: mine the first mentions (default) or every entity-bearing sentence",
    )
    parser.add_argument(
        "--sentences-in",
        default=None,
        help="stage 4: read the sentences from this folder instead of deriving it",
    )
    parser.add_argument(
        "--max-context-entries",
        type=int,
        default=0,
        help="stage 4: cap the dictionary re-sent with each chapter (0 = all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stages 3/4/5/6/7/8: only the first N units (chapters, or chunks for "
        "stages 6/8) — a quick look, and how to price the glossary or a "
        "chunk/translation run before the whole book; --first-chapters is the "
        "chapter-aligned alternative",
    )
    parser.add_argument(
        "--toc-scan-lines",
        type=int,
        default=TOC_SCAN_LINES,
        help="recon: front matter lines to scan",
    )
    parser.add_argument(
        "--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES, help="recon: candidate cap"
    )
    parser.add_argument(
        "--examples", type=int, default=3, help="recon/debug: damage examples per shape"
    )

    parser.add_argument("--provider", choices=sorted(PROVIDERS), default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="shared sampling temperature for every stage (default: the config file, "
        "else 0.0); use --llm <stage>.temperature to set one stage only",
    )
    parser.add_argument(
        "--llm-config",
        default=None,
        metavar="FILE",
        help=f"per-stage model config (default: ./{LLM_CONFIG_FILENAME}, or "
        f"$BOOK_TRANSLATOR_CONFIG); one [llm] table plus an optional [llm.<stage>] "
        "table per stage",
    )
    parser.add_argument(
        "--llm",
        action="append",
        default=[],
        metavar="STAGE.FIELD=VALUE",
        help="override one model field for one stage, repeatable; stages: "
        f"{', '.join(STAGES)}; fields: provider, model, base_url, temperature, "
        "timeout, max_retries, api_key_env (e.g. --llm translate.provider=lmstudio "
        "--llm translate.model=my-model)",
    )
    parser.add_argument("--recursion-limit", type=int, default=DEFAULT_RECURSION_LIMIT)

    parser.add_argument(
        "--start-at",
        choices=PHASES,
        default=PHASES[0],
        help="begin here, skipping earlier phases (default: recon)",
    )
    parser.add_argument(
        "--stop-after",
        choices=PHASES,
        default=PHASES[-1],
        help=f"stop after this phase (default: {PHASES[-1]})",
    )
    parser.add_argument(
        "--with-preprocess",
        action="store_true",
        help="accepted for older command lines; stage 3 is on by default",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="drop stage 3 (chapter cleaning) from this run (it is on by default)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="also print headers, files and damage examples"
    )
    parser.add_argument(
        "--log-file", default=None, help="where to write the log (default: <book>/pipeline.log)"
    )
    parser.add_argument("--no-log", action="store_true", help="do not write a log file")
    parser.add_argument(
        "--json", action="store_true", help="print one machine-readable run summary"
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="suppress human narration (errors still go to stderr)",
    )
    parser.add_argument(
        "--interactive",
        "--checkpoint",
        action="store_true",
        dest="interactive",
        help="pause for human confirmation and review at key checkpoints (glossary, front matter, postprocess)",
    )
    return parser




def main(
    argv: list[str] | None = None,
    *,
    prog: str = "book_translator pipeline",
) -> int:
    """Parse, validate and run the whole pipeline through the shared driver."""
    args = build_parser(prog).parse_args(argv)
    options = PipelineOptions.from_namespace(args)
    errors = options.validate()
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 2
    try:
        driver = PipelineDriver(options, ns=args)
    except (FileNotFoundError, ToolError, ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        return driver.run()
    finally:
        driver.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
