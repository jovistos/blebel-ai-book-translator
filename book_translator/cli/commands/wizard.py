"""Interactive CLI wizard for configuring and running the book translation pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .. import parents


def add_parser(sub: Any) -> None:
    """Register the wizard command parser."""
    p_wizard = sub.add_parser(
        "wizard",
        help="interactive wizard to guide and run the book translation pipeline",
    )
    parents.add_book(p_wizard, required=False)
    p_wizard.add_argument(
        "--first-chapters",
        "--chapters",
        dest="first_chapters",
        type=int,
        default=None,
        help="process only the first N body chapters",
    )
    p_wizard.add_argument("--json", action="store_true", help="machine-readable result")
    p_wizard.add_argument("--quiet", "-q", action="store_true")
    p_wizard.set_defaults(func=cmd_wizard)


def _format_size(size_bytes: int) -> str:
    from book_translator.core.report import human_bytes

    return human_bytes(size_bytes)


def _print_language_codes(console: Any = None) -> None:
    from book_translator.stages.glossary.prompts import LANGUAGE_NAMES

    items = sorted(LANGUAGE_NAMES.items(), key=lambda x: x[0].lower())
    chunked = [items[i : i + 3] for i in range(0, len(items), 3)]

    if console:
        from rich.table import Table

        table = Table(
            title="Supported Language Codes",
            border_style="dim",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Code", style="bold cyan", justify="center")
        table.add_column("Language", style="white")
        table.add_column("Code", style="bold cyan", justify="center")
        table.add_column("Language", style="white")
        table.add_column("Code", style="bold cyan", justify="center")
        table.add_column("Language", style="white")

        for row in chunked:
            row_cells: list[str] = []
            for code, name in row:
                row_cells.extend([code, name])
            while len(row_cells) < 6:
                row_cells.extend(["", ""])
            table.add_row(*row_cells)

        console.print(table)
        console.print("[dim]Tip: You can also enter any other standard ISO 639-1 language code.[/dim]\n")
    else:
        print("\nSupported Language Codes:")
        for row in chunked:
            line = "  ".join(f"{code:>7} : {name:<22}" for code, name in row)
            print(f"  {line}")
        print("  (Tip: You can also enter any other standard ISO 639-1 language code)\n")


def _list_input_books(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    valid_exts = {".md", ".epub", ".pdf", ".txt"}
    files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
    return sorted(files, key=lambda f: f.name.lower())


def cmd_wizard(args: argparse.Namespace) -> int:
    """Run the interactive wizard."""
    from book_translator.paths import default_input_dir

    if getattr(args, "json", False):
        import json

        from book_translator.core.constants import DEFAULT_SOURCE_LANGUAGE, DEFAULT_TARGET_LANGUAGE
        from book_translator.stages.glossary.prompts import LANGUAGE_NAMES

        input_dir = default_input_dir()
        books = _list_input_books(input_dir)
        print(
            json.dumps(
                {
                    "input_dir": str(input_dir),
                    "books": [b.name for b in books],
                    "presets": [
                        "full",
                        "preview",
                        "preparation",
                        "translate_chunks",
                        "convert",
                        "custom",
                    ],
                    "languages": dict(sorted(LANGUAGE_NAMES.items(), key=lambda x: x[0].lower())),
                    "default_source_language": DEFAULT_SOURCE_LANGUAGE,
                    "default_target_language": DEFAULT_TARGET_LANGUAGE,
                },
                indent=2,
            )
        )
        return 0

    try:
        from rich.console import Console as RichConsole
        from rich.panel import Panel
        from rich.table import Table

        console = RichConsole()
    except ImportError:
        console = None

    if not sys.stdin.isatty():
        print("error: wizard requires an interactive terminal (stdin is not a tty)", file=sys.stderr)
        return 1

    # Banner
    if console:
        console.print(
            Panel(
                "[bold cyan]BleBel Book Translation Pipeline[/bold cyan]\n"
                "[dim]Interactive Step-by-Step Configuration Wizard[/dim]",
                border_style="cyan",
                expand=False,
            )
        )
    else:
        print("\n=== BleBel Book Translation Pipeline Wizard ===\n")

    # Step 1: Book Selection
    input_dir = default_input_dir()
    books = _list_input_books(input_dir)

    selected_book: str | None = getattr(args, "book", None)
    if not selected_book:
        if books:
            if console:
                tbl = Table(title=f"Available Books in {input_dir.name}/", border_style="dim")
                tbl.add_column("#", justify="center", width=4, style="bold")
                tbl.add_column("Filename", style="bold cyan")
                tbl.add_column("Format", justify="center", width=8, style="yellow")
                tbl.add_column("Size", justify="right", width=10, style="dim")

                for idx, b in enumerate(books, start=1):
                    tbl.add_row(
                        str(idx),
                        b.name,
                        b.suffix.upper().lstrip("."),
                        _format_size(b.stat().st_size),
                    )
                console.print(tbl)
            else:
                print(f"Available Books in {input_dir}:")
                for idx, b in enumerate(books, start=1):
                    print(f"  [{idx}] {b.name} ({_format_size(b.stat().st_size)})")

            while not selected_book:
                prompt_text = f"Select book number (1-{len(books)}) or enter custom path"
                choice = input(f"{prompt_text}: ").strip()
                if not choice:
                    if books:
                        selected_book = books[0].name
                        break
                    continue
                if choice.isdigit() and 1 <= int(choice) <= len(books):
                    selected_book = books[int(choice) - 1].name
                else:
                    selected_book = choice
        else:
            selected_book = input("Enter book path or filename in data/input/: ").strip()

    if console:
        console.print(f"\n[bold green]Selected Book:[/] [cyan]{selected_book}[/]\n")
    else:
        print(f"\nSelected Book: {selected_book}\n")

    # Step 2: Workflow Preset
    presets = [
        (
            "Full Pipeline",
            "Stages 0-11: Convert, Split, Preprocess, Dictionary, Glossary, Chunk, Translate, Assemble, Postprocess, EPUB",
            ["--with-translate", "--with-postprocess"],
        ),
        (
            "Preview / Test Run",
            "Translate only the first N chapters (quick verification of translation quality)",
            ["--with-translate", "--with-postprocess"],
        ),
        (
            "Preparation & Glossary Only",
            "Stages 1-6: Structure, cleanup, dictionary, locked glossary and chunking (no translation cost)",
            ["--no-translate"],
        ),
        (
            "Translate & Assemble Existing Chunks",
            "Stages 7-12: Resume after chunks & glossary are already prepared",
            ["--start-at", "matter", "--with-translate", "--with-postprocess"],
        ),
        (
            "Convert Only",
            "Stage 1: Convert EPUB/PDF to Markdown and extract illustrations",
            ["--stop-after", "convert"],
        ),
        (
            "Custom Window",
            "Manually specify --start-at and --stop-after phases",
            [],
        ),
    ]

    if console:
        ptbl = Table(title="Select Workflow Preset", border_style="dim")
        ptbl.add_column("#", justify="center", width=4, style="bold")
        ptbl.add_column("Preset", style="bold cyan", width=28)
        ptbl.add_column("Description", style="dim")
        for idx, (pname, pdesc, _) in enumerate(presets, start=1):
            ptbl.add_row(str(idx), pname, pdesc)
        console.print(ptbl)
    else:
        print("Select Workflow Preset:")
        for idx, (pname, pdesc, _) in enumerate(presets, start=1):
            print(f"  [{idx}] {pname}: {pdesc}")

    p_choice = input(f"Choose preset (1-{len(presets)}, default 1): ").strip()
    preset_idx = int(p_choice) - 1 if p_choice.isdigit() and 1 <= int(p_choice) <= len(presets) else 0
    pname, _, selected_flags = presets[preset_idx]
    cmd_flags = list(selected_flags)

    initial_first_chapters = getattr(args, "first_chapters", None)
    if pname == "Preview / Test Run":
        if initial_first_chapters is not None:
            cmd_flags.extend(["--first-chapters", str(initial_first_chapters)])
        else:
            ch_choice = input("How many chapters to translate for the preview? (default 3): ").strip()
            num_ch = int(ch_choice) if ch_choice.isdigit() and int(ch_choice) > 0 else 3
            cmd_flags.extend(["--first-chapters", str(num_ch)])

    if pname == "Custom Window":
        start_at = input("Start at phase (default: convert/split): ").strip()
        stop_after = input("Stop after phase (default: epub): ").strip()
        if start_at:
            cmd_flags.extend(["--start-at", start_at])
        if stop_after:
            cmd_flags.extend(["--stop-after", stop_after])
        with_trans = input("Include translation? [y/N]: ").strip().lower()
        if with_trans in {"y", "yes"}:
            cmd_flags.append("--with-translate")

    is_translation_run = "--with-translate" in cmd_flags or (
        "--no-translate" not in cmd_flags and "--stop-after" not in cmd_flags and preset_idx == 0
    )

    # Step 3: Translation Configuration
    if is_translation_run:
        if console:
            console.print("\n[bold cyan]Translation Engine Configuration[/bold cyan]")
            console.print("  [1] From book_translator.toml (Recommended — uses your config file)")
            console.print("  [2] DeepSeek (Cloud API)")
            console.print("  [3] LM Studio (Local / offline — auto-detects currently loaded model)")
            console.print("  [4] OpenAI (gpt-4.1-mini / gpt-4o / gpt-5)")
            console.print("  [5] Gemini (gemini-3.8-flash)")
            console.print("  [6] Claude (claude-haiku-5-5 via OpenRouter / proxy)")
        else:
            print("\nTranslation Engine Configuration:")
            print("  [1] From book_translator.toml (Recommended — uses your config file)")
            print("  [2] DeepSeek (Cloud API)")
            print("  [3] LM Studio (Local / offline — auto-detects currently loaded model)")
            print("  [4] OpenAI (gpt-4.1-mini / gpt-4o / gpt-5)")
            print("  [5] Gemini (gemini-3.8-flash)")
            print("  [6] Claude (claude-haiku-5-5 via OpenRouter / proxy)")

        t_choice = input("Select provider (1-6, default 1): ").strip()
        if t_choice == "2":
            cmd_flags.extend(["--translate-provider", "deepseek"])
        elif t_choice == "3":
            cmd_flags.extend(["--translate-provider", "lmstudio"])
            if console:
                console.print("  [dim]LM Studio will use whichever model is currently loaded.[/dim]")
            else:
                print("  LM Studio will use whichever model is currently loaded.")
        elif t_choice == "4":
            cmd_flags.extend(["--llm", "translate.provider=openai"])
        elif t_choice == "5":
            cmd_flags.extend(["--llm", "translate.provider=gemini"])
        elif t_choice == "6":
            cmd_flags.extend(["--llm", "translate.provider=claude"])
        else:
            pass

    if pname != "Convert Only":
        from book_translator.core.constants import DEFAULT_SOURCE_LANGUAGE, DEFAULT_TARGET_LANGUAGE
        from book_translator.stages.glossary.prompts import LANGUAGE_NAMES, language_name

        if console:
            console.print("\n[bold cyan]Language Configuration[/bold cyan]")
        else:
            print("\nLanguage Configuration:")

        _print_language_codes(console)

        canonical_codes = {k.lower(): k for k in LANGUAGE_NAMES}

        src_name = language_name(DEFAULT_SOURCE_LANGUAGE)
        src_prompt = (
            f"Source language code (default '{DEFAULT_SOURCE_LANGUAGE}' for {src_name}, Enter to keep default): "
        )
        src_input = input(src_prompt).strip()
        if src_input:
            src_code = canonical_codes.get(src_input.lower(), src_input)
            if src_code != DEFAULT_SOURCE_LANGUAGE:
                cmd_flags.extend(["--source-language", src_code])
        else:
            src_code = DEFAULT_SOURCE_LANGUAGE

        dest_name = language_name(DEFAULT_TARGET_LANGUAGE)
        dest_prompt = (
            f"Destination language code (default '{DEFAULT_TARGET_LANGUAGE}' for {dest_name}, Enter to keep default): "
        )
        dest_input = input(dest_prompt).strip()
        if dest_input:
            dest_code = canonical_codes.get(dest_input.lower(), dest_input)
            if dest_code != DEFAULT_TARGET_LANGUAGE:
                cmd_flags.extend(["--target-language", dest_code])
        else:
            dest_code = DEFAULT_TARGET_LANGUAGE

        if console:
            console.print(
                f"  [dim]Languages selected:[/] [bold cyan]{src_code}[/] ({language_name(src_code)}) "
                f"→ [bold cyan]{dest_code}[/] ({language_name(dest_code)})\n"
            )
        else:
            print(
                f"  Languages selected: {src_code} ({language_name(src_code)}) "
                f"→ {dest_code} ({language_name(dest_code)})\n"
            )

    # Step 4: Safety & Checkpoints
    if console:
        console.print("\n[bold cyan]Execution Options & Checkpoints[/bold cyan]")
    else:
        print("\nExecution Options & Checkpoints:")

    if pname not in {"Preview / Test Run", "Convert Only"}:
        if initial_first_chapters is not None:
            cmd_flags.extend(["--first-chapters", str(initial_first_chapters)])
        else:
            ch_input = input("Limit run to first N body chapters? (press Enter for full book): ").strip()
            if ch_input.isdigit() and int(ch_input) > 0:
                cmd_flags.extend(["--first-chapters", ch_input])

    chk_choice = input("Pause at checkpoints (Glossary, Title, Postprocess) for review? [Y/n]: ").strip().lower()
    if chk_choice not in {"n", "no"}:
        cmd_flags.append("--interactive")

    skip_choice = input("Skip already completed chapters/chunks (--skip-if-present)? [Y/n]: ").strip().lower()
    if skip_choice not in {"n", "no"}:
        cmd_flags.append("--skip-if-present")

    cpu_choice = input("Force CPU for NER tagging (--cpu)? [y/N]: ").strip().lower()
    if cpu_choice in {"y", "yes"}:
        cmd_flags.append("--cpu")

    # Step 5: Synthesize Command
    full_argv = [selected_book, *cmd_flags]
    cmd_str = f"python scripts/run_pipeline.py {full_argv[0]}"
    for flag in full_argv[1:]:
        if " " in flag:
            cmd_str += f' "{flag}"'
        else:
            cmd_str += f" {flag}"

    if console:
        console.print(
            Panel(
                f"[bold green]Ready to Execute:[/bold green]\n\n"
                f"[bold white]{cmd_str}[/bold white]\n\n"
                f"[bold][r][/bold] Run now\n"
                f"[bold][p][/bold] Print command and exit\n"
                f"[bold][q][/bold] Quit",
                border_style="green",
            )
        )
    else:
        print(f"\nReady to Execute:\n  {cmd_str}\n")
        print("[r] Run now / [p] Print command and exit / [q] Quit")

    action = input("Select [r/p/q] (default 'r'): ").strip().lower()
    if action in {"", "r", "run"}:
        if console:
            console.print("\n[bold cyan]Starting pipeline...[/bold cyan]\n")
        from book_translator.pipeline.parser import main as pipeline_main

        return pipeline_main(full_argv, prog="run_pipeline.py")
    elif action in {"p", "print"}:
        print(cmd_str)
        return 0
    else:
        print("Cancelled.")
        return 0


def main(argv: list[str] | None = None, *, prog: str = "book_translator wizard") -> int:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parents.add_book(parser, required=False)
    parser.add_argument(
        "--first-chapters",
        "--chapters",
        dest="first_chapters",
        type=int,
        default=None,
        help="process only the first N body chapters",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable result")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args(argv)
    if args.first_chapters is not None and int(args.first_chapters) < 0:
        print("error: --first-chapters cannot be negative", file=sys.stderr)
        return 2
    return cmd_wizard(args)


__all__ = ["add_parser", "cmd_wizard", "main"]
