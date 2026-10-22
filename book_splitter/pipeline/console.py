"""The pipeline's console: stdout, the run log and the live progress bar."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any

RULE = "─" * 72


class Console:
    """Narrates the run to stdout **and** to the run's log file.

    The log receives every line, debug included; the console only shows debug
    lines with ``--debug``. Writes are locked, because stage 2 reports progress
    from worker threads when ``--workers`` is greater than one.
    """

    def __init__(
        self, log_path: Path | None, debug: bool = False, quiet: bool = False
    ) -> None:
        self.debug_enabled = debug
        self.quiet = quiet
        self.log_path = log_path
        self._lock = threading.Lock()
        self._log = None
        self.bar_enabled = sys.stdout.isatty()
        self._bar_drawn = False
        self._bar_width = 0
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = log_path.open("w", encoding="utf-8")

    def emit(self, text: str = "", level: str = "info", force: bool = False) -> None:
        with self._lock:
            self._clear_bar_locked()
            if self._log is not None:
                self._log.write(text + "\n")
                self._log.flush()
            if self.quiet and level != "error" and not force:
                return
            if level == "debug" and not self.debug_enabled:
                return
            stream = sys.stderr if level == "error" else sys.stdout
            print(text, file=stream, flush=True)

    # -- the live bar ----------------------------------------------------- #
    def bar(self, text: str) -> None:
        """Redraw the one-line progress bar in place (terminal only).

        The bar is deliberately *not* written to the log: it is redrawn several
        times a second, and the log is a transcript of the run, not a recording
        of a cursor. Anything that must survive the run goes through ``info``.
        """
        if not self.bar_enabled or self.quiet:
            return
        with self._lock:
            padded = text + " " * max(0, self._bar_width - len(text))
            print("\r" + padded, end="", file=sys.stdout, flush=True)
            self._bar_width = max(self._bar_width, len(text))
            self._bar_drawn = True

    def stream_line(self, text: str) -> None:
        """A transient progress line for a non-terminal stdout (piped / logged).

        Same job as ``bar`` where a cursor cannot be moved: print the state, but
        keep it out of the log — the log keeps the durable per-chapter lines.
        """
        if self.quiet:
            return
        with self._lock:
            self._clear_bar_locked()
            print(text, file=sys.stdout, flush=True)

    def clear_bar(self) -> None:
        with self._lock:
            self._clear_bar_locked()

    def _clear_bar_locked(self) -> None:
        if self._bar_drawn:
            print("\r" + " " * self._bar_width + "\r", end="", file=sys.stdout, flush=True)
            self._bar_drawn = False

    def info(self, text: str = "") -> None:
        self.emit(text)

    def debug(self, text: str = "") -> None:
        self.emit(text, "debug")

    def error(self, text: str = "") -> None:
        self.emit(text, "error")

    def json(self, payload: Any) -> None:
        """Write the one machine-readable object, even in quiet mode."""
        self.emit(json.dumps(payload, ensure_ascii=False, indent=2), force=True)

    def log_only(self, text: str = "") -> None:
        """Write to the log without touching the console (used for provenance)."""
        with self._lock:
            if self._log is not None:
                self._log.write(text + "\n")
                self._log.flush()

    def kv(self, key: str, value: Any, indent: int = 2, level: str = "info") -> None:
        self.emit(f"{' ' * indent}{key:<14}: {value}", level)

    def phase(self, index: int, total: int, name: str, title: str) -> None:
        self.emit("")
        self.emit(RULE)
        self.emit(f" [{index}/{total}] {name.upper()}")
        self.emit(f"       {title}")
        self.emit(RULE)

    def close(self) -> None:
        with self._lock:
            if self._log is not None:
                self._log.close()
                self._log = None


__all__ = ["RULE", "Console"]
