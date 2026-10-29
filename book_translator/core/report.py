"""The small reporting surface both front ends share.

The pipeline and the command line each render their own stage narration, but two
values are genuinely common and must not drift:

* **formatting** — :func:`human_seconds`, :func:`human_bytes` and
  :func:`human_count`, used by the pipeline driver and the command summaries;
* **a payload summary** — :func:`summarize_payload` turns the deterministic
  ``recon``/``scan`` reports into a :class:`StageSummary` the CLI renders.

Typed stage results are summarized by the front end that owns them (the driver's
phase bodies and the CLI's ``_print_*_summary`` helpers). This module deliberately
imports no stage package: importing it must stay cheap, and a summary of a plain
payload does not need the agent runtime.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


# --------------------------------------------------------------------------- #
# numbers & progress bars
# --------------------------------------------------------------------------- #
def render_bar(done: int, total: int, width: int = 22) -> str:
    """Render a fixed-width ``█░`` progress bar."""
    filled = min(width, int(width * done / total)) if total > 0 else 0
    return "█" * filled + "░" * (width - filled)


def estimate_eta(
    done: int,
    total: int,
    elapsed: float,
    durations: Sequence[float] = (),
) -> float | None:
    """Estimate remaining seconds from per-unit durations or overall average."""
    average = (
        sum(durations) / len(durations)
        if durations
        else (elapsed / done if done > 0 else None)
    )
    return average * max(total - done, 0) if average is not None else None


class BackgroundTicker:
    """Runs a 1-second daemon progress loop with a periodic non-TTY heartbeat."""

    def __init__(
        self,
        name: str,
        *,
        on_tick: Callable[[], None],
        on_heartbeat: Callable[[], None] | None = None,
        is_tty: Callable[[], bool] | bool = True,
        heartbeat_seconds: float = 10.0,
    ) -> None:
        self.name = name
        self._on_tick = on_tick
        self._on_heartbeat = on_heartbeat
        self._is_tty = is_tty
        self.heartbeat_seconds = heartbeat_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._loop, name=self.name, daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None

    def _tty(self) -> bool:
        return self._is_tty() if callable(self._is_tty) else bool(self._is_tty)

    def _loop(self) -> None:
        last_heartbeat = 0.0
        while not self._stop.wait(1.0):
            if self._tty():
                self._on_tick()
                continue
            if self._on_heartbeat is not None:
                now = time.monotonic()
                if now - last_heartbeat >= self.heartbeat_seconds:
                    last_heartbeat = now
                    self._on_heartbeat()
def human_seconds(seconds: float) -> str:
    """``0.0213`` -> ``21 ms``; ``389.42`` -> ``6m 29.4s``."""
    value = max(0.0, float(seconds or 0.0))
    if value < 1:
        return f"{value * 1000:.0f} ms"
    if value < 60:
        return f"{value:.2f} s"
    minutes, rest = divmod(value, 60)
    if minutes < 60:
        return f"{int(minutes)}m {rest:04.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes:02d}m {rest:04.1f}s"


def human_bytes(size: float) -> str:
    """``411144`` -> ``401.5 KiB``."""
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024:
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def human_count(value: Any) -> str:
    """``12345`` -> ``12,345``; ``None`` -> ``0``."""
    try:
        return f"{int(value or 0):,}"
    except (TypeError, ValueError):
        return str(value)


# --------------------------------------------------------------------------- #
# a payload summary
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StageSummary:
    """The rows one deterministic phase reports, ready to render anywhere."""

    headline: str
    rows: tuple[tuple[str, str], ...] = ()


def summarize_payload(name: str, payload: Mapping[str, Any]) -> StageSummary:
    """Summarize the deterministic ``recon``/``scan`` payloads.

    Those phases do not return a typed result; their payload is the report. The
    rows here are the same numbers the CLI prints for them.
    """
    payload = dict(payload or {})
    if name == "recon":
        assessment = payload.get("header_assessment") or {}
        toc = payload.get("printed_toc_block") or {}
        rows = [
            ("source", str(payload.get("source") or "—")),
            ("lines", human_count(payload.get("line_count"))),
            ("headers", human_count(payload.get("header_count"))),
            ("verdict", str(assessment.get("verdict") or "—")),
            ("candidates", human_count(payload.get("candidate_count"))),
            ("printed TOC", "found" if toc.get("found") else "not found"),
        ]
        return StageSummary(
            headline=(
                f"{human_count(payload.get('line_count'))} lines · "
                f"{human_count(payload.get('header_count'))} headers "
                f"({assessment.get('verdict') or '—'}) · "
                f"{human_count(payload.get('candidate_count'))} candidates"
            ),
            rows=tuple(rows),
        )
    if name == "scan":
        counts = payload.get("counts") or {}
        rows = [
            ("chapters", human_count(payload.get("chapter_count"))),
            ("findings", human_count(payload.get("total_findings"))),
            ("shapes", str(len(counts))),
        ]
        rows.extend((str(key), human_count(value)) for key, value in sorted(counts.items()))
        return StageSummary(
            headline=(
                f"{human_count(payload.get('total_findings'))} findings in "
                f"{len(counts)} shape(s)"
            ),
            rows=tuple(rows),
        )
    generic_rows = tuple(
        (str(key), str(value))
        for key, value in payload.items()
        if isinstance(value, (str, int, float))
    )
    return StageSummary(headline=name, rows=generic_rows)


__all__ = [
    "BackgroundTicker",
    "StageSummary",
    "estimate_eta",
    "human_bytes",
    "human_count",
    "human_seconds",
    "render_bar",
    "summarize_payload",
]
