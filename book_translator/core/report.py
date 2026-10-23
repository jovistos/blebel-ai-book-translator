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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


# --------------------------------------------------------------------------- #
# numbers
# --------------------------------------------------------------------------- #
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
    "StageSummary",
    "human_bytes",
    "human_count",
    "human_seconds",
    "summarize_payload",
]
