"""The run shell every stage shares: a clock, a progress hook and isolation.

Each stage runner grew the same preamble — start a monotonic clock, carry
``dry_run``/``limited``/``settings`` through to the result, guard the progress
hook so a broken UI cannot kill a run, and catch one item's exception so the
book continues. :class:`RunContext` holds those facts once;
:func:`run_items` is the one per-item loop with error isolation.

The stage still owns *what* a result is and how it is written. This module only
promises that the run is timed from one place, that every finished item is
reported in order, and that a failed item becomes a record instead of an
aborted book.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")

#: Monotonic clock used for timings; injectable so tests can be exact.
Clock = Callable[[], float]
#: Called once per finished item as ``(record, completed, total)``.
ProgressHook = Callable[[Any, int, int], None]


@dataclass
class RunContext:
    """The facts a stage run carries from its first line to its result.

    ``settings`` is deliberately untyped: some stages hold an
    :class:`~book_translator.core.settings.LLMSettings`, some a plain dict of
    stage options. The context only carries it; the stage interprets it.
    """

    clock: Clock = time.perf_counter
    dry_run: bool = False
    settings: Any = None
    limited: bool = False
    progress: ProgressHook | None = None
    started: float = field(init=False)
    completed: int = 0

    def __post_init__(self) -> None:
        self.started = self.clock()

    def elapsed(self) -> float:
        """Seconds since the run started, on the injected clock."""
        return self.clock() - self.started

    def emit(self, record: Any, completed: int, total: int) -> None:
        """Call the progress hook, swallowing its failures.

        Reporting is never allowed to end a paid run: a broken callback loses a
        progress line, not the book.
        """
        self.completed = max(self.completed, int(completed))
        if self.progress is None:
            return
        with contextlib.suppress(Exception):
            self.progress(record, completed, total)

    def finish(self, result: R) -> R:
        """Stamp the run's common fields onto ``result`` and return it.

        Only attributes the result actually has are touched, so a stage keeps
        its own shape: a ``dry_run`` flag, a ``limited`` flag and a
        ``clock``-measured ``run_seconds`` in ``timing`` when it times itself.
        """
        if hasattr(result, "dry_run"):
            result.dry_run = self.dry_run
        if hasattr(result, "limited"):
            result.limited = self.limited
        timing = getattr(result, "timing", None)
        if isinstance(timing, dict):
            timing.setdefault("run_seconds", round(self.elapsed(), 4))
        return result


@dataclass
class ItemOutcome(Generic[T, R]):
    """One item's result, or the exception that replaced it."""

    item: T
    value: R | None = None
    error: Exception | None = None
    ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "error": f"{type(self.error).__name__}: {self.error}" if self.error else None,
        }


def run_items(
    items: Iterable[T],
    worker: Callable[[T, int], R],
    *,
    on_error: Callable[[T, Exception], R] | None = None,
    on_item: Callable[[ItemOutcome[T, R], int, int], None] | None = None,
    progress: ProgressHook | None = None,
    completed_before: int = 0,
) -> list[ItemOutcome[T, R]]:
    """Run ``worker(item, position)`` over ``items`` in order, isolated item by item.

    ``position`` is 1-based, so a stage can use it for prompt numbering and for
    progress without keeping a parallel counter. An exception from ``worker``
    does not stop the loop: ``on_error(item, exc)`` builds the failed record (a
    stage's "this chapter failed but the book continues" branch), and the loop
    moves on.

    ``on_item`` sees every outcome after it is produced — that is where a runner
    appends its record, updates book-wide totals and takes a manifest snapshot.
    It is *not* suppressed: those are correctness steps, and a bug there should
    fail the run instead of silently dropping an item. ``progress`` is the
    external reporting hook and *is* suppressed, because a broken UI must not
    end a paid run.
    """
    ordered = list(items)
    total = len(ordered)
    outcomes: list[ItemOutcome[T, R]] = []
    for position, item in enumerate(ordered, start=1):
        try:
            value = worker(item, position)
        except Exception as exc:  # one bad item must not stop the run
            failed = on_error(item, exc) if on_error is not None else None
            outcome: ItemOutcome[T, R] = ItemOutcome(
                item=item, value=failed, error=exc, ok=False
            )
        else:
            outcome = ItemOutcome(item=item, value=value)
        outcomes.append(outcome)
        completed = completed_before + position
        if on_item is not None:
            on_item(outcome, completed, total)
        if progress is not None:
            with contextlib.suppress(Exception):
                progress(outcome.value, completed, total)
    return outcomes


__all__ = [
    "Clock",
    "ItemOutcome",
    "ProgressHook",
    "RunContext",
    "run_items",
]
