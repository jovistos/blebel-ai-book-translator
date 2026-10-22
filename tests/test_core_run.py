"""Tests for the shared run shell: clock, progress, finish and item isolation."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from book_splitter.core.run import RunContext, run_items


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


@dataclass
class Result:
    dry_run: bool = True
    limited: bool = True
    timing: dict[str, float] = field(default_factory=dict)


def test_run_context_measures_on_the_injected_clock_and_finishes_a_result():
    clock = FakeClock()
    context = RunContext(clock=clock, dry_run=False, limited=True, settings={"x": 1})
    clock.value = 2.5

    result = context.finish(Result())
    assert result.dry_run is False
    assert result.limited is True
    assert result.timing["run_seconds"] == 2.5


def test_a_broken_progress_hook_never_breaks_the_run():
    seen: list[tuple[object, int, int]] = []

    def broken(_record, _completed, _total):
        raise RuntimeError("UI went away")

    context = RunContext(progress=broken)
    context.emit("record", 1, 2)  # must not raise
    assert context.completed == 1

    context = RunContext(progress=lambda record, done, total: seen.append((record, done, total)))
    context.emit("record", 1, 2)
    assert seen == [("record", 1, 2)]


def test_run_items_continues_after_a_failed_item_and_keeps_the_order():
    items = [1, 2, 3, 4]

    def worker(item: int, _position: int) -> int:
        if item == 2:
            raise ValueError("bad item")
        return item * 10

    def on_error(item: int, exc: Exception) -> int:
        assert isinstance(exc, ValueError)
        return -item

    seen: list[tuple[int, int, int]] = []
    outcomes = run_items(
        items,
        worker,
        on_error=on_error,
        on_item=lambda outcome, done, total: seen.append((outcome.item, done, total)),
    )

    assert [outcome.value for outcome in outcomes] == [10, -2, 30, 40]
    assert [outcome.ok for outcome in outcomes] == [True, False, True, True]
    assert outcomes[1].error is not None
    assert seen == [(1, 1, 4), (2, 2, 4), (3, 3, 4), (4, 4, 4)]


def test_run_items_passes_a_one_based_position():
    positions: list[tuple[int, int]] = []

    def worker(item: int, position: int) -> int:
        positions.append((item, position))
        return item

    run_items(["a", "b", "c"], worker)
    assert positions == [("a", 1), ("b", 2), ("c", 3)]


def test_a_broken_on_item_fails_the_run_instead_of_dropping_an_item():
    """on_item is bookkeeping, not reporting: a bug there must not be silent."""

    def worker(item: int, _position: int) -> int:
        return item

    def broken(_outcome, _completed, _total) -> None:
        raise RuntimeError("collector bug")

    with pytest.raises(RuntimeError, match="collector bug"):
        run_items([1], worker, on_item=broken)


def test_run_items_honours_completed_before_and_a_broken_progress_hook():
    def worker(item: int, _position: int) -> int:
        return item

    def broken(_record, _completed, _total):
        raise RuntimeError("ignored")

    outcomes = run_items([1, 2], worker, progress=broken, completed_before=10)
    assert [outcome.value for outcome in outcomes] == [1, 2]


def test_run_items_without_an_error_handler_records_none():
    def worker(item: int, _position: int) -> int:
        raise RuntimeError(f"no {item}")

    outcomes = run_items([1], worker)
    assert outcomes[0].ok is False
    assert outcomes[0].value is None
