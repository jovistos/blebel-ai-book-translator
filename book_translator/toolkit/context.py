"""The explicit context a stage's tools resolve their calls against.

A tool call from the model carries only the arguments the model chose. The
*folder*, *book*, *section* and *chunk* the graph is working on are not the
model's to invent: they belong to the run. Before this module they arrived
through a :class:`contextvars.ContextVar` set by the runner, which made them
ambient, invisible in a signature and easy to leak between concurrent runs.

They are now a :class:`ToolContext` value passed to the tool executor and
injected into the tools that declare a ``ctx`` parameter. A tool signature is
the whole story again, and a test can build a context without a runner.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ToolContext:
    """What a stage's tools know about the run beyond the model's arguments."""

    #: The folder holding the book's chapter files (all stages that take paths).
    chapters_path: str | None = None
    #: The one chapter a chapter-level graph is editing, when there is one.
    chapter_path: str | None = None
    #: The folder holding the matter chunks (matter stage only).
    chunks_path: str | None = None
    #: The matter stage's book inventory.
    book: Any = None
    #: The matter section the current chunk belongs to.
    section: Any = None
    #: The matter chunk being localized.
    chunk: Any = None
    #: The binding book plan, for tools that must obey it.
    plan: Mapping[str, Any] | None = None
    #: Layout flags the localization validators depend on.
    is_first_front_chunk: bool = False
    is_first_section_chunk: bool = False

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> ToolContext:
        """Build the context a graph's state already carries."""
        plan = state.get("plan")
        return cls(
            chapters_path=_as_text(state.get("chapters_path")),
            chapter_path=_as_text(state.get("chapter_path")),
            chunks_path=_as_text(state.get("chunks_path")),
            book=state.get("book"),
            section=state.get("section"),
            chunk=state.get("chunk"),
            plan=plan if isinstance(plan, Mapping) else None,
            is_first_front_chunk=bool(state.get("is_first_front_chunk")),
            is_first_section_chunk=bool(state.get("is_first_section_chunk")),
        )

    @property
    def chapters_dir(self) -> Path | None:
        """The chapter folder as a path, or ``None`` when the state had none."""
        return _path_or_none(self.chapters_path)


def wants_context(tool: Any) -> bool:
    """True when ``tool`` declares the injected ``ctx`` argument.

    Injected arguments are absent from the tool's public schema (the model never
    sees them), so the executor asks the wrapped function instead.
    """
    func = getattr(tool, "func", None)
    if func is None:
        return False
    try:
        return "ctx" in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _path_or_none(value: str | None) -> Path | None:
    return Path(value).expanduser() if value else None


__all__ = ["ToolContext", "wants_context"]
