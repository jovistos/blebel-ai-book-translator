"""Tool registry, grouped by pipeline stage.

Tools are declared once, in the module that implements them, and tagged with the
stage they belong to:

``shared``
    Useful to more than one stage (line shapes, corpus discovery, the manifest
    reader). These are the generic tools: if two stages need the same evidence,
    the tool lives here instead of being duplicated.
``split``
    Stage 1 only — turning one long Markdown book into one file per chapter.
``preprocess``
    Stage 2 only — repairing each chapter's headers and conversion damage.

The tag is part of registration, so the toolset a stage binds to its model is a
one-line filter (:func:`tools_for_stage`), and `book_splitter tools` can show the
grouping without a hand-maintained list.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

#: Stage names, also used as the grouping keys in `book_splitter tools`.
SHARED = "shared"
SPLIT = "split"
PREPROCESS = "preprocess"
STAGES: tuple[str, ...] = (SHARED, SPLIT, PREPROCESS)

_REGISTRY: dict[str, Any] = {}
_STAGE: dict[str, str] = {}

#: Tools that must never be reachable before a plan has been validated.
WRITE_TOOLS: frozenset[str] = frozenset(
    {
        "slice_book_by_line_numbers",
        "verify_sliced_output",
        "apply_chapter_cleaning",
        "verify_preprocessed_chapters",
    }
)

#: Write tools belonging to each stage, used by that stage's write policy.
SPLIT_WRITE_TOOLS: frozenset[str] = frozenset({"slice_book_by_line_numbers"})
PREPROCESS_WRITE_TOOLS: frozenset[str] = frozenset({"apply_chapter_cleaning"})


def register_tool(tool: Any, stage: str = SHARED) -> Any:
    """Add a LangChain tool to the registry under ``stage`` (used as a decorator)."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    existing = _STAGE.get(tool.name)
    if existing is not None and existing != stage:
        raise ValueError(
            f"tool {tool.name!r} is already registered in stage {existing!r}; "
            "a tool belongs to exactly one stage group"
        )
    _REGISTRY[tool.name] = tool
    _STAGE[tool.name] = stage
    return tool


def shared(tool: Any) -> Any:
    """Decorator: register a tool usable by every stage."""
    return register_tool(tool, SHARED)


def split_stage(tool: Any) -> Any:
    """Decorator: register a stage-1 (chapter splitting) tool."""
    return register_tool(tool, SPLIT)


def preprocess_stage(tool: Any) -> Any:
    """Decorator: register a stage-2 (chapter preprocessing) tool."""
    return register_tool(tool, PREPROCESS)


def stage_of(name: str) -> str | None:
    """The stage a registered tool belongs to, or ``None``."""
    return _STAGE.get(name)


def all_tools() -> list[Any]:
    """Every registered tool, in definition order."""
    return list(_REGISTRY.values())


def tools_for_stage(stage: str, include_shared: bool = True) -> list[Any]:
    """Tools a stage may bind: its own, plus the generic ones by default."""
    wanted = {stage, SHARED} if include_shared else {stage}
    return [t for t in _REGISTRY.values() if _STAGE.get(t.name) in wanted]


def tool_by_name(name: str) -> Any | None:
    """Look a tool up by the name the model used, or ``None``."""
    return _REGISTRY.get(name)


def tool_names(stage: str | None = None) -> list[str]:
    """Names of the registered tools, optionally filtered by stage.

    Filtering by a pipeline stage includes the generic (``shared``) tools,
    because that is exactly the set the stage's model is given.
    """
    if stage is None:
        return list(_REGISTRY)
    if stage == SHARED:
        return [name for name, s in _STAGE.items() if s == SHARED]
    return [name for name, s in _STAGE.items() if s in {stage, SHARED}]


def is_write_tool(name: str) -> bool:
    """True when a tool writes (or verifies) book content on disk."""
    return name in WRITE_TOOLS


def describe_tools(stage: str | None = None) -> list[tuple[str, str, str]]:
    """``(stage, name, first docstring line)`` for each tool, for the CLI."""
    out: list[tuple[str, str, str]] = []
    for tool in all_tools():
        tool_stage = _STAGE.get(tool.name, SHARED)
        if stage is not None and tool_stage != stage:
            continue
        first = (tool.description or "").strip().splitlines()
        out.append((tool_stage, tool.name, first[0] if first else ""))
    return out


def grouped_tools() -> list[tuple[str, list[tuple[str, str]]]]:
    """``(stage, [(name, summary)])`` in :data:`STAGES` order — the CLI grouping."""
    described = describe_tools()
    return [
        (s, [(name, summary) for st, name, summary in described if st == s]) for s in STAGES
    ]


def reset() -> None:
    """Drop the registry (test helper)."""
    _REGISTRY.clear()
    _STAGE.clear()


def register(function: Callable[..., Any]) -> Callable[..., Any]:
    """Alias kept for readability when decorating a tool factory."""
    return register_tool(function)


def bind_all(tools: Iterable[Any]) -> dict[str, Any]:
    """``name -> tool`` lookup for a stage's executor table."""
    return {t.name: t for t in tools}
