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
``matter``
    The front/back matter localization agent that runs before translation.
``postprocess``
    Stage 9 only — repairing the *translated* chapters' headings and TOC.

The tag is part of registration, so the toolset a stage binds to its model is a
one-line filter (:func:`tools_for_stage`), and `book_splitter tools` can show the
grouping without a hand-maintained list. A genuinely useful tool can belong to
more than one stage (``web_search`` is used by the splitter *and* the matter
agent); :func:`register_tool` records every stage, while :func:`stage_of` keeps
returning the first one it was registered under for backwards compatibility.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, cast

#: Stage names, also used as the grouping keys in `book_splitter tools`.
SHARED = "shared"
SPLIT = "split"
PREPROCESS = "preprocess"
MATTER = "matter"
POSTPROCESS = "postprocess"
STAGES: tuple[str, ...] = (SHARED, SPLIT, PREPROCESS, MATTER, POSTPROCESS)

_REGISTRY: dict[str, Any] = {}
_STAGE: dict[str, str] = {}
#: Additional stage tags for a tool that is useful to more than one agent.
_EXTRA_STAGES: dict[str, set[str]] = {}

#: Write tools belonging to each stage, used by that stage's write policy.
#: Stage 1 has no model-reachable write tool: the deterministic ``slice`` tail
#: writes only after validation, so the executor has nothing to gate here.
SPLIT_WRITE_TOOLS: frozenset[str] = frozenset()
PREPROCESS_WRITE_TOOLS: frozenset[str] = frozenset({"apply_chapter_cleaning"})
POSTPROCESS_WRITE_TOOLS: frozenset[str] = frozenset(
    {
        "apply_chapter_postprocess",
        "apply_book_toc",
        "verify_postprocessed_chapters",
    }
)
#: The matter agent submits a finished translation through a report tool; the
#: graph writes the file. There is no model-reachable write tool.
MATTER_WRITE_TOOLS: frozenset[str] = frozenset()


def register_tool(tool: Any, stage: str = SHARED) -> Any:
    """Add a LangChain tool to the registry under ``stage`` (used as a decorator).

    Registering the same tool under a second stage is allowed and adds an extra
    tag; registering it twice under the same stage is a no-op.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    existing = _STAGE.get(tool.name)
    if existing is None:
        _REGISTRY[tool.name] = tool
        _STAGE[tool.name] = stage
        return tool
    if existing == stage:
        return tool
    _EXTRA_STAGES.setdefault(tool.name, set()).add(stage)
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


def matter_stage(tool: Any) -> Any:
    """Decorator: register a front/back matter localization tool."""
    return register_tool(tool, MATTER)


def postprocess_stage(tool: Any) -> Any:
    """Decorator: register a stage-9 (translated-heading postprocessing) tool."""
    return register_tool(tool, POSTPROCESS)


def stages_of(name: str) -> tuple[str, ...]:
    """Every stage tag a registered tool carries, primary stage first."""
    primary = _STAGE.get(name)
    if primary is None:
        return ()
    extras = sorted(_EXTRA_STAGES.get(name, set()))
    return (primary, *extras)


def stage_of(name: str) -> str | None:
    """The primary stage a registered tool belongs to, or ``None``."""
    return _STAGE.get(name)


def all_tools() -> list[Any]:
    """Every registered tool, in definition order."""
    return list(_REGISTRY.values())


def tools_for_stage(stage: str, include_shared: bool = True) -> list[Any]:
    """Tools a stage may bind: its own, the extra-tagged ones, and the generic ones."""
    wanted = {stage, SHARED} if include_shared else {stage}
    return [
        t
        for t in _REGISTRY.values()
        if _STAGE.get(t.name) in wanted
        or bool(_EXTRA_STAGES.get(t.name, set()) & wanted)
    ]


def tool_by_name(name: str) -> Any | None:
    """Look a tool up by the name the model used, or ``None``."""
    return _REGISTRY.get(name)


def tool_names(stage: str | None = None) -> list[str]:
    """Names of the registered tools, optionally filtered by a pipeline stage.

    Filtering by a pipeline stage includes the generic (``shared``) tools,
    because that is exactly the set the stage's model is given. A multi-stage
    tool appears for every stage it is tagged with.
    """
    if stage is None:
        return list(_REGISTRY)
    if stage == SHARED:
        return [name for name in _REGISTRY if _STAGE.get(name) == SHARED]
    wanted = {stage, SHARED}
    return [
        name
        for name in _REGISTRY
        if _STAGE.get(name) in wanted or bool(_EXTRA_STAGES.get(name, set()) & wanted)
    ]


def describe_tools(stage: str | None = None) -> list[tuple[str, str, str]]:
    """``(stage, name, first docstring line)`` for each tool, for the CLI.

    With no ``stage`` each tool appears once, under its primary stage. With a
    stage filter every tool tagged for that stage appears, including a
    multi-stage tool that was first registered elsewhere.
    """
    out: list[tuple[str, str, str]] = []
    for tool in all_tools():
        tags = stages_of(tool.name)
        primary = tags[0] if tags else SHARED
        if stage is not None and stage not in tags:
            continue
        first = (tool.description or "").strip().splitlines()
        out.append((stage or primary, tool.name, first[0] if first else ""))
    return out


def grouped_tools() -> list[tuple[str, list[tuple[str, str]]]]:
    """``(stage, [(name, summary)])`` in :data:`STAGES` order — the CLI grouping.

    A tool tagged with two stages is listed under both, because both agents can
    call it.
    """
    out: list[tuple[str, list[tuple[str, str]]]] = []
    for stage in STAGES:
        entries: list[tuple[str, str]] = []
        for tool in all_tools():
            if stage not in stages_of(tool.name):
                continue
            first = (tool.description or "").strip().splitlines()
            entries.append((tool.name, first[0] if first else ""))
        out.append((stage, entries))
    return out


def reset() -> None:
    """Drop the registry (test helper)."""
    _REGISTRY.clear()
    _STAGE.clear()
    _EXTRA_STAGES.clear()


def register(function: Callable[..., Any]) -> Callable[..., Any]:
    """Alias kept for readability when decorating a tool factory."""
    return cast(Callable[..., Any], register_tool(function))


def bind_all(tools: Iterable[Any]) -> dict[str, Any]:
    """``name -> tool`` lookup for a stage's executor table."""
    return {t.name: t for t in tools}
