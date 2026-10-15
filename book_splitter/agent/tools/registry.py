"""Tool registry.

Each tool module decorates its LangChain tools with :func:`register_tool`, so
adding a tool is a single-file change: define it, decorate it, and it shows up
in the schemas sent to the model and in the executor's lookup table. The
registry is also what the executor consults to reject unknown tool names.
"""

from __future__ import annotations

from typing import Any, Callable

_REGISTRY: dict[str, Any] = {}

#: Tools that must never be reachable before a plan has been validated.
WRITE_TOOLS: frozenset[str] = frozenset({"slice_book_by_line_numbers", "verify_sliced_output"})


def register_tool(tool: Any) -> Any:
    """Add a LangChain tool to the registry (used as a decorator)."""
    _REGISTRY[tool.name] = tool
    return tool


def all_tools() -> list[Any]:
    """Every registered tool, in definition order."""
    return list(_REGISTRY.values())


def tool_by_name(name: str) -> Any | None:
    """Look a tool up by the name the model used, or ``None``."""
    return _REGISTRY.get(name)


def tool_names() -> list[str]:
    """Names of every registered tool."""
    return list(_REGISTRY)


def is_write_tool(name: str) -> bool:
    """True when a tool writes (or verifies) book content on disk."""
    return name in WRITE_TOOLS


def describe_tools() -> list[tuple[str, str]]:
    """``(name, first docstring line)`` for each tool — used by ``book_splitter tools``."""
    out: list[tuple[str, str]] = []
    for tool in all_tools():
        first = (tool.description or "").strip().splitlines()
        out.append((tool.name, first[0] if first else ""))
    return out


def reset() -> None:
    """Drop the registry (test helper)."""
    _REGISTRY.clear()


def register(function: Callable[..., Any]) -> Callable[..., Any]:
    """Alias kept for readability when decorating a tool factory."""
    return register_tool(function)
