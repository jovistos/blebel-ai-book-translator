"""Exception types shared across the package."""

from __future__ import annotations


class ToolError(Exception):
    """Raised for recoverable, model-facing problems.

    A tool that raises this reports a clean message back to the agent instead of
    crashing the graph. It is also used by the deterministic layer (planning,
    validation, slicing) for input the caller can fix.
    """


class ConfigError(Exception):
    """Raised when the LLM provider settings cannot be resolved."""
