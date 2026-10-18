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


class MissingDependencyError(ToolError):
    """Raised when an optional stage dependency is not installed.

    Stage 3 (dictionary) needs a tagger — ``classla``, ``stanza`` or
    ``lingua-language-detector`` — that the rest of the pipeline never touches.
    Stage 5 needs ``chonkie``/``tiktoken`` and stage 6 needs ``rapidfuzz`` plus the
    same tagger. Those imports are lazy, so this error carries the exact install
    command instead of an ``ImportError`` raised deep inside a third-party package.
    """


class ExtractionError(ToolError):
    """Raised when a chapter's glossary extraction never produced a valid answer.

    Stage 4 asks a model for a JSON object and validates it with Pydantic. The
    call is retried, and this is what is raised once every attempt has failed —
    a transport error, a provider refusal, or an answer that stayed unparseable.
    The message carries the last failure so a run can report which chapter is
    missing from the dictionary instead of a raw traceback.
    """
