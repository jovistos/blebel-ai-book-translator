"""The tools each pipeline stage may call, grouped by the stage that owns them.

``preprocess``
    Stage 2's chapter-cleaning tools.
``postprocess``
    Stage 9's translated-heading and TOC tools.
``matter``
    The front/back matter localization tools.

Stage 1's tools stay in :mod:`book_splitter.agent.tools` next to the splitter
they serve, and the tools shared by more than one stage (headers, line windows,
web search, the opening audit) stay there too because that is where they are
implemented.

Each stage package imports its ``tools`` module for the registration side effect
and exposes that stage's toolset and dispatch table. :mod:`toolkit.context` holds
the :class:`ToolContext` every path-resolving tool now takes explicitly instead
of reading a process-global scope.
"""

__all__ = ["import_all"]


def import_all() -> None:
    """Import every stage package, registering its tools in the shared registry.

    Code that reports on the whole toolset (the ``tools`` CLI command) calls this
    first; a stage's own graph imports only its package.
    """
    from . import matter, postprocess, preprocess  # noqa: F401
