"""The one orchestrator for the whole book pipeline.

Public surface:

* :func:`book_splitter.pipeline.parser.main` — the ``book_splitter pipeline``
  command and the implementation behind ``scripts/run_pipeline.py``;
* :class:`~book_splitter.pipeline.driver.PipelineDriver` — phase order,
  start/stop, records and the final report;
* :class:`~book_splitter.pipeline.options.PipelineOptions` and
  :class:`~book_splitter.pipeline.options.StageOptions` — normalized options and
  the exact runner calls;
* :class:`~book_splitter.pipeline.paths.StagePaths` — the canonical folder
  layout;
* :mod:`book_splitter.pipeline.sources` — the input-folder fallbacks.

The CLI package imports :func:`main` lazily so importing ``book_splitter.cli``
does not pull the whole pipeline parser in until the ``pipeline`` subcommand is
used.
"""

from __future__ import annotations

from .driver import PipelineDriver
from .options import (
    PHASES,
    PhaseSelection,
    PipelineOptions,
    StageCall,
    StageOptions,
    StageRuntime,
)
from .paths import StagePaths
from .sources import StageSources

__all__ = [
    "PHASES",
    "PhaseSelection",
    "PipelineDriver",
    "PipelineOptions",
    "StageCall",
    "StageOptions",
    "StagePaths",
    "StageRuntime",
    "StageSources",
]
