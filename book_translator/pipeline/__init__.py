"""The one orchestrator for the whole book pipeline.

Public surface:

* :func:`book_translator.pipeline.parser.main` — the ``book_translator pipeline``
  command and the implementation behind ``scripts/run_pipeline.py``;
* :class:`~book_translator.pipeline.driver.PipelineDriver` — phase order,
  start/stop, records and the final report;
* :class:`~book_translator.pipeline.options.PipelineOptions` and
  :class:`~book_translator.pipeline.options.StageOptions` — normalized options and
  the exact runner calls;
* :class:`~book_translator.pipeline.paths.StagePaths` — the canonical folder
  layout;
* :mod:`book_translator.pipeline.sources` — the input-folder fallbacks.

The CLI package imports :func:`main` lazily so importing ``book_translator.cli``
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
