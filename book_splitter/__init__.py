"""LangGraph agent that splits a long Markdown book into chapter files.

Design contract (see ``README-book-splitter.md``):

* The LLM never reads the body of the book. It only ever sees line numbers and
  short candidate/header strings produced by deterministic tools.
* Every byte of the output files is copied verbatim from the source file by the
  slicer; the LLM only decides *where* to cut.
* Documents live in dedicated folders: sources in ``data/input/``, every
  generated artifact under ``data/output/<book>/`` (see :mod:`book_splitter.paths`).

Package layout::

    book_splitter/
        paths.py        where documents live (input/output folders)
        cli.py          `inspect` / `split` / `tools` commands
        core/           constants, errors, LLM settings
        documents/      reading, markdown inspection, text normalisation
        analysis/       TOC detection, header assessment, title location
        slicing/        planning, verbatim slicing, verification
        agent/          prompts, model, tools, graph nodes, routing, graph
"""

from .agent import SplitterResult, SplitterState, build_graph, run_split

__all__ = ["SplitterResult", "SplitterState", "build_graph", "run_split"]
__version__ = "2.0.0"
