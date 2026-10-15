"""Two LangGraph agents that turn a long Markdown book into clean chapter files.

Stage 1 — :mod:`book_splitter.agent` — splits one long Markdown book into one
file per chapter **without reading its body**. Design contract:

* The LLM never reads the body of the book. It only ever sees line numbers and
  short candidate/header strings produced by deterministic tools.
* Every byte of the output files is copied verbatim from the source file by the
  slicer; the LLM only decides *where* to cut.
* Documents live in dedicated folders: sources in ``data/input/``, every
  generated artifact under ``data/output/<book>/`` (see :mod:`book_splitter.paths`).

Stage 2 — :mod:`book_splitter.preprocess_agent` — repairs each chapter file,
**one chapter at a time**, with its own specialised agent and toolset:

* It starts from the chapter's *headers*, not from its prose: page numbers,
  figure captions marked up as headings, and headings the PDF conversion split
  across two lines are detected by shape and confirmed by the agent.
* It never edits text; it submits a plan in a fixed vocabulary
  (``keep`` / ``demote`` / ``remove`` / ``merge_with_next`` plus a few named text
  fixes) and a deterministic validator checks each entry against the real file.
* A per-line provenance map proves that only the declared lines changed, and
  that the only lines deleted were page artefacts.

Package layout::

    book_splitter/
        paths.py        where documents live (input/output folders)
        cli.py          `inspect` / `split` / `preprocess` / `tools` commands
        core/           constants, errors, LLM settings
        documents/      reading, markdown/bodyline shapes, chapter discovery
        analysis/       TOC detection, header assessment, title location
        slicing/        stage 1: planning, verbatim slicing, verification
        preprocess/     stage 2: header inspection, damage scan, repair engine
        agent/          stage 1 agent: prompts, model, tools, graph
        preprocess_agent/  stage 2 agent: prompts, model, tools binding, graph
"""

from .agent import SplitterResult, SplitterState, build_graph, run_split
from .paths import (
    default_input_dir,
    default_output_dir,
    resolve_book_path,
    resolve_output_dir,
)
from .preprocess_agent import (
    PreprocessResult,
    PreprocessState,
    build_preprocess_graph,
    run_preprocess,
)

__all__ = [
    "PreprocessResult",
    "PreprocessState",
    "SplitterResult",
    "SplitterState",
    "build_graph",
    "build_preprocess_graph",
    "default_input_dir",
    "default_output_dir",
    "resolve_book_path",
    "resolve_output_dir",
    "run_preprocess",
    "run_split",
]
__version__ = "2.1.0"
