"""Stage 10: concatenate the postprocessed chapters into one Markdown book.

This is the final, deterministic step. It reads ``09-postprocessed-chapters/`` in
reading order — front matter, the table of contents (when it is a separate file),
the chapters, and back matter — and writes one ``10-translated-book.md``. It inserts
exactly one blank line between files and nothing else; the postprocessed files
remain the canonical per-chapter artifacts.
"""

from .runner import BindResult, run_bind

__all__ = ["BindResult", "run_bind"]
