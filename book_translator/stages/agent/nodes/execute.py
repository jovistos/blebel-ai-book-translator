"""The execution nodes: slice the book, then verify the result.

``slice`` moves text (or previews the move in ``--dry-run``); ``verify`` proves
with SHA-256 that nothing was lost. Both are deterministic — no model involved.
"""

from __future__ import annotations

from typing import Any

from book_translator.core.constants import FRONT_MATTER_FILENAME
from book_translator.stages.slicing.slice import preview_split, slice_book
from book_translator.stages.slicing.verify import verify_output

from ..state import SplitterState


def make_slice_node():
    """Build the ``slice`` node (writes the chapter files unless dry-running)."""

    def slice_node(state: SplitterState) -> dict[str, Any]:
        include_front_matter = bool(state.get("include_front_matter", True))
        if state.get("dry_run"):
            manifest = preview_split(
                state["source_path"],
                state["proposed_splits"],
                state["output_dir"],
                include_front_matter=include_front_matter,
                front_matter_filename=FRONT_MATTER_FILENAME,
            )
            return {
                "manifest": manifest,
                "log": [
                    f"slice: DRY RUN, would write {manifest['file_count']} files "
                    f"({manifest['total_lines']}/{manifest['source_lines']} lines)"
                ],
            }
        manifest = slice_book(
            state["source_path"],
            state["proposed_splits"],
            state["output_dir"],
            include_front_matter=include_front_matter,
            front_matter_filename=FRONT_MATTER_FILENAME,
            write_manifest=True,
        )
        return {
            "manifest": manifest,
            "log": [
                f"slice: wrote {manifest['file_count']} files "
                f"({manifest['total_lines']}/{manifest['source_lines']} lines) "
                f"to {manifest['output_dir']}"
            ],
        }

    return slice_node


def make_verify_node():
    """Build the ``verify`` node that re-reads the output from disk."""

    def verify_node(state: SplitterState) -> dict[str, Any]:
        report = verify_output(state["manifest"])
        return {
            "verification": report,
            "log": [
                f"verify: ok={report['ok']} "
                f"byte_for_byte={report['reassembles_byte_for_byte']} "
                f"lines={report['output_lines']}/{report['source_lines']}"
            ],
        }

    return verify_node
