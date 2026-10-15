"""Tools that touch the filesystem: slicing and verification.

They are the only tools allowed to write book content, and the graph blocks
them until a plan has passed validation (see ``policy.py``). Both return
bookkeeping metadata only — never the text they wrote.
"""

from __future__ import annotations

from langchain_core.tools import tool

from ...core.constants import FRONT_MATTER_FILENAME
from ...documents.text import jdump
from ...slicing.slice import slice_book
from ...slicing.verify import verify_output_dir
from .registry import split_stage


@split_stage
@tool
def slice_book_by_line_numbers(
    file_path: str,
    split_mapping: list[dict],
    output_dir: str,
    include_front_matter: bool = True,
) -> str:
    """Physically write one file per chapter, copying the source lines verbatim.

    This is the only tool that writes book content. It never reads the text back
    to the model: it returns a manifest with filenames, line ranges, byte counts
    and SHA-256 hashes. A `_manifest.json` is written into `output_dir` with the
    same information, and any content before the first split point is saved to
    `00-front-matter.md` when `include_front_matter` is true.

    Do not call this until `validate_split_map_tool` returns `ok: true`.
    """
    manifest = slice_book(
        file_path,
        split_mapping,
        output_dir,
        include_front_matter=include_front_matter,
        front_matter_filename=FRONT_MATTER_FILENAME,
        write_manifest=True,
    )
    return jdump(manifest)


@split_stage
@tool
def verify_sliced_output(output_dir: str) -> str:
    """Verify the split output: files present, ranges contiguous, no content lost.

    Re-reads `_manifest.json` (or the files themselves) and checks that the
    chapter files concatenate back to the exact source bytes, using SHA-256.
    Returns `{"ok", "reassembles_byte_for_byte", "output_lines", ...}`.
    Call this last and report the result.
    """
    return jdump(verify_output_dir(output_dir))
