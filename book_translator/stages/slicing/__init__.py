"""Planning, writing and verifying the chapter files.

The contract of this package: text is only ever *moved*, never rewritten. The
planner decides the ranges, the slicer copies the bytes, and the verifier proves
with SHA-256 that the chapter files concatenate back to the source.
"""

from .hashing import sha256_file, sha256_lines
from .plan import (
    describe_first_line,
    gap_warnings,
    plan_files,
    reconcile_filename,
    unsafe_filename,
)
from .slice import preview_split, slice_book, write_manifest_file
from .validation import validate_source, validate_split_map
from .verify import source_digest_for_manifest, verify_output, verify_output_dir

__all__ = [
    "describe_first_line",
    "gap_warnings",
    "plan_files",
    "preview_split",
    "reconcile_filename",
    "sha256_file",
    "sha256_lines",
    "slice_book",
    "source_digest_for_manifest",
    "unsafe_filename",
    "validate_source",
    "validate_split_map",
    "verify_output",
    "verify_output_dir",
    "write_manifest_file",
]
