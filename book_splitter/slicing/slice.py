"""Writing the chapter files: the only place book content is ever written."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..core.artifacts import Sha256, write_manifest
from ..core.constants import FRONT_MATTER_FILENAME, MANIFEST_FILENAME
from ..documents.io import load_lines
from ..documents.text import excerpt
from .hashing import sha256_file
from .plan import describe_first_line, plan_files


def preview_split(
    source_path: str,
    split_mapping: Iterable[dict[str, Any]],
    output_dir: str,
    include_front_matter: bool = True,
    front_matter_filename: str = FRONT_MATTER_FILENAME,
) -> dict[str, Any]:
    """Compute the exact chapter files that *would* be written, writing nothing."""
    lines = load_lines(source_path)
    planned = plan_files(lines, split_mapping, include_front_matter, front_matter_filename)
    files = []
    for item in planned:
        span = lines[item["start_line"] - 1 : item["end_line"]]
        files.append(
            {
                **item,
                "lines": len(span),
                "bytes": sum(len(line.encode("utf-8")) for line in span),
                "first_line": describe_first_line(lines, item["start_line"], item["end_line"]),
            }
        )
    return {
        "dry_run": True,
        "source_path": str(Path(source_path).expanduser().resolve()),
        "output_dir": str(Path(output_dir).expanduser().resolve()),
        "source_lines": len(lines),
        "file_count": len(files),
        "chapter_count": sum(1 for f in files if f["kind"] == "chapter"),
        "total_lines": sum(f["lines"] for f in files),
        "files": files,
    }


def slice_book(
    source_path: str,
    split_mapping: Iterable[dict[str, Any]],
    output_dir: str,
    include_front_matter: bool = True,
    front_matter_filename: str = FRONT_MATTER_FILENAME,
    write_manifest: bool = True,
) -> dict[str, Any]:
    """Copy the source into one file per chapter, verbatim, by line range.

    Returns a manifest (filenames, line ranges, sizes, hashes) — never content.
    """
    lines = load_lines(source_path)
    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    planned = plan_files(lines, split_mapping, include_front_matter, front_matter_filename)
    files = [
        _write_range(
            out_dir / item["filename"],
            lines,
            item["start_line"],
            item["end_line"],
            item["filename"],
            item["kind"],
        )
        for item in planned
    ]

    manifest = {
        "source_path": str(Path(source_path).expanduser().resolve()),
        "source_lines": len(lines),
        "source_sha256": sha256_file(source_path),
        "output_dir": str(out_dir.resolve()),
        "include_front_matter": include_front_matter,
        "file_count": len(files),
        "chapter_count": sum(1 for f in files if f["kind"] == "chapter"),
        "total_lines": sum(f["lines"] for f in files),
        "files": files,
    }

    if write_manifest:
        write_manifest_file(out_dir, manifest)
    return manifest


def write_manifest_file(out_dir: Path, manifest: dict[str, Any]) -> Path:
    """Persist the manifest next to the chapter files."""
    return write_manifest(out_dir, MANIFEST_FILENAME, manifest)


def _write_range(
    path: Path, lines: list[str], start: int, end: int, name: str, kind: str
) -> dict[str, Any]:
    """Write one verbatim line range and return its bookkeeping entry."""
    digest = Sha256()
    n_bytes = 0
    n_lines = 0
    with path.open("w", encoding="utf-8", newline="") as fh:
        for i in range(start - 1, end):
            line = lines[i]
            fh.write(line)
            digest.update(line)
            n_bytes += len(line.encode("utf-8"))
            n_lines += 1
    return {
        "filename": name,
        "path": str(path),
        "kind": kind,
        "start_line": start,
        "end_line": end,
        "lines": n_lines,
        "bytes": n_bytes,
        "sha256": digest.hexdigest(),
        "first_line": excerpt(
            next((line for line in lines[start - 1 : end + 3] if line.strip()), ""), 80
        ),
    }
