"""Proving that the produced chapter files reassemble into the source.

The verifier re-reads everything from disk, so it also catches a truncated,
partially written or since-edited output directory — not just a bad plan.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.artifacts import Sha256
from ..core.constants import MANIFEST_FILENAME
from ..documents.io import load_lines
from .hashing import sha256_lines

_CHUNK = 1 << 20


def source_digest_for_manifest(manifest: dict[str, Any]) -> str:
    """Hash the exact source region the manifest claims to cover."""
    lines = load_lines(manifest["source_path"])
    start = 1 if manifest.get("include_front_matter") else manifest["files"][0]["start_line"]
    return sha256_lines(lines, start)


def verify_output(manifest: dict[str, Any]) -> dict[str, Any]:
    """Prove the chapter files reassemble into the source, byte for byte."""
    out_dir = Path(manifest["output_dir"])
    problems: list[str] = []
    missing: list[str] = []
    lines_total = 0
    concatenated = Sha256()
    contiguous = True
    expected_next = None

    for f in manifest["files"]:
        path = out_dir / f["filename"]
        if not path.is_file():
            missing.append(f["filename"])
            contiguous = False
            continue
        digest = Sha256()
        n_bytes = 0
        n_lines = 0
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(_CHUNK), b""):
                digest.update(chunk)
                concatenated.update(chunk)
                n_bytes += len(chunk)
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for _ in fh:
                n_lines += 1
        lines_total += n_lines
        if digest.hexdigest() != f["sha256"]:
            problems.append(f"{f['filename']}: content hash changed since slicing")
        if n_bytes != f["bytes"]:
            problems.append(f"{f['filename']}: {n_bytes} bytes on disk, expected {f['bytes']}")
        if expected_next is not None and f["start_line"] != expected_next:
            contiguous = False
            problems.append(
                f"{f['filename']}: starts at line {f['start_line']}, expected {expected_next}"
            )
        expected_next = f["end_line"] + 1

    expected_digest = source_digest_for_manifest(manifest)
    return {
        "ok": not problems and not missing and lines_total == manifest["total_lines"],
        "reassembles_byte_for_byte": concatenated.hexdigest() == expected_digest,
        "source_sha256": manifest["source_sha256"],
        "expected_region_sha256": expected_digest,
        "output_sha256": concatenated.hexdigest(),
        "source_lines": manifest["source_lines"],
        "output_lines": lines_total,
        "files_checked": len(manifest["files"]),
        "missing_files": missing,
        "problems": problems,
        "line_ranges_contiguous": contiguous,
    }


def verify_output_dir(output_dir: str | Path) -> dict[str, Any]:
    """Load ``_manifest.json`` from an output folder and verify the files."""
    manifest_path = Path(output_dir).expanduser() / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return {"ok": False, "error": f"no manifest at {manifest_path}"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return verify_output(manifest)
