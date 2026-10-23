"""One run's manifest/trace pair: where it goes, whether it is written, and
what paths the result should report.

Every stage used to repeat the same five lines at the end of its runner::

    if write_manifest and not dry_run:
        result.manifest_path = str(write_manifest(out_dir, NAME, payload))
        result.trace_path = str(write_trace(out_dir, TRACE, result.to_dict()))

and the all-resumed paths repeated a variant that *adopts* the files an earlier
run already wrote. :class:`ArtifactStore` keeps that state once, while the stage
keeps ownership of the payload: this module never inspects or builds a manifest.

The store is deliberately narrow. It does not write chapter files, chunk files,
``glossary.json``, prompt records or debug records — those have stage-specific
verification and snapshot rules. It only coordinates the two stage-level
artifacts whose names are fixed and whose payloads are the result's own record.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

from .artifacts import write_manifest, write_trace

R = TypeVar("R")


@dataclass
class ArtifactStore:
    """The manifest/trace pair of one stage run.

    ``enabled`` is the stage's own write decision — usually
    ``write_manifest and not dry_run``, sometimes also ``and save``. When it is
    false nothing is written and both path fields stay ``None``.

    ``manifest_path``/``trace_path`` are strings because that is what every
    result class exposes. They are set when a file is written, or adopted from a
    previous run; a stage that reports an existing artifact without rewriting it
    calls :meth:`adopt`.
    """

    root: Path
    manifest_filename: str
    trace_filename: str
    enabled: bool = True
    manifest_path: str | None = None
    trace_path: str | None = None

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser()

    # ------------------------------------------------------------------ #
    # writing
    # ------------------------------------------------------------------ #
    def write_manifest(self, payload: dict[str, Any]) -> Path | None:
        """Write the manifest and record its path; no-op when disabled."""
        if not self.enabled:
            return None
        path = write_manifest(self.root, self.manifest_filename, payload)
        self.manifest_path = str(path)
        return path

    def write_trace(self, payload: Any) -> Path | None:
        """Write the trace and record its path; no-op when disabled."""
        if not self.enabled:
            return None
        path = write_trace(self.root, self.trace_filename, payload)
        self.trace_path = str(path)
        return path

    def write(self, manifest_payload: dict[str, Any], trace_payload: Any) -> None:
        """Write the pair, manifest first."""
        self.write_manifest(manifest_payload)
        self.write_trace(trace_payload)

    def write_result(
        self,
        result: R,
        manifest_payload: dict[str, Any],
        *,
        trace_payload: Any = None,
    ) -> R:
        """Write the pair for ``result`` and attach both paths.

        The manifest is written first and its path is put on ``result`` before
        the trace is serialised: the trace records the manifest it belongs to,
        which is the order every stage already used.
        """
        if not self.enabled:
            return result
        path = self.write_manifest(manifest_payload)
        if path is not None and hasattr(result, "manifest_path"):
            result.manifest_path = str(path)
        to_dict = cast(Any, result).to_dict
        self.write_trace(to_dict() if trace_payload is None else trace_payload)
        return self.attach(result)

    def snapshot(self, payload: dict[str, Any]) -> Path | None:
        """Best-effort manifest write for an in-progress run.

        A snapshot exists so a killed run keeps the work already done; a disk
        error must cost the snapshot, not the run that is still paying for model
        calls. The final :meth:`write` is not best-effort: a run that cannot
        record its result should fail.
        """
        if not self.enabled:
            return None
        try:
            return self.write_manifest(payload)
        except OSError:
            return None

    # ------------------------------------------------------------------ #
    # adopting and attaching
    # ------------------------------------------------------------------ #
    def adopt(
        self,
        *,
        manifest_path: str | Path | None = None,
        trace_path: str | Path | None = None,
    ) -> None:
        """Point at artifacts an earlier run wrote, without writing them.

        The fully reused branch of a stage deliberately skips every model and
        tokenizer; it still reports the previous run's files so a caller can
        open them.
        """
        if manifest_path is not None:
            self.manifest_path = str(manifest_path)
        if trace_path is not None:
            self.trace_path = str(trace_path)

    def attach(self, result: R) -> R:
        """Copy the recorded paths onto ``result`` and return it."""
        if hasattr(result, "manifest_path"):
            result.manifest_path = self.manifest_path
        if hasattr(result, "trace_path"):
            result.trace_path = self.trace_path
        return result

    def existing(self, name: str) -> Path | None:
        """An already-written artifact under ``root``, or ``None``.

        Used by the all-reused branches: they must not create a file, but they
        still report the one on disk.
        """
        path = self.root / name
        return path if path.is_file() else None


__all__ = ["ArtifactStore"]
