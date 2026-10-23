"""Loading a chapter map from a JSON file: the ``--map`` escape hatch.

The split agent cannot know what the file does not say. When a chapter's heading
was dropped by the conversion, deciding *which* paragraph opens it needs the
published contents list and a reader's judgement — sometimes a human's. This
module lets that judgement be written down once and replayed deterministically:

    python -m book_translator split BOOK.md --map my-map.json

No model, no API key, no web: the map is validated with exactly the same checks
the graph applies to an agent's plan, then sliced and verified byte for byte.

Accepted JSON shapes
--------------------
A bare list of entries::

    [{"line_number": 227, "filename": "03-2.md",
      "role": "prose-chapter", "evidence": "publisher contents list: chapter 2"}]

…or an object holding that list under one of ``splits``, ``split_mapping``,
``map``, ``chapters`` or ``entries``. A richer report (for example a recovered
map with per-entry ``provenance`` and ``confidence``) can be passed unchanged:
only the fields below are read, everything else is ignored.

Per entry:

``line_number``
    Required. Aliases: ``line``, ``start_line``.
``filename``
    Required. Aliases: ``file``, ``name``, ``output``. Normalised by the slicer,
    so directory components cannot escape the output folder.
``title``
    Optional. Aliases: ``original_header``.
``role``
    Optional. Aliases: ``kind``. ``prose-chapter`` for a chapter whose heading is
    gone, ``back-matter`` for the trailing block — anything else is treated as a
    normal chapter.
``evidence``
    Required for ``role: "prose-chapter"``. Aliases: ``reason``,
    ``justification``, ``why``, ``source``, ``provenance``.

An entry for line 1 whose filename is the front-matter file is dropped with a
note: everything before the first split point is written to
``00-front-matter.md`` automatically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from book_translator.core.constants import FRONT_MATTER_FILENAME
from book_translator.core.errors import ToolError
from book_translator.documents.text import normalize_filename
from book_translator.stages.analysis.prose import EVIDENCE_FIELDS

#: Object keys that may hold the list of entries.
MAP_LIST_KEYS: tuple[str, ...] = ("splits", "split_mapping", "map", "chapters", "entries")

#: Field aliases, in the order they are read.
LINE_FIELDS: tuple[str, ...] = ("line_number", "line", "start_line")
NAME_FIELDS: tuple[str, ...] = ("filename", "file", "name", "output")
TITLE_FIELDS: tuple[str, ...] = ("title", "original_header")
ROLE_FIELDS: tuple[str, ...] = ("role", "kind")


def load_split_map(path: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read a chapter map from ``path``.

    Returns ``(mapping, notes)``: the entries in the shape
    :func:`~book_translator.stages.slicing.validation.validate_split_map` expects, plus
    human-readable notes about anything that was adjusted or ignored. Raises
    :class:`~book_translator.core.errors.ToolError` with every problem it found when
    the file cannot be used.
    """
    file_path = Path(str(path)).expanduser()
    if not file_path.is_file():
        raise ToolError(f"map file not found: {file_path}")
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ToolError(f"map file {file_path} is not valid JSON: {exc}") from exc
    except OSError as exc:  # unreadable file / permission
        raise ToolError(f"map file {file_path} could not be read: {exc}") from exc

    entries = _entries_of(payload, file_path)
    notes: list[str] = []
    mapping: list[dict[str, Any]] = []
    problems: list[str] = []

    for index, raw in enumerate(entries, start=1):
        if not isinstance(raw, dict):
            problems.append(f"entry {index} is not an object")
            continue
        line = _first(raw, LINE_FIELDS)
        try:
            line_number = int(line)
        except (TypeError, ValueError):
            problems.append(f"entry {index} has no integer line number (got {line!r})")
            continue
        name = _first(raw, NAME_FIELDS)
        if not name:
            problems.append(f"entry {index} (line {line_number}) has no filename")
            continue

        item: dict[str, Any] = {"line_number": line_number, "filename": str(name)}
        title = _first(raw, TITLE_FIELDS)
        if title:
            item["title"] = str(title)
        role = _first(raw, ROLE_FIELDS)
        if role:
            item["role"] = str(role).strip().lower()
        evidence = _first(raw, EVIDENCE_FIELDS)
        if evidence:
            item["evidence"] = str(evidence).strip()

        if line_number == 1 and normalize_filename(str(name)) == normalize_filename(
            FRONT_MATTER_FILENAME
        ):
            notes.append(
                f"line 1 ({normalize_filename(str(name))}) was ignored: content before the "
                "first split point is written to the front-matter file automatically"
            )
            continue
        mapping.append(item)

    if problems:
        raise ToolError(
            f"map file {file_path} cannot be used: " + "; ".join(problems)
        )
    if not mapping:
        raise ToolError(f"map file {file_path} holds no split points")

    notes.append(f"map file {file_path}: {len(mapping)} split point(s), no model calls")
    return mapping, notes


def _entries_of(payload: Any, file_path: Path) -> list[Any]:
    """The list of entries inside whatever shape the file used."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in MAP_LIST_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                return value
        raise ToolError(
            f"map file {file_path} has no list of split points under any of "
            f"{', '.join(MAP_LIST_KEYS)}"
        )
    raise ToolError(
        f"map file {file_path} must hold a list of split points or an object with one "
        f"(found {type(payload).__name__})"
    )


def _first(entry: dict[str, Any], fields: tuple[str, ...]) -> Any:
    """The first non-empty value among ``fields``."""
    for field in fields:
        value = entry.get(field)
        if value not in (None, "", []):
            return value
    return None


__all__ = ["LINE_FIELDS", "MAP_LIST_KEYS", "NAME_FIELDS", "load_split_map"]
