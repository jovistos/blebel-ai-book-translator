"""Fuzzy glossary retrieval for stage 6: an NER term in, a locked entry out.

Stage 4 already decided how every entity is written in the target language. This
module does **not** translate anything; it is the lookup that answers one
question per extracted term:

    *which entry of ``glossary.json`` is the term this chunk mentions?*

The lookup is deliberately fuzzy, because the NER tagger and the glossary do not
always surface a name identically:

* the tagger may return an inflected or slightly clipped form (``Džordža`` for
  ``Džordž``, ``Hamilton`` for ``Hamilton-Proctor``);
* a PDF conversion leaves punctuation and doubled spaces in one copy and not the
  other;
* the glossary's canonical form may be longer than the mention (a full name versus
  the surname the sentence used).

``rapidfuzz`` scores every glossary term against the extracted one on a 0..100
scale, and the caller keeps the best candidate only when it reaches the
configured threshold (95 % by default). The scorer is a flag, because the right
one depends on the book: ``ratio`` for near-exact names, ``token_sort_ratio``
when word order moves, ``wratio`` when a mention is a fragment of a longer name.

Everything here is import-light: ``rapidfuzz`` is imported when a matcher is
built, not when the package is imported, so a split-only install keeps working.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..core.constants import (
    CHUNK_GLOSSARY_SCORERS,
    DEFAULT_CHUNK_GLOSSARY_CANDIDATES,
    DEFAULT_CHUNK_GLOSSARY_SCORER,
    DEFAULT_CHUNK_GLOSSARY_THRESHOLD,
    GLOSSARY_FILENAME,
)
from ..core.errors import MissingDependencyError, ToolError

_RAPIDFUZZ_HINT = (
    "the chunk-glossary stage needs rapidfuzz: pip install rapidfuzz "
    "(already in requirements-book-translator.txt)"
)

#: Punctuation that can cling to an NER mention (``Džordž,`` / ``"London"``) but
#: is not part of the name. Hyphens and apostrophes are deliberately **not**
#: stripped: they live inside names and removing them would change the match.
_EDGE_PUNCTUATION = "\"'“”‘’«»‹›.,;:!?()[]{}…"


def clean_term(text: Any) -> str:
    """The display form of a term: trimmed, single-spaced, Unicode-normalised."""
    value = unicodedata.normalize("NFC", str(text or ""))
    value = " ".join(value.split())
    # Strip repeatedly: `"Džordž,"` has a quote at the edge and a comma just
    # behind it, so one pass would leave the comma in place.
    previous = None
    while value and value != previous:
        previous = value
        value = value.strip(_EDGE_PUNCTUATION).strip()
    return value


def match_key(text: Any) -> str:
    """The identity of a term for matching: case-folded, single-spaced.

    Case folding is right here for the same reason it is right in stage 4's
    ``entry_key``: the dictionary is keyed on a *name*, and a tagger that
    capitalises or inflects it differently is still naming the same entity.
    Diacritics are kept on purpose — ``Džordž`` and ``Dzordz`` are close enough
    for rapidfuzz to score, and folding them would make ``Džordž`` collide with
    an unrelated ``Dzordz`` in the book.
    """
    return clean_term(text).casefold()


def normalize_threshold(value: Any = DEFAULT_CHUNK_GLOSSARY_THRESHOLD) -> float:
    """Validate the 0..100 fuzzy threshold, with an actionable message."""
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ToolError(f"match threshold must be a number between 0 and 100, got {value!r}") from exc
    if not 0.0 <= threshold <= 100.0:
        raise ToolError(
            f"match threshold must be between 0 and 100 (it is a percentage), got {threshold}"
        )
    return threshold


def normalize_scorer(value: str | None = None) -> str:
    """Validate the rapidfuzz scorer name."""
    scorer = str(value or DEFAULT_CHUNK_GLOSSARY_SCORER).strip().lower()
    if scorer not in CHUNK_GLOSSARY_SCORERS:
        raise ToolError(
            f"unknown fuzzy scorer {scorer!r}; pick one of "
            f"{', '.join(CHUNK_GLOSSARY_SCORERS)}"
        )
    return scorer


@dataclass(frozen=True)
class LexiconEntry:
    """One locked glossary entry, as stage 6 reads it from ``glossary.json``."""

    term: str
    translation: str
    category: str = "terminology"
    chapter_index: int = 0
    chapter_filename: str = ""

    @property
    def key(self) -> str:
        return match_key(self.term)

    def to_dict(self) -> dict[str, Any]:
        return {
            "term": self.term,
            "category": self.category,
            "translation": self.translation,
            "chapter_index": self.chapter_index,
            "chapter_filename": self.chapter_filename,
        }


@dataclass(frozen=True)
class MatchCandidate:
    """One glossary entry retrieved for a term, and rapidfuzz's score for it."""

    entry: LexiconEntry
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {**self.entry.to_dict(), "score": round(float(self.score), 2)}


class TermMatcher(Protocol):
    """What the chunk-glossary runner needs from a matcher.

    Kept as a protocol so the tests — and anything wanting a different retrieval
    strategy — can inject a matcher without a rapidfuzz dependency. The runner
    only ever calls :meth:`match`; it never imports rapidfuzz itself.
    """

    def match(self, term: str, *, limit: int | None = None) -> list[MatchCandidate]:
        """The best glossary candidates for ``term``, highest score first."""
        ...


def glossary_file(path: str | Path) -> Path:
    """Resolve a glossary folder or file to the ``glossary.json`` inside it."""
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        return candidate / GLOSSARY_FILENAME
    return candidate


def load_glossary(path: str | Path) -> list[LexiconEntry]:
    """Read ``glossary.json`` into a de-duplicated, order-preserving entry list.

    Stage 4 writes ``{"entries": [...]}}``. A bare list is accepted too, because
    the file is also consumed by hand and by other tools. Entries without a term
    or a translation are skipped: they cannot be attached to a chunk, and failing
    the whole run over one malformed row would make a good dictionary unusable.
    """
    file = glossary_file(path)
    if not file.is_file():
        raise ToolError(
            f"glossary not found: {file} — run `glossary` first to build it"
        )
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ToolError(f"could not read {file}: {type(exc).__name__}: {exc}") from exc

    raw = payload.get("entries") if isinstance(payload, dict) else payload
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise ToolError(f"{file} does not hold an entries list")

    entries: list[LexiconEntry] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        term = clean_term(item.get("term"))
        translation = " ".join(str(item.get("translation") or "").split())
        if not term or not translation:
            continue
        key = match_key(term)
        if not key or key in seen:
            continue
        seen.add(key)
        try:
            chapter_index = int(item.get("chapter_index") or 0)
        except (TypeError, ValueError):
            chapter_index = 0
        entries.append(
            LexiconEntry(
                term=term,
                translation=translation,
                category=str(item.get("category") or "terminology"),
                chapter_index=chapter_index,
                chapter_filename=str(item.get("chapter_filename") or ""),
            )
        )
    return entries


def _rapidfuzz():
    """Import rapidfuzz lazily, with the install command in the error."""
    try:
        from rapidfuzz import fuzz, process
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise MissingDependencyError(_RAPIDFUZZ_HINT) from exc
    return fuzz, process


def _scorer_function(name: str):
    """Map a scorer name from the CLI onto ``rapidfuzz.fuzz``."""
    fuzz, _process = _rapidfuzz()
    return {
        "ratio": fuzz.ratio,
        "token_sort_ratio": fuzz.token_sort_ratio,
        "token_set_ratio": fuzz.token_set_ratio,
        "wratio": fuzz.WRatio,
        "partial_ratio": fuzz.partial_ratio,
        "qratio": fuzz.QRatio,
    }[name]


class RapidFuzzMatcher:
    """The default matcher: score one term against every glossary term.

    The glossary is small (hundreds of entries), so a linear ``process.extract``
    is simpler and more than fast enough; no index has to be kept in sync. The
    normalized keys are computed once, when the matcher is built, because the
    matching phase scores many terms against the same dictionary.
    """

    def __init__(
        self,
        entries: list[LexiconEntry] | tuple[LexiconEntry, ...],
        *,
        scorer: str = DEFAULT_CHUNK_GLOSSARY_SCORER,
        candidates: int = DEFAULT_CHUNK_GLOSSARY_CANDIDATES,
    ) -> None:
        self.entries = list(entries)
        self.scorer = normalize_scorer(scorer)
        self.candidates = max(1, int(candidates))
        _fuzz, process = _rapidfuzz()
        self._process = process
        self._score = _scorer_function(self.scorer)
        self._keys = [match_key(entry.term) for entry in self.entries]

    def match(self, term: str, *, limit: int | None = None) -> list[MatchCandidate]:
        """Return the top candidates for ``term``, best first.

        Every score is returned, including one below the threshold: the caller
        needs the near miss for its per-chunk report. The threshold is applied
        there, not here, so the same matcher can be reused at another threshold.
        """
        if not self.entries:
            return []
        query = match_key(term)
        if not query:
            return []
        count = self.candidates if limit is None else max(1, int(limit))
        results = self._process.extract(
            query,
            self._keys,
            scorer=self._score,
            limit=count,
            score_cutoff=0,
        )
        candidates: list[MatchCandidate] = []
        for _choice, score, index in results:
            position = int(index)
            if 0 <= position < len(self.entries):
                candidates.append(
                    MatchCandidate(entry=self.entries[position], score=float(score))
                )
        return candidates


__all__ = [
    "LexiconEntry",
    "MatchCandidate",
    "RapidFuzzMatcher",
    "TermMatcher",
    "clean_term",
    "glossary_file",
    "load_glossary",
    "match_key",
    "normalize_scorer",
    "normalize_threshold",
]
