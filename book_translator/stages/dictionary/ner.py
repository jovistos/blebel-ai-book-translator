"""The NER gate: which sentences survive, and which tagger decides.

The rule the stage implements is deliberately blunt: **keep a sentence only if a
tagger found at least one PER, LOC, ORG or MISC mention in it.** Zero-entity
sentences — generic dialogue, pure action, generic description — are dropped,
because they cannot teach a dictionary anything about a noun.

The tagger is a *gatekeeper, never the dictionary builder*. CLASSLA's NER is a
traditional four-type model: it has no idea that ``Mithril`` is a substance and
will tag it MISC or nothing at all. That is fine here — the surviving sentences
are handed to :mod:`book_translator.stages.glossary` (stage 4), which is the part that
can read context.

Stage 6 (:mod:`book_translator.stages.chunk_glossary`) reuses exactly these backends, but
with a different question: it tags every chunk to *find* the terms a chunk
mentions, then retrieves their locked renderings from the glossary. Same models,
same language routing, same device selection.

Two backends, chosen by language (:func:`backend_for_language`):

``classla``
    Serbian, Croatian, Slovenian and Bulgarian (``mk`` has no NER model).
``stanza``
    everything else this pipeline meets, English today.

Both are imported lazily; neither is a hard dependency of the package. Models are
downloaded on first use, and the physical GPU is chosen through
``CUDA_VISIBLE_DEVICES`` (:func:`configure_gpu`) because that is the one switch
both libraries honour.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Protocol

from book_translator.core.constants import (
    CLASSLA_LANGUAGES,
    CLASSLA_NER_LANGUAGES,
    DEFAULT_GPU_INDEX,
    DEFAULT_KEEP_ENTITY_TYPES,
    STANZA_LANGUAGES,
)
from book_translator.core.errors import MissingDependencyError, ToolError

_CLASSLA_HINT = (
    "the Serbian/Croatian/Slovenian/Bulgarian tagger needs CLASSLA: "
    "pip install -r requirements-dictionary.txt  (or: pip install classla)"
)
_STANZA_HINT = (
    "the English tagger needs stanza: "
    "pip install -r requirements-dictionary.txt  (or: pip install stanza)"
)

#: Entity types both backends emit, upper-cased, with any IOB prefix stripped.
_KEEP_TYPES: tuple[str, ...] = DEFAULT_KEEP_ENTITY_TYPES

#: The only two processors this stage uses: sentence splitting and the NER gate.
#: Requesting the same set at *download* time keeps the first run small — the
#: default packages add pos/lemma/depparse (and, for Serbian, a 110 MB pretrain
#: set) that stage 3 would never load: ~450 MB instead of ~800 MB for Serbian,
#: and ~60 MB instead of ~250 MB for English.
_PROCESSORS = "tokenize,ner"

#: stanza's models emit the **18 OntoNotes types**, CLASSLA's emit four. The gate
#: is written against the four, so stanza's labels are folded onto them the way
#: the CoNLL-2003 conversion does. Without this every ``PERSON`` and ``GPE`` falls
#: through the gate as an unknown type — which silently threw away every person
#: and city in an entire English novel (244 sentences kept, 0 PER).
#:
#: Anything absent from this map (``DATE``, ``TIME``, ``CARDINAL``, ``MONEY``,
#: ``PERCENT``, ``QUANTITY``, ``ORDINAL``) stays unmapped, so a date cannot save a
#: sentence on its own.
_STANZA_TO_CLASSLA: dict[str, str] = {
    "PERSON": "PER",
    "GPE": "LOC",  # countries, cities, states
    "LOC": "LOC",  # natural locations: rivers, seas, mountains
    "FAC": "LOC",  # facilities: buildings, airports, bridges
    "ORG": "ORG",
    "NORP": "MISC",  # nationalities, religious or political groups
    "EVENT": "MISC",
    "WORK_OF_ART": "MISC",
    "LAW": "MISC",
    "LANGUAGE": "MISC",
    "PRODUCT": "MISC",
}

_MONTHS = frozenset(
    {
        "januar", "februar", "mart", "april", "maj", "jun", "jul", "avgust",
        "septembar", "oktobar", "novembar", "decembar",
        # genitive forms, which is what a date inside a sentence uses
        "januara", "februara", "marta", "aprila", "maja", "juna", "jula",
        "avgusta", "septembra", "oktobra", "novembra", "decembra",
        "siječnja", "veljače", "ožujka", "travnja", "svibnja", "lipnja", "srpnja",
        "kolovoza", "rujna", "listopada", "studenog", "prosinca",
        "january", "february", "march", "may", "june", "july",
        "august", "september", "october", "november", "december",
    }
)
_WEEKDAYS = frozenset(
    {
        "ponedeljak", "utorak", "sreda", "četvrtak", "petak", "subota", "nedelja",
        "ponedjeljak", "srijeda", "nedjelja",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    }
)
_YEAR = re.compile(r"^(20\d{2}|19\d{2}|18\d{2})\s*\.?\s*(g|god|godine)?\.?$", re.IGNORECASE)
_CLOCK = re.compile(r"^\d{1,2}[:.]\d{2}(\s*(h|sati|am|pm))?$", re.IGNORECASE)
_DATE = re.compile(r"^\d{1,2}\s*\.\s*\d{1,2}\s*\.\s*\d{2,4}\.?$")
_TEMPORAL_WORDS = frozenset(
    {
        "dan", "danas", "juče", "jutros", "sutra", "noć", "noćas", "veče", "godina",
        "godine", "mesec", "mjesec", "nedelja", "tjedan", "week", "day", "today",
        "yesterday", "tomorrow", "tonight", "morning", "evening", "night", "year",
        "month", "summer", "winter", "spring", "autumn", "fall", "leto", "zima",
        "proleće", "jesen",
    }
)


@dataclass(frozen=True)
class Entity:
    """One mention a tagger found: its surface text and its type."""

    text: str
    type: str

    def to_dict(self) -> dict:
        return {"text": self.text, "type": self.type}


class NerBackend(Protocol):
    """What the runner needs from a tagger."""

    name: str
    language: str

    def entities(self, text: str) -> list[Entity]:
        """Entity mentions in ``text`` (usually one sentence)."""


def backend_device(backend: object) -> str:
    """The device a built backend actually runs on: ``cuda``, ``cpu`` or ``""``.

    Both classla and stanza decide the device themselves
    (``torch.cuda.is_available() and use_gpu``) and expose the verdict on their
    pipeline, so this reports what happened rather than what was asked for — a
    ``--gpu 1`` run on a box whose driver is unreachable is a CPU run, and the
    summary should say so.
    """
    pipeline = getattr(backend, "_pipeline", None)
    device = getattr(pipeline, "device", None)  # stanza >= 1.5: "cpu" / "cuda:0"
    if isinstance(device, str) and device:
        return "cuda" if device.startswith("cuda") else device
    use_gpu = getattr(pipeline, "use_gpu", None)  # classla
    if use_gpu is None:
        return ""
    return "cuda" if use_gpu else "cpu"


# --------------------------------------------------------------------------- #
# device selection
# --------------------------------------------------------------------------- #
def configure_gpu(
    gpu_index: int | None = None,
    *,
    use_gpu: bool = True,
    default: int = DEFAULT_GPU_INDEX,
) -> int | None:
    """Pin the NER tagger to one physical GPU, before torch is imported.

    ``classla`` and ``stanza`` both decide the device from
    ``CUDA_VISIBLE_DEVICES`` at import/construction time, so this must run before
    the backend is built. The rules, in order:

    1. ``use_gpu=False`` (``--cpu``) clears the variable and reports ``None``;
    2. an explicit ``gpu_index`` (``--gpu 1``) wins;
    3. otherwise an already-exported ``CUDA_VISIBLE_DEVICES`` is respected;
    4. otherwise the machine default (:data:`DEFAULT_GPU_INDEX`) is used.

    Returns the physical index in effect, or ``None`` for CPU.
    """
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return None
    if gpu_index is None:
        existing = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if existing:
            head = existing.split(",")[0].strip()
            return int(head) if head.isdigit() else None
        gpu_index = default
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu_index))
    return int(gpu_index)


def backend_for_language(language: str) -> str:
    """Name the backend that serves ``language`` (``classla`` or ``stanza``)."""
    code = (language or "").strip().lower()
    if code in CLASSLA_NER_LANGUAGES:
        return "classla"
    if code in CLASSLA_LANGUAGES:
        raise ToolError(
            f"CLASSLA has no NER model for {code!r} (it is tokenize/POS/lemma only); "
            "stage 3 cannot gate sentences with it"
        )
    if code in STANZA_LANGUAGES:
        return "stanza"
    supported = ", ".join(sorted(set(CLASSLA_NER_LANGUAGES) | set(STANZA_LANGUAGES)))
    raise ToolError(f"no NER backend for language {code!r}; supported: {supported}")


def _looks_like_missing_models(exc: Exception) -> bool:
    message = str(exc).lower()
    return "resources" in message or "not found" in message or "no such file" in message


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
class ClasslaBackend:
    """CLASSLA's neural NER, for Serbian, Croatian, Slovenian and Bulgarian."""

    name = "classla"

    def __init__(
        self,
        language: str = "sr",
        *,
        use_gpu: bool = True,
        verbose: bool = False,
        auto_download: bool = True,
    ) -> None:
        code = (language or "").strip().lower()
        if code not in CLASSLA_NER_LANGUAGES:
            raise ToolError(
                f"CLASSLA NER does not cover {code!r}; "
                f"use one of {', '.join(sorted(CLASSLA_NER_LANGUAGES))}"
            )
        try:  # lazy: torch only gets imported when stage 3 actually runs
            import classla
        except ImportError as exc:
            raise MissingDependencyError(_CLASSLA_HINT) from exc

        self.language = code
        self._classla = classla
        self._pipeline = self._build(use_gpu, verbose, auto_download)

    def _build(self, use_gpu: bool, verbose: bool, auto_download: bool):
        kwargs = {
            "processors": _PROCESSORS,
            "use_gpu": bool(use_gpu),
            "verbose": bool(verbose),
            "logging_level": "WARNING",
        }
        try:
            return self._classla.Pipeline(self.language, **kwargs)
        except Exception as exc:
            if not auto_download or not _looks_like_missing_models(exc):
                raise
            self._classla.download(self.language, processors=_PROCESSORS)
            return self._classla.Pipeline(self.language, **kwargs)

    def entities(self, text: str) -> list[Entity]:
        return _document_entities(self._pipeline(text))


class StanzaBackend:
    """stanza's NER, for the languages CLASSLA does not cover (English today)."""

    name = "stanza"

    def __init__(
        self,
        language: str = "en",
        *,
        use_gpu: bool = True,
        verbose: bool = False,
        auto_download: bool = True,
    ) -> None:
        code = (language or "").strip().lower()
        try:  # lazy: torch only gets imported when stage 3 actually runs
            import stanza
        except ImportError as exc:
            raise MissingDependencyError(_STANZA_HINT) from exc

        self.language = code
        self._stanza = stanza
        self._pipeline = self._build(use_gpu, verbose, auto_download)

    def _build(self, use_gpu: bool, verbose: bool, auto_download: bool):
        kwargs = {
            "lang": self.language,
            "processors": _PROCESSORS,
            "use_gpu": bool(use_gpu),
            "verbose": bool(verbose),
            "logging_level": "WARNING",
        }
        try:
            return self._stanza.Pipeline(**kwargs)
        except Exception as exc:
            if not auto_download or not _looks_like_missing_models(exc):
                raise
            self._stanza.download(self.language, processors=_PROCESSORS)
            return self._stanza.Pipeline(**kwargs)

    def entities(self, text: str) -> list[Entity]:
        return _document_entities(self._pipeline(text), type_map=_STANZA_TO_CLASSLA)


def _document_entities(doc, type_map: dict[str, str] | None = None) -> list[Entity]:
    """Pull ``(text, type)`` out of a classla/stanza document.

    Reads per-sentence entities (present in every version) and falls back to the
    document-level list when a sentence carries none. ``type_map`` folds a
    backend's own label set onto CLASSLA's four (see ``_STANZA_TO_CLASSLA``).
    """
    def convert(ent) -> Entity | None:
        text = getattr(ent, "text", "")
        if not text:
            return None
        entity_type = str(getattr(ent, "type", ""))
        if type_map:
            entity_type = type_map.get(entity_type.upper(), entity_type)
        return Entity(text=str(text), type=entity_type)

    out: list[Entity] = []
    sentences = getattr(doc, "sentences", None) or []
    for sentence in sentences:
        for ent in getattr(sentence, "ents", None) or []:
            entity = convert(ent)
            if entity is not None:
                out.append(entity)
    if not out:
        for ent in getattr(doc, "ents", None) or []:
            entity = convert(ent)
            if entity is not None:
                out.append(entity)
    return out


def make_backend(
    language: str,
    *,
    use_gpu: bool = True,
    verbose: bool = False,
    auto_download: bool = True,
    backend: str | None = None,
) -> NerBackend:
    """Build the backend for ``language``, or the explicitly named one."""
    name = (backend or backend_for_language(language)).strip().lower()
    if name == "classla":
        return ClasslaBackend(
            language, use_gpu=use_gpu, verbose=verbose, auto_download=auto_download
        )
    if name == "stanza":
        return StanzaBackend(
            language, use_gpu=use_gpu, verbose=verbose, auto_download=auto_download
        )
    raise ToolError(f"unknown NER backend {backend!r}; expected 'classla' or 'stanza'")


# --------------------------------------------------------------------------- #
# the gate itself
# --------------------------------------------------------------------------- #
def normalize_type(entity_type: str) -> str:
    """``B-PER`` / ``I-LOC`` / ``per`` -> ``PER`` / ``LOC``."""
    value = str(entity_type or "").strip().upper()
    for prefix in ("B-", "I-", "E-", "S-", "U-", "L-"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
            break
    return value


def is_temporal(entity: Entity) -> bool:
    """True for time and date mentions.

    CLASSLA and stanza have no TIME/DATE type — a date is usually MISC or no
    entity at all — so this is a small surface-form test, used only when
    ``--drop-temporal-only`` is on.
    """
    text = str(entity.text or "").strip().strip(".,;:")
    if not text:
        return False
    lowered = text.lower()
    if _YEAR.match(text) or _CLOCK.match(text) or _DATE.match(text):
        return True
    words = [w for w in re.split(r"[\s.]+", lowered) if w]
    if not words:
        return False
    return all(
        w in _MONTHS or w in _WEEKDAYS or w in _TEMPORAL_WORDS or w.isdigit() for w in words
    )


@dataclass(frozen=True)
class GateDecision:
    """What the gate decided about one sentence, and why."""

    entities: list[Entity]
    keep: bool
    reason: str  #: ``kept`` | ``no_entity`` | ``temporal_only``

    @property
    def types(self) -> list[str]:
        return [normalize_type(e.type) for e in self.entities]

    def to_dict(self) -> dict:
        return {
            "keep": self.keep,
            "reason": self.reason,
            "entities": [e.to_dict() for e in self.entities],
        }


def gate_entities(
    entities: list[Entity],
    *,
    keep_types: tuple[str, ...] = _KEEP_TYPES,
    drop_temporal_only: bool = False,
) -> GateDecision:
    """Decide whether a sentence survives the NER gate.

    A sentence with no entity of a wanted type is dropped (``no_entity``). With
    ``drop_temporal_only`` on, a sentence whose *only* wanted entities are
    temporal is dropped too (``temporal_only``), so ``"Sreli smo se 1996."`` does
    not survive on the strength of a year.
    """
    wanted = {str(t).strip().upper() for t in keep_types}
    kept = [e for e in entities if normalize_type(e.type) in wanted]
    if not kept:
        return GateDecision(entities=[], keep=False, reason="no_entity")
    if drop_temporal_only and all(is_temporal(e) for e in kept):
        return GateDecision(entities=kept, keep=False, reason="temporal_only")
    return GateDecision(entities=kept, keep=True, reason="kept")


__all__ = [
    "ClasslaBackend",
    "Entity",
    "GateDecision",
    "NerBackend",
    "StanzaBackend",
    "backend_device",
    "backend_for_language",
    "configure_gpu",
    "gate_entities",
    "is_temporal",
    "make_backend",
    "normalize_type",
]
