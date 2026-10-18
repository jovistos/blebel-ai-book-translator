"""Which language is this book in? Sampling, detection, routing.

CLASSLA's own models only cover the South Slavic languages (``sl``, ``hr``,
``sr``, ``bg``, ``mk``), so an English book has to go to stanza instead. Rather
than make the caller guess, stage 3 samples a few random snippets — one per
chapter — detects their language with ``lingua-py`` and routes the whole book to
the tagger that speaks it. ``--language sr`` skips detection entirely, which is
what a mixed or very short book needs.

Serbian and Croatian are the same language written two ways — *ekavian* (``vreme``,
``dete``, ``reč``) and *ijekavian* (``vrijeme``, ``dijete``, ``riječ``) — and
``lingua-py`` cannot separate them: it labels an ekavian Serbian novel ``hr`` with
confidence 1.0, which would send the book to CLASSLA's *Croatian* models. So after
the vote, the sampled text is checked for those reflex forms
(:func:`resolve_south_slavic`) and an ``hr``/``bs`` vote that is overwhelmingly
ekavian is read as ``sr`` (and the reverse). The tie-breaker only fires on a clear
margin, so a genuine Croatian or Bosnian book keeps its own language.

``lingua-py`` is imported lazily, like the taggers: the rest of the pipeline never
needs it, so a missing wheel produces one actionable line instead of a traceback.
"""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass, field
from typing import Protocol

from ..core.constants import (
    DETECTABLE_LANGUAGES,
    LANGUAGE_SAMPLE_CHAPTERS,
    LANGUAGE_SAMPLE_CHARS,
    LANGUAGE_SAMPLE_SEED,
)
from ..core.errors import MissingDependencyError, ToolError
from ..documents.chapters import Chapter

#: How the language was decided.
METHOD_EXPLICIT = "explicit"
METHOD_LINGUA = "lingua"
METHOD_LINGUA_REFLEX = "lingua+reflex"

_LINGUA_INSTALL_HINT = (
    "language detection needs lingua-py: pip install lingua-language-detector "
    "(or pass --language sr to skip detection)"
)

_WORD = re.compile(r"\w+", re.UNICODE)

#: The same words spelled the ekavian (standard Serbian) way.
_EKAVIAN_FORMS = frozenset(
    {
        "vreme", "vremena", "lepo", "lepa", "lep", "lepota", "lepotu", "mleko",
        "dete", "deteta", "detetu", "deca", "deci", "decom", "reč", "reči", "rečima",
        "cvet", "cveta", "cveće", "pesma", "pesme", "pesama", "mesto", "mesta",
        "mestu", "deo", "dela", "delu", "delova", "smeh", "smeha", "sneg", "snega",
        "pena", "sećanje", "sećanja", "nedelja", "nedelje", "vera", "vere",
        "zvezda", "zvezde", "vek", "veka", "ceo", "cela", "celo", "smeo", "umio",
        "sedeo", "predeo", "dve", "sledeći", "sledeće", "sledeća", "poslednji",
        "poslednja", "poslednje", "posledica", "posledice", "uspeh", "uspeha",
        "uspeo", "uspeti", "uspešno", "najpre", "naučiti",
    }
)
#: …and the ijekavian (Croatian/Bosnian) way. Words that are identical in both
#: (``sreća``, ``žena``, ``dva``) are deliberately absent: they discriminate
#: nothing and would only dilute the ratio.
_IJEKAVIAN_FORMS = frozenset(
    {
        "vrijeme", "vremena", "lijepo", "lijepa", "lijep", "ljepota", "ljepotu",
        "mlijeko", "dijete", "djeteta", "djetetu", "djeca", "djeci", "djecom",
        "riječ", "riječi", "riječima", "cvijet", "cvijeta", "cvijeće", "pjesma",
        "pjesme", "pjesama", "mjesto", "mjesta", "mjestu", "dio", "djela", "dijelu",
        "dijelova", "smijeh", "smijeha", "snijeg", "snijega", "pjena", "sjećanje",
        "sjećanja", "nedjelja", "nedjelje", "vjera", "vjere", "zvijezda", "zvijezde",
        "vijek", "vijeka", "cio", "cijela", "cijelo", "smio", "umio", "sjedio",
        "predio", "dvije", "sljedeći", "sljedeće", "sljedeća", "posljednji",
        "posljednja", "posljednje", "posljedica", "posljedice", "uspjeh", "uspjeha",
        "uspio", "uspjeti", "uspješno", "najprije", "naučiti",
    }
)
#: Which detections the reflex test is allowed to overrule.
_IJEKAVIAN_LANGS = frozenset({"hr", "bs"})
#: Minimum evidence: a handful of forms, and one side at least twice the other.
_REFLEX_MINIMUM = 5
_REFLEX_RATIO = 2.0


class LanguageDetector(Protocol):
    """Anything that can name the language of one snippet."""

    def detect(self, text: str) -> tuple[str, float]:
        """``(iso-639-1 code, confidence in 0..1)``."""


@dataclass(frozen=True)
class LanguageSample:
    """One detected snippet: which chapter it came from and what it looked like."""

    chapter: str
    language: str
    confidence: float
    chars: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LanguageGuess:
    """The book's language, how it was decided, and the evidence behind it."""

    language: str
    confidence: float
    method: str
    samples: list[LanguageSample] = field(default_factory=list)
    #: Set when the South Slavic reflex test overruled the detector.
    note: str = ""
    #: ``{"ekavian": n, "ijekavian": m}`` over the sampled text.
    reflex: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "confidence": round(float(self.confidence), 4),
            "method": self.method,
            "note": self.note,
            "reflex": dict(self.reflex),
            "samples": [s.to_dict() for s in self.samples],
        }


class LinguaDetector:
    """``lingua-py`` restricted to the languages the taggers can actually serve.

    Restricting the candidate set matters: Serbian, Croatian and Bosnian are
    near-identical in writing, so an unrestricted detector happily returns a
    language no backend supports.
    """

    def __init__(self, languages: tuple[str, ...] = DETECTABLE_LANGUAGES) -> None:
        try:  # lazy: the pipeline works without this wheel
            from lingua import Language, LanguageDetectorBuilder
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise MissingDependencyError(_LINGUA_INSTALL_HINT) from exc

        by_code: dict[str, object] = {}
        for member in _lingua_members(Language):
            code = getattr(getattr(member, "iso_code_639_1", None), "name", "").lower()
            if code:
                by_code[code] = member
        wanted = [code for code in languages if code in by_code]
        missing = sorted(set(languages) - set(wanted))
        if missing:
            raise ToolError(f"lingua-py cannot detect {', '.join(missing)}")
        if not wanted:
            raise ToolError("no detectable languages configured")
        self.languages = tuple(wanted)
        self._detector = LanguageDetectorBuilder.from_languages(
            *[by_code[code] for code in wanted]
        ).build()

    def detect(self, text: str) -> tuple[str, float]:
        values = self._detector.compute_language_confidence_values(text)
        if not values:
            return "", 0.0
        best = values[0]
        code = getattr(getattr(best.language, "iso_code_639_1", None), "name", "").lower()
        return code, float(best.value)


def _lingua_members(language_enum) -> list:
    """Every member of lingua's ``Language`` enum, across both major versions.

    lingua 2.x is a PyO3 binding: its enums are **not** iterable and the members
    are listed by ``Language.all()``. lingua 1.x is a plain Python enum. Both
    expose ``iso_code_639_1.name``, which is the only thing used here.
    """
    all_members = getattr(language_enum, "all", None)
    if callable(all_members):
        return list(all_members())
    return list(language_enum)


def sample_chapter_text(
    chapters: list[Chapter],
    *,
    count: int = LANGUAGE_SAMPLE_CHAPTERS,
    chars: int = LANGUAGE_SAMPLE_CHARS,
    seed: int = LANGUAGE_SAMPLE_SEED,
) -> list[tuple[Chapter, str]]:
    """Pick up to ``count`` chapters and cut one random window out of each.

    The sample is deterministic (fixed seed), taken from a random offset past the
    first quarter of each chapter, so a title or a heading cannot dominate the
    snippet and the evidence is not always the same paragraph.
    """
    if not chapters or chars <= 0:
        return []
    ordered = sorted(chapters, key=lambda c: c.filename)
    rng = random.Random(seed)
    picked = rng.sample(ordered, k=min(count, len(ordered)))
    picked.sort(key=lambda c: c.filename)

    samples: list[tuple[Chapter, str]] = []
    for chapter in picked:
        try:
            lines = _read(chapter)
        except OSError:  # pragma: no cover - unreadable chapter is reported upstream
            continue
        text = " ".join(" ".join(lines).split())
        if not text:
            continue
        if len(text) <= chars:
            samples.append((chapter, text))
            continue
        head = min(len(text) // 4, max(len(text) - chars, 0))
        start = rng.randint(head, len(text) - chars)
        samples.append((chapter, text[start : start + chars]))
    return samples


def _read(chapter: Chapter) -> list[str]:
    from ..documents.io import load_lines

    return load_lines(chapter.path)


def count_reflexes(text: str) -> dict[str, int]:
    """Count ekavian and ijekavian word forms in ``text``.

    ``{"ekavian": 300, "ijekavian": 0}`` for an ekavian Serbian chapter. Only the
    forms that actually differ are counted; identical words are ignored.
    """
    words = _WORD.findall(str(text).lower())
    return {
        "ekavian": sum(1 for word in words if word in _EKAVIAN_FORMS),
        "ijekavian": sum(1 for word in words if word in _IJEKAVIAN_FORMS),
    }


def resolve_south_slavic(
    winner: str,
    evidence: str,
    *,
    minimum: int = _REFLEX_MINIMUM,
    ratio: float = _REFLEX_RATIO,
) -> tuple[str, str]:
    """``(language, note)``: overrule an sr/hr vote that the reflexes contradict.

    Serbian (ekavian) and Croatian/Bosnian (ijekavian) are the same language
    spelled two ways, and ``lingua-py`` cannot tell them apart — an ekavian
    Serbian novel comes back as ``hr`` with confidence 1.0. The reflex forms can:
    a book that says ``vreme``/``dete``/``reč`` throughout is Serbian whatever the
    detector says, and one that says ``vrijeme``/``dijete``/``riječ`` is Croatian.

    The test is deliberately timid. It needs at least ``minimum`` forms on one
    side, at least ``ratio`` times the other side, and a winner it is allowed to
    question (``hr``/``bs`` → ``sr``; ``sr`` → ``hr``). Anything less and the
    detector's answer stands, so a short sample or a mixed book is left alone.
    """
    counts = count_reflexes(evidence)
    ekavian, ijekavian = counts["ekavian"], counts["ijekavian"]
    if winner in _IJEKAVIAN_LANGS and ekavian >= minimum and ekavian >= ratio * ijekavian:
        return "sr", (
            f"{winner} vote overruled: {ekavian} ekavian vs {ijekavian} ijekavian "
            "form(s) -> sr"
        )
    if winner == "sr" and ijekavian >= minimum and ijekavian >= ratio * ekavian:
        return "hr", (
            f"sr vote overruled: {ijekavian} ijekavian vs {ekavian} ekavian "
            "form(s) -> hr"
        )
    return winner, ""


def detect_language(
    chapters: list[Chapter],
    *,
    language: str | None = None,
    detector: LanguageDetector | None = None,
    sample_chapters: int = LANGUAGE_SAMPLE_CHAPTERS,
    sample_chars: int = LANGUAGE_SAMPLE_CHARS,
    seed: int = LANGUAGE_SAMPLE_SEED,
    candidates: tuple[str, ...] = DETECTABLE_LANGUAGES,
) -> LanguageGuess:
    """The book's language: explicit when given, otherwise a sampled vote.

    An explicit ``language`` wins over any detector, so ``--language sr`` never
    touches lingua-py. Otherwise every sampled chapter votes, and the language
    with the most votes (ties broken by summed confidence) is the book's — unless
    the South Slavic reflex test contradicts it
    (:func:`resolve_south_slavic`), which is the only thing that can change a vote.
    """
    wanted = (language or "").strip().lower()
    if wanted and wanted != "auto":
        return LanguageGuess(language=wanted, confidence=1.0, method=METHOD_EXPLICIT)

    samples = sample_chapter_text(
        chapters, count=sample_chapters, chars=sample_chars, seed=seed
    )
    if not samples:
        raise ToolError("no chapter text to detect the language from")

    engine = detector or LinguaDetector(candidates)
    votes: dict[str, list[float]] = {}
    evidence: list[LanguageSample] = []
    for chapter, text in samples:
        code, confidence = engine.detect(text)
        code = (code or "").strip().lower()
        evidence.append(
            LanguageSample(
                chapter=chapter.filename,
                language=code or "unknown",
                confidence=float(confidence),
                chars=len(text),
            )
        )
        if code:
            votes.setdefault(code, []).append(float(confidence))

    if not votes:
        raise ToolError("language detection found no known language in the sampled chapters")
    voted = max(votes, key=lambda code: (len(votes[code]), sum(votes[code])))
    sampled_text = " ".join(text for _, text in samples)
    winner, note = resolve_south_slavic(voted, sampled_text)
    return LanguageGuess(
        language=winner,
        confidence=sum(votes[voted]) / len(votes[voted]),
        method=METHOD_LINGUA_REFLEX if note else METHOD_LINGUA,
        samples=evidence,
        note=note,
        reflex=count_reflexes(sampled_text),
    )


__all__ = [
    "METHOD_EXPLICIT",
    "METHOD_LINGUA",
    "METHOD_LINGUA_REFLEX",
    "LanguageDetector",
    "LanguageGuess",
    "LanguageSample",
    "LinguaDetector",
    "count_reflexes",
    "detect_language",
    "resolve_south_slavic",
    "sample_chapter_text",
]
