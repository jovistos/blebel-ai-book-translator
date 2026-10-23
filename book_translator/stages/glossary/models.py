"""The shape of the model's answer: one glossary entry, one chapter's extraction.

The glossary is built by asking a model to read the first-mention sentences of a
chapter and to name the entities in them. That answer has to be *machine-readable*
before it can be trusted with the consistency of a whole translation, so it is
validated with Pydantic rather than parsed by hand:

``GlossaryEntry``
    one entity — its source ``term``, which ``category`` it belongs to, and the
    locked ``translation`` in the target language;
``ChapterExtraction``
    the object one chapter's call returns: ``{"entries": [...]}``. Object-rooted
    on purpose — an LLM asked for a bare JSON array forgets the wrapper, and a
    schema with one named field is the shape every provider's structured-output
    mode accepts.

Entities are the subject of the stage, not vocabulary: a name, a place, a guild,
an invented device, a recurring term of art. The categories are the five the
translation pipeline actually needs to keep apart, because each one is rendered
differently — a person keeps their name shape, a place gets the established
target-language exonym, an organisation is translated or left alone depending on
whether it is real, and a term of art is *defined* once and then reused verbatim.

The validators are deliberately forgiving about spelling and strict about
emptiness: a model that answers ``"Characters & Aliases"`` or ``"PERSON"`` is
naming the right category, so it is normalised instead of rejected, while an
entry with no term or no translation is a real failure and is rejected, which is
what makes the caller retry.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: The five entity families the glossary keeps apart, in the order the prompts
#: present them. Values are what lands in ``glossary.json``.
CHARACTER = "character"
LOCATION = "location"
ORGANIZATION = "organization"
ARTIFACT = "artifact"
TERMINOLOGY = "terminology"


class TermCategory(StrEnum):
    """Which kind of entity an entry pins down."""

    CHARACTER = CHARACTER
    LOCATION = LOCATION
    ORGANIZATION = ORGANIZATION
    ARTIFACT = ARTIFACT
    TERMINOLOGY = TERMINOLOGY


#: Every label a model may plausibly answer with, mapped onto the five categories.
#: Kept generous on purpose: the category is a filing decision, and a retry spent
#: arguing about the word "person" is a retry not spent on the book.
CATEGORY_ALIASES: dict[str, TermCategory] = {}


def _register(category: TermCategory, *labels: str) -> None:
    for label in labels:
        CATEGORY_ALIASES[label] = category


_register(
    TermCategory.CHARACTER,
    "character",
    "characters",
    "characters & aliases",
    "character & alias",
    "person",
    "people",
    "persons",
    "name",
    "names",
    "proper name",
    "proper names",
    "person name",
    "person names",
    "per",
    "alias",
    "aliases",
    "title",
    "titles",
    "protagonist",
)
_register(
    TermCategory.LOCATION,
    "location",
    "locations",
    "geographic location",
    "geographic locations",
    "place",
    "places",
    "place name",
    "place names",
    "geo",
    "geography",
    "setting",
    "settings",
    "loc",
    "city",
    "cities",
    "country",
    "region",
    "realm",
    "river",
    "building",
    "buildings",
)
_register(
    TermCategory.ORGANIZATION,
    "organization",
    "organizations",
    "organisation",
    "organisations",
    "organizations & factions",
    "org",
    "orgs",
    "faction",
    "factions",
    "group",
    "groups",
    "guild",
    "guilds",
    "company",
    "companies",
    "military unit",
    "military units",
    "family",
    "families",
    "institution",
    "institutions",
)
_register(
    TermCategory.ARTIFACT,
    "artifact",
    "artifacts",
    "artefact",
    "artefacts",
    "artifacts & objects",
    "object",
    "objects",
    "item",
    "items",
    "vehicle",
    "vehicles",
    "device",
    "devices",
    "weapon",
    "weapons",
    "misc",
)
_register(
    TermCategory.TERMINOLOGY,
    "terminology",
    "domain-specific terminology",
    "domain specific terminology",
    "domain terminology",
    "term",
    "terms",
    "concept",
    "concepts",
    "technical",
    "technical term",
    "technical terms",
    "unit",
    "units",
    "measurement",
    "measurements",
    "measurement unit",
    "measurement units",
    "historical role",
    "historical roles",
    "procedural term",
    "procedural terms",
    "jargon",
    "slang",
    "idiom",
    "idioms",
)


def normalize_category(value: Any) -> TermCategory:
    """Map a model's label onto a category, falling back to terminology.

    A label the table does not know is not worth a retry: the entity was found,
    and ``terminology`` is the honest bucket for "something that needs a locked
    translation but is not a person, place, group or object". ``"Characters &
    Aliases"``, ``"PERSON"`` and ``"place names"`` all resolve here, since the
    models this stage runs against are not consistent about their own labels.
    """
    if isinstance(value, TermCategory):
        return value
    label = " ".join(str(value or "").strip().lower().replace("_", " ").split())
    if not label:
        return TermCategory.TERMINOLOGY
    if label in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[label]
    # "character / person", "location (city)" — the first known label wins.
    for separator in ("/", "|", ",", "(", " - ", "—"):
        if separator in label:
            head = label.split(separator, 1)[0].strip()
            if head in CATEGORY_ALIASES:
                return CATEGORY_ALIASES[head]
    for alias, category in CATEGORY_ALIASES.items():
        if alias in label:
            return category
    return TermCategory.TERMINOLOGY


def entry_key(term: str) -> str:
    """The identity of an entry in the dictionary: case-folded, space-collapsed.

    ``"New York"`` and ``"new york"`` are one entity, and so are ``"Potter "``
    and ``"Potter"``. Case-folding is right here — unlike an inflection — because
    the dictionary is keyed on the *name*, and a model that capitalises it
    differently in two chapters is still talking about the same person.
    """
    return " ".join(str(term or "").split()).casefold()


def _as_text(value: Any) -> str:
    """A required string field: anything scalar is coerced, containers rejected."""
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, int | float | bool):
        return str(value)
    raise ValueError("expected a string")


class GlossaryEntry(BaseModel):
    """One entity, with the rendering a translation has to reuse everywhere.

    Three fields, and the third is the point: ``term`` names the entity in the
    book, ``category`` files it, and ``translation`` is the *locked* target form,
    given once, in its base shape (nominative singular for an inflected
    language). That is the whole entry — the dictionary is re-sent with every
    chapter, so anything else in it is paid for once per chapter of the book.
    """

    model_config = ConfigDict(extra="ignore")

    #: The entity as the source text names it, in its canonical source form.
    term: str = Field(min_length=1, max_length=200)
    category: TermCategory = TermCategory.TERMINOLOGY
    #: The locked rendering in the target language, base form.
    translation: str = Field(min_length=1, max_length=200)

    @field_validator("term", "translation", mode="before")
    @classmethod
    def _text(cls, value: Any) -> str:
        return _as_text(value)

    @field_validator("category", mode="before")
    @classmethod
    def _category(cls, value: Any) -> TermCategory:
        return normalize_category(value)

    @property
    def key(self) -> str:
        return entry_key(self.term)

    def to_dict(self) -> dict[str, Any]:
        return {
            "term": self.term,
            "category": self.category.value,
            "translation": self.translation,
        }


class ChapterExtraction(BaseModel):
    """What one chapter's call must return: an object with an ``entries`` list.

    An empty list is a valid, expected answer — a chapter can introduce nothing
    new — so the field defaults to ``[]`` rather than being required. The
    ``before`` validator also accepts a bare JSON array (``[ {...}, {...} ]``) and
    a wrapper the model may have invented (``{"glossary": [...]}``), because the
    same prompt runs against several providers and none of them is consistent
    about the envelope. Only when no list can be found at all is the answer
    invalid, which is what makes the caller retry.
    """

    model_config = ConfigDict(extra="ignore")

    entries: list[GlossaryEntry] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _unwrap(cls, value: Any) -> Any:
        """Accept the three envelopes a provider may return the list in.

        The prompt asks for ``{"entries": [...]}``, and that is what a native
        structured-output mode produces. A bare array and an invented wrapper
        (``{"glossary": [...]}``) are the same answer with a different envelope,
        and rejecting them would spend a retry on punctuation rather than on the
        book. ``None`` means "nothing new", which is a valid chapter answer.
        """
        if value is None:
            return {"entries": []}
        if isinstance(value, list):
            return {"entries": value}
        if isinstance(value, dict) and "entries" not in value:
            for key in ("glossary", "terms", "items", "results", "data"):
                if key in value:
                    return {"entries": value[key]}
        return value

    @field_validator("entries", mode="before")
    @classmethod
    def _entries(cls, value: Any) -> Any:
        return [] if value is None else value

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [entry.to_dict() for entry in self.entries]}


__all__ = [
    "ARTIFACT",
    "CATEGORY_ALIASES",
    "CHARACTER",
    "LOCATION",
    "ORGANIZATION",
    "TERMINOLOGY",
    "ChapterExtraction",
    "GlossaryEntry",
    "TermCategory",
    "entry_key",
    "normalize_category",
]
