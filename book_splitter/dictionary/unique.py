"""The second stage-3 filter: one sentence per noun, first mention wins.

The NER gate in :mod:`book_splitter.dictionary.ner` keeps *every* sentence that
carries an entity, and a novel says ``Marko`` in nine hundred of them. A
dictionary needs the noun once, not nine hundred times, so this module drops a
sentence whose nouns have all been met before::

    03-reduced/06-glava-3.md          03-reduced-unique/06-glava-3.md
    Marko je ušao u sobu.          Marko je ušao u sobu.     <- PER "Marko" is new
    Marko je seo za sto.           (dropped: "Marko" alone, already met)
    Ana je ušla za njim.           Ana je ušla za njim.      <- PER "Ana" is new
    Ana je donela knjigu.          (dropped: "Ana" alone, already met)
    Marko i Ana su razgovarali.    (dropped: both already met)

The rule is deliberately "**keep only if at least one noun is new**", not "drop as
soon as any noun repeats". The aggressive version throws away a first mention
that happens to share a sentence with a known name — and with it the only
sentence that would ever have introduced that noun.

Two properties are worth being explicit about, because both are visible in the
output:

* **Book-wide and ordered.** Chapters are visited first to last and the set of
  known nouns is carried across the whole book, so a noun introduced in chapter 3
  suppresses its repeats in every later chapter. Re-running therefore needs the
  order, which the runner keeps by walking ``discover_chapters``' own list.
* **Surface forms, not lemmas.** Two spellings of the same name are two nouns
  (``Marko`` / ``Marku``), and a noun that the tagger typed differently in two
  places is one noun here: the key ignores the entity type, because the point is
  *which words* the dictionary has to pin down, not how a tagger labelled them.

Nothing here needs a tagger: the step reads the entity mentions the gate already
produced, so it costs nothing on top of stage 3 — which is why it runs in the
same pass rather than as a second model call.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from .ner import Entity
from .sentences import EXAMPLE_CHARS

#: Characters stripped from both ends of a mention before it becomes a key:
#: punctuation and quotes the tagger may have swallowed (``"Marko,"``).
_KEY_PUNCTUATION = ".,;:!?…\"'“”‘’()[]{}«»„-–—"
#: Dropped-sentence examples kept per chapter for the trace.
DEFAULT_UNIQUE_EXAMPLES = 5


def noun_key(text: str) -> str:
    """The identity of a mention: case-folded, trimmed, inner space collapsed.

    ``" Marko,"`` and ``"MARKO"`` are the same noun; ``"Novi Sad"`` stays one key
    rather than two. An empty result (a mention that was only punctuation) is not
    a noun at all and is dropped by the callers.
    """
    value = str(text or "").strip().strip(_KEY_PUNCTUATION).strip()
    return " ".join(value.split()).casefold()


def sentence_noun_keys(entities: Iterable[Entity]) -> list[str]:
    """The distinct nouns of one sentence, in the order the tagger reported them."""
    return _clean_keys(getattr(entity, "text", "") for entity in entities)


def _clean_keys(texts: Iterable[str]) -> list[str]:
    """Normalize mentions to distinct noun keys, keeping first-seen order."""
    keys: list[str] = []
    seen: set[str] = set()
    for text in texts:
        key = noun_key(text)
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


@dataclass
class UniqueSelection:
    """What the first-mention filter did to one chapter's kept sentences."""

    kept: list[str] = field(default_factory=list)
    #: Nouns per kept sentence, aligned with ``kept`` — the index a rerun reads.
    kept_keys: list[list[str]] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    dropped_keys: list[list[str]] = field(default_factory=list)
    #: Nouns this chapter introduced, in first-seen order.
    new_keys: list[str] = field(default_factory=list)
    #: Distinct nouns known before and after the chapter (book-wide counters).
    seen_before: int = 0
    seen_after: int = 0
    examples: list[str] = field(default_factory=list)

    @property
    def dropped_duplicate(self) -> int:
        return len(self.dropped)

    @property
    def new_nouns(self) -> int:
        return len(self.new_keys)

    def to_dict(self) -> dict:
        """The compact shape stored in the trace: counts and nouns, not the sentences.

        The kept sentences *are* ``03-reduced-unique/<same-filename>.md`` and the dropped
        ones are the difference against ``03-reduced/<same-filename>.md``, so repeating
        them in the audit trail would add tens of kilobytes of prose to explain what a
        two-folder ``diff`` already shows. What the trace keeps is the part the files
        cannot show: which nouns each sentence brought, and which ones it repeated.
        """
        return {
            "kept": len(self.kept),
            "dropped": len(self.dropped),
            "kept_keys": [list(keys) for keys in self.kept_keys],
            "dropped_keys": [list(keys) for keys in self.dropped_keys],
            "new_keys": list(self.new_keys),
            "seen_before": self.seen_before,
            "seen_after": self.seen_after,
            "dropped_duplicate": self.dropped_duplicate,
            "new_nouns": self.new_nouns,
            "examples": list(self.examples),
        }


def select_unique_sentences(
    sentences: Iterable[str],
    keys: Iterable[Iterable[str]],
    *,
    seen: set[str],
    example_limit: int = DEFAULT_UNIQUE_EXAMPLES,
) -> UniqueSelection:
    """Keep the sentences that introduce a noun, dropping the rest.

    ``seen`` is the book-wide set of known nouns and is **updated in place**, so
    the caller can walk the chapters in order and hand the same set to each. A
    sentence survives only when at least one of its nouns is not in ``seen`` yet;
    a sentence that survives contributes all of its nouns to ``seen``, not just
    the new ones, because a known noun is already there anyway and the book's noun
    set is what later chapters are tested against.

    ``keys`` must be aligned with ``sentences`` — one noun list per sentence —
    which ``strict=True`` enforces rather than silently dropping the tail.
    """
    selection = UniqueSelection(seen_before=len(seen))
    for sentence, sentence_keys in zip(sentences, keys, strict=True):
        nouns = _clean_keys(sentence_keys)
        fresh = [key for key in nouns if key not in seen]
        if fresh:
            selection.kept.append(sentence)
            selection.kept_keys.append(nouns)
            # ``fresh`` holds each new noun once, so extending the book-wide set
            # and recording the chapter's own additions are the same operation.
            seen.update(fresh)
            selection.new_keys.extend(fresh)
            continue
        selection.dropped.append(sentence)
        selection.dropped_keys.append(nouns)
        if len(selection.examples) < max(int(example_limit), 0):
            selection.examples.append(sentence[:EXAMPLE_CHARS])
    selection.seen_after = len(seen)
    return selection


def extend_seen(seen: set[str], keys: Iterable[str]) -> None:
    """Add a reused chapter's recorded nouns to the book-wide set."""
    for key in _clean_keys(keys):
        seen.add(key)


__all__ = [
    "DEFAULT_UNIQUE_EXAMPLES",
    "UniqueSelection",
    "extend_seen",
    "noun_key",
    "select_unique_sentences",
    "sentence_noun_keys",
]
