"""The sentence funnel and one chapter's reduction."""

from __future__ import annotations

import time
from pathlib import Path

from book_translator.core.artifacts import sha256_text, write_text
from book_translator.core.constants import (
    DEFAULT_KEEP_ENTITY_TYPES,
    DEFAULT_MIN_SENTENCE_WORDS,
    DEFAULT_REQUIRE_UPPERCASE,
)
from book_translator.documents.chapters import Chapter
from book_translator.documents.io import load_lines

from .models import ChapterReduction, Clock, TextReduction
from .ner import NerBackend, gate_entities, normalize_type
from .sentences import EXAMPLE_CHARS, prefilter_sentences, split_sentences
from .unique import select_unique_sentences, sentence_noun_keys

#: Dropped-sentence examples kept per reason, per chapter, for the trace.
DEFAULT_EXAMPLE_LIMIT = 5
#: Entity mentions recorded per chapter, for the trace. Counts stay exact.
MENTION_LIMIT = 200


def reduce_text(
    text: str,
    *,
    backend: NerBackend,
    min_words: int = DEFAULT_MIN_SENTENCE_WORDS,
    require_uppercase: bool = DEFAULT_REQUIRE_UPPERCASE,
    keep_types: tuple[str, ...] = DEFAULT_KEEP_ENTITY_TYPES,
    drop_temporal_only: bool = False,
    example_limit: int = DEFAULT_EXAMPLE_LIMIT,
    clock: Clock = time.perf_counter,
) -> TextReduction:
    """Run the whole funnel over one chapter's text.

    Only the survivors of the two cheap filters reach the tagger, which is what
    keeps the expensive step small. The seconds spent *inside* the tagger are
    accumulated separately (``ner_seconds``), so the cost of the NER stage can be
    read on its own — the sentence splitting and filtering are free by comparison.
    """
    sentences = split_sentences(text)
    pre = prefilter_sentences(
        sentences, min_words=min_words, require_uppercase=require_uppercase
    )
    reduction = TextReduction(
        source_sentences=len(sentences),
        dropped_short=len(pre.dropped_short),
        dropped_no_uppercase=len(pre.dropped_no_uppercase),
        examples={
            "short": _examples(pre.dropped_short, example_limit),
            "no_uppercase": _examples(pre.dropped_no_uppercase, example_limit),
            "no_entity": [],
            "temporal_only": [],
        },
    )

    for sentence in pre.kept:
        started = clock()
        entities = backend.entities(sentence)
        reduction.ner_seconds += clock() - started
        reduction.ner_calls += 1
        decision = gate_entities(
            entities,
            keep_types=keep_types,
            drop_temporal_only=drop_temporal_only,
        )
        if decision.keep:
            reduction.kept.append(sentence)
            reduction.kept_keys.append(sentence_noun_keys(decision.entities))
            for entity in decision.entities:
                entity_type = normalize_type(entity.type)
                reduction.entities[entity_type] = (
                    reduction.entities.get(entity_type, 0) + 1
                )
                if len(reduction.entity_mentions) < MENTION_LIMIT:
                    reduction.entity_mentions.append(
                        {
                            "sentence": len(reduction.kept),
                            "text": entity.text,
                            "type": entity_type,
                        }
                    )
            continue
        if decision.reason == "temporal_only":
            reduction.dropped_temporal += 1
            _remember(reduction.examples["temporal_only"], sentence, example_limit)
        else:
            reduction.dropped_no_entity += 1
            _remember(reduction.examples["no_entity"], sentence, example_limit)

    return reduction


def _examples(sentences: list[str], limit: int) -> list[str]:
    return [s[:EXAMPLE_CHARS] for s in sentences[: max(int(limit), 0)]]


def _remember(examples: list[str], sentence: str, limit: int) -> None:
    if len(examples) < max(int(limit), 0):
        examples.append(sentence[:EXAMPLE_CHARS])


def reduced_payload(sentences: list[str]) -> str:
    """A reduced chapter as it is written: one raw sentence per line.

    Both outputs use this shape — ``03-reduced/`` and ``03-reduced-unique/`` differ
    in *which* sentences they hold, never in how a sentence is written.
    """
    return "".join(f"{sentence.rstrip()}\n" for sentence in sentences)


def reduce_chapter(
    chapter: Chapter,
    out_dir: Path,
    *,
    unique_out_dir: Path,
    backend: NerBackend,
    min_words: int,
    require_uppercase: bool,
    keep_types: tuple[str, ...],
    drop_temporal_only: bool,
    example_limit: int,
    save_reduced: bool,
    save_unique: bool,
    unique_example_limit: int,
    seen: set[str],
    dry_run: bool,
    clock: Clock = time.perf_counter,
) -> ChapterReduction:
    """Reduce one chapter and (unless told not to) write its two output files.

    ``seen`` is the book-wide noun set, updated in place by the first-mention step
    so the next chapter continues from where this one stopped.
    """
    chapter_started = clock()
    text = "".join(load_lines(chapter.path))
    reduction = reduce_text(
        text,
        backend=backend,
        min_words=min_words,
        require_uppercase=require_uppercase,
        keep_types=keep_types,
        drop_temporal_only=drop_temporal_only,
        example_limit=example_limit,
        clock=clock,
    )
    record = ChapterReduction(
        chapter_id=chapter.chapter_id,
        filename=chapter.filename,
        source_path=chapter.path,
        title=chapter.title,
        start_line=chapter.start_line,
        end_line=chapter.end_line,
        reduction=reduction,
        output_lines=len(reduction.kept),
        ner_seconds=reduction.ner_seconds,
        ner_calls=reduction.ner_calls,
        noun_keys=reduction.noun_keys,
    )
    record.log.append(
        f"{reduction.source_sentences} sentence(s) -> {len(reduction.kept)} kept "
        f"(short={reduction.dropped_short}, no-uppercase={reduction.dropped_no_uppercase}, "
        f"no-entity={reduction.dropped_no_entity}, temporal={reduction.dropped_temporal})"
    )
    if save_reduced and not dry_run:
        path = Path(out_dir) / chapter.filename
        payload = reduced_payload(reduction.kept)
        write_text(path, payload)
        record.output_path = str(path)
        record.output_bytes = len(payload.encode("utf-8"))
        record.sha256 = sha256_text(payload)
    elif save_reduced:
        record.output_path = str(Path(out_dir) / chapter.filename)

    if save_unique:
        selection = select_unique_sentences(
            reduction.kept,
            reduction.kept_keys,
            seen=seen,
            example_limit=unique_example_limit,
        )
        record.unique = selection
        record.unique_lines = len(selection.kept)
        record.log.append(
            f"{len(reduction.kept)} kept -> {len(selection.kept)} first mention(s) "
            f"(duplicate-noun={selection.dropped_duplicate}, "
            f"new-nouns={selection.new_nouns}, nouns-after={selection.seen_after})"
        )
        if not dry_run:
            path = Path(unique_out_dir) / chapter.filename
            payload = reduced_payload(selection.kept)
            write_text(path, payload)
            record.unique_path = str(path)
            record.unique_bytes = len(payload.encode("utf-8"))
            record.unique_sha256 = sha256_text(payload)
        else:
            record.unique_path = str(Path(unique_out_dir) / chapter.filename)

    record.elapsed_seconds = clock() - chapter_started
    return record


__all__ = [
    "DEFAULT_EXAMPLE_LIMIT",
    "MENTION_LIMIT",
    "reduce_chapter",
    "reduce_text",
    "reduced_payload",
]
