"""The ordered chunk translation loop."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

from book_translator.core.errors import ConfigError, TranslationError
from book_translator.core.llm import TokenUsage
from book_translator.documents.chunks import chunk_glossary_filename
from book_translator.documents.images import (
    IMAGE_TOKEN_RE,
    protect_images,
    restore_images,
    strip_images,
)

from .plan import TranslationPlan
from .prompts import build_chunk_prompt
from .records import ChunkTranslation
from .support import (
    _carried_records,
    _context_tail,
    _debug_payload,
    _finish_record,
    _last_reused_translation,
    _load_existing_translation,
    _previous_usage,
    _snapshot_manifest,
    _totals,
    _update_debug_answer,
    _write_prompt_files,
    _write_translation_file,
    load_chunk_glossary_rows,
    preserve_trailing_newlines,
)


@dataclass
class LoopOutcome:
    records: list[ChunkTranslation] = field(default_factory=list)
    totals: TokenUsage = field(default_factory=TokenUsage)
    errors: list[str] = field(default_factory=list)
    settings_payload: dict = field(default_factory=dict)


def translate_chunks(plan: TranslationPlan) -> LoopOutcome:
    if plan.pending and not plan.dry_run and plan.save:
        plan.out_dir.mkdir(parents=True, exist_ok=True)
        if plan.write_prompts:
            plan.prompt_dir.mkdir(parents=True, exist_ok=True)

    records: list[ChunkTranslation] = []
    totals = (
        _previous_usage(plan.previous_manifest)
        if plan.resumed
        else TokenUsage()
    )
    errors: list[str] = []
    previous_translation = _last_reused_translation(
        plan.inputs, plan.reused, plan.out_dir
    )
    completed = 0

    for position, (chunk, is_reused) in enumerate(
        zip(plan.inputs, plan.reused, strict=True), start=1
    ):
        if plan.scope == "chunk":
            rows = load_chunk_glossary_rows(
                plan.resolved_chunk_glossary / chunk_glossary_filename(chunk.filename)
            )
            glossary_missing = rows is None
            terms = rows or []
        else:
            terms = list(plan.book_rows)
            glossary_missing = False

        # Pictures are not continuity context: drop them so the model cannot
        # copy an image from the previous chunk into this one.
        context_text = _context_tail(
            strip_images(previous_translation), plan.context_size
        )
        record = ChunkTranslation(
            chunk=chunk,
            terms=terms,
            glossary_missing=glossary_missing,
            context_chars=len(context_text),
            source_chars=len(chunk.text),
            resume_key=plan.resume_keys[chunk.filename],
        )
        # The model never sees a real image path: each reference becomes a
        # [[IMG_n]] marker, and the marker is restored before the file is written.
        protected_text, image_map = protect_images(chunk.text)
        prose_text = IMAGE_TOKEN_RE.sub("", protected_text)
        records.append(record)
        if glossary_missing:
            record.log.append(
                f"glossary: no file for this chunk in {plan.resolved_chunk_glossary}; "
                "translating without it"
            )

        if is_reused:
            record.reused = True
            _load_existing_translation(record, plan.out_dir / chunk.filename)
            previous_translation = record.translation or previous_translation
            completed += 1
            _snapshot_manifest(
                plan.out_dir,
                records,
                source_language=plan.source_language,
                target_language=plan.target_language,
                tokens=totals,
                enabled=plan.write_manifest and plan.save and not plan.dry_run,
                extra_files=_carried_records(
                    plan.carried_from,
                    {record.chunk.filename for record in records},
                    plan.out_dir,
                ),
            )
            plan.context.emit(record.to_dict(), completed, len(plan.inputs))
            continue

        image_only = bool(image_map) and not any(ch.isalnum() for ch in prose_text)
        if not chunk.text.strip() or image_only:
            record.translation = chunk.text
            _finish_record(record, record.translation)
            if not plan.dry_run and plan.save:
                _write_translation_file(plan.out_dir, record)
            record.log.append(
                "image-only chunk — passed through unchanged to keep its picture(s)"
                if image_only
                else "empty chunk — nothing to translate"
            )
            completed += 1
            _snapshot_manifest(
                plan.out_dir,
                records,
                source_language=plan.source_language,
                target_language=plan.target_language,
                tokens=totals,
                enabled=plan.write_manifest and plan.save and not plan.dry_run,
                extra_files=_carried_records(
                    plan.carried_from,
                    {record.chunk.filename for record in records},
                    plan.out_dir,
                ),
            )
            plan.context.emit(record.to_dict(), completed, len(plan.inputs))
            continue

        user_prompt = build_chunk_prompt(
            chunk_index=position,
            total_chunks=len(plan.inputs),
            chunk_filename=chunk.filename,
            chapter_filename=chunk.chapter_filename,
            source_text=protected_text,
            source_language=plan.source_language,
            target_language=plan.target_language,
            terms=terms,
            context_text=context_text,
            book_title=plan.title,
        )
        record.debug = _debug_payload(
            record,
            system_prompt=plan.system_prompt,
            user_prompt=user_prompt,
            context_text=context_text,
            source_language=plan.source_language,
            target_language=plan.target_language,
            scope=plan.scope,
            chunk_glossary_path=str(plan.resolved_chunk_glossary),
            glossary_path=str(plan.resolved_glossary),
            provider=str(getattr(plan.settings, "provider", "") or ""),
            model=str(getattr(plan.settings, "model", "") or ""),
            translator=plan.translator_name,
        )
        record.debug["images"] = {
            "count": len(image_map),
            "markers": dict(image_map),
        }
        if not plan.dry_run and plan.save and plan.write_prompts:
            _write_prompt_files(plan.prompt_dir, record)

        if plan.chunk_start is not None:
            with contextlib.suppress(Exception):
                plan.chunk_start(
                    {
                        "index": position,
                        "total": len(plan.inputs),
                        "done": completed,
                        "filename": chunk.filename,
                        "chapter_filename": chunk.chapter_filename,
                        "tokens": totals.to_dict(),
                    }
                )

        call_started = plan.clock()
        try:
            if plan.translator is None:  # pragma: no cover - guarded above
                raise ConfigError("no translator was provided for the translation stage")
            outcome = plan.translator.translate(
                system_prompt=plan.system_prompt,
                user_prompt=user_prompt,
                source_text=protected_text,
            )
            record.attempts = int(outcome.attempts)
            record.method = outcome.method
            record.usage = outcome.usage
            record.raw_answer = outcome.raw_text
            totals += outcome.usage
            restored, missing, duplicated = restore_images(outcome.text, image_map)
            if missing or duplicated:
                # The real translator's own validation retries a dropped marker;
                # this guard also covers injected translators and is the last
                # line of defence before a picture disappears from the book.
                parts: list[str] = []
                if missing:
                    parts.append(f"dropped {', '.join(missing)}")
                if duplicated:
                    parts.append(f"repeated {', '.join(duplicated)}")
                raise TranslationError(
                    "the translation did not preserve every image marker "
                    f"({'; '.join(parts)})"
                )
            record.translation = preserve_trailing_newlines(restored, chunk.text)
            if outcome.error:
                record.log.append(f"retried after: {outcome.error}")
            if outcome.recovered:
                record.log.append("recovered the translation from malformed JSON")
            _finish_record(record, record.translation)
            if not plan.dry_run and plan.save:
                _write_translation_file(plan.out_dir, record)
                if plan.write_prompts:
                    _write_prompt_files(plan.prompt_dir, record)
            record.log.append(
                f"translated {record.source_chars:,} -> {record.translation_chars:,} chars "
                f"with {int(outcome.usage.total_tokens):,} token(s), "
                f"{outcome.attempts} attempt(s)"
            )
            previous_translation = record.translation
        except Exception as exc:  # one bad chunk must not stop the book
            record.ok = False
            record.error = f"{type(exc).__name__}: {exc}"
            # A failed chunk still spent its attempts. The translator carries the
            # usage on the exception so the run's token total stays honest.
            failed_usage = getattr(exc, "usage", None)
            if isinstance(failed_usage, TokenUsage):
                record.usage = failed_usage
                totals += failed_usage
            record.attempts = int(getattr(exc, "attempts", 0) or 0)
            errors.append(f"{chunk.filename}: {record.error}")
            record.log.append(f"translation failed: {record.error}")
            if not plan.dry_run and plan.save:
                _update_debug_answer(record)
                if plan.write_prompts:
                    _write_prompt_files(plan.prompt_dir, record)
        record.elapsed_seconds = plan.clock() - call_started
        completed += 1
        _snapshot_manifest(
            plan.out_dir,
            records,
            source_language=plan.source_language,
            target_language=plan.target_language,
            tokens=totals,
            enabled=plan.write_manifest and plan.save and not plan.dry_run,
            extra_files=_carried_records(
                plan.carried_from,
                {record.chunk.filename for record in records},
                plan.out_dir,
            ),
        )
        plan.context.emit(record.to_dict(), completed, len(plan.inputs))

    settings_payload = dict(plan.settings_payload)
    # A provider that rejected the schema mid-run leaves the translator in JSON
    # mode; recording it here is what lets the summary say so.
    settings_payload["json_only"] = bool(
        getattr(plan.translator, "json_only", False)
    )
    return LoopOutcome(
        records=records,
        totals=totals,
        errors=errors,
        settings_payload=settings_payload,
    )


def totals_payload(outcome: LoopOutcome) -> dict[str, int]:
    return _totals(outcome.records)


__all__ = ["LoopOutcome", "translate_chunks", "totals_payload"]
