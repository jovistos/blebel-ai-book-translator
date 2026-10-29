"""Validation and final formatting for the matter localization stage.

The agent produces prose, but the book needs a *contract*. Two deterministic
rules stand between the model and the translated edition:

1. **A valid plan.** The localized title is required, the front matter must name
   the translator (``BleBel``), and every omitted phrase is a real source phrase
   the final text is checked against.
2. **A valid translation.** An omitted chunk is allowed and must say why; a
   non-omitted chunk must be non-empty, must not be the source unchanged, and
   must not contain any phrase the plan said to omit.

The formatting helpers are deliberately small. They guarantee the two things a
book cannot get wrong — the localized title and the translator credit appear on
the first front-matter chunk, with clean Markdown spacing, and above them a cover
picture stays the first block of the book — while leaving the agent's wording and
layout alone.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from book_translator.core.constants import (
    DEFAULT_MATTER_CREDIT,
    MATTER_TRANSLATOR_NAME,
    MIN_MATTER_LENGTH_RATIO,
)
from book_translator.documents.images import image_refs, is_image_line
from book_translator.documents.text import preserve_trailing_newlines
from book_translator.stages.matter.images import is_translated_image_match
from book_translator.stages.translation.translator import clean_translation

#: A heading line, with its level.
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
#: A horizontal rule, which may separate the blocks of a title page.
_RULE_RE = re.compile(r"^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$")
#: A short line ending in a full stop is prose, not a title to promote.
_SENTENCE_END_RE = re.compile(r"[.…]\s*$")
#: The credit must contain this; only the surrounding sentence is localized.
_CREDIT_NAME = MATTER_TRANSLATOR_NAME


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _phrase_list(value: Any) -> list[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        phrase = _text(item)
        if not phrase:
            continue
        key = phrase.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(phrase)
    return out


def section_plan(plan: dict[str, Any], filename: str) -> dict[str, Any]:
    """The per-section overrides for one matter file (an empty dict when none)."""
    sections = plan.get("sections") if isinstance(plan, dict) else None
    if not isinstance(sections, dict):
        return {}
    wanted = str(filename or "")
    for name, value in sections.items():
        if str(name) == wanted and isinstance(value, dict):
            return value
    return {}


def omitted_phrases(plan: dict[str, Any], filename: str = "") -> list[str]:
    """Every source phrase the plan says must not appear in the translation.

    The book-level list applies to the whole front/back matter; a section may add
    its own. The comparison is case-insensitive, because a model that drops a
    sentence might still quote a fragment of it.
    """
    phrases = _phrase_list((plan or {}).get("omit"))
    section = section_plan(plan or {}, filename)
    phrases.extend(_phrase_list(section.get("omit")))
    seen: set[str] = set()
    out: list[str] = []
    for phrase in phrases:
        key = phrase.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(phrase)
    return out


def clean_markdown(text: str) -> str:
    """Prose from the agent as book-ready Markdown.

    Strips a wrapping code fence and a conversational preamble (the same cleaner
    stage 7 uses), normalizes line endings, collapses runs of blank lines to one,
    and makes sure a heading is followed by a blank line.
    """
    value = clean_translation(str(text or "")).replace("\r\n", "\n").replace("\r", "\n")
    lines = value.split("\n")
    out: list[str] = []
    for line in lines:
        clean = line.rstrip()
        if not clean:
            if out and out[-1] == "":
                continue
            out.append("")
            continue
        out.append(clean)
    # A heading needs air after it; a heading directly followed by prose reads
    # as an underline in several Markdown renderers.
    spaced: list[str] = []
    for index, line in enumerate(out):
        spaced.append(line)
        if _HEADING_RE.match(line):
            nxt = out[index + 1] if index + 1 < len(out) else ""
            if nxt.strip():
                spaced.append("")
    while spaced and not spaced[0].strip():
        spaced.pop(0)
    while spaced and not spaced[-1].strip():
        spaced.pop()
    return "\n".join(spaced)


def _title_and_credit(plan: dict[str, Any]) -> tuple[str, str]:
    title = _text(plan.get("translated_title")) or _text(plan.get("title"))
    credit = _text(plan.get("translation_credit"))
    if not credit or _CREDIT_NAME not in credit:
        credit = _text(plan.get("translated_by"))
    if not credit or _CREDIT_NAME not in credit:
        credit = DEFAULT_MATTER_CREDIT
    elif _CREDIT_NAME not in credit:
        credit = f"{credit} {_CREDIT_NAME}"
    return title, credit


def _insert_credit(lines: list[str], credit: str) -> list[str]:
    """Put the credit under the title, or keep the model's localized line."""
    if any(_CREDIT_NAME in line for line in lines):
        return lines
    credit_line = f"*{credit.strip('*').strip()}*"
    # Find the opening heading. The title is line 0 after `_ensure_front_title`,
    # so the credit goes directly below it with one blank line between.
    heading = next(
        (index for index, line in enumerate(lines) if _HEADING_RE.match(line)),
        None,
    )
    insert_at = 2 if heading == 0 and len(lines) > 1 and not lines[1].strip() else 1
    if heading not in (0, None):
        insert_at = heading + 1
        if insert_at < len(lines) and not lines[insert_at].strip():
            insert_at += 1
    block = [credit_line]
    if insert_at < len(lines) and lines[insert_at].strip():
        block.append("")
    return lines[:insert_at] + block + lines[insert_at:]


def _cover_and_body(lines: list[str]) -> tuple[list[str], list[str]]:
    """Split the cover picture from the opening title page, cover first.

    The translated edition opens with its cover. A model sometimes writes the
    title page first and the cover picture directly below it; lifting the picture
    keeps the cover first without dropping or rewriting it. Only the opening
    title block is inspected — a picture that follows prose is content, not the
    cover, and stays where the model put it.
    """
    first = next((index for index, line in enumerate(lines) if line.strip()), None)
    if first is None:
        return [], lines
    if is_image_line(lines[first]):
        picture: int | None = first
    else:
        picture = next(
            (index for index in range(first + 1, len(lines)) if is_image_line(lines[index])),
            None,
        )
    if picture is None:
        return [], lines
    opening = [line for line in lines[first:picture] if line.strip()]
    # A picture under prose is an illustration, not the cover. Above the picture
    # there is either the model's title page (which carries the credit) or the
    # source cover caption; both belong below the cover in the translated book.
    title_page = bool(opening) and all(
        _HEADING_RE.match(line)
        or _CREDIT_NAME in line
        or _RULE_RE.match(line.strip())
        or (len(line.strip()) <= 160 and not _SENTENCE_END_RE.search(line))
        for line in opening
    )
    caption = (
        len(opening) == 1
        and len(opening[0].strip()) <= 160
        and not _SENTENCE_END_RE.search(opening[0])
    )
    if opening:
        # A picture appearing after the title is only lifted above it if it
        # explicitly identifies as a cover picture (by filename or alt text).
        # Normal publisher logos, imprints, or illustrations stay where they are.
        if "cover" not in lines[picture].lower():
            return [], lines
        if not (title_page and (any(_CREDIT_NAME in line for line in opening) or caption)):
            return [], lines
    cover = [lines[picture]]
    end = picture + 1
    while end < len(lines) and (not lines[end].strip() or is_image_line(lines[end])):
        if is_image_line(lines[end]):
            cover.append(lines[end])
        end += 1
    return cover, lines[:picture] + lines[end:]


def _ensure_front_title(text: str, plan: dict[str, Any]) -> str:
    """The first front-matter chunk opens with the cover, then title and credit.

    The localized title and the translator credit are guaranteed on the first
    front-matter chunk, but they never push the cover picture down: the cover
    stays the first block of the translated book.
    """
    title, credit = _title_and_credit(plan)
    cover, lines = _cover_and_body(text.split("\n") if text else [])
    first = next((index for index, line in enumerate(lines) if line.strip()), None)
    if title:
        heading = f"# {title}"
        if first is None:
            lines = [heading, "", f"*{credit}*"]
            first = 0
        else:
            match = _HEADING_RE.match(lines[first])
            if match:
                lines[first] = heading
            else:
                candidate = lines[first].strip()
                # A short standalone line at the very top is the old title page;
                # promote it. A sentence is not, so the heading is inserted.
                if len(candidate) <= 160 and not _SENTENCE_END_RE.search(candidate):
                    lines[first] = heading
                else:
                    lines[first:first] = [heading, ""]
                    first += 2
    elif first is None:
        lines = []
    lines = _insert_credit(lines, credit)
    edition_note = _text(plan.get("edition_note"))
    if edition_note and edition_note.casefold() not in "\n".join(lines).casefold():
        credit_index = next(
            (i for i, line in enumerate(lines) if _CREDIT_NAME in line), None
        )
        if credit_index is None:
            credit_index = 0
        lines.insert(credit_index + 1, "")
        lines.insert(credit_index + 2, f"*{edition_note.strip('*').strip()}*")
        if credit_index + 3 < len(lines) and lines[credit_index + 3].strip():
            lines.insert(credit_index + 3, "")
    body = clean_markdown("\n".join(lines))
    if cover:
        joined = "\n\n".join(cover)
        return clean_markdown(f"{joined}\n\n{body}" if body else joined)
    return body


def _ensure_back_title(text: str, title: str) -> str:
    """The first chunk of a back-matter section gets its localized heading."""
    clean = clean_markdown(text)
    if not title:
        return clean
    lines = clean.split("\n") if clean else []
    first_index = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first_index is None:
        return clean_markdown(f"# {title}\n\n{clean}")
    first_line = lines[first_index]
    match = _HEADING_RE.match(first_line)
    if match:
        # The plan's section title is binding: an agent that headed the section
        # with the book title (or the source title) is corrected here.
        if title.casefold() in match.group(2).casefold():
            return clean
        lines[first_index] = f"# {title}"
        return clean_markdown("\n".join(lines))
    candidate = first_line.strip()
    if candidate and len(candidate) <= 160 and not _SENTENCE_END_RE.search(candidate):
        lines[first_index] = f"# {title}"
        return clean_markdown("\n".join(lines))
    return clean_markdown(f"# {title}\n\n{clean}")


def finalize_translation(
    *,
    section_kind: str,
    section_title: str,
    plan: dict[str, Any],
    translation: str,
    omitted: bool = False,
    is_first_front_chunk: bool = False,
    is_first_section_chunk: bool = False,
) -> str:
    """The text that may be written for one matter chunk.

    An omitted chunk becomes an empty file, except the very first front-matter
    chunk: the translated edition still needs its title page and translator
    credit, even when the original imprint page it happened to contain is gone.
    """
    if omitted:
        if is_first_front_chunk:
            return _ensure_front_title("", plan)
        return ""
    text = clean_markdown(translation)
    if section_kind == "front" and is_first_front_chunk:
        text = _ensure_front_title(text, plan)
    elif section_kind == "back" and is_first_section_chunk:
        text = _ensure_back_title(text, section_title)
    return text


#: Localized headings for the common generated back-matter section. Only used
#: when the planning model could not run (no provider, no network).
_SECTION_TITLE_FALLBACKS = {
    "about the author": {"sr": "O autoru", "hr": "O autoru", "bs": "O autoru"},
    "notes": {"sr": "Beleške", "hr": "Bilješke", "bs": "Bilješke"},
    "preface": {"sr": "Predgovor", "hr": "Predgovor", "bs": "Predgovor"},
    "introduction": {"sr": "Uvod", "hr": "Uvod", "bs": "Uvod"},
    "afterword": {"sr": "Pogovor", "hr": "Pogovor", "bs": "Pogovor"},
    "acknowledgements": {"sr": "Zahvalnice", "hr": "Zahvale", "bs": "Zahvale"},
}


def _fallback_section_title(title: str, target_language: str) -> str:
    language = str(target_language or "").strip().lower().split("-")[0]
    key = str(title or "").strip().casefold()
    return _SECTION_TITLE_FALLBACKS.get(key, {}).get(language) or str(title or "").strip()


def fallback_matter_plan(book: Any, *, target_language: str = "") -> dict[str, Any]:
    """A valid plan built only from the filename and the matter inventory.

    Used when the planning graph cannot reach a provider or exhausts its turns
    while searching. The localized title is the filename title (a warning in
    validation records that it is not a published translation); the credit and
    section titles are still filled in.
    """
    title = str(getattr(book, "book_title", "") or "").strip() or "Untitled"
    sections: dict[str, Any] = {}
    for section in getattr(book, "sections", []) or []:
        section_title = _fallback_section_title(section.title, target_language)
        if section_title:
            sections[section.filename] = {"title": section_title}
    credit = "Prevod: BleBel" if str(target_language).lower().startswith("sr") else "Translated by BleBel"
    return {
        "original_title": title,
        "author": str(getattr(book, "book_author", "") or ""),
        "translated_title": title,
        "title_source": "filename",
        "title_evidence": "",
        "translation_credit": credit,
        "edition_note": "",
        "omit": [],
        "sections": sections,
    }


def validate_matter_plan(
    book_or_sections: Any,
    plan: dict[str, Any],
    *,
    target_language: str = "",
) -> dict[str, Any]:
    """Validate the book-level localization plan.

    ``book_or_sections`` may be a :class:`~book_translator.stages.matter.inspect.MatterBook`
    or a list of sections. The plan is normalized in the returned ``plan`` field,
    so a missing credit is filled with the default instead of failing the run.
    """
    sections = getattr(book_or_sections, "sections", book_or_sections)
    sections = list(sections or [])
    issues: list[str] = []
    warnings: list[str] = []
    normalized = dict(plan or {})

    title = _text(normalized.get("translated_title") or normalized.get("title"))
    if not title:
        issues.append("translated_title is required")
    elif len(title) > 300:
        issues.append("translated_title is longer than 300 characters")
    normalized["translated_title"] = title

    original = _text(normalized.get("original_title"))
    if original and title and original.casefold() == title.casefold():
        warnings.append(
            "translated_title is identical to original_title; verify that the target "
            "language really uses the same title, or reuse the published translation"
        )

    credit = _text(normalized.get("translation_credit"))
    if not credit:
        translated_by = _text(normalized.get("translated_by"))
        credit = (
            translated_by
            if _CREDIT_NAME in translated_by
            else DEFAULT_MATTER_CREDIT
        )
    if _CREDIT_NAME not in credit:
        credit = f"{credit} ({_CREDIT_NAME})" if credit else DEFAULT_MATTER_CREDIT
    normalized["translation_credit"] = credit
    normalized["translated_by"] = _CREDIT_NAME
    if not any(section.kind == "front" for section in sections):
        warnings.append("the book has no front matter; the translator credit was not placed")

    known = {section.filename for section in sections}
    raw_sections = normalized.get("sections")
    clean_sections: dict[str, Any] = {}
    if raw_sections is not None:
        if not isinstance(raw_sections, dict):
            issues.append("sections must be an object keyed by matter filename")
        else:
            for filename, value in raw_sections.items():
                name = str(filename)
                if name not in known:
                    warnings.append(f"sections names an unknown matter file: {name}")
                if not isinstance(value, dict):
                    issues.append(f"sections[{name}] must be an object")
                    continue
                entry: dict[str, Any] = {}
                section_title = _text(value.get("title"))
                if section_title:
                    entry["title"] = section_title
                entry["omit"] = _phrase_list(value.get("omit"))
                style = _text(value.get("style"))
                if style:
                    entry["style"] = style
                clean_sections[name] = entry
    normalized["sections"] = clean_sections

    omitted = _phrase_list(normalized.get("omit"))
    section_text: dict[str, str] = {}
    for section in sections:
        path = Path(section.path)
        try:
            section_text[section.filename] = (
                path.read_text(encoding="utf-8") if path.is_file() else ""
            ).casefold()
        except (OSError, UnicodeError):
            section_text[section.filename] = ""
    for phrase in omitted:
        found = any(
            phrase.casefold() in text for text in section_text.values()
        )
        if not found:
            warnings.append(
                f"omit phrase was not found in the matter files: {phrase!r} "
                "(it may be a category rather than an exact phrase)"
            )
    normalized["omit"] = omitted

    evidence = _text(normalized.get("title_evidence"))
    normalized["title_evidence"] = evidence
    if not evidence and target_language:
        warnings.append(
            "no title_evidence/url was recorded; if a published translation exists, "
            "search again and reuse its exact title"
        )
    return {
        "ok": not issues,
        "issues": issues,
        "warnings": warnings,
        "plan": normalized,
        "translated_title": title,
        "sections": sorted(known),
    }


def validate_matter_translation(
    *,
    source_text: str,
    section_kind: str,
    section_filename: str,
    section_title: str,
    plan: dict[str, Any],
    translation: str,
    omitted: bool = False,
    omit_reason: str = "",
    is_first_front_chunk: bool = False,
    is_first_section_chunk: bool = False,
) -> dict[str, Any]:
    """Validate one chunk's localization and return the text to write.

    The returned ``text`` is the *final* form (title/credit formatting applied),
    so the apply node can write exactly what was verified.
    """
    issues: list[str] = []
    warnings: list[str] = []
    reason = _text(omit_reason)
    # A section title in the plan is binding for the translated edition; fall
    # back to the source title when the plan has no override.
    planned_title = _text(section_plan(plan, section_filename).get("title")) or section_title
    if omitted:
        if not reason:
            issues.append("an omitted chunk must explain why in omit_reason")
        text = finalize_translation(
            section_kind=section_kind,
            section_title=planned_title,
            plan=plan,
            translation="",
            omitted=True,
            is_first_front_chunk=is_first_front_chunk,
            is_first_section_chunk=is_first_section_chunk,
        )
    else:
        text = finalize_translation(
            section_kind=section_kind,
            section_title=planned_title,
            plan=plan,
            translation=translation,
            omitted=False,
            is_first_front_chunk=is_first_front_chunk,
            is_first_section_chunk=is_first_section_chunk,
        )
        if not text.strip():
            issues.append("the localized chunk is empty")
        elif not any(character.isalnum() for character in text):
            issues.append("the localized chunk holds no words")
        source = " ".join(str(source_text or "").split())
        answer = " ".join(text.split())
        if source and len(source) >= 80 and answer.casefold() == source.casefold():
            issues.append("the localized chunk is the source text unchanged")
        if (
            source
            and len(source) >= 200
            and len(answer) < MIN_MATTER_LENGTH_RATIO * len(source)
        ):
            warnings.append(
                f"the localized chunk is much shorter than its source "
                f"({len(answer)} for {len(source)} characters); make sure omitted "
                "material was listed in the plan"
            )

    issues.extend(
        f"omitted phrase still appears in the translation: {phrase!r}"
        for phrase in omitted_phrases(plan, section_filename)
        if phrase.casefold() in text.casefold()
    )

    # Front/back matter can carry a cover, a portrait or a facsimile. The
    # localization may rearrange the layout, but it may not lose a picture.
    # A wholly omitted legal page is the one case where dropping its pictures is
    # intended, so the check only applies to content that stays in the book.
    if not omitted:
        source_images = image_refs(source_text)
        remaining_images = image_refs(text)
        for reference in source_images:
            if reference in remaining_images:
                remaining_images.remove(reference)
            else:
                matched_candidate = None
                for candidate in remaining_images:
                    if is_translated_image_match(reference, candidate):
                        matched_candidate = candidate
                        break
                if matched_candidate is not None:
                    remaining_images.remove(matched_candidate)
                else:
                    issues.append(
                        "the localization dropped an image reference; copy every image "
                        f"exactly as it is ({reference[:80]!r})"
                    )
        if remaining_images:
            warnings.append(
                f"the localization added {len(remaining_images)} image reference(s) that "
                "were not in the source; remove them unless the plan asks for a picture"
            )

    if section_kind == "front" and is_first_front_chunk:
        title, credit = _title_and_credit(plan)
        if title and title.casefold() not in text.casefold():
            issues.append("the first front-matter chunk does not carry translated_title")
        if _CREDIT_NAME not in text:
            issues.append(
                f"the first front-matter chunk does not name the translator {_CREDIT_NAME}"
            )
        if not credit:
            warnings.append("no translation_credit was supplied")

    return {
        "ok": not issues,
        "issues": issues,
        "warnings": warnings,
        "omitted": bool(omitted),
        "omit_reason": reason,
        "text": text,
    }


__all__ = [
    "clean_markdown",
    "finalize_translation",
    "omitted_phrases",
    "preserve_trailing_newlines",
    "fallback_matter_plan",
    "section_plan",
    "validate_matter_plan",
    "validate_matter_translation",
]
