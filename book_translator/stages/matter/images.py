"""End-to-end multimodal image translation and recreation for front/back matter chunks.

This module sends each front/back matter image together with a contextual prompt
(book title, author, translator credit, and localized matter text) to a multimodal
image-generation model, which recreates the entire image from scratch with the
translated text naturally integrated into the artwork and typography (without
bounding boxes or PIL rectangle overlays).
"""

from __future__ import annotations

import base64
import contextlib
import mimetypes
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from book_translator.core.constants import IMAGES_DIRNAME, MARKDOWN_DIRNAME
from book_translator.core.llm import parse_json_object
from book_translator.documents.images import image_refs

DEFAULT_MULTIMODAL_MODEL = "gemini-nano-banana-2.1"
DEFAULT_MULTIMODAL_PROVIDER = "gemini"


@dataclass
class MatterImageRecord:
    """Record of one image processed during the end-to-end matter image recreation phase."""

    source_ref: str
    source_path: str
    target_ref: str
    target_path: str | None = None
    has_text: bool = False
    extracted_text: str = ""
    translated_text: str = ""
    recreated: bool = False
    error: str | None = None
    log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# markup and reference helpers
# --------------------------------------------------------------------------- #
def parse_image_src(ref: str) -> str:
    """Extract the src/path from a Markdown or HTML image reference."""
    text = str(ref or "").strip()
    m_md = re.match(
        r"!\[(?P<alt>[^\]\n]*)\]\(\s*(?P<url><[^>\n]+>|[^)\s]+)(?P<title>\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)",
        text,
    )
    if m_md:
        url = m_md.group("url")
        if url.startswith("<") and url.endswith(">"):
            url = url[1:-1]
        return url
    m_html = re.search(r'src=["\']([^"\']+)["\']', text, re.IGNORECASE)
    if m_html:
        return m_html.group(1)
    return ""


def replace_image_src(ref: str, new_src: str) -> str:
    """Replace only the image path in a Markdown or HTML reference."""
    text = str(ref or "").strip()
    m_md = re.match(r"(!\[[^\]\n]*\]\(\s*)(?:<[^>\n]+>|[^)\s]+)(.*?\))", text)
    if m_md:
        return f"{m_md.group(1)}{new_src}{m_md.group(2)}"
    m_html = re.search(r'src=(["\'])(?:[^"\']+)(["\'])', text, re.IGNORECASE)
    if m_html:
        quote = m_html.group(1)
        return text[: m_html.start()] + f"src={quote}{new_src}{quote}" + text[m_html.end() :]
    return ref


def is_translated_image_match(source_ref: str, target_ref: str) -> bool:
    """True when target_ref is the original reference or its translated counterpart."""
    if source_ref == target_ref:
        return True
    src_a = parse_image_src(source_ref)
    src_b = parse_image_src(target_ref)
    if not src_a or not src_b:
        return False
    path_a = Path(src_a)
    path_b = Path(src_b)
    stem_a = path_a.stem
    stem_b = path_b.stem
    return stem_b == f"{stem_a}_translated" or stem_b.startswith(f"{stem_a}_translated")


def resolve_matter_image_path(
    src: str,
    *,
    candidates: Sequence[Path | str] | None = None,
) -> Path | None:
    """Find the actual image file on disk matching `src`."""
    clean_src = parse_image_src(src) or src
    if not clean_src:
        return None
    candidate_path = Path(clean_src).expanduser()
    if candidate_path.is_file():
        return candidate_path

    filename = candidate_path.name
    search_dirs: list[Path] = []
    for item in candidates or ():
        p = Path(item).expanduser()
        if p.is_file():
            p = p.parent
        search_dirs.append(p)
        search_dirs.append(p / IMAGES_DIRNAME)
        search_dirs.append(p / MARKDOWN_DIRNAME / IMAGES_DIRNAME)
        search_dirs.append(p.parent / MARKDOWN_DIRNAME / IMAGES_DIRNAME)
        search_dirs.append(p.parent / IMAGES_DIRNAME)
        search_dirs.append(p.parent / MARKDOWN_DIRNAME / IMAGES_DIRNAME)

    for directory in search_dirs:
        option = directory / filename
        if option.is_file():
            return option
        option_rel = directory / clean_src
        if option_rel.is_file():
            return option_rel

    return None


# --------------------------------------------------------------------------- #
# end-to-end multimodal image generation
# --------------------------------------------------------------------------- #
def _image_base64_and_mime(image_path: Path) -> tuple[str, str]:
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
    b64_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return b64_data, mime_type


def build_image_recreation_prompt(
    *,
    target_language: str = "Serbian",
    source_language: str = "English",
    book_title: str = "",
    book_author: str = "",
    translation_credit: str = "",
    matter_context: str = "",
) -> str:
    """Build the prompt instructing the model to recreate the entire image end-to-end."""
    return f"""Recreate this book image end-to-end from scratch with all text translated from {source_language} into {target_language} (Latin alphabet).

Instructions:
1. Generate the whole image from scratch as a complete, seamless image. Do NOT output bounding boxes or draw solid rectangular patches over the original image.
2. Preserve the exact visual composition, artwork, illustrations, background details, ornamental borders, colors, lighting, and aspect ratio of the source image.
3. Render all translated lettering directly and naturally into the recreated scene, matching the original calligraphic, hand-lettered, or typographic style, curvature, and placement.
4. Use the following book translation context to keep titles, author names, and credits consistent:
   - Localized Book Title: {book_title or 'Untitled'}
   - Author: {book_author or 'Unknown'}
   - Translator Credit: {translation_credit or 'Prevod: BleBel'}
   - Surrounding Translated Matter Context:
\"\"\"{matter_context[:1000]}\"\"\"

5. If the image contains NO translatable words at all (for example, a pure illustration without text, a portrait, a publisher logo icon, or an ornamental divider line), do NOT generate a new image and respond only with:
{{"recreated": false, "has_text": false}}"""


def _extract_image_bytes_and_metadata(response: Any) -> tuple[bytes | None, str, str]:
    """Extract generated image bytes and optional text metadata from a model response."""
    if isinstance(response, (bytes, bytearray)):
        return bytes(response), "", ""

    content = getattr(response, "content", response)
    direct_image_bytes: bytes | None = None
    text_parts: list[str] = []

    if isinstance(content, (bytes, bytearray)):
        return bytes(content), "", ""

    if isinstance(content, list):
        for part in content:
            if isinstance(part, (bytes, bytearray)):
                direct_image_bytes = bytes(part)
            elif isinstance(part, dict):
                # Gemini inlineData / inline_data format
                inline = part.get("inlineData") or part.get("inline_data")
                if isinstance(inline, dict) and inline.get("data"):
                    with contextlib.suppress(Exception):
                        direct_image_bytes = base64.b64decode(inline["data"])
                # LangChain / OpenAI image_url or image part format
                if part.get("type") in ("image_url", "image") or "image_url" in part:
                    url_val = (
                        part.get("image_url", {}).get("url", "")
                        if isinstance(part.get("image_url"), dict)
                        else (part.get("url") or part.get("data") or "")
                    )
                    if "base64," in str(url_val):
                        with contextlib.suppress(Exception):
                            direct_image_bytes = base64.b64decode(str(url_val).split("base64,", 1)[1])
                    elif isinstance(url_val, str) and len(url_val) > 64:
                        with contextlib.suppress(Exception):
                            direct_image_bytes = base64.b64decode(url_val)
                if "text" in part:
                    text_parts.append(str(part["text"]))
            elif isinstance(part, str):
                text_parts.append(part)
        raw_text = " ".join(text_parts).strip()
    else:
        raw_text = str(content or "").strip()

    extracted_text = ""
    translated_text = ""

    if raw_text:
        # Check for inline data URI in text
        if direct_image_bytes is None and "data:image/" in raw_text and ";base64," in raw_text:
            m_uri = re.search(r"data:image/[a-zA-Z0-9.+-]+;base64,([A-Za-z0-9+/=\s]+)", raw_text)
            if m_uri:
                with contextlib.suppress(Exception):
                    direct_image_bytes = base64.b64decode(re.sub(r"\s+", "", m_uri.group(1)))

        with contextlib.suppress(Exception):
            data = parse_json_object(raw_text)
            if isinstance(data, dict):
                extracted_text = str(data.get("extracted_text") or data.get("original_text") or "").strip()
                translated_text = str(data.get("translated_text") or "").strip()
                if direct_image_bytes is None:
                    for b64_key in ("image_b64", "image_base64", "image", "generated_image", "data"):
                        val = data.get(b64_key)
                        if isinstance(val, str) and len(val) > 16:
                            b64_clean = val.split("base64,", 1)[1] if "base64," in val else val
                            with contextlib.suppress(Exception):
                                direct_image_bytes = base64.b64decode(b64_clean)
                            if direct_image_bytes:
                                break

    return direct_image_bytes, extracted_text, translated_text


def synthesize_and_translate_image(
    image_path: Path,
    model: Any,
    *,
    target_language: str = "Serbian",
    source_language: str = "English",
    book_title: str = "",
    book_author: str = "",
    translation_credit: str = "",
    matter_context: str = "",
) -> tuple[bool, bytes | None, str, str]:
    """Recreate an image end-to-end from scratch using a multimodal image-generation model.

    Takes the source image and contextual prompt, and asks the model to generate
    the complete translated image from scratch (no bounding boxes or in-place PIL patching).

    Returns ``(recreated, image_bytes, extracted_text, translated_text)``.
    """
    if not image_path.is_file() or model is None:
        return False, None, "", ""

    from langchain_core.messages import HumanMessage

    b64_data, mime_type = _image_base64_and_mime(image_path)
    prompt = build_image_recreation_prompt(
        target_language=target_language,
        source_language=source_language,
        book_title=book_title,
        book_author=book_author,
        translation_credit=translation_credit,
        matter_context=matter_context,
    )

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64_data}"},
            },
        ]
    )

    try:
        if hasattr(model, "invoke"):
            response = model.invoke([message])
        elif callable(model):
            response = model([message])
        else:
            return False, None, "", ""
    except Exception:
        return False, None, "", ""

    image_bytes, extracted_text, translated_text = _extract_image_bytes_and_metadata(response)
    if not image_bytes:
        return False, None, extracted_text, translated_text

    return True, image_bytes, extracted_text, translated_text


# --------------------------------------------------------------------------- #
# chunk-level phase runner
# --------------------------------------------------------------------------- #
def recreate_matter_images_in_chunk(
    chunk_text: str,
    *,
    chunk_path: Path | None = None,
    out_dir: Path,
    model: Any | None,
    source_language: str = "English",
    target_language: str = "Serbian",
    book_title: str = "",
    book_author: str = "",
    translation_credit: str = "",
    matter_context: str = "",
    dry_run: bool = False,
    cache: dict[str, MatterImageRecord] | None = None,
) -> tuple[str, list[MatterImageRecord]]:
    """Scan and recreate all images containing text inside one matter chunk end-to-end."""
    updated_text = str(chunk_text or "")
    references = image_refs(updated_text)
    if not references:
        return updated_text, []

    records: list[MatterImageRecord] = []
    seen_cache = cache if cache is not None else {}
    target_images_dir = out_dir / IMAGES_DIRNAME

    for ref in references:
        src = parse_image_src(ref)
        if not src:
            continue

        cached = seen_cache.get(src)
        if cached is not None:
            if cached.recreated and cached.target_ref:
                updated_text = updated_text.replace(ref, cached.target_ref)
            records.append(cached)
            continue

        resolved = resolve_matter_image_path(
            src,
            candidates=[chunk_path or out_dir, out_dir],
        )
        if resolved is None or not resolved.is_file():
            record = MatterImageRecord(
                source_ref=ref,
                source_path=src,
                target_ref=ref,
                has_text=False,
                recreated=False,
                error=f"image file not found: {src}",
                log=[f"skipped: could not resolve path on disk for {src}"],
            )
            seen_cache[src] = record
            records.append(record)
            continue

        # Destination paths for the recreated image
        stem = re.sub(r"(_translated)+$", "", resolved.stem)
        suffix = resolved.suffix
        new_filename = f"{stem}_translated{suffix}"
        dest_path = target_images_dir / new_filename

        # Check if an already-recreated file exists on disk to reuse
        existing_candidate: Path | None = None
        for cand in [dest_path, resolved.parent / new_filename]:
            if cand.is_file() and cand.stat().st_size > 0:
                existing_candidate = cand
                break

        if existing_candidate is not None:
            if not dry_run:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                if existing_candidate != dest_path:
                    dest_path.write_bytes(existing_candidate.read_bytes())
                source_dest = resolved.parent / new_filename
                if existing_candidate != source_dest:
                    with contextlib.suppress(OSError):
                        source_dest.write_bytes(existing_candidate.read_bytes())

            new_src = (
                f"{IMAGES_DIRNAME}/{new_filename}"
                if "/" not in src
                else src.replace(resolved.name, new_filename)
            )
            new_ref = replace_image_src(ref, new_src)
            updated_text = updated_text.replace(ref, new_ref)

            record = MatterImageRecord(
                source_ref=ref,
                source_path=str(resolved),
                target_ref=new_ref,
                target_path=str(dest_path),
                has_text=True,
                recreated=True,
                log=[f"reused existing end-to-end translated image {dest_path.name}"],
            )
            seen_cache[src] = record
            records.append(record)
            continue

        if model is None:
            record = MatterImageRecord(
                source_ref=ref,
                source_path=str(resolved),
                target_ref=ref,
                target_path=str(resolved),
                has_text=False,
                recreated=False,
                log=[f"inspected {resolved.name}: no image model available"],
            )
            seen_cache[src] = record
            records.append(record)
            continue

        recreated, direct_image_bytes, extracted_text, translated_text = synthesize_and_translate_image(
            resolved,
            model,
            target_language=target_language,
            source_language=source_language,
            book_title=book_title,
            book_author=book_author,
            translation_credit=translation_credit,
            matter_context=matter_context or chunk_text,
        )

        if not recreated or not direct_image_bytes:
            record = MatterImageRecord(
                source_ref=ref,
                source_path=str(resolved),
                target_ref=ref,
                target_path=str(resolved),
                has_text=False,
                recreated=False,
                log=[f"inspected {resolved.name}: no text detected, original preserved"],
            )
            seen_cache[src] = record
            records.append(record)
            continue

        if not dry_run:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(direct_image_bytes)
            source_dest = resolved.parent / new_filename
            if source_dest != dest_path:
                with contextlib.suppress(OSError):
                    source_dest.write_bytes(direct_image_bytes)

        new_src = (
            f"{IMAGES_DIRNAME}/{new_filename}"
            if "/" not in src
            else src.replace(resolved.name, new_filename)
        )
        new_ref = replace_image_src(ref, new_src)
        updated_text = updated_text.replace(ref, new_ref)

        record = MatterImageRecord(
            source_ref=ref,
            source_path=str(resolved),
            target_ref=new_ref,
            target_path=str(dest_path),
            has_text=True,
            extracted_text=extracted_text,
            translated_text=translated_text,
            recreated=True,
            log=[
                f"recreated {resolved.name} -> {dest_path.name} "
                f"(end-to-end generated, {len(direct_image_bytes)} bytes)"
            ],
        )
        seen_cache[src] = record
        records.append(record)

    return updated_text, records


__all__ = [
    "DEFAULT_MULTIMODAL_MODEL",
    "DEFAULT_MULTIMODAL_PROVIDER",
    "MatterImageRecord",
    "build_image_recreation_prompt",
    "is_translated_image_match",
    "parse_image_src",
    "recreate_matter_images_in_chunk",
    "replace_image_src",
    "resolve_matter_image_path",
    "synthesize_and_translate_image",
]
