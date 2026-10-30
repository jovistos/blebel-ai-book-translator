"""Chat models bound to the matter agent's toolsets.

The book-level model sees the planning tools; the per-chunk model sees the
localization tools plus the readers. Both include the splitter's web tools and
the shared structural tools, because the matter stage reuses them instead of
growing a second copy.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage

from book_translator.core.llm import build_chat_model
from book_translator.core.settings import LLMSettings, resolve_settings
from book_translator.stages.agent.llm import bind_tools
from book_translator.toolkit.matter import BOOK_TOOL_NAMES, CHUNK_TOOL_NAMES
from book_translator.toolkit.registry import tool_by_name

DEFAULT_MATTER_IMAGE_MODEL = "gemini-nano-banana-2.1"
DEFAULT_MATTER_IMAGE_PROVIDER = "gemini"
GEMINI_GENERATE_CONTENT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiImageGenerationModel:
    """Native Gemini multimodal image generation client (end-to-end image synthesis)."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MATTER_IMAGE_MODEL,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model or DEFAULT_MATTER_IMAGE_MODEL
        self.api_key = api_key or ""
        self.timeout = timeout

    def invoke(self, messages: Sequence[Any], *_: Any, **__: Any) -> AIMessage:
        parts: list[dict[str, Any]] = []
        for msg in messages:
            content = getattr(msg, "content", msg)
            if isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        parts.append({"text": item})
                    elif isinstance(item, dict):
                        if item.get("type") == "text" and "text" in item:
                            parts.append({"text": str(item["text"])})
                        elif item.get("type") in ("image_url", "image") or "image_url" in item:
                            url_val = (
                                item.get("image_url", {}).get("url", "")
                                if isinstance(item.get("image_url"), dict)
                                else str(item.get("url") or "")
                            )
                            if url_val.startswith("data:") and ";base64," in url_val:
                                header, b64 = url_val.split(";base64,", 1)
                                mime = header[5:] or "image/png"
                                parts.append({"inlineData": {"mimeType": mime, "data": b64}})

        url = f"{GEMINI_GENERATE_CONTENT_BASE_URL}/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            cand_parts = (
                (data.get("candidates") or [{}])[0]
                .get("content", {})
                .get("parts", [])
            )
            return AIMessage(content=cand_parts)
        except Exception:
            return AIMessage(content='{"recreated": false, "has_text": false}')


def _tools(names: tuple[str, ...]) -> list[Any]:
    tools = []
    for name in names:
        tool = tool_by_name(name)
        if tool is not None:
            tools.append(tool)
    return tools


def make_matter_llm(settings: LLMSettings, tools: list[Any] | None = None) -> Any:
    """A chat model with the matter toolset bound to it."""
    return bind_tools(
        build_chat_model(settings),
        _tools(CHUNK_TOOL_NAMES) if tools is None else tools,
    )


def make_book_llm(settings: LLMSettings) -> Any:
    """The model used by the book-level localization-plan graph."""
    return make_matter_llm(settings, _tools(BOOK_TOOL_NAMES))


def make_chunk_llm(settings: LLMSettings) -> Any:
    """The model used by one matter chunk's localization graph."""
    return make_matter_llm(settings, _tools(CHUNK_TOOL_NAMES))


def make_matter_image_llm(settings: LLMSettings | None = None) -> Any:
    """The multimodal model used by the end-to-end matter image recreation phase."""
    if settings is None:
        settings = resolve_settings(
            provider=DEFAULT_MATTER_IMAGE_PROVIDER,
            model=DEFAULT_MATTER_IMAGE_MODEL,
            stage="matter",
            require_key=False,
        )
    if settings.provider == "gemini":
        model_name = settings.model or DEFAULT_MATTER_IMAGE_MODEL
        return GeminiImageGenerationModel(
            model=model_name,
            api_key=settings.api_key,
            timeout=settings.timeout,
        )
    import os
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        return GeminiImageGenerationModel(
            model=DEFAULT_MATTER_IMAGE_MODEL,
            api_key=gemini_key,
            timeout=getattr(settings, "timeout", 120.0) if settings else 120.0,
        )
    model = build_chat_model(settings)
    if hasattr(model, "max_retries"):
        model.max_retries = 1
    return model


__all__ = [
    "DEFAULT_MATTER_IMAGE_MODEL",
    "DEFAULT_MATTER_IMAGE_PROVIDER",
    "GeminiImageGenerationModel",
    "make_book_llm",
    "make_chunk_llm",
    "make_matter_image_llm",
    "make_matter_llm",
]
