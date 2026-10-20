"""Deprecated import path — use :mod:`book_splitter.core.settings` instead.

Kept so existing code (and notebooks) that imported ``book_splitter.config``
keep working unchanged.
"""

from __future__ import annotations

from .core.settings import (
    PROVIDERS,
    TRANSLATION_CHOICES,
    TRANSLATION_PROVIDERS,
    LLMSettings,
    load_dotenv_files,
    resolve_api_key,
    resolve_settings,
    resolve_translation_settings,
    translation_model_label,
)

__all__ = [
    "PROVIDERS",
    "TRANSLATION_CHOICES",
    "TRANSLATION_PROVIDERS",
    "LLMSettings",
    "load_dotenv_files",
    "resolve_api_key",
    "resolve_settings",
    "resolve_translation_settings",
    "translation_model_label",
]
