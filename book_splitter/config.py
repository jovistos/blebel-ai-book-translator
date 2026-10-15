"""Deprecated import path — use :mod:`book_splitter.core.settings` instead.

Kept so existing code (and notebooks) that imported ``book_splitter.config``
keep working unchanged.
"""

from __future__ import annotations

from .core.settings import (
    PROVIDERS,
    LLMSettings,
    load_dotenv_files,
    resolve_api_key,
    resolve_settings,
)

__all__ = [
    "PROVIDERS",
    "LLMSettings",
    "load_dotenv_files",
    "resolve_api_key",
    "resolve_settings",
]
