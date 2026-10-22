"""Deprecated import path — use :mod:`book_splitter.core.settings` instead.

Kept so existing code (and notebooks) that imported ``book_splitter.config``
keep working unchanged.
"""

from __future__ import annotations

from .core.settings import (
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_STAGE,
    LLM_CONFIG_ENV,
    LLM_CONFIG_FILENAME,
    PROVIDERS,
    STAGE_ALIASES,
    STAGES,
    TRANSLATION_CHOICES,
    TRANSLATION_PROVIDERS,
    LLMConfig,
    LLMSettings,
    llm_config_path,
    load_dotenv_files,
    load_llm_config,
    normalize_stage,
    parse_llm_overrides,
    resolve_api_key,
    resolve_settings,
    resolve_translation_settings,
    stage_model_label,
    stage_overrides,
    translation_model_label,
)

__all__ = [
    "DEFAULT_DEEPSEEK_MODEL",
    "DEFAULT_STAGE",
    "LLMConfig",
    "LLM_CONFIG_ENV",
    "LLM_CONFIG_FILENAME",
    "LLMSettings",
    "PROVIDERS",
    "STAGES",
    "STAGE_ALIASES",
    "TRANSLATION_CHOICES",
    "TRANSLATION_PROVIDERS",
    "llm_config_path",
    "load_dotenv_files",
    "load_llm_config",
    "normalize_stage",
    "parse_llm_overrides",
    "resolve_api_key",
    "resolve_settings",
    "resolve_translation_settings",
    "stage_model_label",
    "stage_overrides",
    "translation_model_label",
]
