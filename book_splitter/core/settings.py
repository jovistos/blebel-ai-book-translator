"""Model and credential resolution for the splitting agent.

Resolution order for credentials (first hit wins):

1. explicit CLI argument
2. environment variable (``SPLITTER_API_KEY``, ``DEEPSEEK_API_KEY``, ``OPENAI_API_KEY``, ...)
3. a ``.env`` file at the project root (via python-dotenv, not overriding the env)
4. ``~/.dsh/.credentials.yaml`` ``refs:`` (so an existing DSH key can be reused)

Nothing is ever written back or logged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..paths import repo_root
from .errors import ConfigError

PROVIDERS: dict[str, dict[str, str]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-flash",
        "key_env": "DEEPSEEK_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4.1-mini",
        "key_env": "OPENAI_API_KEY",
    },
    "lmstudio": {
        "base_url": "http://127.0.0.1:1234/v1",
        "model": "local-model",
        "key_env": "LM_STUDIO_API_KEY",
    },
    "ollama": {
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen3:14b",
        "key_env": "OLLAMA_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "google/gemini-2.5-flash",
        "key_env": "OPENROUTER_API_KEY",
    },
}

_DSH_CREDENTIALS = Path.home() / ".dsh" / ".credentials.yaml"
_LOCAL_HOSTS = ("127.0.0.1", "localhost", "0.0.0.0")


def load_dotenv_files() -> None:
    """Load the project ``.env`` (and the CWD one) without overriding the env."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - optional dependency
        return
    for candidate in (repo_root() / ".env", Path.cwd() / ".env"):
        if candidate.is_file():
            load_dotenv(candidate, override=False)


def _dsh_ref(name: str) -> str | None:
    if not _DSH_CREDENTIALS.is_file():
        return None
    try:
        import yaml  # type: ignore
    except ImportError:
        return None
    try:
        data = yaml.safe_load(_DSH_CREDENTIALS.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    value = (data.get("refs") or {}).get(name)
    return str(value) if value else None


def resolve_api_key(provider: str, explicit: str | None = None) -> str | None:
    """Find an API key for the provider, allowing local servers to run keyless."""
    if explicit:
        return explicit
    load_dotenv_files()
    spec = PROVIDERS.get(provider)
    if not spec:
        return os.environ.get("SPLITTER_API_KEY")
    for env_name in ("SPLITTER_API_KEY", spec["key_env"]):
        value = os.environ.get(env_name)
        if value:
            return value
    return _dsh_ref(spec["key_env"])


@dataclass(slots=True)
class LLMSettings:
    """Resolved connection settings for the routing model."""

    provider: str
    model: str
    base_url: str
    api_key: str | None
    temperature: float = 0.0
    timeout: float = 180.0
    max_retries: int = 3

    @property
    def is_local(self) -> bool:
        return any(host in self.base_url for host in _LOCAL_HOSTS)


def resolve_settings(
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
) -> LLMSettings:
    """Build :class:`LLMSettings` from flags, environment and provider defaults."""
    load_dotenv_files()
    provider = (provider or os.environ.get("SPLITTER_PROVIDER") or "deepseek").lower()
    spec = PROVIDERS.get(provider, {})
    resolved_base = base_url or os.environ.get("SPLITTER_BASE_URL") or spec.get("base_url")
    resolved_model = model or os.environ.get("SPLITTER_MODEL") or spec.get("model")
    if not resolved_base or not resolved_model:
        raise ConfigError(
            f"unknown provider {provider!r}; pass --base-url and --model, or choose one of "
            f"{', '.join(sorted(PROVIDERS))}"
        )
    key = resolve_api_key(provider, api_key)
    if not key and not any(host in resolved_base for host in _LOCAL_HOSTS):
        raise ConfigError(
            f"no API key found for provider {provider!r}. Set "
            f"{spec.get('key_env', 'SPLITTER_API_KEY')} in the environment or in .env, "
            "or pass --api-key."
        )
    return LLMSettings(
        provider=provider,
        model=resolved_model,
        base_url=resolved_base,
        api_key=key or "not-needed",
        temperature=temperature,
    )
