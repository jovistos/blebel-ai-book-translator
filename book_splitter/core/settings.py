"""Model and credential resolution for the splitting agent.

Resolution order for credentials (first hit wins):

1. explicit CLI argument
2. environment variable (``SPLITTER_API_KEY``, ``DEEPSEEK_API_KEY``, ``OPENAI_API_KEY``, ...)
3. a ``.env`` file at the project root (via python-dotenv, not overriding the env)
4. ``~/.dsh/.credentials.yaml`` ``refs:`` (so an existing DSH key can be reused)

Stage 7 (translation) is stricter: it offers exactly two backends, DeepSeek and a
local LM Studio server, and resolves them with :func:`resolve_translation_settings`.
Its ``TRANSLATION_*`` variables only fall back to the shared ``SPLITTER_*`` ones
when both stages name the same provider, so a ``.env`` written for DeepSeek cannot
leak ``SPLITTER_MODEL=deepseek-flash`` into a local LM Studio run.

Nothing is ever written back or logged.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
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

#: The two backends stage 7 offers. DeepSeek is the cloud default; LM Studio is
#: the model the user is hosting locally on its OpenAI-compatible endpoint.
TRANSLATION_PROVIDERS: dict[str, dict[str, str]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-flash",
        "key_env": "DEEPSEEK_API_KEY",
    },
    "lmstudio": {
        "base_url": "http://127.0.0.1:1234/v1",
        # LM Studio serves whatever model is loaded; an unset model is discovered
        # from the running server rather than guessed.
        "model": "",
        "key_env": "LM_STUDIO_API_KEY",
    },
}
TRANSLATION_CHOICES = tuple(TRANSLATION_PROVIDERS)
DEFAULT_TRANSLATION_PROVIDER = "deepseek"

#: Provider-specific names that may stand in for a ``TRANSLATION_*`` variable.
_TRANSLATION_ENV = {
    "lmstudio": {
        "MODEL": "LMSTUDIO_MODEL",
        "BASE_URL": "LMSTUDIO_BASE_URL",
        "API_KEY": "LM_STUDIO_API_KEY",
    },
}

#: How long model discovery waits for LM Studio before giving up (seconds).
LMSTUDIO_DISCOVERY_TIMEOUT = 2.0

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


# --------------------------------------------------------------------------- #
# stage 7: DeepSeek or a local LM Studio, and nothing else
# --------------------------------------------------------------------------- #
def _shared_provider() -> str:
    """The provider the rest of the pipeline is configured with, lower-cased."""
    return (os.environ.get("SPLITTER_PROVIDER") or "").strip().lower()


def _translation_provider(provider: str | None = None) -> str:
    """Resolve the stage-7 provider, defaulting to DeepSeek.

    ``TRANSLATION_PROVIDER`` wins over ``SPLITTER_PROVIDER`` so the translation
    stage can be switched to LM Studio while the agent stages stay on DeepSeek.
    A shared provider outside the two choices is ignored rather than turned into
    an error: ``SPLITTER_PROVIDER=ollama`` describes the other stages.
    """
    chosen = (provider or os.environ.get("TRANSLATION_PROVIDER") or "").strip().lower()
    if not chosen:
        shared = _shared_provider()
        chosen = shared if shared in TRANSLATION_PROVIDERS else DEFAULT_TRANSLATION_PROVIDER
    if chosen not in TRANSLATION_PROVIDERS:
        raise ConfigError(
            f"unknown translation provider {chosen!r}; the translation stage offers "
            f"{', '.join(sorted(TRANSLATION_PROVIDERS))}"
        )
    return chosen


def _translation_var(chosen: str, name: str) -> str | None:
    """A stage-7 setting from its dedicated, provider-specific or shared env var.

    The shared ``SPLITTER_*`` variable is only consulted when the rest of the
    pipeline uses the same provider. That is what keeps ``SPLITTER_MODEL`` (set
    for DeepSeek) from being sent to a local LM Studio server.
    """
    names = [f"TRANSLATION_{name}"]
    specific = _TRANSLATION_ENV.get(chosen, {}).get(name)
    if specific:
        names.append(specific)
    if _shared_provider() == chosen:
        names.append(f"SPLITTER_{name}")
    for env_name in names:
        value = os.environ.get(env_name)
        if value:
            return value
    return None


def list_lmstudio_models(
    base_url: str, timeout: float = LMSTUDIO_DISCOVERY_TIMEOUT
) -> list[str]:
    """The model ids a running LM Studio server reports, or ``[]`` when it is down.

    Uses the OpenAI-compatible ``GET /v1/models`` endpoint with the standard
    library only, so discovery never becomes a hard dependency on langchain (or
    makes the offline tests import one).
    """
    endpoint = str(base_url or "").rstrip("/") + "/models"
    if not endpoint.startswith(("http://", "https://")):
        return []
    try:
        import json
        import urllib.request

        with urllib.request.urlopen(endpoint, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:  # server down, timeout, HTML error page, bad JSON…
        return []
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    models: list[str] = []
    for item in data:
        if isinstance(item, dict) and item.get("id"):
            models.append(str(item["id"]))
    return models


def pick_lmstudio_model(models: Sequence[str]) -> str:
    """Choose the model to translate with from the server's loaded ones.

    A translation-tuned id (``…-en2sr…``) is preferred when several models are
    loaded; otherwise the first one LM Studio lists is used. Embedding models are
    skipped: LM Studio may list them next to the chat model, but they cannot
    translate a chunk.
    """
    usable = [
        str(model)
        for model in models
        if str(model).strip() and "embed" not in str(model).casefold()
    ]
    if not usable:
        return ""
    for hint in ("en2sr", "translat"):
        for model in usable:
            if hint in model.casefold():
                return model
    return usable[0]


def translation_model_label(provider: str | None = None, model: str | None = None) -> str:
    """``deepseek/deepseek-flash`` or ``lmstudio/auto`` — for a banner, never raises.

    ``auto`` means the model will be discovered from the running LM Studio server
    when the first chunk is translated.
    """
    try:
        chosen = _translation_provider(provider)
    except ConfigError:
        return str(provider or "—")
    resolved = (
        model
        or _translation_var(chosen, "MODEL")
        or TRANSLATION_PROVIDERS[chosen].get("model")
        or "auto"
    )
    return f"{chosen}/{resolved}"


def resolve_translation_settings(
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    discover_model: bool = True,
) -> LLMSettings:
    """Build the stage-7 connection: DeepSeek, or the local LM Studio server.

    The model for LM Studio is resolved in this order: explicit ``model`` →
    ``TRANSLATION_MODEL`` → ``LMSTUDIO_MODEL`` → the running server's
    ``GET /v1/models`` (preferring a translation-tuned id) → an error naming the
    server. A local endpoint needs no key; a remote one still does.
    """
    load_dotenv_files()
    chosen = _translation_provider(provider)
    spec = TRANSLATION_PROVIDERS[chosen]
    resolved_base = (
        base_url or _translation_var(chosen, "BASE_URL") or spec.get("base_url") or ""
    )
    resolved_model = (
        model or _translation_var(chosen, "MODEL") or spec.get("model") or ""
    )
    if not resolved_model and chosen == "lmstudio" and discover_model:
        resolved_model = pick_lmstudio_model(list_lmstudio_models(resolved_base))
    if not resolved_model:
        if chosen == "lmstudio":
            raise ConfigError(
                f"no model to translate with: LM Studio is not answering at "
                f"{resolved_base} (GET {resolved_base.rstrip('/')}/models). Start the "
                "server, or pass --translator lmstudio --model <id>."
            )
        raise ConfigError(
            f"no model configured for translation provider {chosen!r}; pass --model."
        )
    resolved_key = api_key or _translation_var(chosen, "API_KEY")
    if not resolved_key and chosen == "deepseek":
        resolved_key = resolve_api_key("deepseek")
    if not resolved_key and not any(host in resolved_base for host in _LOCAL_HOSTS):
        raise ConfigError(
            f"no API key found for translation provider {chosen!r}. Set "
            f"{spec.get('key_env', 'SPLITTER_API_KEY')} or TRANSLATION_API_KEY in the "
            "environment or in .env, or pass --api-key."
        )
    return LLMSettings(
        provider=chosen,
        model=resolved_model,
        base_url=resolved_base,
        api_key=resolved_key or "not-needed",
        temperature=temperature,
    )


__all__ = [
    "DEFAULT_TRANSLATION_PROVIDER",
    "LMSTUDIO_DISCOVERY_TIMEOUT",
    "PROVIDERS",
    "TRANSLATION_CHOICES",
    "TRANSLATION_PROVIDERS",
    "LLMSettings",
    "list_lmstudio_models",
    "load_dotenv_files",
    "pick_lmstudio_model",
    "resolve_api_key",
    "resolve_settings",
    "resolve_translation_settings",
    "translation_model_label",
]
