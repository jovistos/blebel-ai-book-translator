"""Model and credential resolution for every stage that calls a model.

One file holds the model of every pipeline stage that calls an LLM —
``book_translator.toml`` (or the file named by ``--llm-config`` /
``$BOOK_TRANSLATOR_CONFIG``). Its ``[llm]`` table holds the shared defaults and each
``[llm.<stage>]`` table overrides the stages that need a different model::

    [llm]
    provider = "deepseek"
    model = "deepseek-flash"

    [llm.translate]
    provider = "lmstudio"
    base_url = "http://127.0.0.1:1234/v1"

Every field is optional. A stage inherits ``model`` / ``base_url`` from ``[llm]``
only when the two agree on the provider, so switching translation to a local
server never inherits the cloud model id.

Resolution order for one stage and one field (first hit wins):

1. ``--llm <stage>.<field>=VALUE`` (the most specific command-line switch)
2. the stage's own flag (``--translate-model`` on the pipeline, ``--model`` on a
   single ``book_translator`` command)
3. stage-specific environment (``SPLITTER_<STAGE>_<FIELD>``, ``TRANSLATION_*``,
   ``LMSTUDIO_*``)
4. the shared command-line flags (``--provider``, ``--model``, ``--base-url``,
   ``--api-key``, ``--temperature``) of the pipeline
5. ``[llm.<stage>]`` in the config file
6. ``[llm]`` in the config file (when the provider matches)
7. the legacy shared environment (``SPLITTER_PROVIDER``, ``SPLITTER_MODEL``, ...)
8. the provider's built-in default

That list is executable, not prose: :data:`FIELD_SOURCES` is the ordered table the
resolver walks, and ``tests/test_settings_precedence.py`` exercises every adjacent
pair. Providers have their own :data:`PROVIDER_SOURCES` table (the stage-specific
sources may be rejected by a restricted stage, while a broad one is skipped), and
credentials have :data:`CREDENTIAL_SOURCES` because a key has sources the config
file must never hold.

Credentials are never stored in the config file. Keys come from the environment
(a project ``.env`` is loaded without overriding the real environment), then from
``~/.dsh/.credentials.yaml`` ``refs:`` so an existing DSH key can be reused.

Stage 7 (translation) keeps its two-backend choice (DeepSeek or a local LM Studio
server): its provider is resolved like the others but restricted to those two.

Nothing is ever written back or logged.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..paths import repo_root
from .errors import ConfigError

#: DeepSeek's default model: V4.1 Flash. Every stage uses it unless a stage
#: section, an environment variable or a command-line override says otherwise.
DEFAULT_DEEPSEEK_MODEL = "deepseek-flash"

PROVIDERS: dict[str, dict[str, str]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": DEFAULT_DEEPSEEK_MODEL,
        "key_env": "DEEPSEEK_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4.1-mini",
        "key_env": "OPENAI_API_KEY",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-3.8-flash",
        "key_env": "GEMINI_API_KEY",
    },
    "claude": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "anthropic/claude-3.5-haiku",
        "key_env": "ANTHROPIC_API_KEY",
    },
    "anthropic": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "anthropic/claude-3.5-haiku",
        "key_env": "ANTHROPIC_API_KEY",
    },
    "lmstudio": {
        "base_url": "http://127.0.0.1:1234/v1",
        "model": "",
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
        "model": DEFAULT_DEEPSEEK_MODEL,
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

#: Every pipeline part that calls a model, in pipeline order. ``dictionary`` is
#: the glossary extraction half of the dictionary phase (stage 4).
STAGES: dict[str, str] = {
    "split": "stage 1 — the routing agent",
    "preprocess": "stage 2 — the per-chapter cleaning agent",
    "dictionary": "stages 3–4 — the glossary extractor",
    "matter": "stage 6b — the front/back matter agent",
    "matter_image": "stage 6b — multimodal matter image recreation",
    "translate": "stage 7 — the chunk translator",
    "postprocess": "stage 9 — the headings/TOC agent",
}

#: The stage ``resolve_settings()`` selects when the caller does not name one.
DEFAULT_STAGE = "split"

#: Friendly names accepted in a config section or a ``--llm`` switch.
STAGE_ALIASES = {
    "glossary": "dictionary",
    "translation": "translate",
    "pre_process": "preprocess",
    "post_process": "postprocess",
    "matter_images": "matter_image",
    "matter_img": "matter_image",
}

#: Provider-specific names that may stand in for a stage setting.
_PROVIDER_ENV = {
    "lmstudio": {
        "MODEL": "LMSTUDIO_MODEL",
        "BASE_URL": "LMSTUDIO_BASE_URL",
        "API_KEY": "LM_STUDIO_API_KEY",
    },
}


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """One field a ``[llm]`` section, an environment variable or a flag may set."""

    name: str
    kind: type[str] | type[float] | type[int]
    default: Any = None
    #: Only inherit from a broad source written for the same provider.
    provider_scoped: bool = False
    #: A ``[llm.<stage>]`` value is ignored when the stage's provider is switched.
    section_scoped: bool = False

    def env_names(self, stage: str, provider: str | None = None) -> list[str]:
        """Environment names that may hold this field, most specific first."""
        upper = self.name.upper()
        names = [f"SPLITTER_{stage.upper()}_{upper}"]
        if stage == "translate":
            names.append(f"TRANSLATION_{upper}")
        if provider:
            specific = _PROVIDER_ENV.get(provider, {}).get(upper)
            if specific:
                names.append(specific)
        return names


#: The fields a ``[llm]`` / ``[llm.<stage>]`` section or a ``--llm`` switch may set.
FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("provider", str),
    FieldSpec("model", str, provider_scoped=True, section_scoped=True),
    FieldSpec("base_url", str, provider_scoped=True, section_scoped=True),
    FieldSpec("temperature", float, 0.0),
    FieldSpec("timeout", float, 180.0),
    FieldSpec("max_retries", int, 3),
    FieldSpec("api_key_env", str, provider_scoped=True),
)
FIELD_BY_NAME: dict[str, FieldSpec] = {spec.name: spec for spec in FIELDS}
LLM_FIELDS = tuple(spec.name for spec in FIELDS)

#: The one file that answers "which model does each stage use?".
LLM_CONFIG_FILENAME = "book_translator.toml"
#: Environment variable naming an alternative config file.
LLM_CONFIG_ENV = "BOOK_TRANSLATOR_CONFIG"

#: How long model discovery waits for LM Studio before giving up (seconds).
LMSTUDIO_DISCOVERY_TIMEOUT = 2.0

_DSH_CREDENTIALS = Path.home() / ".dsh" / ".credentials.yaml"
_LOCAL_HOSTS = ("127.0.0.1", "localhost", "0.0.0.0")


# --------------------------------------------------------------------------- #
# environment and the project .env
# --------------------------------------------------------------------------- #
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
        import yaml
    except ImportError:
        return None
    try:
        data = yaml.safe_load(_DSH_CREDENTIALS.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    value = (data.get("refs") or {}).get(name)
    return str(value) if value else None


def _text(value: Any) -> str | None:
    """A stripped string, or ``None`` when the value is empty/absent."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _env(name: str) -> str | None:
    return _text(os.environ.get(name))


def _env_field(name: str, field_name: str) -> Any:
    """Read one environment variable and coerce it to the field's type."""
    raw = _env(name)
    if raw is None:
        return None
    spec = FIELD_BY_NAME[field_name]
    if spec.kind is float:
        try:
            return float(raw)
        except ValueError as exc:
            raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    if spec.kind is int:
        try:
            return int(raw)
        except ValueError as exc:
            raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    return raw


def _is_local(base_url: str | None) -> bool:
    return any(host in str(base_url or "") for host in _LOCAL_HOSTS)


# --------------------------------------------------------------------------- #
# the config file
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class LLMConfig:
    """The parsed ``book_translator.toml``: shared defaults and per-stage overrides."""

    path: Path | None = None
    defaults: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)

    def stage(self, name: str) -> dict[str, Any]:
        """A copy of one stage's section (empty when the stage has none)."""
        return dict(self.stages.get(normalize_stage(name) or "", {}))


def normalize_stage(stage: str | None) -> str | None:
    """Canonical stage name, or ``None`` when it is not one of :data:`STAGES`."""
    name = str(stage or "").strip().lower().replace("-", "_")
    name = STAGE_ALIASES.get(name, name)
    return name if name in STAGES else None


def _canonical_stage(stage: str | None) -> str:
    if not str(stage or "").strip():
        return DEFAULT_STAGE
    canonical = normalize_stage(stage)
    if canonical is None:
        raise ConfigError(
            f"unknown pipeline stage {stage!r}; choose one of {', '.join(STAGES)}"
        )
    return canonical


def llm_config_path(explicit: str | Path | None = None) -> Path | None:
    """The config file to read: explicit path → ``$BOOK_TRANSLATOR_CONFIG`` → cwd/repo."""
    if explicit:
        return Path(explicit).expanduser()
    env = _env(LLM_CONFIG_ENV)
    if env:
        return Path(env).expanduser()
    seen: set[Path] = set()
    for base in (Path.cwd(), repo_root()):
        candidate = base / LLM_CONFIG_FILENAME
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return None


def load_llm_config(path: str | Path | None = None) -> LLMConfig:
    """Read the per-stage model config, or an empty one when no file exists."""
    resolved = llm_config_path(path)
    if resolved is None:
        return LLMConfig()
    if not resolved.is_file():
        raise ConfigError(
            f"LLM config file not found: {resolved}. Create it, or drop --llm-config / "
            f"{LLM_CONFIG_ENV}."
        )
    try:
        import tomllib

        raw = tomllib.loads(resolved.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read the LLM config {resolved}: {exc}") from exc
    except ValueError as exc:  # tomllib.TOMLDecodeError subclasses ValueError
        raise ConfigError(f"{resolved} is not valid TOML: {exc}") from exc
    table = raw.get("llm")
    if table is None:
        if not raw:
            return LLMConfig(path=resolved)
        raise ConfigError(
            f"{resolved}: expected an [llm] table (found {', '.join(sorted(raw))}); "
            f"see {LLM_CONFIG_FILENAME} for the format"
        )
    if not isinstance(table, dict):
        raise ConfigError(f"{resolved}: [llm] must be a table")
    defaults: dict[str, Any] = {}
    stages: dict[str, dict[str, Any]] = {}
    for key, value in table.items():
        if isinstance(value, dict):
            stage = normalize_stage(key)
            if stage is None:
                raise ConfigError(
                    f"{resolved}: unknown stage {key!r} in [llm]; choose one of "
                    f"{', '.join(STAGES)} (write it as [llm.{key}])"
                )
            stages[stage] = _config_section(f"{resolved} [llm.{key}]", value)
            continue
        if key not in LLM_FIELDS:
            raise ConfigError(
                f"{resolved}: unknown key {key!r} in [llm]; use one of "
                f"{', '.join(LLM_FIELDS)}, or a stage table [llm.<stage>]"
            )
        defaults[key] = _coerce_field(key, value, where=f"{resolved} [llm]")
    if any(name in defaults for name in ("model", "base_url", "api_key_env")) and not defaults.get(
        "provider"
    ):
        raise ConfigError(
            f"{resolved}: [llm] sets model/base_url/api_key_env without a provider; add "
            'provider = "deepseek" (or the provider those values belong to) to [llm]'
        )
    return LLMConfig(path=resolved, defaults=defaults, stages=stages)


def _config_section(where: str, section: Mapping[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in section.items():
        if key not in LLM_FIELDS:
            raise ConfigError(
                f"{where}: unknown key {key!r}; use one of {', '.join(LLM_FIELDS)}"
            )
        cleaned[key] = _coerce_field(key, value, where=where)
    return cleaned


def _coerce_field(name: str, value: Any, *, where: str) -> Any:
    spec = FIELD_BY_NAME.get(name)
    if spec is None:
        raise ConfigError(f"{where}: unknown key {name!r}")
    try:
        if spec.kind is str:
            if not isinstance(value, str):
                raise TypeError("expected a string")
            return value.strip() or None
        if spec.kind is float:
            return float(value)
        if spec.kind is int:
            return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where}: {name} = {value!r} is not valid ({exc})") from exc
    raise ConfigError(f"{where}: unknown key {name!r}")


def parse_llm_overrides(
    values: Sequence[str] | None,
    *,
    default_stage: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Turn repeated ``[STAGE.]FIELD=VALUE`` switches into per-stage overrides.

    ``default_stage`` lets a single ``book_translator`` command accept ``--llm
    model=...`` (that command's stage). The pipeline has no such default: there,
    every switch names its stage, e.g. ``--llm translate.provider=lmstudio``.
    """
    overrides: dict[str, dict[str, Any]] = {}
    for raw in values or ():
        text = str(raw).strip()
        if not text:
            continue
        if "=" not in text:
            raise ConfigError(f"--llm expects [STAGE.]FIELD=VALUE, got {text!r}")
        left, value = text.split("=", 1)
        left = left.strip()
        if "." in left:
            stage_part, field_name = left.rsplit(".", 1)
            stage = normalize_stage(stage_part)
            if stage is None:
                raise ConfigError(
                    f"--llm {text!r}: unknown stage {stage_part!r}; choose one of "
                    f"{', '.join(STAGES)}"
                )
        else:
            field_name = left
            if default_stage is None:
                raise ConfigError(
                    f"--llm {text!r}: name the stage, e.g. --llm split.{field_name}=..."
                )
            stage = _canonical_stage(default_stage)
        field_name = field_name.strip().lower().replace("-", "_")
        if field_name not in LLM_FIELDS:
            raise ConfigError(
                f"--llm {text!r}: unknown field {field_name!r}; use one of "
                f"{', '.join(LLM_FIELDS)}"
            )
        overrides.setdefault(stage, {})[field_name] = _coerce_field(
            field_name, value, where="--llm"
        )
    return overrides


def stage_overrides(
    overrides: Mapping[str, Mapping[str, Any]] | None,
    stage: str,
) -> dict[str, Any]:
    """The overrides that belong to one stage."""
    canonical = normalize_stage(stage) or stage
    return dict((overrides or {}).get(canonical, {}))


# --------------------------------------------------------------------------- #
# credentials
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class _CredentialContext:
    provider: str
    stage: str
    explicit: str | None
    config: LLMConfig
    overrides: Mapping[str, Any]
    shared: Mapping[str, Any]
    provider_specific: bool
    shared_key_ok: bool
    env_shared_key_ok: bool
    spec: Mapping[str, str]


def _credential_explicit(ctx: _CredentialContext) -> str | None:
    return ctx.explicit


def _credential_override_key(ctx: _CredentialContext) -> str | None:
    value = ctx.overrides.get("api_key")
    return str(value) if value else None


def _credential_override_key_env(ctx: _CredentialContext) -> str | None:
    name = ctx.overrides.get("api_key_env")
    return _env(str(name)) if name else None


def _credential_shared(ctx: _CredentialContext) -> str | None:
    return _text(ctx.shared.get("api_key")) if ctx.shared_key_ok else None


def _credential_stage_env(ctx: _CredentialContext) -> str | None:
    for env_name in _stage_env_names(ctx.stage, "API_KEY", ctx.provider):
        value = _env(env_name)
        if value:
            return value
    return None


def _credential_stage_section(ctx: _CredentialContext) -> str | None:
    name = ctx.config.stages.get(ctx.stage, {}).get("api_key_env")
    return _env(str(name)) if name else None


def _credential_defaults(ctx: _CredentialContext) -> str | None:
    name = ctx.config.defaults.get("api_key_env")
    defaults_provider = _text(ctx.config.defaults.get("provider"))
    if defaults_provider == ctx.provider and name:
        return _env(str(name))
    return None


def _credential_provider_env(ctx: _CredentialContext) -> str | None:
    name = ctx.spec.get("key_env")
    if name:
        val = _env(name)
        if val:
            return val
    if ctx.provider == "gemini":
        return _env("GOOGLE_API_KEY")
    if ctx.provider in {"claude", "anthropic"}:
        return _env("CLAUDE_API_KEY") or _env("OPENROUTER_API_KEY")
    return None


def _credential_legacy_shared(ctx: _CredentialContext) -> str | None:
    return _env("SPLITTER_API_KEY") if ctx.env_shared_key_ok else None


def _credential_dsh(ctx: _CredentialContext) -> str | None:
    name = ctx.spec.get("key_env")
    return _dsh_ref(name) if name else None


@dataclass(frozen=True, slots=True)
class CredentialSource:
    """One place an API key may come from, in resolution order."""

    key: str
    label: str
    resolve: Callable[[_CredentialContext], str | None]


#: The key lookup order. It is separate from the field table because a credential
#: has its own sources (the DSH credential store, provider-owned variables) and
#: never appears in the config file itself.
CREDENTIAL_SOURCES: tuple[CredentialSource, ...] = (
    CredentialSource("explicit", "--api-key", _credential_explicit),
    CredentialSource("override_key", "--llm <stage>.api_key", _credential_override_key),
    CredentialSource(
        "override_key_env", "--llm <stage>.api_key_env", _credential_override_key_env
    ),
    CredentialSource("shared_key", "the shared --api-key", _credential_shared),
    CredentialSource(
        "stage_env",
        "SPLITTER_<STAGE>_API_KEY / TRANSLATION_API_KEY / provider key variable",
        _credential_stage_env,
    ),
    CredentialSource(
        "stage_section", "[llm.<stage>].api_key_env", _credential_stage_section
    ),
    CredentialSource("defaults", "[llm].api_key_env", _credential_defaults),
    CredentialSource("provider_env", "the provider's key_env", _credential_provider_env),
    CredentialSource("legacy_shared", "SPLITTER_API_KEY", _credential_legacy_shared),
    CredentialSource("dsh", "~/.dsh/.credentials.yaml refs", _credential_dsh),
)


def resolve_api_key(
    provider: str,
    explicit: str | None = None,
    *,
    stage: str | None = None,
    config: LLMConfig | None = None,
    config_path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    shared: Mapping[str, Any] | None = None,
    provider_specific: bool = False,
) -> str | None:
    """Find an API key for the provider, allowing local servers to run keyless."""
    if explicit:
        return explicit
    load_dotenv_files()
    cfg = config if config is not None else load_llm_config(config_path)
    canonical = _canonical_stage(stage)
    shared_data = dict(shared or {})
    spec = PROVIDERS.get(provider) or TRANSLATION_PROVIDERS.get(provider) or {}
    shared_provider = _text(shared_data.get("provider"))
    env_shared_provider = _env("SPLITTER_PROVIDER")
    ctx = _CredentialContext(
        provider=provider,
        stage=canonical,
        explicit=explicit,
        config=cfg,
        overrides=dict(overrides or {}),
        shared=shared_data,
        provider_specific=provider_specific,
        shared_key_ok=shared_provider == provider
        or (shared_provider is None and not provider_specific),
        env_shared_key_ok=env_shared_provider == provider
        or (env_shared_provider is None and not provider_specific),
        spec=spec,
    )
    for source in CREDENTIAL_SOURCES:
        value = source.resolve(ctx)
        if value:
            return str(value)
    return None


def _stage_env_names(stage: str, field_name: str, provider: str | None = None) -> list[str]:
    """Environment names that may hold one stage's setting, most specific first."""
    spec = FIELD_BY_NAME.get(field_name)
    if spec is not None:
        return spec.env_names(stage, provider)
    upper = field_name.upper()
    names = [f"SPLITTER_{stage.upper()}_{upper}"]
    if stage == "translate":
        names.append(f"TRANSLATION_{upper}")
    if provider:
        specific = _PROVIDER_ENV.get(provider, {}).get(upper)
        if specific:
            names.append(specific)
    return names


# --------------------------------------------------------------------------- #
# the resolved connection
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class LLMSettings:
    """Resolved connection settings for one pipeline stage."""

    provider: str
    model: str
    base_url: str
    api_key: str | None
    temperature: float = 0.0
    timeout: float = 180.0
    max_retries: int = 3
    stage: str = ""
    config_path: str | None = None

    @property
    def is_local(self) -> bool:
        return _is_local(self.base_url)


def _shared_field_ok(
    source_provider: str | None,
    provider: str,
    *,
    provider_specific: bool,
) -> bool:
    """Whether a shared/broad source may set a field for this stage."""
    if source_provider is not None:
        return source_provider == provider
    return not provider_specific


_MISSING = object()


@dataclass(slots=True)
class _ProviderContext:
    stage: str
    explicit: str | None
    config: LLMConfig
    overrides: Mapping[str, Any]
    shared: Mapping[str, Any]


def _provider_override(ctx: _ProviderContext) -> Any:
    return ctx.overrides.get("provider")


def _provider_explicit(ctx: _ProviderContext) -> Any:
    return ctx.explicit


def _provider_stage_env(ctx: _ProviderContext) -> Any:
    return _env(f"SPLITTER_{ctx.stage.upper()}_PROVIDER")


def _provider_translation_env(ctx: _ProviderContext) -> Any:
    return _env("TRANSLATION_PROVIDER") if ctx.stage == "translate" else None


def _provider_shared(ctx: _ProviderContext) -> Any:
    return ctx.shared.get("provider")


def _provider_stage_section(ctx: _ProviderContext) -> Any:
    return ctx.config.stages.get(ctx.stage, {}).get("provider")


def _provider_defaults(ctx: _ProviderContext) -> Any:
    return ctx.config.defaults.get("provider")


def _provider_shared_env(ctx: _ProviderContext) -> Any:
    return _env("SPLITTER_PROVIDER")


@dataclass(frozen=True, slots=True)
class ProviderSource:
    """One place a stage's provider may be named, in resolution order."""

    key: str
    label: str
    specific: bool
    resolve: Callable[[_ProviderContext], Any]


#: The provider lookup order. ``specific`` sources belong to this one stage, so
#: an unsupported name from one of them is an error; a broad source is ignored
#: instead, which lets a global ``SPLITTER_PROVIDER`` coexist with a stage 7 that
#: only offers deepseek/lmstudio.
PROVIDER_SOURCES: tuple[ProviderSource, ...] = (
    ProviderSource("override", "--llm <stage>.provider", True, _provider_override),
    ProviderSource("explicit", "the command line", True, _provider_explicit),
    ProviderSource(
        "stage_env", "SPLITTER_<STAGE>_PROVIDER", True, _provider_stage_env
    ),
    ProviderSource(
        "translation_env", "TRANSLATION_PROVIDER", True, _provider_translation_env
    ),
    ProviderSource("shared", "--provider", False, _provider_shared),
    ProviderSource("stage_section", "[llm.<stage>]", True, _provider_stage_section),
    ProviderSource("defaults", "[llm]", False, _provider_defaults),
    ProviderSource("shared_env", "SPLITTER_PROVIDER", False, _provider_shared_env),
)


def _resolve_provider(
    stage: str,
    *,
    explicit: str | None,
    config: LLMConfig,
    overrides: Mapping[str, Any],
    shared: Mapping[str, Any],
    allowed: Collection[str] | None,
) -> tuple[str, bool]:
    """The stage's provider and whether a stage-specific source named it."""
    ctx = _ProviderContext(
        stage=stage,
        explicit=explicit,
        config=config,
        overrides=overrides,
        shared=shared,
    )
    for source in PROVIDER_SOURCES:
        name = _text(source.resolve(ctx))
        if name is None:
            continue
        name = name.lower()
        if allowed is not None and name not in allowed:
            if source.specific:
                raise ConfigError(
                    f"unknown {stage} provider {name!r} (from {source.label}); choose "
                    f"one of {', '.join(sorted(allowed))}"
                )
            continue
        return name, source.specific
    if allowed is not None:
        return DEFAULT_TRANSLATION_PROVIDER, False
    return "deepseek", False


@dataclass(slots=True)
class _FieldContext:
    stage: str
    explicit: Any
    config: LLMConfig
    overrides: Mapping[str, Any]
    shared: Mapping[str, Any]
    provider: str
    shared_ok: bool
    env_shared_ok: bool
    defaults_ok: bool
    stage_ok: bool


def _field_override(spec: FieldSpec, ctx: _FieldContext) -> Any:
    return ctx.overrides.get(spec.name, _MISSING)


def _field_explicit(spec: FieldSpec, ctx: _FieldContext) -> Any:
    return _MISSING if ctx.explicit is None else ctx.explicit


def _field_stage_env(spec: FieldSpec, ctx: _FieldContext) -> Any:
    for env_name in spec.env_names(ctx.stage, ctx.provider):
        value = _env_field(env_name, spec.name)
        if value is not None:
            return value
    return _MISSING


def _field_shared(spec: FieldSpec, ctx: _FieldContext) -> Any:
    value = ctx.shared.get(spec.name)
    return _MISSING if value is None else value


def _field_stage_section(spec: FieldSpec, ctx: _FieldContext) -> Any:
    section = ctx.config.stages.get(ctx.stage, {})
    return section.get(spec.name, _MISSING)


def _field_defaults(spec: FieldSpec, ctx: _FieldContext) -> Any:
    return ctx.config.defaults.get(spec.name, _MISSING)


def _field_shared_env(spec: FieldSpec, ctx: _FieldContext) -> Any:
    value = _env_field(f"SPLITTER_{spec.name.upper()}", spec.name)
    return _MISSING if value is None else value


def _field_builtin(spec: FieldSpec, ctx: _FieldContext) -> Any:
    return spec.default


@dataclass(frozen=True, slots=True)
class FieldSource:
    """One place a field value may come from, in resolution order."""

    key: str
    label: str
    #: The context attribute that gates this source for provider-scoped fields.
    gate: str | None
    resolve: Callable[[FieldSpec, _FieldContext], Any]


#: The field lookup order documented in the module docstring, README and
#: book_translator.toml. ``gate`` is None for sources that always apply; the
#: provider/section gates are only consulted for the fields marked for them in
#: :data:`FIELDS`, so a generic field still inherits from the broadest source.
FIELD_SOURCES: tuple[FieldSource, ...] = (
    FieldSource("override", "--llm <stage>.<field>", None, _field_override),
    FieldSource("explicit", "the stage's own flag", None, _field_explicit),
    FieldSource("stage_env", "stage-specific environment", None, _field_stage_env),
    FieldSource("shared", "shared pipeline flags", "shared_ok", _field_shared),
    FieldSource("stage_section", "[llm.<stage>]", "stage_ok", _field_stage_section),
    FieldSource("defaults", "[llm]", "defaults_ok", _field_defaults),
    FieldSource("shared_env", "legacy SPLITTER_* environment", "env_shared_ok", _field_shared_env),
    FieldSource("builtin", "the provider's built-in default", None, _field_builtin),
)


def _resolve_field(
    spec: FieldSpec,
    *,
    stage: str,
    explicit: Any,
    config: LLMConfig,
    overrides: Mapping[str, Any],
    shared: Mapping[str, Any],
    provider: str,
    shared_ok: bool,
    env_shared_ok: bool,
    defaults_ok: bool,
    stage_ok: bool,
) -> Any:
    """One field from the ordered sources in :data:`FIELD_SOURCES`."""
    ctx = _FieldContext(
        stage=stage,
        explicit=explicit,
        config=config,
        overrides=overrides,
        shared=shared,
        provider=provider,
        shared_ok=shared_ok,
        env_shared_ok=env_shared_ok,
        defaults_ok=defaults_ok,
        stage_ok=stage_ok,
    )
    for source in FIELD_SOURCES:
        if source.gate == "stage_ok":
            if spec.section_scoped and not ctx.stage_ok:
                continue
        elif (
            source.gate is not None
            and spec.provider_scoped
            and not getattr(ctx, source.gate)
        ):
            continue
        value = source.resolve(spec, ctx)
        if value is not _MISSING:
            return value
    return spec.default


def resolve_settings(
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    *,
    stage: str | None = None,
    config: LLMConfig | None = None,
    config_path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    shared: Mapping[str, Any] | None = None,
    allowed: Collection[str] | None = None,
    require_key: bool = True,
    discover_model: bool = True,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> LLMSettings:
    """Build one stage's :class:`LLMSettings` from flags, env, config and defaults.

    ``overrides`` are that stage's parsed ``--llm`` values (see
    :func:`stage_overrides`); ``shared`` holds the pipeline's global flags so they
    only reach a stage whose provider they were written for. ``allowed`` restricts
    the provider choice (stage 7 passes its two choices). ``require_key=False`` is
    used by the banners, which must not fail on a missing credential.
    """
    load_dotenv_files()
    canonical = _canonical_stage(stage)
    cfg = config if config is not None else load_llm_config(config_path)
    ov = dict(overrides or {})
    shared_data = dict(shared or {})
    defaults = cfg.defaults
    defaults_provider = _text(defaults.get("provider"))
    shared_provider = _text(shared_data.get("provider"))
    env_shared_provider = _env("SPLITTER_PROVIDER")

    resolved_provider, provider_specific = _resolve_provider(
        canonical,
        explicit=provider,
        config=cfg,
        overrides=ov,
        shared=shared_data,
        allowed=allowed,
    )
    shared_ok = _shared_field_ok(
        shared_provider, resolved_provider, provider_specific=provider_specific
    )
    env_shared_ok = _shared_field_ok(
        env_shared_provider, resolved_provider, provider_specific=provider_specific
    )
    defaults_ok = defaults_provider == resolved_provider

    # Stage 7 keeps its own table: LM Studio deliberately has no default model
    # (the loaded id is discovered), while the generic provider table names one.
    if canonical == "translate":
        spec = (
            TRANSLATION_PROVIDERS.get(resolved_provider)
            or PROVIDERS.get(resolved_provider)
            or {}
        )
    else:
        spec = PROVIDERS.get(resolved_provider) or {}

    # Only the connection fields belong to one provider: a shared model id or
    # endpoint must not leak into a stage pinned to another provider, while
    # temperature/timeouts are generic and inherit from the broadest source.
    # FieldSpec carries those two scopes, so the resolution loop stays generic.
    stage_section = cfg.stages.get(canonical, {})
    stage_provider = _text(stage_section.get("provider"))
    # A stage section's model/endpoint is written for one provider. When the
    # stage's provider is switched (CLI, env, or a provider in that section),
    # the section's model must not be handed to the new provider.
    if stage_provider is not None:
        stage_ok = stage_provider == resolved_provider
    else:
        stage_ok = defaults_provider is None or defaults_provider == resolved_provider

    def field(name: str, value: Any) -> Any:
        return _resolve_field(
            FIELD_BY_NAME[name],
            stage=canonical,
            explicit=value,
            config=cfg,
            overrides=ov,
            shared=shared_data,
            provider=resolved_provider,
            shared_ok=shared_ok,
            env_shared_ok=env_shared_ok,
            defaults_ok=defaults_ok,
            stage_ok=stage_ok,
        )

    resolved_base = field("base_url", base_url) or _text(spec.get("base_url")) or ""
    if not resolved_base:
        raise ConfigError(
            f"no base URL for {canonical} provider {resolved_provider!r}; set base_url "
            f"in {LLM_CONFIG_FILENAME}, or pass --base-url / --llm "
            f"{canonical}.base_url=... (known providers: {', '.join(sorted(PROVIDERS))})"
        )

    resolved_model = field("model", model) or _text(spec.get("model")) or ""
    if not resolved_model and resolved_provider == "lmstudio" and discover_model:
        resolved_model = pick_lmstudio_model(list_lmstudio_models(resolved_base))
    if not resolved_model and (discover_model or resolved_provider != "lmstudio"):
        if resolved_provider == "lmstudio":
            raise ConfigError(
                f"no model to translate with: LM Studio is not answering at "
                f"{resolved_base} (GET {resolved_base.rstrip('/')}/models). Start the "
                f"server, set [llm.{canonical}].model in {LLM_CONFIG_FILENAME}, or "
                "pass --model."
            )
        raise ConfigError(
            f"no model configured for {canonical} provider {resolved_provider!r}; set "
            f"[llm.{canonical}].model in {LLM_CONFIG_FILENAME}, or pass --model / --llm "
            f"{canonical}.model=..."
        )

    resolved_temperature = field("temperature", temperature)
    if resolved_temperature is None:
        resolved_temperature = FIELD_BY_NAME["temperature"].default
    resolved_timeout = field("timeout", timeout)
    if resolved_timeout is None:
        resolved_timeout = FIELD_BY_NAME["timeout"].default
    resolved_retries = field("max_retries", max_retries)
    if resolved_retries is None:
        resolved_retries = FIELD_BY_NAME["max_retries"].default

    resolved_key = resolve_api_key(
        resolved_provider,
        api_key,
        stage=canonical,
        config=cfg,
        overrides=ov,
        shared=shared_data,
        provider_specific=provider_specific,
    )
    if not resolved_key and not _is_local(resolved_base):
        if require_key:
            raise ConfigError(
                f"no API key found for {canonical} provider {resolved_provider!r}. Set "
                f"{spec.get('key_env', 'SPLITTER_API_KEY')} in the environment or in "
                f".env (keys do not belong in {LLM_CONFIG_FILENAME}), or pass --api-key / "
                f"--llm {canonical}.api_key_env=MY_KEY."
            )
        resolved_key = None
    return LLMSettings(
        provider=resolved_provider,
        model=resolved_model,
        base_url=resolved_base,
        api_key=resolved_key or "not-needed",
        temperature=float(resolved_temperature),
        timeout=float(resolved_timeout),
        max_retries=int(resolved_retries),
        stage=canonical,
        config_path=str(cfg.path) if cfg.path else None,
    )


def resolve_translation_settings(
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    *,
    discover_model: bool = True,
    config: LLMConfig | None = None,
    config_path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    shared: Mapping[str, Any] | None = None,
    require_key: bool = True,
) -> LLMSettings:
    """Build stage 7's connection: DeepSeek, or the local LM Studio server.

    The provider is restricted to :data:`TRANSLATION_PROVIDERS`. For LM Studio the
    model is resolved as explicit → ``TRANSLATION_MODEL`` / ``LMSTUDIO_MODEL`` →
    the running server's ``GET /v1/models`` (preferring a translation-tuned id).
    A local endpoint needs no key; a remote one still does.
    """
    return resolve_settings(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        stage="translate",
        config=config,
        config_path=config_path,
        overrides=overrides,
        shared=shared,
        allowed=TRANSLATION_PROVIDERS,
        require_key=require_key,
        discover_model=discover_model,
    )


# --------------------------------------------------------------------------- #
# labels (never raise, never need a key)
# --------------------------------------------------------------------------- #
def stage_model_label(
    stage: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    *,
    base_url: str | None = None,
    temperature: float | None = None,
    config: LLMConfig | None = None,
    config_path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    shared: Mapping[str, Any] | None = None,
    allowed: Collection[str] | None = None,
) -> str:
    """``deepseek/deepseek-flash`` or ``lmstudio/auto`` for a banner, never raises.

    ``auto`` means the model will be discovered from the running LM Studio server
    when the first unit is processed.
    """
    try:
        canonical = _canonical_stage(stage)
    except ConfigError:
        return str(stage or "—")
    try:
        resolved = resolve_settings(
            provider=provider,
            model=model,
            base_url=base_url,
            temperature=temperature,
            stage=canonical,
            config=config,
            config_path=config_path,
            overrides=overrides,
            shared=shared,
            allowed=allowed,
            require_key=False,
            discover_model=False,
        )
    except ConfigError:
        return str(provider or "—")
    return f"{resolved.provider}/{resolved.model or 'auto'}"


def translation_model_label(provider: str | None = None, model: str | None = None) -> str:
    """Stage 7's label, restricted to its two backends."""
    return stage_model_label(
        "translate",
        provider=provider,
        model=model,
        allowed=TRANSLATION_PROVIDERS,
    )


# --------------------------------------------------------------------------- #
# LM Studio model discovery
# --------------------------------------------------------------------------- #
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
    return [
        str(item["id"]) for item in data if isinstance(item, dict) and item.get("id")
    ]


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


__all__ = [
    "CREDENTIAL_SOURCES",
    "DEFAULT_DEEPSEEK_MODEL",
    "DEFAULT_STAGE",
    "DEFAULT_TRANSLATION_PROVIDER",
    "FIELD_BY_NAME",
    "FIELD_SOURCES",
    "FIELDS",
    "LLM_CONFIG_ENV",
    "LLM_CONFIG_FILENAME",
    "LLM_FIELDS",
    "LMSTUDIO_DISCOVERY_TIMEOUT",
    "PROVIDER_SOURCES",
    "PROVIDERS",
    "STAGES",
    "STAGE_ALIASES",
    "TRANSLATION_CHOICES",
    "TRANSLATION_PROVIDERS",
    "LLMConfig",
    "LLMSettings",
    "list_lmstudio_models",
    "llm_config_path",
    "load_dotenv_files",
    "load_llm_config",
    "normalize_stage",
    "parse_llm_overrides",
    "pick_lmstudio_model",
    "resolve_api_key",
    "resolve_settings",
    "resolve_translation_settings",
    "stage_model_label",
    "stage_overrides",
    "translation_model_label",
]
