"""The model/credential resolution is a precedence table, and it is tested as one.

The old tests checked a handful of pairs by hand. These tests walk the declared
tables in [book_translator.core.settings](book_translator/core/settings.py) so a new
source inserted without a test — or a source moved without updating the docs —
fails here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from book_translator.core import settings as S
from book_translator.core.errors import ConfigError

#: Every variable these tests may set, cleared before each one.
_ENV_NAMES = (
    "SPLITTER_PROVIDER",
    "SPLITTER_MODEL",
    "SPLITTER_BASE_URL",
    "SPLITTER_TEMPERATURE",
    "SPLITTER_TIMEOUT",
    "SPLITTER_MAX_RETRIES",
    "SPLITTER_API_KEY_ENV",
    "SPLITTER_API_KEY",
    "SPLITTER_TRANSLATE_PROVIDER",
    "SPLITTER_TRANSLATE_MODEL",
    "SPLITTER_TRANSLATE_BASE_URL",
    "SPLITTER_TRANSLATE_TEMPERATURE",
    "SPLITTER_TRANSLATE_TIMEOUT",
    "SPLITTER_TRANSLATE_MAX_RETRIES",
    "SPLITTER_TRANSLATE_API_KEY_ENV",
    "SPLITTER_TRANSLATE_API_KEY",
    "TRANSLATION_PROVIDER",
    "TRANSLATION_MODEL",
    "TRANSLATION_BASE_URL",
    "TRANSLATION_TEMPERATURE",
    "TRANSLATION_TIMEOUT",
    "TRANSLATION_MAX_RETRIES",
    "TRANSLATION_API_KEY_ENV",
    "TRANSLATION_API_KEY",
    "LMSTUDIO_MODEL",
    "LMSTUDIO_BASE_URL",
    "LM_STUDIO_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "OLLAMA_API_KEY",
    "MY_OVERRIDE_KEY",
    "MY_STAGE_KEY",
    "MY_DEFAULT_KEY",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    monkeypatch.setattr(S, "load_dotenv_files", lambda: None)
    monkeypatch.setattr(S, "_dsh_ref", lambda _name: None)
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def _toml(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    return repr(value)


def _config(
    tmp_path: Path,
    *,
    defaults: dict[str, Any] | None = None,
    stage: dict[str, Any] | None = None,
) -> S.LLMConfig:
    lines = [
        "[llm]",
        'provider = "lmstudio"',
    ]
    for key, value in (defaults or {}).items():
        lines.append(f"{key} = {_toml(value)}")
    if stage:
        lines.append("[llm.translate]")
        for key, value in stage.items():
            lines.append(f"{key} = {_toml(value)}")
    path = tmp_path / "book_translator.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return S.load_llm_config(path)


def _field_value(spec: S.FieldSpec, index: int) -> Any:
    if spec.kind is str:
        return f"{spec.name}-source-{index}"
    if spec.kind is float:
        return 10.0 + index
    return 100 + index


def _resolve_field(
    spec: S.FieldSpec,
    config: S.LLMConfig,
    *,
    overrides: dict[str, Any] | None = None,
    explicit: Any = None,
    shared: dict[str, Any] | None = None,
    shared_ok: bool = True,
    env_shared_ok: bool = True,
    defaults_ok: bool = True,
    stage_ok: bool = True,
) -> Any:
    return S._resolve_field(
        spec,
        stage="translate",
        explicit=explicit,
        config=config,
        overrides=overrides or {},
        shared=shared or {},
        provider="lmstudio",
        shared_ok=shared_ok,
        env_shared_ok=env_shared_ok,
        defaults_ok=defaults_ok,
        stage_ok=stage_ok,
    )


def _set_field_source(
    source: S.FieldSource,
    spec: S.FieldSpec,
    value: Any,
    state: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if source.key == "override":
        state["overrides"][spec.name] = value
    elif source.key == "explicit":
        state["explicit"] = value
    elif source.key == "stage_env":
        monkeypatch.setenv(spec.env_names("translate", "lmstudio")[0], str(value))
    elif source.key == "shared":
        state["shared"][spec.name] = value
    elif source.key == "stage_section":
        state["stage"][spec.name] = value
    elif source.key == "defaults":
        state["defaults"][spec.name] = value
    elif source.key == "shared_env":
        monkeypatch.setenv(f"SPLITTER_{spec.name.upper()}", str(value))
    elif source.key == "builtin":
        return
    else:  # pragma: no cover - a new table row must be handled here
        raise AssertionError(f"untested field source: {source.key}")


# --------------------------------------------------------------------------- #
# the declared tables
# --------------------------------------------------------------------------- #
def test_the_field_table_is_the_documented_order() -> None:
    assert [source.key for source in S.FIELD_SOURCES] == [
        "override",
        "explicit",
        "stage_env",
        "shared",
        "stage_section",
        "defaults",
        "shared_env",
        "builtin",
    ]
    assert S.LLM_FIELDS == (
        "provider",
        "model",
        "base_url",
        "temperature",
        "timeout",
        "max_retries",
        "api_key_env",
    )


def test_the_provider_table_is_the_documented_order() -> None:
    assert [source.key for source in S.PROVIDER_SOURCES] == [
        "override",
        "explicit",
        "stage_env",
        "translation_env",
        "shared",
        "stage_section",
        "defaults",
        "shared_env",
    ]


def test_the_credential_table_is_the_documented_order() -> None:
    assert [source.key for source in S.CREDENTIAL_SOURCES] == [
        "explicit",
        "override_key",
        "override_key_env",
        "shared_key",
        "stage_env",
        "stage_section",
        "defaults",
        "provider_env",
        "legacy_shared",
        "dsh",
    ]


def test_every_field_has_a_kind_and_a_default() -> None:
    assert S.FIELD_BY_NAME["temperature"].default == 0.0
    assert S.FIELD_BY_NAME["timeout"].default == 180.0
    assert S.FIELD_BY_NAME["max_retries"].default == 3
    assert S.FIELD_BY_NAME["model"].provider_scoped
    assert S.FIELD_BY_NAME["base_url"].section_scoped
    assert not S.FIELD_BY_NAME["temperature"].provider_scoped


# --------------------------------------------------------------------------- #
# field precedence: every adjacent pair, for every field
# --------------------------------------------------------------------------- #
_FIELDS = ("model", "base_url", "temperature", "timeout", "max_retries", "api_key_env")
_PAIRS = list(range(len(S.FIELD_SOURCES) - 1))


@pytest.mark.parametrize("field_name", _FIELDS)
@pytest.mark.parametrize("rank", _PAIRS, ids=[s.key for s in S.FIELD_SOURCES[:-1]])
def test_a_field_source_beats_the_next_one(
    field_name: str, rank: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = S.FIELD_BY_NAME[field_name]
    higher = S.FIELD_SOURCES[rank]
    lower = S.FIELD_SOURCES[rank + 1]
    state: dict[str, Any] = {
        "overrides": {},
        "explicit": None,
        "shared": {},
        "defaults": {},
        "stage": {},
    }
    higher_value = _field_value(spec, rank)
    lower_value = _field_value(spec, rank + 1)
    _set_field_source(higher, spec, higher_value, state, monkeypatch)
    _set_field_source(lower, spec, lower_value, state, monkeypatch)
    config = _config(tmp_path, defaults=state["defaults"], stage=state["stage"])

    resolved = _resolve_field(
        spec,
        config,
        overrides=state["overrides"],
        explicit=state["explicit"],
        shared=state["shared"],
    )

    assert resolved == higher_value, f"{higher.key} must beat {lower.key}"


@pytest.mark.parametrize("field_name", _FIELDS)
def test_the_builtin_default_is_the_last_resort(field_name: str, tmp_path: Path) -> None:
    spec = S.FIELD_BY_NAME[field_name]
    config = _config(tmp_path)

    assert _resolve_field(spec, config) == spec.default


# --------------------------------------------------------------------------- #
# provider precedence
# --------------------------------------------------------------------------- #
def _provider_config(
    tmp_path: Path,
    *,
    defaults: str | None = None,
    stage: str | None = None,
) -> S.LLMConfig:
    lines = ["[llm]"]
    if defaults:
        lines.append(f'provider = "{defaults}"')
    else:
        lines.append('provider = "deepseek"')
    if stage:
        lines += ["[llm.translate]", f'provider = "{stage}"']
    path = tmp_path / "book_translator.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return S.load_llm_config(path)


def _resolve_provider(
    config: S.LLMConfig,
    *,
    explicit: str | None = None,
    overrides: dict[str, Any] | None = None,
    shared: dict[str, Any] | None = None,
) -> str:
    name, _specific = S._resolve_provider(
        "translate",
        explicit=explicit,
        config=config,
        overrides=overrides or {},
        shared=shared or {},
        allowed=None,
    )
    return name


@pytest.mark.parametrize("rank", _PAIRS, ids=[s.key for s in S.PROVIDER_SOURCES[:-1]])
def test_a_provider_source_beats_the_next_one(
    rank: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    higher = S.PROVIDER_SOURCES[rank]
    lower = S.PROVIDER_SOURCES[rank + 1]
    higher_name = f"higher-{rank}"
    lower_name = f"lower-{rank + 1}"
    overrides: dict[str, Any] = {}
    explicit = None
    shared: dict[str, Any] = {}
    defaults = "deepseek"
    stage = None

    if higher.key == "override":
        overrides["provider"] = higher_name
    elif higher.key == "explicit":
        explicit = higher_name
    elif higher.key == "stage_env":
        monkeypatch.setenv("SPLITTER_TRANSLATE_PROVIDER", higher_name)
    elif higher.key == "translation_env":
        monkeypatch.setenv("TRANSLATION_PROVIDER", higher_name)
    elif higher.key == "shared":
        shared["provider"] = higher_name
    elif higher.key == "stage_section":
        stage = higher_name
    elif higher.key == "defaults":
        defaults = higher_name
    elif higher.key == "shared_env":
        monkeypatch.setenv("SPLITTER_PROVIDER", higher_name)

    if lower.key == "override":
        overrides["provider"] = lower_name
    elif lower.key == "explicit":
        explicit = lower_name
    elif lower.key == "stage_env":
        monkeypatch.setenv("SPLITTER_TRANSLATE_PROVIDER", lower_name)
    elif lower.key == "translation_env":
        monkeypatch.setenv("TRANSLATION_PROVIDER", lower_name)
    elif lower.key == "shared":
        shared["provider"] = lower_name
    elif lower.key == "stage_section":
        stage = lower_name
    elif lower.key == "defaults":
        defaults = lower_name
    elif lower.key == "shared_env":
        monkeypatch.setenv("SPLITTER_PROVIDER", lower_name)

    config = _provider_config(tmp_path, defaults=defaults, stage=stage)
    assert _resolve_provider(config, explicit=explicit, overrides=overrides, shared=shared) == higher_name


def test_the_provider_default_is_deepseek(tmp_path: Path) -> None:
    assert _resolve_provider(_provider_config(tmp_path)) == "deepseek"


def test_an_unsupported_specific_provider_is_an_error(tmp_path: Path) -> None:
    config = _provider_config(tmp_path)

    with pytest.raises(ConfigError, match="unknown translate provider"):
        S._resolve_provider(
            "translate",
            explicit="openai",
            config=config,
            overrides={},
            shared={},
            allowed=S.TRANSLATION_PROVIDERS,
        )


def test_an_unsupported_broad_provider_is_ignored(tmp_path: Path) -> None:
    config = _provider_config(tmp_path, defaults="deepseek")

    # A shared provider that stage 7 does not offer is skipped, not fatal.
    assert (
        S._resolve_provider(
            "translate",
            explicit=None,
            config=config,
            overrides={},
            shared={"provider": "ollama"},
            allowed=S.TRANSLATION_PROVIDERS,
        )[0]
        == "deepseek"
    )


# --------------------------------------------------------------------------- #
# credentials
# --------------------------------------------------------------------------- #
def _credential_config(
    tmp_path: Path,
    *,
    stage_key_env: str | None = None,
    default_key_env: str | None = None,
) -> S.LLMConfig:
    lines = ["[llm]", 'provider = "deepseek"']
    if default_key_env:
        lines.append(f'api_key_env = "{default_key_env}"')
    if stage_key_env:
        lines += ["[llm.translate]", f'api_key_env = "{stage_key_env}"']
    path = tmp_path / "book_translator.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return S.load_llm_config(path)


def _resolve_key(
    config: S.LLMConfig,
    *,
    explicit: str | None = None,
    overrides: dict[str, Any] | None = None,
    shared: dict[str, Any] | None = None,
) -> str | None:
    return S.resolve_api_key(
        "deepseek",
        explicit,
        stage="translate",
        config=config,
        overrides=overrides or {},
        shared=shared or {},
        provider_specific=True,
    )


def _set_credential_source(
    source: S.CredentialSource,
    value: str,
    state: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if source.key == "explicit":
        state["explicit"] = value
    elif source.key == "override_key":
        state["overrides"]["api_key"] = value
    elif source.key == "override_key_env":
        state["overrides"]["api_key_env"] = "MY_OVERRIDE_KEY"
        monkeypatch.setenv("MY_OVERRIDE_KEY", value)
    elif source.key == "shared_key":
        state["shared"]["api_key"] = value
    elif source.key == "stage_env":
        monkeypatch.setenv("SPLITTER_TRANSLATE_API_KEY", value)
    elif source.key == "stage_section":
        state["stage_key_env"] = "MY_STAGE_KEY"
        monkeypatch.setenv("MY_STAGE_KEY", value)
    elif source.key == "defaults":
        state["default_key_env"] = "MY_DEFAULT_KEY"
        monkeypatch.setenv("MY_DEFAULT_KEY", value)
    elif source.key == "provider_env":
        monkeypatch.setenv("DEEPSEEK_API_KEY", value)
    elif source.key == "legacy_shared":
        monkeypatch.setenv("SPLITTER_API_KEY", value)
    elif source.key == "dsh":
        monkeypatch.setattr(S, "_dsh_ref", lambda _name: value)
    else:  # pragma: no cover - a new table row must be handled here
        raise AssertionError(f"untested credential source: {source.key}")


@pytest.mark.parametrize("rank", list(range(len(S.CREDENTIAL_SOURCES) - 1)), ids=[s.key for s in S.CREDENTIAL_SOURCES[:-1]])
def test_a_credential_source_beats_the_next_one(
    rank: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPLITTER_PROVIDER", "deepseek")
    higher = S.CREDENTIAL_SOURCES[rank]
    lower = S.CREDENTIAL_SOURCES[rank + 1]
    state: dict[str, Any] = {
        "explicit": None,
        "overrides": {},
        "shared": {"provider": "deepseek"},
        "stage_key_env": None,
        "default_key_env": None,
    }
    higher_value = f"key-{higher.key}"
    lower_value = f"key-{lower.key}"
    _set_credential_source(higher, higher_value, state, monkeypatch)
    _set_credential_source(lower, lower_value, state, monkeypatch)
    config = _credential_config(
        tmp_path,
        stage_key_env=state["stage_key_env"],
        default_key_env=state["default_key_env"],
    )

    resolved = _resolve_key(
        config,
        explicit=state["explicit"],
        overrides=state["overrides"],
        shared=state["shared"],
    )

    assert resolved == higher_value, f"{higher.key} must beat {lower.key}"


def test_no_credential_source_returns_none(tmp_path: Path) -> None:
    assert _resolve_key(_credential_config(tmp_path)) is None


def test_the_dsh_store_is_the_last_credential_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(S, "_dsh_ref", lambda _name: "key-from-dsh")

    assert _resolve_key(_credential_config(tmp_path)) == "key-from-dsh"


# --------------------------------------------------------------------------- #
# provider/section gating
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("gate", "spec_name", "value"),
    [
        ("shared_ok", "model", "shared-model"),
        ("env_shared_ok", "model", "env-model"),
        ("defaults_ok", "model", "default-model"),
        ("stage_ok", "model", "stage-model"),
    ],
)
def test_a_gated_source_does_not_cross_a_provider_for_connection_fields(
    gate: str,
    spec_name: str,
    value: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = S.FIELD_BY_NAME[spec_name]
    config = _config(
        tmp_path,
        defaults={"model": value} if gate == "defaults_ok" else None,
        stage={"model": value} if gate == "stage_ok" else None,
    )
    overrides: dict[str, Any] = {}
    shared = {"model": value} if gate == "shared_ok" else {}
    if gate == "env_shared_ok":
        monkeypatch.setenv("SPLITTER_MODEL", value)

    resolved = _resolve_field(
        spec,
        config,
        overrides=overrides,
        shared=shared,
        **{gate: False},
    )

    assert resolved == spec.default


def test_generic_fields_ignore_the_provider_gates(tmp_path: Path) -> None:
    spec = S.FIELD_BY_NAME["temperature"]
    config = _config(tmp_path)
    shared = {"temperature": 0.25}

    resolved = _resolve_field(
        spec,
        config,
        shared=shared,
        shared_ok=False,
        env_shared_ok=False,
        defaults_ok=False,
        stage_ok=False,
    )

    assert resolved == 0.25
