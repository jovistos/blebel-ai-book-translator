"""Offline tests for stage 7's two model choices: DeepSeek and LM Studio.

The translation stage is deliberately narrower than the rest of the pipeline: it
offers exactly these two providers, and its ``TRANSLATION_*`` settings do not let
a DeepSeek ``.env`` leak a cloud model id or URL into a local run. No test talks
to LM Studio; discovery is a stub.
"""

from __future__ import annotations

import pytest
from book_splitter.core import settings as settings_module

#: Every variable these tests may set, cleared before each one so the developer's
#: real environment cannot change the result.
_ENV_NAMES = (
    "SPLITTER_PROVIDER",
    "SPLITTER_MODEL",
    "SPLITTER_BASE_URL",
    "SPLITTER_API_KEY",
    "TRANSLATION_PROVIDER",
    "TRANSLATION_MODEL",
    "TRANSLATION_BASE_URL",
    "TRANSLATION_API_KEY",
    "LMSTUDIO_MODEL",
    "LMSTUDIO_BASE_URL",
    "LM_STUDIO_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """A blank environment: no project ``.env``, no DSH credentials, no keys."""
    monkeypatch.setattr(settings_module, "load_dotenv_files", lambda: None)
    monkeypatch.setattr(settings_module, "_dsh_ref", lambda _name: None)
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_the_translation_stage_defaults_to_deepseek(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    resolved = settings_module.resolve_translation_settings()

    assert resolved.provider == "deepseek"
    assert resolved.model == "deepseek-flash"
    assert resolved.base_url == "https://api.deepseek.com/v1"
    assert resolved.api_key == "sk-test"


def test_lmstudio_is_the_second_option_and_needs_no_key(monkeypatch):
    monkeypatch.setattr(
        settings_module,
        "list_lmstudio_models",
        lambda _base_url, timeout=0: ["gemma-4-31b-en2sr-v3@q4_k_m", "qwen/qwen3.8-27b"],
    )

    resolved = settings_module.resolve_translation_settings(provider="lmstudio")

    assert resolved.provider == "lmstudio"
    assert resolved.model == "gemma-4-31b-en2sr-v3@q4_k_m"
    assert resolved.base_url == "http://127.0.0.1:1234/v1"
    assert resolved.api_key == "not-needed"
    assert resolved.is_local


def test_a_translation_tuned_model_is_preferred_when_several_are_loaded():
    picked = settings_module.pick_lmstudio_model(
        ["qwen/qwen3.8-27b", "gemma-4-31b-en2sr-v3@q4_k_m", "google/gemma-4-12b"]
    )
    assert picked == "gemma-4-31b-en2sr-v3@q4_k_m"
    assert settings_module.pick_lmstudio_model([]) == ""
    # An embedding model listed next to the chat model is never picked.
    assert (
        settings_module.pick_lmstudio_model(
            ["text-embedding-nomic-embed-text-v1.5", "qwen/qwen3.8-27b"]
        )
        == "qwen/qwen3.8-27b"
    )
    assert settings_module.pick_lmstudio_model(["text-embedding-nomic"]) == ""


def test_the_splitter_model_does_not_leak_into_lmstudio(monkeypatch):
    monkeypatch.setenv("SPLITTER_PROVIDER", "deepseek")
    monkeypatch.setenv("SPLITTER_MODEL", "deepseek-flash")
    monkeypatch.setenv("SPLITTER_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("SPLITTER_API_KEY", "sk-deepseek")
    monkeypatch.setattr(
        settings_module,
        "list_lmstudio_models",
        lambda _base_url, timeout=0: ["local-translator"],
    )

    resolved = settings_module.resolve_translation_settings(provider="lmstudio")

    assert resolved.model == "local-translator"
    assert resolved.base_url == "http://127.0.0.1:1234/v1"
    # The cloud key must not be sent to a localhost endpoint.
    assert resolved.api_key == "not-needed"


def test_translation_variables_override_the_shared_settings(monkeypatch):
    monkeypatch.setenv("SPLITTER_PROVIDER", "deepseek")
    monkeypatch.setenv("SPLITTER_MODEL", "deepseek-flash")
    monkeypatch.setenv("TRANSLATION_PROVIDER", "lmstudio")
    monkeypatch.setenv("TRANSLATION_MODEL", "my-local-model")
    monkeypatch.setenv("TRANSLATION_BASE_URL", "http://127.0.0.1:9999/v1")

    resolved = settings_module.resolve_translation_settings()

    assert resolved.provider == "lmstudio"
    assert resolved.model == "my-local-model"
    assert resolved.base_url == "http://127.0.0.1:9999/v1"


def test_a_shared_provider_outside_the_two_choices_is_ignored(monkeypatch):
    monkeypatch.setenv("SPLITTER_PROVIDER", "ollama")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    resolved = settings_module.resolve_translation_settings()

    assert resolved.provider == "deepseek"


def test_a_dead_lmstudio_server_is_a_clear_config_error(monkeypatch):
    monkeypatch.setattr(
        settings_module, "list_lmstudio_models", lambda _base_url, timeout=0: []
    )

    with pytest.raises(settings_module.ConfigError) as caught:
        settings_module.resolve_translation_settings(provider="lmstudio")

    message = str(caught.value)
    assert "LM Studio is not answering" in message
    assert "/v1/models" in message
    assert "--model" in message


def test_deepseek_still_requires_a_key():
    with pytest.raises(settings_module.ConfigError) as caught:
        settings_module.resolve_translation_settings(provider="deepseek")

    assert "no API key" in str(caught.value)


def test_an_unknown_translation_provider_is_rejected():
    with pytest.raises(settings_module.ConfigError) as caught:
        settings_module.resolve_translation_settings(provider="openai")

    assert "deepseek" in str(caught.value) and "lmstudio" in str(caught.value)


def test_the_label_says_auto_when_lmstudio_will_be_discovered(monkeypatch):
    assert settings_module.translation_model_label("deepseek") == "deepseek/deepseek-flash"
    assert settings_module.translation_model_label("lmstudio") == "lmstudio/auto"
    assert (
        settings_module.translation_model_label("lmstudio", "my-model")
        == "lmstudio/my-model"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
