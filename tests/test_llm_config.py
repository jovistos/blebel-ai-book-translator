"""Offline tests for the per-stage model config (``book_splitter.toml``).

The config file is the one place that says which model each pipeline stage uses;
the environment still holds the keys, and ``--llm`` / ``--llm-config`` override
one stage for one run. No test talks to a provider: discovery is stubbed and the
credentials are environment strings.
"""

from __future__ import annotations

import pytest
from book_splitter.core import settings as settings_module
from book_splitter.core.errors import ConfigError
from book_splitter.paths import repo_root

#: Every variable these tests may set, cleared before each one so the developer's
#: real environment cannot change the result.
_ENV_NAMES = (
    "SPLITTER_PROVIDER",
    "SPLITTER_MODEL",
    "SPLITTER_BASE_URL",
    "SPLITTER_API_KEY",
    "SPLITTER_TEMPERATURE",
    "SPLITTER_SPLIT_PROVIDER",
    "SPLITTER_SPLIT_MODEL",
    "SPLITTER_SPLIT_BASE_URL",
    "SPLITTER_SPLIT_API_KEY",
    "SPLITTER_TRANSLATE_MODEL",
    "SPLITTER_PREPROCESS_MODEL",
    "TRANSLATION_PROVIDER",
    "TRANSLATION_MODEL",
    "TRANSLATION_BASE_URL",
    "TRANSLATION_API_KEY",
    "LMSTUDIO_MODEL",
    "LMSTUDIO_BASE_URL",
    "LM_STUDIO_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "OLLAMA_API_KEY",
    "MY_TRANSLATION_KEY",
    "BOOK_SPLITTER_CONFIG",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """No project ``.env``, no DSH credentials, no real keys."""
    monkeypatch.setattr(settings_module, "load_dotenv_files", lambda: None)
    monkeypatch.setattr(settings_module, "_dsh_ref", lambda _name: None)
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def write_config(tmp_path, text: str):
    path = tmp_path / "book_splitter.toml"
    path.write_text(text, encoding="utf-8")
    return settings_module.load_llm_config(path)


BASE_CONFIG = """
[llm]
provider = "deepseek"
model = "deepseek-flash"
base_url = "https://api.deepseek.com/v1"
temperature = 0.0

[llm.split]
model = "deepseek-v4-pro"
temperature = 0.3

[llm.translate]
provider = "lmstudio"
base_url = "http://127.0.0.1:1234/v1"
"""


# --------------------------------------------------------------------------- #
# the file itself
# --------------------------------------------------------------------------- #
def test_the_project_file_parses_and_names_every_model_stage():
    loaded = settings_module.load_llm_config(repo_root() / "book_splitter.toml")

    assert loaded.path is not None
    assert loaded.defaults["provider"] == "deepseek"
    assert loaded.defaults["model"] == "deepseek-flash"
    # Every stage table is accepted (and may be empty until the user edits it).
    assert set(loaded.stages) <= set(settings_module.STAGES)
    assert "translate" in loaded.stages


def test_a_stage_section_overrides_the_shared_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    cfg = write_config(tmp_path, BASE_CONFIG)

    split = settings_module.resolve_settings(stage="split", config=cfg)
    preprocess = settings_module.resolve_settings(stage="preprocess", config=cfg)

    assert (split.provider, split.model) == ("deepseek", "deepseek-v4-pro")
    assert split.temperature == 0.3
    assert split.stage == "split"
    assert (preprocess.provider, preprocess.model) == ("deepseek", "deepseek-flash")
    assert preprocess.temperature == 0.0


def test_a_pipeline_override_beats_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    cfg = write_config(tmp_path, BASE_CONFIG)

    split = settings_module.resolve_settings(
        stage="split", config=cfg, overrides={"model": "custom-small"}
    )

    assert split.model == "custom-small"


def test_the_stage_environment_beats_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    cfg = write_config(tmp_path, BASE_CONFIG)
    monkeypatch.setenv("SPLITTER_SPLIT_MODEL", "env-model")

    split = settings_module.resolve_settings(stage="split", config=cfg)

    assert split.model == "env-model"


def test_the_legacy_shared_environment_is_only_a_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    cfg = write_config(tmp_path, BASE_CONFIG)
    monkeypatch.setenv("SPLITTER_MODEL", "legacy-model")
    monkeypatch.setenv("SPLITTER_PROVIDER", "deepseek")

    split = settings_module.resolve_settings(stage="split", config=cfg)

    # A config file is the deliberate choice; the old shared variable is below it.
    assert split.model == "deepseek-v4-pro"


def test_a_stage_that_switches_provider_does_not_inherit_the_cloud_model(tmp_path, monkeypatch):
    cfg = write_config(tmp_path, BASE_CONFIG)
    monkeypatch.setattr(
        settings_module,
        "list_lmstudio_models",
        lambda _base_url, timeout=0: ["local-translator"],
    )

    translation = settings_module.resolve_translation_settings(config=cfg)

    assert translation.provider == "lmstudio"
    assert translation.model == "local-translator"  # discovered, not "deepseek-flash"
    assert translation.base_url == "http://127.0.0.1:1234/v1"
    assert translation.api_key == "not-needed"


def test_api_key_env_names_the_stage_key(tmp_path, monkeypatch):
    cfg = write_config(
        tmp_path,
        """
[llm]
provider = "deepseek"
model = "deepseek-flash"

[llm.translate]
api_key_env = "MY_TRANSLATION_KEY"
""",
    )
    monkeypatch.setenv("MY_TRANSLATION_KEY", "sk-stage")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-provider")

    translation = settings_module.resolve_translation_settings(config=cfg)

    assert translation.api_key == "sk-stage"


def test_a_stage_may_name_a_keyless_local_provider(tmp_path):
    cfg = write_config(
        tmp_path,
        """
[llm]
provider = "deepseek"
model = "deepseek-flash"

[llm.preprocess]
provider = "ollama"
base_url = "http://127.0.0.1:11434/v1"
model = "qwen3:14b"
""",
    )

    preprocess = settings_module.resolve_settings(stage="preprocess", config=cfg)

    assert (preprocess.provider, preprocess.model) == ("ollama", "qwen3:14b")
    assert preprocess.api_key == "not-needed"


def test_an_unknown_key_or_stage_is_reported_as_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="unknown key 'modle'"):
        write_config(tmp_path, '[llm]\nprovider = "deepseek"\nmodle = "x"\n')

    with pytest.raises(ConfigError, match="unknown stage 'translater'"):
        write_config(tmp_path, "[llm]\n[llm.translater]\nmodel = \"x\"\n")


def test_a_missing_explicit_file_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        settings_module.load_llm_config(tmp_path / "missing.toml")


def test_llm_config_path_follows_the_env_and_explicit_path(tmp_path, monkeypatch):
    path = tmp_path / "other.toml"
    path.write_text("[llm]\nprovider = \"deepseek\"\n", encoding="utf-8")

    monkeypatch.setenv("BOOK_SPLITTER_CONFIG", str(path))
    assert settings_module.llm_config_path() == path
    assert settings_module.llm_config_path(tmp_path / "explicit.toml").name == "explicit.toml"


# --------------------------------------------------------------------------- #
# --llm parsing
# --------------------------------------------------------------------------- #
def test_parse_llm_overrides_names_the_stage_and_coerces_types():
    parsed = settings_module.parse_llm_overrides(
        [
            "split.model=deepseek-v4-pro",
            "translate.temperature=0.2",
            "translate.max_retries=5",
            "translate.provider=lmstudio",
        ]
    )

    assert parsed == {
        "split": {"model": "deepseek-v4-pro"},
        "translate": {"temperature": 0.2, "max_retries": 5, "provider": "lmstudio"},
    }


def test_parse_llm_overrides_accepts_a_command_default_stage():
    parsed = settings_module.parse_llm_overrides(["model=x"], default_stage="preprocess")

    assert parsed == {"preprocess": {"model": "x"}}


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (["nope.model=x"], "unknown stage"),
        (["split.modle=x"], "unknown field"),
        (["model=x"], "name the stage"),
        (["split.model"], "expects"),
    ],
)
def test_parse_llm_overrides_rejects_bad_input(values, message):
    with pytest.raises(ConfigError, match=message):
        settings_module.parse_llm_overrides(values)


def test_stage_aliases_are_understood():
    assert settings_module.normalize_stage("glossary") == "dictionary"
    assert settings_module.normalize_stage("translation") == "translate"
    assert settings_module.normalize_stage("post-process") == "postprocess"
    assert settings_module.normalize_stage("nope") is None


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #
def test_the_labels_read_the_stage_sections(tmp_path):
    cfg = write_config(tmp_path, BASE_CONFIG)

    assert (
        settings_module.stage_model_label("split", config=cfg)
        == "deepseek/deepseek-v4-pro"
    )
    assert (
        settings_module.stage_model_label(
            "translate", config=cfg, allowed=settings_module.TRANSLATION_CHOICES
        )
        == "lmstudio/auto"
    )


# --------------------------------------------------------------------------- #
# the command line
# --------------------------------------------------------------------------- #
def test_the_cli_accepts_a_bare_field_for_this_commands_stage():
    from book_splitter import cli

    args = cli.build_parser().parse_args(["translate", "--llm", "model=x"])

    assert args.llm == ["model=x"]
    assert args.llm_config is None


def test_the_cli_rejects_a_bad_llm_switch_before_it_needs_a_model(capsys):
    from book_splitter import cli

    code = cli.main(["split", "whatever.md", "--llm", "nope.model=x"])

    assert code == 2
    assert "unknown stage 'nope'" in capsys.readouterr().err


def test_generic_settings_inherit_even_when_a_stage_switches_provider(tmp_path, monkeypatch):
    cfg = write_config(
        tmp_path,
        """
[llm]
provider = "deepseek"
model = "deepseek-flash"
temperature = 0.15
timeout = 30.0

[llm.translate]
provider = "lmstudio"
model = "local-translator"
""",
    )

    translation = settings_module.resolve_translation_settings(config=cfg)

    # The connection stays lmstudio's; temperature/timeout are not provider-bound.
    assert translation.provider == "lmstudio"
    assert translation.temperature == 0.15
    assert translation.timeout == 30.0


def test_the_glossary_alias_loads_as_the_dictionary_stage(tmp_path):
    cfg = write_config(
        tmp_path,
        """
[llm]
provider = "deepseek"
model = "deepseek-flash"

[llm.glossary]
model = "gloss-model"
""",
    )

    assert cfg.stages["dictionary"]["model"] == "gloss-model"
    assert "glossary" not in cfg.stages


def test_deepseek_flash_is_the_default_in_both_provider_tables():
    assert settings_module.DEFAULT_DEEPSEEK_MODEL == "deepseek-flash"
    assert settings_module.PROVIDERS["deepseek"]["model"] == "deepseek-flash"
    assert settings_module.TRANSLATION_PROVIDERS["deepseek"]["model"] == "deepseek-flash"


def test_the_project_config_pins_deepseek_flash_in_every_stage():
    loaded = settings_module.load_llm_config(repo_root() / "book_splitter.toml")

    assert loaded.defaults["model"] == "deepseek-flash"
    for stage in settings_module.STAGES:
        assert loaded.stages[stage]["model"] == "deepseek-flash"
        resolved = settings_module.resolve_settings(
            stage=stage, config=loaded, require_key=False
        )
        assert (resolved.provider, resolved.model) == ("deepseek", "deepseek-flash")


def test_a_stage_model_does_not_follow_a_provider_switch(tmp_path, monkeypatch):
    """An explicit stage model belongs to the config's provider, not to a new one."""
    cfg = write_config(
        tmp_path,
        """
[llm]
provider = "deepseek"
model = "deepseek-flash"

[llm.translate]
model = "deepseek-flash"
""",
    )
    monkeypatch.setattr(
        settings_module,
        "list_lmstudio_models",
        lambda _base_url, timeout=0: ["local-translator"],
    )

    translation = settings_module.resolve_translation_settings(
        provider="lmstudio", config=cfg
    )

    assert translation.provider == "lmstudio"
    assert translation.model == "local-translator"
