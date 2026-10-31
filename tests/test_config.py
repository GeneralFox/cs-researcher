import json

import pytest

from cs_researcher.config import (
    AppConfig,
    ConfigError,
    DashboardSettings,
    LLMSettings,
    OllamaSettings,
    ResearchSettings,
    TelemetrySettings,
)

MINIMAL = {"coordinator": {"model": "big:12b"}, "analysts": [{"model": "small-1:4b"}, {"model": "small.2:4b"}]}


def write(tmp_path, data) -> AppConfig:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    return AppConfig.load(path)


def test_shipped_config_matches_the_code_defaults():
    # a custom file can leave sections out: they must equal the shipped ones
    config = AppConfig.load()
    assert config.ollama == OllamaSettings()
    assert config.llm_defaults == LLMSettings()
    assert config.research == ResearchSettings()
    assert config.telemetry == TelemetrySettings()
    assert config.dashboard == DashboardSettings()


def test_minimal_config(tmp_path):
    config = write(tmp_path, MINIMAL)
    assert config.coordinator.role == "research_coordinator"
    assert [a.role for a in config.analysts] == ["analyst_small_1_4b", "analyst_small_2_4b"]


def test_llm_defaults_apply_unless_overridden(tmp_path):
    data = MINIMAL | {
        "llm_defaults": {"num_ctx": 8192, "temperature": 0.5},
        "analysts": [{"model": "small:4b", "temperature": 0.0}],
    }
    [analyst] = write(tmp_path, data).analysts
    assert (analyst.num_ctx, analyst.temperature, analyst.max_tokens) == (8192, 0.0, LLMSettings().max_tokens)


def test_unknown_key_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="num_ctxx"):
        write(tmp_path, MINIMAL | {"llm_defaults": {"num_ctxx": 8192}})


def test_agent_names_must_be_unique(tmp_path):
    with pytest.raises(ConfigError, match="analyst_small_4b"):
        write(tmp_path, MINIMAL | {"analysts": [{"model": "small:4b"}, {"model": "small-4b"}]})


def test_unreadable_and_invalid_files(tmp_path):
    with pytest.raises(ConfigError, match="cannot read"):
        AppConfig.load(tmp_path / "missing.json")
    (tmp_path / "bad.json").write_text("{")
    with pytest.raises(ConfigError, match="not valid JSON"):
        AppConfig.load(tmp_path / "bad.json")


def test_updated_ignores_none():
    config = AppConfig.load()
    assert config.updated("research", max_analyst_calls=None) is config
    assert config.updated("research", max_analyst_calls=3).research.max_analyst_calls == 3


def test_to_json_round_trips(tmp_path):
    config = AppConfig.load()
    assert write(tmp_path, json.loads(config.to_json())) == config
