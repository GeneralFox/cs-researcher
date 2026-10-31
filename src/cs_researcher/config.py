"""Configuration: a JSON file validated into typed, read-only settings.

The package ships a default config.json; `--config path.json` replaces it.
Sections left out of a custom file take the defaults below, which match the
shipped file. `cs-researcher config` prints the configuration in use, a
starting point for a custom file.
"""

import json
import re
from importlib import resources
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

DEFAULT_CONFIG = resources.files(__package__).joinpath("config.json")


class ConfigError(Exception):
    """The configuration file is missing, not valid JSON, or has invalid settings."""


class Settings(BaseModel):
    # a misspelt key is an error, not a setting silently ignored
    model_config = ConfigDict(extra="forbid", frozen=True)


class OllamaSettings(Settings):
    base_url: str = "http://localhost:11434"


class LLMSettings(Settings):
    """Generation settings: `llm_defaults`, overridable per agent."""

    num_ctx: int = Field(16384, gt=0)
    temperature: float = Field(0.1, ge=0)
    # hard cap on every answer: the coordinator is asked for ~800 words
    # (~1100 tokens), so this only cuts a model stuck repeating itself
    max_tokens: int = Field(3000, gt=0)
    # seconds per model call, model load included
    timeout: float = Field(420, gt=0)
    num_retries: int = Field(2, ge=0)
    think: bool = False


class AgentSettings(LLMSettings):
    """An agent's model, with any LLMSettings key overriding llm_defaults."""

    model: str
    name: str | None = None

    @property
    def role(self) -> str:
        # "nemotron-3.5-bolt:e4b" -> "analyst_nemotron_3_5_bolt_e4b"
        return self.name or "analyst_" + re.sub(r"[^0-9a-zA-Z]", "_", self.model)


class CoordinatorSettings(AgentSettings):
    name: str = "research_coordinator"


class ResearchSettings(Settings):
    # a healthy analyst uses 3-6 model calls: the cap stops one stuck
    # repeating the same tool call
    max_analyst_calls: int = Field(12, gt=0)
    search_results: int = Field(10, gt=0)
    max_page_chars: int = Field(8000, gt=0)
    fetch_timeout: float = Field(20, gt=0)
    trusted_sites: tuple[str, ...] = Field(
        (
            "nist.gov",
            "cisa.gov",
            "attack.mitre.org",
            "isc.sans.edu",
            "bleepingcomputer.com",
            "thehackernews.com",
        ),
        min_length=1,
    )


class TelemetrySettings(Settings):
    enabled: bool = True
    db_path: Path = Path("telemetry.db")  # relative to the working directory


class DashboardSettings(Settings):
    host: str = "127.0.0.1"
    port: int = Field(8765, ge=0, le=65535)


class AppConfig(Settings):
    ollama: OllamaSettings = OllamaSettings()
    llm_defaults: LLMSettings = LLMSettings()
    coordinator: CoordinatorSettings
    analysts: tuple[AgentSettings, ...] = Field(min_length=1)
    research: ResearchSettings = ResearchSettings()
    telemetry: TelemetrySettings = TelemetrySettings()
    dashboard: DashboardSettings = DashboardSettings()

    @model_validator(mode="before")
    @classmethod
    def _apply_llm_defaults(cls, data: Any) -> Any:
        """Fill each agent's unset LLM settings from llm_defaults."""
        if not isinstance(data, dict):
            return data
        defaults = data.get("llm_defaults") or {}
        if not isinstance(defaults, dict):
            return data
        data = dict(data)
        if isinstance(data.get("coordinator"), dict):
            data["coordinator"] = defaults | data["coordinator"]
        if isinstance(data.get("analysts"), list):
            data["analysts"] = [defaults | a if isinstance(a, dict) else a for a in data["analysts"]]
        return data

    @model_validator(mode="after")
    def _unique_roles(self) -> Self:
        # the coordinator calls each analyst by its role: names must differ
        roles = [self.coordinator.role, *(a.role for a in self.analysts)]
        duplicates = sorted({r for r in roles if roles.count(r) > 1})
        if duplicates:
            raise ValueError(f"agent names must be unique, repeated: {', '.join(duplicates)}")
        return self

    @classmethod
    def load(cls, path: Path | None = None) -> Self:
        """Load a config file, or the shipped default without a path."""
        source = path or DEFAULT_CONFIG
        try:
            return cls.model_validate(json.loads(source.read_text()))
        except OSError as e:
            raise ConfigError(f"cannot read {source}: {e.strerror}") from e
        except json.JSONDecodeError as e:
            raise ConfigError(f"{source} is not valid JSON: {e}") from e
        except ValidationError as e:
            raise ConfigError(f"{source}: invalid configuration\n{e}") from e

    def updated(self, section: str, **values: Any) -> Self:
        """A copy with some values of a section replaced (e.g. by CLI options)."""
        values = {k: v for k, v in values.items() if v is not None}
        if not values:
            return self
        return self.model_copy(update={section: getattr(self, section).model_copy(update=values)})

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)
