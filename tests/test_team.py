"""The team is built offline: no model is called."""

import litellm
import pytest

from cs_researcher.config import AppConfig
from cs_researcher.llm import OllamaLLMFactory
from cs_researcher.team import ResearchTeam


@pytest.fixture(scope="module")
def config():
    return AppConfig.load()


@pytest.fixture(scope="module")
def team(config):
    return ResearchTeam(config)


def test_llm_factory(config):
    llm = OllamaLLMFactory(config.ollama).create(config.coordinator)
    assert llm.context_window_size == int(config.coordinator.num_ctx * 0.85)
    assert litellm.supports_function_calling(f"ollama_chat/{config.coordinator.model}")


def test_coordinator_has_a_tool_per_analyst(team, config):
    assert [t.name for t in team.coordinator.tools] == [a.role for a in config.analysts]
    for analyst in config.analysts:
        assert f"- {analyst.role}" in team.coordinator.backstory


class Output:
    def __init__(self, raw):
        self.raw = raw


@pytest.mark.parametrize(
    ("kickoff", "expected"),
    [
        (lambda brief: Output("report"), "report"),
        (lambda brief: Output("  "), "ANALYST FAILED: {name} (empty report)"),
        (lambda brief: (_ for _ in ()).throw(TimeoutError("slow")), "ANALYST FAILED: {name} (TimeoutError: slow)"),
    ],
)
def test_analyst_research_never_raises(team, monkeypatch, kickoff, expected):
    analyst = team.analysts[0]
    monkeypatch.setattr(type(analyst.agent), "kickoff", lambda self, brief: kickoff(brief))
    assert analyst.research("brief") == expected.format(name=analyst.name)


def test_failed_question_is_reported(team, monkeypatch):
    def fail(self, question):
        raise TimeoutError("slow")

    monkeypatch.setattr(type(team.coordinator), "kickoff", fail)
    result = team.ask("What is Log4Shell?")
    assert (result.answer, result.error) == (None, "TimeoutError: slow")
