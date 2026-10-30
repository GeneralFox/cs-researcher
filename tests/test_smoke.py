"""Offline smoke tests: no Ollama and no network needed."""

import subprocess
import sys

import litellm
import pytest

from crew_ai_experiment import deep_search_multi_crewai as ds
from crew_ai_experiment import research_tools as rt


def test_slug():
    assert ds.slug("nemotron-3.5-bolt:e4b") == "analyst_nemotron_3_5_bolt_e4b"


def test_master_prompt_lists_analysts():
    names = [ds.slug(m) for m in ds.ANALYST_MODELS]
    prompt = ds.master_prompt(names)
    for name in names:
        assert f"- {name}" in prompt


def test_declare_tool_calling():
    ds.declare_tool_calling([ds.MASTER_MODEL])
    assert litellm.supports_function_calling(f"ollama_chat/{ds.MASTER_MODEL}")


def test_make_llm_declares_context_window():
    llm = ds.make_llm(ds.MASTER_MODEL)
    assert llm.context_window_size == int(ds.NUM_CTX * 0.85)


def test_master_delegates_to_each_analyst():
    master = ds.build_master(ds.MAX_ANALYST_CALLS)
    assert sorted(t.name for t in master.tools) == sorted(ds.slug(m) for m in ds.ANALYST_MODELS)


class _Analyst:
    role = "analyst_test"

    def __init__(self, result=None, error=None):
        self.result, self.error = result, error

    def kickoff(self, brief):
        if self.error:
            raise self.error
        return type("Output", (), {"raw": self.result})()


@pytest.mark.parametrize(
    ("analyst", "expected"),
    [
        (_Analyst(result="report"), "report"),
        (_Analyst(result="  "), "ANALYST FAILED: analyst_test (empty report)"),
        (_Analyst(error=TimeoutError("slow")), "ANALYST FAILED: analyst_test (TimeoutError: slow)"),
    ],
)
def test_run_analyst_never_raises(analyst, expected):
    assert ds.run_analyst(analyst, "brief") == expected


def test_fetch_page_refuses_untrusted_site():
    fetch_page = rt.make_fetch_page()
    assert fetch_page("https://example.com/page").startswith("Refused:")


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "crew_ai_experiment", "--help"], capture_output=True, text=True, check=True
    )
    assert "--max-analyst-calls" in result.stdout
