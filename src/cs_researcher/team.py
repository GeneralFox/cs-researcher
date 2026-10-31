"""The research team: a coordinator delegating to analysts on different models.

The coordinator does no research itself: it gives the question to every
analyst, then compares their reports and writes a weighted synthesis. Each
analyst is a CrewAI Agent exposed to the coordinator as a tool taking a
`brief`: the coordinator calls them all in one turn and CrewAI runs them in
parallel (a thread pool), so all the models must fit in RAM together.
"""

import contextlib
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass

from crewai import LLM, Agent
from crewai.hooks.llm_hooks import register_before_llm_call_hook
from crewai.tools import BaseTool, tool

from .config import AgentSettings, AppConfig
from .llm import OllamaLLMFactory
from .prompts import analyst_prompt, coordinator_prompt
from .tools import PageFetcher, PageReader, SiteSearch, TrustedSites


@dataclass(frozen=True)
class ResearchResult:
    question: str
    answer: str | None  # None when the coordinator failed
    elapsed: float  # seconds
    error: str | None = None


class Analyst:
    """A research analyst: an agent on its own model, with its own tools."""

    def __init__(self, settings: AgentSettings, llm: LLM, tools: list[BaseTool], prompt: str):
        self.name = settings.role
        self.model = settings.model
        self.agent = Agent(
            role=self.name,
            goal="Research the brief on the trusted sources and report with references.",
            backstory=prompt,
            llm=llm,
            tools=tools,
            allow_delegation=False,
            verbose=False,
        )

    def research(self, brief: str) -> str:
        """The analyst's report on the brief; never raises.

        A crashed analyst (e.g. Ollama rejecting a malformed tool call, or the
        call limit) or an empty report (a model that spent its tokens without
        writing any text) becomes an "ANALYST FAILED" report the coordinator
        can work around. Not retried: that would cost a full research run.
        """
        try:
            report = self.agent.kickoff(brief).raw
        except Exception as e:
            return f"ANALYST FAILED: {self.name} ({type(e).__name__}: {e})"
        if not (report or "").strip():
            return f"ANALYST FAILED: {self.name} (empty report)"
        return report

    def as_tool(self) -> BaseTool:
        """The tool the coordinator calls, named after the analyst."""

        # the parameter is called `brief`: with `objective` models tend to
        # rewrite the question instead of passing it verbatim
        def delegate(brief: str) -> str:
            return self.research(brief)

        # the tool's description is the function's docstring
        delegate.__doc__ = (
            f"Cybersecurity research analyst powered by {self.model}. "
            "Give it a research brief; it investigates the trusted sites and "
            "returns a summary with references."
        )
        return tool(self.name)(delegate)


class ResearchTeam:
    """The coordinator and its analysts, built from the configuration."""

    def __init__(
        self,
        config: AppConfig,
        session: Callable[[], AbstractContextManager] = contextlib.nullcontext,
    ):
        """`session` wraps each question, e.g. Telemetry.session."""
        self.config = config
        self._session = session
        llms = OllamaLLMFactory(config.ollama)
        research = config.research

        sites = TrustedSites(research.trusted_sites)
        search = SiteSearch(sites, research.search_results)
        reader = PageReader(sites, research.max_page_chars, research.fetch_timeout)  # shared download cache
        prompt = analyst_prompt(sites.domains)
        self.analysts = [
            Analyst(
                settings,
                llms.create(settings),
                [search.as_tool(), PageFetcher(reader).as_tool()],  # own "already read" memory
                prompt,
            )
            for settings in config.analysts
        ]
        self._limit_analyst_calls(research.max_analyst_calls)

        coordinator = config.coordinator
        self.coordinator_model = coordinator.model
        self.coordinator = Agent(
            role=coordinator.role,
            goal="Delegate the question to the analysts and write the weighted synthesis.",
            backstory=coordinator_prompt(a.name for a in self.analysts),
            llm=llms.create(coordinator),
            tools=[a.as_tool() for a in self.analysts],  # no research tools: delegation only
            allow_delegation=False,  # CrewAI's own coworker delegation: not used
            verbose=False,
        )

    def _limit_analyst_calls(self, max_calls: int) -> None:
        """Stop an analyst after max_calls model calls.

        A before_llm_call hook returning False blocks the call: CrewAI then
        raises, and Analyst.research turns that into an "ANALYST FAILED" report.
        """
        names = {a.name for a in self.analysts}

        def hook(context):
            # iterations counts the calls already made in this run
            if context.agent is not None and context.agent.role in names and context.iterations >= max_calls:
                return False
            return None

        register_before_llm_call_hook(hook)

    def ask(self, question: str) -> ResearchResult:
        """Research a question; a failure is reported in the result, not raised."""
        start = time.perf_counter()
        try:
            with self._session():
                answer, error = self.coordinator.kickoff(question).raw, None
        except Exception as e:
            # e.g. the coordinator's own model call timing out
            answer, error = None, f"{type(e).__name__}: {e}"
        return ResearchResult(question, answer, time.perf_counter() - start, error)
