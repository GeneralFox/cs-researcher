"""Terminal output of a research run."""

from crewai.events import crewai_event_bus
from crewai.events.types.tool_usage_events import ToolUsageFinishedEvent, ToolUsageStartedEvent

from .team import ResearchResult, ResearchTeam

RULE = "=" * 70
PREVIEW_CHARS = 100


class ConsoleReporter:
    """Prints the team, every tool call as it happens, and the results."""

    def __init__(self):
        self._installed = False

    def install(self) -> None:
        """Print every tool call and its result, tagged with the agent (once)."""
        if self._installed:
            return
        self._installed = True

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def started(source, event):
            print(f"  [{event.agent_role}] → {event.tool_name}({str(event.tool_args)[:PREVIEW_CHARS]})", flush=True)

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def finished(source, event):
            preview = str(event.output).replace("\n", " ")[:PREVIEW_CHARS]
            print(f"  [{event.agent_role}]   ✓ {preview}", flush=True)

    def team(self, team: ResearchTeam) -> None:
        print(f"Coordinator: {team.coordinator_model}")
        print(f"Analysts: {', '.join(a.model for a in team.analysts)}")
        print(f"Sources: {', '.join(team.config.research.trusted_sites)}", flush=True)

    def result(self, result: ResearchResult) -> None:
        print("\n" + RULE)
        print(result.answer or f"No final answer: the synthesis failed ({result.error}).")
        print(RULE)
        minutes, seconds = divmod(result.elapsed, 60)
        print(f"Elapsed: {int(minutes)}m {seconds:.1f}s\n", flush=True)
