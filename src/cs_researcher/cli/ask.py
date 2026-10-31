import argparse

from ..config import AppConfig
from ..dashboard import DashboardServer, TelemetryStore
from .base import Command


class AskCommand(Command):
    name = "ask"
    help = "research a question (without one, ask questions interactively)"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--max-analyst-calls",
            type=int,
            metavar="N",
            help="cap on each analyst's model calls (default: research.max_analyst_calls in the configuration)",
        )
        parser.add_argument(
            "--dashboard", action="store_true", help="also serve the telemetry dashboard, to follow the run live"
        )
        parser.add_argument(
            "--port", type=int, help="port of the --dashboard (default: dashboard.port in the configuration)"
        )
        parser.add_argument("question", nargs="*", help="the research question")

    def run(self, args: argparse.Namespace, config: AppConfig) -> int:
        config = config.updated("research", max_analyst_calls=args.max_analyst_calls).updated(
            "dashboard", port=args.port
        )

        # CrewAI takes a while to import: only for the commands that use it
        from ..console import ConsoleReporter
        from ..team import ResearchTeam
        from ..telemetry import Telemetry

        telemetry = Telemetry(config.telemetry)
        telemetry.install()
        console = ConsoleReporter()
        console.install()
        team = ResearchTeam(config, session=telemetry.session)
        dashboard = self._start_dashboard(config) if args.dashboard else None
        console.team(team)

        if args.question:
            console.result(team.ask(" ".join(args.question)))
            if dashboard:
                print("Dashboard still running — Ctrl+C to stop")
                dashboard.wait()
            return 0

        print('Type your question and press Enter. Type "exit" to quit.\n')
        while True:
            try:
                question = input("Q: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if question.lower() == "exit":
                return 0
            if question:
                console.result(team.ask(question))

    @staticmethod
    def _start_dashboard(config: AppConfig) -> DashboardServer | None:
        """The dashboard in the background; the research runs anyway if it can't start."""
        if not config.telemetry.enabled:
            print("Dashboard not started: telemetry is disabled in the configuration.")
            return None
        settings = config.dashboard
        try:
            server = DashboardServer(TelemetryStore(config.telemetry.db_path), settings.host, settings.port)
        except OSError as e:
            print(f"Dashboard not started on port {settings.port}: {e}")
            return None
        server.start()
        # observra writes every 0.5 s and the page refreshes itself: the run shows up live
        print(f"Dashboard on {server.url}  (db: {config.telemetry.db_path})")
        return server
