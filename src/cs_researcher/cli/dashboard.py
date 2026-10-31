import argparse

from ..config import AppConfig
from ..dashboard import DashboardServer, TelemetryStore
from .base import Command


class DashboardCommand(Command):
    name = "dashboard"
    help = "serve the telemetry dashboard, to browse the past runs"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--port", type=int, help="port (default: dashboard.port in the configuration)")
        parser.add_argument("--db", help="telemetry database (default: telemetry.db_path in the configuration)")

    def run(self, args: argparse.Namespace, config: AppConfig) -> int:
        settings = config.updated("dashboard", port=args.port).dashboard
        store = TelemetryStore(args.db or config.telemetry.db_path)
        try:
            server = DashboardServer(store, settings.host, settings.port)
        except OSError as e:
            print(f"cs-researcher: cannot serve on port {settings.port}: {e}")
            return 1
        print(f"Dashboard on {server.url}  (db: {store.db_path}) — Ctrl+C to stop")
        server.serve_forever()
        return 0
