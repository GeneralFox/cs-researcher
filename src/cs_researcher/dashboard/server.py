"""Local web server for the telemetry dashboard: an HTML page and a JSON API."""

import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from urllib.parse import parse_qs, urlparse

from .store import TelemetryStore

PAGE = resources.files(__package__).joinpath("index.html")


class DashboardServer:
    """Serves the dashboard of a TelemetryStore on host:port.

    Port 0 picks a free port. Raises OSError when the port is taken.
    """

    def __init__(self, store: TelemetryStore, host: str = "127.0.0.1", port: int = 8765):
        self.store = store
        self._server = ThreadingHTTPServer((host, port), self._handler())

    @property
    def port(self) -> int:
        return self._server.server_port

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}"

    def serve_forever(self) -> None:
        """Serve until Ctrl+C."""
        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self._server.server_close()

    def start(self) -> None:
        """Serve in a background thread, which ends with the program."""
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def wait(self) -> None:
        """Block until Ctrl+C, e.g. to keep a background dashboard up."""
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        store = self.store

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                url = urlparse(self.path)
                try:
                    if url.path == "/":
                        self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                    elif url.path == "/api/sessions":
                        self._json(store.sessions())
                    elif url.path == "/api/session":
                        self._json(store.session(parse_qs(url.query).get("id", [""])[0]))
                    else:
                        self._json({"error": "not found"}, 404)
                except sqlite3.OperationalError as e:
                    # e.g. database not created yet: no run executed
                    self._json({"error": str(e)}, 503)

            def _send(self, status: int, body: bytes, content_type: str):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _json(self, payload, status: int = 200):
                self._send(status, json.dumps(payload).encode(), "application/json")

            def log_message(self, *args):
                pass  # no log line for every auto-refresh request

        return Handler
