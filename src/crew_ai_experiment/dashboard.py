"""Local dashboard for the observra telemetry of the multi-agent examples.

Reads, read-only, the SQLite database written by observra
(deep_search_multi.py and deep_search_multi_adk.py) and serves an HTML
page backed by a small JSON API. Standard library only.

    uv run dashboard.py [--db telemetry.db] [--port 8765]
"""

import argparse
import json
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HTML_PATH = Path(__file__).with_name("dashboard.html")

SESSIONS_SQL = """
SELECT session_id,
       MAX(framework)                                   AS framework,
       MIN(timestamp)                                   AS start,
       MAX(timestamp)                                   AS end,
       COUNT(*)                                         AS events,
       SUM(event_type = 'model_response')               AS model_calls,
       SUM(CASE WHEN event_type = 'model_response'
                THEN json_extract(data, '$.input_tokens') END)  AS input_tokens,
       SUM(CASE WHEN event_type = 'model_response'
                THEN json_extract(data, '$.output_tokens') END) AS output_tokens,
       SUM(event_type = 'tool_start')                   AS tool_calls,
       SUM(event_type LIKE '%error%')                   AS errors,
       COUNT(DISTINCT model_name)                       AS models
FROM events
WHERE session_id IS NOT NULL
GROUP BY session_id
ORDER BY start DESC
LIMIT 50
"""

EVENTS_SQL = """
SELECT timestamp, event_type, agent_name, model_name, tool_name, data
FROM events
WHERE session_id = ?
ORDER BY timestamp
LIMIT 5000
"""

# plumbing events hidden from the timeline unless "show all" is ticked
NOISE_EVENTS = {"stream_event", "model_request", "turn_duration", "user_message"}


def connect(db_path: str) -> sqlite3.Connection:
    # read-only: thanks to WAL we can read while a run is writing
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def rows(conn: sqlite3.Connection, sql: str, *params) -> list[dict]:
    return [dict(r) for r in conn.execute(sql, params)]


def prepare_events(events: list[dict]) -> list[dict]:
    """Parse event data, tidy model names and flag plumbing events as noise.

    Agent names are recorded at the source by telemetry.py. LangChain runs
    recorded before it have an "unknown" agent_start (and an unnamed
    agent_end) for every internal graph node: those are flagged as noise.
    """
    for e in events:
        e["data"] = json.loads(e["data"]) if e["data"] else {}
        e["model_name"] = (e["model_name"] or "").removeprefix("ollama_chat/") or None
        if e["agent_name"] == "unknown":
            e["agent_name"] = None
        is_boundary = e["event_type"] in ("agent_start", "agent_end")
        e["noise"] = e["event_type"] in NOISE_EVENTS or (is_boundary and not e["agent_name"])
    return events


def model_stats(events: list[dict]) -> list[dict]:
    """Calls and tokens per (agent, model) over the model_response events."""
    stats: dict[tuple, dict] = {}
    for e in events:
        if e["event_type"] != "model_response":
            continue
        key = (e["agent_name"] or "", e["model_name"] or "unknown")
        st = stats.setdefault(key, {"agent_name": key[0], "model_name": key[1],
                                    "calls": 0, "input_tokens": 0, "output_tokens": 0})
        st["calls"] += 1
        st["input_tokens"] += e["data"].get("input_tokens") or 0
        st["output_tokens"] += e["data"].get("output_tokens") or 0
    return sorted(stats.values(), key=lambda st: st["input_tokens"] + st["output_tokens"], reverse=True)


def sessions(db_path: str) -> list[dict]:
    with connect(db_path) as conn:
        return rows(conn, SESSIONS_SQL)


def tool_stats(events: list[dict]) -> list[dict]:
    """Calls, errors and durations per tool.

    The ADK plugin leaves duration_ms empty on tool_end: in that case the
    duration is derived by pairing each tool_end with the last open
    tool_start of the same tool by the same agent (analysts run in parallel).
    """
    stats: dict[str, dict] = {}
    open_starts: dict[tuple, list[float]] = {}
    for e in events:
        name = e["tool_name"]
        key = (e["agent_name"], name)
        st = stats.setdefault(name, {"tool_name": name, "calls": 0, "errors": 0, "durations": []})
        if e["event_type"] == "tool_start":
            st["calls"] += 1
            open_starts.setdefault(key, []).append(e["timestamp"])
        elif e["event_type"] in ("tool_end", "tool_error"):
            started = open_starts.get(key, [])
            start_ts = started.pop() if started else None
            if e["event_type"] == "tool_error":
                st["errors"] += 1
            elif e["data"].get("duration_ms") is not None:
                st["durations"].append(e["data"]["duration_ms"])
            elif start_ts is not None:
                st["durations"].append((e["timestamp"] - start_ts) * 1000)

    result = []
    for st in stats.values():
        durations = st.pop("durations")
        st["avg_ms"] = sum(durations) / len(durations) if durations else None
        st["max_ms"] = max(durations) if durations else None
        result.append(st)
    return sorted(result, key=lambda st: st["calls"], reverse=True)


def session_detail(db_path: str, session_id: str) -> dict:
    with connect(db_path) as conn:
        events = prepare_events(rows(conn, EVENTS_SQL, session_id))
    return {
        "models": model_stats(events),
        "tools": tool_stats([e for e in events if e["tool_name"]]),
        "timeline": events,
        "errors": [e for e in events if "error" in e["event_type"]],
    }


def make_handler(db_path: str):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload, status: int = 200):
            self._send(status, json.dumps(payload).encode(), "application/json")

        def do_GET(self):
            url = urlparse(self.path)
            try:
                if url.path == "/":
                    self._send(200, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/sessions":
                    self._json(sessions(db_path))
                elif url.path == "/api/session":
                    session_id = parse_qs(url.query).get("id", [""])[0]
                    self._json(session_detail(db_path, session_id))
                else:
                    self._json({"error": "not found"}, 404)
            except sqlite3.OperationalError as e:
                # e.g. database not created yet: no run executed
                self._json({"error": str(e)}, 503)

        def log_message(self, *args):
            pass  # no log line for every auto-refresh request

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Observra telemetry dashboard")
    parser.add_argument("--db", default="telemetry.db", help="SQLite database written by observra")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.db))
    print(f"Dashboard on http://localhost:{args.port}  (db: {args.db}) — Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
