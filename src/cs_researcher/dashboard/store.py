"""Read-only queries on the telemetry database."""

import json
import sqlite3
from pathlib import Path

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
LIMIT ?
"""

EVENTS_SQL = """
SELECT timestamp, event_type, agent_name, model_name, tool_name, data
FROM events
WHERE session_id = ?
ORDER BY timestamp
LIMIT ?
"""

# plumbing events hidden from the timeline unless "show all" is ticked
NOISE_EVENTS = {"model_request"}


class TelemetryStore:
    """Sessions and their events, read from observra's SQLite database.

    Read-only: thanks to WAL it can be read while a run is writing. A missing
    database (no run yet) raises sqlite3.OperationalError.
    """

    MAX_SESSIONS = 50
    MAX_EVENTS = 5000

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)

    def sessions(self) -> list[dict]:
        """The latest sessions, newest first, with their totals."""
        return self._rows(SESSIONS_SQL, self.MAX_SESSIONS)

    def session(self, session_id: str) -> dict:
        """A session's per-model and per-tool stats, timeline and errors."""
        events = [self._prepare(e) for e in self._rows(EVENTS_SQL, session_id, self.MAX_EVENTS)]
        return {
            "models": self._model_stats(events),
            "tools": self._tool_stats([e for e in events if e["tool_name"]]),
            "timeline": events,
            "errors": [e for e in events if "error" in e["event_type"]],
        }

    def _rows(self, sql: str, *params) -> list[dict]:
        with sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute(sql, params)]

    @staticmethod
    def _prepare(event: dict) -> dict:
        event["data"] = json.loads(event["data"]) if event["data"] else {}
        event["noise"] = event["event_type"] in NOISE_EVENTS
        return event

    @staticmethod
    def _model_stats(events: list[dict]) -> list[dict]:
        """Calls and tokens per (agent, model), heaviest first."""
        stats: dict[tuple, dict] = {}
        for e in events:
            if e["event_type"] != "model_response":
                continue
            key = (e["agent_name"] or "", e["model_name"] or "unknown")
            st = stats.setdefault(
                key, {"agent_name": key[0], "model_name": key[1], "calls": 0, "input_tokens": 0, "output_tokens": 0}
            )
            st["calls"] += 1
            st["input_tokens"] += e["data"].get("input_tokens") or 0
            st["output_tokens"] += e["data"].get("output_tokens") or 0
        return sorted(stats.values(), key=lambda st: st["input_tokens"] + st["output_tokens"], reverse=True)

    @staticmethod
    def _tool_stats(events: list[dict]) -> list[dict]:
        """Calls, errors and durations per tool, most called first."""
        stats: dict[str, dict] = {}
        for e in events:
            st = stats.setdefault(
                e["tool_name"], {"tool_name": e["tool_name"], "calls": 0, "errors": 0, "durations": []}
            )
            if e["event_type"] == "tool_start":
                st["calls"] += 1
            elif e["event_type"] == "tool_error":
                st["errors"] += 1
            elif e["event_type"] == "tool_end" and e["data"].get("duration_ms") is not None:
                st["durations"].append(e["data"]["duration_ms"])

        for st in stats.values():
            durations = st.pop("durations")
            st["avg_ms"] = sum(durations) / len(durations) if durations else None
            st["max_ms"] = max(durations) if durations else None
        return sorted(stats.values(), key=lambda st: st["calls"], reverse=True)
