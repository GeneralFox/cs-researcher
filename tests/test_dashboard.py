"""Dashboard tests on a throwaway SQLite database with observra's events schema."""

import json
import sqlite3
import urllib.error
import urllib.request

import pytest

from cs_researcher.dashboard import DashboardServer, TelemetryStore

SCHEMA = """
CREATE TABLE events (
    event_id TEXT PRIMARY KEY, timestamp REAL NOT NULL, session_id TEXT NOT NULL,
    trace_id TEXT, span_id TEXT, event_type TEXT NOT NULL, agent_name TEXT,
    tool_name TEXT, model_name TEXT, framework TEXT, host TEXT, data TEXT,
    library_version TEXT
)
"""

EVENTS = [
    ("e1", 1.0, "s1", "session_start", None, None, None, "{}"),
    (
        "e2",
        2.0,
        "s1",
        "model_response",
        "analyst_a",
        None,
        "qwen3.5:4b-mlx",
        json.dumps({"input_tokens": 100, "output_tokens": 20}),
    ),
    ("e3", 3.0, "s1", "tool_start", "analyst_a", "fetch_page", None, "{}"),
    ("e4", 4.5, "s1", "tool_end", "analyst_a", "fetch_page", None, json.dumps({"duration_ms": 1500})),
]


@pytest.fixture
def server(tmp_path):
    db = tmp_path / "telemetry.db"
    with sqlite3.connect(db) as conn:
        conn.execute(SCHEMA)
        conn.executemany(
            "INSERT INTO events (event_id, timestamp, session_id, event_type, agent_name, tool_name, model_name, data)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            EVENTS,
        )
    server = DashboardServer(TelemetryStore(db), port=0)  # port 0: any free port
    server.start()
    yield server
    server.shutdown()


def get(url):
    with urllib.request.urlopen(url) as resp:
        return resp.read()


def test_serves_the_page(server):
    assert b"<html" in get(server.url + "/").lower()


def test_sessions(server):
    [session] = json.loads(get(server.url + "/api/sessions"))
    assert session["session_id"] == "s1"
    assert (session["model_calls"], session["tool_calls"], session["input_tokens"]) == (1, 1, 100)


def test_session_detail(server):
    detail = json.loads(get(server.url + "/api/session?id=s1"))
    assert detail["models"][0]["model_name"] == "qwen3.5:4b-mlx"
    assert detail["tools"][0]["avg_ms"] == pytest.approx(1500)


def test_missing_db_is_503(tmp_path):
    server = DashboardServer(TelemetryStore(tmp_path / "none.db"), port=0)
    server.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as err:
            get(server.url + "/api/sessions")
        assert err.value.code == 503
    finally:
        server.shutdown()


def test_port_taken(server):
    with pytest.raises(OSError):
        DashboardServer(server.store, port=server.port)
