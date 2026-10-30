"""Dashboard tests on a throwaway SQLite database with observra's events schema."""

import json
import sqlite3
import urllib.error
import urllib.request

import pytest

from crew_ai_experiment import deep_search_multi_crewai as ds

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
        "ollama_chat/qwen3.5:4b-mlx",
        json.dumps({"input_tokens": 100, "output_tokens": 20}),
    ),
    ("e3", 3.0, "s1", "tool_start", "analyst_a", "fetch_page", None, "{}"),
    ("e4", 4.5, "s1", "tool_end", "analyst_a", "fetch_page", None, "{}"),
]


@pytest.fixture
def server(tmp_path, monkeypatch):
    db = tmp_path / "telemetry.db"
    with sqlite3.connect(db) as conn:
        conn.execute(SCHEMA)
        conn.executemany(
            "INSERT INTO events (event_id, timestamp, session_id, event_type, agent_name, tool_name, model_name, data)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            EVENTS,
        )
    monkeypatch.setattr(ds, "TELEMETRY_DB", str(db))
    server = ds.start_dashboard(0)  # port 0: any free port
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def get(url):
    with urllib.request.urlopen(url) as resp:
        return resp.read()


def test_serves_the_page(server):
    assert b"<html" in get(server + "/").lower()


def test_sessions(server):
    [session] = json.loads(get(server + "/api/sessions"))
    assert session["session_id"] == "s1"
    assert (session["model_calls"], session["tool_calls"], session["input_tokens"]) == (1, 1, 100)


def test_session_detail(server):
    detail = json.loads(get(server + "/api/session?id=s1"))
    assert detail["models"][0]["model_name"] == "qwen3.5:4b-mlx"  # ollama_chat/ prefix removed
    assert detail["tools"][0]["avg_ms"] == pytest.approx(1500)


def test_missing_db_is_503(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "TELEMETRY_DB", str(tmp_path / "none.db"))
    server = ds.start_dashboard(0)
    try:
        with pytest.raises(urllib.error.HTTPError) as err:
            get(f"http://127.0.0.1:{server.server_port}/api/sessions")
        assert err.value.code == 503
    finally:
        server.shutdown()


def test_port_taken_does_not_raise(server):
    port = int(server.rsplit(":", 1)[1])
    assert ds.start_dashboard(port) is None
