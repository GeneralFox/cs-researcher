# cs-researcher

Multi-agent cybersecurity research on local models. A coordinator model hands each question to a team of analysts, each running a different model. The analysts search and read only trusted sources (NIST, CISA, MITRE ATT&CK, SANS ISC, …), then the coordinator compares their reports and writes a weighted synthesis: what they agree on, where they diverge, a summary and the references.

Everything runs on [Ollama](https://ollama.com), through [CrewAI](https://crewai.com); nothing leaves the machine except the web searches and page downloads.

## Requirements

- [uv](https://docs.astral.sh/uv/): it installs Python 3.13 by itself (CrewAI does not support 3.14 yet)
- Ollama on `localhost:11434` with the models of the configuration; by default `gemma4:12b-mlx` (coordinator), `nemotron-3.5-bolt:e4b`, `bonsai-2:4b` and `qwen3.5:4b-mlx` (analysts). The analysts run in parallel, so all the models must fit in RAM together.

## Usage

```bash
uv sync
uv run cs-researcher ask "What is Log4Shell and what mitigations does CISA recommend?"
```

| Command | What it does |
|---------|--------------|
| `cs-researcher ask "question"` | Research a question. Without one, ask questions interactively. `--dashboard` also serves the telemetry dashboard to follow the run live (it stays up after the answer until Ctrl+C); `--port`, `--max-analyst-calls N` override the configuration |
| `cs-researcher dashboard` | Serve the telemetry dashboard on http://localhost:8765 to browse past runs (`--port`, `--db`) |
| `cs-researcher config` | Print the configuration in use, as JSON |
| `cs-researcher --config my.json <command>` | Use another configuration file |

`uv run python -m cs_researcher …` works too.

## Configuration

The package ships [`config.json`](src/cs_researcher/config.json); `--config` replaces it. To start a custom one:

```bash
uv run cs-researcher config > my.json
```

| Section | Keys |
|---------|------|
| `ollama` | `base_url` |
| `llm_defaults` | `num_ctx`, `temperature`, `max_tokens` (hard cap on an answer), `timeout` (seconds per call), `num_retries`, `think` |
| `coordinator` | `model`, optional `name`, and any `llm_defaults` key to override it |
| `analysts` | a list of `{ "model": … }`, each with optional `name` (default `analyst_<model>`) and `llm_defaults` overrides |
| `research` | `max_analyst_calls` (per analyst and question), `search_results`, `max_page_chars`, `fetch_timeout`, `trusted_sites` |
| `telemetry` | `enabled`, `db_path` (relative to the working directory) |
| `dashboard` | `host`, `port` |

Sections left out take the shipped values; unknown keys are an error, so a typo doesn't go unnoticed. Any number of analysts works: the coordinator's instructions list them.

## Telemetry

Every question is recorded in `telemetry.db` (SQLite, through [observra](https://pypi.org/project/observra/)): agent runs, model calls with tokens, tool calls with arguments, results and durations, errors. The dashboard shows the sessions, per-model and per-tool stats, an agent swimlane and the errors, refreshing itself every 5 s.

CrewAI's own anonymous telemetry is turned off (`OTEL_SDK_DISABLED`, `CREWAI_TRACING_ENABLED`).

## Code

All under `src/cs_researcher/`:

| Module | Contents |
|--------|----------|
| `config.py` | `AppConfig` and its sections: the JSON configuration, validated with pydantic |
| `team.py` | `ResearchTeam` (coordinator + analysts, `ask(question)`), `Analyst`, `ResearchResult` |
| `llm.py` | `OllamaLLMFactory`: CrewAI LLMs for Ollama models (see below) |
| `tools.py` | `TrustedSites`, `SiteSearch` (DuckDuckGo), `PageReader` (HTML and PDF, shared download cache), `PageFetcher` (an agent's `fetch_page` tool) |
| `prompts.py` | The analysts' and the coordinator's instructions |
| `telemetry.py` | `Telemetry`: CrewAI's event bus to observra |
| `console.py` | `ConsoleReporter`: tool calls as they happen, and the answers |
| `dashboard/` | `TelemetryStore` (queries), `DashboardServer` (HTML page + JSON API), standard library only |
| `cli/` | `Command` base class and the `ask`, `dashboard`, `config` commands |

A new command is a `Command` subclass (`name`, `help`, `add_arguments`, `run`) added to `COMMANDS` in `cli/__init__.py`.

### CrewAI defaults overridden for local models

- **Ollama through LiteLLM** (`is_litellm=True`, `ollama_chat/` prefix): CrewAI's native Ollama provider uses the OpenAI-compatible `/v1` endpoint, which ignores `num_ctx` and loads the models with a 4096 context.
- **Native tool calling declared** (`litellm.register_model`): LiteLLM's catalog doesn't know local models and reports no function calling, so CrewAI falls back to ReAct text. The coordinator then delegates to one analyst at a time and keeps writing invented steps until the output cap.
- **Context window declared**: CrewAI assumes 8192 for unknown models and replaces any tool result above 25% of it (one fetched page) with a preview the agent must read back with an extra call, which eats the analysts' call budget.
- **Thinking off** by default (`think: false`, sent to Ollama by LiteLLM as `reasoning_effort="none"`).

## Development

```bash
uv run pytest          # offline: no Ollama or network needed
uv run ruff check .
uv run ruff format .
```
