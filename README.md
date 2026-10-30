# crew-ai-experiment

CrewAI port of the multi-agent deep search from [ai-experiment](https://github.com/GeneralFox/ai-experiment): a master model orchestrating three analysts, each on a different local model, all running on Ollama. Same models, prompts, tools and limits as the deepagents, LangGraph, Google ADK and AG2 versions there, so the five frameworks can be compared.

It lives in its own repo because CrewAI requires Python < 3.14, while ai-experiment is a Python 3.14 project.

## Requirements

- [uv](https://docs.astral.sh/uv/) and Python 3.13 (the project venv is `crew-ai-py3.13/`)
- [Ollama](https://ollama.com) running on `localhost:11434` with `gemma4:12b-mlx`, `nemotron-3.5-bolt:e4b`, `bonsai-2:4b` and `qwen3.5:4b-mlx`

```bash
export UV_PROJECT_ENVIRONMENT=crew-ai-py3.13
uv sync
uv run deep_search_multi_crewai.py "What is Log4Shell and what mitigations does CISA recommend?"
```

`--max-analyst-calls N` caps each analyst's model calls (default 12).

## Files

| File | What it does |
|------|--------------|
| `deep_search_multi_crewai.py` | Master + 3 analysts; each analyst is a CrewAI `Agent` exposed to the master as a delegation tool taking a `brief`, and the three run in parallel |
| `research_tools.py` | Copy of ai-experiment's shared tools (DuckDuckGo search on trusted sites, HTML/PDF page fetcher, prompts): keep the two copies identical |
| `telemetry_crewai.py` | observra telemetry from CrewAI's event bus, same events and fields as the other frameworks |

## Telemetry

Events go to `telemetry.db` (SQLite). Open them with ai-experiment's dashboard, on a second port to keep both open:

```bash
cd ../ai-experiment
uv run dashboard.py --db /opt/crew-ai-experiment/telemetry.db --port 8766
```

CrewAI's own anonymous telemetry is turned off in the script (`OTEL_SDK_DISABLED`, `CREWAI_TRACING_ENABLED`): nothing leaves the machine.

## CrewAI defaults that had to change for a fair comparison

- **Ollama through LiteLLM** (`is_litellm=True`, `ollama_chat/` prefix): CrewAI's native Ollama provider uses the OpenAI-compatible `/v1` endpoint, which ignores `num_ctx` and loads the models with a 4096 context instead of 16384.
- **Declare native tool calling** (`litellm.register_model`): LiteLLM's catalog doesn't know local models and reports no function calling, so CrewAI falls back to ReAct text. The master then delegated to one analyst at a time and kept writing invented steps until the 3000-token output cap, 3-5 minutes per call.
- **Declare the context window**: CrewAI assumes 8192 for unknown models and replaces any tool result above 25% of it (one fetched page) with a preview the agent must read back with an extra `read_a_deferred_tool_result` call. That consumed the analysts' call budget.
- **Thinking off**: `reasoning_effort="none"`, which LiteLLM sends to Ollama as `think: false`, as in the other frameworks.
