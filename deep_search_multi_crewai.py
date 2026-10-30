"""Multi-agent deep search with CrewAI: a master orchestrating three analysts.

Port of deep_search_multi_ag2.py / deep_search_multi_adk.py from the
ai-experiment repo: the master (the most capable model) does no research
itself, it delegates the same question to three analysts, each running a
different model, then compares the three answers and writes a weighted
synthesis. Same models, prompts, tools and limits as the other versions.

Each analyst is a CrewAI Agent exposed to the master as a delegation tool
taking a `brief`: the master calls the three tools in one turn and CrewAI
runs them in parallel (a thread pool), so the total size of master +
analysts must fit in RAM.

Ollama is reached through LiteLLM (`is_litellm=True`, `ollama_chat/`
prefix), like ADK: CrewAI's native Ollama provider uses the OpenAI-compatible
/v1 endpoint, which ignores num_ctx and would load the models with a 4096
context instead of 16384.

Telemetry goes to telemetry.db through telemetry_crewai.py; CrewAI's own
anonymous telemetry is turned off (everything here stays local).
"""

import argparse
import os
import time
import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

# CrewAI sends anonymous usage telemetry by default: off before importing it
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

import litellm
import observra
from crewai import LLM, Agent
from crewai.events import crewai_event_bus
from crewai.events.types.tool_usage_events import ToolUsageFinishedEvent, ToolUsageStartedEvent
from crewai.hooks.llm_hooks import register_before_llm_call_hook
from crewai.tools import tool

from telemetry_crewai import register_handlers, telemetry_session
from research_tools import search_sites, make_fetch_page, ANALYST_PROMPT, SITES

MASTER_MODEL = "gemma4:12b-mlx"
MASTER_NAME = "research_coordinator"
ANALYST_MODELS = [
    "nemotron-3.5-bolt:e4b",
    "bonsai-2:4b",
    "qwen3.5:4b-mlx",
]
NUM_CTX = 16384
# hard cap on every answer (master and analysts): the master is asked for
# ~800 words (~1100 tokens) and analysts write less, so this only cuts a
# runaway generation (a model stuck repeating itself until the context is full)
MAX_OUTPUT_TOKENS = 3000
# per-call timeout in seconds: the slowest healthy call measured so far took
# ~275 s (model load included); the hangs we saw lasted 10+ minutes
LLM_TIMEOUT = 420
# default cap on each analyst's model calls (--max-analyst-calls): a healthy
# analyst uses 3-6; the cap stops a model stuck repeating the same tool call
# (granite4.2:3b fetched one page 146 times in 45 minutes)
MAX_ANALYST_CALLS = 12
TELEMETRY_DB = "telemetry.db"  # read by ai-experiment's dashboard.py --db


def declare_tool_calling(model_names: list[str]):
    """Tell LiteLLM that the local models support native function calling.

    LiteLLM's catalog doesn't know them and answers False, and CrewAI then
    falls back to ReAct text ("Action: ... Action Input: ..."): one tool
    call per turn, so the master delegated to the analysts one at a time,
    and it kept writing invented steps until the 3000-token cap (3-5 min per
    call). Ollama does parse their tool calls: the other frameworks use them.
    """
    litellm.register_model({
        f"ollama_chat/{name}": {
            "litellm_provider": "ollama_chat",
            "mode": "chat",
            "supports_function_calling": True,
        }
        for name in model_names
    })


def make_llm(name: str, num_retries: int = 0) -> LLM:
    llm = LLM(
        model=f"ollama_chat/{name}",
        is_litellm=True,  # Ollama's native API, not /v1 (see the module docstring)
        base_url="http://localhost:11434",
        temperature=0.1,
        num_ctx=NUM_CTX,
        # thinking off (LiteLLM sends think=false to Ollama), as in the
        # other frameworks: same conditions for all
        reasoning_effort="none",
        max_tokens=MAX_OUTPUT_TOKENS,  # mapped to Ollama's num_predict
        timeout=LLM_TIMEOUT,  # LiteLLM's default for Ollama is 600 s
        num_retries=num_retries,  # LiteLLM retries a failed call
    )
    # CrewAI doesn't know the local models' context and assumes 8192: then a
    # tool result above 25% of it (~2000 tokens, one fetched page) is replaced
    # by a preview the agent must read back with an extra tool call
    # (read_a_deferred_tool_result), which ate the analysts' call budget.
    # Declare the real window, minus the 15% CrewAI reserves for catalog sizes.
    llm.context_window_size = int(NUM_CTX * 0.85)
    return llm


def slug(model_name: str) -> str:
    # "nemotron-3.5-bolt:e4b" -> "analyst_nemotron_3_5_bolt_e4b"
    return "analyst_" + model_name.replace(":", "_").replace(".", "_").replace("-", "_")


def limit_analyst_calls(analyst_names: set[str], max_calls: int):
    """Stop an analyst after max_calls model calls, like the other versions.

    A before_llm_call hook that returns False blocks the call: CrewAI raises
    LLMCallBlockedError, which run_analyst turns into "ANALYST FAILED".
    LiteLLM's retries happen inside one call, so they don't count.
    """

    def hook(context):
        if context.agent is not None and context.agent.role in analyst_names:
            # iterations counts the calls already made in this run
            if context.iterations >= max_calls:
                return False
        return None

    register_before_llm_call_hook(hook)


def show_progress():
    """Print every tool call and its result, tagged with the agent."""

    @crewai_event_bus.on(ToolUsageStartedEvent)
    def started(source, event):
        print(f"  [{event.agent_role}] → {event.tool_name}({str(event.tool_args)[:100]})", flush=True)

    @crewai_event_bus.on(ToolUsageFinishedEvent)
    def finished(source, event):
        preview = str(event.output).replace("\n", " ")[:100]
        print(f"  [{event.agent_role}]   ✓ {preview}", flush=True)


def make_analysts() -> list[Agent]:
    analysts = []
    for model_name in ANALYST_MODELS:
        analysts.append(
            Agent(
                role=slug(model_name),
                goal="Research the brief on the trusted sources and report with references.",
                backstory=ANALYST_PROMPT,  # same prompt as the other frameworks' analysts
                # retry a failed model call twice before the analyst fails,
                # like ModelRetryMiddleware in the LangChain version
                llm=make_llm(model_name, num_retries=2),
                tools=[
                    tool("search_sites")(search_sites),
                    tool("fetch_page")(make_fetch_page()),  # own "already read" memory
                ],
                allow_delegation=False,
                verbose=False,
            )
        )
    return analysts


def run_analyst(analyst: Agent, brief: str) -> str:
    """Run an analyst on the brief and return its report, never raising.

    A crashed analyst (e.g. Ollama rejecting a malformed tool call, or the
    call limit) or an empty report (a model that spent its tokens without
    writing any text) becomes an "ANALYST FAILED" report the master can work
    around. Not retried: that would cost a full research run.
    """
    try:
        report = analyst.kickoff(brief).raw
    except Exception as e:
        return f"ANALYST FAILED: {analyst.role} ({type(e).__name__}: {e})"
    if not (report or "").strip():
        return f"ANALYST FAILED: {analyst.role} (empty report)"
    return report


def analyst_tool(analyst: Agent, model_name: str):
    def delegate(brief: str) -> str:
        return run_analyst(analyst, brief)

    # the tool's description is the function's docstring; the parameter is
    # called `brief`, as in the AG2 and LangGraph versions (with `objective`
    # AG2's master rewrote the question instead of passing it verbatim)
    delegate.__doc__ = (
        f"Cybersecurity research analyst powered by {model_name}. "
        "Give it a research brief; it investigates the trusted sites and "
        "returns a summary with references."
    )
    return tool(analyst.role)(delegate)  # the master calls the analysts by the listed names


def master_prompt(analyst_names: list[str]) -> str:
    names = "\n".join(f"- {n}" for n in analyst_names)
    return f"""You are the coordinator of a cybersecurity research team.
You NEVER research topics yourself: you delegate to your analysts.

Your analysts (each one uses a different LLM, so their answers differ):
{names}

Workflow — follow it exactly:
1. Call EACH analyst listed above (three calls in total), passing the
   user's question as the research brief. Use the same brief for every
   analyst.
2. Compare the three reports and write the final answer with these sections:
   - **Consensus**: facts that at least two analysts agree on (high confidence).
   - **Divergences**: claims found by only one analyst, or contradictions
     (flag them as lower confidence).
   - **Summary**: your weighted synthesis of the question's answer.
   - **References**: the union of all URLs cited by the analysts, one per
     line, no duplicates.

If a report starts with "ANALYST FAILED", list that analyst as missing under
Divergences and synthesize from the remaining reports.
Keep the final answer under ~800 words.
Answer in the same language as the user's question.
Never add facts or URLs that do not come from the analysts' reports.
"""


def build_master(max_calls: int) -> Agent:
    analysts = make_analysts()
    limit_analyst_calls({a.role for a in analysts}, max_calls)
    return Agent(
        role=MASTER_NAME,
        goal="Delegate the question to the three analysts and write the weighted synthesis.",
        backstory=master_prompt([a.role for a in analysts]),
        llm=make_llm(MASTER_MODEL),
        # no research tools: delegation only
        tools=[analyst_tool(a, model) for a, model in zip(analysts, ANALYST_MODELS)],
        allow_delegation=False,  # CrewAI's own coworker delegation: not used
        verbose=False,
    )


def run(question: str, master: Agent):
    print(f"\nMaster: {MASTER_MODEL} — analysts: {', '.join(ANALYST_MODELS)}\n", flush=True)
    start = time.perf_counter()

    try:
        with telemetry_session():
            final_text = master.kickoff(question).raw
    except Exception as e:
        # e.g. the master's own model call timing out
        print(f"  [{MASTER_NAME}] failed: {type(e).__name__}: {e}", flush=True)
        final_text = None

    elapsed = time.perf_counter() - start

    print("\n" + "=" * 70)
    print(final_text or f"No final answer from {MASTER_NAME}: the synthesis failed (see errors above).")
    print("=" * 70)
    minutes, seconds = divmod(elapsed, 60)
    print(f"Elapsed: {int(minutes)}m {seconds:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-agent cybersecurity deep search (CrewAI)"
    )
    parser.add_argument(
        "--max-analyst-calls",
        type=int,
        default=MAX_ANALYST_CALLS,
        help=f"cap on each analyst's model calls (default {MAX_ANALYST_CALLS})",
    )
    parser.add_argument("question", nargs="*", help="The research question")
    args = parser.parse_args()

    # telemetry to SQLite; the final flush happens automatically at exit
    observra.initialize(backend="sqlite", path=TELEMETRY_DB)
    register_handlers()
    show_progress()
    declare_tool_calling([MASTER_MODEL, *ANALYST_MODELS])
    master = build_master(args.max_analyst_calls)

    if args.question:
        run(" ".join(args.question), master)
        return

    print(f"Multi-agent deep search (CrewAI) — master: {MASTER_MODEL}")
    print(f"Analysts: {', '.join(ANALYST_MODELS)}")
    print(f"Sources: {', '.join(SITES)}")
    print('Type your question and press Enter. Type "exit" to quit.\n')
    while True:
        question = input("Q: ").strip()
        if not question:
            continue
        if question.lower() == "exit":
            break
        run(question, master)


if __name__ == "__main__":
    main()
