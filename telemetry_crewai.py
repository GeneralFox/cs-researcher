"""observra telemetry for the CrewAI example (observra ships no CrewAI adapter).

Same approach as telemetry_ag2.py in the ai-experiment repo: the events are
written by hand with observra.emit(), here from handlers on CrewAI's event
bus, with the same event types and fields the stock adapters produce, so
ai-experiment's dashboard.py can read this repo's telemetry.db (--db):

    session_start / session_end     one per question (telemetry_session)
    agent_start / agent_end         one per agent run (Agent.kickoff)
    model_request / model_response  one per LLM call, with tokens and the
                                    output shape (text chars, tool calls, finish reason)
    tool_start / tool_end / tool_error  one per tool call, with args, result
                                    and duration

CrewAI runs sync handlers in a thread pool, but with a copy of the emitting
thread's context: observra's session and trace ids (ContextVars) arrive
intact, also from the analysts running in parallel.
"""

import contextlib

import observra
from crewai.events import crewai_event_bus
from crewai.events.types.agent_events import (
    LiteAgentExecutionCompletedEvent,
    LiteAgentExecutionErrorEvent,
    LiteAgentExecutionStartedEvent,
)
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
)
from crewai.events.types.tool_usage_events import (
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)
from observra.core.context import initialize_trace, new_span

FRAMEWORK = "crewai"
MAX_TOOL_DATA = 4096  # like observra's payload_max_bytes default


def _emit(event_type: str, **kwargs):
    observra.emit(event_type, framework=FRAMEWORK, **kwargs)


def _truncate(text: str) -> str:
    return text if len(text) <= MAX_TOOL_DATA else text[:MAX_TOOL_DATA] + "...[truncated]"


def _model(name: str | None) -> str | None:
    # "ollama_chat/qwen3.5:4b-mlx" -> "qwen3.5:4b-mlx", as LangChain and AG2 report it
    return name.split("/", 1)[-1] if name else None


@contextlib.contextmanager
def telemetry_session():
    """One observra session (and trace) around a question."""
    initialize_trace()
    observra.initialize_session()
    _emit("session_start")
    error = None
    try:
        yield
    except BaseException as e:
        error = str(e)
        raise
    finally:
        # handlers run in a thread pool: wait for them before closing the session
        crewai_event_bus.flush()
        _emit("session_end", error_message=error)


def _response_shape(response) -> dict:
    """Where the output went: text or tool calls (see telemetry.py in ai-experiment)."""
    if isinstance(response, str):
        return {"text_chars": len(response), "tool_calls": 0}
    if isinstance(response, list):  # native tool calls
        return {"text_chars": 0, "tool_calls": len(response)}
    return {"text_chars": len(str(response or "")), "tool_calls": 0}


def register_handlers():
    """Write CrewAI's agent, LLM and tool events to observra (call once)."""

    @crewai_event_bus.on(LiteAgentExecutionStartedEvent)
    def agent_start(source, event):
        new_span()
        _emit("agent_start", agent_name=event.agent_info.get("role"))

    @crewai_event_bus.on(LiteAgentExecutionCompletedEvent)
    def agent_end(source, event):
        _emit("agent_end", agent_name=event.agent_info.get("role"))

    @crewai_event_bus.on(LiteAgentExecutionErrorEvent)
    def agent_error(source, event):
        _emit("agent_end", agent_name=event.agent_info.get("role"), result="failure", error_message=event.error)

    # LLM events carry the agent in agent_role (CrewAI clears from_agent)
    @crewai_event_bus.on(LLMCallStartedEvent)
    def llm_start(source, event):
        _emit("model_request", agent_name=event.agent_role, model_name=_model(event.model))

    @crewai_event_bus.on(LLMCallCompletedEvent)
    def llm_end(source, event):
        usage = event.usage or {}
        _emit(
            "model_response",
            agent_name=event.agent_role,
            model_name=_model(event.model),
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            # same output shape as the other frameworks (thinking is off)
            reasoning_chars=0,
            done_reason=event.finish_reason,
            **_response_shape(event.response),
        )

    @crewai_event_bus.on(LLMCallFailedEvent)
    def llm_error(source, event):
        _emit(
            "model_error",
            agent_name=event.agent_role,
            model_name=_model(event.model),
            error_message=event.error,
        )

    @crewai_event_bus.on(ToolUsageStartedEvent)
    def tool_start(source, event):
        new_span()  # a span per call: the analysts' tool calls overlap
        _emit(
            "tool_start",
            agent_name=event.agent_role,
            tool_name=event.tool_name,
            tool_args=_truncate(str(event.tool_args)),
        )

    @crewai_event_bus.on(ToolUsageFinishedEvent)
    def tool_end(source, event):
        _emit(
            "tool_end",
            agent_name=event.agent_role,
            tool_name=event.tool_name,
            tool_args=_truncate(str(event.tool_args)),
            tool_result=_truncate(str(event.output)),
            duration_ms=(event.finished_at - event.started_at).total_seconds() * 1000,
        )

    @crewai_event_bus.on(ToolUsageErrorEvent)
    def tool_error(source, event):
        _emit(
            "tool_error",
            agent_name=event.agent_role,
            tool_name=event.tool_name,
            tool_args=_truncate(str(event.tool_args)),
            error_message=str(event.error),
        )
