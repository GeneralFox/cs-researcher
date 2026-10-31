"""observra telemetry from CrewAI's event bus (observra has no CrewAI adapter).

The events are written with observra.emit() from handlers on CrewAI's event
bus, with the event types and fields of observra's stock adapters, into the
SQLite database the dashboard reads:

    session_start / session_end     one per question (Telemetry.session)
    agent_start / agent_end         one per agent run (Agent.kickoff)
    model_request / model_response  one per LLM call, with tokens and the
                                    output shape (text chars, tool calls, finish reason)
    model_error                     a failed LLM call
    tool_start / tool_end / tool_error  one per tool call, with args, result
                                    and duration

CrewAI runs sync handlers in a thread pool, but with a copy of the emitting
thread's context: observra's session and trace ids (ContextVars) arrive
intact, also from the analysts running in parallel.
"""

import contextlib
from collections.abc import Iterator

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

from .config import TelemetrySettings


class Telemetry:
    """Writes CrewAI's events to observra's SQLite database, when enabled."""

    FRAMEWORK = "crewai"
    MAX_TOOL_DATA = 4096  # like observra's payload_max_bytes default

    def __init__(self, settings: TelemetrySettings):
        self.enabled = settings.enabled
        self.db_path = settings.db_path
        self._installed = False

    def install(self) -> None:
        """Open the database and subscribe to CrewAI's events (once)."""
        if not self.enabled or self._installed:
            return
        # the final flush happens automatically at exit
        observra.initialize(backend="sqlite", path=str(self.db_path))
        self._register_handlers()
        self._installed = True

    @contextlib.contextmanager
    def session(self) -> Iterator[None]:
        """One observra session (and trace) around a question."""
        if not self._installed:
            yield
            return
        initialize_trace()
        observra.initialize_session()
        self._emit("session_start")
        error = None
        try:
            yield
        except BaseException as e:
            error = str(e)
            raise
        finally:
            # handlers run in a thread pool: wait for them before closing the session
            crewai_event_bus.flush()
            self._emit("session_end", error_message=error)

    def _emit(self, event_type: str, **kwargs) -> None:
        observra.emit(event_type, framework=self.FRAMEWORK, **kwargs)

    @classmethod
    def _truncate(cls, value) -> str:
        text = str(value)
        return text if len(text) <= cls.MAX_TOOL_DATA else text[: cls.MAX_TOOL_DATA] + "...[truncated]"

    @staticmethod
    def _model(name: str | None) -> str | None:
        # "ollama_chat/qwen3.5:4b-mlx" -> "qwen3.5:4b-mlx"
        return name.split("/", 1)[-1] if name else None

    @staticmethod
    def _response_shape(response) -> dict:
        """Where the output went: text or tool calls."""
        if isinstance(response, list):  # native tool calls
            return {"text_chars": 0, "tool_calls": len(response)}
        return {"text_chars": len(str(response or "")), "tool_calls": 0}

    def _register_handlers(self) -> None:
        emit, truncate, model = self._emit, self._truncate, self._model

        @crewai_event_bus.on(LiteAgentExecutionStartedEvent)
        def agent_start(source, event):
            new_span()
            emit("agent_start", agent_name=event.agent_info.get("role"))

        @crewai_event_bus.on(LiteAgentExecutionCompletedEvent)
        def agent_end(source, event):
            emit("agent_end", agent_name=event.agent_info.get("role"))

        @crewai_event_bus.on(LiteAgentExecutionErrorEvent)
        def agent_error(source, event):
            emit("agent_end", agent_name=event.agent_info.get("role"), result="failure", error_message=event.error)

        # LLM events carry the agent in agent_role (CrewAI clears from_agent)
        @crewai_event_bus.on(LLMCallStartedEvent)
        def llm_start(source, event):
            emit("model_request", agent_name=event.agent_role, model_name=model(event.model))

        @crewai_event_bus.on(LLMCallCompletedEvent)
        def llm_end(source, event):
            usage = event.usage or {}
            emit(
                "model_response",
                agent_name=event.agent_role,
                model_name=model(event.model),
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
                total_tokens=int(usage.get("total_tokens") or 0),
                reasoning_chars=0,  # thinking is off
                done_reason=event.finish_reason,
                **self._response_shape(event.response),
            )

        @crewai_event_bus.on(LLMCallFailedEvent)
        def llm_error(source, event):
            emit("model_error", agent_name=event.agent_role, model_name=model(event.model), error_message=event.error)

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def tool_start(source, event):
            new_span()  # a span per call: the analysts' tool calls overlap
            emit(
                "tool_start",
                agent_name=event.agent_role,
                tool_name=event.tool_name,
                tool_args=truncate(event.tool_args),
            )

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def tool_end(source, event):
            emit(
                "tool_end",
                agent_name=event.agent_role,
                tool_name=event.tool_name,
                tool_args=truncate(event.tool_args),
                tool_result=truncate(event.output),
                duration_ms=(event.finished_at - event.started_at).total_seconds() * 1000,
            )

        @crewai_event_bus.on(ToolUsageErrorEvent)
        def tool_error(source, event):
            emit(
                "tool_error",
                agent_name=event.agent_role,
                tool_name=event.tool_name,
                tool_args=truncate(event.tool_args),
                error_message=str(event.error),
            )
