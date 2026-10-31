"""CrewAI LLMs for local Ollama models."""

import litellm
from crewai import LLM

from .config import AgentSettings, OllamaSettings


class OllamaLLMFactory:
    """Builds CrewAI LLMs for Ollama models, fixing three CrewAI defaults.

    - Provider: CrewAI's native Ollama provider uses the OpenAI-compatible
      /v1 endpoint, which ignores num_ctx and loads the models with a 4096
      context. LiteLLM's `ollama_chat/` provider uses Ollama's own API.
    - Tool calling: LiteLLM's catalog doesn't know local models and reports
      no function calling, so CrewAI falls back to ReAct text, one tool call
      per turn: the coordinator would delegate to one analyst at a time.
      Ollama does parse their tool calls, so it is declared here.
    - Context window: CrewAI assumes 8192 for unknown models and replaces a
      tool result above 25% of it (one fetched page) with a preview the agent
      must read back with an extra tool call, which eats the call budget.
    """

    PROVIDER = "ollama_chat"
    # share of the context CrewAI keeps for itself on top of the declared window
    CONTEXT_RESERVE = 0.15

    def __init__(self, ollama: OllamaSettings):
        self.base_url = ollama.base_url

    def create(self, agent: AgentSettings) -> LLM:
        model = f"{self.PROVIDER}/{agent.model}"
        litellm.register_model(
            {model: {"litellm_provider": self.PROVIDER, "mode": "chat", "supports_function_calling": True}}
        )
        llm = LLM(
            model=model,
            is_litellm=True,
            base_url=self.base_url,
            temperature=agent.temperature,
            num_ctx=agent.num_ctx,
            # LiteLLM sends "none" to Ollama as think=false
            reasoning_effort=None if agent.think else "none",
            max_tokens=agent.max_tokens,  # mapped to Ollama's num_predict
            timeout=agent.timeout,  # LiteLLM's default for Ollama is 600 s
            num_retries=agent.num_retries,
        )
        llm.context_window_size = int(agent.num_ctx * (1 - self.CONTEXT_RESERVE))
        return llm
