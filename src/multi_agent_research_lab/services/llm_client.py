"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import logging
from dataclasses import dataclass
from typing import Any

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client implementation."""

    def __init__(self, model_name: str | None = None, api_key: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.openai_model
        self.api_key = api_key or settings.openai_api_key
        self._client: Any = None

        if self.api_key and self.api_key.startswith("sk-"):
            try:
                import openai

                self._client = openai.OpenAI(api_key=self.api_key)
            except Exception as exc:
                logger.warning(f"Failed to initialize OpenAI client: {exc}")

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion with token usage and cost estimation."""
        if self._client:
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                )
                content = response.choices[0].message.content or ""
                in_tokens = response.usage.prompt_tokens if response.usage else 0
                out_tokens = response.usage.completion_tokens if response.usage else 0

                # Pricing model for gpt-4o-mini ($0.15 / 1M prompt, $0.60 / 1M completion)
                if "gpt-4o-mini" in self.model_name:
                    cost = (in_tokens * 0.15 + out_tokens * 0.60) / 1_000_000
                else:
                    cost = (in_tokens * 2.50 + out_tokens * 10.00) / 1_000_000

                return LLMResponse(
                    content=content,
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                    cost_usd=cost,
                )
            except Exception as exc:
                logger.warning(
                    f"OpenAI API call failed: {exc}. Falling back to deterministic mock completion."
                )

        # Deterministic mock fallback for offline / test environments
        mock_content = f"Synthesized analysis based on system prompt ({system_prompt[:30]}...) and query: {user_prompt}"
        in_tokens = len((system_prompt + user_prompt).split())
        out_tokens = len(mock_content.split())
        return LLMResponse(
            content=mock_content,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            cost_usd=0.00001,
        )
