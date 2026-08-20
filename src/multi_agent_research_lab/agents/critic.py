"""Critic agent implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = AgentName.CRITIC

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings."""
        if not state.final_answer:
            return state

        logger.info("CriticAgent evaluating final answer quality and citation coverage.")
        system_prompt = (
            "You are a quality assurance critic. Review the research answer against the sources and research notes. "
            "Verify accuracy, citation completeness, and safety."
        )
        user_prompt = (
            f"Final Answer:\n{state.final_answer}\n\nSources Count: {len(state.sources)}\n"
        )

        llm_resp = self.llm_client.complete(system_prompt, user_prompt)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=llm_resp.content,
                metadata={
                    "input_tokens": llm_resp.input_tokens,
                    "output_tokens": llm_resp.output_tokens,
                    "cost_usd": llm_resp.cost_usd,
                },
            )
        )
        state.add_trace_event("critic_complete", {"passed": True})
        return state
