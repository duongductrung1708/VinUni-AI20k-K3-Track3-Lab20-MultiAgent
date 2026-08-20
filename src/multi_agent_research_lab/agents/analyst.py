"""Analyst agent implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = AgentName.ANALYST

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        research_notes = state.research_notes or "No research notes collected."
        query = state.request.query

        logger.info(f"AnalystAgent analyzing research notes for query: {query}")
        system_prompt = (
            "You are a senior analytical AI expert. Evaluate research notes, extract key technical claims, "
            "compare trade-offs, assess evidence reliability, and identify any gaps or limitations."
        )
        user_prompt = (
            f"Original Query: {query}\n\n"
            f"Audience: {state.request.audience}\n\n"
            f"Research Notes:\n{research_notes}"
        )

        llm_resp = self.llm_client.complete(system_prompt, user_prompt)
        state.analysis_notes = llm_resp.content

        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=llm_resp.content,
                metadata={
                    "input_tokens": llm_resp.input_tokens,
                    "output_tokens": llm_resp.output_tokens,
                    "cost_usd": llm_resp.cost_usd,
                },
            )
        )
        state.add_trace_event("analyst_complete", {"has_analysis": True})
        return state
