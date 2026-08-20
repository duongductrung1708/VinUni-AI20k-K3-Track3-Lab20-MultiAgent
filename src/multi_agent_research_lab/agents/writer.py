"""Writer agent implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = AgentName.WRITER

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        query = state.request.query
        research_notes = state.research_notes or "N/A"
        analysis_notes = state.analysis_notes or "N/A"

        sources_ref = "\n".join(
            [
                f"[{idx + 1}] {doc.title} - {doc.url or 'N/A'}"
                for idx, doc in enumerate(state.sources)
            ]
        )

        logger.info(f"WriterAgent drafting final answer for query: {query}")
        system_prompt = (
            "You are a technical writer drafting a comprehensive research report. "
            "Integrate the research facts and analytical insights into a clear, professional synthesis. "
            "Mandatory rule: Include numerical inline citations (e.g. [1], [2]) pointing to the source references provided."
        )
        user_prompt = (
            f"Query: {query}\n"
            f"Audience: {state.request.audience}\n\n"
            f"Research Notes:\n{research_notes}\n\n"
            f"Analysis Notes:\n{analysis_notes}\n\n"
            f"Available Sources:\n{sources_ref}"
        )

        llm_resp = self.llm_client.complete(system_prompt, user_prompt)

        # Ensure references section is appended if not already present
        final_text = llm_resp.content
        if "References" not in final_text and state.sources:
            final_text += "\n\n### References\n" + sources_ref

        state.final_answer = final_text

        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=final_text,
                metadata={
                    "input_tokens": llm_resp.input_tokens,
                    "output_tokens": llm_resp.output_tokens,
                    "cost_usd": llm_resp.cost_usd,
                },
            )
        )
        state.add_trace_event("writer_complete", {"has_final_answer": True})
        return state
