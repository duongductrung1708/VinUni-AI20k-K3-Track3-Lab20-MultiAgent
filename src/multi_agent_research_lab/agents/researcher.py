"""Researcher agent implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = AgentName.RESEARCHER

    def __init__(
        self,
        search_client: SearchClient | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.search_client = search_client or SearchClient()
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        query = state.request.query
        max_sources = state.request.max_sources

        logger.info(f"ResearcherAgent searching for query: {query}")
        sources = self.search_client.search(query, max_results=max_sources)
        state.sources = sources

        snippets = "\n\n".join(
            [
                f"Source [{idx + 1}] ({doc.title} - {doc.url}):\n{doc.snippet}"
                for idx, doc in enumerate(sources)
            ]
        )

        system_prompt = (
            "You are an expert technical researcher. Summarize key facts, definitions, "
            "and architectural concepts from the provided sources into structured research notes."
        )
        user_prompt = f"Target Query: {query}\n\nSources:\n{snippets}"

        llm_resp = self.llm_client.complete(system_prompt, user_prompt)
        state.research_notes = llm_resp.content

        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=llm_resp.content,
                metadata={
                    "num_sources": len(sources),
                    "input_tokens": llm_resp.input_tokens,
                    "output_tokens": llm_resp.output_tokens,
                    "cost_usd": llm_resp.cost_usd,
                },
            )
        )
        state.add_trace_event("researcher_complete", {"num_sources": len(sources)})
        return state
