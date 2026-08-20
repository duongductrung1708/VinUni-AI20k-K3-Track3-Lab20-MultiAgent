"""LangGraph workflow implementation."""

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(self) -> None:
        self.supervisor = SupervisorAgent()
        self.researcher = ResearcherAgent()
        self.analyst = AnalystAgent()
        self.writer = WriterAgent()
        self.critic = CriticAgent()
        self._compiled_graph: Any = None

    def build(self) -> StateGraph:
        """Create and configure a LangGraph StateGraph."""
        builder = StateGraph(ResearchState)

        builder.add_node(AgentName.SUPERVISOR, self.supervisor.run)
        builder.add_node(AgentName.RESEARCHER, self.researcher.run)
        builder.add_node(AgentName.ANALYST, self.analyst.run)
        builder.add_node(AgentName.WRITER, self.writer.run)
        builder.add_node(AgentName.CRITIC, self.critic.run)

        builder.set_entry_point(AgentName.SUPERVISOR)

        def route_next(state: ResearchState) -> str:
            if isinstance(state, dict):
                history = state.get("route_history", [])
            else:
                history = state.route_history

            if not history:
                return "done"
            last = history[-1]
            if last in (
                AgentName.RESEARCHER,
                AgentName.ANALYST,
                AgentName.WRITER,
                AgentName.CRITIC,
            ):
                return last
            return "done"

        builder.add_conditional_edges(
            AgentName.SUPERVISOR,
            route_next,
            {
                AgentName.RESEARCHER: AgentName.RESEARCHER,
                AgentName.ANALYST: AgentName.ANALYST,
                AgentName.WRITER: AgentName.WRITER,
                AgentName.CRITIC: AgentName.CRITIC,
                "done": END,
            },
        )

        builder.add_edge(AgentName.RESEARCHER, AgentName.SUPERVISOR)
        builder.add_edge(AgentName.ANALYST, AgentName.SUPERVISOR)
        builder.add_edge(AgentName.WRITER, AgentName.SUPERVISOR)
        builder.add_edge(AgentName.CRITIC, AgentName.SUPERVISOR)

        return builder

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""
        if self._compiled_graph is None:
            builder = self.build()
            self._compiled_graph = builder.compile()

        logger.info(f"Starting MultiAgentWorkflow execution for query: {state.request.query}")
        result = self._compiled_graph.invoke(state)

        if isinstance(result, ResearchState):
            return result

        if isinstance(result, dict):
            return ResearchState(**result)

        return state
