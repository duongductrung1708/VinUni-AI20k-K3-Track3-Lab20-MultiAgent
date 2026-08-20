"""Supervisor / router implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = AgentName.SUPERVISOR

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route."""
        settings = get_settings()

        # Stop condition: reached max iteration cap
        if state.iteration >= settings.max_iterations:
            logger.info("Supervisor reached max iterations; routing to done.")
            next_route = "done"
        elif state.final_answer is not None and len(state.final_answer.strip()) > 0:
            next_route = "done"
        elif not state.sources or not state.research_notes:
            next_route = AgentName.RESEARCHER
        elif not state.analysis_notes:
            next_route = AgentName.ANALYST
        elif state.final_answer is None:
            next_route = AgentName.WRITER
        else:
            next_route = "done"

        state.record_route(next_route)
        state.add_trace_event(
            "supervisor_decision",
            {
                "next_route": next_route,
                "iteration": state.iteration,
                "has_sources": bool(state.sources),
                "has_research_notes": bool(state.research_notes),
                "has_analysis_notes": bool(state.analysis_notes),
                "has_final_answer": bool(state.final_answer),
            },
        )
        return state
