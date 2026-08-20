"""Unit tests for multi-agent components."""

from multi_agent_research_lab.agents import (
    AnalystAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


def test_supervisor_routing_policy() -> None:
    supervisor = SupervisorAgent()

    # Step 1: Initial state -> Route to Researcher
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state = supervisor.run(state)
    assert state.route_history[-1] == AgentName.RESEARCHER

    # Step 2: Sources present -> Route to Analyst
    state.sources = [SourceDocument(title="Doc1", snippet="Snippet1")]
    state.research_notes = "Research summary notes"
    state = supervisor.run(state)
    assert state.route_history[-1] == AgentName.ANALYST

    # Step 3: Analysis present -> Route to Writer
    state.analysis_notes = "Analytical summary notes"
    state = supervisor.run(state)
    assert state.route_history[-1] == AgentName.WRITER

    # Step 4: Final answer present -> Route to Done
    state.final_answer = "Comprehensive final answer [1]."
    state = supervisor.run(state)
    assert state.route_history[-1] == "done"


def test_supervisor_max_iterations_cap() -> None:
    supervisor = SupervisorAgent()
    state = ResearchState(request=ResearchQuery(query="Test query"), iteration=10)
    state = supervisor.run(state)
    assert state.route_history[-1] == "done"


def test_worker_agents_execution() -> None:
    state = ResearchState(request=ResearchQuery(query="Research GraphRAG state-of-the-art"))

    researcher = ResearcherAgent()
    state = researcher.run(state)
    assert len(state.sources) > 0
    assert state.research_notes is not None

    analyst = AnalystAgent()
    state = analyst.run(state)
    assert state.analysis_notes is not None

    writer = WriterAgent()
    state = writer.run(state)
    assert state.final_answer is not None


def test_multi_agent_workflow_end_to_end() -> None:
    workflow = MultiAgentWorkflow()
    initial_state = ResearchState(request=ResearchQuery(query="Research GraphRAG state-of-the-art"))
    final_state = workflow.run(initial_state)

    assert final_state.final_answer is not None
    assert "done" in final_state.route_history
    assert len(final_state.sources) > 0
