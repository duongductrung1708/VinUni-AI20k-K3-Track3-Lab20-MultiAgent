"""Command-line entrypoint for the lab starter."""

from time import perf_counter
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline LLM call."""

    _init()
    request = _parse_query(query)
    state = ResearchState(request=request)

    start_t = perf_counter()
    llm = LLMClient()
    sys_prompt = "You are a helpful research assistant. Provide a direct, comprehensive summary answering the user query."
    resp = llm.complete(sys_prompt, request.query)
    elapsed = perf_counter() - start_t

    state.final_answer = resp.content
    state.agent_results.append(
        AgentResult(
            agent=AgentName.SUPERVISOR,
            content=resp.content,
            metadata={
                "latency_seconds": elapsed,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "cost_usd": resp.cost_usd,
            },
        )
    )

    console.print(
        Panel(state.final_answer, title="[bold green]Single-Agent Baseline Answer[/bold green]")
    )

    stats_table = Table(title="Execution Metrics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="magenta")
    stats_table.add_row("Latency", f"{elapsed:.2f}s")
    stats_table.add_row("Input Tokens", str(resp.input_tokens or 0))
    stats_table.add_row("Output Tokens", str(resp.output_tokens or 0))
    stats_table.add_row("Est. Cost (USD)", f"${(resp.cost_usd or 0):.6f}")
    console.print(stats_table)


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow system."""

    _init()
    state = ResearchState(request=_parse_query(query))
    workflow = MultiAgentWorkflow()

    start_t = perf_counter()
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    elapsed = perf_counter() - start_t

    console.print(
        Panel(
            result.final_answer or "No answer generated.",
            title="[bold blue]Multi-Agent Final Synthesis[/bold blue]",
        )
    )

    info_table = Table(title="Multi-Agent Run Details")
    info_table.add_column("Field", style="cyan")
    info_table.add_column("Value", style="yellow")
    info_table.add_row("Total Latency", f"{elapsed:.2f}s")
    info_table.add_row("Route History", " -> ".join(result.route_history))
    info_table.add_row("Total Sources Collected", str(len(result.sources)))
    info_table.add_row("Agent Iterations", str(result.iteration))
    console.print(info_table)


if __name__ == "__main__":
    app()
