"""Benchmark evaluation for single-agent vs multi-agent."""

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Execute a runner, measure latency, token costs, quality, and citation coverage."""
    started = perf_counter()
    errors: list[str] = []

    try:
        state = runner(query)
    except Exception as exc:
        state = ResearchState(request=query if isinstance(query, object) else None)  # type: ignore
        errors.append(str(exc))

    latency = perf_counter() - started

    # Calculate token costs across all agent executions
    total_cost = sum((res.metadata.get("cost_usd") or 0.0) for res in state.agent_results)

    # Citation coverage estimation
    final_ans = state.final_answer or ""
    citations_found = len(re.findall(r"\[\d+\]", final_ans))
    source_count = len(state.sources)
    citation_coverage = (
        min(1.0, citations_found / max(1, source_count))
        if source_count > 0
        else (1.0 if citations_found > 0 else 0.0)
    )

    # Quality score estimation (scale 0-10)
    quality_score = 5.0
    if len(final_ans) > 200:
        quality_score += 2.0
    if state.analysis_notes:
        quality_score += 1.5
    if citation_coverage > 0.5:
        quality_score += 1.5
    quality_score = min(10.0, quality_score)

    failure_rate = 1.0 if (errors or not state.final_answer) else 0.0

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=total_cost,
        quality_score=quality_score,
        citation_coverage=citation_coverage,
        failure_rate=failure_rate,
        notes=f"Iterations: {state.iteration}, Route: {' -> '.join(state.route_history)}",
    )
    return state, metrics
