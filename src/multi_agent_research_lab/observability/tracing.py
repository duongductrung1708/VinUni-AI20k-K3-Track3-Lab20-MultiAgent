"""Tracing hooks.

Supports LangSmith, Langfuse, OpenTelemetry, and structured JSON traces.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Span context supporting local telemetry and optional provider integration."""
    settings = get_settings()
    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "attributes": attributes or {},
        "duration_seconds": None,
        "langsmith_project": settings.langsmith_project if settings.langsmith_api_key else None,
    }

    try:
        yield span
    finally:
        span["duration_seconds"] = perf_counter() - started
        logger.debug(f"[TRACE] {name} completed in {span['duration_seconds']:.4f}s")
