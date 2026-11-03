"""Benchmark suite: a fixed set of operational questions with expected outcomes.

Run it directly to see a scored report::

    python -m app.benchmarks.runner

The same harness backs the benchmark tests, so a routing regression fails the
test suite rather than waiting for someone to notice.
"""

from app.benchmarks.evaluator import EvaluationReport, RoutingCoverage, evaluate, routing_coverage
from app.benchmarks.questions import (
    BENCHMARK_CASES,
    BenchmarkCase,
    CaseOutcome,
    case_by_id,
    supported_cases,
    unsupported_cases,
)

__all__ = [
    "BENCHMARK_CASES",
    "BenchmarkCase",
    "CaseOutcome",
    "EvaluationReport",
    "RoutingCoverage",
    "case_by_id",
    "evaluate",
    "routing_coverage",
    "supported_cases",
    "unsupported_cases",
]
