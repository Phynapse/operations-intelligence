"""Benchmark evaluation harness.

Runs every case in :mod:`app.benchmarks.questions` through the real pipeline and
scores the outcome on four things:

**Routing**
    Did the system choose the analysis type, metric, periods and dimensions the
    case expects? This is the property most worth protecting: a correct
    calculation over the wrong window is still a wrong answer.

**Refusal**
    For cases the platform must refuse, was the question rejected with a
    message that explains why?

**Evidence**
    Did the run actually produce numbers, and does the response say how many it
    checked?

**Grounding**
    Was every number in the narrative traceable to the evidence package?

The harness works with or without an LLM provider. Without one the narrative is
the deterministic one, which is a legitimate configuration: it measures the
platform's own guarantees rather than the model's eloquence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.ai.pipeline import QuestionPipeline
from app.ai.providers.openai_compatible import LLMProvider
from app.benchmarks.questions import (
    BENCHMARK_CASES,
    BenchmarkCase,
    CaseOutcome,
    supported_cases,
    unsupported_cases,
)
from app.core.logging import get_logger
from app.utils.errors import OperationsIntelligenceError

logger = get_logger(__name__)


@dataclass(slots=True)
class EvaluationReport:
    """Aggregate result of one benchmark run."""

    outcomes: tuple[CaseOutcome, ...] = ()
    total_duration_ms: float = 0.0

    @property
    def passed(self) -> tuple[CaseOutcome, ...]:
        return tuple(item for item in self.outcomes if item.passed)

    @property
    def failed(self) -> tuple[CaseOutcome, ...]:
        return tuple(item for item in self.outcomes if not item.passed)

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def pass_rate(self) -> float:
        return len(self.passed) / self.total if self.total else 0.0

    @property
    def all_grounded(self) -> bool:
        return all(item.grounded for item in self.outcomes)

    def describe(self) -> dict[str, object]:
        supported = supported_cases()
        refused = unsupported_cases()
        return {
            "total_cases": self.total,
            "passed": len(self.passed),
            "failed": len(self.failed),
            "pass_rate": round(self.pass_rate, 4),
            "routing_cases": len(supported),
            "routing_failures": sum(
                1 for item in self.outcomes if not item.passed and not item.rejected
            ),
            "refusal_cases": len(refused),
            "refusal_failures": sum(
                1 for item in self.outcomes if item.rejected and not item.passed
            ),
            "all_narratives_grounded": self.all_grounded,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "mean_duration_ms": (
                round(sum(item.duration_ms for item in self.outcomes) / self.total, 2)
                if self.total
                else 0.0
            ),
            "generated_by_llm": sum(1 for item in self.outcomes if item.generated_by == "llm"),
            "generated_by_fallback": sum(
                1 for item in self.outcomes if item.generated_by == "deterministic_fallback"
            ),
        }

    def render(self) -> str:
        lines = ["Benchmark results", "=" * 60]
        for outcome in self.outcomes:
            lines.append(outcome.describe())
        summary = self.describe()
        lines.append("-" * 60)
        for key, value in summary.items():
            lines.append(f"{key:<32} {value}")
        return "\n".join(lines)


async def evaluate(
    cases: tuple[BenchmarkCase, ...] = BENCHMARK_CASES,
    *,
    provider: LLMProvider | None = None,
    pipeline: QuestionPipeline | None = None,
) -> EvaluationReport:
    """Run every case and return a scored report."""
    runner = pipeline or QuestionPipeline(provider=provider)
    outcomes: list[CaseOutcome] = []
    started = time.perf_counter()

    for case in cases:
        outcomes.append(await _run_case(case, runner))

    return EvaluationReport(
        outcomes=tuple(outcomes),
        total_duration_ms=(time.perf_counter() - started) * 1000.0,
    )


async def _run_case(case: BenchmarkCase, runner: QuestionPipeline) -> CaseOutcome:
    started = time.perf_counter()
    try:
        response = await runner.answer(case.question)
    except OperationsIntelligenceError as exc:
        duration = (time.perf_counter() - started) * 1000.0
        if case.expect_rejection:
            failures: list[str] = []
            if case.rejection_hint and case.rejection_hint.lower() not in exc.message.lower():
                failures.append(
                    f"rejected with '{exc.code}' but the message did not mention "
                    f"{case.rejection_hint!r}: {exc.message}"
                )
            return CaseOutcome(
                case_id=case.case_id,
                question=case.question,
                passed=not failures,
                failures=tuple(failures),
                rejected=True,
                error=exc.code,
                duration_ms=duration,
            )
        return CaseOutcome(
            case_id=case.case_id,
            question=case.question,
            passed=False,
            failures=(f"unexpected rejection: {exc.code}: {exc.message}",),
            rejected=True,
            error=exc.code,
            duration_ms=duration,
        )

    duration = (time.perf_counter() - started) * 1000.0
    failures = _check_expectations(case, response)

    return CaseOutcome(
        case_id=case.case_id,
        question=case.question,
        passed=not failures,
        failures=tuple(failures),
        analysis_type=response.plan.analysis_type.value,
        metric=response.plan.metric.value,
        period=response.plan.period,
        comparison_period=response.plan.comparison_period,
        evidence_count=len(response.evidence),
        findings_count=len(response.findings),
        generated_by=response.generated_by,
        grounded=response.grounding.grounded and not response.unsupported_claims_detected,
        duration_ms=duration,
        unsupported_numbers=response.grounding.unsupported,
        extras={
            "dimensions": [item.value for item in response.plan.dimensions],
            "kpi_count": len(response.kpis),
            "caveats": len(response.caveats),
        },
    )


def _check_expectations(case: BenchmarkCase, response: object) -> list[str]:
    """Compare a response against every expectation the case states."""
    failures: list[str] = []

    plan = response.plan  # type: ignore[attr-defined]

    if plan.analysis_type is not case.expected_analysis_type:
        failures.append(
            f"analysis_type was '{plan.analysis_type.value}', expected "
            f"'{case.expected_analysis_type.value}'"
        )
    if plan.metric is not case.expected_metric:
        failures.append(
            f"metric was '{plan.metric.value}', expected '{case.expected_metric.value}'"
        )
    if case.expected_period is not None and plan.period != case.expected_period:
        failures.append(f"period was '{plan.period}', expected '{case.expected_period}'")
    if (
        case.expected_comparison_period is not None
        and plan.comparison_period != case.expected_comparison_period
    ):
        failures.append(
            f"comparison_period was '{plan.comparison_period}', expected "
            f"'{case.expected_comparison_period}'"
        )
    if set(plan.dimensions) != set(case.expected_dimensions):
        failures.append(
            f"dimensions were {sorted(item.value for item in plan.dimensions)}, expected "
            f"{sorted(item.value for item in case.expected_dimensions)}"
        )

    evidence = response.evidence  # type: ignore[attr-defined]
    if len(evidence) < case.min_evidence:
        failures.append(
            f"produced {len(evidence)} evidence item(s), expected at least {case.min_evidence}"
        )

    grounding = response.grounding  # type: ignore[attr-defined]
    if not grounding.grounded:
        failures.append(f"{grounding.unsupported} cited number(s) had no matching evidence value")

    if not response.answer.strip():  # type: ignore[attr-defined]
        failures.append("returned an empty answer")

    return failures


@dataclass(slots=True)
class RoutingCoverage:
    """Which analysis types the benchmark set actually exercises."""

    covered: frozenset[str] = field(default_factory=frozenset)
    missing: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.missing


def routing_coverage() -> RoutingCoverage:
    """Report analysis types with no benchmark case."""
    from app.analytics.registry import supported_analysis_types

    covered = {case.expected_analysis_type.value for case in supported_cases()}
    every = set(supported_analysis_types())
    return RoutingCoverage(
        covered=frozenset(covered),
        missing=tuple(sorted(every - covered)),
    )


__all__ = [
    "EvaluationReport",
    "RoutingCoverage",
    "evaluate",
    "routing_coverage",
]
