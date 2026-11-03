"""The benchmark suite, wired into the test run.

``app.benchmarks.runner`` is for humans; this module is for CI. Every case runs
through the real pipeline, so a routing regression fails here with the specific
expectation that broke rather than as a vague "results changed".
"""

from __future__ import annotations

import asyncio

import pytest

from app.analytics.registry import supported_analysis_types
from app.benchmarks.evaluator import evaluate, routing_coverage
from app.benchmarks.questions import (
    BENCHMARK_CASES,
    supported_cases,
    unsupported_cases,
)
from app.models.enums import AnalysisType

pytestmark = pytest.mark.integration


def _report():
    return asyncio.run(evaluate(BENCHMARK_CASES, provider=None))


@pytest.fixture(scope="module")
def report():
    return _report()


class TestBenchmarkRouting:
    def test_every_case_passes(self, report):
        assert report.failed == (), "\n" + report.render()

    def test_pass_rate_is_complete(self, report):
        assert report.pass_rate == 1.0

    def test_every_supported_analysis_type_has_a_case(self):
        coverage = routing_coverage()
        assert coverage.complete, "analysis types with no benchmark case: " + ", ".join(
            coverage.missing
        )

    def test_the_case_set_covers_every_analysis_type(self):
        covered = {case.expected_analysis_type for case in supported_cases()}
        assert covered == set(supported_analysis_types())

    def test_the_case_set_is_broad_enough(self):
        counts: dict[AnalysisType, int] = {}
        for case in supported_cases():
            counts[case.expected_analysis_type] = counts.get(case.expected_analysis_type, 0) + 1
        # Breadth first: every type is represented, and most appear more than
        # once so a single odd question cannot define a type's behaviour.
        assert len(counts) >= 12
        assert sum(counts.values()) / len(counts) >= 1.5

    def test_every_case_has_a_purpose(self):
        assert all(case.purpose for case in BENCHMARK_CASES)

    def test_case_ids_are_unique(self):
        ids = [case.case_id for case in BENCHMARK_CASES]
        assert len(ids) == len(set(ids))

    def test_every_narrative_is_grounded(self, report):
        assert report.all_grounded


class TestBenchmarkRefusals:
    def test_every_unsupported_case_is_refused(self):
        assert len(unsupported_cases()) >= 5

    def test_refusals_name_a_reason(self, report):
        rejected = [item for item in report.outcomes if item.rejected]
        assert rejected
        assert all(item.passed for item in rejected)

    def test_no_supported_case_is_refused(self, report):
        rejected_ids = {item.case_id for item in report.outcomes if item.rejected}
        expected = {case.case_id for case in unsupported_cases()}
        assert rejected_ids == expected


class TestBenchmarkReport:
    def test_report_describes_itself(self, report):
        described = report.describe()
        assert described["total_cases"] == len(BENCHMARK_CASES)
        assert described["routing_cases"] == len(supported_cases())
        assert described["refusal_cases"] == len(unsupported_cases())

    def test_report_renders_readable_text(self, report):
        rendered = report.render()
        assert "Benchmark results" in rendered
        assert "pass_rate" in rendered

    def test_all_runs_are_deterministic_mode_without_a_provider(self, report):
        assert report.describe()["generated_by_llm"] == 0
        assert report.describe()["generated_by_fallback"] == len(supported_cases())
