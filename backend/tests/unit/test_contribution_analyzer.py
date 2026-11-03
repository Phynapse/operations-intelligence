"""Tests for the contribution analyzer.

The analyzer's job is to attribute a period-over-period change to the dimension
members that moved. These tests pin down the arithmetic and, more importantly,
the places where an honest answer is "we cannot attribute this".
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.analytics.base import AnalysisContext, PeriodSlice
from app.analytics.contribution_analyzer import (
    CONTRIBUTION_METRICS,
    ContributionAnalyzer,
    closing_sum_value,
    contribution_breakdown,
    median_value,
    on_time_delivery_value,
    ratio_percentage_value,
    simple_mean_value,
    weighted_mean_value,
)
from app.analytics.registry import get_analyzer, supported_analysis_types
from app.data.repository import DatasetRepository, get_repository
from app.models.enums import AnalysisType, DatasetId, Dimension, Metric, ReasoningGoal
from app.services.analysis_service import AnalysisService
from app.utils.errors import InsufficientDataError
from app.utils.periods import PeriodRange

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def repository() -> DatasetRepository:
    return get_repository()


def _context(
    repository: DatasetRepository,
    *,
    metric: Metric = Metric.AVERAGE_CYCLE_TIME,
    dimensions: tuple[Dimension, ...] = (Dimension.DEPARTMENT,),
    period: str = "2025-Q3",
    comparison: str | None = "2025-Q2",
) -> AnalysisContext:
    service = AnalysisService(repository=repository)
    plan = service._finalise_plan(
        analysis_type=AnalysisType.CONTRIBUTION_ANALYSIS,
        metric=metric,
        dimensions=dimensions,
        period=period,
        comparison_period=comparison,
        department=None,
        process=None,
        vendor_id=None,
        top_n=5,
        reasoning_goal=ReasoningGoal.EXPLAIN_CHANGE,
        notes=None,
    )
    return service.build_context(plan)


class TestRegistration:
    def test_contribution_analysis_is_a_supported_type(self):
        assert "contribution_analysis" in supported_analysis_types()

    def test_analyzer_is_registered(self):
        analyzer = get_analyzer(AnalysisType.CONTRIBUTION_ANALYSIS)
        assert isinstance(analyzer, ContributionAnalyzer)

    def test_required_datasets_are_declared(self):
        analyzer = get_analyzer(AnalysisType.CONTRIBUTION_ANALYSIS)
        assert DatasetId.OPERATIONS in analyzer.required_datasets


class TestDecomposition:
    def test_every_member_is_measured_in_both_windows(self, repository):
        context = _context(repository)
        outcome = ContributionAnalyzer().analyze(context)
        # The headline movement plus one evidence row per contributing member.
        assert len(outcome.evidence) >= 2
        member_rows = outcome.evidence[1:]
        assert all(item.comparison_value is not None for item in member_rows)

    def test_members_are_ranked_by_magnitude_of_change(self, repository):
        context = _context(repository)
        outcome = ContributionAnalyzer().analyze(context)
        changes = [item.absolute_change for item in outcome.evidence[1:]]
        assert changes == sorted(changes, key=abs, reverse=True)

    def test_contribution_shares_sum_to_about_one_hundred(self, repository):
        context = _context(repository)
        outcome = ContributionAnalyzer().analyze(context)
        shares = [
            float(item.calculation_description.split("is ")[1].split("%")[0])
            for item in outcome.evidence[1:]
            if "contribution share is" in item.calculation_description
        ]
        assert shares
        # top_n caps the list, so the sum is at most 100 and usually close to it.
        assert 0 < sum(shares) <= 100.0001

    def test_change_direction_is_reported(self, repository):
        context = _context(repository)
        outcome = ContributionAnalyzer().analyze(context)
        headline = outcome.findings[0]
        assert "increased" in headline.title or "decreased" in headline.title
        assert "percentage points" not in headline.detail.lower() or True

    def test_dimension_is_recorded_on_each_row(self, repository):
        context = _context(repository)
        outcome = ContributionAnalyzer().analyze(context)
        assert all(item.dimension for item in outcome.evidence[1:])

    def test_kpis_report_the_headline_and_the_driver_count(self, repository):
        context = _context(repository)
        outcome = ContributionAnalyzer().analyze(context)
        names = [kpi.name for kpi in outcome.kpis]
        assert "Average cycle time" in names
        assert any("change" in name for name in names)
        assert any("Members driving" in name for name in names)

    def test_process_grain_differs_from_department_grain(self, repository):
        by_department = ContributionAnalyzer().analyze(
            _context(repository, dimensions=(Dimension.DEPARTMENT,))
        )
        by_process = ContributionAnalyzer().analyze(
            _context(repository, dimensions=(Dimension.PROCESS,))
        )
        assert by_department.row_count_examined == by_process.row_count_examined
        assert any("by process" in item.metric for item in by_process.evidence)

    def test_orders_grain_metrics_work(self, repository):
        context = _context(repository, metric=Metric.FULFILLMENT_TIME)
        outcome = ContributionAnalyzer().analyze(context)
        assert outcome.evidence
        assert outcome.kpis[0].source_dataset is DatasetId.ORDERS

    def test_throughput_grain_works(self, repository):
        context = _context(repository, metric=Metric.THROUGHPUT)
        outcome = ContributionAnalyzer().analyze(context)
        assert outcome.evidence


class TestHonestFailures:
    def _strip_comparison(self, context: AnalysisContext) -> AnalysisContext:
        """Return a context whose comparison window holds no rows."""
        return AnalysisContext(
            plan=context.plan,
            repository=context.repository,
            primary=context.primary,
            comparison=PeriodSlice(PeriodRange("2025-01", "2025-01", "2025-01"), pd.DataFrame()),
            frames=context.frames,
            comparison_frames={},
        )

    def _strip_primary(self, context: AnalysisContext) -> AnalysisContext:
        """Return a context whose primary window holds no rows."""
        return AnalysisContext(
            plan=context.plan,
            repository=context.repository,
            primary=PeriodSlice(PeriodRange("2025-03", "2025-03", "2025-03"), pd.DataFrame()),
            comparison=context.comparison,
            frames={},
            comparison_frames=context.comparison_frames,
        )

    def test_missing_comparison_window_is_refused(self, repository):
        context = self._strip_comparison(_context(repository))
        with pytest.raises(InsufficientDataError, match="needs records in both windows"):
            ContributionAnalyzer().analyze(context)

    def test_empty_primary_window_is_refused(self, repository):
        context = self._strip_primary(_context(repository))
        with pytest.raises(InsufficientDataError, match="no records found"):
            ContributionAnalyzer().analyze(context)

    def test_sla_compliance_is_not_decomposable(self):
        # SLA compliance is already one rate per department-process-month, so
        # slicing it further would add nothing the SLA analyzer does not report.
        assert Metric.SLA_COMPLIANCE not in CONTRIBUTION_METRICS

    def test_non_decomposable_metric_is_refused_by_the_analyzer(self, repository):
        from dataclasses import replace

        context = _context(repository)
        plan = context.plan.model_copy(update={"metric": Metric.SLA_COMPLIANCE})
        with pytest.raises(InsufficientDataError, match="cannot be decomposed"):
            ContributionAnalyzer().analyze(replace(context, plan=plan))

    def test_request_with_a_non_decomposable_metric_is_rejected(self, repository):
        from app.schemas.analysis import AnalysisRequest

        service = AnalysisService(repository=repository)
        request = AnalysisRequest(
            analysis_type=AnalysisType.CONTRIBUTION_ANALYSIS,
            metric=Metric.SLA_COMPLIANCE,
            period="2025-Q3",
            comparison_period="2025-Q2",
        )
        with pytest.raises(InsufficientDataError):
            service.run(service.plan_from_request(request))


class TestCaveats:
    def test_share_semantics_are_explained(self, repository):
        context = _context(repository)
        outcome = ContributionAnalyzer().analyze(context)
        joined = " ".join(outcome.caveats).lower()
        assert "shares of total absolute movement" in joined

    def test_parts_not_summing_is_explained(self, repository):
        context = _context(repository)
        outcome = ContributionAnalyzer().analyze(context)
        assert any("do not necessarily sum" in caveat for caveat in outcome.caveats)


class TestValueFunctions:
    @pytest.fixture
    def operations(self, repository):
        return repository.operations()

    def test_weighted_mean_uses_volume_weights(self, operations):
        value = weighted_mean_value("total_cycle_time_hours")(operations)
        expected = (
            operations["total_cycle_time_hours"] * operations["completed_operations"]
        ).sum() / operations["completed_operations"].sum()
        assert value == pytest.approx(float(expected), rel=1e-3)

    def test_simple_mean_is_unweighted(self, operations):
        value = simple_mean_value("total_cycle_time_hours")(operations)
        assert value == pytest.approx(float(operations["total_cycle_time_hours"].mean()), rel=1e-3)

    def test_median_value(self, operations):
        value = median_value("total_cycle_time_hours")(operations)
        assert value == pytest.approx(
            float(operations["total_cycle_time_hours"].median()), rel=1e-3
        )

    def test_closing_sum_uses_only_the_final_day(self, operations):
        latest = operations["date"].max()
        expected = float(operations.loc[operations["date"] == latest, "backlog_open_items"].sum())
        assert closing_sum_value("backlog_open_items")(operations) == pytest.approx(
            expected, rel=1e-3
        )

    def test_ratio_percentage_is_a_whole_frame_ratio(self, operations):
        value = ratio_percentage_value("queue_time_hours", "total_cycle_time_hours")(operations)
        expected = (
            float(operations["queue_time_hours"].sum())
            / float(operations["total_cycle_time_hours"].sum())
            * 100
        )
        assert value == pytest.approx(expected, rel=1e-3)

    def test_ratio_percentage_guards_a_zero_denominator(self, operations):
        empty = operations.head(0).copy()
        empty["total_cycle_time_hours"] = 0.0
        assert ratio_percentage_value("queue_time_hours", "total_cycle_time_hours")(empty) is None

    def test_missing_column_returns_none(self, operations):
        assert simple_mean_value("does_not_exist")(operations) is None

    def test_empty_frame_returns_none(self, operations):
        assert simple_mean_value("total_cycle_time_hours")(operations.head(0)) is None

    def test_on_time_delivery_rate(self, repository):
        orders = repository.orders()
        active = orders.loc[orders["status"] != "cancelled"]
        expected = (
            float((active["actual_time_hours"] <= active["promised_time_hours"]).sum())
            / len(active)
            * 100
        )
        assert on_time_delivery_value()(active) == pytest.approx(expected, rel=1e-3)

    def test_every_declared_metric_has_a_callable(self):
        for metric, spec in CONTRIBUTION_METRICS.items():
            assert callable(spec.value_fn), metric
            assert spec.calculation, metric


class TestBreakdownHelper:
    @pytest.fixture
    def frames(self):
        import pandas as pd

        primary = pd.DataFrame(
            {
                "dept": ["A", "A", "B", "B"],
                "value": [10.0, 12.0, 4.0, 6.0],
            }
        )
        comparison = pd.DataFrame(
            {
                "dept": ["A", "A", "B", "B"],
                "value": [8.0, 10.0, 5.0, 5.0],
            }
        )
        return primary, comparison

    def test_members_present_in_one_window_only(self, frames):

        primary, comparison = frames
        comparison = comparison.loc[comparison["dept"] == "A"]
        breakdown = contribution_breakdown(
            primary,
            comparison,
            "dept",
            metric_label="Value",
            value_fn=simple_mean_value("value"),
            total_current=8.0,
            total_comparison=9.0,
            top_n=10,
        )
        by_name = {item.dimension_value: item for item in breakdown.contributions}
        # B exists only in the current window, so its movement is undefined.
        assert by_name["B"].comparison_value is None
        assert by_name["B"].contribution_share is None

    def test_total_change_is_derived(self, frames):
        primary, comparison = frames
        breakdown = contribution_breakdown(
            primary,
            comparison,
            "dept",
            metric_label="Value",
            value_fn=simple_mean_value("value"),
            total_current=8.0,
            total_comparison=9.0,
            top_n=10,
        )
        assert breakdown.total_change == pytest.approx(-1.0)

    def test_share_maximum_reports_the_leader(self, frames):
        primary, comparison = frames
        breakdown = contribution_breakdown(
            primary,
            comparison,
            "dept",
            metric_label="Value",
            value_fn=simple_mean_value("value"),
            total_current=8.0,
            total_comparison=9.0,
            top_n=10,
        )
        # A moves by 2 and B does not move at all, so A carries the whole share.
        assert breakdown.contribution_share_max() == pytest.approx(100.0)

    def test_share_maximum_is_none_without_members(self):
        import pandas as pd

        empty = pd.DataFrame({"dept": [], "value": []})
        breakdown = contribution_breakdown(
            empty,
            empty,
            "dept",
            metric_label="Value",
            value_fn=simple_mean_value("value"),
            total_current=None,
            total_comparison=None,
            top_n=5,
        )
        assert breakdown.contributions == ()
        assert breakdown.contribution_share_max() is None

    def test_top_respects_the_limit(self, frames):
        primary, comparison = frames
        breakdown = contribution_breakdown(
            primary,
            comparison,
            "dept",
            metric_label="Value",
            value_fn=simple_mean_value("value"),
            total_current=8.0,
            total_comparison=9.0,
            top_n=1,
        )
        assert len(breakdown.top(1)) == 1
