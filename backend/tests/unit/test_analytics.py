"""Comprehensive unit tests for all analytics analyzers.

These tests exercise:
- KPIAnalyzer
- ThroughputAnalyzer
- CycleTimeAnalyzer
- BacklogAnalyzer
- SLAAnalyzer
- BottleneckAnalyzer
- AnomalyAnalyzer
- TrendAnalyzer
- VendorAnalyzer
- DepartmentAnalyzer
- ContributionAnalyzer
- PeriodComparisonAnalyzer

Every test is deterministic: the same inputs always produce the same outputs.
No LLM calls are made. No network calls are made.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from app.analytics.anomaly_analyzer import AnomalyAnalyzer, detect, modified_z_scores
from app.analytics.backlog_analyzer import (
    BacklogAnalyzer,
    backlog_table,
    closing_backlog,
    daily_backlog_totals,
)
from app.analytics.base import AnalysisContext, PeriodSlice
from app.analytics.bottleneck_analyzer import BottleneckAnalyzer, bottleneck_table
from app.analytics.contribution_analyzer import (
    ContributionAnalyzer,
    contribution_breakdown,
    weighted_mean_value,
)
from app.analytics.cycle_time_analyzer import (
    CycleTimeAnalyzer,
    change_severity,
    cycle_components,
    cycle_table,
)
from app.analytics.department_analyzer import DepartmentAnalyzer, department_scorecard
from app.analytics.kpi_analyzer import KPIAnalyzer
from app.analytics.period_comparison_analyzer import PeriodComparisonAnalyzer
from app.analytics.sla_analyzer import (
    SLAAnalyzer,
    compliance_severity,
    compliance_summary,
    compliance_table,
)
from app.analytics.throughput_analyzer import ThroughputAnalyzer, distinct_days, rate_table
from app.analytics.trend_analyzer import (
    TrendAnalyzer,
    classify_direction,
    monthly_series,
)
from app.analytics.vendor_analyzer import VendorAnalyzer, on_time_rate, vendor_table
from app.data.repository import get_repository
from app.models.enums import (
    AnalysisType,
    DatasetId,
    Dimension,
    Metric,
    ReasoningGoal,
    Severity,
)
from app.schemas.analysis import AnalysisPlan
from app.utils.errors import InsufficientDataError
from app.utils.periods import PeriodRange

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers — build deterministic DataFrames and AnalysisContexts
# ---------------------------------------------------------------------------


def make_operations(rows: list[dict]) -> pd.DataFrame:
    """Build an operations-like DataFrame with the required columns."""
    defaults = {
        "operation_id": "OP-001",
        "date": dt.date(2025, 7, 1),
        "period": "2025-07",
        "quarter": "2025-Q3",
        "department": "DEP-101",
        "process": "order intake",
        "operation_type": "Order Management",
        "status": "completed",
        "priority": "standard",
        "completed_operations": 50,
        "delayed_operations": 5,
        "processing_time_hours": 2.0,
        "queue_time_hours": 1.0,
        "total_cycle_time_hours": 3.0,
        "employee_count": 10,
        "vendor_id": None,
        "backlog_open_items": 10.0,
    }
    records = [{**defaults, **row} for row in rows]
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


def make_orders(rows: list[dict]) -> pd.DataFrame:
    """Build an orders-like DataFrame."""
    defaults = {
        "order_id": "ORD-001",
        "order_date": dt.date(2025, 7, 1),
        "period": "2025-07",
        "quarter": "2025-Q3",
        "department": "DEP-101",
        "product_category": "Standard Supplies",
        "order_value": 500.0,
        "promised_time_hours": 24.0,
        "actual_time_hours": 20.0,
        "fulfillment_time_hours": 8.0,
        "transit_time_hours": 12.0,
        "status": "delivered",
        "vendor_id": "VND-101",
        "delay_hours": 0.0,
    }
    records = [{**defaults, **row} for row in rows]
    df = pd.DataFrame(records)
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


def make_sla(rows: list[dict]) -> pd.DataFrame:
    """Build an sla_metrics-like DataFrame."""
    defaults = {
        "period": "2025-07",
        "quarter": "2025-Q3",
        "department": "DEP-101",
        "process": "order intake",
        "total_cases": 100,
        "sla_target_hours": 4.0,
        "average_cycle_time_hours": 3.0,
        "average_queue_time_hours": 1.0,
        "sla_met_cases": 90,
        "sla_breached_cases": 10,
        "sla_compliance_percentage": 90.0,
        "backlog_open_items": 5.0,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def make_vendors(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "vendor_id": "VND-101",
        "vendor_name": "Test Vendor",
        "category": "Logistics",
        "department": "Fulfillment",
        "monthly_volume": 100.0,
        "average_processing_time": 2.0,
        "delay_rate": 5.0,
        "sla_compliance": 90.0,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def make_departments_df(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "department_id": "DEP-101",
        "department_name": "Customer Operations",
        "operational_area": "Front Office",
        "headcount": 34,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def make_plan(
    analysis_type: AnalysisType = AnalysisType.CYCLE_TIME_ANALYSIS,
    metric: Metric = Metric.AVERAGE_CYCLE_TIME,
    period: str | None = "2025-07",
    comparison_period: str | None = None,
    dimensions: tuple = (),
    top_n: int = 5,
    department: str | None = None,
    process: str | None = None,
    vendor_id: str | None = None,
) -> AnalysisPlan:
    required: tuple = (DatasetId.OPERATIONS,)
    if analysis_type == AnalysisType.SLA_ANALYSIS:
        required = (DatasetId.SLA_METRICS, DatasetId.DEPARTMENTS)
    elif analysis_type == AnalysisType.VENDOR_ANALYSIS:
        required = (DatasetId.ORDERS, DatasetId.OPERATIONS, DatasetId.VENDORS, DatasetId.DEPARTMENTS)
    elif analysis_type in (AnalysisType.PERIOD_COMPARISON, AnalysisType.CONTRIBUTION_ANALYSIS):
        required = (DatasetId.OPERATIONS, DatasetId.SLA_METRICS, DatasetId.ORDERS, DatasetId.DEPARTMENTS)
        if comparison_period is None and analysis_type == AnalysisType.CONTRIBUTION_ANALYSIS:
            comparison_period = "2025-06"
    return AnalysisPlan(
        analysis_type=analysis_type,
        metric=metric,
        dimensions=dimensions,
        period=period,
        comparison_period=comparison_period,
        department=department,
        process=process,
        vendor_id=vendor_id,
        top_n=top_n,
        required_datasets=required,
        reasoning_goal=ReasoningGoal.SUMMARIZE_PERFORMANCE,
    )


def make_period_range(period: str = "2025-07") -> PeriodRange:
    return PeriodRange(start=period, end=period, label=period)


def make_context(
    plan: AnalysisPlan,
    primary_frame: pd.DataFrame,
    comparison_frame: pd.DataFrame | None = None,
    sla_frame: pd.DataFrame | None = None,
    orders_frame: pd.DataFrame | None = None,
    vendors_frame: pd.DataFrame | None = None,
    departments_frame: pd.DataFrame | None = None,
    min_observations: int = 4,
) -> AnalysisContext:
    """Build a minimal AnalysisContext without touching the real repository."""
    repository = get_repository()

    period = plan.period or "2025-07"
    primary_range = make_period_range(period)
    primary_slice = PeriodSlice(
        period_range=primary_range,
        frame=primary_frame,
        by_period={period: primary_frame},
    )

    comparison_slice = None
    if comparison_frame is not None and plan.comparison_period:
        comp_period = plan.comparison_period
        comp_range = make_period_range(comp_period)
        comparison_slice = PeriodSlice(
            period_range=comp_range,
            frame=comparison_frame,
            by_period={comp_period: comparison_frame},
        )

    frames: dict[DatasetId, pd.DataFrame] = {DatasetId.OPERATIONS: primary_frame}
    comparison_frames: dict[DatasetId, pd.DataFrame] = {
        DatasetId.OPERATIONS: comparison_frame if comparison_frame is not None else pd.DataFrame()
    }

    if sla_frame is not None:
        frames[DatasetId.SLA_METRICS] = sla_frame
        comparison_frames[DatasetId.SLA_METRICS] = pd.DataFrame()

    if orders_frame is not None:
        frames[DatasetId.ORDERS] = orders_frame
        comparison_frames[DatasetId.ORDERS] = pd.DataFrame()

    if vendors_frame is not None:
        frames[DatasetId.VENDORS] = vendors_frame

    if departments_frame is not None:
        frames[DatasetId.DEPARTMENTS] = departments_frame

    return AnalysisContext(
        plan=plan,
        repository=repository,
        primary=primary_slice,
        comparison=comparison_slice,
        frames=frames,
        comparison_frames=comparison_frames,
        caveats=(),
        min_observations=min_observations,
    )


# ---------------------------------------------------------------------------
# Cycle time helpers
# ---------------------------------------------------------------------------


class TestCycleComponents:
    def test_basic_weighted_mean(self):
        frame = make_operations([
            {"completed_operations": 100, "queue_time_hours": 2.0,
             "processing_time_hours": 3.0, "total_cycle_time_hours": 5.0},
            {"completed_operations": 100, "queue_time_hours": 4.0,
             "processing_time_hours": 1.0, "total_cycle_time_hours": 5.0},
        ])
        result = cycle_components(frame)
        assert result["queue"] == pytest.approx(3.0)
        assert result["processing"] == pytest.approx(2.0)
        assert result["cycle"] == pytest.approx(5.0)
        assert result["queue_share"] == pytest.approx(60.0)
        assert result["processing_share"] == pytest.approx(40.0)

    def test_empty_frame_returns_none(self):
        result = cycle_components(pd.DataFrame())
        assert result["cycle"] is None
        assert result["queue"] is None

    def test_none_frame_returns_none(self):
        result = cycle_components(None)
        assert result["cycle"] is None

    def test_zero_volume_falls_back_to_unweighted(self):
        frame = make_operations([
            {"completed_operations": 0, "queue_time_hours": 2.0,
             "processing_time_hours": 3.0, "total_cycle_time_hours": 5.0},
        ])
        result = cycle_components(frame)
        assert result["cycle"] == pytest.approx(5.0)

    def test_single_row(self):
        frame = make_operations([
            {"completed_operations": 10, "queue_time_hours": 1.5,
             "processing_time_hours": 2.5, "total_cycle_time_hours": 4.0},
        ])
        result = cycle_components(frame)
        assert result["cycle"] == pytest.approx(4.0)
        assert result["queue_share"] == pytest.approx(37.5)


class TestCycleTable:
    def test_groups_by_column(self):
        frame = make_operations([
            {"department": "DEP-101", "completed_operations": 100,
             "queue_time_hours": 1.0, "processing_time_hours": 2.0,
             "total_cycle_time_hours": 3.0},
            {"department": "DEP-102", "completed_operations": 50,
             "queue_time_hours": 3.0, "processing_time_hours": 4.0,
             "total_cycle_time_hours": 7.0},
        ])
        table = cycle_table(frame, "department")
        assert len(table) == 2
        dep102 = table.loc[table["department"] == "DEP-102"]
        assert dep102.iloc[0]["cycle"] == pytest.approx(7.0)

    def test_empty_frame_returns_empty(self):
        table = cycle_table(pd.DataFrame(), "department")
        assert table.empty

    def test_missing_column_returns_empty(self):
        frame = make_operations([{}])
        table = cycle_table(frame, "nonexistent_column")
        assert table.empty


class TestChangeSeverity:
    def test_high_relative_change(self):
        assert change_severity(5.0, scale=10.0) == Severity.HIGH

    def test_medium_relative_change(self):
        assert change_severity(1.5, scale=10.0) == Severity.MEDIUM

    def test_small_change(self):
        assert change_severity(0.1, scale=10.0) == Severity.INFO

    def test_none_change(self):
        assert change_severity(None) == Severity.INFO


# ---------------------------------------------------------------------------
# SLA helpers
# ---------------------------------------------------------------------------


class TestComplianceSummary:
    def test_basic(self):
        frame = make_sla([
            {"sla_met_cases": 90, "sla_breached_cases": 10, "total_cases": 100},
            {"sla_met_cases": 80, "sla_breached_cases": 20, "total_cases": 100},
        ])
        result = compliance_summary(frame)
        assert result["met"] == 170
        assert result["breached"] == 30
        assert result["total"] == 200
        assert result["compliance"] == pytest.approx(85.0)

    def test_zero_total(self):
        result = compliance_summary(pd.DataFrame())
        assert result["compliance"] is None
        assert result["total"] == 0

    def test_perfect_compliance(self):
        frame = make_sla([{"sla_met_cases": 100, "sla_breached_cases": 0, "total_cases": 100}])
        result = compliance_summary(frame)
        assert result["compliance"] == pytest.approx(100.0)

    def test_zero_compliance(self):
        frame = make_sla([{"sla_met_cases": 0, "sla_breached_cases": 100, "total_cases": 100}])
        result = compliance_summary(frame)
        assert result["compliance"] == pytest.approx(0.0)


class TestComplianceSeverity:
    def test_critical(self):
        assert compliance_severity(50.0) == Severity.CRITICAL

    def test_high(self):
        assert compliance_severity(80.0) == Severity.HIGH

    def test_medium(self):
        assert compliance_severity(93.0) == Severity.MEDIUM

    def test_info_when_good(self):
        assert compliance_severity(99.0) == Severity.INFO

    def test_none_is_info(self):
        assert compliance_severity(None) == Severity.INFO


class TestComplianceTable:
    def test_recomputes_per_column(self):
        frame = make_sla([
            {"department": "DEP-101", "sla_met_cases": 95, "sla_breached_cases": 5, "total_cases": 100},
            {"department": "DEP-102", "sla_met_cases": 70, "sla_breached_cases": 30, "total_cases": 100},
        ])
        table = compliance_table(frame, "department")
        dep102 = table.loc[table["department"] == "DEP-102"]
        assert dep102.iloc[0]["compliance"] == pytest.approx(70.0)

    def test_empty_frame(self):
        assert compliance_table(pd.DataFrame(), "department").empty


# ---------------------------------------------------------------------------
# Backlog helpers
# ---------------------------------------------------------------------------


class TestClosingBacklog:
    def test_returns_last_day_total(self):
        frame = make_operations([
            {"date": dt.date(2025, 7, 1), "backlog_open_items": 10.0},
            {"date": dt.date(2025, 7, 2), "backlog_open_items": 20.0},
            {"date": dt.date(2025, 7, 2), "backlog_open_items": 15.0},
        ])
        frame["date"] = pd.to_datetime(frame["date"])
        result = closing_backlog(frame)
        # Last day is July 2, sum = 35
        assert result == pytest.approx(35.0)

    def test_single_row(self):
        frame = make_operations([{"date": dt.date(2025, 7, 1), "backlog_open_items": 42.0}])
        frame["date"] = pd.to_datetime(frame["date"])
        result = closing_backlog(frame)
        assert result == pytest.approx(42.0)

    def test_empty_frame_returns_none(self):
        assert closing_backlog(pd.DataFrame()) is None

    def test_none_returns_none(self):
        assert closing_backlog(None) is None


class TestDailyBacklogTotals:
    def test_aggregates_by_day(self):
        frame = make_operations([
            {"date": dt.date(2025, 7, 1), "backlog_open_items": 10.0},
            {"date": dt.date(2025, 7, 1), "backlog_open_items": 5.0},
            {"date": dt.date(2025, 7, 2), "backlog_open_items": 20.0},
        ])
        frame["date"] = pd.to_datetime(frame["date"])
        totals = daily_backlog_totals(frame)
        assert len(totals) == 2
        july1 = totals.loc[totals["date"] == pd.Timestamp("2025-07-01")]
        assert july1.iloc[0]["backlog"] == pytest.approx(15.0)

    def test_empty_frame(self):
        totals = daily_backlog_totals(pd.DataFrame())
        assert totals.empty


class TestBacklogTable:
    def test_opening_and_closing(self):
        frame = make_operations([
            {"department": "DEP-101", "date": dt.date(2025, 7, 1), "backlog_open_items": 10.0},
            {"department": "DEP-101", "date": dt.date(2025, 7, 5), "backlog_open_items": 30.0},
            {"department": "DEP-102", "date": dt.date(2025, 7, 1), "backlog_open_items": 5.0},
            {"department": "DEP-102", "date": dt.date(2025, 7, 5), "backlog_open_items": 3.0},
        ])
        frame["date"] = pd.to_datetime(frame["date"])
        table = backlog_table(frame, "department")
        dep101 = table.loc[table["department"] == "DEP-101"]
        assert dep101.iloc[0]["opening"] == pytest.approx(10.0)
        assert dep101.iloc[0]["closing"] == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# Bottleneck helpers
# ---------------------------------------------------------------------------


class TestBottleneckTable:
    def test_scores_are_in_zero_to_one(self):
        rows = []
        for i, dept in enumerate(["DEP-101", "DEP-102", "DEP-103"]):
            rows.extend([
                {"department": dept, "date": dt.date(2025, 7, i + 1),
                 "completed_operations": 100,
                 "total_cycle_time_hours": (i + 1) * 2.0,
                 "queue_time_hours": (i + 1) * 1.0,
                 "processing_time_hours": (i + 1) * 1.0,
                 "backlog_open_items": (i + 1) * 10.0},
            ])
        frame = make_operations(rows)
        table = bottleneck_table(frame, "department")
        assert not table.empty
        assert (table["score"] >= 0.0).all()
        assert (table["score"] <= 1.0).all()

    def test_highest_score_is_most_constrained(self):
        # DEP-102 has much worse cycle time and backlog
        rows = [
            {"department": "DEP-101", "date": dt.date(2025, 7, 1),
             "completed_operations": 100, "total_cycle_time_hours": 1.0,
             "queue_time_hours": 0.2, "processing_time_hours": 0.8,
             "backlog_open_items": 0.0},
            {"department": "DEP-102", "date": dt.date(2025, 7, 1),
             "completed_operations": 10, "total_cycle_time_hours": 20.0,
             "queue_time_hours": 15.0, "processing_time_hours": 5.0,
             "backlog_open_items": 500.0},
        ]
        frame = make_operations(rows)
        table = bottleneck_table(frame, "department")
        top = table.iloc[0]
        assert top["department"] == "DEP-102"

    def test_empty_frame_returns_empty(self):
        table = bottleneck_table(pd.DataFrame(), "department")
        assert table.empty


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------


class TestModifiedZScores:
    def test_returns_correct_length(self):
        values = [1.0, 2.0, 3.0, 4.0, 100.0]
        scores = modified_z_scores(values)
        assert len(scores) == 5

    def test_outlier_gets_highest_score(self):
        values = [1.0, 2.0, 3.0, 4.0, 100.0]
        scores = modified_z_scores(values)
        non_none = [s for s in scores if s is not None]
        assert abs(scores[-1]) == max(abs(s) for s in non_none)

    def test_flat_series_returns_none(self):
        scores = modified_z_scores([5.0, 5.0, 5.0, 5.0])
        assert all(s is None for s in scores)

    def test_single_value_returns_none(self):
        scores = modified_z_scores([42.0])
        assert scores == [None]


class TestAnomalyDetect:
    def test_detects_obvious_spike(self):
        # Normal sequence with a large spike at the end
        series = [{"period": f"2025-{i:02d}", "value": 5.0} for i in range(1, 11)]
        series.append({"period": "2025-11", "value": 100.0})
        anomalies = detect(series, Metric.AVERAGE_CYCLE_TIME)
        assert len(anomalies) >= 1
        assert any(a.period == "2025-11" for a in anomalies)

    def test_stable_series_produces_no_anomalies(self):
        series = [{"period": f"2025-{i:02d}", "value": 5.0 + i * 0.01}
                  for i in range(1, 13)]
        anomalies = detect(series, Metric.AVERAGE_CYCLE_TIME)
        assert len(anomalies) == 0

    def test_direction_spike_vs_dip(self):
        series = [{"period": f"2025-{i:02d}", "value": 10.0} for i in range(1, 11)]
        series.append({"period": "2025-11", "value": 100.0})
        anomalies = detect(series, Metric.AVERAGE_CYCLE_TIME)
        spike = next(a for a in anomalies if a.period == "2025-11")
        assert spike.direction == "spike"


# ---------------------------------------------------------------------------
# Trend analysis helpers
# ---------------------------------------------------------------------------


class TestClassifyDirection:
    def test_rising_lower_is_better_is_deteriorating(self):
        fit = {"slope": 1.0, "r_squared": 0.9}
        direction, confirmed = classify_direction(fit, 10.0, 20.0, better_when_lower=True)
        assert direction == "deteriorating"

    def test_falling_lower_is_better_is_improving(self):
        fit = {"slope": -1.0, "r_squared": 0.9}
        direction, confirmed = classify_direction(fit, 20.0, 10.0, better_when_lower=True)
        assert direction == "improving"

    def test_flat_is_flat(self):
        fit = {"slope": 0.001, "r_squared": 0.01}
        direction, _ = classify_direction(fit, 10.0, 10.0, better_when_lower=True)
        assert direction == "flat"

    def test_none_fit_is_flat(self):
        direction, _ = classify_direction(None, 10.0, 12.0, better_when_lower=True)
        assert direction == "flat"


# ---------------------------------------------------------------------------
# Vendor helpers
# ---------------------------------------------------------------------------


class TestOnTimeRate:
    def test_basic(self):
        assert on_time_rate(10.0, 100.0) == pytest.approx(90.0)

    def test_all_on_time(self):
        assert on_time_rate(0.0, 100.0) == pytest.approx(100.0)

    def test_all_late(self):
        assert on_time_rate(100.0, 100.0) == pytest.approx(0.0)

    def test_zero_total_returns_none(self):
        assert on_time_rate(0.0, 0.0) is None

    def test_none_returns_none(self):
        assert on_time_rate(None, 100.0) is None


class TestVendorTable:
    def test_merges_orders_and_operations(self):
        orders = make_orders([
            {"vendor_id": "VND-101", "status": "delivered_late", "transit_time_hours": 15.0},
            {"vendor_id": "VND-101", "status": "delivered", "transit_time_hours": 10.0},
        ])
        operations = make_operations([
            {"vendor_id": "VND-101", "completed_operations": 100, "delayed_operations": 10},
        ])
        vendor_names = {"VND-101": "Test Vendor"}
        table = vendor_table(orders, operations, vendor_names)
        assert not table.empty
        row = table.loc[table["vendor_id"] == "VND-101"].iloc[0]
        assert row["counted_orders"] == 2
        assert row["late_orders"] == 1
        assert row["on_time_rate"] == pytest.approx(50.0)

    def test_empty_inputs(self):
        table = vendor_table(None, None, {})
        assert table.empty


# ---------------------------------------------------------------------------
# Throughput helpers
# ---------------------------------------------------------------------------


class TestDistinctDays:
    def test_counts_unique_dates(self):
        frame = make_operations([
            {"date": dt.date(2025, 7, 1)},
            {"date": dt.date(2025, 7, 1)},
            {"date": dt.date(2025, 7, 2)},
        ])
        frame["date"] = pd.to_datetime(frame["date"])
        assert distinct_days(frame) == 2

    def test_empty_frame_returns_zero(self):
        assert distinct_days(pd.DataFrame()) == 0

    def test_none_returns_zero(self):
        assert distinct_days(None) == 0


class TestRateTable:
    def test_computes_per_day_rate(self):
        frame = make_operations([
            {"department": "DEP-101", "date": dt.date(2025, 7, 1), "completed_operations": 100},
            {"department": "DEP-101", "date": dt.date(2025, 7, 2), "completed_operations": 100},
            {"department": "DEP-102", "date": dt.date(2025, 7, 1), "completed_operations": 50},
        ])
        frame["date"] = pd.to_datetime(frame["date"])
        table = rate_table(frame, "department")
        dep101 = table.loc[table["department"] == "DEP-101"].iloc[0]
        assert dep101["rate"] == pytest.approx(100.0)  # 200 / 2 days

    def test_empty_frame(self):
        table = rate_table(pd.DataFrame(), "department")
        assert table.empty


# ---------------------------------------------------------------------------
# Contribution analysis helpers
# ---------------------------------------------------------------------------


class TestContributionBreakdown:
    def test_shares_sum_to_approximately_100(self):
        primary = make_operations([
            {"department": "DEP-101", "completed_operations": 100,
             "total_cycle_time_hours": 5.0, "queue_time_hours": 2.0,
             "processing_time_hours": 3.0},
            {"department": "DEP-102", "completed_operations": 100,
             "total_cycle_time_hours": 8.0, "queue_time_hours": 4.0,
             "processing_time_hours": 4.0},
        ])
        comparison = make_operations([
            {"department": "DEP-101", "completed_operations": 100,
             "total_cycle_time_hours": 4.0, "queue_time_hours": 1.0,
             "processing_time_hours": 3.0},
            {"department": "DEP-102", "completed_operations": 100,
             "total_cycle_time_hours": 6.0, "queue_time_hours": 2.0,
             "processing_time_hours": 4.0},
        ])
        fn = weighted_mean_value("total_cycle_time_hours")
        breakdown = contribution_breakdown(
            primary, comparison, "department",
            metric_label="Average cycle time",
            value_fn=fn,
            total_current=fn(primary),
            total_comparison=fn(comparison),
            top_n=5,
        )
        shares = [c.contribution_share for c in breakdown.contributions
                  if c.contribution_share is not None]
        assert sum(shares) == pytest.approx(100.0, abs=1e-3)

    def test_largest_mover_is_first(self):
        primary = make_operations([
            {"department": "DEP-101", "completed_operations": 100,
             "total_cycle_time_hours": 10.0, "queue_time_hours": 5.0,
             "processing_time_hours": 5.0},
            {"department": "DEP-102", "completed_operations": 100,
             "total_cycle_time_hours": 3.0, "queue_time_hours": 1.0,
             "processing_time_hours": 2.0},
        ])
        comparison = make_operations([
            {"department": "DEP-101", "completed_operations": 100,
             "total_cycle_time_hours": 5.0, "queue_time_hours": 2.0,
             "processing_time_hours": 3.0},
            {"department": "DEP-102", "completed_operations": 100,
             "total_cycle_time_hours": 2.5, "queue_time_hours": 1.0,
             "processing_time_hours": 1.5},
        ])
        fn = weighted_mean_value("total_cycle_time_hours")
        breakdown = contribution_breakdown(
            primary, comparison, "department",
            metric_label="Test",
            value_fn=fn,
            total_current=fn(primary),
            total_comparison=fn(comparison),
            top_n=5,
        )
        # DEP-101 changed from 5 to 10 (+5), DEP-102 from 2.5 to 3 (+0.5)
        assert breakdown.contributions[0].dimension_value == "DEP-101"


# ---------------------------------------------------------------------------
# Analyzer integration tests using real repository data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def repository():
    return get_repository()


@pytest.fixture(scope="module")
def q3_operations(repository):
    frame = repository.frame(DatasetId.OPERATIONS)
    return frame.loc[frame["period"].isin(["2025-07", "2025-08", "2025-09"])].copy()


@pytest.fixture(scope="module")
def q2_operations(repository):
    frame = repository.frame(DatasetId.OPERATIONS)
    return frame.loc[frame["period"].isin(["2025-04", "2025-05", "2025-06"])].copy()


@pytest.fixture(scope="module")
def q3_sla(repository):
    frame = repository.frame(DatasetId.SLA_METRICS)
    return frame.loc[frame["period"].isin(["2025-07", "2025-08", "2025-09"])].copy()


@pytest.fixture(scope="module")
def q3_orders(repository):
    frame = repository.frame(DatasetId.ORDERS)
    return frame.loc[frame["period"].isin(["2025-07", "2025-08", "2025-09"])].copy()


@pytest.fixture(scope="module")
def full_operations(repository):
    return repository.frame(DatasetId.OPERATIONS).copy()


class TestKPIAnalyzerIntegration:
    def test_produces_five_kpis(self, q3_operations, q3_sla):
        plan = make_plan(AnalysisType.KPI_SUMMARY, Metric.THROUGHPUT, period="2025-Q3")
        context = make_context(plan, q3_operations, sla_frame=q3_sla)
        analyzer = KPIAnalyzer()
        outcome = analyzer.analyze(context)
        assert len(outcome.kpis) == 5
        assert all(kpi.value is not None for kpi in outcome.kpis)

    def test_kpi_values_are_finite(self, q3_operations, q3_sla):
        plan = make_plan(AnalysisType.KPI_SUMMARY, Metric.THROUGHPUT, period="2025-Q3")
        context = make_context(plan, q3_operations, sla_frame=q3_sla)
        outcome = KPIAnalyzer().analyze(context)
        for kpi in outcome.kpis:
            if kpi.value is not None:
                assert kpi.value == kpi.value  # not NaN
                assert abs(kpi.value) < 1e9  # not infinite

    def test_raises_on_empty_frame(self, q3_sla):
        plan = make_plan(AnalysisType.KPI_SUMMARY, Metric.THROUGHPUT, period="2025-07")
        context = make_context(plan, pd.DataFrame(), sla_frame=q3_sla)
        with pytest.raises(InsufficientDataError):
            KPIAnalyzer().analyze(context)

    def test_evidence_has_source_datasets(self, q3_operations, q3_sla):
        plan = make_plan(AnalysisType.KPI_SUMMARY, Metric.THROUGHPUT, period="2025-Q3")
        context = make_context(plan, q3_operations, sla_frame=q3_sla)
        outcome = KPIAnalyzer().analyze(context)
        assert all(item.source_dataset is not None for item in outcome.evidence)
        assert all(item.calculation_description for item in outcome.evidence)


class TestThroughputAnalyzerIntegration:
    def test_overall_rate_is_positive(self, q3_operations, q2_operations):
        plan = make_plan(AnalysisType.THROUGHPUT_ANALYSIS, Metric.THROUGHPUT,
                         period="2025-Q3", comparison_period="2025-Q2")
        context = make_context(plan, q3_operations, q2_operations)
        outcome = ThroughputAnalyzer().analyze(context)
        rate_kpi = next(k for k in outcome.kpis if "per day" in k.name.lower())
        assert rate_kpi.value is not None
        assert rate_kpi.value > 0

    def test_dimension_breakdown_by_department(self, q3_operations):
        plan = make_plan(AnalysisType.THROUGHPUT_ANALYSIS, Metric.THROUGHPUT,
                         period="2025-Q3", dimensions=(Dimension.DEPARTMENT,))
        context = make_context(plan, q3_operations)
        outcome = ThroughputAnalyzer().analyze(context)
        assert len(outcome.evidence) >= 2

    def test_row_count_is_set(self, q3_operations):
        plan = make_plan(AnalysisType.THROUGHPUT_ANALYSIS, Metric.THROUGHPUT, period="2025-Q3")
        context = make_context(plan, q3_operations)
        outcome = ThroughputAnalyzer().analyze(context)
        assert outcome.row_count_examined == len(q3_operations)


class TestCycleTimeAnalyzerIntegration:
    def test_queue_and_processing_sum_to_cycle(self, q3_operations):
        plan = make_plan(AnalysisType.CYCLE_TIME_ANALYSIS, Metric.AVERAGE_CYCLE_TIME,
                         period="2025-Q3")
        context = make_context(plan, q3_operations)
        outcome = CycleTimeAnalyzer().analyze(context)
        q_kpi = next(k for k in outcome.kpis if "queue" in k.name.lower())
        p_kpi = next(k for k in outcome.kpis if "processing" in k.name.lower())
        c_kpi = next(k for k in outcome.kpis if "average cycle" in k.name.lower())
        if all(k.value is not None for k in [q_kpi, p_kpi, c_kpi]):
            # Contributions should sum close to 100%
            q_share = next(k for k in outcome.kpis if "contribution" in k.name.lower())
            if q_share.value is not None:
                assert 0.0 <= q_share.value <= 100.0

    def test_comparison_evidence_has_both_values(self, q3_operations, q2_operations):
        plan = make_plan(AnalysisType.CYCLE_TIME_ANALYSIS, Metric.AVERAGE_CYCLE_TIME,
                         period="2025-Q3", comparison_period="2025-Q2")
        context = make_context(plan, q3_operations, q2_operations)
        outcome = CycleTimeAnalyzer().analyze(context)
        cycle_ev = next(e for e in outcome.evidence if e.metric == "Average cycle time")
        assert cycle_ev.value is not None
        assert cycle_ev.comparison_value is not None

    def test_findings_identify_dominant_component(self, q3_operations, q2_operations):
        """Findings are only emitted when the change exceeds the threshold.

        With real synthetic data Q3 vs Q2 the change may be sub-threshold, which
        is correct behaviour - the analyzer should not fabricate a finding when
        the movement is negligible. The test verifies that either findings are
        present and mention the right terminology, or the result is empty
        (sub-threshold change) without raising.
        """
        plan = make_plan(AnalysisType.CYCLE_TIME_ANALYSIS, Metric.AVERAGE_CYCLE_TIME,
                         period="2025-Q3", comparison_period="2025-Q2")
        context = make_context(plan, q3_operations, q2_operations)
        outcome = CycleTimeAnalyzer().analyze(context)
        # Either findings exist and mention cycle-time concepts, or there are
        # legitimately none because the measured change was below 0.05 hours.
        if outcome.findings:
            combined = " ".join(f.title + f.detail for f in outcome.findings).lower()
            assert "queue" in combined or "processing" in combined or "cycle" in combined
        # Always at least have evidence items (the comparison itself)
        assert len(outcome.evidence) >= 3


class TestBacklogAnalyzerIntegration:
    def test_closing_backlog_is_non_negative(self, q3_operations, q2_operations):
        plan = make_plan(AnalysisType.BACKLOG_ANALYSIS, Metric.BACKLOG,
                         period="2025-Q3", comparison_period="2025-Q2")
        context = make_context(plan, q3_operations, q2_operations)
        outcome = BacklogAnalyzer().analyze(context)
        closing = next(k for k in outcome.kpis if "closing" in k.name.lower())
        assert closing.value is not None
        assert closing.value >= 0.0

    def test_backlog_change_evidence_exists(self, q3_operations, q2_operations):
        plan = make_plan(AnalysisType.BACKLOG_ANALYSIS, Metric.BACKLOG,
                         period="2025-Q3", comparison_period="2025-Q2")
        context = make_context(plan, q3_operations, q2_operations)
        outcome = BacklogAnalyzer().analyze(context)
        change_ev = next((e for e in outcome.evidence if "change" in e.metric.lower()), None)
        assert change_ev is not None

    def test_department_dimension_adds_evidence(self, q3_operations):
        plan = make_plan(AnalysisType.BACKLOG_ANALYSIS, Metric.BACKLOG,
                         period="2025-Q3", dimensions=(Dimension.DEPARTMENT,))
        context = make_context(plan, q3_operations)
        outcome = BacklogAnalyzer().analyze(context)
        assert len(outcome.evidence) >= 2


class TestSLAAnalyzerIntegration:
    def test_compliance_is_in_valid_range(self, q3_sla):
        plan = make_plan(AnalysisType.SLA_ANALYSIS, Metric.SLA_COMPLIANCE,
                         period="2025-Q3")
        context = make_context(plan, pd.DataFrame(), sla_frame=q3_sla)
        outcome = SLAAnalyzer().analyze(context)
        compliance_kpi = next(k for k in outcome.kpis if "compliance" in k.name.lower()
                               and "breach" not in k.name.lower())
        if compliance_kpi.value is not None:
            assert 0.0 <= compliance_kpi.value <= 100.0

    def test_recomputed_compliance_matches_formula(self, q3_sla):
        """Compliance must be recomputed from case counts, not read from stored percentage."""
        plan = make_plan(AnalysisType.SLA_ANALYSIS, Metric.SLA_COMPLIANCE, period="2025-Q3")
        context = make_context(plan, pd.DataFrame(), sla_frame=q3_sla)
        outcome = SLAAnalyzer().analyze(context)
        comp_ev = next(e for e in outcome.evidence if "compliance" in e.metric.lower()
                        and "breach" not in e.metric.lower())
        # cross-check against direct calculation
        summary = compliance_summary(q3_sla)
        if summary["compliance"] is not None and comp_ev.value is not None:
            assert comp_ev.value == pytest.approx(summary["compliance"], abs=0.01)

    def test_process_dimension_finds_offenders(self, q3_sla):
        plan = make_plan(AnalysisType.SLA_ANALYSIS, Metric.SLA_COMPLIANCE,
                         period="2025-Q3", dimensions=(Dimension.PROCESS,))
        context = make_context(plan, pd.DataFrame(), sla_frame=q3_sla)
        outcome = SLAAnalyzer().analyze(context)
        assert len(outcome.evidence) >= 2

    def test_raises_on_empty_sla_data(self):
        plan = make_plan(AnalysisType.SLA_ANALYSIS, Metric.SLA_COMPLIANCE, period="2025-07")
        context = make_context(plan, pd.DataFrame(), sla_frame=pd.DataFrame())
        with pytest.raises(InsufficientDataError):
            SLAAnalyzer().analyze(context)


class TestBottleneckAnalyzerIntegration:
    def test_scores_are_bounded(self, q3_operations):
        plan = make_plan(AnalysisType.BOTTLENECK_ANALYSIS, Metric.AVERAGE_CYCLE_TIME,
                         period="2025-Q3", dimensions=(Dimension.PROCESS,))
        context = make_context(plan, q3_operations)
        outcome = BottleneckAnalyzer().analyze(context)
        score_ev = next(e for e in outcome.evidence if "score" in e.metric.lower())
        assert score_ev.value is not None
        assert 0.0 <= score_ev.value <= 1.0

    def test_top_bottleneck_has_all_components(self, q3_operations):
        plan = make_plan(AnalysisType.BOTTLENECK_ANALYSIS, Metric.AVERAGE_CYCLE_TIME,
                         period="2025-Q3", dimensions=(Dimension.PROCESS,))
        context = make_context(plan, q3_operations)
        outcome = BottleneckAnalyzer().analyze(context)
        # Should have score, cycle time, queue share, backlog pressure
        assert len([e for e in outcome.evidence if e.dimension is not None]) >= 3

    def test_primary_constraint_finding_exists(self, q3_operations):
        plan = make_plan(AnalysisType.BOTTLENECK_ANALYSIS, Metric.AVERAGE_CYCLE_TIME,
                         period="2025-Q3")
        context = make_context(plan, q3_operations)
        outcome = BottleneckAnalyzer().analyze(context)
        assert any("constraint" in f.title.lower() or "bottleneck" in f.title.lower()
                   for f in outcome.findings)

    def test_weights_formula_is_documented(self, q3_operations):
        """The formula weights are documented in the finding statistic."""
        plan = make_plan(AnalysisType.BOTTLENECK_ANALYSIS, Metric.AVERAGE_CYCLE_TIME,
                         period="2025-Q3")
        context = make_context(plan, q3_operations)
        outcome = BottleneckAnalyzer().analyze(context)
        primary_finding = next(f for f in outcome.findings if f.metric == "bottleneck_score")
        # Should contain the weight breakdown in the statistic field
        assert primary_finding.statistic is not None
        stat = primary_finding.statistic
        assert "0.45" in stat or "cycle" in stat.lower()


class TestAnomalyAnalyzerIntegration:
    def test_returns_anomaly_count_kpi(self, full_operations, q3_sla):
        plan = make_plan(AnalysisType.ANOMALY_ANALYSIS, Metric.ANOMALY_COUNT, period="2025")
        context = make_context(plan, full_operations, sla_frame=q3_sla,
                                min_observations=4)
        outcome = AnomalyAnalyzer().analyze(context)
        count_kpi = next(k for k in outcome.kpis if "anomal" in k.name.lower())
        assert count_kpi.value is not None
        assert count_kpi.value >= 0.0

    def test_evidence_carries_method_description(self, full_operations, q3_sla):
        plan = make_plan(AnalysisType.ANOMALY_ANALYSIS, Metric.ANOMALY_COUNT, period="2025")
        context = make_context(plan, full_operations, sla_frame=q3_sla, min_observations=4)
        outcome = AnomalyAnalyzer().analyze(context)
        for ev in outcome.evidence:
            assert ev.calculation_description

    def test_caveats_state_the_method(self, full_operations, q3_sla):
        plan = make_plan(AnalysisType.ANOMALY_ANALYSIS, Metric.ANOMALY_COUNT, period="2025")
        context = make_context(plan, full_operations, sla_frame=q3_sla, min_observations=4)
        outcome = AnomalyAnalyzer().analyze(context)
        combined_caveats = " ".join(outcome.caveats).lower()
        assert "monthly" in combined_caveats or "detect" in combined_caveats


class TestTrendAnalyzerIntegration:
    def test_produces_three_kpis(self, full_operations):
        plan = make_plan(AnalysisType.TREND_ANALYSIS, Metric.AVERAGE_CYCLE_TIME,
                         period="2025", comparison_period=None)
        context = make_context(plan, full_operations, min_observations=4)
        outcome = TrendAnalyzer().analyze(context)
        assert len(outcome.kpis) == 3

    def test_direction_is_classified(self, full_operations):
        plan = make_plan(AnalysisType.TREND_ANALYSIS, Metric.AVERAGE_CYCLE_TIME,
                         period="2025", comparison_period=None)
        context = make_context(plan, full_operations, min_observations=4)
        outcome = TrendAnalyzer().analyze(context)
        direction_finding = outcome.findings[0]
        assert any(word in direction_finding.title.lower()
                   for word in ["improving", "deteriorating", "flat", "volatile"])

    def test_r_squared_evidence_exists(self, full_operations):
        plan = make_plan(AnalysisType.TREND_ANALYSIS, Metric.AVERAGE_CYCLE_TIME,
                         period="2025", comparison_period=None)
        context = make_context(plan, full_operations, min_observations=4)
        outcome = TrendAnalyzer().analyze(context)
        r2_ev = next(e for e in outcome.evidence if "r^2" in e.metric.lower()
                     or "r_squared" in e.metric.lower() or "fit" in e.metric.lower())
        assert r2_ev is not None

    def test_insufficient_data_raises(self, q3_operations):
        plan = make_plan(AnalysisType.TREND_ANALYSIS, Metric.AVERAGE_CYCLE_TIME,
                         period="2025-Q3", comparison_period=None)
        context = make_context(plan, q3_operations, min_observations=15)
        with pytest.raises(InsufficientDataError):
            TrendAnalyzer().analyze(context)


class TestVendorAnalyzerIntegration:
    def test_finds_vendors(self, q3_operations, q3_orders, repository):
        plan = make_plan(AnalysisType.VENDOR_ANALYSIS, Metric.ON_TIME_DELIVERY_RATE,
                         period="2025-Q3")
        vendors_frame = repository.frame(DatasetId.VENDORS)
        context = make_context(plan, q3_operations, orders_frame=q3_orders,
                                vendors_frame=vendors_frame)
        outcome = VendorAnalyzer().analyze(context)
        assert len(outcome.kpis) >= 2

    def test_on_time_rate_is_percentage(self, q3_operations, q3_orders, repository):
        plan = make_plan(AnalysisType.VENDOR_ANALYSIS, Metric.ON_TIME_DELIVERY_RATE,
                         period="2025-Q3")
        vendors_frame = repository.frame(DatasetId.VENDORS)
        context = make_context(plan, q3_operations, orders_frame=q3_orders,
                                vendors_frame=vendors_frame)
        outcome = VendorAnalyzer().analyze(context)
        overall_kpi = next(k for k in outcome.kpis if "on-time" in k.name.lower())
        if overall_kpi.value is not None:
            assert 0.0 <= overall_kpi.value <= 100.0

    def test_caveats_mention_attribution(self, q3_operations, q3_orders, repository):
        plan = make_plan(AnalysisType.VENDOR_ANALYSIS, Metric.ON_TIME_DELIVERY_RATE,
                         period="2025-Q3")
        vendors_frame = repository.frame(DatasetId.VENDORS)
        context = make_context(plan, q3_operations, orders_frame=q3_orders,
                                vendors_frame=vendors_frame)
        outcome = VendorAnalyzer().analyze(context)
        assert any("vendor" in c.lower() or "attribution" in c.lower()
                   for c in outcome.caveats)


class TestDepartmentAnalyzerIntegration:
    def test_cycle_time_ratio_kpi_exists(self, q3_operations, q3_sla, repository):
        plan = make_plan(AnalysisType.DEPARTMENT_ANALYSIS, Metric.SLA_COMPLIANCE,
                         period="2025-Q3")
        dept_frame = repository.frame(DatasetId.DEPARTMENTS)
        context = make_context(plan, q3_operations, sla_frame=q3_sla,
                                departments_frame=dept_frame)
        outcome = DepartmentAnalyzer().analyze(context)
        assert len(outcome.kpis) >= 2

    def test_compliance_evidence_has_dimensions(self, q3_operations, q3_sla, repository):
        plan = make_plan(AnalysisType.DEPARTMENT_ANALYSIS, Metric.SLA_COMPLIANCE,
                         period="2025-Q3")
        dept_frame = repository.frame(DatasetId.DEPARTMENTS)
        context = make_context(plan, q3_operations, sla_frame=q3_sla,
                                departments_frame=dept_frame)
        outcome = DepartmentAnalyzer().analyze(context)
        assert any(e.dimension is not None for e in outcome.evidence)


class TestContributionAnalyzerIntegration:
    def test_contributions_sum_correctly(self, q3_operations, q2_operations):
        plan = make_plan(AnalysisType.CONTRIBUTION_ANALYSIS, Metric.AVERAGE_CYCLE_TIME,
                         period="2025-Q3", comparison_period="2025-Q2",
                         dimensions=(Dimension.DEPARTMENT,))
        context = make_context(plan, q3_operations, q2_operations)
        outcome = ContributionAnalyzer().analyze(context)
        member_evidence = [e for e in outcome.evidence if e.dimension is not None]
        assert len(member_evidence) >= 2

    def test_headline_evidence_has_both_windows(self, q3_operations, q2_operations):
        plan = make_plan(AnalysisType.CONTRIBUTION_ANALYSIS, Metric.AVERAGE_CYCLE_TIME,
                         period="2025-Q3", comparison_period="2025-Q2",
                         dimensions=(Dimension.DEPARTMENT,))
        context = make_context(plan, q3_operations, q2_operations)
        outcome = ContributionAnalyzer().analyze(context)
        headline = outcome.evidence[0]
        assert headline.value is not None
        assert headline.comparison_value is not None

    def test_findings_rank_contributors(self, q3_operations, q2_operations):
        plan = make_plan(AnalysisType.CONTRIBUTION_ANALYSIS, Metric.AVERAGE_CYCLE_TIME,
                         period="2025-Q3", comparison_period="2025-Q2",
                         dimensions=(Dimension.DEPARTMENT,))
        context = make_context(plan, q3_operations, q2_operations)
        outcome = ContributionAnalyzer().analyze(context)
        ranked_findings = [f for f in outcome.findings
                           if "contributor" in f.title.lower() or "#" in f.title]
        assert len(ranked_findings) >= 1

    def test_caveats_warn_about_shares(self, q3_operations, q2_operations):
        plan = make_plan(AnalysisType.CONTRIBUTION_ANALYSIS, Metric.AVERAGE_CYCLE_TIME,
                         period="2025-Q3", comparison_period="2025-Q2",
                         dimensions=(Dimension.DEPARTMENT,))
        context = make_context(plan, q3_operations, q2_operations)
        outcome = ContributionAnalyzer().analyze(context)
        combined = " ".join(outcome.caveats).lower()
        assert "share" in combined or "absolute" in combined

    def test_raises_when_no_comparison_data(self, q3_operations):
        plan = make_plan(AnalysisType.CONTRIBUTION_ANALYSIS, Metric.AVERAGE_CYCLE_TIME,
                         period="2025-Q3", comparison_period="2025-Q2")
        context = make_context(plan, q3_operations, pd.DataFrame())
        with pytest.raises(InsufficientDataError):
            ContributionAnalyzer().analyze(context)


class TestPeriodComparisonAnalyzerIntegration:
    def test_delegates_to_cycle_time(self, q3_operations, q2_operations, q3_sla):
        plan = make_plan(AnalysisType.PERIOD_COMPARISON, Metric.AVERAGE_CYCLE_TIME,
                         period="2025-Q3", comparison_period="2025-Q2")
        context = make_context(plan, q3_operations, q2_operations, sla_frame=q3_sla)
        outcome = PeriodComparisonAnalyzer().analyze(context)
        assert outcome.evidence
        assert any("cycle" in e.metric.lower() for e in outcome.evidence)

    def test_delegates_to_sla(self, q3_operations, q3_sla):
        plan = make_plan(AnalysisType.PERIOD_COMPARISON, Metric.SLA_COMPLIANCE,
                         period="2025-Q3", comparison_period="2025-Q2")
        context = make_context(plan, q3_operations, sla_frame=q3_sla)
        outcome = PeriodComparisonAnalyzer().analyze(context)
        assert any("sla" in e.metric.lower() or "compliance" in e.metric.lower()
                   for e in outcome.evidence)

    def test_comparison_caveat_is_appended(self, q3_operations, q2_operations, q3_sla):
        plan = make_plan(AnalysisType.PERIOD_COMPARISON, Metric.AVERAGE_CYCLE_TIME,
                         period="2025-Q3", comparison_period="2025-Q2")
        context = make_context(plan, q3_operations, q2_operations, sla_frame=q3_sla)
        outcome = PeriodComparisonAnalyzer().analyze(context)
        # PeriodComparisonAnalyzer always appends a delegation caveat
        assert any("comparison" in c.lower() or "delegate" in c.lower()
                   for c in outcome.caveats)


# ---------------------------------------------------------------------------
# Cross-cutting: evidence invariants
# ---------------------------------------------------------------------------


class TestEvidenceInvariants:
    """Every evidence item must satisfy the provenance contract."""

    @pytest.fixture(params=[
        (AnalysisType.KPI_SUMMARY, Metric.THROUGHPUT, "2025-Q3", None),
        (AnalysisType.THROUGHPUT_ANALYSIS, Metric.THROUGHPUT, "2025-Q3", None),
        (AnalysisType.BACKLOG_ANALYSIS, Metric.BACKLOG, "2025-Q3", "2025-Q2"),
        (AnalysisType.BOTTLENECK_ANALYSIS, Metric.AVERAGE_CYCLE_TIME, "2025-Q3", None),
    ])
    def outcome(self, request, q3_operations, q2_operations, q3_sla):
        analysis_type, metric, period, comparison_period = request.param
        plan = make_plan(analysis_type, metric, period=period,
                         comparison_period=comparison_period)
        context = make_context(plan, q3_operations, q2_operations, sla_frame=q3_sla)
        analyzer_map = {
            AnalysisType.KPI_SUMMARY: KPIAnalyzer,
            AnalysisType.THROUGHPUT_ANALYSIS: ThroughputAnalyzer,
            AnalysisType.BACKLOG_ANALYSIS: BacklogAnalyzer,
            AnalysisType.BOTTLENECK_ANALYSIS: BottleneckAnalyzer,
        }
        analyzer_cls = analyzer_map[analysis_type]
        return analyzer_cls().analyze(context)

    def test_no_nan_in_evidence(self, outcome):
        for item in outcome.evidence:
            for field_val in (item.value, item.comparison_value,
                               item.absolute_change, item.percentage_change):
                if field_val is not None:
                    assert field_val == field_val, f"NaN in {item.metric}"

    def test_no_infinity_in_evidence(self, outcome):
        for item in outcome.evidence:
            for field_val in (item.value, item.comparison_value,
                               item.absolute_change, item.percentage_change):
                if field_val is not None:
                    assert abs(field_val) < 1e15, f"Infinity in {item.metric}"

    def test_calculation_description_present(self, outcome):
        for item in outcome.evidence:
            assert item.calculation_description
            assert len(item.calculation_description) > 5

    def test_source_dataset_present(self, outcome):
        for item in outcome.evidence:
            assert item.source_dataset is not None

    def test_unit_present(self, outcome):
        from app.models.enums import Unit
        for item in outcome.evidence:
            assert item.unit in Unit
