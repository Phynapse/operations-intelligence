"""Trend analysis: direction, steepness and statistical strength of a metric.

The method is deliberately conservative. A point is only called an anomaly or a
trend when the evidence supports it:

* every metric is reduced to a monthly series, so a single noisy day cannot
  create a trend;
* a trend requires at least ``MIN_OBSERVATIONS_FOR_TREND`` months and a
  non-degenerate least-squares fit;
* direction is derived from the slope, not from comparing only the first and last
  points, and is confirmed against the first/last change so that a mid-period
  spike followed by a return is not reported as growth.
"""

from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd

from app.analytics.base import (
    AnalysisContext,
    AnalysisOutcome,
    BaseAnalyzer,
    build_evidence,
    build_finding,
    build_kpi,
)
from app.analytics.sla_analyzer import compliance_summary
from app.models.enums import AnalysisType, DatasetId, Metric, Severity, Unit
from app.schemas.analysis import Finding
from app.utils.calculations import (
    linear_trend,
    percentage_change,
    round_or_none,
    safe_float,
)
from app.utils.calculations import (
    throughput as rate_per_day,
)
from app.utils.errors import InsufficientDataError
from app.utils.periods import period_sort_key


class MetricSpec(TypedDict):
    """How one metric is read off the monthly series.

    `column` is `None` when the series comes from the SLA rollup rather than
    from a column of the operations dataset.
    """

    column: str | None
    aggregation: str
    unit: Unit
    reading: str


#: Metric -> (operations.csv column or None, aggregation, unit, reading)
METRIC_SPECS: dict[Metric, MetricSpec] = {
    Metric.THROUGHPUT: {
        "column": "completed_operations",
        "aggregation": "per_day",
        "unit": Unit.OPERATIONS,
        "reading": "higher is better",
    },
    Metric.AVERAGE_CYCLE_TIME: {
        "column": "total_cycle_time_hours",
        "aggregation": "weighted_mean",
        "unit": Unit.HOURS,
        "reading": "lower is better",
    },
    Metric.PROCESSING_TIME: {
        "column": "processing_time_hours",
        "aggregation": "weighted_mean",
        "unit": Unit.HOURS,
        "reading": "lower is better",
    },
    Metric.QUEUE_TIME: {
        "column": "queue_time_hours",
        "aggregation": "weighted_mean",
        "unit": Unit.HOURS,
        "reading": "lower is better",
    },
    Metric.BACKLOG: {
        "column": "backlog_open_items",
        "aggregation": "closing_sum",
        "unit": Unit.ITEMS,
        "reading": "lower is better",
    },
    Metric.DELAY_RATE: {
        "column": "delayed_operations",
        "aggregation": "delay_rate",
        "unit": Unit.PERCENT,
        "reading": "lower is better",
    },
    Metric.SLA_COMPLIANCE: {
        "column": None,
        "aggregation": "sla_compliance",
        "unit": Unit.PERCENT,
        "reading": "higher is better",
    },
}

DIMENSION_COLUMNS: dict[str, str] = {
    "department": "department",
    "process": "process",
    "operation_type": "operation_type",
    "vendor": "vendor_id",
}


class TrendAnalyzer(BaseAnalyzer):
    """Monthly trend of one metric with direction and fit statistics."""

    name = "trend_analyzer"
    supported_types = frozenset({AnalysisType.TREND_ANALYSIS})
    required_datasets = (DatasetId.OPERATIONS, DatasetId.SLA_METRICS, DatasetId.DEPARTMENTS)

    def analyze(self, context: AnalysisContext) -> AnalysisOutcome:
        metric = context.plan.metric
        spec = METRIC_SPECS.get(metric)
        if spec is None:
            raise InsufficientDataError(
                f"metric '{metric.value}' has no monthly trend definition; supported metrics: "
                + ", ".join(sorted(item.value for item in METRIC_SPECS))
            )

        frame = context.primary_frame()
        if frame.empty:
            raise InsufficientDataError(
                f"no records found for {context.plan.period or 'the requested period'}"
            )

        series = monthly_series(frame, metric, context)
        outcome = AnalysisOutcome()
        points = [(row["period"], row["value"]) for row in series if row["value"] is not None]

        if len(points) < context.min_observations:
            raise InsufficientDataError(
                f"trend analysis needs at least {context.min_observations} "
                f"periods of data but only {len(points)} were available"
            )

        values = [value for _, value in points]
        fit = linear_trend(values)
        first_period, first_value = points[0]
        last_period, last_value = points[-1]
        overall_change = percentage_change(last_value, first_value)
        direction, confirmed = classify_direction(
            fit, first_value, last_value, better_when_lower=str(spec["reading"]).startswith("lower")
        )

        r_squared = fit.get("r_squared") if fit else None
        slope = safe_float(fit.get("slope")) if fit else None

        outcome.kpis = [
            build_kpi(
                name=f"{_label(metric)} - latest period",
                value=last_value,
                unit=spec["unit"],
                source_dataset=_source_for(metric),
                period=last_period,
                calculation=(
                    f"monthly {spec['aggregation']} of {spec['column'] or 'SLA case counts'}"
                ),
            ),
            build_kpi(
                name=f"{_label(metric)} - change over period",
                value=overall_change,
                unit=Unit.PERCENT,
                source_dataset=_source_for(metric),
                period=f"{first_period} to {last_period}",
                calculation=(
                    f"(value in {last_period} - value in {first_period}) / value in "
                    f"{first_period} * 100"
                ),
            ),
            build_kpi(
                name=f"{_label(metric)} - monthly slope",
                value=slope,
                unit=spec["unit"],
                source_dataset=_source_for(metric),
                period=f"{first_period} to {last_period}",
                calculation=(
                    "least-squares slope of the monthly series, expressed in the metric unit per "
                    "month"
                ),
            ),
        ]

        outcome.evidence = [
            build_evidence(
                metric=_label(metric),
                value=last_value,
                comparison_value=first_value,
                unit=spec["unit"],
                source_dataset=_source_for(metric),
                period=last_period,
                comparison_period=first_period,
                calculation_description=(
                    f"Monthly {spec['aggregation']} of "
                    f"{spec['column'] or 'sla_met_cases / total_cases * 100'}; the change is "
                    "reported between the first and last month present in the data"
                ),
                sample_size=len(values),
                severity=direction_severity(direction, overall_change, spec["reading"]),
            ),
            build_evidence(
                metric=f"{_label(metric)} - monthly series",
                value=round_or_none(mean_of(values)),
                unit=spec["unit"],
                source_dataset=_source_for(metric),
                calculation_description=(
                    "Arithmetic mean of the monthly values used to fit the trend"
                ),
                sample_size=len(values),
            ),
            build_evidence(
                metric=f"{_label(metric)} - trend fit R^2",
                value=round_or_none(r_squared, 4),
                unit=Unit.PERCENT,
                source_dataset=_source_for(metric),
                calculation_description=(
                    "Coefficient of determination of the least-squares fit of the monthly series"
                ),
                sample_size=len(values),
                severity=_fit_severity(r_squared),
            ),
        ]

        outcome.findings.append(
            build_finding(
                title=f"{_label(metric)} is {_direction_phrase(direction)}",
                detail=_trend_detail(
                    metric,
                    direction,
                    confirmed,
                    first_period,
                    last_period,
                    first_value,
                    last_value,
                    overall_change,
                    slope,
                    r_squared,
                    spec,
                ),
                severity=direction_severity(direction, overall_change, str(spec["reading"])),
                metric=metric.value,
                period=f"{first_period} to {last_period}",
                is_statistical=True,
                statistic=(
                    f"least-squares slope = {slope:+.4f} per month"
                    + (f", R^2 = {r_squared:.3f}" if isinstance(r_squared, float) else "")
                ),
            )
        )
        if not confirmed:
            outcome.findings.append(
                build_finding(
                    title="Trend endpoint disagrees with fitted slope",
                    detail=(
                        f"The fitted slope is {slope:+.4f} per month but the first-to-last change "
                        f"is {overall_change:+.1f}%. The series is not monotonic, so the "
                        "direction should be read as provisional."
                    ),
                    severity=Severity.LOW,
                    metric=metric.value,
                    is_statistical=True,
                    statistic=f"slope {slope:+.4f} vs endpoint change {overall_change:+.2f}%",
                )
            )

        outcome.findings.extend(self._dimension_findings(context, metric, spec, frame))
        outcome.extend_caveats(
            f"trend is fitted over {len(values)} monthly observations from "
            f"{_source_for(metric).value}"
            + (
                ""
                if isinstance(r_squared, float)
                else "; the fit has no explanatory power because the series is flat"
            )
        )
        outcome.row_count_examined = len(frame)
        return outcome

    def _dimension_findings(
        self,
        context: AnalysisContext,
        metric: Metric,
        spec: MetricSpec,
        frame: pd.DataFrame,
    ) -> list[Finding]:
        column = None
        for dimension in context.plan.dimensions:
            column = DIMENSION_COLUMNS.get(dimension.value)
            if column:
                break
        if not column:
            return []
        names = context.repository.department_names
        rows = frame.groupby(column, dropna=True)
        findings: list[Finding] = []
        scores: list[tuple[float, str, float, float | None]] = []
        for key, group in rows:
            sub_series = monthly_series(group, metric, context)
            values = [row["value"] for row in sub_series if row["value"] is not None]
            if len(values) < context.min_observations:
                continue
            change = percentage_change(values[-1], values[0])
            if change is None:
                continue
            lower_is_better = str(spec["reading"]).startswith("lower")
            worse = change > 0 if lower_is_better else change < 0
            scores.append((abs(change), str(key), change, values[-1] if worse else None))
        if not scores:
            return []
        scores.sort(key=lambda item: item[0], reverse=True)
        lower_is_better = str(spec["reading"]).startswith("lower")
        for magnitude, key, change, _ in scores[: context.top_n]:
            worse = (change > 0) if lower_is_better else (change < 0)
            label = names.get(key, key)
            findings.append(
                build_finding(
                    title=f"{label}: {_label(metric)} {change:+.1f}% over the period",
                    detail=(
                        f"{label} moved {_label(metric).lower()} by {change:+.1f}% between the "
                        f"first and last month present in the data"
                        + (
                            f", which is the largest adverse move of any {column} value"
                            if worse
                            else ""
                        )
                        + "."
                    ),
                    severity=Severity.MEDIUM if magnitude >= 20 else Severity.LOW,
                    metric=metric.value,
                    dimension=label,
                    is_statistical=True,
                    statistic=f"first-to-last change = {change:+.2f}%",
                )
            )
        return findings


# ---------------------------------------------------------------------------
# Series construction
# ---------------------------------------------------------------------------


def monthly_series(
    frame: pd.DataFrame, metric: Metric, context: AnalysisContext
) -> list[dict[str, Any]]:
    """Reduce a frame to one value per month for the requested metric."""
    if frame.empty:
        return []
    rows = _aggregate_by_period(frame, metric, context)
    return rows


def _aggregate_by_period(
    frame: pd.DataFrame, metric: Metric, context: AnalysisContext
) -> list[dict]:
    spec = METRIC_SPECS[metric]
    aggregation = spec["aggregation"]
    column = spec["column"]
    if aggregation == "sla_compliance":
        sla_frame = context.frame(DatasetId.SLA_METRICS)
        if sla_frame.empty:
            return []
        sla_frame = _align_sla_to_operations_months(sla_frame, frame)
        compliance_rows: list[dict[str, Any]] = []
        for period, group in sla_frame.groupby("period", dropna=True):
            summary = compliance_summary(group)
            compliance_rows.append({"period": str(period), "value": summary["compliance"]})
        return sorted(compliance_rows, key=lambda row: period_sort_key(row["period"]))

    if aggregation == "per_day":
        grouped = frame.groupby("period", dropna=True).agg(
            completed=("completed_operations", "sum"), days=("date", "nunique")
        )
        return sorted(
            (
                {"period": str(period), "value": rate_per_day(row.completed, row.days)}
                for period, row in grouped.iterrows()
            ),
            key=lambda row: period_sort_key(row["period"]),
        )

    if aggregation == "closing_sum":
        # Backlog is a stock: the value per period is the total on the last day
        # present *in that period*, not the last day of the whole window.
        stock_rows: list[dict[str, Any]] = []
        for period, group in frame.groupby("period", dropna=True):
            last_date = group["date"].max()
            closing = float(group.loc[group["date"] == last_date, column].sum())
            stock_rows.append({"period": str(period), "value": round_or_none(closing, 1)})
        return sorted(stock_rows, key=lambda row: period_sort_key(row["period"]))

    if aggregation == "delay_rate":
        grouped = frame.groupby("period", dropna=True).agg(
            delayed=("delayed_operations", "sum"), completed=("completed_operations", "sum")
        )
        return sorted(
            (
                {
                    "period": str(period),
                    "value": round_or_none(row.delayed / row.completed * 100.0)
                    if row.completed
                    else None,
                }
                for period, row in grouped.iterrows()
            ),
            key=lambda row: period_sort_key(row["period"]),
        )

    grouped_rows: list[dict] = []
    for period, group in frame.groupby("period", dropna=True):
        weights = group["completed_operations"].fillna(0).astype("float64")
        total_weight = float(weights.sum())
        values = group[column].fillna(0).astype("float64")
        value = (
            float((values * weights).sum() / total_weight)
            if total_weight > 0
            else float(values.mean())
        )
        grouped_rows.append({"period": str(period), "value": round_or_none(value)})
    return sorted(grouped_rows, key=lambda row: period_sort_key(row["period"]))


def _align_sla_to_operations_months(
    sla_frame: pd.DataFrame, operations: pd.DataFrame
) -> pd.DataFrame:
    """Keep only SLA rows for months the analysis window actually covers."""
    months = set(operations["period"].astype(str))
    return sla_frame.loc[sla_frame["period"].astype(str).isin(months)]


# ---------------------------------------------------------------------------
# Interpretation helpers
# ---------------------------------------------------------------------------


def classify_direction(
    fit: dict[str, float | None] | None,
    first_value: float | None,
    last_value: float | None,
    better_when_lower: bool = False,
) -> tuple[str, bool]:
    """Classify a trend as ``improving``, ``deteriorating``, ``flat`` or ``volatile``.

    Returns ``(direction, endpoints_confirm)``. The slope decides the direction;
    the confirmation flag records whether the first-to-last change agrees with it.
    """
    slope = safe_float(fit.get("slope")) if fit else None
    change = percentage_change(last_value, first_value)
    if slope is None or change is None:
        return "flat", False

    reference = max(abs(first_value or 0.0), 1e-9)
    relative_slope = slope / reference
    if abs(relative_slope) < 0.005 or abs(change) < 2.0:
        return "flat", abs(change) < 2.0

    rising = relative_slope > 0
    direction = "deteriorating" if rising == better_when_lower else "improving"
    confirmed = (change > 0) == rising
    return direction, confirmed


def direction_severity(direction: str, change_percent: float | None, reading: str) -> Severity:
    if direction == "flat":
        return Severity.INFO
    magnitude = abs(change_percent or 0.0)
    if magnitude >= 30.0:
        return Severity.HIGH
    if magnitude >= 12.0:
        return Severity.MEDIUM
    return Severity.LOW


def _fit_severity(r_squared: float | None) -> Severity:
    value = safe_float(r_squared)
    if value is None:
        return Severity.INFO
    if value < 0.2:
        return Severity.LOW
    if value < 0.5:
        return Severity.MEDIUM
    return Severity.INFO


def _direction_phrase(direction: str) -> str:
    return {
        "improving": "improving",
        "deteriorating": "deteriorating",
        "flat": "broadly flat",
        "volatile": "volatile",
    }.get(direction, direction)


def _trend_detail(
    metric: Metric,
    direction: str,
    confirmed: bool,
    first_period: str,
    last_period: str,
    first_value: float,
    last_value: float,
    change: float | None,
    slope: float | None,
    r_squared: float | None,
    spec: MetricSpec,
) -> str:
    unit = spec["unit"]
    unit_text = (
        "hours"
        if unit is Unit.HOURS
        else "operations"
        if unit is Unit.OPERATIONS
        else "items"
        if unit is Unit.ITEMS
        else "percentage points"
    )
    fit_text = (
        f" The least-squares fit explains {r_squared * 100:.0f}% of the monthly variation."
        if isinstance(r_squared, float)
        else ""
    )
    if direction == "flat":
        return (
            f"{_label(metric)} moved from {first_value:,.2f} in {first_period} to "
            f"{last_value:,.2f} in {last_period}, a change of {change:+.1f}%, which is within "
            f"normal operating variation.{fit_text}"
        )
    caveat = (
        ""
        if confirmed
        else (" The monthly series is not monotonic, so this direction is provisional.")
    )
    return (
        f"{_label(metric)} moved from {first_value:,.2f} in {first_period} to {last_value:,.2f} "
        f"in {last_period}, a change of {change:+.1f}% "
        f"({slope:+.3f} {unit_text} per month on a least-squares fit).{fit_text}{caveat}"
    )


def _label(metric: Metric) -> str:
    return metric.value.replace("_", " ").capitalize()


def _source_for(metric: Metric) -> DatasetId:
    return DatasetId.SLA_METRICS if metric is Metric.SLA_COMPLIANCE else DatasetId.OPERATIONS


def mean_of(values: list[float]) -> float | None:
    return round_or_none(sum(values) / len(values)) if values else None


__all__ = ["METRIC_SPECS", "TrendAnalyzer", "classify_direction", "monthly_series"]
