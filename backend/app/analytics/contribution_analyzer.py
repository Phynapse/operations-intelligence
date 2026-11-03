"""Contribution analysis: decompose a metric change by dimension.

A "why did X change?" question is not answered by the headline delta alone. This
module attributes the overall change to the dimension members that drove it, so
the evidence package can carry a ranked contribution hierarchy (which department,
which process, which vendor moved the metric most).

The decomposition is deterministic and additive where the metric is additive
(e.g. hours contributed), and difference-based where it is an average or a rate.
Each contribution carries the current value, the comparison value, the absolute
change and the share of the total movement, so the LLM can state *what*
contributed without inventing *why*.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.analytics.base import (
    AnalysisContext,
    AnalysisOutcome,
    BaseAnalyzer,
    build_evidence,
    build_finding,
    build_kpi,
)
from app.models.enums import AnalysisType, DatasetId, Metric, Severity, Unit
from app.schemas.analysis import EvidenceItem, Finding
from app.utils.calculations import (
    absolute_change,
    percentage_change,
    round_or_none,
    safe_float,
)
from app.utils.errors import InsufficientDataError


@dataclass(frozen=True, slots=True)
class Contribution:
    """One dimension member's contribution to a metric change."""

    dimension: str
    dimension_value: str
    current_value: float | None
    comparison_value: float | None
    absolute_change: float | None
    percentage_change: float | None
    #: share of the total absolute movement attributable to this member (percent)
    contribution_share: float | None
    sample_size: int


@dataclass(frozen=True, slots=True)
class ContributionBreakdown:
    """A ranked set of contributions for one dimension."""

    dimension: str
    metric: str
    total_current: float | None
    total_comparison: float | None
    total_change: float | None
    contributions: tuple[Contribution, ...]

    def top(self, n: int) -> tuple[Contribution, ...]:
        return self.contributions[:n]

    def contribution_share_max(self) -> float | None:
        """The largest single-member share of total absolute movement.

        Used to set the severity of the headline movement: a change driven by one
        member is more actionable than the same change spread thinly.
        """
        shares = [
            c.contribution_share for c in self.contributions if c.contribution_share is not None
        ]
        return max(shares) if shares else None


def contribution_breakdown(
    primary: pd.DataFrame,
    comparison: pd.DataFrame,
    dimension_column: str,
    *,
    metric_label: str,
    value_fn: ValueFn,
    total_current: float | None,
    total_comparison: float | None,
    top_n: int = 5,
) -> ContributionBreakdown:
    """Attribute a metric change across the values of ``dimension_column``.

    ``value_fn`` maps a sub-frame to the metric value for one dimension member.
    It is applied per dimension member in each period so that the contribution
    numbers are computed with the same deterministic function as the headline
    metric. The share of movement is the member's absolute change divided by the
    sum of absolute changes across all members (so shares are comparable and
    sum to ~100% in magnitude).
    """
    members: dict[str, dict[str, Any]] = {}

    for value, group in primary.groupby(dimension_column, dropna=True):
        members.setdefault(str(value), {})["current"] = value_fn(group)
        members[str(value)]["current_n"] = len(group)
    for value, group in comparison.groupby(dimension_column, dropna=True):
        members.setdefault(str(value), {})["comparison"] = value_fn(group)
        members[str(value)]["comparison_n"] = len(group)

    rows: list[Contribution] = []
    for member, data in members.items():
        current = safe_float(data.get("current"))
        previous = safe_float(data.get("comparison"))
        change = absolute_change(current, previous)
        rows.append(
            Contribution(
                dimension=dimension_column,
                dimension_value=member,
                current_value=round_or_none(current),
                comparison_value=round_or_none(previous),
                absolute_change=change,
                percentage_change=percentage_change(current, previous),
                contribution_share=None,  # filled below
                sample_size=int(data.get("current_n") or data.get("comparison_n") or 0),
            )
        )

    total_abs = sum(abs(c.absolute_change) for c in rows if c.absolute_change is not None)
    ranked: list[Contribution] = []
    for contribution in rows:
        share = (
            round_or_none(abs(contribution.absolute_change) / total_abs * 100.0)
            if contribution.absolute_change is not None and total_abs > 0
            else None
        )
        ranked.append(
            Contribution(
                dimension=contribution.dimension,
                dimension_value=contribution.dimension_value,
                current_value=contribution.current_value,
                comparison_value=contribution.comparison_value,
                absolute_change=contribution.absolute_change,
                percentage_change=contribution.percentage_change,
                contribution_share=share,
                sample_size=contribution.sample_size,
            )
        )

    # Rank by the magnitude of absolute change (largest movers first).
    ranked.sort(
        key=lambda c: abs(c.absolute_change) if c.absolute_change is not None else -1.0,
        reverse=True,
    )

    return ContributionBreakdown(
        dimension=dimension_column,
        metric=metric_label,
        total_current=round_or_none(total_current),
        total_comparison=round_or_none(total_comparison),
        total_change=absolute_change(total_current, total_comparison),
        contributions=tuple(ranked[:top_n]),
    )


#: A `value_fn` maps one dimension member slice to its metric value.
ValueFn = Callable[[pd.DataFrame], float | None]

#: A factory builds the `value_fn` for one contributed metric.
ValueFnFactory = Callable[[], ValueFn]


def weighted_mean_value(column: str, weight_column: str = "completed_operations") -> ValueFn:
    """Return a ``value_fn`` computing a completed-ops-weighted mean of ``column``."""

    def _fn(frame: pd.DataFrame) -> float | None:
        if frame.empty or column not in frame.columns:
            return None
        weights = (
            frame[weight_column].fillna(0).astype("float64")
            if weight_column in frame.columns
            else None
        )
        if weights is None or float(weights.sum()) <= 0:
            return safe_float(frame[column].mean())
        return safe_float((frame[column] * weights).sum() / float(weights.sum()))

    return _fn


def simple_mean_value(column: str) -> ValueFn:
    """Return a ``value_fn`` computing an unweighted mean of ``column``."""

    def _fn(frame: pd.DataFrame) -> float | None:
        if frame.empty or column not in frame.columns:
            return None
        return safe_float(frame[column].mean())

    return _fn


def rate_per_day_value(count_column: str = "completed_operations") -> ValueFn:
    """Return a ``value_fn`` computing completed count per distinct day."""

    def _fn(frame: pd.DataFrame) -> float | None:
        if frame.empty or count_column not in frame.columns:
            return None
        days = int(frame["date"].nunique()) if "date" in frame.columns else 0
        if days <= 0:
            return None
        return safe_float(float(frame[count_column].sum()) / days)

    return _fn


def median_value(column: str) -> ValueFn:
    """Return a ``value_fn`` computing the unweighted median of ``column``."""

    def _fn(frame: pd.DataFrame) -> float | None:
        if frame.empty or column not in frame.columns:
            return None
        return safe_float(frame[column].median())

    return _fn


def closing_sum_value(column: str, date_column: str = "date") -> ValueFn:
    """Return a ``value_fn`` summing ``column`` on the latest date in the frame.

    Stock metrics such as open backlog are a point-in-time balance: summing every
    row in a window would add the same balance up once per day.
    """

    def _fn(frame: pd.DataFrame) -> float | None:
        if frame.empty or column not in frame.columns:
            return None
        if date_column not in frame.columns:
            return safe_float(frame[column].sum())
        latest = frame[date_column].max()
        return safe_float(frame.loc[frame[date_column] == latest, column].sum())

    return _fn


def ratio_percentage_value(numerator: str, denominator: str) -> ValueFn:
    """Return a ``value_fn`` computing ``numerator / denominator * 100``.

    Used for share metrics (queue contribution, delay rate, on-time delivery)
    where a mean of per-row ratios would over-weight small rows.
    """

    def _fn(frame: pd.DataFrame) -> float | None:
        if frame.empty or numerator not in frame.columns or denominator not in frame.columns:
            return None
        bottom = float(frame[denominator].fillna(0).sum())
        if bottom <= 0:
            return None
        return safe_float(float(frame[numerator].fillna(0).sum()) / bottom * 100.0)

    return _fn


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """How one metric is measured for contribution purposes.

    ``value_fn`` is applied identically to the whole window and to each dimension
    member, which is what keeps the headline number and the parts consistent.
    """

    #: ``"operations"`` or ``"orders"`` - which dataset carries the metric.
    grain: str
    unit: Unit
    label: str
    calculation: str
    value_fn: ValueFnFactory


def _spec(
    grain: str, unit: Unit, label: str, calculation: str, factory: ValueFnFactory
) -> MetricSpec:
    return MetricSpec(
        grain=grain,
        unit=unit,
        label=label,
        calculation=calculation,
        value_fn=factory,
    )


#: Metrics that can be decomposed across a dimension, and how.
CONTRIBUTION_METRICS: dict[Metric, MetricSpec] = {
    Metric.AVERAGE_CYCLE_TIME: _spec(
        "operations",
        Unit.HOURS,
        "Average cycle time",
        "completed_operations-weighted mean of total_cycle_time_hours",
        lambda: weighted_mean_value("total_cycle_time_hours"),
    ),
    Metric.MEDIAN_CYCLE_TIME: _spec(
        "operations",
        Unit.HOURS,
        "Median cycle time",
        "median of total_cycle_time_hours",
        lambda: median_value("total_cycle_time_hours"),
    ),
    Metric.PROCESSING_TIME: _spec(
        "operations",
        Unit.HOURS,
        "Average processing time",
        "completed_operations-weighted mean of processing_time_hours",
        lambda: weighted_mean_value("processing_time_hours"),
    ),
    Metric.QUEUE_TIME: _spec(
        "operations",
        Unit.HOURS,
        "Average queue time",
        "completed_operations-weighted mean of queue_time_hours",
        lambda: weighted_mean_value("queue_time_hours"),
    ),
    Metric.QUEUE_CONTRIBUTION: _spec(
        "operations",
        Unit.PERCENT,
        "Queue contribution to cycle time",
        "sum of queue_time_hours / sum of total_cycle_time_hours * 100",
        lambda: ratio_percentage_value("queue_time_hours", "total_cycle_time_hours"),
    ),
    Metric.THROUGHPUT: _spec(
        "operations",
        Unit.OPERATIONS,
        "Throughput",
        "completed_operations summed per distinct day",
        lambda: rate_per_day_value("completed_operations"),
    ),
    Metric.DELAY_RATE: _spec(
        "operations",
        Unit.PERCENT,
        "Delay rate",
        "delayed_operations / completed_operations * 100",
        lambda: ratio_percentage_value("delayed_operations", "completed_operations"),
    ),
    Metric.BACKLOG: _spec(
        "operations",
        Unit.ITEMS,
        "Closing backlog",
        "backlog_open_items summed on the final day of the window",
        lambda: closing_sum_value("backlog_open_items"),
    ),
    Metric.FULFILLMENT_TIME: _spec(
        "orders",
        Unit.HOURS,
        "Average order fulfilment time",
        "mean of fulfillment_time_hours over non-cancelled orders",
        lambda: simple_mean_value("fulfillment_time_hours"),
    ),
    Metric.ON_TIME_DELIVERY_RATE: _spec(
        "orders",
        Unit.PERCENT,
        "On-time delivery rate",
        "orders with actual_time_hours <= promised_time_hours / non-cancelled orders * 100",
        lambda: on_time_delivery_value(),
    ),
    Metric.ORDER_VALUE: _spec(
        "orders",
        Unit.CURRENCY,
        "Average order value",
        "mean of order_value over non-cancelled orders",
        lambda: simple_mean_value("order_value"),
    ),
}

#: Dimension -> column, per grain.
DIMENSION_COLUMNS: dict[str, dict[str, str]] = {
    "operations": {
        "department": "department",
        "process": "process",
        "operation_type": "operation_type",
        "priority": "priority",
        "vendor": "vendor_id",
        "status": "status",
    },
    "orders": {
        "department": "department",
        "product_category": "product_category",
        "vendor": "vendor_id",
        "status": "status",
    },
}

#: Default dimension per grain when the plan requests none.
DEFAULT_DIMENSION: dict[str, str] = {"operations": "department", "orders": "department"}

#: Minimum members before a contribution share is worth reporting.
MIN_MEMBERS = 2


def on_time_delivery_value() -> ValueFn:
    """Return a ``value_fn`` computing the on-time delivery percentage."""

    def _fn(frame: pd.DataFrame) -> float | None:
        if frame.empty:
            return None
        if "actual_time_hours" not in frame.columns or "promised_time_hours" not in frame.columns:
            return None
        counted = len(frame)
        if counted <= 0:
            return None
        on_time = int((frame["actual_time_hours"] <= frame["promised_time_hours"]).sum())
        return safe_float(on_time / counted * 100.0)

    return _fn


class ContributionAnalyzer(BaseAnalyzer):
    """Attribute a metric's period-over-period change to dimension members.

    This answers the question a plain period comparison cannot: not *that* cycle
    time rose, but *which department's* cycle time accounts for the rise. The
    method is explicit and reported with every number:

    * each member's value is computed in both windows with the same deterministic
      function used for the headline metric;
    * the member's movement is its current value minus its comparison value;
    * its share of movement is ``|member change| / sum(|member change|) * 100``,
      so shares express which members moved most in magnitude, not which pushed
      the total in a particular direction.

    A member appearing in only one window keeps its share as undefined rather
    than being treated as a change from zero, because "absent" and "zero" are
    different facts.
    """

    name = "contribution_analyzer"
    supported_types = frozenset({AnalysisType.CONTRIBUTION_ANALYSIS})
    required_datasets = (DatasetId.OPERATIONS, DatasetId.ORDERS, DatasetId.DEPARTMENTS)

    def analyze(self, context: AnalysisContext) -> AnalysisOutcome:
        spec = CONTRIBUTION_METRICS.get(context.plan.metric)
        if spec is None:
            raise InsufficientDataError(
                f"metric '{context.plan.metric.value}' cannot be decomposed across a "
                "dimension; decomposable metrics: "
                + ", ".join(sorted(item.value for item in CONTRIBUTION_METRICS))
            )

        dimension = self._dimension_column(context, spec.grain)
        primary, comparison = self._frames(context, spec.grain)
        if primary.empty:
            raise InsufficientDataError(
                f"no records found for {context.plan.period or 'the requested period'} to "
                f"decompose {spec.label} across {dimension}"
            )
        if comparison is None or comparison.empty:
            raise InsufficientDataError(
                "contribution analysis needs records in both windows; "
                f"{context.plan.comparison_period or 'the comparison period'} has no rows"
            )

        outcome = AnalysisOutcome()
        value_fn = spec.value_fn()
        breakdown = contribution_breakdown(
            primary,
            comparison,
            dimension,
            metric_label=spec.label,
            value_fn=value_fn,
            total_current=value_fn(primary),
            total_comparison=value_fn(comparison),
            top_n=context.top_n,
        )
        if not breakdown.contributions:
            raise InsufficientDataError(
                f"no {dimension} values are present in both windows, so no contribution can "
                "be attributed"
            )

        period_label = context.plan.period or "the current window"
        comparison_label = context.plan.comparison_period or "the comparison window"
        names = context.repository.department_names
        dataset = DatasetId.OPERATIONS if spec.grain == "operations" else DatasetId.ORDERS

        outcome.kpis = [
            build_kpi(
                name=spec.label,
                value=breakdown.total_current,
                unit=spec.unit,
                source_dataset=dataset,
                period=period_label,
                calculation=spec.calculation,
            ),
            build_kpi(
                name=f"{spec.label} change",
                value=breakdown.total_change,
                unit=spec.unit,
                source_dataset=dataset,
                period=period_label,
                calculation=(
                    f"{spec.calculation} in {comparison_label}, subtracted from the same "
                    "measure in the current window"
                ),
            ),
            build_kpi(
                name=f"Members driving the change by {dimension}",
                value=float(len(breakdown.contributions)),
                unit=Unit.COUNT,
                source_dataset=dataset,
                period=period_label,
                calculation=(
                    f"count of {dimension} values with a non-zero movement between the two windows"
                ),
            ),
        ]

        outcome.evidence = _contribution_evidence(
            breakdown, spec, dataset, dimension, period_label, comparison_label, names
        )
        outcome.findings = _contribution_findings(
            breakdown, spec, dimension, period_label, comparison_label, names
        )

        outcome.extend_caveats(
            "contribution shares are shares of total absolute movement, not of the signed "
            "total: a member that improved is as likely to appear as one that worsened",
            f"each {dimension} value is measured independently in both windows, so the parts "
            "do not necessarily sum to the headline change when the metric is an average or "
            "a rate",
        )
        if len(breakdown.contributions) < MIN_MEMBERS:
            outcome.extend_caveats(
                f"only {len(breakdown.contributions)} {dimension} value(s) moved between the "
                "two windows, so no meaningful driver can be identified"
            )
        outcome.row_count_examined = len(primary) + len(comparison)
        return outcome

    # -- resolution -----------------------------------------------------
    @staticmethod
    def _dimension_column(context: AnalysisContext, grain: str) -> str:
        columns = DIMENSION_COLUMNS[grain]
        for dimension in context.plan.dimensions:
            column = columns.get(dimension.value)
            if column:
                return column
        return DEFAULT_DIMENSION[grain]

    @staticmethod
    def _frames(context: AnalysisContext, grain: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Return the primary and comparison frames for the metric's grain."""
        if grain == "orders":
            primary = context.frame(DatasetId.ORDERS)
            comparison = context.frame(DatasetId.ORDERS, window="comparison")
            return _active_orders(primary), _active_orders(comparison)
        return context.primary_frame(), context.comparison_frame()


def _active_orders(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Drop cancelled orders, which never reached the delivery stage."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    active = frame.loc[frame["status"] != "cancelled"]
    return active if not active.empty else frame


def _label(value: str, dimension: str, department_names: dict[str, str]) -> str:
    return department_names.get(value, value) if dimension == "department" else value


def _contribution_evidence(
    breakdown: ContributionBreakdown,
    spec: MetricSpec,
    dataset: DatasetId,
    dimension: str,
    period_label: str,
    comparison_label: str,
    names: dict[str, str],
) -> list[EvidenceItem]:
    """One evidence item per contributing member, plus the headline movement."""
    items: list[EvidenceItem] = [
        build_evidence(
            metric=f"{spec.label} change by {dimension}",
            value=breakdown.total_current,
            comparison_value=breakdown.total_comparison,
            unit=spec.unit,
            source_dataset=dataset,
            period=period_label,
            calculation_description=(
                f"{spec.calculation} across all {dimension} values, compared with the same "
                f"measure over {comparison_label}"
            ),
            severity=Severity.HIGH if abs(breakdown.total_change or 0.0) > 0 else Severity.INFO,
        )
    ]
    for contribution in breakdown.contributions:
        label = _label(contribution.dimension_value, dimension, names)
        items.append(
            build_evidence(
                metric=f"{spec.label} change by {dimension}",
                value=contribution.current_value,
                comparison_value=contribution.comparison_value,
                unit=spec.unit,
                source_dataset=dataset,
                dimension=label,
                period=period_label,
                calculation_description=(
                    f"{spec.calculation} for {label}, minus the same measure over "
                    f"{comparison_label}; contribution share is "
                    f"{contribution.contribution_share:.1f}% of the total absolute movement "
                    "across all members"
                )
                if contribution.contribution_share is not None
                else (
                    f"{spec.calculation} for {label}, minus the same measure over "
                    f"{comparison_label}; no share is reported because the member exists in "
                    "only one of the two windows"
                ),
                sample_size=contribution.sample_size,
                severity=_share_severity(contribution.contribution_share),
            )
        )
    return items


def _contribution_findings(
    breakdown: ContributionBreakdown,
    spec: MetricSpec,
    dimension: str,
    period_label: str,
    comparison_label: str,
    names: dict[str, str],
) -> list[Finding]:
    """Ranked findings: the headline movement, then the top drivers."""
    direction = "increased" if (breakdown.total_change or 0.0) >= 0 else "decreased"
    findings: list[Finding] = [
        build_finding(
            title=f"{spec.label} {direction} between the two windows",
            detail=(
                f"{spec.label} moved from {breakdown.total_comparison} to "
                f"{breakdown.total_current} {spec.unit.value} between {comparison_label} and "
                f"{period_label}, a change of {breakdown.total_change:+.4f} "
                f"({percentage_change(breakdown.total_current, breakdown.total_comparison)}%). "
                f"The movement is decomposed below by {dimension}."
            ),
            severity=_share_severity(breakdown.contribution_share_max()),
            metric=spec.label.lower().replace(" ", "_"),
            period=period_label,
            statistic=(
                f"{breakdown.total_current} - {breakdown.total_comparison} = "
                f"{breakdown.total_change:+.4f} {spec.unit.value}"
            ),
        )
    ]
    for rank, contribution in enumerate(breakdown.contributions[:3], start=1):
        label = _label(contribution.dimension_value, dimension, names)
        share = contribution.contribution_share
        moved = "increased" if (contribution.absolute_change or 0.0) >= 0 else "decreased"
        findings.append(
            build_finding(
                title=f"#{rank} contributor by {dimension}: {label}",
                detail=(
                    f"{label}'s {spec.label.lower()} {moved} from "
                    f"{contribution.comparison_value} to {contribution.current_value} "
                    f"{spec.unit.value}, a change of {contribution.absolute_change:+.4f}"
                    + (
                        f", accounting for {share:.1f}% of the total absolute movement across "
                        "all members"
                        if share is not None
                        else ""
                    )
                    + "."
                ),
                severity=_share_severity(share),
                metric=spec.label.lower().replace(" ", "_"),
                dimension=label,
                period=period_label,
                statistic=(
                    f"contribution share = {share:.1f}%"
                    if share is not None
                    else "share unavailable"
                ),
            )
        )
    return findings


def _share_severity(share: float | None) -> Severity:
    value = safe_float(share)
    if value is None:
        return Severity.LOW
    if value >= 50.0:
        return Severity.HIGH
    if value >= 25.0:
        return Severity.MEDIUM
    if value >= 10.0:
        return Severity.LOW
    return Severity.INFO


__all__ = [
    "CONTRIBUTION_METRICS",
    "DIMENSION_COLUMNS",
    "Contribution",
    "ContributionAnalyzer",
    "ContributionBreakdown",
    "MetricSpec",
    "closing_sum_value",
    "contribution_breakdown",
    "median_value",
    "on_time_delivery_value",
    "rate_per_day_value",
    "ratio_percentage_value",
    "simple_mean_value",
    "weighted_mean_value",
]
