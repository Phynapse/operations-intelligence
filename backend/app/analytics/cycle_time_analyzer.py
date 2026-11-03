"""Cycle time analysis: where the time in an operation actually goes.

Cycle time decomposes into two components:

    total_cycle_time = queue_time + processing_time

The key operational question this analyzer answers is *which component* moved.
A rise driven by queue time points at under-capacity or a prioritisation problem;
a rise driven by processing time points at the work itself. That distinction is
what makes the evidence useful for root-cause reasoning.
"""

from __future__ import annotations

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
    contribution,
    percentage_change,
    percentage_of_total,
    round_or_none,
    weighted_mean,
)
from app.utils.errors import InsufficientDataError

DIMENSION_COLUMNS: dict[str, str] = {
    "department": "department",
    "process": "process",
    "operation_type": "operation_type",
    "priority": "priority",
    "vendor": "vendor_id",
}

#: Metrics that are measured on ``orders.csv`` rather than ``operations.csv``.
ORDER_METRICS: frozenset[Metric] = frozenset(
    {Metric.FULFILLMENT_TIME, Metric.ON_TIME_DELIVERY_RATE, Metric.ORDER_VALUE}
)

#: orders.csv dimension columns.
ORDER_DIMENSION_COLUMNS: dict[str, str] = {
    "department": "department",
    "vendor": "vendor_id",
    "process": "product_category",
    "status": "status",
}


class CycleTimeAnalyzer(BaseAnalyzer):
    """Cycle time with a wait/work decomposition.

    Two grains are supported because "cycle time" means two different things in
    this dataset:

    * operations (``operations.csv``) -
      ``total_cycle_time = queue_time + processing_time``;
    * orders (``orders.csv``) -
      ``actual_time = fulfillment_time + transit_time``, where fulfilment time
      is our own handling and transit time belongs to the carrier.

    The decomposition logic is shared; only the column names and the labels
    differ. Order grain is selected when the requested metric is order-derived.
    """

    name = "cycle_time_analyzer"
    supported_types = frozenset(
        {
            AnalysisType.CYCLE_TIME_ANALYSIS,
            AnalysisType.PROCESS_ANALYSIS,
            AnalysisType.PERIOD_COMPARISON,
            AnalysisType.DEPARTMENT_ANALYSIS,
        }
    )
    required_datasets = (DatasetId.OPERATIONS, DatasetId.ORDERS)

    def analyze(self, context: AnalysisContext) -> AnalysisOutcome:
        if context.plan.metric in ORDER_METRICS:
            return self._analyze_orders(context)
        return self._analyze_operations(context)

    # -- operations grain ----------------------------------------------
    def _analyze_operations(self, context: AnalysisContext) -> AnalysisOutcome:
        primary = context.primary_frame()
        if primary.empty:
            raise InsufficientDataError(
                f"no operational records found for {context.plan.period or 'the requested period'}"
            )

        outcome = AnalysisOutcome()
        period_label = context.plan.period or "all periods"
        comparison = context.comparison_frame()

        current = cycle_components(primary)
        baseline = cycle_components(comparison) if comparison is not None else None
        volume = int(primary["completed_operations"].sum())

        outcome.kpis = [
            build_kpi(
                name="Average cycle time",
                value=current["cycle"],
                unit=Unit.HOURS,
                source_dataset=DatasetId.OPERATIONS,
                period=period_label,
                calculation=("completed_operations-weighted mean of total_cycle_time_hours"),
            ),
            build_kpi(
                name="Queue contribution",
                value=current["queue_share"],
                unit=Unit.PERCENT,
                source_dataset=DatasetId.OPERATIONS,
                period=period_label,
                calculation=("weighted mean of queue_time_hours / total_cycle_time_hours * 100"),
            ),
            build_kpi(
                name="Processing contribution",
                value=current["processing_share"],
                unit=Unit.PERCENT,
                source_dataset=DatasetId.OPERATIONS,
                period=period_label,
                calculation=(
                    "weighted mean of processing_time_hours / total_cycle_time_hours * 100"
                ),
            ),
        ]

        outcome.evidence = [
            build_evidence(
                metric="Average cycle time",
                value=current["cycle"],
                comparison_value=baseline["cycle"] if baseline else None,
                unit=Unit.HOURS,
                source_dataset=DatasetId.OPERATIONS,
                period=period_label,
                comparison_period=context.plan.comparison_period,
                calculation_description=(
                    "Mean of total_cycle_time_hours weighted by completed_operations"
                ),
                sample_size=volume,
                severity=change_severity(
                    absolute_change(current["cycle"], baseline["cycle"] if baseline else None),
                    scale=current["cycle"] or 1.0,
                ),
            ),
            build_evidence(
                metric="Average queue time",
                value=current["queue"],
                comparison_value=baseline["queue"] if baseline else None,
                unit=Unit.HOURS,
                source_dataset=DatasetId.OPERATIONS,
                period=period_label,
                comparison_period=context.plan.comparison_period,
                calculation_description=(
                    "Mean of queue_time_hours weighted by completed_operations"
                ),
                sample_size=volume,
                severity=change_severity(
                    absolute_change(current["queue"], baseline["queue"] if baseline else None),
                    scale=current["queue"] or 1.0,
                ),
            ),
            build_evidence(
                metric="Average processing time",
                value=current["processing"],
                comparison_value=baseline["processing"] if baseline else None,
                unit=Unit.HOURS,
                source_dataset=DatasetId.OPERATIONS,
                period=period_label,
                comparison_period=context.plan.comparison_period,
                calculation_description=(
                    "Mean of processing_time_hours weighted by completed_operations"
                ),
                sample_size=volume,
                severity=change_severity(
                    absolute_change(
                        current["processing"], baseline["processing"] if baseline else None
                    ),
                    scale=current["processing"] or 1.0,
                ),
            ),
            build_evidence(
                metric="Queue contribution to cycle time",
                value=current["queue_share"],
                comparison_value=baseline["queue_share"] if baseline else None,
                unit=Unit.PERCENT,
                source_dataset=DatasetId.OPERATIONS,
                period=period_label,
                comparison_period=context.plan.comparison_period,
                calculation_description=(
                    "queue_time_hours / total_cycle_time_hours * 100, weighted by "
                    "completed_operations"
                ),
                sample_size=volume,
            ),
        ]

        outcome.findings.extend(self._component_findings(current, baseline, period_label))

        column = self._dimension_column(context)
        if column:
            outcome.evidence.extend(self._dimension_evidence(primary, comparison, column))
            outcome.findings.extend(
                self._dimension_findings(primary, comparison, column, period_label)
            )

        outcome.extend_caveats(*self._consistency_caveats(primary))
        outcome.row_count_examined = len(primary)
        return outcome

    # -- orders grain ---------------------------------------------------
    def _analyze_orders(self, context: AnalysisContext) -> AnalysisOutcome:
        """Cycle time measured per order: our handling versus carrier transit."""
        primary = context.frame(DatasetId.ORDERS)
        if primary.empty:
            raise InsufficientDataError(
                f"no orders found for {context.plan.period or 'the requested period'}"
            )
        active = primary.loc[primary["status"] != "cancelled"]
        if active.empty:
            active = primary

        outcome = AnalysisOutcome()
        period_label = context.plan.period or "all periods"
        comparison_frame = context.frame(DatasetId.ORDERS, window="comparison")
        comparison = (
            comparison_frame.loc[comparison_frame["status"] != "cancelled"]
            if not comparison_frame.empty
            else None
        )
        if comparison is not None and comparison.empty:
            comparison = None

        current = order_components(active)
        baseline = order_components(comparison) if comparison is not None else None
        counted = len(active)

        outcome.kpis = [
            build_kpi(
                name="Average order fulfilment time",
                value=current["handling"],
                unit=Unit.HOURS,
                source_dataset=DatasetId.ORDERS,
                period=period_label,
                calculation="mean of fulfillment_time_hours over non-cancelled orders",
            ),
            build_kpi(
                name="Average transit time",
                value=current["transit"],
                unit=Unit.HOURS,
                source_dataset=DatasetId.ORDERS,
                period=period_label,
                calculation="mean of transit_time_hours over non-cancelled orders",
            ),
            build_kpi(
                name="Average end-to-end order time",
                value=current["total"],
                unit=Unit.HOURS,
                source_dataset=DatasetId.ORDERS,
                period=period_label,
                calculation=(
                    "mean of actual_time_hours over non-cancelled orders "
                    "(fulfillment_time_hours + transit_time_hours)"
                ),
            ),
            build_kpi(
                name="On-time delivery rate",
                value=current["on_time_rate"],
                unit=Unit.PERCENT,
                source_dataset=DatasetId.ORDERS,
                period=period_label,
                calculation=(
                    "orders with actual_time_hours <= promised_time_hours, divided by "
                    "non-cancelled orders, times 100"
                ),
            ),
        ]

        outcome.evidence = [
            build_evidence(
                metric="Average order fulfilment time",
                value=current["handling"],
                comparison_value=baseline["handling"] if baseline else None,
                unit=Unit.HOURS,
                source_dataset=DatasetId.ORDERS,
                period=period_label,
                comparison_period=context.plan.comparison_period,
                calculation_description=(
                    "Mean of fulfillment_time_hours, the portion of the order spent inside "
                    "our own operation"
                ),
                sample_size=counted,
                severity=change_severity(
                    absolute_change(
                        current["handling"], baseline["handling"] if baseline else None
                    ),
                    scale=current["handling"] or 1.0,
                ),
            ),
            build_evidence(
                metric="Average transit time",
                value=current["transit"],
                comparison_value=baseline["transit"] if baseline else None,
                unit=Unit.HOURS,
                source_dataset=DatasetId.ORDERS,
                period=period_label,
                comparison_period=context.plan.comparison_period,
                calculation_description=(
                    "Mean of transit_time_hours, the portion of the order spent with the carrier"
                ),
                sample_size=counted,
                severity=change_severity(
                    absolute_change(current["transit"], baseline["transit"] if baseline else None),
                    scale=current["transit"] or 1.0,
                ),
            ),
            build_evidence(
                metric="On-time delivery rate",
                value=current["on_time_rate"],
                comparison_value=baseline["on_time_rate"] if baseline else None,
                unit=Unit.PERCENT,
                source_dataset=DatasetId.ORDERS,
                period=period_label,
                comparison_period=context.plan.comparison_period,
                calculation_description=(
                    "Orders delivered within promised_time_hours / non-cancelled orders * 100"
                ),
                sample_size=counted,
                severity=change_severity(
                    absolute_change(
                        current["on_time_rate"], baseline["on_time_rate"] if baseline else None
                    ),
                    scale=10.0,
                ),
            ),
            build_evidence(
                metric="Carrier share of end-to-end order time",
                value=current["transit_share"],
                unit=Unit.PERCENT,
                source_dataset=DatasetId.ORDERS,
                period=period_label,
                calculation_description=(
                    "transit_time_hours / actual_time_hours * 100 averaged over non-cancelled "
                    "orders"
                ),
                sample_size=counted,
            ),
        ]

        outcome.findings.extend(_order_component_findings(current, baseline, period_label))

        column = self._order_dimension_column(context)
        if column:
            outcome.evidence.extend(self._order_dimension_evidence(active, comparison, column))
            outcome.findings.extend(
                self._order_dimension_findings(active, comparison, column, context)
            )

        outcome.extend_caveats(
            "cancelled orders are excluded from fulfilment-time averages because they never "
            "reached the delivery stage; they are still counted in throughput analysis",
            "transit time is attributed entirely to the recorded carrier, so a carrier change "
            "appears as a transit movement even when our own handling is unchanged",
        )
        if counted < 50:
            outcome.extend_caveats(
                f"only {counted} orders back this result, so the averages are sensitive to a "
                "handful of long-running orders"
            )
        outcome.row_count_examined = len(primary)
        return outcome

    @staticmethod
    def _order_dimension_column(context: AnalysisContext) -> str | None:
        for dimension in context.plan.dimensions:
            column = ORDER_DIMENSION_COLUMNS.get(dimension.value)
            if column:
                return column
        return None

    def _order_dimension_evidence(
        self, primary: pd.DataFrame, comparison: pd.DataFrame | None, column: str
    ) -> list[EvidenceItem]:
        table = order_table(primary, column)
        if table.empty:
            return []
        slowest = table.sort_values("handling", ascending=False).iloc[0]
        items = [
            build_evidence(
                metric=f"Highest average fulfilment time by {column}",
                value=round_or_none(slowest["handling"]),
                unit=Unit.HOURS,
                source_dataset=DatasetId.ORDERS,
                dimension=str(slowest[column]),
                calculation_description=(
                    f"Mean of fulfillment_time_hours per {column} value over non-cancelled "
                    "orders, ranked highest first; the slowest is reported"
                ),
                sample_size=int(slowest["orders"]),
                severity=Severity.MEDIUM,
            )
        ]
        if comparison is not None and not comparison.empty:
            movement = largest_order_movement(table, order_table(comparison, column), column)
            if movement is not None:
                items.append(
                    build_evidence(
                        metric=f"Largest fulfilment time movement by {column}",
                        value=round_or_none(movement["handling"]),
                        comparison_value=round_or_none(movement["comparison_handling"]),
                        unit=Unit.HOURS,
                        source_dataset=DatasetId.ORDERS,
                        dimension=str(movement[column]),
                        calculation_description=(
                            f"Mean fulfillment_time_hours per {column} value in both periods, "
                            "ranked by absolute difference between periods"
                        ),
                        sample_size=int(movement.get("orders") or 0),
                        severity=change_severity(
                            movement["change"], scale=movement["handling"] or 1.0
                        ),
                    )
                )
        return items

    def _order_dimension_findings(
        self,
        primary: pd.DataFrame,
        comparison: pd.DataFrame | None,
        column: str,
        context: AnalysisContext,
    ) -> list[Finding]:
        table = order_table(primary, column)
        if table.empty:
            return []
        names = context.repository.department_names
        worst = table.sort_values("on_time_rate").iloc[0]
        label = names.get(str(worst[column]), str(worst[column]))
        findings = [
            build_finding(
                title=f"Lowest on-time delivery: {label}",
                detail=(
                    f"{label} delivered {worst['on_time_rate']:.1f}% of {worst['orders']:,.0f} "
                    f"orders within the promise, with an average fulfilment time of "
                    f"{worst['handling']:.2f} hours and transit of {worst['transit']:.2f} hours."
                ),
                severity=Severity.MEDIUM if worst["on_time_rate"] < 85.0 else Severity.LOW,
                metric="on_time_delivery_rate",
                dimension=label,
                period=context.plan.period,
                statistic=(
                    f"{worst['on_time_delivered']:,.0f}/{worst['orders']:,.0f} orders on time"
                ),
            )
        ]
        if comparison is not None and not comparison.empty:
            movement = largest_order_movement(table, order_table(comparison, column), column)
            if movement is not None and movement["change"]:
                label = names.get(str(movement[column]), str(movement[column]))
                direction = "increased" if movement["change"] > 0 else "decreased"
                handling_pct = (
                    percentage_change(movement["handling"], movement["comparison_handling"])
                    if movement["comparison_handling"]
                    else "n/a"
                )
                findings.append(
                    build_finding(
                        title=f"Largest fulfilment time movement: {label}",
                        detail=(
                            f"Average fulfilment time for {label} {direction} by "
                            f"{abs(movement['change']):.2f} hours between the two periods "
                            f"({handling_pct}%)."
                        ),
                        severity=change_severity(
                            movement["change"], scale=movement["handling"] or 1.0
                        ),
                        metric="average_cycle_time",
                        dimension=label,
                        statistic=f"fulfilment time delta = {movement['change']:+.2f} hours",
                    )
                )
        return findings

    # -- internals -----------------------------------------------------
    @staticmethod
    def _dimension_column(context: AnalysisContext) -> str | None:
        for dimension in context.plan.dimensions:
            column = DIMENSION_COLUMNS.get(dimension.value)
            if column:
                return column
        return None

    @staticmethod
    def _component_findings(
        current: dict[str, float | None],
        baseline: dict[str, float | None] | None,
        period_label: str,
    ) -> list[Finding]:
        if baseline is None:
            return []
        queue_delta = absolute_change(current["queue"], baseline["queue"])
        processing_delta = absolute_change(current["processing"], baseline["processing"])
        if queue_delta is None and processing_delta is None:
            return []

        findings: list[Finding] = []
        dominant = "processing"
        dominant_delta = processing_delta
        if queue_delta is not None and processing_delta is not None:
            dominant = "queue" if abs(queue_delta) >= abs(processing_delta) else "processing"
            dominant_delta = queue_delta if dominant == "queue" else processing_delta
        elif queue_delta is not None:
            dominant = "queue"
            dominant_delta = queue_delta

        if dominant_delta and abs(dominant_delta) > 0.05:
            direction = "increased" if dominant_delta > 0 else "decreased"
            opposite = "processing" if dominant == "queue" else "queue"
            opposite_delta = processing_delta if dominant == "queue" else queue_delta
            contrast = (
                f" while {opposite} time moved by {opposite_delta:+.2f} hours"
                if opposite_delta is not None
                else ""
            )
            findings.append(
                build_finding(
                    title=f"Cycle time change driven by {dominant} time",
                    detail=(
                        f"{dominant.capitalize()} time {direction} by "
                        f"{abs(dominant_delta):.2f} hours between the two periods{contrast}. "
                        "This points to a capacity or prioritisation cause rather than a change "
                        "in the work itself."
                    ),
                    severity=change_severity(dominant_delta, scale=current["cycle"] or 1.0),
                    metric=f"{dominant}_time",
                    period=period_label,
                    statistic=f"{dominant}_time delta = {dominant_delta:+.2f} hours",
                )
            )

        share_delta = absolute_change(current["queue_share"], baseline["queue_share"])
        if share_delta is not None and abs(share_delta) >= 3.0:
            findings.append(
                build_finding(
                    title="Queue share of cycle time shifted",
                    detail=(
                        f"Waiting now accounts for {current['queue_share']:.1f}% of cycle time "
                        f"versus {baseline['queue_share']:.1f}% previously, a change of "
                        f"{share_delta:+.1f} percentage points."
                    ),
                    severity=Severity.MEDIUM if abs(share_delta) >= 8.0 else Severity.LOW,
                    metric="queue_contribution",
                    period=period_label,
                    statistic=f"queue share delta = {share_delta:+.2f} pp",
                )
            )
        return findings

    def _dimension_evidence(
        self, primary: pd.DataFrame, comparison: pd.DataFrame | None, column: str
    ) -> list[EvidenceItem]:
        table = cycle_table(primary, column)
        if table.empty:
            return []
        leader = table.sort_values("cycle", ascending=False).iloc[0]
        items = [
            build_evidence(
                metric=f"Slowest average cycle time by {column}",
                value=round_or_none(leader["cycle"]),
                unit=Unit.HOURS,
                source_dataset=DatasetId.OPERATIONS,
                dimension=str(leader[column]),
                calculation_description=(
                    f"completed_operations-weighted mean of total_cycle_time_hours per {column} "
                    "value, ranked highest first; the slowest value is reported"
                ),
                sample_size=int(leader.get("volume") or 0),
                severity=Severity.MEDIUM,
            )
        ]
        if comparison is not None and not comparison.empty:
            movement = largest_cycle_movement(table, cycle_table(comparison, column), column)
            if movement is not None:
                items.append(
                    build_evidence(
                        metric=f"Largest cycle time movement by {column}",
                        value=round_or_none(movement["cycle"]),
                        comparison_value=round_or_none(movement["comparison_cycle"]),
                        unit=Unit.HOURS,
                        source_dataset=DatasetId.OPERATIONS,
                        dimension=str(movement[column]),
                        calculation_description=(
                            f"Weighted mean cycle time for each {column} value in both periods, "
                            "ranked by absolute difference between periods"
                        ),
                        sample_size=int(movement.get("volume") or 0),
                        severity=change_severity(
                            movement["change"], scale=movement["cycle"] or 1.0
                        ),
                    )
                )
        return items

    def _dimension_findings(
        self,
        primary: pd.DataFrame,
        comparison: pd.DataFrame | None,
        column: str,
        period_label: str,
    ) -> list[Finding]:
        table = cycle_table(primary, column)
        if table.empty:
            return []
        slowest = table.sort_values("cycle", ascending=False).iloc[0]
        findings = [
            build_finding(
                title=f"Slowest process by cycle time: {slowest[column]}",
                detail=(
                    f"{slowest[column]} averages {slowest['cycle']:.2f} hours end to end, of which "
                    f"{slowest['queue']:.2f} hours is queue time "
                    f"({slowest['queue_share']:.0f}% of the total)."
                ),
                severity=Severity.MEDIUM,
                metric="average_cycle_time",
                dimension=str(slowest[column]),
                period=period_label,
                statistic=f"weighted mean cycle time = {slowest['cycle']:.2f} hours",
            )
        ]
        if comparison is not None and not comparison.empty:
            movement = largest_cycle_movement(table, cycle_table(comparison, column), column)
            if movement is not None and movement["change"]:
                direction = "increased" if movement["change"] > 0 else "decreased"
                cycle_pct = (
                    percentage_change(movement["cycle"], movement["comparison_cycle"])
                    if movement["comparison_cycle"]
                    else "n/a"
                )
                findings.append(
                    build_finding(
                        title=f"Largest cycle time movement: {movement[column]}",
                        detail=(
                            f"Average cycle time for {movement[column]} {direction} by "
                            f"{abs(movement['change']):.2f} hours between periods "
                            f"({cycle_pct}%)."
                        ),
                        severity=change_severity(
                            movement["change"], scale=movement["cycle"] or 1.0
                        ),
                        metric="average_cycle_time",
                        dimension=str(movement[column]),
                        statistic=(
                            f"weighted mean cycle time delta = {movement['change']:+.2f} hours"
                        ),
                    )
                )
        return findings

    @staticmethod
    def _consistency_caveats(primary: pd.DataFrame) -> list[str]:
        """Report rows whose components do not sum to the stated cycle time."""
        if primary.empty:
            return []
        tolerance = 0.011
        components = primary["queue_time_hours"] + primary["processing_time_hours"]
        mismatched = int((components - primary["total_cycle_time_hours"]).abs().gt(tolerance).sum())
        if mismatched == 0:
            return []
        return [
            f"{mismatched} record(s) have a total_cycle_time_hours that does not equal "
            "queue_time_hours + processing_time_hours; cycle-time shares for those rows are "
            "computed from the stated components rather than the stated total"
        ]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def cycle_components(frame: pd.DataFrame | None) -> dict[str, float | None]:
    """Weighted cycle time, queue time, processing time and their shares."""
    if frame is None or frame.empty:
        return {
            "cycle": None,
            "queue": None,
            "processing": None,
            "queue_share": None,
            "processing_share": None,
        }
    weights = frame["completed_operations"].fillna(0).astype("float64")
    if float(weights.sum()) <= 0:
        weights = pd.Series(1.0, index=frame.index)
    queue = weighted_mean(frame["queue_time_hours"], weights)
    processing = weighted_mean(frame["processing_time_hours"], weights)
    cycle = weighted_mean(frame["total_cycle_time_hours"], weights)
    total = (queue or 0.0) + (processing or 0.0)
    return {
        "cycle": cycle,
        "queue": queue,
        "processing": processing,
        "queue_share": contribution(queue, total) if total > 0 else None,
        "processing_share": contribution(processing, total) if total > 0 else None,
    }


def cycle_table(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Cycle-time breakdown for each value of ``column``."""
    if frame is None or frame.empty or column not in frame.columns:
        return pd.DataFrame(
            columns=[column, "cycle", "queue", "processing", "queue_share", "volume"]
        )
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(column, dropna=True):
        components = cycle_components(group)
        rows.append(
            {
                column: key,
                "cycle": components["cycle"],
                "queue": components["queue"],
                "processing": components["processing"],
                "queue_share": components["queue_share"],
                "volume": int(group["completed_operations"].sum()),
            }
        )
    return pd.DataFrame(rows)


def largest_cycle_movement(
    primary: pd.DataFrame, comparison: pd.DataFrame, column: str
) -> dict[str, Any] | None:
    """The ``column`` value with the biggest absolute weighted cycle-time change."""
    if primary.empty or comparison.empty:
        return None
    joined = primary[[column, "cycle", "volume"]].merge(
        comparison[[column, "cycle"]], on=column, how="inner", suffixes=("", "_comparison")
    )
    if joined.empty:
        return None
    joined["change"] = joined["cycle"] - joined["cycle_comparison"]
    ordered = joined.reindex(joined["change"].abs().sort_values(ascending=False).index)
    row = ordered.iloc[0].to_dict()
    row["comparison_cycle"] = row.pop("cycle_comparison")
    return row


def change_severity(change: float | None, scale: float | None = None) -> Severity:
    """Severity from an absolute change, scaled by the metric's own magnitude."""
    magnitude = abs(change or 0.0)
    reference = abs(scale or 0.0) or 1.0
    relative = magnitude / reference
    if relative >= 0.25 or magnitude >= 10.0:
        return Severity.HIGH
    if relative >= 0.12 or magnitude >= 4.0:
        return Severity.MEDIUM
    if relative >= 0.05 or magnitude >= 1.0:
        return Severity.LOW
    return Severity.INFO


# ---------------------------------------------------------------------------
# Order-grain helpers (orders.csv)
# ---------------------------------------------------------------------------


def order_components(frame: pd.DataFrame | None) -> dict[str, float | None]:
    """Fulfilment, transit, end-to-end and on-time rate for a set of orders."""
    if frame is None or frame.empty:
        return {
            "handling": None,
            "transit": None,
            "total": None,
            "on_time_rate": None,
            "transit_share": None,
        }
    handling = round_or_none(frame["fulfillment_time_hours"].mean())
    transit = round_or_none(frame["transit_time_hours"].mean())
    total = round_or_none(frame["actual_time_hours"].mean())
    counted = len(frame)
    on_time = int((frame["actual_time_hours"] <= frame["promised_time_hours"]).sum())
    return {
        "handling": handling,
        "transit": transit,
        "total": total,
        "on_time_rate": percentage_of_total(on_time, counted),
        "transit_share": contribution(transit, total),
    }


def order_table(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Order-level cycle time breakdown for each value of ``column``."""
    columns = [
        column,
        "orders",
        "handling",
        "transit",
        "total",
        "on_time_rate",
        "on_time_delivered",
    ]
    if frame is None or frame.empty or column not in frame.columns:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(column, dropna=True):
        components = order_components(group)
        rows.append(
            {
                column: key,
                "orders": len(group),
                "on_time_delivered": int(
                    (group["actual_time_hours"] <= group["promised_time_hours"]).sum()
                ),
                **components,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def largest_order_movement(
    primary: pd.DataFrame, comparison: pd.DataFrame, column: str
) -> dict[str, Any] | None:
    """The ``column`` value with the biggest absolute fulfilment-time change."""
    if primary.empty or comparison.empty:
        return None
    joined = primary[[column, "handling", "orders"]].merge(
        comparison[[column, "handling"]],
        on=column,
        how="inner",
        suffixes=("", "_comparison"),
    )
    if joined.empty:
        return None
    joined["change"] = joined["handling"] - joined["handling_comparison"]
    ordered = joined.reindex(joined["change"].abs().sort_values(ascending=False).index)
    row = ordered.iloc[0].to_dict()
    row["comparison_handling"] = row.pop("handling_comparison")
    return row


def _order_component_findings(
    current: dict[str, float | None],
    baseline: dict[str, float | None] | None,
    period_label: str,
) -> list[Finding]:
    """Attribute an end-to-end change to either our handling or the carrier."""
    if baseline is None:
        return []
    handling_delta = absolute_change(current["handling"], baseline["handling"])
    transit_delta = absolute_change(current["transit"], baseline["transit"])
    if handling_delta is None and transit_delta is None:
        return []

    dominant = "transit"
    dominant_delta = transit_delta
    opposite_delta = handling_delta
    # A movement only transfers blame between the two legs when both moved.
    if (
        handling_delta is not None
        and transit_delta is not None
        and abs(handling_delta) >= abs(transit_delta)
    ):
        dominant = "handling"
        dominant_delta = handling_delta
        opposite_delta = transit_delta

    if not dominant_delta or abs(dominant_delta) < 0.05:
        return []

    owner = (
        "our own fulfilment operation" if dominant == "handling" else "the carrier's transit leg"
    )
    direction = "increased" if dominant_delta > 0 else "decreased"
    return [
        build_finding(
            title=f"End-to-end order time moved with {dominant} time",
            detail=(
                f"{dominant.capitalize()} time {direction} by {abs(dominant_delta):.2f} hours "
                f"between the two periods, against {opposite_delta:+.2f} hours for "
                f"{'transit' if dominant == 'handling' else 'fulfilment'} time. The movement sits "
                f"with {owner}."
            ),
            severity=change_severity(dominant_delta, scale=current["total"] or 1.0),
            metric=f"{dominant}_time",
            period=period_label,
            statistic=(
                f"{dominant}_time delta = {dominant_delta:+.2f} hours; "
                f"opposite component = {opposite_delta:+.2f} hours"
            ),
        )
    ]
