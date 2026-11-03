"""Throughput analysis: how much work completed, and how it changed.

Throughput is defined as ``completed_operations`` per day. Rates are normalised
by the number of distinct days actually present in the data, never by the number
of rows, so daily aggregates, monthly rollups and partial periods remain
comparable.
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
from app.models.enums import AnalysisType, DatasetId, Severity, Unit
from app.schemas.analysis import EvidenceItem, Finding
from app.utils.calculations import round_or_none, safe_float
from app.utils.calculations import throughput as rate_per_day
from app.utils.errors import InsufficientDataError

#: Requested dimension -> operations.csv column.
DIMENSION_COLUMNS: dict[str, str] = {
    "department": "department",
    "process": "process",
    "operation_type": "operation_type",
    "priority": "priority",
    "status": "status",
    "vendor": "vendor_id",
}


class ThroughputAnalyzer(BaseAnalyzer):
    """Completed operations per day, overall and by dimension."""

    name = "throughput_analyzer"
    supported_types = frozenset({AnalysisType.THROUGHPUT_ANALYSIS})
    required_datasets = (DatasetId.OPERATIONS,)

    def analyze(self, context: AnalysisContext) -> AnalysisOutcome:
        primary = context.primary_frame()
        if primary.empty:
            raise InsufficientDataError(
                f"no operational records found for {context.plan.period or 'the requested period'}"
            )

        outcome = AnalysisOutcome()
        period_label = context.plan.period or "all periods"
        comparison = context.comparison_frame()

        days = distinct_days(primary)
        total_completed = float(primary["completed_operations"].sum())
        overall_rate = rate_per_day(total_completed, days)
        comparison_days = distinct_days(comparison) if comparison is not None else 0
        comparison_total = (
            float(comparison["completed_operations"].sum()) if comparison is not None else 0.0
        )
        comparison_rate = (
            rate_per_day(comparison_total, comparison_days) if comparison_days else None
        )

        outcome.kpis = [
            build_kpi(
                name="Completed operations",
                value=total_completed,
                unit=Unit.OPERATIONS,
                source_dataset=DatasetId.OPERATIONS,
                period=period_label,
                calculation=(
                    "sum of completed_operations over all matching operations.csv records"
                ),
            ),
            build_kpi(
                name="Throughput per day",
                value=overall_rate,
                unit=Unit.OPERATIONS,
                source_dataset=DatasetId.OPERATIONS,
                period=period_label,
                calculation=f"sum(completed_operations) / {days} distinct days present in the data",
            ),
        ]

        outcome.evidence = [
            build_evidence(
                metric="Throughput per day",
                value=overall_rate,
                comparison_value=comparison_rate,
                unit=Unit.OPERATIONS,
                source_dataset=DatasetId.OPERATIONS,
                period=period_label,
                comparison_period=context.plan.comparison_period,
                calculation_description=(
                    "Completed operations divided by the number of distinct calendar days "
                    "present in operations.csv for the period"
                ),
                sample_size=days,
            )
        ]

        column = self._dimension_column(context)
        if column:
            outcome.evidence.extend(self._dimension_evidence(primary, comparison, column))
            outcome.findings.extend(
                self._dimension_findings(primary, comparison, column, period_label)
            )

        if days < 7:
            outcome.extend_caveats(
                f"only {days} day(s) of data cover this period, so the daily rate is sensitive "
                "to weekends and month-end activity"
            )
        outcome.row_count_examined = len(primary)
        return outcome

    # -- internals -----------------------------------------------------
    @staticmethod
    def _dimension_column(context: AnalysisContext) -> str | None:
        for dimension in context.plan.dimensions:
            column = DIMENSION_COLUMNS.get(dimension.value)
            if column:
                return column
        return None

    def _dimension_evidence(
        self, primary: pd.DataFrame, comparison: pd.DataFrame | None, column: str
    ) -> list[EvidenceItem]:
        ranking = rate_table(primary, column)
        if ranking.empty:
            return []
        leader = ranking.iloc[0]
        items = [
            build_evidence(
                metric=f"Throughput per day by {column}",
                value=safe_float(leader["rate"]),
                unit=Unit.OPERATIONS,
                source_dataset=DatasetId.OPERATIONS,
                dimension=str(leader[column]),
                calculation_description=(
                    f"Daily completed operations per {column} value, ranked highest first; "
                    "the leading value is reported"
                ),
                sample_size=len(ranking),
            )
        ]
        if comparison is not None and not comparison.empty:
            movement = largest_movement(
                rate_table(primary, column), rate_table(comparison, column), column
            )
            if movement is not None:
                items.append(
                    build_evidence(
                        metric=f"Largest throughput movement by {column}",
                        value=round_or_none(movement["rate"]),
                        comparison_value=round_or_none(movement["comparison_rate"]),
                        unit=Unit.OPERATIONS,
                        source_dataset=DatasetId.OPERATIONS,
                        dimension=str(movement[column]),
                        calculation_description=(
                            f"Per-day completed operations for each {column} value in both "
                            "periods, ranked by absolute difference between periods"
                        ),
                        sample_size=int(movement.get("days") or 0),
                        severity=movement_severity(movement["change"]),
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
        ranking = rate_table(primary, column)
        if ranking.empty:
            return []
        leader = ranking.iloc[0]
        findings = [
            build_finding(
                title=f"Highest throughput: {leader[column]}",
                detail=(
                    f"{leader[column]} completed {safe_float(leader['rate']):,.1f} operations per "
                    f"day on average over {period_label}, the highest of {len(ranking)} "
                    f"{column} values."
                ),
                severity=Severity.INFO,
                metric="throughput",
                dimension=str(leader[column]),
                period=period_label,
                statistic=(
                    f"mean of completed_operations per day for {column} = "
                    f"{safe_float(leader['rate']):,.2f}"
                ),
            )
        ]
        if comparison is not None and not comparison.empty:
            movement = largest_movement(
                rate_table(primary, column), rate_table(comparison, column), column
            )
            if movement is not None and movement["change"] is not None:
                direction = "increased" if movement["change"] > 0 else "decreased"
                findings.append(
                    build_finding(
                        title=f"Largest throughput movement: {movement[column]}",
                        detail=(
                            f"{movement[column]} daily throughput {direction} by "
                            f"{abs(movement['change']):,.1f} operations per day between the two "
                            "periods, the largest absolute movement of any "
                            f"{column} value."
                        ),
                        severity=movement_severity(movement["change"]),
                        metric="throughput",
                        dimension=str(movement[column]),
                        statistic=(
                            f"per-day rate difference = {movement['change']:+.2f} operations/day"
                        ),
                    )
                )
        return findings


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def distinct_days(frame: pd.DataFrame | None) -> int:
    """Number of distinct calendar days in a frame."""
    if frame is None or frame.empty or "date" not in frame.columns:
        return 0
    return int(frame["date"].nunique())


def rate_table(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Per-day throughput for each value of ``column``."""
    if frame.empty or column not in frame.columns:
        return pd.DataFrame(columns=[column, "rate", "days", "completed"])
    grouped = (
        frame.groupby(column, dropna=True)
        .agg(completed=("completed_operations", "sum"), days=("date", "nunique"))
        .reset_index()
    )
    grouped["rate"] = [
        rate_per_day(value, days)
        for value, days in zip(grouped["completed"], grouped["days"], strict=True)
    ]
    return grouped.sort_values("rate", ascending=False).reset_index(drop=True)


def largest_movement(
    primary: pd.DataFrame, comparison: pd.DataFrame, column: str
) -> dict[str, Any] | None:
    """The ``column`` value with the biggest absolute per-day rate change."""
    if primary.empty or comparison.empty:
        return None
    joined = primary[[column, "rate", "days"]].merge(
        comparison[[column, "rate"]], on=column, how="inner", suffixes=("", "_comparison")
    )
    if joined.empty:
        return None
    joined["change"] = joined["rate"] - joined["rate_comparison"]
    ordered = joined.reindex(joined["change"].abs().sort_values(ascending=False).index)
    row = ordered.iloc[0].to_dict()
    row["comparison_rate"] = row.pop("rate_comparison")
    return row


def movement_severity(change: float | None) -> Severity:
    """Map an absolute daily-rate change to a severity level."""
    magnitude = abs(safe_float(change) or 0.0)
    if magnitude >= 8.0:
        return Severity.HIGH
    if magnitude >= 3.0:
        return Severity.MEDIUM
    if magnitude >= 1.0:
        return Severity.LOW
    return Severity.INFO
