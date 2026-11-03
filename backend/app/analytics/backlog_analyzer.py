"""Backlog analysis: open work, and whether it is growing.

Backlog is a *stock* measure, so it is read as a level at the end of a period
and a change between periods. Two properties matter operationally and are made
explicit in the evidence:

* the closing backlog of the last day in the period is the reported level, not
  the average, because the average hides a growing queue;
* the change is compared against the previous period's closing backlog, not its
  first value, so a steadily growing queue is not mistaken for seasonal noise.
"""

from __future__ import annotations

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
from app.utils.calculations import (
    absolute_change,
    linear_trend,
    percentage_change,
    round_or_none,
    safe_float,
)
from app.utils.errors import InsufficientDataError
from app.utils.periods import days_in_month, period_sort_key

DIMENSION_COLUMNS: dict[str, str] = {
    "department": "department",
    "process": "process",
    "operation_type": "operation_type",
    "priority": "priority",
    "vendor": "vendor_id",
}


class BacklogAnalyzer(BaseAnalyzer):
    """Closing backlog levels, period-over-period change and accumulation trend."""

    name = "backlog_analyzer"
    supported_types = frozenset({AnalysisType.BACKLOG_ANALYSIS})
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

        closing = closing_backlog(primary)
        if closing is None:
            raise InsufficientDataError(
                f"no measurable backlog in {context.plan.period or 'the requested period'}"
            )
        comparison_closing = closing_backlog(comparison) if comparison is not None else None
        change = absolute_change(closing, comparison_closing)
        max_backlog, _max_date = max_backlog_record(primary)

        outcome.kpis = [
            build_kpi(
                name="Closing backlog",
                value=closing,
                unit=Unit.ITEMS,
                source_dataset=DatasetId.OPERATIONS,
                period=period_label,
                calculation=(
                    "backlog_open_items summed across all process records on the final day "
                    "present in the period"
                ),
            ),
            build_kpi(
                name="Backlog change",
                value=change,
                unit=Unit.ITEMS,
                source_dataset=DatasetId.OPERATIONS,
                period=period_label,
                calculation=(
                    "closing backlog of this period minus closing backlog of the comparison period"
                ),
            ),
            build_kpi(
                name="Peak backlog",
                value=max_backlog,
                unit=Unit.ITEMS,
                source_dataset=DatasetId.OPERATIONS,
                period=period_label,
                calculation="highest daily backlog_open_items total within the period",
            ),
        ]

        outcome.evidence = [
            build_evidence(
                metric="Closing backlog",
                value=closing,
                comparison_value=comparison_closing,
                unit=Unit.ITEMS,
                source_dataset=DatasetId.OPERATIONS,
                period=period_label,
                comparison_period=context.plan.comparison_period,
                calculation_description=(
                    "Sum of backlog_open_items over all operations.csv records on the last "
                    "calendar day of the period"
                ),
                severity=backlog_severity(change),
            ),
            build_evidence(
                metric="Backlog change",
                value=change,
                unit=Unit.ITEMS,
                source_dataset=DatasetId.OPERATIONS,
                period=period_label,
                comparison_period=context.plan.comparison_period,
                calculation_description=(
                    "Closing backlog (this period) minus closing backlog (comparison period)"
                ),
                severity=backlog_severity(change),
            ),
            build_evidence(
                metric="Peak backlog within period",
                value=max_backlog,
                unit=Unit.ITEMS,
                source_dataset=DatasetId.OPERATIONS,
                period=period_label,
                calculation_description=(
                    "Maximum of the daily backlog_open_items total observed in the period"
                ),
            ),
        ]

        column = self._dimension_column(context)
        if column:
            outcome.evidence.extend(self._dimension_evidence(primary, column))
            outcome.findings.extend(self._dimension_findings(primary, column, period_label))

        outcome.findings.extend(self._accumulation_findings(primary, closing, period_label))
        outcome.extend_caveats(*self._coverage_caveats(primary, period_label))
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

    def _dimension_evidence(self, primary: pd.DataFrame, column: str) -> list[EvidenceItem]:
        table = backlog_table(primary, column)
        if table.empty:
            return []
        leader = table.sort_values("closing", ascending=False).iloc[0]
        return [
            build_evidence(
                metric=f"Closing backlog by {column}",
                value=round_or_none(leader["closing"], 1),
                unit=Unit.ITEMS,
                source_dataset=DatasetId.OPERATIONS,
                dimension=str(leader[column]),
                calculation_description=(
                    f"Closing backlog_open_items per {column} value, ranked highest first; "
                    "the largest holder is reported"
                ),
                sample_size=len(table),
                severity=backlog_severity(leader["closing"] - leader["opening"], reference=100.0),
            )
        ]

    def _dimension_findings(
        self, primary: pd.DataFrame, column: str, period_label: str
    ) -> list[Finding]:
        table = backlog_table(primary, column)
        if table.empty:
            return []
        ranked = table.sort_values("closing", ascending=False).iloc[0]
        closing_value = safe_float(ranked["closing"])
        opening_value = safe_float(ranked["opening"])
        growth = (
            closing_value - opening_value
            if closing_value is not None and opening_value is not None
            else None
        )
        findings = [
            build_finding(
                title=f"Largest backlog holder: {ranked[column]}",
                detail=(
                    f"{ranked[column]} closed {ranked['closing']:,.1f} open items at the end of "
                    f"{period_label}, against {ranked['opening']:,.1f} at the start."
                ),
                severity=backlog_severity(growth, reference=100.0),
                metric="backlog",
                dimension=str(ranked[column]),
                period=period_label,
                statistic=(
                    f"closing minus opening backlog = {growth:+,.1f} items"
                    if growth is not None
                    else "opening backlog unavailable"
                ),
            )
        ]
        growing = table.assign(growth=table["closing"] - table["opening"])
        growing = growing[growing["growth"] > 0.5].sort_values("growth", ascending=False)
        if not growing.empty:
            top = growing.iloc[0]
            findings.append(
                build_finding(
                    title=f"Backlog accumulating in {top[column]}",
                    detail=(
                        f"{top[column]} added {top['growth']:,.1f} net items of backlog over "
                        f"{period_label}, the fastest accumulation of any {column} value."
                    ),
                    severity=backlog_severity(top["growth"], reference=100.0),
                    metric="backlog_change",
                    dimension=str(top[column]),
                    period=period_label,
                    statistic=f"net backlog change = {top['growth']:+,.1f} items",
                )
            )
        return findings

    @staticmethod
    def _accumulation_findings(
        primary: pd.DataFrame, closing: float, period_label: str
    ) -> list[Finding]:
        """Detect a monotonically accumulating backlog across the period."""
        series = daily_backlog_series(primary)
        if len(series) < 4:
            return []
        trend = linear_trend(series["backlog"].tolist())
        if trend is None:
            return []
        slope = safe_float(trend.get("slope"))
        if slope is None or slope <= 0.5:
            return []
        r_squared = trend.get("r_squared")
        confidence = f" (R^2 = {r_squared:.2f})" if isinstance(r_squared, float) else ""
        return [
            build_finding(
                title="Backlog is accumulating steadily",
                detail=(
                    f"Total backlog rises by an average of {slope:,.1f} items per day across "
                    f"{period_label}{confidence}, reaching {closing:,.1f} open items. This is a "
                    "capacity shortfall rather than a one-off spike."
                ),
                severity=backlog_severity(slope * 30.0, reference=100.0),
                metric="backlog",
                period=period_label,
                is_statistical=True,
                statistic=(
                    f"least-squares slope = {slope:+.2f} items/day"
                    + (f", R^2 = {r_squared:.2f}" if isinstance(r_squared, float) else "")
                ),
            )
        ]

    @staticmethod
    def _coverage_caveats(primary: pd.DataFrame, period_label: str) -> list[str]:
        caveats: list[str] = []
        months = sorted({str(value) for value in primary["period"].unique()}, key=period_sort_key)
        if months and period_label:
            expected = days_in_month(months[-1])
            actual_days = int(primary.loc[primary["period"] == months[-1], "date"].nunique())
            if expected and actual_days < expected:
                caveats.append(
                    f"{months[-1]} is incomplete: {actual_days} of {expected} days are present, "
                    "so period totals for the final month understate a full month"
                )
        return caveats


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def daily_backlog_totals(frame: pd.DataFrame) -> pd.DataFrame:
    """Total backlog_open_items per calendar day."""
    if frame is None or frame.empty or "backlog_open_items" not in frame.columns:
        return pd.DataFrame(columns=["date", "backlog"])
    totals = frame.groupby("date", as_index=False)["backlog_open_items"].sum().sort_values("date")
    return totals.rename(columns={"backlog_open_items": "backlog"})


def daily_backlog_series(frame: pd.DataFrame) -> pd.DataFrame:
    return daily_backlog_totals(frame)


def closing_backlog(frame: pd.DataFrame | None) -> float | None:
    """Backlog level on the final day present in the frame."""
    if frame is None or frame.empty:
        return None
    totals = daily_backlog_totals(frame)
    if totals.empty:
        return None
    return round_or_none(totals.iloc[-1]["backlog"], 1)


def max_backlog_record(frame: pd.DataFrame | None) -> tuple[float | None, object]:
    """Highest daily backlog total, with the date it occurred."""
    if frame is None or frame.empty:
        return None, None
    totals = daily_backlog_totals(frame)
    if totals.empty:
        return None, None
    row = totals.loc[totals["backlog"].idxmax()]
    return round_or_none(row["backlog"], 1), row["date"]


def backlog_table(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Opening and closing backlog for each value of ``column``."""
    if frame is None or frame.empty or column not in frame.columns:
        return pd.DataFrame(columns=[column, "opening", "closing", "peak", "days"])
    ordered = frame.sort_values("date")
    first_date = ordered["date"].min()
    last_date = ordered["date"].max()
    opening = (
        ordered.loc[ordered["date"] == first_date]
        .groupby(column, dropna=True)["backlog_open_items"]
        .sum()
    )
    closing = (
        ordered.loc[ordered["date"] == last_date]
        .groupby(column, dropna=True)["backlog_open_items"]
        .sum()
    )
    peak = ordered.groupby(column, dropna=True)["backlog_open_items"].max()
    days = ordered.groupby(column, dropna=True)["date"].nunique()
    table = pd.DataFrame(
        {
            "opening": opening,
            "closing": closing,
            "peak": peak,
            "days": days,
        }
    ).reset_index()
    table.columns = [column, "opening", "closing", "peak", "days"]
    return table.fillna({"opening": 0.0, "closing": 0.0, "peak": 0.0})


def backlog_severity(change: float | None, reference: float = 50.0) -> Severity:
    """Severity from a net backlog movement."""
    magnitude = abs(safe_float(change) or 0.0)
    if magnitude >= reference * 2:
        return Severity.HIGH
    if magnitude >= reference:
        return Severity.MEDIUM
    if magnitude >= reference * 0.25:
        return Severity.LOW
    return Severity.INFO


def backlog_percentage_change(current: float | None, previous: float | None) -> float | None:
    return percentage_change(current, previous)
