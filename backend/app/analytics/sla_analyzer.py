"""SLA analysis: compliance, breach load and who is missing the target.

SLA compliance is always recomputed from case counts rather than read from the
stored percentage, so a discrepancy between the two is surfaced rather than
silently trusted:

    compliance  = sla_met_cases / total_cases * 100
    breach rate = sla_breached_cases / total_cases * 100

Where the source dataset already carries a percentage, the difference from the
recomputed value is emitted as evidence about data quality.
"""

from __future__ import annotations

from typing import TypedDict

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
from app.utils.calculations import absolute_change, round_or_none, safe_float
from app.utils.errors import InsufficientDataError
from app.utils.periods import period_sort_key

DIMENSION_COLUMNS: dict[str, str] = {
    "department": "department",
    "process": "process",
}

#: Compliance below this is treated as a material failure.
SLA_ALERT_THRESHOLD = 90.0
#: Compliance below this is treated as critical.
SLA_CRITICAL_THRESHOLD = 75.0


class SLAAnalyzer(BaseAnalyzer):
    """SLA compliance and breach exposure from the monthly rollup."""

    name = "sla_analyzer"
    supported_types = frozenset({AnalysisType.SLA_ANALYSIS})
    required_datasets = (DatasetId.SLA_METRICS, DatasetId.DEPARTMENTS)

    def analyze(self, context: AnalysisContext) -> AnalysisOutcome:
        primary = context.frame(DatasetId.SLA_METRICS)
        if primary.empty:
            raise InsufficientDataError(
                f"no SLA records found for {context.plan.period or 'the requested period'}"
            )

        outcome = AnalysisOutcome()
        period_label = context.plan.period or "all periods"
        comparison = context.frame(DatasetId.SLA_METRICS, window="comparison")

        current = compliance_summary(primary)
        baseline = compliance_summary(comparison) if comparison is not None else None
        names = context.repository.department_names

        outcome.kpis = [
            build_kpi(
                name="SLA compliance",
                value=current["compliance"],
                unit=Unit.PERCENT,
                source_dataset=DatasetId.SLA_METRICS,
                period=period_label,
                calculation=(
                    "sla_met_cases / total_cases * 100 across all matching sla_metrics rows"
                ),
            ),
            build_kpi(
                name="SLA breach rate",
                value=current["breach_rate"],
                unit=Unit.PERCENT,
                source_dataset=DatasetId.SLA_METRICS,
                period=period_label,
                calculation=(
                    "sla_breached_cases / total_cases * 100 across all matching sla_metrics rows"
                ),
            ),
            build_kpi(
                name="Breached cases",
                value=current["breached"],
                unit=Unit.COUNT,
                source_dataset=DatasetId.SLA_METRICS,
                period=period_label,
                calculation="sum of sla_breached_cases across all matching sla_metrics rows",
            ),
        ]

        outcome.evidence = [
            build_evidence(
                metric="SLA compliance",
                value=current["compliance"],
                comparison_value=baseline["compliance"] if baseline else None,
                unit=Unit.PERCENT,
                source_dataset=DatasetId.SLA_METRICS,
                period=period_label,
                comparison_period=context.plan.comparison_period,
                calculation_description=(
                    "sla_met_cases / total_cases * 100, aggregated over sla_metrics.csv rows "
                    "matching the period"
                ),
                sample_size=current["total"],
                severity=compliance_severity(current["compliance"]),
            ),
            build_evidence(
                metric="SLA breach rate",
                value=current["breach_rate"],
                comparison_value=baseline["breach_rate"] if baseline else None,
                unit=Unit.PERCENT,
                source_dataset=DatasetId.SLA_METRICS,
                period=period_label,
                comparison_period=context.plan.comparison_period,
                calculation_description=(
                    "sla_breached_cases / total_cases * 100, aggregated over sla_metrics.csv rows "
                    "matching the period"
                ),
                sample_size=current["total"],
                severity=compliance_severity(current["compliance"]),
            ),
        ]

        if baseline is not None and baseline["compliance"] is not None:
            delta = absolute_change(current["compliance"], baseline["compliance"])
            outcome.findings.append(
                build_finding(
                    title="SLA compliance moved between periods",
                    detail=(
                        f"Compliance was {baseline['compliance']:.1f}% and is now "
                        f"{current['compliance']:.1f}%, a change of {delta:+.1f} "
                        f"percentage points across {current['total']:,} cases."
                    ),
                    severity=compliance_severity(current["compliance"]),
                    metric="sla_compliance",
                    period=period_label,
                    statistic=(
                        f"sla_met_cases/total_cases = {current['met']:,}/{current['total']:,}"
                    ),
                )
            )

        column = self._dimension_column(context)
        offenders = compliance_table(primary, column) if column else pd.DataFrame()
        if column and not offenders.empty:
            worst = offenders.sort_values("compliance").iloc[0]
            outcome.evidence.append(
                build_evidence(
                    metric=f"Lowest SLA compliance by {column}",
                    value=round_or_none(worst["compliance"]),
                    unit=Unit.PERCENT,
                    source_dataset=DatasetId.SLA_METRICS,
                    dimension=_label(worst, column, names),
                    calculation_description=(
                        f"Recomputed sla_met_cases / total_cases per {column} value, ranked "
                        "lowest first; the weakest is reported"
                    ),
                    sample_size=int(worst["total"]),
                    severity=compliance_severity(worst["compliance"]),
                )
            )
            outcome.findings.extend(
                self._dimension_findings(offenders, column, names, period_label, context.top_n)
            )

        if comparison is not None and not comparison.empty:
            deteriorating = compliance_table(comparison, column or "process")
            outcome.evidence.extend(
                _movement_evidence(
                    offenders if column else pd.DataFrame(),
                    deteriorating if column else pd.DataFrame(),
                    column or "process",
                    names,
                )
            )

        discrepancy = self._stored_percentage_discrepancy(primary)
        if discrepancy:
            outcome.extend_caveats(
                f"{discrepancy} sla_metrics row(s) store a sla_compliance_percentage that differs "
                "from the value recomputed from sla_met_cases and total_cases by more than "
                "0.05 percentage points; the recomputed value is reported"
            )
        if current["total"] < 30:
            outcome.extend_caveats(
                f"only {current['total']} SLA cases back this result, so a single case moves "
                "compliance by more than three percentage points"
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

    def _dimension_findings(
        self,
        offenders: pd.DataFrame,
        column: str,
        names: dict[str, str],
        period_label: str,
        top_n: int,
    ) -> list[Finding]:
        failing = offenders.loc[offenders["compliance"] < SLA_ALERT_THRESHOLD]
        if failing.empty:
            return [
                build_finding(
                    title="All measured processes are meeting their SLA",
                    detail=(
                        f"Every {column} value in {period_label} is above the "
                        f"{SLA_ALERT_THRESHOLD:.0f}% compliance threshold."
                    ),
                    severity=Severity.INFO,
                    metric="sla_compliance",
                    period=period_label,
                )
            ]
        findings = [
            build_finding(
                title=f"SLA failure: {worst[column]}",
                detail=(
                    f"{_label(worst, column, names)} met its SLA in {worst['compliance']:.1f}% of "
                    f"{worst['total']:,.0f} cases against a "
                    f"{worst['target']:.1f} hour target, breaching "
                    f"{worst['breached']:,.0f} cases. Average cycle time was "
                    f"{worst['average_cycle_time_hours']:.2f} hours."
                ),
                severity=compliance_severity(worst["compliance"]),
                metric="sla_compliance",
                dimension=_label(worst, column, names),
                period=period_label,
                statistic=(f"{int(worst['met']):,}/{int(worst['total']):,} cases met the target"),
            )
            for _, worst in failing.sort_values("compliance").head(top_n).iterrows()
        ]
        return findings

    @staticmethod
    def _stored_percentage_discrepancy(frame: pd.DataFrame) -> int:
        """Rows where the stored percentage disagrees with the case counts."""
        valid = frame["total_cases"] > 0
        derived = (frame["sla_met_cases"] / frame["total_cases"] * 100).where(valid)
        mask = (
            derived.notna()
            & frame["sla_compliance_percentage"].notna()
            & (derived - frame["sla_compliance_percentage"]).abs().gt(0.05)
        )
        return int(mask.sum())


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class ComplianceSummary(TypedDict):
    """Case-weighted SLA counts and the percentages derived from them."""

    met: int
    breached: int
    total: int
    compliance: float | None
    breach_rate: float | None


def compliance_summary(frame: pd.DataFrame | None) -> ComplianceSummary:
    """Case-weighted compliance and breach rate for a set of SLA rows."""
    if frame is None or frame.empty:
        return ComplianceSummary(met=0, breached=0, total=0, compliance=None, breach_rate=None)
    met = int(frame["sla_met_cases"].fillna(0).sum())
    breached = int(frame["sla_breached_cases"].fillna(0).sum())
    total = met + breached
    if total <= 0:
        return ComplianceSummary(
            met=met,
            breached=breached,
            total=0,
            compliance=None,
            breach_rate=None,
        )
    return ComplianceSummary(
        met=met,
        breached=breached,
        total=total,
        compliance=round_or_none(met / total * 100.0),
        breach_rate=round_or_none(breached / total * 100.0),
    )


def compliance_table(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Recomputed SLA compliance for each value of ``column``."""
    if frame is None or frame.empty or column not in frame.columns:
        return pd.DataFrame(
            columns=[
                column,
                "met",
                "breached",
                "total",
                "compliance",
                "average_cycle_time_hours",
                "sla_target_hours",
            ]
        )
    grouped = (
        frame.groupby(column, dropna=True)
        .agg(
            met=("sla_met_cases", "sum"),
            breached=("sla_breached_cases", "sum"),
            total=("total_cases", "sum"),
            weighted_cycle=("average_cycle_time_hours", lambda s: float(s.sum())),
            target=("sla_target_hours", "first"),
        )
        .reset_index()
    )
    grouped["compliance"] = [
        round_or_none(met / total * 100.0) if total else None
        for met, total in zip(grouped["met"], grouped["total"], strict=True)
    ]
    grouped["average_cycle_time_hours"] = grouped["weighted_cycle"].round(2)
    return grouped.drop(columns=["weighted_cycle"])


def compliance_by_period(frame: pd.DataFrame) -> pd.DataFrame:
    """Month-by-month compliance for the whole dataset (dashboard use)."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["period", "met", "total", "compliance", "breached"])
    rows: list[dict[str, object]] = []
    for period, group in frame.groupby("period", dropna=True):
        summary = compliance_summary(group)
        rows.append(
            {
                "period": period,
                "met": summary["met"],
                "breached": summary["breached"],
                "total": summary["total"],
                "compliance": summary["compliance"],
            }
        )
    return pd.DataFrame(rows).sort_values("period", key=lambda s: s.map(period_sort_key))


def compliance_severity(compliance: float | None) -> Severity:
    value = safe_float(compliance)
    if value is None:
        return Severity.INFO
    if value < SLA_CRITICAL_THRESHOLD:
        return Severity.CRITICAL
    if value < SLA_ALERT_THRESHOLD:
        return Severity.HIGH
    if value < 97.0:
        return Severity.MEDIUM
    return Severity.INFO


def _movement_evidence(
    current: pd.DataFrame, comparison: pd.DataFrame, column: str, names: dict[str, str]
) -> list[EvidenceItem]:
    """Evidence for the segment whose compliance fell most between periods."""
    if current.empty or comparison.empty or column not in current.columns:
        return []
    joined = current[[column, "compliance", "total"]].merge(
        comparison[[column, "compliance"]], on=column, how="inner", suffixes=("", "_comparison")
    )
    if joined.empty:
        return []
    joined["change"] = joined["compliance"] - joined["compliance_comparison"]
    ordered = joined.reindex(joined["change"].abs().sort_values(ascending=False).index)
    row = ordered.iloc[0].to_dict()
    previous = row.pop("compliance_comparison")
    return [
        build_evidence(
            metric=f"Largest SLA compliance movement by {column}",
            value=round_or_none(row["compliance"]),
            comparison_value=round_or_none(previous),
            unit=Unit.PERCENT,
            source_dataset=DatasetId.SLA_METRICS,
            dimension=_label(row, column, names),
            calculation_description=(
                f"Recomputed sla_met_cases / total_cases for each {column} value in both "
                "periods, ranked by absolute difference between periods"
            ),
            sample_size=int(row.get("total") or 0),
            severity=compliance_severity(row["compliance"]),
        )
    ]


def _label(row: dict, column: str, names: dict[str, str]) -> str:
    value = str(row[column])
    return names.get(value, value)
