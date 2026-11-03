"""Department analysis: a comparable operational scorecard per department.

Departments own very different work, so comparing raw cycle times across them is
misleading: a 24-hour SLA target and a 4-hour SLA target produce comparable
percentages but not comparable hours. This analyzer therefore reports each
department's performance **relative to its own SLA target**, using the monthly
SLA rollup, and combines that with operational volume and backlog pressure.

The scorecard fields are:

* ``sla_compliance``   - recomputed met / total cases;
* ``cycle_time_ratio`` - average cycle time divided by the process SLA target,
  where 1.0 means exactly on target;
* ``throughput``       - completed operations per day;
* ``backlog_days``     - closing backlog expressed as days of that department's
  own recent throughput.
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
from app.analytics.sla_analyzer import compliance_severity, compliance_summary
from app.models.enums import AnalysisType, DatasetId, Severity, Unit
from app.schemas.analysis import EvidenceItem, Finding
from app.utils.calculations import round_or_none, safe_float
from app.utils.errors import InsufficientDataError

#: Cycle time ratio at or above this means the department is past its own target.
OVER_TARGET_RATIO = 1.0
#: Backlog expressed in days of throughput at or above this is material.
MATERIAL_BACKLOG_DAYS = 2.0


class DepartmentAnalyzer(BaseAnalyzer):
    """Per-department scorecard with target-relative comparison."""

    name = "department_analyzer"
    supported_types = frozenset({AnalysisType.DEPARTMENT_ANALYSIS})
    required_datasets = (
        DatasetId.OPERATIONS,
        DatasetId.SLA_METRICS,
        DatasetId.DEPARTMENTS,
    )

    def analyze(self, context: AnalysisContext) -> AnalysisOutcome:
        operations = context.primary_frame()
        if operations.empty:
            raise InsufficientDataError(
                f"no operational records found for {context.plan.period or 'the requested period'}"
            )

        outcome = AnalysisOutcome()
        period_label = context.plan.period or "all periods"
        sla_frame = context.frame(DatasetId.SLA_METRICS)
        if not sla_frame.empty and context.primary is not None:
            months = set(context.primary.months)
            sla_frame = sla_frame.loc[sla_frame["period"].isin(months)]

        names = context.repository.department_names
        headcount = _headcount_by_department(context)
        table = department_scorecard(operations, sla_frame, headcount)
        if table.empty:
            raise InsufficientDataError("no department-level records were found")

        table["label"] = table["department"].map(lambda value: names.get(str(value), str(value)))

        outcome.kpis = [
            build_kpi(
                name="Departments past their own cycle-time target",
                value=float(int((table["cycle_time_ratio"] >= OVER_TARGET_RATIO).sum())),
                unit=Unit.COUNT,
                source_dataset=DatasetId.SLA_METRICS,
                period=period_label,
                calculation=(
                    "count of departments whose average cycle time is at or above the average "
                    "sla_target_hours of its processes"
                ),
            ),
            build_kpi(
                name="Weakest SLA compliance",
                value=round_or_none(table["sla_compliance"].min()),
                unit=Unit.PERCENT,
                source_dataset=DatasetId.SLA_METRICS,
                period=period_label,
                dimension=str(
                    table.loc[table["sla_compliance"].idxmin(), "label"]
                    if table["sla_compliance"].notna().any()
                    else "n/a"
                ),
                calculation=(
                    "minimum of recomputed per-department sla_met_cases / total_cases * 100"
                ),
            ),
            build_kpi(
                name="Largest backlog exposure",
                value=round_or_none(table["backlog_days"].max(), 2),
                unit=Unit.COUNT,
                source_dataset=DatasetId.OPERATIONS,
                period=period_label,
                dimension=str(table.loc[table["backlog_days"].idxmax(), "label"]),
                calculation=(
                    "Closing backlog_open_items divided by the department's average daily "
                    "completed_operations, expressed in days"
                ),
            ),
        ]

        outcome.evidence = self._evidence(table, period_label)
        outcome.findings.extend(self._findings(table, period_label))
        outcome.extend_caveats(
            "cycle_time_ratio compares each department against its own SLA target rather than "
            "against other departments, because targets differ by process",
            "headcount is a static allocation in departments.csv and does not vary by period",
        )
        outcome.row_count_examined = len(operations)
        return outcome

    # -- internals -----------------------------------------------------
    def _evidence(self, table: pd.DataFrame, period_label: str) -> list[EvidenceItem]:
        items: list[EvidenceItem] = []
        ordered = table.sort_values("sla_compliance", ascending=False, na_position="last")
        best = ordered.iloc[0]
        items.append(
            build_evidence(
                metric="Best SLA compliance by department",
                value=round_or_none(best["sla_compliance"]),
                unit=Unit.PERCENT,
                source_dataset=DatasetId.SLA_METRICS,
                dimension=str(best["label"]),
                period=period_label,
                calculation_description=(
                    "Recomputed sla_met_cases / total_cases per department, ranked highest first; "
                    "the strongest department is reported"
                ),
                sample_size=int(best["sla_total"] or 0),
                severity=compliance_severity(best["sla_compliance"]),
            )
        )
        worst = ordered.iloc[-1]
        if len(ordered) > 1 and worst["sla_compliance"] is not None:
            items.append(
                build_evidence(
                    metric="Weakest SLA compliance by department",
                    value=round_or_none(worst["sla_compliance"]),
                    unit=Unit.PERCENT,
                    source_dataset=DatasetId.SLA_METRICS,
                    dimension=str(worst["label"]),
                    period=period_label,
                    calculation_description=(
                        "Recomputed sla_met_cases / total_cases per department, ranked lowest "
                        "first; the weakest department is reported"
                    ),
                    sample_size=int(worst["sla_total"] or 0),
                    severity=compliance_severity(worst["sla_compliance"]),
                )
            )
        slowest = table.sort_values("cycle_time_ratio", ascending=False, na_position="last").iloc[0]
        items.append(
            build_evidence(
                metric="Largest cycle time relative to its own SLA target",
                value=round_or_none(slowest["cycle_time_ratio"], 3),
                unit=Unit.PERCENT,
                source_dataset=DatasetId.SLA_METRICS,
                dimension=str(slowest["label"]),
                period=period_label,
                calculation_description=(
                    "average_cycle_time_hours divided by the mean sla_target_hours of the same "
                    "department; 1.0 means exactly on target"
                ),
                sample_size=int(slowest["volume"] or 0),
                severity=ratio_severity(slowest["cycle_time_ratio"]),
            )
        )
        return items

    def _findings(self, table: pd.DataFrame, period_label: str) -> list[Finding]:
        findings: list[Finding] = []

        over = table.loc[table["cycle_time_ratio"] >= OVER_TARGET_RATIO].sort_values(
            "cycle_time_ratio", ascending=False
        )
        for _, row in over.iterrows():
            findings.append(
                build_finding(
                    title=f"{row['label']} is past its cycle-time target",
                    detail=(
                        f"{row['label']} averages {row['avg_cycle_time']:.2f} hours of cycle time "
                        f"against an average SLA target of {row['avg_sla_target']:.2f} hours "
                        f"({row['cycle_time_ratio']:.2f}x target), with "
                        f"{row['sla_compliance']:.1f}% "
                        f"SLA compliance over {row['sla_total']:,} cases."
                    ),
                    severity=ratio_severity(row["cycle_time_ratio"]),
                    metric="average_cycle_time",
                    dimension=str(row["label"]),
                    period=period_label,
                    statistic=(f"cycle time / target = {row['cycle_time_ratio']:.3f}"),
                )
            )

        backlog_heavy = table.loc[table["backlog_days"] >= MATERIAL_BACKLOG_DAYS].sort_values(
            "backlog_days", ascending=False
        )
        for _, row in backlog_heavy.iterrows():
            findings.append(
                build_finding(
                    title=f"{row['label']} carries {row['backlog_days']:.1f} days of backlog",
                    detail=(
                        f"{row['label']} closed {row['closing_backlog']:,.0f} open items, which is "
                        f"about {row['backlog_days']:.1f} days of its own average throughput of "
                        f"{row['throughput']:,.0f} operations per day."
                    ),
                    severity=Severity.HIGH
                    if row["backlog_days"] >= MATERIAL_BACKLOG_DAYS * 3
                    else Severity.MEDIUM,
                    metric="backlog",
                    dimension=str(row["label"]),
                    period=period_label,
                    statistic=f"backlog / daily throughput = {row['backlog_days']:.2f} days",
                )
            )

        if not findings:
            findings.append(
                build_finding(
                    title="All departments are within their own targets",
                    detail=(
                        f"No department in {period_label} exceeded its average SLA target on cycle "
                        f"time, and none carries more than {MATERIAL_BACKLOG_DAYS:.0f} days of "
                        "backlog."
                    ),
                    severity=Severity.INFO,
                    period=period_label,
                )
            )
        return findings


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def department_scorecard(
    operations: pd.DataFrame,
    sla_frame: pd.DataFrame | None,
    headcount: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Build the per-department scorecard table."""
    if operations is None or operations.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for department, group in operations.groupby("department", dropna=True):
        weights = group["completed_operations"].fillna(0).astype("float64")
        total_weight = float(weights.sum())
        days = max(1, int(group["date"].nunique()))
        throughput = float(group["completed_operations"].sum()) / days
        cycle = (
            float((group["total_cycle_time_hours"] * weights).sum() / total_weight)
            if total_weight > 0
            else round_or_none(group["total_cycle_time_hours"].mean())
        )
        last_date = group["date"].max()
        closing = float(group.loc[group["date"] == last_date, "backlog_open_items"].sum())

        sla_summary = compliance_summary(
            sla_frame.loc[sla_frame["department"] == department]
            if sla_frame is not None and not sla_frame.empty
            else None
        )
        sla_rows = (
            sla_frame.loc[sla_frame["department"] == department]
            if sla_frame is not None and not sla_frame.empty
            else pd.DataFrame()
        )
        avg_target = (
            round_or_none(sla_rows["sla_target_hours"].mean()) if not sla_rows.empty else None
        )
        cycle_ratio = round_or_none(cycle / avg_target, 3) if cycle and avg_target else None

        rows.append(
            {
                "department": str(department),
                "throughput": round_or_none(throughput),
                "avg_cycle_time": round_or_none(cycle),
                "avg_sla_target": avg_target,
                "cycle_time_ratio": cycle_ratio,
                "sla_compliance": sla_summary["compliance"],
                "sla_met": sla_summary["met"],
                "sla_breached": sla_summary["breached"],
                "sla_total": sla_summary["total"],
                "closing_backlog": round_or_none(closing, 1),
                "backlog_days": round_or_none(closing / throughput, 2) if throughput > 0 else None,
                "volume": int(group["completed_operations"].sum()),
                "headcount": int((headcount or {}).get(str(department), 0)),
                "operations_per_head": round_or_none(
                    float(group["completed_operations"].sum())
                    / (headcount or {}).get(str(department), 1)
                )
                if (headcount or {}).get(str(department))
                else None,
                "processes": int(group["process"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _headcount_by_department(context: AnalysisContext) -> dict[str, int]:
    frame = context.frame(DatasetId.DEPARTMENTS)
    if frame.empty:
        try:
            frame = context.repository.departments
        except Exception:  # pragma: no cover - reference data is optional
            return {}
    if frame.empty:
        return {}
    return {str(row["department_id"]): int(row["headcount"]) for _, row in frame.iterrows()}


def ratio_severity(ratio: float | None) -> Severity:
    value = safe_float(ratio)
    if value is None:
        return Severity.INFO
    if value >= 1.5:
        return Severity.CRITICAL
    if value >= 1.25:
        return Severity.HIGH
    if value >= 1.0:
        return Severity.MEDIUM
    return Severity.LOW


__all__ = ["DepartmentAnalyzer", "department_scorecard", "ratio_severity"]
