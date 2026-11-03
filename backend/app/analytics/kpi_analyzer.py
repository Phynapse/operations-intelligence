"""KPI summary analysis: the headline operational numbers for a period.

This is the default analysis. It answers "how is operations doing" with a fixed,
deterministic set of metrics rather than an open-ended search, so the same
period always produces the same summary.
"""

from __future__ import annotations

import pandas as pd

from app.analytics.backlog_analyzer import closing_backlog
from app.analytics.base import (
    AnalysisContext,
    AnalysisOutcome,
    BaseAnalyzer,
    build_evidence,
    build_finding,
    build_kpi,
)
from app.analytics.cycle_time_analyzer import cycle_components
from app.analytics.sla_analyzer import ComplianceSummary, compliance_severity, compliance_summary
from app.analytics.throughput_analyzer import distinct_days, rate_per_day
from app.models.enums import AnalysisType, DatasetId, Metric, Severity, Unit
from app.schemas.analysis import Finding
from app.utils.calculations import delay_rate as delay_rate_of
from app.utils.errors import InsufficientDataError


class KPIAnalyzer(BaseAnalyzer):
    """Throughput, cycle time, SLA compliance, backlog and delay rate."""

    name = "kpi_analyzer"
    supported_types = frozenset({AnalysisType.KPI_SUMMARY})
    required_datasets = (DatasetId.OPERATIONS, DatasetId.SLA_METRICS, DatasetId.DEPARTMENTS)

    #: KPI definitions exposed by this analyzer, in display order.
    METRIC_ORDER = (
        Metric.THROUGHPUT,
        Metric.AVERAGE_CYCLE_TIME,
        Metric.SLA_COMPLIANCE,
        Metric.BACKLOG,
        Metric.DELAY_RATE,
    )

    def analyze(self, context: AnalysisContext) -> AnalysisOutcome:
        operations = context.primary_frame()
        if operations.empty:
            raise InsufficientDataError(
                f"no operational records found for {context.plan.period or 'the requested period'}"
            )

        outcome = AnalysisOutcome()
        period_label = context.plan.period or "all periods"
        comparison = context.comparison_frame()

        days = distinct_days(operations)
        throughput_total = float(operations["completed_operations"].sum())
        delayed_total = float(operations["delayed_operations"].sum())
        throughput_daily = rate_per_day(throughput_total, days)
        cycle = cycle_components(operations)
        backlog = closing_backlog(operations)
        delay = delay_rate_of(delayed_total, throughput_total)

        sla_frame = _sla_slice(context)
        sla = compliance_summary(sla_frame)

        comparison_days = distinct_days(comparison) if comparison is not None else 0
        comparison_throughput = (
            float(comparison["completed_operations"].sum()) if comparison is not None else 0.0
        )
        comparison_daily = (
            rate_per_day(comparison_throughput, comparison_days) if comparison_days else None
        )
        comparison_cycle = cycle_components(comparison) if comparison is not None else None
        comparison_sla = compliance_summary(_comparison_sla_slice(context))
        comparison_backlog = closing_backlog(comparison)
        comparison_delay = (
            delay_rate_of(
                float(comparison["delayed_operations"].sum()),
                float(comparison["completed_operations"].sum()),
            )
            if comparison is not None and not comparison.empty
            else None
        )

        outcome.kpis = [
            build_kpi(
                name="Throughput per day",
                value=throughput_daily,
                unit=Unit.OPERATIONS,
                source_dataset=DatasetId.OPERATIONS,
                period=period_label,
                calculation=f"sum(completed_operations) / {days} distinct days present in the data",
            ),
            build_kpi(
                name="Average cycle time",
                value=cycle["cycle"],
                unit=Unit.HOURS,
                source_dataset=DatasetId.OPERATIONS,
                period=period_label,
                calculation="completed_operations-weighted mean of total_cycle_time_hours",
            ),
            build_kpi(
                name="SLA compliance",
                value=sla["compliance"],
                unit=Unit.PERCENT,
                source_dataset=DatasetId.SLA_METRICS,
                period=period_label,
                calculation="sla_met_cases / total_cases * 100 across sla_metrics.csv rows",
            ),
            build_kpi(
                name="Closing backlog",
                value=backlog,
                unit=Unit.ITEMS,
                source_dataset=DatasetId.OPERATIONS,
                period=period_label,
                calculation=(
                    "sum of backlog_open_items over all operations.csv records on the final day "
                    "of the period"
                ),
            ),
            build_kpi(
                name="Delay rate",
                value=delay,
                unit=Unit.PERCENT,
                source_dataset=DatasetId.OPERATIONS,
                period=period_label,
                calculation="sum(delayed_operations) / sum(completed_operations) * 100",
            ),
        ]

        outcome.evidence = [
            build_evidence(
                metric="Throughput per day",
                value=throughput_daily,
                comparison_value=comparison_daily,
                unit=Unit.OPERATIONS,
                source_dataset=DatasetId.OPERATIONS,
                period=period_label,
                comparison_period=context.plan.comparison_period,
                calculation_description=(
                    "Completed operations divided by the number of distinct calendar days in "
                    "operations.csv"
                ),
                sample_size=days,
            ),
            build_evidence(
                metric="Average cycle time",
                value=cycle["cycle"],
                comparison_value=comparison_cycle["cycle"] if comparison_cycle else None,
                unit=Unit.HOURS,
                source_dataset=DatasetId.OPERATIONS,
                period=period_label,
                comparison_period=context.plan.comparison_period,
                calculation_description=(
                    "completed_operations-weighted mean of total_cycle_time_hours"
                ),
                sample_size=int(throughput_total),
            ),
            build_evidence(
                metric="SLA compliance",
                value=sla["compliance"],
                comparison_value=comparison_sla["compliance"],
                unit=Unit.PERCENT,
                source_dataset=DatasetId.SLA_METRICS,
                period=period_label,
                comparison_period=context.plan.comparison_period,
                calculation_description=(
                    "sla_met_cases / total_cases * 100 across the monthly SLA rollup"
                ),
                sample_size=sla["total"],
                severity=compliance_severity(sla["compliance"]),
            ),
            build_evidence(
                metric="Closing backlog",
                value=backlog,
                comparison_value=comparison_backlog,
                unit=Unit.ITEMS,
                source_dataset=DatasetId.OPERATIONS,
                period=period_label,
                comparison_period=context.plan.comparison_period,
                calculation_description=("Total backlog_open_items on the final day of the period"),
            ),
            build_evidence(
                metric="Delay rate",
                value=delay,
                comparison_value=comparison_delay,
                unit=Unit.PERCENT,
                source_dataset=DatasetId.OPERATIONS,
                period=period_label,
                comparison_period=context.plan.comparison_period,
                calculation_description=(
                    "sum(delayed_operations) / sum(completed_operations) * 100"
                ),
                sample_size=int(throughput_total),
            ),
            build_evidence(
                metric="Queue contribution to cycle time",
                value=cycle["queue_share"],
                unit=Unit.PERCENT,
                source_dataset=DatasetId.OPERATIONS,
                period=period_label,
                calculation_description=(
                    "queue_time_hours / total_cycle_time_hours * 100, weighted by "
                    "completed_operations"
                ),
                sample_size=int(throughput_total),
            ),
        ]

        outcome.findings.extend(_summary_findings(cycle, sla, backlog, delay, period_label))
        outcome.extend_caveats(*_summary_caveats(sla, days, period_label))
        outcome.row_count_examined = len(operations)
        return outcome


def _sla_slice(context: AnalysisContext) -> pd.DataFrame:
    frame = context.frame(DatasetId.SLA_METRICS)
    if frame.empty or context.primary is None:
        return frame
    months = set(context.primary.months)
    return frame.loc[frame["period"].isin(months)]


def _comparison_sla_slice(context: AnalysisContext) -> pd.DataFrame:
    frame = context.frame(DatasetId.SLA_METRICS)
    if frame.empty or context.comparison is None:
        return frame
    months = set(context.comparison.months)
    return frame.loc[frame["period"].isin(months)]


def _summary_findings(
    cycle: dict[str, float | None],
    sla: ComplianceSummary,
    backlog: float | None,
    delay: float | None,
    period_label: str,
) -> list[Finding]:
    findings = []

    queue_share = cycle.get("queue_share")
    if queue_share is not None and queue_share >= 40.0:
        findings.append(
            build_finding(
                title="Waiting dominates cycle time",
                detail=(
                    f"{queue_share:.1f}% of average cycle time is spent queued rather than being "
                    f"worked on ({cycle.get('queue') or 0.0:.2f} of "
                    f"{cycle.get('cycle') or 0.0:.2f} hours). Capacity, not work content, is the "
                    "dominant constraint."
                ),
                severity=Severity.HIGH if queue_share >= 60.0 else Severity.MEDIUM,
                metric="queue_contribution",
                period=period_label,
                statistic=f"weighted queue share = {queue_share:.2f}%",
            )
        )

    compliance = sla.get("compliance")
    if compliance is not None and compliance < 90.0:
        findings.append(
            build_finding(
                title="SLA compliance below target",
                detail=(
                    f"Only {compliance:.1f}% of {sla['total']:,} measured cases met their SLA "
                    f"target in {period_label}, leaving {sla['breached']:,} breaches."
                ),
                severity=compliance_severity(compliance),
                metric="sla_compliance",
                period=period_label,
                statistic=f"{sla['met']:,}/{sla['total']:,} cases met target",
            )
        )

    if backlog is not None and backlog >= 100.0:
        findings.append(
            build_finding(
                title=f"Backlog stands at {backlog:,.0f} items",
                detail=(
                    f"{backlog:,.0f} open items remained at the end of {period_label}. Work "
                    "incoming is currently outpacing the capacity clearing it."
                ),
                severity=Severity.HIGH if backlog >= 500 else Severity.MEDIUM,
                metric="backlog",
                period=period_label,
                statistic=f"closing backlog_open_items = {backlog:,.1f}",
            )
        )

    if delay is not None and delay >= 10.0:
        findings.append(
            build_finding(
                title="Elevated delay rate",
                detail=(
                    f"{delay:.1f}% of completed operations were late against the operational "
                    f"promise in {period_label}."
                ),
                severity=Severity.HIGH if delay >= 20.0 else Severity.MEDIUM,
                metric="delay_rate",
                period=period_label,
                statistic=f"delay rate = {delay:.2f}%",
            )
        )

    if not findings:
        findings.append(
            build_finding(
                title="Operations within tolerance",
                detail=(
                    f"Throughput, cycle time, SLA compliance, backlog and delay rate were all "
                    f"within normal operating tolerance in {period_label}."
                ),
                severity=Severity.INFO,
                period=period_label,
            )
        )
    return findings


def _summary_caveats(sla: ComplianceSummary, days: int, period_label: str) -> list[str]:
    caveats: list[str] = []
    if not sla["total"]:
        caveats.append(
            f"no SLA cases were available for {period_label}; SLA compliance is reported as "
            "undefined rather than zero"
        )
    elif int(sla["total"]) < 100:
        caveats.append(
            f"only {int(sla['total'])} SLA cases back this summary, so compliance is sensitive to "
            "a small number of breaches"
        )
    if days and days < 14:
        caveats.append(
            f"the period contains {days} day(s) of data; short windows exaggerate weekly and "
            "month-end effects"
        )
    return caveats


__all__ = ["KPIAnalyzer"]
