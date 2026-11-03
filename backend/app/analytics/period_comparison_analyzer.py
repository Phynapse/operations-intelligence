"""Period comparison analysis: "what changed, and where".

``period_comparison`` is a composite rather than a separate calculation. It
selects the underlying analyzer from the requested metric and runs it against the
two resolved windows, so a comparison question never needs its own arithmetic:

* cycle-time, queue-time and processing-time metrics delegate to
  :class:`~app.analytics.cycle_time_analyzer.CycleTimeAnalyzer`;
* SLA compliance and breach rate delegate to
  :class:`~app.analytics.sla_analyzer.SLAAnalyzer`;
* throughput and backlog delegate to their own analyzers.

The composite declares the union of the delegated datasets so the context
builder loads everything any branch might need.
"""

from __future__ import annotations

from app.analytics.base import AnalysisContext, AnalysisOutcome, BaseAnalyzer
from app.models.enums import AnalysisType, DatasetId, Metric
from app.utils.errors import UnsupportedAnalysisError

#: Metric -> analysis type whose analyzer performs the comparison.
METRIC_TO_ANALYSIS: dict[Metric, AnalysisType] = {
    Metric.AVERAGE_CYCLE_TIME: AnalysisType.CYCLE_TIME_ANALYSIS,
    Metric.MEDIAN_CYCLE_TIME: AnalysisType.CYCLE_TIME_ANALYSIS,
    Metric.PROCESSING_TIME: AnalysisType.CYCLE_TIME_ANALYSIS,
    Metric.QUEUE_TIME: AnalysisType.CYCLE_TIME_ANALYSIS,
    Metric.QUEUE_CONTRIBUTION: AnalysisType.CYCLE_TIME_ANALYSIS,
    Metric.FULFILLMENT_TIME: AnalysisType.CYCLE_TIME_ANALYSIS,
    Metric.THROUGHPUT: AnalysisType.THROUGHPUT_ANALYSIS,
    Metric.PRODUCTIVITY_PER_HEAD: AnalysisType.THROUGHPUT_ANALYSIS,
    Metric.BACKLOG: AnalysisType.BACKLOG_ANALYSIS,
    Metric.BACKLOG_CHANGE: AnalysisType.BACKLOG_ANALYSIS,
    Metric.SLA_COMPLIANCE: AnalysisType.SLA_ANALYSIS,
    Metric.SLA_BREACH_RATE: AnalysisType.SLA_ANALYSIS,
    Metric.DELAY_RATE: AnalysisType.THROUGHPUT_ANALYSIS,
    Metric.ON_TIME_DELIVERY_RATE: AnalysisType.VENDOR_ANALYSIS,
    Metric.ANOMALY_COUNT: AnalysisType.ANOMALY_ANALYSIS,
    Metric.ORDER_VALUE: AnalysisType.CYCLE_TIME_ANALYSIS,
}

#: Analyzers that must never be invoked through this composite.
DELEGATION_BLOCKLIST: frozenset[AnalysisType] = frozenset(
    {
        AnalysisType.PERIOD_COMPARISON,
        AnalysisType.KPI_SUMMARY,
        AnalysisType.DEPARTMENT_ANALYSIS,
        AnalysisType.PROCESS_ANALYSIS,
    }
)


class PeriodComparisonAnalyzer(BaseAnalyzer):
    """Route a period-over-period question to the analyzer that owns the metric."""

    name = "period_comparison_analyzer"
    supported_types = frozenset({AnalysisType.PERIOD_COMPARISON})
    required_datasets = (
        DatasetId.OPERATIONS,
        DatasetId.SLA_METRICS,
        DatasetId.DEPARTMENTS,
        DatasetId.ORDERS,
        DatasetId.VENDORS,
    )

    def analyze(self, context: AnalysisContext) -> AnalysisOutcome:
        # Imported here to avoid a cycle: the registry instantiates this class.
        from app.analytics.registry import get_analyzer

        metric = context.plan.metric
        target = METRIC_TO_ANALYSIS.get(metric)
        if target is None or target in DELEGATION_BLOCKLIST:
            raise UnsupportedAnalysisError(
                f"metric '{metric.value}' cannot be compared between periods; comparable "
                "metrics: " + ", ".join(sorted(item.value for item in METRIC_TO_ANALYSIS))
            )

        delegate = get_analyzer(target)
        outcome = delegate.analyze(context)
        outcome.extend_caveats(
            f"comparison performed by the {delegate.name} for metric '{metric.value}' across "
            f"{context.plan.period or 'the requested window'} and "
            f"{context.plan.comparison_period or 'its immediate predecessor'}"
        )
        return outcome


__all__ = ["METRIC_TO_ANALYSIS", "PeriodComparisonAnalyzer"]
