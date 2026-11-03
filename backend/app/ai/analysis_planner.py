"""Analysis planner: validates and enriches a question intent.

The planner takes the structured intent produced by the question interpreter
and produces a fully validated :class:`~app.schemas.analysis.AnalysisPlan`.
Its responsibilities:

* map the intent's free-form metric/analysis-type strings to the controlled
  :class:`~app.models.enums.AnalysisType` and :class:`~app.models.enums.Metric`
  enums;
* infer a comparison period when the question implies one;
* attach the list of datasets the analysis will need;
* attach a reasoning goal that guides the explanation layer.

The planner does NOT execute the analysis. That happens downstream in the
:mod:`~app.services.analysis_service`.
"""

from __future__ import annotations

from contextlib import suppress

from app.core.logging import get_logger
from app.models.enums import (
    COMPARISON_ANALYSIS_TYPES,
    AnalysisType,
    DatasetId,
    Dimension,
    Metric,
    ReasoningGoal,
)
from app.schemas.ai import QuestionIntent
from app.schemas.analysis import AnalysisPlan
from app.utils.periods import months_covered_by, preceding_period, resolve_period_range

logger = get_logger(__name__)

#: Analysis types that always need a comparison window.
COMPARISON_TYPES: frozenset[AnalysisType] = COMPARISON_ANALYSIS_TYPES

#: Default analysis type per free-form type string the interpreter may return.
_TYPE_MAP: dict[str, AnalysisType] = {
    "kpi_summary": AnalysisType.KPI_SUMMARY,
    "trend_analysis": AnalysisType.TREND_ANALYSIS,
    "period_comparison": AnalysisType.PERIOD_COMPARISON,
    "cycle_time_analysis": AnalysisType.CYCLE_TIME_ANALYSIS,
    "throughput_analysis": AnalysisType.THROUGHPUT_ANALYSIS,
    "backlog_analysis": AnalysisType.BACKLOG_ANALYSIS,
    "sla_analysis": AnalysisType.SLA_ANALYSIS,
    "bottleneck_analysis": AnalysisType.BOTTLENECK_ANALYSIS,
    "vendor_analysis": AnalysisType.VENDOR_ANALYSIS,
    "department_analysis": AnalysisType.DEPARTMENT_ANALYSIS,
    "anomaly_analysis": AnalysisType.ANOMALY_ANALYSIS,
    "process_analysis": AnalysisType.PROCESS_ANALYSIS,
    "contribution_analysis": AnalysisType.CONTRIBUTION_ANALYSIS,
}

#: Default metric per analysis type.
DEFAULT_METRIC_BY_TYPE: dict[AnalysisType, Metric] = {
    AnalysisType.KPI_SUMMARY: Metric.THROUGHPUT,
    AnalysisType.TREND_ANALYSIS: Metric.AVERAGE_CYCLE_TIME,
    AnalysisType.PERIOD_COMPARISON: Metric.AVERAGE_CYCLE_TIME,
    AnalysisType.CYCLE_TIME_ANALYSIS: Metric.AVERAGE_CYCLE_TIME,
    AnalysisType.THROUGHPUT_ANALYSIS: Metric.THROUGHPUT,
    AnalysisType.BACKLOG_ANALYSIS: Metric.BACKLOG,
    AnalysisType.SLA_ANALYSIS: Metric.SLA_COMPLIANCE,
    AnalysisType.BOTTLENECK_ANALYSIS: Metric.AVERAGE_CYCLE_TIME,
    AnalysisType.VENDOR_ANALYSIS: Metric.ON_TIME_DELIVERY_RATE,
    AnalysisType.DEPARTMENT_ANALYSIS: Metric.SLA_COMPLIANCE,
    AnalysisType.ANOMALY_ANALYSIS: Metric.ANOMALY_COUNT,
    AnalysisType.PROCESS_ANALYSIS: Metric.AVERAGE_CYCLE_TIME,
    AnalysisType.CONTRIBUTION_ANALYSIS: Metric.AVERAGE_CYCLE_TIME,
}

#: Datasets required per analysis type.
_DATASET_MAP: dict[AnalysisType, tuple[DatasetId, ...]] = {
    AnalysisType.KPI_SUMMARY: (DatasetId.OPERATIONS, DatasetId.SLA_METRICS, DatasetId.DEPARTMENTS),
    AnalysisType.TREND_ANALYSIS: (
        DatasetId.OPERATIONS,
        DatasetId.SLA_METRICS,
        DatasetId.DEPARTMENTS,
    ),
    AnalysisType.PERIOD_COMPARISON: (
        DatasetId.OPERATIONS,
        DatasetId.SLA_METRICS,
        DatasetId.ORDERS,
        DatasetId.DEPARTMENTS,
    ),
    AnalysisType.CYCLE_TIME_ANALYSIS: (
        DatasetId.OPERATIONS,
        DatasetId.ORDERS,
        DatasetId.DEPARTMENTS,
    ),
    AnalysisType.THROUGHPUT_ANALYSIS: (DatasetId.OPERATIONS, DatasetId.DEPARTMENTS),
    AnalysisType.BACKLOG_ANALYSIS: (DatasetId.OPERATIONS, DatasetId.DEPARTMENTS),
    AnalysisType.SLA_ANALYSIS: (DatasetId.SLA_METRICS, DatasetId.DEPARTMENTS),
    AnalysisType.BOTTLENECK_ANALYSIS: (DatasetId.OPERATIONS, DatasetId.DEPARTMENTS),
    AnalysisType.VENDOR_ANALYSIS: (
        DatasetId.ORDERS,
        DatasetId.OPERATIONS,
        DatasetId.VENDORS,
        DatasetId.DEPARTMENTS,
    ),
    AnalysisType.DEPARTMENT_ANALYSIS: (
        DatasetId.OPERATIONS,
        DatasetId.SLA_METRICS,
        DatasetId.DEPARTMENTS,
    ),
    AnalysisType.ANOMALY_ANALYSIS: (
        DatasetId.OPERATIONS,
        DatasetId.SLA_METRICS,
        DatasetId.DEPARTMENTS,
    ),
    AnalysisType.PROCESS_ANALYSIS: (DatasetId.OPERATIONS, DatasetId.DEPARTMENTS),
    AnalysisType.CONTRIBUTION_ANALYSIS: (
        DatasetId.OPERATIONS,
        DatasetId.ORDERS,
        DatasetId.DEPARTMENTS,
    ),
}


def plan_from_intent(
    intent: QuestionIntent,
    available_periods: list[str],
    available_period_set: frozenset[str],
) -> AnalysisPlan:
    """Convert a parsed question intent into a validated analysis plan."""
    analysis_type = _resolve_analysis_type(intent)
    metric = _resolve_metric(intent, analysis_type)
    dimensions = _resolve_dimensions(intent)
    period, comparison_period, notes = _resolve_periods(
        intent, analysis_type, available_periods, available_period_set
    )
    required_datasets = _DATASET_MAP.get(analysis_type, ())
    reasoning_goal = _resolve_reasoning(intent, analysis_type)

    return AnalysisPlan(
        analysis_type=analysis_type,
        metric=metric,
        dimensions=dimensions,
        period=period,
        comparison_period=comparison_period,
        department=intent.department,
        process=intent.process,
        vendor_id=intent.vendor_id,
        top_n=5,
        required_datasets=required_datasets,
        reasoning_goal=reasoning_goal,
        notes=notes or None,
    )


def _resolve_analysis_type(intent: QuestionIntent) -> AnalysisType:
    """Map the interpreter's free-form type string to an enum member."""
    candidate = _TYPE_MAP.get(intent.analysis_type)
    if candidate is not None:
        return candidate
    # Try partial match on the analysis_type string.
    text = intent.analysis_type.lower().replace(" ", "_")
    for key, value in _TYPE_MAP.items():
        if key in text or text in key:
            return value
    return AnalysisType.KPI_SUMMARY


def _resolve_metric(intent: QuestionIntent, analysis_type: AnalysisType) -> Metric:
    """Resolve the metric, falling back to the default for the analysis type."""
    try:
        return Metric(intent.metric)
    except (ValueError, AttributeError):
        return DEFAULT_METRIC_BY_TYPE.get(analysis_type, Metric.AVERAGE_CYCLE_TIME)


def _resolve_dimensions(intent: QuestionIntent) -> tuple[Dimension, ...]:
    """Keep only the dimensions that exist in the controlled vocabulary."""
    resolved: list[Dimension] = []
    for item in intent.dimensions:
        with suppress(ValueError, AttributeError):
            resolved.append(Dimension(item))
    return tuple(resolved)


def _resolve_periods(
    intent: QuestionIntent,
    analysis_type: AnalysisType,
    available: list[str],
    available_set: frozenset[str],
) -> tuple[str | None, str | None, str]:
    """Resolve periods against the months that actually exist.

    Two-window analyses need two real windows, so when the question names only
    one period the immediately preceding window of the same size is used. When it
    names none at all, the most recent month becomes the primary window. Both
    inferences are recorded in the plan's notes, because a reader must be able to
    tell which window was asked for and which was supplied.
    """
    notes: list[str] = []
    period = intent.period if _period_has_data(intent.period, available_set) else None
    comparison_period = (
        intent.comparison_period
        if _period_has_data(intent.comparison_period, available_set)
        else None
    )

    needs_comparison = analysis_type in COMPARISON_TYPES or intent.comparison_period is not None

    if needs_comparison and period is None and comparison_period is not None:
        # "Compare the previous quarter with Q3": the later window is the primary.
        comparison_range, _, _ = resolve_period_range(comparison_period, available_set)
        preceding = (
            preceding_period(comparison_range, available_set)
            if comparison_range is not None
            else None
        )
        period = preceding.start if preceding is not None else None
        notes.append(f"the later window {comparison_period} was treated as the current period")

    if needs_comparison and period is None and available:
        period = available[-1]
        notes.append(
            f"no period was named in the question, so the most recent month {period} was used"
        )

    if needs_comparison and comparison_period is None and period is not None:
        primary, _, _ = resolve_period_range(period, available_set)
        if primary is not None:
            preceding = preceding_period(primary, available_set)
            if preceding is not None:
                comparison_period = (
                    preceding.start
                    if preceding.month_count == 1
                    else f"{preceding.start} to {preceding.end}"
                )
                notes.append(
                    f"no comparison period was named, so {comparison_period} (the immediately "
                    f"preceding window) was used"
                )

    if notes:
        plan_notes = "; ".join(notes)
    else:
        plan_notes = ""

    return period, comparison_period, plan_notes


def _period_has_data(period: str | None, available_months: frozenset[str]) -> bool:
    """True when a requested period overlaps the months the dataset contains.

    The available set holds month buckets, while a question may name a year, a
    month or a quarter. Membership is therefore tested by expanding the request
    and checking for any overlap, rather than by string equality.
    """
    if not period:
        return False
    months = months_covered_by(period)
    if not months:
        return False
    return any(month in available_months for month in months)


def _resolve_reasoning(intent: QuestionIntent, analysis_type: AnalysisType) -> ReasoningGoal:
    return intent.reasoning_goal


__all__ = ["COMPARISON_TYPES", "plan_from_intent"]
