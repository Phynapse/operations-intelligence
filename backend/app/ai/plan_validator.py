"""Strict analysis-plan validation.

This module is the security and correctness boundary between any planner
(deterministic or LLM-driven) and the execution layer. Nothing is executed until
it passes :func:`validate_plan`.

The validator is intentionally allow-list based. A field is accepted only when
its value is a member of the controlled vocabulary (an enum) *and* the
combination of fields is one the registered analyzers can actually run. There is
no code path that turns a plan field into a filesystem path, a Python
expression, a SQL fragment, or an HTTP request — a plan can only ever name an
``AnalysisType``, a ``Metric``, a ``Dimension``, a known period, and a known
department/process/vendor id.

Because the planner may be an LLM, the inputs are treated as untrusted. Unknown
strings are rejected with an explanatory message rather than coerced.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.data.repository import DatasetRepository
from app.models.enums import (
    AnalysisType,
    DatasetId,
    Dimension,
    Metric,
    ReasoningGoal,
)
from app.schemas.analysis import AnalysisPlan
from app.utils.errors import InvalidRequestError, UnsupportedAnalysisError
from app.utils.periods import months_covered_by

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Supported metric / dimension combinations per analysis type.
#
# A metric or dimension is accepted for an analysis type only if it appears in
# these allow-lists. This stops a syntactically valid but semantically
# nonsensical plan (e.g. "throughput analysis of SLA breach rate") from running.
# ---------------------------------------------------------------------------

#: Metrics each analysis type can legitimately operate on.
ALLOWED_METRICS: dict[AnalysisType, frozenset[Metric]] = {
    AnalysisType.KPI_SUMMARY: frozenset(Metric),
    AnalysisType.TREND_ANALYSIS: frozenset(
        {
            Metric.THROUGHPUT,
            Metric.AVERAGE_CYCLE_TIME,
            Metric.PROCESSING_TIME,
            Metric.QUEUE_TIME,
            Metric.BACKLOG,
            Metric.DELAY_RATE,
            Metric.SLA_COMPLIANCE,
        }
    ),
    AnalysisType.PERIOD_COMPARISON: frozenset(
        {
            Metric.AVERAGE_CYCLE_TIME,
            Metric.MEDIAN_CYCLE_TIME,
            Metric.PROCESSING_TIME,
            Metric.QUEUE_TIME,
            Metric.QUEUE_CONTRIBUTION,
            Metric.FULFILLMENT_TIME,
            Metric.ON_TIME_DELIVERY_RATE,
            Metric.ORDER_VALUE,
            Metric.THROUGHPUT,
            Metric.DELAY_RATE,
            Metric.BACKLOG,
            Metric.BACKLOG_CHANGE,
            Metric.SLA_COMPLIANCE,
            Metric.SLA_BREACH_RATE,
            Metric.PRODUCTIVITY_PER_HEAD,
            Metric.ANOMALY_COUNT,
        }
    ),
    AnalysisType.CYCLE_TIME_ANALYSIS: frozenset(
        {
            Metric.AVERAGE_CYCLE_TIME,
            Metric.MEDIAN_CYCLE_TIME,
            Metric.PROCESSING_TIME,
            Metric.QUEUE_TIME,
            Metric.QUEUE_CONTRIBUTION,
            Metric.FULFILLMENT_TIME,
            Metric.ON_TIME_DELIVERY_RATE,
            Metric.ORDER_VALUE,
        }
    ),
    AnalysisType.THROUGHPUT_ANALYSIS: frozenset({Metric.THROUGHPUT, Metric.PRODUCTIVITY_PER_HEAD}),
    AnalysisType.BACKLOG_ANALYSIS: frozenset({Metric.BACKLOG, Metric.BACKLOG_CHANGE}),
    AnalysisType.SLA_ANALYSIS: frozenset({Metric.SLA_COMPLIANCE, Metric.SLA_BREACH_RATE}),
    AnalysisType.BOTTLENECK_ANALYSIS: frozenset(
        {Metric.AVERAGE_CYCLE_TIME, Metric.QUEUE_TIME, Metric.QUEUE_CONTRIBUTION}
    ),
    AnalysisType.VENDOR_ANALYSIS: frozenset(
        {Metric.ON_TIME_DELIVERY_RATE, Metric.DELAY_RATE, Metric.FULFILLMENT_TIME}
    ),
    AnalysisType.DEPARTMENT_ANALYSIS: frozenset(
        {Metric.SLA_COMPLIANCE, Metric.AVERAGE_CYCLE_TIME, Metric.THROUGHPUT, Metric.BACKLOG}
    ),
    AnalysisType.ANOMALY_ANALYSIS: frozenset(
        {
            Metric.ANOMALY_COUNT,
            Metric.AVERAGE_CYCLE_TIME,
            Metric.SLA_COMPLIANCE,
            Metric.THROUGHPUT,
            Metric.BACKLOG,
            Metric.QUEUE_TIME,
            Metric.DELAY_RATE,
        }
    ),
    AnalysisType.PROCESS_ANALYSIS: frozenset(
        {
            Metric.AVERAGE_CYCLE_TIME,
            Metric.QUEUE_TIME,
            Metric.PROCESSING_TIME,
            Metric.QUEUE_CONTRIBUTION,
        }
    ),
    AnalysisType.CONTRIBUTION_ANALYSIS: frozenset(
        {
            Metric.AVERAGE_CYCLE_TIME,
            Metric.MEDIAN_CYCLE_TIME,
            Metric.PROCESSING_TIME,
            Metric.QUEUE_TIME,
            Metric.QUEUE_CONTRIBUTION,
            Metric.THROUGHPUT,
            Metric.DELAY_RATE,
            Metric.BACKLOG,
            Metric.FULFILLMENT_TIME,
            Metric.ON_TIME_DELIVERY_RATE,
            Metric.ORDER_VALUE,
        }
    ),
}

#: Dimensions each analysis type can slice by. Empty frozenset = no dimension.
ALLOWED_DIMENSIONS: dict[AnalysisType, frozenset[Dimension]] = {
    AnalysisType.KPI_SUMMARY: frozenset({Dimension.DEPARTMENT, Dimension.PROCESS}),
    AnalysisType.TREND_ANALYSIS: frozenset(
        {Dimension.DEPARTMENT, Dimension.PROCESS, Dimension.OPERATION_TYPE, Dimension.VENDOR}
    ),
    AnalysisType.PERIOD_COMPARISON: frozenset(
        {
            Dimension.DEPARTMENT,
            Dimension.PROCESS,
            Dimension.OPERATION_TYPE,
            Dimension.VENDOR,
            Dimension.PRODUCT_CATEGORY,
            Dimension.PRIORITY,
            Dimension.STATUS,
        }
    ),
    AnalysisType.CYCLE_TIME_ANALYSIS: frozenset(
        {
            Dimension.DEPARTMENT,
            Dimension.PROCESS,
            Dimension.OPERATION_TYPE,
            Dimension.PRIORITY,
            Dimension.VENDOR,
            Dimension.PRODUCT_CATEGORY,
            Dimension.STATUS,
        }
    ),
    AnalysisType.THROUGHPUT_ANALYSIS: frozenset(
        {
            Dimension.DEPARTMENT,
            Dimension.PROCESS,
            Dimension.OPERATION_TYPE,
            Dimension.PRIORITY,
            Dimension.STATUS,
            Dimension.VENDOR,
        }
    ),
    AnalysisType.BACKLOG_ANALYSIS: frozenset(
        {
            Dimension.DEPARTMENT,
            Dimension.PROCESS,
            Dimension.OPERATION_TYPE,
            Dimension.PRIORITY,
            Dimension.VENDOR,
        }
    ),
    AnalysisType.SLA_ANALYSIS: frozenset({Dimension.DEPARTMENT, Dimension.PROCESS}),
    AnalysisType.BOTTLENECK_ANALYSIS: frozenset(
        {Dimension.DEPARTMENT, Dimension.PROCESS, Dimension.OPERATION_TYPE}
    ),
    AnalysisType.VENDOR_ANALYSIS: frozenset(
        {Dimension.VENDOR, Dimension.DEPARTMENT, Dimension.PRODUCT_CATEGORY, Dimension.STATUS}
    ),
    AnalysisType.DEPARTMENT_ANALYSIS: frozenset({Dimension.DEPARTMENT}),
    AnalysisType.ANOMALY_ANALYSIS: frozenset(
        {Dimension.DEPARTMENT, Dimension.PROCESS, Dimension.OPERATION_TYPE, Dimension.VENDOR}
    ),
    AnalysisType.PROCESS_ANALYSIS: frozenset(
        {Dimension.PROCESS, Dimension.DEPARTMENT, Dimension.OPERATION_TYPE, Dimension.PRIORITY}
    ),
    AnalysisType.CONTRIBUTION_ANALYSIS: frozenset(
        {
            Dimension.DEPARTMENT,
            Dimension.PROCESS,
            Dimension.OPERATION_TYPE,
            Dimension.PRIORITY,
            Dimension.VENDOR,
            Dimension.PRODUCT_CATEGORY,
            Dimension.STATUS,
        }
    ),
}

#: Datasets each analysis type is permitted to read.
REQUIRED_DATASETS: dict[AnalysisType, tuple[DatasetId, ...]] = {
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
        DatasetId.VENDORS,
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

#: Analysis types that require two periods.
_COMPARISON_TYPES: frozenset[AnalysisType] = frozenset(
    {AnalysisType.PERIOD_COMPARISON, AnalysisType.CONTRIBUTION_ANALYSIS}
)

#: A plan may slice by at most this many dimensions.
MAX_DIMENSIONS = 4


@dataclass(slots=True)
class ValidationResult:
    """Outcome of validating a plan."""

    plan: AnalysisPlan
    adjustments: list[str] = field(default_factory=list)

    @property
    def was_adjusted(self) -> bool:
        return bool(self.adjustments)


def validate_plan(
    plan: AnalysisPlan,
    repository: DatasetRepository,
    *,
    strict: bool = False,
) -> ValidationResult:
    """Validate a plan against the controlled vocabulary and the loaded data.

    When ``strict`` is ``True`` any disallowed metric/dimension/filter raises.
    When ``False`` (the default used for LLM plans) a disallowed *dimension* is
    dropped with a recorded adjustment, while a disallowed *metric*,
    *analysis type*, *period*, or *filter value* still raises, because those
    change the meaning of the analysis rather than merely narrowing it.
    """
    adjustments: list[str] = []

    analysis_type = _validate_analysis_type(plan.analysis_type)
    metric = _validate_metric(analysis_type, plan.metric, strict, adjustments)
    dimensions = _validate_dimensions(analysis_type, plan.dimensions, strict, adjustments)
    period = _validate_period(plan.period, repository, "period")
    comparison_period = _validate_period(plan.comparison_period, repository, "comparison_period")
    _validate_comparison_requirement(analysis_type, period, comparison_period)
    _validate_filters(plan, repository)

    required = REQUIRED_DATASETS[analysis_type]
    _validate_datasets_available(required, repository)

    validated = AnalysisPlan(
        analysis_type=analysis_type,
        metric=metric,
        dimensions=dimensions,
        period=period,
        comparison_period=comparison_period,
        department=plan.department,
        process=plan.process,
        vendor_id=plan.vendor_id,
        top_n=min(max(plan.top_n, 1), 50),
        required_datasets=required,
        reasoning_goal=plan.reasoning_goal
        if isinstance(plan.reasoning_goal, ReasoningGoal)
        else ReasoningGoal(plan.reasoning_goal),
        notes=plan.notes,
    )
    logger.info(
        "plan_validated",
        extra={
            "analysis_type": validated.analysis_type.value,
            "metric": validated.metric.value,
            "adjustments": len(adjustments),
        },
    )
    return ValidationResult(plan=validated, adjustments=adjustments)


# ---------------------------------------------------------------------------
# Field validators
# ---------------------------------------------------------------------------


def _validate_analysis_type(value: AnalysisType) -> AnalysisType:
    if not isinstance(value, AnalysisType):
        raise UnsupportedAnalysisError(
            f"analysis type '{value}' is not supported; allowed: "
            + ", ".join(sorted(a.value for a in AnalysisType))
        )
    if value not in ALLOWED_METRICS:
        raise UnsupportedAnalysisError(f"analysis type '{value.value}' has no validation profile")
    return value


def _validate_metric(
    analysis_type: AnalysisType,
    metric: Metric,
    strict: bool,
    adjustments: list[str],
) -> Metric:
    allowed = ALLOWED_METRICS[analysis_type]
    if metric in allowed:
        return metric
    # A wrong metric changes the analysis meaning, so this always raises.
    raise UnsupportedAnalysisError(
        f"metric '{metric.value}' is not supported for analysis type "
        f"'{analysis_type.value}'; allowed metrics: " + ", ".join(sorted(m.value for m in allowed))
    )


def _validate_dimensions(
    analysis_type: AnalysisType,
    dimensions: tuple[Dimension, ...],
    strict: bool,
    adjustments: list[str],
) -> tuple[Dimension, ...]:
    allowed = ALLOWED_DIMENSIONS[analysis_type]
    kept: list[Dimension] = []
    seen: set[Dimension] = set()
    for dimension in dimensions:
        if dimension in seen:
            continue
        seen.add(dimension)
        if dimension not in allowed:
            if strict:
                raise UnsupportedAnalysisError(
                    f"dimension '{dimension.value}' is not supported for analysis type "
                    f"'{analysis_type.value}'; allowed: "
                    + (", ".join(sorted(d.value for d in allowed)) or "none")
                )
            adjustments.append(
                f"dropped unsupported dimension '{dimension.value}' for '{analysis_type.value}'"
            )
            continue
        kept.append(dimension)
    if len(kept) > MAX_DIMENSIONS:
        adjustments.append(
            f"limited dimensions to the first {MAX_DIMENSIONS} of {len(kept)} requested"
        )
        kept = kept[:MAX_DIMENSIONS]
    return tuple(kept)


def _validate_period(period: str | None, repository: DatasetRepository, label: str) -> str | None:
    if period is None:
        return None
    months = months_covered_by(period)
    if months is None:
        raise InvalidRequestError(
            f"{label} '{period}' is not a recognised period; use a year (2025), a month "
            "(2025-03), a quarter (2025-Q3), or a range (2025-01 to 2025-03)"
        )
    available = repository.available_period_set
    if not any(month in available for month in months):
        raise InvalidRequestError(
            f"{label} '{period}' contains no data in the loaded dataset; available periods "
            f"range from {repository.available_periods[0]} to {repository.available_periods[-1]}"
        )
    return period


def _validate_comparison_requirement(
    analysis_type: AnalysisType, period: str | None, comparison_period: str | None
) -> None:
    if analysis_type in _COMPARISON_TYPES and period is None and comparison_period is None:
        # The analysis service resolves a default comparison window, so this is
        # allowed; the requirement is enforced at execution time against the
        # resolved periods. Nothing to reject here.
        return
    if comparison_period is not None and period is not None and period == comparison_period:
        raise InvalidRequestError("period and comparison_period must differ")


def _validate_filters(plan: AnalysisPlan, repository: DatasetRepository) -> None:
    """Reject filter values that are not known entities in the dataset."""
    if plan.department and plan.department not in repository.department_ids:
        raise InvalidRequestError(
            f"unknown department '{plan.department}'; known departments: "
            + ", ".join(repository.department_ids)
        )
    if plan.process and plan.process not in repository.process_names:
        raise InvalidRequestError(
            f"unknown process '{plan.process}'; known processes: "
            + ", ".join(repository.process_names[:8])
            + ("..." if len(repository.process_names) > 8 else "")
        )
    if plan.vendor_id and plan.vendor_id not in repository.vendor_ids:
        raise InvalidRequestError(
            f"unknown vendor '{plan.vendor_id}'; known vendors: " + ", ".join(repository.vendor_ids)
        )


def _validate_datasets_available(
    required: tuple[DatasetId, ...], repository: DatasetRepository
) -> None:
    available = set(repository.available_datasets())
    missing = [d.value for d in required if d not in available]
    if missing:
        raise InvalidRequestError(
            f"this analysis requires datasets that are not loaded: {', '.join(missing)}"
        )


def supported_metrics_for(analysis_type: AnalysisType) -> list[str]:
    return sorted(m.value for m in ALLOWED_METRICS.get(analysis_type, frozenset()))


def supported_dimensions_for(analysis_type: AnalysisType) -> list[str]:
    return sorted(d.value for d in ALLOWED_DIMENSIONS.get(analysis_type, frozenset()))


__all__ = [
    "ALLOWED_DIMENSIONS",
    "ALLOWED_METRICS",
    "REQUIRED_DATASETS",
    "ValidationResult",
    "supported_dimensions_for",
    "supported_metrics_for",
    "validate_plan",
]
