"""Analysis execution service.

Responsibilities:

* build a validated :class:`AnalysisPlan` from a request, a question intent or an
  explicit analysis type;
* resolve requested periods against the months that actually exist, and supply a
  default comparison window when the question implies one;
* build the :class:`~app.analytics.base.AnalysisContext` with filtered frames;
* run the registered analyzer and return a structured
  :class:`~app.schemas.analysis.AnalysisResult`;
* keep an in-memory index of runs so ``GET /analysis/{id}`` works.

The service performs **no LLM calls** and no arithmetic of its own beyond plan
resolution. That separation is what lets the same code path serve both
``POST /analysis`` (explicit plan) and ``POST /questions`` (LLM-planned plan).
"""

from __future__ import annotations

import datetime as dt
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Sequence

import pandas as pd

from app.analytics.base import AnalysisContext, PeriodSlice
from app.analytics.registry import (
    default_reasoning_goal_for,
    get_analyzer,
    required_datasets_for,
)
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.data.repository import (
    COLUMN_ALIASES,
    FILTERABLE_COLUMNS,
    DatasetRepository,
    FilterSpec,
    get_repository,
)
from app.models.enums import (
    COMPARISON_ANALYSIS_TYPES,
    AnalysisType,
    DatasetId,
    Dimension,
    Metric,
)
from app.schemas.ai import AnalysisPlanProposal, QuestionIntent
from app.schemas.analysis import AnalysisListEntry, AnalysisPlan, AnalysisRequest, AnalysisResult
from app.utils.errors import (
    AnalysisNotFoundError,
    InvalidRequestError,
    UnsupportedAnalysisError,
)
from app.utils.periods import PeriodRange, preceding_period, resolve_period_range

logger = get_logger(__name__)

#: Maximum number of analysis runs retained in memory.
MAX_RETAINED_RUNS = 200

#: Analysis types that always need a second window of data.
COMPARISON_TYPES: frozenset[AnalysisType] = COMPARISON_ANALYSIS_TYPES

#: Default metric per analysis type, used when a request omits one.
DEFAULT_METRIC: dict[AnalysisType, Metric] = {
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
    AnalysisType.ANOMALY_ANALYSIS: Metric.AVERAGE_CYCLE_TIME,
    AnalysisType.PROCESS_ANALYSIS: Metric.AVERAGE_CYCLE_TIME,
    AnalysisType.CONTRIBUTION_ANALYSIS: Metric.AVERAGE_CYCLE_TIME,
}


class AnalysisService:
    """Executes validated analysis plans against the loaded datasets."""

    def __init__(
        self,
        repository: DatasetRepository | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._repository = repository or get_repository()
        self._lock = threading.Lock()
        self._runs: OrderedDict[str, AnalysisResult] = OrderedDict()

    # -- plan construction ---------------------------------------------
    def plan_from_request(self, request: AnalysisRequest) -> AnalysisPlan:
        """Build a plan from an explicit request (``POST /analysis``)."""
        if request.analysis_type is None:
            raise InvalidRequestError("analysis_type is required when no question is supplied")
        analysis_type = request.analysis_type
        metric = request.metric or DEFAULT_METRIC.get(analysis_type)
        if metric is None:
            raise InvalidRequestError(
                f"metric is required for analysis type '{analysis_type.value}'"
            )
        return self._finalise_plan(
            analysis_type=analysis_type,
            metric=metric,
            dimensions=request.dimensions,
            period=request.period,
            comparison_period=request.comparison_period,
            department=request.department,
            process=request.process,
            vendor_id=request.vendor_id,
            top_n=request.top_n,
            reasoning_goal=default_reasoning_goal_for(analysis_type),
            notes=None,
        )

    def plan_from_intent(self, intent: QuestionIntent) -> AnalysisPlan:
        """Build a plan from an LLM-produced intent (``POST /questions``)."""
        return self._finalise_plan(
            analysis_type=AnalysisType(intent.analysis_type),
            metric=Metric(intent.metric),
            dimensions=tuple(Dimension(item) for item in intent.dimensions),
            period=intent.period,
            comparison_period=intent.comparison_period,
            department=intent.department,
            process=intent.process,
            vendor_id=intent.vendor_id,
            top_n=5,
            reasoning_goal=intent.reasoning_goal,
            notes=None,
        )

    def plan_from_proposal(self, proposal: AnalysisPlanProposal) -> AnalysisPlan:
        """Build a plan from an LLM-produced planner proposal."""
        analysis_type = AnalysisType(proposal.analysis_type)
        return self._finalise_plan(
            analysis_type=analysis_type,
            metric=Metric(proposal.metric),
            dimensions=tuple(Dimension(item) for item in proposal.dimensions),
            period=proposal.period,
            comparison_period=proposal.comparison_period,
            department=proposal.department,
            process=proposal.process,
            vendor_id=proposal.vendor_id,
            top_n=5,
            reasoning_goal=proposal.reasoning_goal,
            notes=proposal.planning_notes,
        )

    def _finalise_plan(
        self,
        *,
        analysis_type: AnalysisType,
        metric: Metric,
        dimensions: tuple[Dimension, ...],
        period: str | None,
        comparison_period: str | None,
        department: str | None,
        process: str | None,
        vendor_id: str | None,
        top_n: int,
        reasoning_goal: object,
        notes: str | None,
    ) -> AnalysisPlan:
        """Validate every plan field against the enums and dataset contents."""
        required = required_datasets_for(analysis_type)
        resolved_period, resolved_comparison, caveats = self._resolve_periods(
            analysis_type, period, comparison_period
        )
        self._validate_filters(analysis_type, department, process, vendor_id)

        return AnalysisPlan(
            analysis_type=analysis_type,
            metric=metric,
            dimensions=dimensions or (),
            period=resolved_period,
            comparison_period=resolved_comparison,
            department=department,
            process=process,
            vendor_id=vendor_id,
            top_n=min(top_n, self._settings.evidence_top_n * 4),
            required_datasets=required,
            reasoning_goal=reasoning_goal,  # type: ignore[arg-type]
            notes=(notes or "; ".join(caveats))[:500] or None,
        )

    def _resolve_periods(
        self,
        analysis_type: AnalysisType,
        period: str | None,
        comparison_period: str | None,
    ) -> tuple[str | None, str | None, list[str]]:
        available = self._repository.available_period_set
        primary, comparison, caveats = resolve_period_range(period, available, comparison_period)
        needs_comparison = analysis_type in COMPARISON_TYPES or comparison_period is not None
        if needs_comparison and comparison is None and primary is not None:
            comparison = preceding_period(primary, available)
            if comparison is None:
                caveats.append(
                    "no earlier period is available in this dataset, so a comparison could not "
                    "be produced; only the requested window is reported"
                )
            else:
                caveats.append(
                    f"no comparison period was given, so {comparison.label} (the immediately "
                    f"preceding window) was used"
                )
        return (
            _period_label(primary),
            _period_label(comparison),
            caveats,
        )

    def _validate_filters(
        self,
        analysis_type: AnalysisType,
        department: str | None,
        process: str | None,
        vendor_id: str | None,
    ) -> None:
        """Reject filter values that do not exist in the loaded datasets.

        A filter naming an unknown department, process or vendor is an error
        rather than a silently empty result, so the caller learns the request was
        wrong instead of reading a zero.
        """
        required = set(required_datasets_for(analysis_type))
        if department and DatasetId.DEPARTMENTS not in required:
            return
        if process and DatasetId.OPERATIONS not in required:
            return
        if vendor_id and DatasetId.OPERATIONS not in required:
            return

        if department and department not in self._repository.department_ids:
            raise InvalidRequestError(
                f"unknown department '{department}'; known departments: "
                + ", ".join(self._repository.department_ids)
            )
        if process and process not in self._repository.process_names:
            raise InvalidRequestError(
                f"unknown process '{process}'; known processes: "
                + ", ".join(self._repository.process_names[:10])
                + ("..." if len(self._repository.process_names) > 10 else "")
            )
        if vendor_id and vendor_id not in self._repository.vendor_ids:
            raise InvalidRequestError(
                f"unknown vendor '{vendor_id}'; known vendors: "
                + ", ".join(self._repository.vendor_ids)
            )

    # -- execution ------------------------------------------------------
    def run(self, plan: AnalysisPlan, question: str | None = None) -> AnalysisResult:
        """Execute a validated plan and persist the result in memory."""
        analyzer = get_analyzer(plan.analysis_type)
        started = time.perf_counter()
        context = self.build_context(plan)
        outcome = analyzer.analyze(context)
        duration_ms = (time.perf_counter() - started) * 1000.0

        result = AnalysisResult(
            analysis_id=uuid.uuid4().hex,
            created_at=dt.datetime.now(dt.UTC),
            question=question,
            plan=plan,
            evidence=tuple(outcome.evidence),
            kpis=tuple(outcome.kpis),
            findings=tuple(outcome.findings),
            caveats=tuple(dict.fromkeys(list(context.caveats) + list(outcome.caveats))),
            row_count_examined=outcome.row_count_examined,
            duration_ms=round(duration_ms, 3),
            status="completed",
        )
        self._store(result)
        logger.info(
            "analysis_completed",
            extra={
                "analysis_id": result.analysis_id,
                "analysis_type": plan.analysis_type.value,
                "metric": plan.metric.value,
                "period": plan.period,
                "evidence_items": len(result.evidence),
                "findings": len(result.findings),
                "duration_ms": result.duration_ms,
            },
        )
        return result

    def build_context(self, plan: AnalysisPlan) -> AnalysisContext:
        """Resolve periods, filters and frames for a plan."""
        available = self._repository.available_period_set
        primary_range, comparison_range, caveats = resolve_period_range(
            plan.period, available, plan.comparison_period
        )
        if primary_range is None:
            raise InvalidRequestError("the dataset contains no usable periods for this request")

        required = set(required_datasets_for(plan.analysis_type))
        primary_filters = self._filters_for(plan, primary_range)
        comparison_filters = (
            self._filters_for(plan, comparison_range) if comparison_range is not None else None
        )

        primary_slice = self._slice(DatasetId.OPERATIONS, primary_filters, primary_range)
        comparison_slice = (
            self._slice(DatasetId.OPERATIONS, comparison_filters, comparison_range)
            if comparison_filters is not None
            else None
        )

        frames: dict[DatasetId, pd.DataFrame] = {DatasetId.OPERATIONS: primary_slice.frame}
        comparison_frames: dict[DatasetId, pd.DataFrame] = {
            DatasetId.OPERATIONS: comparison_slice.frame
            if comparison_slice is not None
            else pd.DataFrame()
        }
        for dataset_id in sorted(required, key=lambda item: item.value):
            if dataset_id is DatasetId.OPERATIONS:
                continue
            applicable = self._applicable_filters(dataset_id, primary_filters or ())
            comparison_applicable = self._applicable_filters(dataset_id, comparison_filters or ())
            try:
                frames[dataset_id] = self._repository.apply_filters(
                    self._repository.frame(dataset_id), dataset_id, applicable
                )
                if comparison_filters is None or not comparison_applicable:
                    comparison_frames[dataset_id] = pd.DataFrame()
                else:
                    comparison_frames[dataset_id] = self._repository.apply_filters(
                        self._repository.frame(dataset_id), dataset_id, comparison_applicable
                    )
            except Exception as exc:
                logger.warning(
                    "dataset_filter_failed",
                    extra={"dataset_id": dataset_id.value, "error": type(exc).__name__},
                )

        return AnalysisContext(
            plan=plan,
            repository=self._repository,
            primary=primary_slice,
            comparison=comparison_slice,
            frames=frames,
            comparison_frames=comparison_frames,
            caveats=tuple(caveats),
            min_observations=self._settings.min_observations_for_trend,
        )

    def _applicable_filters(
        self, dataset_id: DatasetId, filters: Sequence[FilterSpec]
    ) -> list[FilterSpec]:
        """Drop filters a dataset cannot honour.

        Reference datasets (``departments.csv``, ``vendors.csv``) have no time
        column, so a period filter is meaningless for them. Passing it anyway
        would raise inside the repository, so the filter is removed here and the
        dataset is loaded whole.
        """
        if not filters:
            return []
        allowed = FILTERABLE_COLUMNS.get(dataset_id, frozenset())
        kept: list[FilterSpec] = []
        for spec in filters:
            column = COLUMN_ALIASES.get(spec.column, spec.column)
            if column in allowed:
                kept.append(spec)
        return kept

    def _filters_for(
        self, plan: AnalysisPlan, period_range: PeriodRange | None
    ) -> list[FilterSpec]:
        filters: list[FilterSpec] = []
        if period_range is not None:
            filters.append(
                FilterSpec(column="period", values=tuple(period_range.months), include_nulls=True)
            )
        if plan.department:
            filters.append(FilterSpec(column="department", values=(plan.department,)))
        if plan.process:
            filters.append(FilterSpec(column="process", values=(plan.process,)))
        if plan.vendor_id:
            filters.append(
                FilterSpec(column="vendor_id", values=(plan.vendor_id,), include_nulls=True)
            )
        return filters

    def _slice(
        self, dataset_id: DatasetId, filters: list[FilterSpec], period_range: PeriodRange | None
    ) -> PeriodSlice:
        frame = self._repository.frame(dataset_id)
        filtered = self._repository.apply_filters(frame, dataset_id, filters)
        months = period_range.months if period_range is not None else ()
        by_period = {
            month: filtered.loc[filtered["period"] == month]
            for month in months
            if "period" in filtered.columns
        }
        return PeriodSlice(
            period_range=period_range or PeriodRange("", "", ""),
            frame=filtered,
            by_period=by_period,
        )

    # -- in-memory index -------------------------------------------------
    def _store(self, result: AnalysisResult) -> None:
        with self._lock:
            self._runs[result.analysis_id] = result
            while len(self._runs) > MAX_RETAINED_RUNS:
                self._runs.popitem(last=False)

    def get_result(self, analysis_id: str) -> AnalysisResult:
        with self._lock:
            result = self._runs.get(analysis_id)
        if result is None:
            raise AnalysisNotFoundError(
                f"no analysis with id '{analysis_id}' is retained in memory; analysis results "
                "are not persisted between requests"
            )
        return result

    def list_runs(self) -> list[AnalysisListEntry]:
        with self._lock:
            items = list(self._runs.values())
        return [
            AnalysisListEntry(
                analysis_id=item.analysis_id,
                created_at=item.created_at,
                question=item.question,
                analysis_type=item.plan.analysis_type,
                metric=item.plan.metric,
                status=item.status,
            )
            for item in reversed(items)
        ]

    def delete(self, analysis_id: str) -> None:
        """Remove a single analysis run from the in-memory index."""
        with self._lock:
            if analysis_id not in self._runs:
                raise AnalysisNotFoundError(
                    f"no analysis with id '{analysis_id}' is retained in memory"
                )
            del self._runs[analysis_id]

    def clear(self) -> None:
        with self._lock:
            self._runs.clear()

    @property
    def repository(self) -> DatasetRepository:
        return self._repository


def _period_label(period_range: PeriodRange | None) -> str | None:
    if period_range is None:
        return None
    if period_range.label and " to " in period_range.label:
        return period_range.label
    if period_range.months:
        if period_range.month_count == 1:
            return period_range.months[0]
        return f"{period_range.months[0]} to {period_range.months[-1]}"
    return None


_service: AnalysisService | None = None


def get_analysis_service() -> AnalysisService:
    """Process-wide analysis service singleton."""
    global _service
    if _service is None:
        _service = AnalysisService()
    return _service


def reset_analysis_service() -> None:
    """Drop the singleton (used by tests)."""
    global _service
    _service = None


__all__ = [
    "COMPARISON_TYPES",
    "DEFAULT_METRIC",
    "AnalysisService",
    "UnsupportedAnalysisError",
    "get_analysis_service",
    "reset_analysis_service",
]
