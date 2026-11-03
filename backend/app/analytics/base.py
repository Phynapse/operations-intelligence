"""Analytics layer base classes.

Two base types are defined here:

``AnalysisContext``
    Immutable, validated inputs for one analysis run. It carries the resolved
    period ranges, the filtered frames, and the caveats raised while resolving
    them. An analyzer never re-derives periods or re-reads the repository.

``BaseAnalyzer``
    The contract every analyzer implements. Analyzers are pure with respect to
    I/O: they receive a context and return an :class:`AnalysisResult`. They
    cannot make HTTP calls, cannot reach an LLM, and cannot execute user
    supplied code. That restriction is what makes them independently testable
    and what keeps operational arithmetic out of the AI layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import pandas as pd

from app.data.repository import DatasetRepository
from app.models.enums import AnalysisType, DatasetId, Severity, Unit
from app.schemas.analysis import KPI, AnalysisPlan, EvidenceItem, Finding
from app.utils.calculations import (
    PERCENT_PRECISION,
    absolute_change,
    percentage_change,
    round_or_none,
)
from app.utils.periods import PeriodRange


@dataclass(frozen=True, slots=True)
class PeriodSlice:
    """A resolved window of data, with the frame it was taken from."""

    period_range: PeriodRange
    frame: pd.DataFrame
    #: month key -> already-filtered sub-frame, so analyzers do not re-filter.
    by_period: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def months(self) -> tuple[str, ...]:
        return self.period_range.months

    @property
    def row_count(self) -> int:
        return len(self.frame)


@dataclass(frozen=True, slots=True)
class AnalysisContext:
    """Everything an analyzer is allowed to look at.

    ``frames`` and ``comparison_frames`` hold the *same required datasets*, each
    filtered down to the plan's filters and to one time window. ``primary`` and
    ``comparison`` are the ``operations.csv`` slices for the two windows, since
    that dataset carries the pre-grouped period buckets.
    """

    plan: AnalysisPlan
    repository: DatasetRepository
    primary: PeriodSlice | None = None
    comparison: PeriodSlice | None = None
    frames: dict[DatasetId, pd.DataFrame] = field(default_factory=dict)
    comparison_frames: dict[DatasetId, pd.DataFrame] = field(default_factory=dict)
    caveats: tuple[str, ...] = ()
    #: Minimum observations before trend or anomaly statistics are trusted.
    min_observations: int = 4

    @property
    def analysis_type(self) -> AnalysisType:
        return self.plan.analysis_type

    @property
    def top_n(self) -> int:
        return self.plan.top_n

    def frame(self, dataset_id: DatasetId, window: str = "primary") -> pd.DataFrame:
        """Filtered frame for a dataset, or an empty frame when unavailable."""
        source = self.comparison_frames if window == "comparison" else self.frames
        return source.get(dataset_id, pd.DataFrame())

    def primary_frame(self) -> pd.DataFrame:
        return self.primary.frame if self.primary is not None else pd.DataFrame()

    def comparison_frame(self) -> pd.DataFrame | None:
        return self.comparison.frame if self.comparison is not None else None

    def with_caveats(self, *caveats: str) -> AnalysisContext:
        """Return a copy with additional caveats appended."""
        clean = tuple(item for item in caveats if item)
        if not clean:
            return self
        return replace(self, caveats=self.caveats + clean)


class BaseAnalyzer(ABC):
    """Base class for every deterministic analyzer.

    Subclasses declare the analysis types and datasets they support, then
    implement :meth:`analyze`. The framework owns plan validation, so a subclass
    cannot be reached with a plan it does not support.
    """

    #: Human readable name used in evidence and findings.
    name: str = "analyzer"
    #: Analysis types this analyzer implements.
    supported_types: frozenset[AnalysisType] = frozenset()
    #: Datasets this analyzer needs.
    required_datasets: tuple[DatasetId, ...] = ()

    def supports(self, analysis_type: AnalysisType) -> bool:
        return analysis_type in self.supported_types

    @abstractmethod
    def analyze(self, context: AnalysisContext) -> AnalysisOutcome:
        """Run the deterministic calculation and return a structured outcome."""

    # -- helpers available to subclasses -------------------------------
    @staticmethod
    def describe(metric: str, calculation: str) -> str:
        """Build the standard calculation description string."""
        return f"{metric}: {calculation}"


@dataclass(slots=True)
class AnalysisOutcome:
    """Intermediate result produced by an analyzer.

    Collections are plain mutable lists so an analyzer can build up evidence and
    findings incrementally. :meth:`as_result_kwargs` converts them to the
    immutable tuples that :class:`~app.schemas.analysis.AnalysisResult` expects.
    """

    kpis: list[KPI] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    row_count_examined: int = 0

    def extend_caveats(self, *caveats: str) -> None:
        for item in caveats:
            if item and item not in self.caveats:
                self.caveats.append(item)

    def as_result_kwargs(self) -> dict[str, object]:
        return {
            "kpis": tuple(self.kpis),
            "evidence": tuple(self.evidence),
            "findings": tuple(self.findings),
            "caveats": tuple(self.caveats),
            "row_count_examined": self.row_count_examined,
        }


# ---------------------------------------------------------------------------
# Evidence construction helpers
#
# Analyzers must use these rather than constructing evidence by hand: the
# helpers guarantee that every emitted number carries a unit, a source dataset
# and a human-readable calculation, and that non-finite values become ``None``.
# ---------------------------------------------------------------------------


def build_kpi(
    *,
    name: str,
    value: float | None,
    unit: Unit,
    source_dataset: DatasetId,
    calculation: str,
    period: str | None = None,
    dimension: str | None = None,
) -> KPI:
    """Create a KPI, coercing undefined values to ``None``."""
    return KPI(
        name=name,
        value=round_or_none(value, PERCENT_PRECISION),
        unit=unit,
        period=period,
        dimension=dimension,
        source_dataset=source_dataset,
        calculation=calculation,
    )


def build_evidence(
    *,
    metric: str,
    value: float | None,
    unit: Unit,
    source_dataset: DatasetId,
    calculation_description: str,
    comparison_value: float | None = None,
    dimension: str | None = None,
    period: str | None = None,
    comparison_period: str | None = None,
    severity: Severity = Severity.INFO,
    sample_size: int | None = None,
    is_estimate: bool = False,
) -> EvidenceItem:
    """Create an evidence item, deriving the change fields deterministically."""
    change_absolute = absolute_change(value, comparison_value)
    change_percentage = percentage_change(value, comparison_value)
    return EvidenceItem(
        metric=metric,
        value=round_or_none(value),
        comparison_value=round_or_none(comparison_value),
        absolute_change=change_absolute,
        percentage_change=change_percentage,
        unit=unit,
        dimension=dimension,
        period=period,
        comparison_period=comparison_period,
        source_dataset=source_dataset,
        calculation_description=calculation_description,
        severity=severity,
        sample_size=sample_size,
        is_estimate=is_estimate,
    )


def build_finding(
    *,
    title: str,
    detail: str,
    severity: Severity = Severity.MEDIUM,
    metric: str | None = None,
    dimension: str | None = None,
    period: str | None = None,
    supporting_evidence: Sequence[str] = (),
    is_statistical: bool = False,
    statistic: str | None = None,
) -> Finding:
    return Finding(
        title=title,
        detail=detail,
        severity=severity,
        metric=metric,
        dimension=dimension,
        period=period,
        supporting_evidence=tuple(supporting_evidence),
        is_statistical=is_statistical,
        statistic=statistic,
    )


def empty_slice() -> pd.DataFrame:
    """A typed empty frame, used when a comparison period has no data."""
    return pd.DataFrame()
