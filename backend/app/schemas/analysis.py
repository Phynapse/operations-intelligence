"""Schemas for analysis requests, plans, evidence and results.

This is the contract between the deterministic analytics layer and the AI
explanation layer. The critical invariant enforced here is that
:class:`EvidenceItem` values are *provenance-bearing*: every number handed to an
LLM must state which dataset it came from and how it was calculated.
"""

from __future__ import annotations

import datetime as dt
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.enums import (
    COMPARISON_ANALYSIS_TYPES,
    AnalysisType,
    DatasetId,
    Dimension,
    Metric,
    ReasoningGoal,
    Severity,
    Unit,
)

#: A period is a year, a month, a quarter, or an explicit month range. The
#: analysis service resolves a request into a range and stores that label here.
PERIOD_PATTERN = (
    r"^(?:\d{4}(?:-\d{2})?(?:-Q[1-4])?|\d{4}-\d{2} to \d{4}-\d{2}|\d{4}-\d{2} to \d{4}-Q[1-4])$"
)
DIMENSION_VALUES = [d.value for d in Dimension]


class AnalysisRequest(BaseModel):
    """A request to run deterministic analysis, optionally with a question."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str | None = Field(default=None, max_length=600)
    analysis_type: AnalysisType | None = None
    metric: Metric | None = None
    dimensions: tuple[Dimension, ...] = ()
    period: str | None = Field(default=None, pattern=PERIOD_PATTERN)
    comparison_period: str | None = Field(default=None, pattern=PERIOD_PATTERN)
    department: str | None = Field(default=None, max_length=64)
    process: str | None = Field(default=None, max_length=96)
    vendor_id: str | None = Field(default=None, max_length=32)
    top_n: int = Field(default=5, ge=1, le=50)

    @field_validator("question")
    @classmethod
    def _require_meaningful_question(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.lower() in {"?", "??", "help"}:
            raise ValueError("question must be a substantive operational question")
        return stripped

    @model_validator(mode="after")
    def _require_an_intent(self) -> AnalysisRequest:
        """A request must say *something*.

        Either a natural-language question or an explicit analysis type is
        required. ``metric`` stays optional because the analysis service applies
        a documented default per analysis type.
        """
        if self.question is None and self.analysis_type is None:
            raise ValueError("either a question or an analysis_type must be supplied")
        if self.comparison_period and not self.period:
            raise ValueError("comparison_period requires period")
        return self

    @property
    def is_question_driven(self) -> bool:
        return self.question is not None


class QuestionRequest(AnalysisRequest):
    """``POST /questions`` payload. A question is mandatory here."""

    question: str = Field(min_length=3, max_length=600)


class AnalysisPlan(BaseModel):
    """A validated, executable description of the analysis to run.

    The planner may only populate values from the controlled enums. A plan that
    references an unsupported analysis type or metric is rejected before any
    data is read.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    analysis_type: AnalysisType
    metric: Metric
    dimensions: tuple[Dimension, ...] = ()
    period: str | None = Field(default=None, pattern=PERIOD_PATTERN)
    comparison_period: str | None = Field(default=None, pattern=PERIOD_PATTERN)
    department: str | None = Field(default=None, max_length=64)
    process: str | None = Field(default=None, max_length=96)
    vendor_id: str | None = Field(default=None, max_length=32)
    top_n: int = Field(default=5, ge=1, le=50)
    required_datasets: tuple[DatasetId, ...] = ()
    reasoning_goal: ReasoningGoal = ReasoningGoal.SUMMARIZE_PERFORMANCE
    notes: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _check_period_pair(self) -> AnalysisPlan:
        if self.analysis_type in COMPARISON_ANALYSIS_TYPES and not self.comparison_period:
            raise ValueError(
                f"{self.analysis_type.value} requires a comparison_period because it "
                "decomposes a change between two windows"
            )
        if self.period and self.comparison_period and self.period == self.comparison_period:
            raise ValueError("period and comparison_period must differ")
        return self


class KPI(BaseModel):
    """A single headline metric for a period and dimension."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=128)
    value: float | None = None
    unit: Unit
    period: str | None = Field(default=None, max_length=32)
    dimension: str | None = Field(default=None, max_length=96)
    source_dataset: DatasetId
    calculation: str = Field(min_length=1, max_length=400)

    @field_validator("value")
    @classmethod
    def _reject_non_finite(cls, value: float | None) -> float | None:
        """NaN must never reach a response body or an LLM prompt."""
        if value is None:
            return None
        if value != value:  # NaN
            return None
        if value in (float("inf"), float("-inf")):
            return None
        return float(value)


class EvidenceItem(BaseModel):
    """One auditable number, with the dataset and calculation that produced it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: str = Field(min_length=1, max_length=128)
    value: float | None = None
    comparison_value: float | None = None
    absolute_change: float | None = None
    percentage_change: float | None = None
    unit: Unit
    dimension: str | None = Field(default=None, max_length=96)
    period: str | None = Field(default=None, max_length=32)
    comparison_period: str | None = Field(default=None, max_length=32)
    source_dataset: DatasetId
    calculation_description: str = Field(min_length=1, max_length=400)
    severity: Severity = Severity.INFO
    sample_size: int | None = Field(default=None, ge=0)
    is_estimate: bool = False

    @field_validator("value", "comparison_value", "absolute_change", "percentage_change")
    @classmethod
    def _reject_non_finite(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return round(float(value), 4)


class Finding(BaseModel):
    """A deterministic observation, phrased without causal claims."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=200)
    detail: str = Field(min_length=1, max_length=600)
    severity: Severity = Severity.MEDIUM
    metric: str | None = Field(default=None, max_length=128)
    dimension: str | None = Field(default=None, max_length=96)
    period: str | None = Field(default=None, max_length=32)
    supporting_evidence: tuple[str, ...] = ()
    is_statistical: bool = False
    statistic: str | None = Field(default=None, max_length=200)


class AnalysisResult(BaseModel):
    """The deterministic output of an analysis run."""

    model_config = ConfigDict(extra="forbid")

    analysis_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))
    question: str | None = Field(default=None, max_length=600)
    plan: AnalysisPlan
    evidence: tuple[EvidenceItem, ...] = ()
    kpis: tuple[KPI, ...] = ()
    findings: tuple[Finding, ...] = ()
    caveats: tuple[str, ...] = ()
    row_count_examined: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0, ge=0.0)
    status: str = Field(default="completed", pattern=r"^(completed|degraded|failed)$")


class AnalysisListEntry(BaseModel):
    """Lightweight entry for the in-memory analysis index."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    analysis_id: str
    created_at: dt.datetime
    question: str | None = None
    analysis_type: AnalysisType
    metric: Metric
    status: str
    has_insight: bool = False


class ExampleQuestion(BaseModel):
    """A curated question shown in the UI to demonstrate capability."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str
    description: str
    category: str
