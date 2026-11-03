"""Schemas for health, dataset catalogue and insight envelopes."""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import ReasoningGoal
from app.schemas.ai import KeyFinding, OperationalInsight, Recommendation
from app.schemas.analysis import (
    KPI,
    AnalysisListEntry,
    AnalysisPlan,
    AnalysisResult,
    EvidenceItem,
    Finding,
)


class HealthStatus(BaseModel):
    """Service and dataset readiness."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str = Field(pattern=r"^(ok|degraded)$")
    service: str
    version: str
    environment: str
    timestamp: dt.datetime
    datasets_loaded: int = Field(ge=0)
    datasets_available: int = Field(ge=0)
    llm_configured: bool
    version_info: dict[str, str] = Field(default_factory=dict)


class DatasetCatalogue(BaseModel):
    """Response for ``GET /datasets``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    count: int = Field(ge=0)
    datasets: tuple[object, ...]


class InsightEnvelope(BaseModel):
    """Combined deterministic result plus AI interpretation."""

    model_config = ConfigDict(extra="forbid")

    analysis_id: str
    question: str = Field(min_length=3, max_length=600)
    plan: AnalysisPlan
    result: AnalysisResult
    insight: OperationalInsight
    created_at: dt.datetime


class GroundingSummary(BaseModel):
    """What the evidence validator found in the generated narrative.

    Returned so a reader, and the tests, can see that the grounding check ran
    and what it concluded, rather than having to take it on trust.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    numbers_checked: int = Field(default=0, ge=0)
    matched: int = Field(default=0, ge=0)
    matched_after_rounding: int = Field(default=0, ge=0)
    unsupported: int = Field(default=0, ge=0)
    grounded: bool = True
    issues: tuple[str, ...] = ()


class QuestionResponse(BaseModel):
    """Complete response envelope for ``POST /questions``.

    Everything the request produced is in one object, flat rather than nested, so
    the contract is legible from the response alone:

    * ``question`` - what was asked;
    * ``plan`` - what the planner decided to run, after validation;
    * ``findings`` / ``evidence`` / ``kpis`` - the deterministic output, which is
      the authoritative part of the response;
    * ``answer`` / ``key_findings`` / ``recommendations`` - the narrative layer,
      which may only restate the evidence;
    * ``caveats`` - the limits on everything above;
    * ``grounding`` - the result of checking the narrative's numbers;
    * ``analysis_id`` - the handle for ``GET /analysis/{id}``.
    """

    model_config = ConfigDict(extra="forbid")

    analysis_id: str
    question: str = Field(min_length=3, max_length=600)
    plan: AnalysisPlan
    kpis: tuple[KPI, ...] = ()
    findings: tuple[Finding, ...] = ()
    evidence: tuple[EvidenceItem, ...] = ()
    answer: str = Field(min_length=1, max_length=2_000)
    key_findings: tuple[KeyFinding, ...] = ()
    recommendations: tuple[Recommendation, ...] = ()
    caveats: tuple[str, ...] = ()
    reasoning_goal: ReasoningGoal = ReasoningGoal.SUMMARIZE_PERFORMANCE
    generated_by: str = Field(pattern=r"^(llm|deterministic_fallback)$")
    model: str | None = Field(default=None, max_length=128)
    unsupported_claims_detected: bool = False
    grounding: GroundingSummary = Field(default_factory=GroundingSummary)
    created_at: dt.datetime
    row_count_examined: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0, ge=0.0)
    status: str = Field(default="completed", pattern=r"^(completed|degraded|failed)$")


class AnalysisIndexEntry(AnalysisListEntry):
    """Analysis list entry including whether an insight has been produced."""


class DashboardResponse(BaseModel):
    """Pre-computed operational overview used by the dashboard."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    generated_at: dt.datetime
    period_start: str | None = None
    period_end: str | None = None
    kpis: tuple[object, ...] = ()
    throughput_trend: tuple[object, ...] = ()
    cycle_time_trend: tuple[object, ...] = ()
    sla_trend: tuple[object, ...] = ()
    department_summary: tuple[object, ...] = ()
    bottlenecks: tuple[object, ...] = ()
    vendor_watchlist: tuple[object, ...] = ()
    anomalies: tuple[object, ...] = ()
