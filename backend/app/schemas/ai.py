"""Schemas for AI-layer inputs and outputs.

Design rule: **every LLM output is untrusted input**. The models in this module
are strict (``extra="forbid"``, bounded strings, finite numbers only) and the AI
services re-validate whatever comes back from the provider before it can reach
the API response or the database-equivalent (in-memory store).

The LLM is never asked to produce a number that is not already present in the
evidence package. ``LLMInsight`` therefore carries no free-form metric values:
it references evidence by id.
"""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.enums import ReasoningGoal, Severity


class QuestionIntent(BaseModel):
    """Structured output of question interpretation.

    Produced by the question interpreter. It describes *what the user wants to
    know*, never the answer itself.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=3, max_length=600)
    analysis_type: str = Field(min_length=1, max_length=64)
    metric: str = Field(min_length=1, max_length=64)
    dimensions: tuple[str, ...] = ()
    period: str | None = Field(default=None, max_length=32)
    comparison_period: str | None = Field(default=None, max_length=32)
    department: str | None = Field(default=None, max_length=64)
    process: str | None = Field(default=None, max_length=96)
    vendor_id: str | None = Field(default=None, max_length=32)
    reasoning_goal: ReasoningGoal = ReasoningGoal.SUMMARIZE_PERFORMANCE
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    rationale: str | None = Field(default=None, max_length=600)
    unsupported_terms: tuple[str, ...] = ()

    @field_validator("dimensions")
    @classmethod
    def _bound_dimensions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) > 6:
            raise ValueError("at most 6 dimensions may be requested")
        return value


class AnalysisPlanProposal(BaseModel):
    """Structured output of the analysis planner.

    The planner's job is to propose a *supported* plan. The service layer then
    validates the proposal against the controlled vocabularies and the actual
    dataset contents before anything executes.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    analysis_type: str = Field(min_length=1, max_length=64)
    metric: str = Field(min_length=1, max_length=64)
    dimensions: tuple[str, ...] = ()
    period: str | None = Field(default=None, max_length=32)
    comparison_period: str | None = Field(default=None, max_length=32)
    department: str | None = Field(default=None, max_length=64)
    process: str | None = Field(default=None, max_length=96)
    vendor_id: str | None = Field(default=None, max_length=32)
    required_datasets: tuple[str, ...] = ()
    reasoning_goal: ReasoningGoal = ReasoningGoal.SUMMARIZE_PERFORMANCE
    planning_notes: str | None = Field(default=None, max_length=600)


class EvidenceSummary(BaseModel):
    """Condensed evidence handed to the explanation layer.

    This is intentionally a *reduction* of the full evidence package: the LLM
    only needs the headline numbers, and a smaller payload reduces the chance of
    it anchoring on an irrelevant row.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str
    metric: str
    value: float | None
    comparison_value: float | None
    absolute_change: float | None
    percentage_change: float | None
    unit: str
    dimension: str | None
    period: str | None
    source_dataset: str
    calculation_description: str


class ExplanationRequest(BaseModel):
    """The full prompt payload for the explanation generator."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=3, max_length=600)
    reasoning_goal: ReasoningGoal
    evidence: tuple[EvidenceSummary, ...]
    deterministic_findings: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()


class KeyFinding(BaseModel):
    """One evidence-grounded insight claim."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    statement: str = Field(min_length=1, max_length=400)
    severity: Severity = Severity.MEDIUM
    evidence_ids: tuple[str, ...] = ()
    metric: str | None = Field(default=None, max_length=128)


class Recommendation(BaseModel):
    """An evidence-grounded suggested action."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: str = Field(min_length=1, max_length=400)
    rationale: str = Field(min_length=1, max_length=600)
    evidence_ids: tuple[str, ...] = ()
    priority: Severity = Severity.MEDIUM
    expected_metric_effect: str | None = Field(default=None, max_length=200)


class OperationalInsight(BaseModel):
    """The structured insight returned to the UI.

    ``answer`` is the only free-form narrative. Every number it can reference is
    available in ``evidence``; the answer must not introduce new figures.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    answer: str = Field(min_length=1, max_length=2_000)
    key_findings: tuple[KeyFinding, ...] = ()
    recommendations: tuple[Recommendation, ...] = ()
    caveats: tuple[str, ...] = ()
    generated_by: str = Field(default="llm", pattern=r"^(llm|deterministic_fallback)$")
    model: str | None = Field(default=None, max_length=128)
    unsupported_claims_detected: bool = False

    @field_validator("key_findings", "recommendations", "caveats")
    @classmethod
    def _bound_collections(cls, value: tuple[object, ...]) -> tuple[object, ...]:
        if len(value) > 12:
            raise ValueError("at most 12 items allowed in insight collections")
        return value


class LLMCallTrace(BaseModel):
    """Non-sensitive record of an LLM interaction, safe to log and return."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: str
    model: str
    duration_ms: float = Field(ge=0.0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    succeeded: bool
    error_code: str | None = Field(default=None, max_length=64)
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))
