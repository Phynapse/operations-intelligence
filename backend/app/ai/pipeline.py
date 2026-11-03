"""The question-to-insight pipeline.

This module is the single place that runs the stages in order, so the ordering
guarantees are stated once rather than reimplemented per route:

    Question Interpreter -> Analysis Planner -> Plan Validator
        -> Deterministic Analytics -> Evidence Builder -> Evidence Validation
        -> Grounded LLM Explanation -> Structured Insight

Two rules are enforced here rather than trusted to the individual stages:

**A plan cannot reach the analytics layer unvalidated.** Whatever the planner
produced - deterministic router or LLM - goes through
:func:`app.ai.plan_validator.validate_plan` immediately before execution.

**An unsupported question is rejected, not approximated.** If the interpreter
reports terms the platform cannot serve, the pipeline raises
:class:`~app.utils.errors.UnsupportedAnalysisError` with the reasons rather than
running a default analysis and returning a confident answer to a question it did
not answer.

Every stage degrades to a deterministic path rather than failing the request.
The analytics layer is the source of truth and it is local, so a missing LLM
costs narrative quality but never correctness.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.ai.analysis_planner import plan_from_intent
from app.ai.explanation_generator import generate_insight, validate_insight_grounding
from app.ai.plan_validator import ValidationResult, validate_plan
from app.ai.providers.openai_compatible import LLMProvider, create_provider
from app.ai.question_interpreter import interpret_question
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.data.repository import DatasetRepository, get_repository
from app.models.enums import AnalysisType
from app.schemas.ai import OperationalInsight, QuestionIntent
from app.schemas.analysis import AnalysisPlan, AnalysisResult
from app.schemas.reports import GroundingSummary, QuestionResponse
from app.services.analysis_service import AnalysisService, get_analysis_service
from app.services.evidence_service import EvidenceService
from app.utils.errors import UnsupportedAnalysisError

logger = get_logger(__name__)

#: Analysis types that must have two windows before they can run.
_TWO_WINDOW_TYPES: frozenset[AnalysisType] = frozenset(
    {AnalysisType.PERIOD_COMPARISON, AnalysisType.CONTRIBUTION_ANALYSIS}
)


class QuestionPipeline:
    """Runs the full question flow and returns the complete response envelope."""

    def __init__(
        self,
        *,
        analysis_service: AnalysisService | None = None,
        evidence_service: EvidenceService | None = None,
        repository: DatasetRepository | None = None,
        settings: Settings | None = None,
        provider: LLMProvider | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._repository = repository or get_repository()
        self._analysis = analysis_service or get_analysis_service()
        self._evidence = evidence_service or EvidenceService(self._settings)
        self._provider = provider
        self._own_provider: LLMProvider | None = None

    # -- stages ---------------------------------------------------------
    def interpret(self, question: str) -> QuestionIntent:
        """Stage 1: natural language to a structured intent."""
        return interpret_question(question)

    def plan(self, intent: QuestionIntent) -> AnalysisPlan:
        """Stages 2 and 3: propose a plan, then validate it.

        The validator is not optional here. Even though the deterministic
        interpreter already works from controlled vocabularies, running the
        validator means an LLM-produced plan would be held to exactly the same
        rule.
        """
        proposed = plan_from_intent(
            intent,
            self._repository.available_periods,
            self._repository.available_period_set,
        )
        result: ValidationResult = validate_plan(proposed, self._repository, strict=False)
        if result.was_adjusted:
            logger.info(
                "plan_adjusted",
                extra={"adjustments": result.adjustments[:5]},
            )
        return result.plan

    def analyze(self, plan: AnalysisPlan, question: str) -> AnalysisResult:
        """Stage 4: deterministic analytics. No LLM is involved here."""
        return self._analysis.run(plan, question=question)

    async def explain(self, result: AnalysisResult) -> OperationalInsight:
        """Stages 5 to 7: evidence package, grounding check, narrative."""
        provider = self._provider
        if provider is None and self._settings.llm_configured:
            provider = self._provider = create_provider()
        return await generate_insight(
            result,
            provider=provider,
            settings=self._settings,
            evidence_service=self._evidence,
        )

    # -- orchestration --------------------------------------------------
    async def answer(self, question: str) -> QuestionResponse:
        """Run every stage in order and assemble the response envelope."""
        intent = self.interpret(question)
        self._reject_if_unsupported(intent)

        plan = self.plan(intent)
        self._ensure_comparisons(plan)

        result = self.analyze(plan, question)
        insight = await self.explain(result)

        package = self._evidence.build(result)
        report = validate_insight_grounding(insight, package)
        grounding = GroundingSummary(
            numbers_checked=report.checked,
            matched=report.matched,
            matched_after_rounding=report.rounded,
            unsupported=report.unsupported_count,
            grounded=report.is_grounded,
            issues=tuple(str(issue) for issue in report.unsupported[:5]),
        )

        logger.info(
            "question_answered",
            extra={
                "analysis_id": result.analysis_id,
                "analysis_type": plan.analysis_type.value,
                "generated_by": insight.generated_by,
                "evidence_items": len(result.evidence),
                "grounded": report.is_grounded,
            },
        )

        return QuestionResponse(
            analysis_id=result.analysis_id,
            question=question,
            plan=plan,
            kpis=result.kpis,
            findings=result.findings,
            evidence=result.evidence,
            answer=insight.answer,
            key_findings=insight.key_findings,
            recommendations=insight.recommendations,
            caveats=insight.caveats,
            reasoning_goal=plan.reasoning_goal,
            generated_by=insight.generated_by,
            model=insight.model,
            unsupported_claims_detected=insight.unsupported_claims_detected,
            grounding=grounding,
            created_at=result.created_at,
            row_count_examined=result.row_count_examined,
            duration_ms=result.duration_ms,
            status=result.status,
        )

    # -- guards ---------------------------------------------------------
    @staticmethod
    def _reject_if_unsupported(intent: QuestionIntent) -> None:
        if not intent.unsupported_terms:
            return
        raise UnsupportedAnalysisError(
            "this platform answers questions about historical operational data "
            "(throughput, cycle time, queue time, backlog, SLA compliance, vendors, "
            "departments and processes), and cannot: " + "; ".join(intent.unsupported_terms)
        )

    def _ensure_comparisons(self, plan: AnalysisPlan) -> None:
        """Fill in a comparison window for two-window analyses.

        ``AnalysisPlan`` requires a ``comparison_period`` for these types, which
        means it has to be resolved before the plan can be constructed. The
        analysis service supplies the immediately preceding window of the same
        size, and records a caveat when it does so.
        """
        if plan.analysis_type not in _TWO_WINDOW_TYPES:
            return
        if plan.comparison_period:
            return
        raise UnsupportedAnalysisError(
            f"analysis type '{plan.analysis_type.value}' compares two windows, and only one "
            "could be resolved from this question; name both periods explicitly"
        )


@dataclass(slots=True)
class PipelineOutcome:
    """Result of a pipeline run, retained for inspection and testing."""

    intent: QuestionIntent
    plan: AnalysisPlan
    result: AnalysisResult
    response: QuestionResponse
    adjustments: list[str] = field(default_factory=list)


__all__ = ["PipelineOutcome", "QuestionPipeline"]
