"""Natural-language question endpoint: ``POST /questions``.

Accepts a question, runs the full pipeline, and returns the complete envelope:
question, plan, KPIs, findings, evidence, answer, key findings, recommendations,
caveats and the analysis id.

The route itself does almost nothing: it enforces the input length limit and
delegates. Stage ordering, plan validation, unsupported-question rejection and
the provider-error fallback all live in
:class:`~app.ai.pipeline.QuestionPipeline`, so the same guarantees apply whether
a question arrives over HTTP, from a test, or from the benchmark harness.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.ai.pipeline import QuestionPipeline
from app.core.config import get_settings
from app.schemas.analysis import QuestionRequest
from app.schemas.reports import QuestionResponse

router = APIRouter()


@router.post("/questions", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest) -> QuestionResponse:
    """Answer one operational question with evidence-backed findings."""
    settings = get_settings()

    if len(request.question) > settings.max_question_length:
        from app.utils.errors import InvalidRequestError

        raise InvalidRequestError(
            f"question exceeds maximum length of {settings.max_question_length} characters"
        )

    pipeline = QuestionPipeline()
    return await pipeline.answer(request.question)
