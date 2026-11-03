"""Explicit analysis endpoint: ``POST /analysis``."""

from __future__ import annotations

from fastapi import APIRouter

from app.schemas.analysis import AnalysisListEntry, AnalysisRequest, AnalysisResult
from app.services.analysis_service import get_analysis_service

router = APIRouter()


@router.post("/analysis", response_model=AnalysisResult)
async def run_analysis(request: AnalysisRequest) -> AnalysisResult:
    service = get_analysis_service()
    plan = service.plan_from_request(request)
    return service.run(plan, question=request.question)


@router.get("/analysis", response_model=list[AnalysisListEntry])
async def list_analyses() -> list[AnalysisListEntry]:
    return get_analysis_service().list_runs()


@router.get("/analysis/{analysis_id}", response_model=AnalysisResult)
async def get_analysis(analysis_id: str) -> AnalysisResult:
    return get_analysis_service().get_result(analysis_id)


@router.delete("/analysis/{analysis_id}", status_code=204)
async def delete_analysis(analysis_id: str) -> None:
    get_analysis_service().delete(analysis_id)
