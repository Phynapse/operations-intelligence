"""Health check: service status and dataset readiness."""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import get_settings
from app.data.repository import get_repository

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str
    timestamp: dt.datetime
    datasets_loaded: int
    datasets_available: int
    llm_configured: bool


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    repository = get_repository()
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
        timestamp=dt.datetime.now(dt.UTC),
        datasets_loaded=len(repository.available_datasets()),
        datasets_available=len(repository.available_datasets()),
        llm_configured=settings.llm_configured,
    )
