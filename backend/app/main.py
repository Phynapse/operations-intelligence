"""FastAPI application entry point.

The API is a modular monolith. Routes are organised by concern and import the
service layer rather than touching the data or analytics layers directly.

Security assumptions:
* secrets come only from environment variables;
* the LLM API key is never returned to the frontend;
* user-supplied paths are never used for file access;
* structured error handling hides stack traces from responses.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.utils.errors import OperationsIntelligenceError

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup / shutdown lifecycle."""
    settings = get_settings()
    configure_logging(settings)
    logger.info(
        "api_starting",
        extra={
            "app": settings.app_name,
            "env": settings.app_env,
            "version": settings.app_version,
            "data_dir": str(settings.data_dir),
        },
    )
    yield
    logger.info("api_shutting_down")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.api_cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    from app.api.routes.analysis import router as analysis_router
    from app.api.routes.dashboard import router as dashboard_router
    from app.api.routes.datasets import router as datasets_router
    from app.api.routes.health import router as health_router
    from app.api.routes.questions import router as questions_router

    app.include_router(health_router, prefix="/api", tags=["health"])
    app.include_router(dashboard_router, prefix="/api", tags=["dashboard"])
    app.include_router(datasets_router, prefix="/api/datasets", tags=["datasets"])
    app.include_router(analysis_router, prefix="/api", tags=["analysis"])
    app.include_router(questions_router, prefix="/api", tags=["questions"])

    @app.exception_handler(OperationsIntelligenceError)
    async def _handle_app_error(request: Request, exc: OperationsIntelligenceError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "request_id": request.headers.get("x-request-id"),
                }
            },
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "unhandled_error",
            extra={
                "method": request.method,
                "path": request.url.path,
            },
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "An unexpected error occurred. Check server logs for details.",
                }
            },
        )

    @app.middleware("http")
    async def _timing(request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Response-Time"] = f"{duration_ms:.1f}ms"
        return response

    return app


# Module-level ASGI application instance for uvicorn (e.g. uvicorn app.main:app)
app = create_app()
