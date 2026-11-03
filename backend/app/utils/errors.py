"""Application exception hierarchy.

The API layer maps these to stable, structured error responses. Tracebacks and
internal messages are never returned to the client; the original exception is
logged server-side and only its type and a safe code are exposed.
"""

from __future__ import annotations


class OperationsIntelligenceError(Exception):
    """Base class for every error raised by this application."""

    code = "internal_error"
    status_code = 500

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code


class ConfigurationError(OperationsIntelligenceError):
    code = "configuration_error"
    status_code = 500


class DatasetNotFoundError(OperationsIntelligenceError):
    code = "dataset_not_found"
    status_code = 404


class DatasetLoadError(OperationsIntelligenceError):
    code = "dataset_load_error"
    status_code = 500


class DataValidationError(OperationsIntelligenceError):
    code = "data_validation_error"
    status_code = 500


class InvalidRequestError(OperationsIntelligenceError):
    code = "invalid_request"
    status_code = 400


class UnsupportedAnalysisError(OperationsIntelligenceError):
    """Raised when a plan requests an operation the platform does not support."""

    code = "unsupported_analysis"
    status_code = 422


class AnalysisNotFoundError(OperationsIntelligenceError):
    code = "analysis_not_found"
    status_code = 404


class AnalysisExecutionError(OperationsIntelligenceError):
    code = "analysis_execution_failed"
    status_code = 500


class InsufficientDataError(OperationsIntelligenceError):
    code = "insufficient_data"
    status_code = 422


class LLMConfigurationError(OperationsIntelligenceError):
    code = "llm_not_configured"
    status_code = 503


class LLMProviderError(OperationsIntelligenceError):
    code = "llm_provider_error"
    status_code = 502


class LLMResponseError(OperationsIntelligenceError):
    """The provider responded, but the payload was unusable or untrusted."""

    code = "llm_invalid_response"
    status_code = 502
