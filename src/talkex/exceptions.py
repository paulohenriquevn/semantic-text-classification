"""Domain-specific exception hierarchy for the TalkEx — Conversation Intelligence Engine.

All project exceptions inherit from EngineError. Each exception accepts a message
and an optional context dict for structured error information, enabling log entries
with enough context to reproduce the problem without a debugger.

Naming convention: all exceptions are prefixed to avoid collision with
third-party exceptions (e.g., EngineValidationError, not ValidationError,
to prevent ambiguity with pydantic.ValidationError).
"""

from typing import Any


class EngineError(Exception):
    """Base exception for all TalkEx — Conversation Intelligence Engine errors.

    Args:
        message: Human-readable description of what went wrong.
        context: Optional dict with structured diagnostic data (IDs, inputs, expected values).
    """

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        self.message = message
        self.context = context or {}
        super().__init__(message)

    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} [{context_str}]"
        return self.message


class EngineValidationError(EngineError):
    """Input validation failure at system boundaries.

    Raised when data entering the system (API handlers, pipeline entry points,
    file parsers) fails validation. After boundaries, data is trusted.
    """


class PipelineError(EngineError):
    """Error during pipeline stage processing.

    Raised when a pipeline stage (ingestion, segmentation, embedding, etc.)
    encounters an irrecoverable error during execution.
    """


class ModelError(EngineError):
    """Error related to ML model operations.

    Raised when loading, initializing, or running inference with an
    embedding model, classifier, or cross-encoder fails.
    """


class RuleError(EngineError):
    """Error in rule parsing, validation, or execution.

    Raised when the semantic rule engine encounters invalid DSL syntax,
    type mismatches in predicates, or execution failures during AST evaluation.
    """


class ConfigurationError(EngineError):
    """Configuration or environment error.

    Raised when required configuration is missing, invalid, or inconsistent.
    Includes missing environment variables, invalid model paths, and
    incompatible version references.
    """
