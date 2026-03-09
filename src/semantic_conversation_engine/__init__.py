"""Semantic Conversation Intelligence Engine.

NLP platform for large-scale conversation analysis in call centers
and digital service channels. Transforms raw conversations into
structured, searchable, and actionable insights using hybrid NLP
architecture (lexical + semantic + rules).
"""

from semantic_conversation_engine.exceptions import (
    ConfigurationError,
    EngineError,
    EngineValidationError,
    ModelError,
    PipelineError,
    RuleError,
)

__all__ = [
    "ConfigurationError",
    "EngineError",
    "EngineValidationError",
    "ModelError",
    "PipelineError",
    "RuleError",
]

__version__ = "0.1.0"
