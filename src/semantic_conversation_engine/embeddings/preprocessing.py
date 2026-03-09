"""Text preprocessing for embedding generation.

Prepares text for embedding models. This is intentionally SEPARATE from
the segmentation normalization (which operates on raw_text → normalized_text).
Embedding preprocessing operates on already-normalized text and applies
model-specific transformations.

Key responsibilities:
    - Task-specific prefixing (e.g., E5 models require "query: " or "passage: ")
    - Optional speaker label inclusion for role-aware context
    - Text truncation awareness (logged, not silently dropped)

All functions are pure — no shared state, no side effects, no mutation
of input objects.
"""

from dataclasses import dataclass

from semantic_conversation_engine.embeddings.inputs import EmbeddingInput
from semantic_conversation_engine.models.enums import ObjectType


@dataclass(frozen=True)
class PreprocessingConfig:
    """Controls how text is prepared for embedding.

    Args:
        task_prefix: Prefix prepended to text before embedding.
            E5 models use "passage: " for documents and "query: "
            for queries. Empty string for no prefix.
        include_object_type_prefix: Whether to prepend a human-readable
            object type tag (e.g., "[WINDOW] ") to the text.
    """

    task_prefix: str = ""
    include_object_type_prefix: bool = False


_OBJECT_TYPE_LABELS: dict[ObjectType, str] = {
    ObjectType.TURN: "[TURN]",
    ObjectType.CONTEXT_WINDOW: "[WINDOW]",
    ObjectType.CONVERSATION: "[CONVERSATION]",
}


def prepare_embedding_text(
    inp: EmbeddingInput,
    config: PreprocessingConfig,
) -> str:
    """Prepare text for embedding generation.

    Applies task prefix and optional object type tag to the input text.
    Does NOT mutate the input — returns a new string.

    Args:
        inp: The embedding input with text and object type.
        config: Preprocessing configuration.

    Returns:
        The prepared text string ready for the embedding model.
    """
    parts: list[str] = []

    if config.task_prefix:
        parts.append(config.task_prefix)

    if config.include_object_type_prefix:
        label = _OBJECT_TYPE_LABELS.get(inp.object_type, "[UNKNOWN]")
        parts.append(label)

    parts.append(inp.text)

    return " ".join(parts) if len(parts) > 1 else inp.text


def prepare_batch_texts(
    inputs: list[EmbeddingInput],
    config: PreprocessingConfig,
) -> list[str]:
    """Prepare texts for a batch of embedding inputs.

    Convenience function applying prepare_embedding_text to each input.

    Args:
        inputs: List of embedding inputs.
        config: Preprocessing configuration.

    Returns:
        List of prepared text strings, same order as inputs.
    """
    return [prepare_embedding_text(inp, config) for inp in inputs]


def estimate_token_count(text: str) -> int:
    """Rough token count estimate for truncation awareness.

    Uses the ~4 characters per token heuristic. This is intentionally
    approximate — the actual tokenizer determines real truncation.
    Used only for logging and warnings, not for actual truncation.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated token count.
    """
    return max(1, len(text) // 4)
