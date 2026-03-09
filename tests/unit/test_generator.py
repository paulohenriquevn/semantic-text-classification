"""Unit tests for embedding generators.

Tests cover: NullEmbeddingGenerator determinism, batch ordering,
provenance, normalization, stats, and SentenceTransformerGenerator
import guard.
"""

import numpy as np
import pytest

from semantic_conversation_engine.embeddings.config import EmbeddingModelConfig
from semantic_conversation_engine.embeddings.generator import (
    GenerationStats,
    NullEmbeddingGenerator,
    SentenceTransformerGenerator,
)
from semantic_conversation_engine.embeddings.inputs import (
    EmbeddingBatch,
    EmbeddingInput,
)
from semantic_conversation_engine.embeddings.preprocessing import PreprocessingConfig
from semantic_conversation_engine.models.enums import ObjectType, PoolingStrategy
from semantic_conversation_engine.models.types import EmbeddingId

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model_config(**overrides: object) -> EmbeddingModelConfig:
    defaults = {
        "model_name": "null-test-model",
        "model_version": "1.0.0",
        "pooling_strategy": PoolingStrategy.MEAN,
        "normalize_vectors": True,
    }
    defaults.update(overrides)
    return EmbeddingModelConfig(**defaults)  # type: ignore[arg-type]


def _inp(
    text: str = "hello world",
    emb_id: str = "emb_001",
    obj_id: str = "obj_001",
    obj_type: ObjectType = ObjectType.CONTEXT_WINDOW,
) -> EmbeddingInput:
    return EmbeddingInput(
        embedding_id=EmbeddingId(emb_id),
        object_type=obj_type,
        object_id=obj_id,
        text=text,
    )


def _batch(*inputs: EmbeddingInput) -> EmbeddingBatch:
    return EmbeddingBatch(items=list(inputs))


def _null_gen(**overrides: object) -> NullEmbeddingGenerator:
    defaults: dict[str, object] = {
        "model_config": _model_config(),
        "dimensions": 8,
    }
    defaults.update(overrides)
    return NullEmbeddingGenerator(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# NullEmbeddingGenerator — basic generation
# ---------------------------------------------------------------------------


class TestNullGeneratorGeneration:
    def test_generates_one_record_per_input(self) -> None:
        gen = _null_gen()
        records = gen.generate(_batch(_inp()))
        assert len(records) == 1

    def test_generates_correct_dimensions(self) -> None:
        gen = _null_gen(dimensions=16)
        records = gen.generate(_batch(_inp()))
        assert records[0].dimensions == 16
        assert len(records[0].vector) == 16

    def test_preserves_batch_order(self) -> None:
        gen = _null_gen()
        batch = _batch(
            _inp(text="first", emb_id="e1", obj_id="o1"),
            _inp(text="second", emb_id="e2", obj_id="o2"),
            _inp(text="third", emb_id="e3", obj_id="o3"),
        )
        records = gen.generate(batch)
        assert [r.source_id for r in records] == ["o1", "o2", "o3"]

    def test_preserves_embedding_id(self) -> None:
        gen = _null_gen()
        records = gen.generate(_batch(_inp(emb_id="emb_custom")))
        assert records[0].embedding_id == "emb_custom"


# ---------------------------------------------------------------------------
# NullEmbeddingGenerator — provenance
# ---------------------------------------------------------------------------


class TestNullGeneratorProvenance:
    def test_carries_model_name(self) -> None:
        gen = _null_gen(model_config=_model_config(model_name="test-e5"))
        records = gen.generate(_batch(_inp()))
        assert records[0].model_name == "test-e5"

    def test_carries_model_version(self) -> None:
        gen = _null_gen(model_config=_model_config(model_version="2.0"))
        records = gen.generate(_batch(_inp()))
        assert records[0].model_version == "2.0"

    def test_carries_pooling_strategy(self) -> None:
        gen = _null_gen(model_config=_model_config(pooling_strategy=PoolingStrategy.MAX))
        records = gen.generate(_batch(_inp()))
        assert records[0].pooling_strategy == PoolingStrategy.MAX

    def test_carries_source_type(self) -> None:
        gen = _null_gen()
        records = gen.generate(_batch(_inp(obj_type=ObjectType.TURN)))
        assert records[0].source_type == ObjectType.TURN


# ---------------------------------------------------------------------------
# NullEmbeddingGenerator — determinism
# ---------------------------------------------------------------------------


class TestNullGeneratorDeterminism:
    def test_same_text_produces_same_vector(self) -> None:
        gen = _null_gen()
        r1 = gen.generate(_batch(_inp(text="identical")))[0]
        r2 = gen.generate(_batch(_inp(text="identical")))[0]
        assert r1.vector == r2.vector

    def test_different_text_produces_different_vector(self) -> None:
        gen = _null_gen()
        r1 = gen.generate(_batch(_inp(text="alpha")))[0]
        r2 = gen.generate(_batch(_inp(text="beta")))[0]
        assert r1.vector != r2.vector

    def test_determinism_across_instances(self) -> None:
        gen1 = _null_gen()
        gen2 = _null_gen()
        r1 = gen1.generate(_batch(_inp(text="stable")))[0]
        r2 = gen2.generate(_batch(_inp(text="stable")))[0]
        assert r1.vector == r2.vector


# ---------------------------------------------------------------------------
# NullEmbeddingGenerator — normalization
# ---------------------------------------------------------------------------


class TestNullGeneratorNormalization:
    def test_normalized_vectors_have_unit_length(self) -> None:
        gen = _null_gen(
            model_config=_model_config(normalize_vectors=True),
            dimensions=64,
        )
        records = gen.generate(_batch(_inp()))
        norm = np.linalg.norm(records[0].vector)
        np.testing.assert_almost_equal(norm, 1.0, decimal=5)

    def test_unnormalized_vectors_may_differ_from_unit(self) -> None:
        gen = _null_gen(
            model_config=_model_config(normalize_vectors=False),
            dimensions=64,
        )
        records = gen.generate(_batch(_inp()))
        norm = np.linalg.norm(records[0].vector)
        # With random vectors, the norm is very unlikely to be exactly 1.0
        assert abs(norm - 1.0) > 0.01


# ---------------------------------------------------------------------------
# NullEmbeddingGenerator — preprocessing integration
# ---------------------------------------------------------------------------


class TestNullGeneratorPreprocessing:
    def test_task_prefix_changes_vector(self) -> None:
        gen_no_prefix = _null_gen()
        gen_with_prefix = NullEmbeddingGenerator(
            model_config=_model_config(),
            preprocessing_config=PreprocessingConfig(task_prefix="query:"),
            dimensions=8,
        )
        r1 = gen_no_prefix.generate(_batch(_inp(text="test")))[0]
        r2 = gen_with_prefix.generate(_batch(_inp(text="test")))[0]
        # Different preprocessing → different hash → different vector
        assert r1.vector != r2.vector


# ---------------------------------------------------------------------------
# NullEmbeddingGenerator — stats / observability
# ---------------------------------------------------------------------------


class TestNullGeneratorStats:
    def test_stats_empty_initially(self) -> None:
        gen = _null_gen()
        assert gen.stats == []

    def test_stats_recorded_per_call(self) -> None:
        gen = _null_gen()
        gen.generate(_batch(_inp()))
        gen.generate(_batch(_inp(), _inp(emb_id="e2", obj_id="o2")))
        assert len(gen.stats) == 2

    def test_stats_contain_batch_size(self) -> None:
        gen = _null_gen()
        gen.generate(_batch(_inp(), _inp(emb_id="e2", obj_id="o2")))
        assert gen.stats[0].batch_size == 2

    def test_stats_contain_model_name(self) -> None:
        gen = _null_gen(model_config=_model_config(model_name="test-m"))
        gen.generate(_batch(_inp()))
        assert gen.stats[0].model_name == "test-m"

    def test_stats_contain_positive_ms(self) -> None:
        gen = _null_gen()
        gen.generate(_batch(_inp()))
        assert gen.stats[0].total_ms >= 0.0

    def test_stats_returns_copy(self) -> None:
        gen = _null_gen()
        gen.generate(_batch(_inp()))
        stats = gen.stats
        stats.clear()
        assert len(gen.stats) == 1  # original not affected


# ---------------------------------------------------------------------------
# GenerationStats
# ---------------------------------------------------------------------------


class TestGenerationStats:
    def test_frozen(self) -> None:
        stats = GenerationStats(batch_size=1, total_ms=5.0, model_name="m")
        with pytest.raises(AttributeError):
            stats.batch_size = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SentenceTransformerGenerator — import guard
# ---------------------------------------------------------------------------


class TestSentenceTransformerGuard:
    def test_raises_import_error_without_library(self) -> None:
        """sentence-transformers is not installed in test env."""
        with pytest.raises(ImportError, match="sentence-transformers"):
            SentenceTransformerGenerator(model_config=_model_config())


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestGeneratorReexport:
    def test_null_generator_importable(self) -> None:
        from semantic_conversation_engine.embeddings import (
            NullEmbeddingGenerator as NG,
        )

        assert NG is NullEmbeddingGenerator

    def test_stats_importable(self) -> None:
        from semantic_conversation_engine.embeddings import (
            GenerationStats as GS,
        )

        assert GS is GenerationStats
