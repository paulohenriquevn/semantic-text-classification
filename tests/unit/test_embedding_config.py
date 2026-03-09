"""Unit tests for EmbeddingModelConfig and EmbeddingRuntimeConfig.

Tests cover: construction, defaults, field validation, strict mode,
immutability, and reexport.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from semantic_conversation_engine.embeddings.config import (
    EmbeddingModelConfig,
    EmbeddingRuntimeConfig,
)
from semantic_conversation_engine.models.enums import PoolingStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model_config(**overrides: Any) -> EmbeddingModelConfig:
    defaults: dict[str, Any] = {
        "model_name": "intfloat/e5-base-v2",
        "model_version": "1.0.0",
    }
    defaults.update(overrides)
    return EmbeddingModelConfig(**defaults)


def _runtime_config(**overrides: Any) -> EmbeddingRuntimeConfig:
    defaults: dict[str, Any] = {}
    defaults.update(overrides)
    return EmbeddingRuntimeConfig(**defaults)


# ---------------------------------------------------------------------------
# EmbeddingModelConfig construction
# ---------------------------------------------------------------------------


class TestModelConfigConstruction:
    def test_minimal_construction(self) -> None:
        cfg = _model_config()
        assert cfg.model_name == "intfloat/e5-base-v2"
        assert cfg.model_version == "1.0.0"

    def test_defaults(self) -> None:
        cfg = _model_config()
        assert cfg.pooling_strategy == PoolingStrategy.MEAN
        assert cfg.normalize_vectors is True
        assert cfg.max_length == 512
        assert cfg.batch_size == 32

    def test_all_fields(self) -> None:
        cfg = _model_config(
            pooling_strategy=PoolingStrategy.MAX,
            normalize_vectors=False,
            max_length=256,
            batch_size=16,
        )
        assert cfg.pooling_strategy == PoolingStrategy.MAX
        assert cfg.normalize_vectors is False
        assert cfg.max_length == 256
        assert cfg.batch_size == 16


# ---------------------------------------------------------------------------
# EmbeddingModelConfig validation
# ---------------------------------------------------------------------------


class TestModelConfigValidation:
    def test_rejects_empty_model_name(self) -> None:
        with pytest.raises(ValidationError, match="model_name"):
            _model_config(model_name="")

    def test_rejects_whitespace_model_name(self) -> None:
        with pytest.raises(ValidationError, match="model_name"):
            _model_config(model_name="   ")

    def test_rejects_empty_model_version(self) -> None:
        with pytest.raises(ValidationError, match="model_version"):
            _model_config(model_version="")

    def test_rejects_zero_max_length(self) -> None:
        with pytest.raises(ValidationError, match="max_length"):
            _model_config(max_length=0)

    def test_rejects_negative_max_length(self) -> None:
        with pytest.raises(ValidationError, match="max_length"):
            _model_config(max_length=-1)

    def test_rejects_zero_batch_size(self) -> None:
        with pytest.raises(ValidationError, match="batch_size"):
            _model_config(batch_size=0)


# ---------------------------------------------------------------------------
# EmbeddingModelConfig strict mode and immutability
# ---------------------------------------------------------------------------


class TestModelConfigStrictAndFrozen:
    def test_rejects_int_for_string(self) -> None:
        with pytest.raises(ValidationError):
            _model_config(model_name=123)

    def test_rejects_string_for_bool(self) -> None:
        with pytest.raises(ValidationError):
            _model_config(normalize_vectors="yes")

    def test_frozen(self) -> None:
        cfg = _model_config()
        with pytest.raises(ValidationError):
            cfg.model_name = "other"


# ---------------------------------------------------------------------------
# EmbeddingRuntimeConfig
# ---------------------------------------------------------------------------


class TestRuntimeConfig:
    def test_defaults(self) -> None:
        cfg = _runtime_config()
        assert cfg.enable_cache is True
        assert cfg.max_retries == 3
        assert cfg.timeout_seconds == 30.0

    def test_custom_values(self) -> None:
        cfg = _runtime_config(enable_cache=False, max_retries=5, timeout_seconds=60.0)
        assert cfg.enable_cache is False
        assert cfg.max_retries == 5
        assert cfg.timeout_seconds == 60.0

    def test_rejects_negative_retries(self) -> None:
        with pytest.raises(ValidationError, match="max_retries"):
            _runtime_config(max_retries=-1)

    def test_rejects_zero_timeout(self) -> None:
        with pytest.raises(ValidationError, match="timeout_seconds"):
            _runtime_config(timeout_seconds=0.0)

    def test_rejects_negative_timeout(self) -> None:
        with pytest.raises(ValidationError, match="timeout_seconds"):
            _runtime_config(timeout_seconds=-1.0)

    def test_frozen(self) -> None:
        cfg = _runtime_config()
        with pytest.raises(ValidationError):
            cfg.enable_cache = False


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestEmbeddingConfigReexport:
    def test_model_config_importable_from_package(self) -> None:
        from semantic_conversation_engine.embeddings import (
            EmbeddingModelConfig as Imported,
        )

        assert Imported is EmbeddingModelConfig

    def test_runtime_config_importable_from_package(self) -> None:
        from semantic_conversation_engine.embeddings import (
            EmbeddingRuntimeConfig as Imported,
        )

        assert Imported is EmbeddingRuntimeConfig
