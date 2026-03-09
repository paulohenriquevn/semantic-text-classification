"""Unit tests for ContextWindowConfig.

Tests cover: construction with defaults, field validation, cross-field
validation, strict mode, immutability, serialization, and reexport.
"""

import pytest

from semantic_conversation_engine.context.config import ContextWindowConfig

# ---------------------------------------------------------------------------
# Construction — defaults
# ---------------------------------------------------------------------------


class TestContextWindowConfigConstruction:
    def test_creates_with_defaults(self) -> None:
        config = ContextWindowConfig()
        assert config.window_size == 5
        assert config.stride == 2
        assert config.min_window_size == 1
        assert config.include_partial_tail is True
        assert config.render_speaker_labels is True
        assert config.render_turn_delimiter == "\n"

    def test_creates_with_custom_values(self) -> None:
        config = ContextWindowConfig(
            window_size=7,
            stride=3,
            min_window_size=3,
            include_partial_tail=False,
            render_speaker_labels=False,
            render_turn_delimiter=" | ",
        )
        assert config.window_size == 7
        assert config.stride == 3
        assert config.min_window_size == 3
        assert config.include_partial_tail is False
        assert config.render_speaker_labels is False
        assert config.render_turn_delimiter == " | "


# ---------------------------------------------------------------------------
# Validation — individual fields
# ---------------------------------------------------------------------------


class TestContextWindowConfigFieldValidation:
    def test_rejects_zero_window_size(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            ContextWindowConfig(window_size=0)

    def test_rejects_negative_window_size(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            ContextWindowConfig(window_size=-1)

    def test_rejects_zero_stride(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            ContextWindowConfig(stride=0)

    def test_rejects_negative_stride(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            ContextWindowConfig(stride=-1)

    def test_rejects_zero_min_window_size(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            ContextWindowConfig(min_window_size=0)

    def test_rejects_negative_min_window_size(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            ContextWindowConfig(min_window_size=-1)


# ---------------------------------------------------------------------------
# Validation — cross-field
# ---------------------------------------------------------------------------


class TestContextWindowConfigCrossFieldValidation:
    def test_rejects_min_window_size_exceeding_window_size(self) -> None:
        with pytest.raises(ValueError, match="must not exceed"):
            ContextWindowConfig(window_size=3, min_window_size=5)

    def test_accepts_min_window_size_equal_to_window_size(self) -> None:
        config = ContextWindowConfig(window_size=5, min_window_size=5)
        assert config.min_window_size == config.window_size

    def test_accepts_min_window_size_below_window_size(self) -> None:
        config = ContextWindowConfig(window_size=5, min_window_size=2)
        assert config.min_window_size < config.window_size


# ---------------------------------------------------------------------------
# Strict mode
# ---------------------------------------------------------------------------


class TestContextWindowConfigStrictMode:
    def test_rejects_string_for_bool(self) -> None:
        with pytest.raises(ValueError):
            ContextWindowConfig(include_partial_tail="true")  # type: ignore[arg-type]

    def test_rejects_float_for_int(self) -> None:
        with pytest.raises(ValueError):
            ContextWindowConfig(window_size=5.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


class TestContextWindowConfigImmutability:
    def test_cannot_assign_to_field(self) -> None:
        config = ContextWindowConfig()
        with pytest.raises(ValueError, match="frozen"):
            config.window_size = 10


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestContextWindowConfigSerialization:
    def test_model_dump_produces_dict(self) -> None:
        config = ContextWindowConfig()
        data = config.model_dump()
        assert isinstance(data, dict)
        assert data["window_size"] == 5
        assert data["stride"] == 2

    def test_reconstructs_from_model_dump(self) -> None:
        config = ContextWindowConfig(window_size=7, stride=3, min_window_size=2)
        data = config.model_dump()
        restored = ContextWindowConfig.model_validate(data, strict=False)
        assert restored == config


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestContextWindowConfigReexport:
    def test_importable_from_context_package(self) -> None:
        from semantic_conversation_engine.context import ContextWindowConfig as Imported

        assert Imported is ContextWindowConfig
