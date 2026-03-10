"""Unit tests for SegmentationConfig.

Tests cover: construction with defaults, field validation, strict mode,
immutability, serialization, and reexport.
"""

import pytest

from talkex.segmentation.config import SegmentationConfig

# ---------------------------------------------------------------------------
# Construction — defaults
# ---------------------------------------------------------------------------


class TestSegmentationConfigConstruction:
    def test_creates_with_defaults(self) -> None:
        config = SegmentationConfig()
        assert config.normalize_unicode is True
        assert config.collapse_whitespace is True
        assert config.strip_lines is True
        assert config.merge_consecutive_same_speaker is True
        assert config.min_turn_chars == 1
        assert config.max_turn_chars == 10_000
        assert isinstance(config.speaker_label_pattern, str)

    def test_creates_with_custom_values(self) -> None:
        config = SegmentationConfig(
            normalize_unicode=False,
            collapse_whitespace=False,
            strip_lines=False,
            merge_consecutive_same_speaker=False,
            min_turn_chars=5,
            max_turn_chars=5000,
            speaker_label_pattern=r"^\w+:",
        )
        assert config.normalize_unicode is False
        assert config.min_turn_chars == 5
        assert config.max_turn_chars == 5000


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestSegmentationConfigValidation:
    def test_rejects_zero_min_turn_chars(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            SegmentationConfig(min_turn_chars=0)

    def test_rejects_negative_min_turn_chars(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            SegmentationConfig(min_turn_chars=-1)

    def test_rejects_zero_max_turn_chars(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            SegmentationConfig(max_turn_chars=0)

    def test_rejects_negative_max_turn_chars(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            SegmentationConfig(max_turn_chars=-1)

    def test_accepts_min_equal_to_max(self) -> None:
        config = SegmentationConfig(min_turn_chars=100, max_turn_chars=100)
        assert config.min_turn_chars == config.max_turn_chars


# ---------------------------------------------------------------------------
# Strict mode
# ---------------------------------------------------------------------------


class TestSegmentationConfigStrictMode:
    def test_rejects_string_for_bool(self) -> None:
        with pytest.raises(ValueError):
            SegmentationConfig(normalize_unicode="true")  # type: ignore[arg-type]

    def test_rejects_float_for_int(self) -> None:
        with pytest.raises(ValueError):
            SegmentationConfig(min_turn_chars=5.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


class TestSegmentationConfigImmutability:
    def test_cannot_assign_to_field(self) -> None:
        config = SegmentationConfig()
        with pytest.raises(ValueError, match="frozen"):
            config.min_turn_chars = 10


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSegmentationConfigSerialization:
    def test_model_dump_produces_dict(self) -> None:
        config = SegmentationConfig()
        data = config.model_dump()
        assert isinstance(data, dict)
        assert data["min_turn_chars"] == 1

    def test_reconstructs_from_model_dump(self) -> None:
        config = SegmentationConfig(min_turn_chars=5, max_turn_chars=500)
        data = config.model_dump()
        restored = SegmentationConfig.model_validate(data, strict=False)
        assert restored == config


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestSegmentationConfigReexport:
    def test_importable_from_segmentation_package(self) -> None:
        from talkex.segmentation import SegmentationConfig as Imported

        assert Imported is SegmentationConfig
