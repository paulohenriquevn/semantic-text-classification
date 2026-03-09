"""Unit tests for text normalization.

Tests cover: Unicode NFKC, whitespace collapsing, line stripping,
flag combinations, and identity when all flags are disabled.
"""

from semantic_conversation_engine.segmentation.config import SegmentationConfig
from semantic_conversation_engine.segmentation.normalization import normalize_text

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(**overrides: object) -> SegmentationConfig:
    """Factory with all normalization flags enabled by default."""
    defaults: dict[str, object] = {
        "normalize_unicode": True,
        "collapse_whitespace": True,
        "strip_lines": True,
    }
    defaults.update(overrides)
    return SegmentationConfig(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Unicode NFKC normalization
# ---------------------------------------------------------------------------


class TestUnicodeNormalization:
    def test_normalizes_fullwidth_to_ascii(self) -> None:
        text = "\uff28\uff45\uff4c\uff4c\uff4f"  # fullwidth "Hello"
        result = normalize_text(text, _config())
        assert result == "Hello"

    def test_normalizes_compatibility_characters(self) -> None:
        text = "\u2126"  # Ω (OHM SIGN) → Ω (GREEK CAPITAL LETTER OMEGA)
        result = normalize_text(text, _config())
        assert result == "\u03a9"

    def test_skips_unicode_when_disabled(self) -> None:
        text = "\uff28\uff45\uff4c\uff4c\uff4f"
        result = normalize_text(text, _config(normalize_unicode=False))
        assert result == "\uff28\uff45\uff4c\uff4c\uff4f"


# ---------------------------------------------------------------------------
# Whitespace collapsing
# ---------------------------------------------------------------------------


class TestWhitespaceCollapsing:
    def test_collapses_multiple_spaces(self) -> None:
        result = normalize_text("hello    world", _config())
        assert result == "hello world"

    def test_collapses_tabs_to_single_space(self) -> None:
        result = normalize_text("hello\t\tworld", _config())
        assert result == "hello world"

    def test_preserves_newlines(self) -> None:
        result = normalize_text("line1\nline2", _config())
        assert result == "line1\nline2"

    def test_collapses_mixed_horizontal_whitespace(self) -> None:
        result = normalize_text("a \t  b", _config())
        assert result == "a b"

    def test_skips_collapsing_when_disabled(self) -> None:
        result = normalize_text("hello    world", _config(collapse_whitespace=False))
        assert result == "hello    world"


# ---------------------------------------------------------------------------
# Line stripping
# ---------------------------------------------------------------------------


class TestLineStripping:
    def test_strips_leading_whitespace_per_line(self) -> None:
        result = normalize_text("  hello\n  world", _config(collapse_whitespace=False))
        assert result == "hello\nworld"

    def test_strips_trailing_whitespace_per_line(self) -> None:
        result = normalize_text("hello  \nworld  ", _config(collapse_whitespace=False))
        assert result == "hello\nworld"

    def test_skips_stripping_when_disabled(self) -> None:
        result = normalize_text("  hello  ", _config(strip_lines=False, collapse_whitespace=False))
        assert result == "  hello  "


# ---------------------------------------------------------------------------
# Flag combinations
# ---------------------------------------------------------------------------


class TestFlagCombinations:
    def test_all_flags_disabled_returns_identity(self) -> None:
        text = "  \uff28ello   world  \n  foo  "
        config = _config(
            normalize_unicode=False,
            collapse_whitespace=False,
            strip_lines=False,
        )
        assert normalize_text(text, config) == text

    def test_all_flags_enabled(self) -> None:
        text = "  \uff28ello   world  \n  foo  "
        result = normalize_text(text, _config())
        # NFKC: fullwidth H -> H, strip per line, collapse multi-space
        assert result == "Hello world\nfoo"

    def test_unicode_and_strip_without_collapse(self) -> None:
        text = "  \uff28ello   world  "
        config = _config(collapse_whitespace=False)
        result = normalize_text(text, config)
        assert result == "Hello   world"

    def test_empty_string_returns_empty(self) -> None:
        assert normalize_text("", _config()) == ""

    def test_whitespace_only_returns_empty_after_strip(self) -> None:
        result = normalize_text("   \n   ", _config())
        assert result == "\n"
