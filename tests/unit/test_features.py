"""Unit tests for lexical feature extraction.

Tests cover: char_count, word_count, has_question, line_count,
avg_word_length, and edge cases (empty string, single word).
"""

import pytest

from semantic_conversation_engine.segmentation.features import extract_lexical_features

# ---------------------------------------------------------------------------
# Individual features
# ---------------------------------------------------------------------------


class TestCharCount:
    def test_counts_characters(self) -> None:
        features = extract_lexical_features("hello")
        assert features["char_count"] == 5

    def test_includes_whitespace_in_count(self) -> None:
        features = extract_lexical_features("hello world")
        assert features["char_count"] == 11

    def test_empty_string_returns_zero(self) -> None:
        features = extract_lexical_features("")
        assert features["char_count"] == 0


class TestWordCount:
    def test_counts_whitespace_separated_tokens(self) -> None:
        features = extract_lexical_features("hello world foo")
        assert features["word_count"] == 3

    def test_single_word(self) -> None:
        features = extract_lexical_features("hello")
        assert features["word_count"] == 1

    def test_empty_string_returns_zero(self) -> None:
        features = extract_lexical_features("")
        assert features["word_count"] == 0


class TestHasQuestion:
    def test_detects_question_mark(self) -> None:
        features = extract_lexical_features("How are you?")
        assert features["has_question"] is True

    def test_no_question_mark(self) -> None:
        features = extract_lexical_features("I am fine.")
        assert features["has_question"] is False

    def test_multiple_question_marks(self) -> None:
        features = extract_lexical_features("What? Really?")
        assert features["has_question"] is True


class TestLineCount:
    def test_single_line(self) -> None:
        features = extract_lexical_features("hello")
        assert features["line_count"] == 1

    def test_multiple_lines(self) -> None:
        features = extract_lexical_features("line1\nline2\nline3")
        assert features["line_count"] == 3

    def test_empty_string_is_one_line(self) -> None:
        features = extract_lexical_features("")
        assert features["line_count"] == 1


class TestAvgWordLength:
    def test_calculates_mean(self) -> None:
        # "hi" (2) + "there" (5) = 7 / 2 = 3.5
        features = extract_lexical_features("hi there")
        assert features["avg_word_length"] == pytest.approx(3.5)

    def test_single_word(self) -> None:
        features = extract_lexical_features("hello")
        assert features["avg_word_length"] == pytest.approx(5.0)

    def test_empty_string_returns_zero(self) -> None:
        features = extract_lexical_features("")
        assert features["avg_word_length"] == 0.0


# ---------------------------------------------------------------------------
# Full feature dict
# ---------------------------------------------------------------------------


class TestFeatureDictStructure:
    def test_returns_all_expected_keys(self) -> None:
        features = extract_lexical_features("hello world")
        expected_keys = {
            "char_count",
            "word_count",
            "has_question",
            "line_count",
            "avg_word_length",
        }
        assert set(features.keys()) == expected_keys

    def test_types_are_correct(self) -> None:
        features = extract_lexical_features("hello world?")
        assert isinstance(features["char_count"], int)
        assert isinstance(features["word_count"], int)
        assert isinstance(features["has_question"], bool)
        assert isinstance(features["line_count"], int)
        assert isinstance(features["avg_word_length"], float)
