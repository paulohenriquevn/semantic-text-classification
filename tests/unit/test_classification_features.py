"""Unit tests for classification feature extraction.

Tests cover: FeatureSet, lexical features, structural features,
feature merging, and reexport.
"""

from semantic_conversation_engine.classification.features import (
    FeatureSet,
    extract_lexical_features,
    extract_structural_features,
    merge_feature_sets,
)


# ---------------------------------------------------------------------------
# FeatureSet
# ---------------------------------------------------------------------------


class TestFeatureSet:
    def test_creates_with_features(self) -> None:
        fs = FeatureSet(
            features={"a": 1.0, "b": 2.0},
            feature_names=["a", "b"],
        )
        assert fs.features["a"] == 1.0
        assert fs.feature_names == ["a", "b"]

    def test_to_vector_preserves_order(self) -> None:
        fs = FeatureSet(
            features={"b": 2.0, "a": 1.0},
            feature_names=["a", "b"],
        )
        assert fs.to_vector() == [1.0, 2.0]

    def test_to_vector_missing_feature_defaults_to_zero(self) -> None:
        fs = FeatureSet(
            features={"a": 1.0},
            feature_names=["a", "b"],
        )
        assert fs.to_vector() == [1.0, 0.0]

    def test_is_frozen(self) -> None:
        fs = FeatureSet(features={}, feature_names=[])
        try:
            fs.feature_names = ["x"]  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Lexical features
# ---------------------------------------------------------------------------


class TestExtractLexicalFeatures:
    def test_extracts_basic_counts(self) -> None:
        fs = extract_lexical_features("hello world")
        assert fs.features["word_count"] == 2.0
        assert fs.features["char_count"] == 11.0

    def test_avg_word_length(self) -> None:
        fs = extract_lexical_features("ab cd ef")
        assert fs.features["avg_word_length"] == 2.0

    def test_question_count(self) -> None:
        fs = extract_lexical_features("what? why? how?")
        assert fs.features["question_count"] == 3.0

    def test_exclamation_count(self) -> None:
        fs = extract_lexical_features("wow! amazing!")
        assert fs.features["exclamation_count"] == 2.0

    def test_uppercase_ratio(self) -> None:
        fs = extract_lexical_features("HELLO world")
        # 5 uppercase out of 10 alpha chars = 0.5
        assert abs(fs.features["uppercase_ratio"] - 0.5) < 1e-9

    def test_digit_ratio(self) -> None:
        fs = extract_lexical_features("abc123")
        # 3 digits out of 6 chars = 0.5
        assert abs(fs.features["digit_ratio"] - 0.5) < 1e-9

    def test_empty_text(self) -> None:
        fs = extract_lexical_features("")
        assert fs.features["word_count"] == 0.0
        assert fs.features["char_count"] == 0.0
        assert fs.features["avg_word_length"] == 0.0

    def test_feature_names_stable(self) -> None:
        fs = extract_lexical_features("test")
        expected = [
            "char_count",
            "word_count",
            "avg_word_length",
            "question_count",
            "exclamation_count",
            "uppercase_ratio",
            "digit_ratio",
        ]
        assert fs.feature_names == expected

    def test_to_vector_produces_correct_length(self) -> None:
        fs = extract_lexical_features("test text")
        vec = fs.to_vector()
        assert len(vec) == 7


# ---------------------------------------------------------------------------
# Structural features
# ---------------------------------------------------------------------------


class TestExtractStructuralFeatures:
    def test_default_values(self) -> None:
        fs = extract_structural_features()
        assert fs.features["is_customer"] == 0.0
        assert fs.features["is_agent"] == 0.0
        assert fs.features["turn_count"] == 1.0
        assert fs.features["speaker_count"] == 1.0

    def test_customer_flag(self) -> None:
        fs = extract_structural_features(is_customer=True)
        assert fs.features["is_customer"] == 1.0
        assert fs.features["is_agent"] == 0.0

    def test_agent_flag(self) -> None:
        fs = extract_structural_features(is_agent=True)
        assert fs.features["is_agent"] == 1.0

    def test_custom_counts(self) -> None:
        fs = extract_structural_features(turn_count=5, speaker_count=2)
        assert fs.features["turn_count"] == 5.0
        assert fs.features["speaker_count"] == 2.0

    def test_feature_names_stable(self) -> None:
        fs = extract_structural_features()
        expected = ["is_customer", "is_agent", "turn_count", "speaker_count"]
        assert fs.feature_names == expected


# ---------------------------------------------------------------------------
# Feature merging
# ---------------------------------------------------------------------------


class TestMergeFeatureSets:
    def test_merges_two_sets(self) -> None:
        fs1 = FeatureSet(features={"a": 1.0}, feature_names=["a"])
        fs2 = FeatureSet(features={"b": 2.0}, feature_names=["b"])
        merged = merge_feature_sets(fs1, fs2)
        assert merged.feature_names == ["a", "b"]
        assert merged.features == {"a": 1.0, "b": 2.0}

    def test_merges_preserves_order(self) -> None:
        fs1 = FeatureSet(features={"b": 1.0}, feature_names=["b"])
        fs2 = FeatureSet(features={"a": 2.0}, feature_names=["a"])
        merged = merge_feature_sets(fs1, fs2)
        assert merged.feature_names == ["b", "a"]

    def test_duplicate_name_uses_last_value(self) -> None:
        fs1 = FeatureSet(features={"a": 1.0}, feature_names=["a"])
        fs2 = FeatureSet(features={"a": 2.0}, feature_names=["a"])
        merged = merge_feature_sets(fs1, fs2)
        assert merged.features["a"] == 2.0
        assert merged.feature_names == ["a"]  # No duplicate names

    def test_merges_empty_sets(self) -> None:
        merged = merge_feature_sets()
        assert merged.features == {}
        assert merged.feature_names == []

    def test_merges_lexical_and_structural(self) -> None:
        lex = extract_lexical_features("hello world")
        struct = extract_structural_features(is_customer=True, turn_count=3)
        merged = merge_feature_sets(lex, struct)
        assert "word_count" in merged.features
        assert "is_customer" in merged.features
        assert len(merged.feature_names) == 11  # 7 lexical + 4 structural

    def test_merged_to_vector(self) -> None:
        fs1 = FeatureSet(features={"a": 1.0}, feature_names=["a"])
        fs2 = FeatureSet(features={"b": 2.0}, feature_names=["b"])
        merged = merge_feature_sets(fs1, fs2)
        assert merged.to_vector() == [1.0, 2.0]


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestFeaturesReexport:
    def test_importable_from_classification_package(self) -> None:
        from semantic_conversation_engine.classification import (
            FeatureSet as FS,
            extract_lexical_features as elf,
            extract_structural_features as esf,
            merge_feature_sets as mfs,
        )

        assert FS is FeatureSet
        assert elf is extract_lexical_features
        assert esf is extract_structural_features
        assert mfs is merge_feature_sets
