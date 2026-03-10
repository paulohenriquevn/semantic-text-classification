"""Unit tests for ClassificationDataset, ClassificationExample, GroundTruthLabel.

Tests cover: construction, properties, immutability, JSON round-trip, reexport.
"""

import json

from talkex.classification_eval.dataset import (
    ClassificationDataset,
    ClassificationExample,
    GroundTruthLabel,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_example(
    example_id: str = "ex_001",
    text: str = "I have a billing issue",
    labels: list[str] | None = None,
    source_type: str = "context_window",
) -> ClassificationExample:
    labels = labels or ["billing"]
    return ClassificationExample(
        example_id=example_id,
        text=text,
        ground_truth=[GroundTruthLabel(label=name) for name in labels],
        source_type=source_type,
    )


def _make_dataset() -> ClassificationDataset:
    return ClassificationDataset(
        name="test-dataset",
        version="1.0",
        examples=[
            _make_example("ex_1", "billing issue", ["billing"]),
            _make_example("ex_2", "cancel subscription", ["cancel"]),
            _make_example("ex_3", "billing and refund", ["billing", "refund"]),
        ],
        label_names=["billing", "cancel", "refund"],
        description="Test classification dataset",
    )


# ---------------------------------------------------------------------------
# GroundTruthLabel
# ---------------------------------------------------------------------------


class TestGroundTruthLabel:
    def test_construction(self) -> None:
        gt = GroundTruthLabel(label="billing")
        assert gt.label == "billing"
        assert gt.relevance == 1

    def test_construction_with_relevance(self) -> None:
        gt = GroundTruthLabel(label="billing", relevance=3)
        assert gt.relevance == 3

    def test_frozen(self) -> None:
        gt = GroundTruthLabel(label="billing")
        try:
            gt.label = "cancel"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# ClassificationExample
# ---------------------------------------------------------------------------


class TestClassificationExample:
    def test_construction(self) -> None:
        ex = _make_example()
        assert ex.example_id == "ex_001"
        assert ex.text == "I have a billing issue"
        assert ex.source_type == "context_window"

    def test_label_set(self) -> None:
        ex = _make_example(labels=["billing", "refund"])
        assert ex.label_set == {"billing", "refund"}

    def test_label_relevance_map(self) -> None:
        ex = ClassificationExample(
            example_id="ex",
            text="text",
            ground_truth=[
                GroundTruthLabel(label="billing", relevance=2),
                GroundTruthLabel(label="cancel", relevance=1),
            ],
        )
        assert ex.label_relevance_map == {"billing": 2, "cancel": 1}

    def test_default_source_type(self) -> None:
        ex = _make_example()
        assert ex.source_type == "context_window"

    def test_custom_source_type(self) -> None:
        ex = _make_example(source_type="turn")
        assert ex.source_type == "turn"

    def test_metadata_default_empty(self) -> None:
        ex = _make_example()
        assert ex.metadata == {}

    def test_metadata_preserved(self) -> None:
        ex = ClassificationExample(
            example_id="ex",
            text="text",
            ground_truth=[],
            metadata={"difficulty": "hard"},
        )
        assert ex.metadata["difficulty"] == "hard"

    def test_frozen(self) -> None:
        ex = _make_example()
        try:
            ex.text = "new text"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# ClassificationDataset
# ---------------------------------------------------------------------------


class TestClassificationDataset:
    def test_construction(self) -> None:
        ds = _make_dataset()
        assert ds.name == "test-dataset"
        assert ds.version == "1.0"
        assert len(ds.examples) == 3
        assert ds.label_names == ["billing", "cancel", "refund"]
        assert ds.description == "Test classification dataset"

    def test_frozen(self) -> None:
        ds = _make_dataset()
        try:
            ds.name = "other"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


class TestDatasetSerialization:
    def test_save_creates_valid_json(self, tmp_path) -> None:
        ds = _make_dataset()
        path = tmp_path / "dataset.json"
        ds.save(path)
        data = json.loads(path.read_text())
        assert data["name"] == "test-dataset"
        assert len(data["examples"]) == 3
        assert data["label_names"] == ["billing", "cancel", "refund"]

    def test_round_trip_preserves_data(self, tmp_path) -> None:
        ds = _make_dataset()
        path = tmp_path / "dataset.json"
        ds.save(path)
        loaded = ClassificationDataset.load(path)
        assert loaded.name == ds.name
        assert loaded.version == ds.version
        assert loaded.description == ds.description
        assert loaded.label_names == ds.label_names
        assert len(loaded.examples) == len(ds.examples)

    def test_round_trip_preserves_labels(self, tmp_path) -> None:
        ds = _make_dataset()
        path = tmp_path / "dataset.json"
        ds.save(path)
        loaded = ClassificationDataset.load(path)
        # Multi-label example
        assert loaded.examples[2].label_set == {"billing", "refund"}

    def test_round_trip_preserves_source_type(self, tmp_path) -> None:
        ds = ClassificationDataset(
            name="test",
            version="1.0",
            examples=[_make_example(source_type="turn")],
            label_names=["billing"],
        )
        path = tmp_path / "dataset.json"
        ds.save(path)
        loaded = ClassificationDataset.load(path)
        assert loaded.examples[0].source_type == "turn"

    def test_round_trip_preserves_metadata(self, tmp_path) -> None:
        ex = ClassificationExample(
            example_id="ex",
            text="text",
            ground_truth=[GroundTruthLabel(label="billing")],
            metadata={"conv_id": "c1"},
        )
        ds = ClassificationDataset(
            name="test",
            version="1.0",
            examples=[ex],
            label_names=["billing"],
        )
        path = tmp_path / "dataset.json"
        ds.save(path)
        loaded = ClassificationDataset.load(path)
        assert loaded.examples[0].metadata["conv_id"] == "c1"

    def test_round_trip_preserves_relevance(self, tmp_path) -> None:
        ex = ClassificationExample(
            example_id="ex",
            text="text",
            ground_truth=[GroundTruthLabel(label="billing", relevance=3)],
        )
        ds = ClassificationDataset(
            name="test",
            version="1.0",
            examples=[ex],
            label_names=["billing"],
        )
        path = tmp_path / "dataset.json"
        ds.save(path)
        loaded = ClassificationDataset.load(path)
        assert loaded.examples[0].ground_truth[0].relevance == 3


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestDatasetReexport:
    def test_importable_from_package(self) -> None:
        from talkex.classification_eval import (
            ClassificationDataset as CD,
        )
        from talkex.classification_eval import (
            ClassificationExample as CE,
        )
        from talkex.classification_eval import (
            GroundTruthLabel as GT,
        )

        assert CD is ClassificationDataset
        assert CE is ClassificationExample
        assert GT is GroundTruthLabel
