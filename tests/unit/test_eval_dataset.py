"""Unit tests for evaluation dataset schema.

Tests cover: RelevanceJudgment, EvaluationExample properties,
EvaluationDataset save/load round-trip, and reexport.
"""

import json
from pathlib import Path

from talkex.evaluation.dataset import (
    EvaluationDataset,
    EvaluationExample,
    RelevanceJudgment,
)

# ---------------------------------------------------------------------------
# RelevanceJudgment
# ---------------------------------------------------------------------------


class TestRelevanceJudgment:
    def test_creates_with_defaults(self) -> None:
        j = RelevanceJudgment(document_id="doc_1")
        assert j.document_id == "doc_1"
        assert j.relevance == 1

    def test_creates_with_graded_relevance(self) -> None:
        j = RelevanceJudgment(document_id="doc_1", relevance=3)
        assert j.relevance == 3

    def test_is_frozen(self) -> None:
        j = RelevanceJudgment(document_id="doc_1")
        try:
            j.document_id = "other"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# EvaluationExample
# ---------------------------------------------------------------------------


class TestEvaluationExample:
    def test_creates_with_required_fields(self) -> None:
        ex = EvaluationExample(
            query_id="q1",
            query_text="billing issue",
            relevant_docs=[RelevanceJudgment(document_id="doc_1")],
        )
        assert ex.query_id == "q1"
        assert ex.query_text == "billing issue"
        assert len(ex.relevant_docs) == 1

    def test_relevant_doc_ids_property(self) -> None:
        ex = EvaluationExample(
            query_id="q1",
            query_text="billing",
            relevant_docs=[
                RelevanceJudgment(document_id="d1"),
                RelevanceJudgment(document_id="d2"),
            ],
        )
        assert ex.relevant_doc_ids == {"d1", "d2"}

    def test_relevance_map_property(self) -> None:
        ex = EvaluationExample(
            query_id="q1",
            query_text="billing",
            relevant_docs=[
                RelevanceJudgment(document_id="d1", relevance=2),
                RelevanceJudgment(document_id="d2", relevance=1),
            ],
        )
        assert ex.relevance_map == {"d1": 2, "d2": 1}

    def test_metadata_defaults_to_empty(self) -> None:
        ex = EvaluationExample(
            query_id="q1",
            query_text="billing",
            relevant_docs=[],
        )
        assert ex.metadata == {}

    def test_metadata_preserved(self) -> None:
        ex = EvaluationExample(
            query_id="q1",
            query_text="billing",
            relevant_docs=[],
            metadata={"category": "payment"},
        )
        assert ex.metadata["category"] == "payment"


# ---------------------------------------------------------------------------
# EvaluationDataset
# ---------------------------------------------------------------------------


class TestEvaluationDataset:
    def test_creates_with_required_fields(self) -> None:
        ds = EvaluationDataset(
            name="test-dataset",
            version="1.0",
            examples=[],
        )
        assert ds.name == "test-dataset"
        assert ds.version == "1.0"
        assert ds.examples == []
        assert ds.description == ""

    def test_creates_with_description(self) -> None:
        ds = EvaluationDataset(
            name="test",
            version="1.0",
            examples=[],
            description="Test dataset for billing queries",
        )
        assert ds.description == "Test dataset for billing queries"

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        examples = [
            EvaluationExample(
                query_id="q1",
                query_text="billing issue",
                relevant_docs=[
                    RelevanceJudgment(document_id="d1", relevance=2),
                    RelevanceJudgment(document_id="d2"),
                ],
                metadata={"category": "payment"},
            ),
            EvaluationExample(
                query_id="q2",
                query_text="cancel subscription",
                relevant_docs=[RelevanceJudgment(document_id="d3")],
            ),
        ]
        original = EvaluationDataset(
            name="billing-eval",
            version="1.0",
            examples=examples,
            description="Billing evaluation set",
        )

        path = tmp_path / "dataset.json"
        original.save(path)
        loaded = EvaluationDataset.load(path)

        assert loaded.name == original.name
        assert loaded.version == original.version
        assert loaded.description == original.description
        assert len(loaded.examples) == 2
        assert loaded.examples[0].query_id == "q1"
        assert loaded.examples[0].query_text == "billing issue"
        assert len(loaded.examples[0].relevant_docs) == 2
        assert loaded.examples[0].relevant_docs[0].relevance == 2
        assert loaded.examples[0].metadata == {"category": "payment"}
        assert loaded.examples[1].query_id == "q2"

    def test_save_produces_valid_json(self, tmp_path: Path) -> None:
        ds = EvaluationDataset(
            name="test",
            version="1.0",
            examples=[
                EvaluationExample(
                    query_id="q1",
                    query_text="test",
                    relevant_docs=[RelevanceJudgment(document_id="d1")],
                ),
            ],
        )
        path = tmp_path / "dataset.json"
        ds.save(path)

        # Should be valid JSON
        data = json.loads(path.read_text())
        assert data["name"] == "test"
        assert len(data["examples"]) == 1


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestDatasetReexport:
    def test_importable_from_evaluation_package(self) -> None:
        from talkex.evaluation import (
            EvaluationDataset as ED,
        )
        from talkex.evaluation import (
            EvaluationExample as EE,
        )
        from talkex.evaluation import (
            RelevanceJudgment as RJ,
        )

        assert ED is EvaluationDataset
        assert EE is EvaluationExample
        assert RJ is RelevanceJudgment
