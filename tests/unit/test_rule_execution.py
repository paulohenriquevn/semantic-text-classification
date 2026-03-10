"""Unit tests for the RuleExecution model — follows the Conversation golden template.

Tests cover: construction, validation (IDs, strings, score, execution_time_ms),
EvidenceItem typing, strict mode behavior, immutability, serialization round-trip,
and re-export.
"""

from typing import Any

import pytest

from talkex.models.enums import ObjectType
from talkex.models.rule_execution import EvidenceItem, RuleExecution
from talkex.models.types import RuleId

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_SAMPLE_EVIDENCE: EvidenceItem = {
    "predicate_type": "lexical",
    "matched_text": "cancel my subscription",
    "score": 0.95,
    "threshold": 0.80,
}

_SAMPLE_SEMANTIC_EVIDENCE: EvidenceItem = {
    "predicate_type": "semantic",
    "score": 0.88,
    "threshold": 0.75,
    "model_name": "e5-large-v2",
    "model_version": "1.0.0",
}


def _make_rule_execution(**overrides: object) -> RuleExecution:
    """Factory with sensible defaults. Override any field via kwargs."""
    defaults: dict[str, Any] = {
        "rule_id": RuleId("rule_abc123"),
        "rule_name": "churn_risk_cancellation",
        "source_id": "turn_abc123",
        "source_type": ObjectType.TURN,
        "matched": True,
        "score": 0.92,
        "execution_time_ms": 12.5,
        "evidence": [_SAMPLE_EVIDENCE],
    }
    defaults.update(overrides)
    return RuleExecution(**defaults)


# ---------------------------------------------------------------------------
# Construction — happy path
# ---------------------------------------------------------------------------


class TestRuleExecutionConstruction:
    def test_creates_matched_with_evidence(self) -> None:
        re = _make_rule_execution()
        assert re.rule_id == "rule_abc123"
        assert re.rule_name == "churn_risk_cancellation"
        assert re.source_id == "turn_abc123"
        assert re.source_type == ObjectType.TURN
        assert re.matched is True
        assert re.score == 0.92
        assert re.execution_time_ms == 12.5
        assert len(re.evidence) == 1
        assert re.metadata == {}

    def test_creates_unmatched_without_evidence(self) -> None:
        re = _make_rule_execution(matched=False, score=0.3, evidence=[])
        assert re.matched is False
        assert re.evidence == []

    def test_creates_with_all_fields(self) -> None:
        re = _make_rule_execution(
            evidence=[_SAMPLE_EVIDENCE, _SAMPLE_SEMANTIC_EVIDENCE],
            metadata={"rule_version": "2.1", "tags": ["compliance"]},
        )
        assert len(re.evidence) == 2
        assert re.evidence[0]["predicate_type"] == "lexical"
        assert re.metadata == {"rule_version": "2.1", "tags": ["compliance"]}

    def test_metadata_defaults_to_empty_dict(self) -> None:
        re = _make_rule_execution()
        assert re.metadata == {}
        assert isinstance(re.metadata, dict)


# ---------------------------------------------------------------------------
# Validation — IDs and strings
# ---------------------------------------------------------------------------


class TestRuleExecutionIdValidation:
    def test_rejects_empty_rule_id(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_rule_execution(rule_id=RuleId(""))

    def test_rejects_whitespace_only_rule_id(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_rule_execution(rule_id=RuleId("   "))

    def test_preserves_rule_id_without_normalizing(self) -> None:
        padded_id = RuleId("  rule_123  ")
        re = _make_rule_execution(rule_id=padded_id)
        assert re.rule_id == "  rule_123  "

    def test_rejects_empty_rule_name(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_rule_execution(rule_name="")

    def test_rejects_whitespace_only_rule_name(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_rule_execution(rule_name="   ")

    def test_rejects_empty_source_id(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _make_rule_execution(source_id="")


# ---------------------------------------------------------------------------
# Validation — score and execution_time_ms
# ---------------------------------------------------------------------------


class TestRuleExecutionScoreValidation:
    def test_rejects_negative_score(self) -> None:
        with pytest.raises(ValueError, match="score must be in"):
            _make_rule_execution(score=-0.1)

    def test_rejects_score_above_one(self) -> None:
        with pytest.raises(ValueError, match="score must be in"):
            _make_rule_execution(score=1.1)

    def test_accepts_score_zero(self) -> None:
        re = _make_rule_execution(score=0.0, matched=False)
        assert re.score == 0.0

    def test_accepts_score_one(self) -> None:
        re = _make_rule_execution(score=1.0)
        assert re.score == 1.0

    def test_rejects_negative_execution_time(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            _make_rule_execution(execution_time_ms=-1.0)

    def test_accepts_zero_execution_time(self) -> None:
        re = _make_rule_execution(execution_time_ms=0.0)
        assert re.execution_time_ms == 0.0

    def test_accepts_positive_execution_time(self) -> None:
        re = _make_rule_execution(execution_time_ms=150.3)
        assert re.execution_time_ms == 150.3


# ---------------------------------------------------------------------------
# Cross-field validation — matched requires evidence
# ---------------------------------------------------------------------------


class TestRuleExecutionCrossFieldValidation:
    """matched=True requires at least one evidence item for auditability."""

    def test_rejects_matched_true_without_evidence(self) -> None:
        with pytest.raises(ValueError, match="requires at least one evidence"):
            _make_rule_execution(matched=True, evidence=[])

    def test_accepts_matched_true_with_evidence(self) -> None:
        re = _make_rule_execution(matched=True, evidence=[_SAMPLE_EVIDENCE])
        assert re.matched is True
        assert len(re.evidence) == 1

    def test_accepts_matched_false_without_evidence(self) -> None:
        """Non-matched rules have no obligation to document why they didn't match."""
        re = _make_rule_execution(matched=False, score=0.3, evidence=[])
        assert re.matched is False
        assert re.evidence == []

    def test_accepts_matched_false_with_evidence(self) -> None:
        """Non-matched rules may still carry partial evaluation evidence."""
        re = _make_rule_execution(matched=False, score=0.4, evidence=[_SAMPLE_EVIDENCE])
        assert re.matched is False
        assert len(re.evidence) == 1

    def test_accepts_matched_true_with_low_score(self) -> None:
        """Binary/structural rules may legitimately match with low aggregate scores."""
        re = _make_rule_execution(matched=True, score=0.1, evidence=[_SAMPLE_EVIDENCE])
        assert re.matched is True
        assert re.score == 0.1

    def test_accepts_unmatched_with_high_score(self) -> None:
        """Partial predicate satisfaction can produce high score without full match."""
        re = _make_rule_execution(matched=False, score=0.9, evidence=[])
        assert re.matched is False
        assert re.score == 0.9


# ---------------------------------------------------------------------------
# EvidenceItem — typed structure
# ---------------------------------------------------------------------------


class TestEvidenceItemTyping:
    """EvidenceItem is a TypedDict with total=False — all fields optional."""

    def test_lexical_evidence_with_matched_text(self) -> None:
        re = _make_rule_execution(evidence=[_SAMPLE_EVIDENCE])
        item = re.evidence[0]
        assert item["predicate_type"] == "lexical"
        assert item["matched_text"] == "cancel my subscription"
        assert item["score"] == 0.95
        assert item["threshold"] == 0.80

    def test_semantic_evidence_with_model_info(self) -> None:
        re = _make_rule_execution(evidence=[_SAMPLE_SEMANTIC_EVIDENCE])
        item = re.evidence[0]
        assert item["predicate_type"] == "semantic"
        assert item["model_name"] == "e5-large-v2"
        assert item["model_version"] == "1.0.0"

    def test_multiple_evidence_items(self) -> None:
        re = _make_rule_execution(evidence=[_SAMPLE_EVIDENCE, _SAMPLE_SEMANTIC_EVIDENCE])
        assert len(re.evidence) == 2

    def test_minimal_evidence_item(self) -> None:
        """Only predicate_type is semantically required, but all fields are optional."""
        minimal: EvidenceItem = {"predicate_type": "structural"}
        re = _make_rule_execution(evidence=[minimal])
        assert re.evidence[0]["predicate_type"] == "structural"

    def test_evidence_with_metadata(self) -> None:
        item: EvidenceItem = {
            "predicate_type": "contextual",
            "metadata": {"window_position": 3, "repeated": True},
        }
        re = _make_rule_execution(evidence=[item])
        assert re.evidence[0]["metadata"] == {"window_position": 3, "repeated": True}


# ---------------------------------------------------------------------------
# Strict mode — no coercion
# ---------------------------------------------------------------------------


class TestRuleExecutionStrictMode:
    def test_rejects_string_source_type_coercion(self) -> None:
        """strict=True means 'turn' (str) won't coerce to ObjectType.TURN."""
        with pytest.raises(ValueError):
            _make_rule_execution(source_type="turn")

    def test_accepts_int_for_float_fields(self) -> None:
        """strict=True still allows int → float (lossless widening in Pydantic v2)."""
        re = _make_rule_execution(score=1)
        assert re.score == 1.0
        assert isinstance(re.score, int | float)

    def test_rejects_string_for_matched(self) -> None:
        """strict=True means str won't coerce to bool."""
        with pytest.raises(ValueError):
            _make_rule_execution(matched="true")

    def test_rejects_int_for_rule_id(self) -> None:
        """strict=True means int won't coerce to str."""
        with pytest.raises(ValueError):
            _make_rule_execution(rule_id=12345)


# ---------------------------------------------------------------------------
# Immutability (frozen=True)
# ---------------------------------------------------------------------------


class TestRuleExecutionImmutability:
    def test_cannot_assign_to_field(self) -> None:
        re = _make_rule_execution()
        with pytest.raises(ValueError, match="frozen"):
            re.rule_id = RuleId("rule_new")

    def test_cannot_assign_to_evidence(self) -> None:
        re = _make_rule_execution()
        with pytest.raises(ValueError, match="frozen"):
            re.evidence = [_SAMPLE_EVIDENCE]

    def test_cannot_assign_to_metadata(self) -> None:
        re = _make_rule_execution()
        with pytest.raises(ValueError, match="frozen"):
            re.metadata = {"new": "data"}


# ---------------------------------------------------------------------------
# Serialization — in-memory (types preserved)
# ---------------------------------------------------------------------------


class TestRuleExecutionSerializationInMemory:
    """In-memory serialization: model_dump() preserves Python types."""

    def test_model_dump_produces_dict(self) -> None:
        re = _make_rule_execution()
        data = re.model_dump()
        assert isinstance(data, dict)
        assert data["rule_id"] == "rule_abc123"
        assert data["matched"] is True

    def test_enum_serializes_as_value(self) -> None:
        re = _make_rule_execution(source_type=ObjectType.CONVERSATION)
        data = re.model_dump()
        assert data["source_type"] == "conversation"
        assert isinstance(data["source_type"], str)

    def test_evidence_serializes_as_list_of_dicts(self) -> None:
        re = _make_rule_execution(evidence=[_SAMPLE_EVIDENCE])
        data = re.model_dump()
        assert isinstance(data["evidence"], list)
        assert isinstance(data["evidence"][0], dict)
        assert data["evidence"][0]["predicate_type"] == "lexical"

    def test_model_dump_mode_json_produces_json_safe_types(self) -> None:
        re = _make_rule_execution(
            evidence=[_SAMPLE_EVIDENCE],
            metadata={"key": "value"},
        )
        data = re.model_dump(mode="json")
        assert isinstance(data["score"], float)
        assert isinstance(data["source_type"], str)


# ---------------------------------------------------------------------------
# Boundary deserialization (dict/JSON → model, uses parsing)
# ---------------------------------------------------------------------------


class TestRuleExecutionBoundaryDeserialization:
    """Boundary parsing: reconstruct model from dict or JSON.

    At system boundaries (API handlers, file parsers), data arrives as plain
    dicts. Pydantic's model_validate with strict=False handles coercion.
    This is the ONLY place where strict=False is acceptable (ADR-002).
    """

    def test_reconstructs_from_model_dump(self) -> None:
        re = _make_rule_execution(
            evidence=[_SAMPLE_EVIDENCE],
            metadata={"key": "value"},
        )
        data = re.model_dump()
        restored = RuleExecution.model_validate(data, strict=False)
        assert restored == re

    def test_preserves_evidence_through_boundary(self) -> None:
        re = _make_rule_execution(evidence=[_SAMPLE_EVIDENCE, _SAMPLE_SEMANTIC_EVIDENCE])
        data = re.model_dump()
        restored = RuleExecution.model_validate(data, strict=False)
        assert len(restored.evidence) == 2
        assert restored.evidence[0]["predicate_type"] == "lexical"
        assert restored.evidence[1]["predicate_type"] == "semantic"

    def test_preserves_metadata_through_boundary(self) -> None:
        meta = {"rule_version": "2.1", "execution_context": "realtime"}
        re = _make_rule_execution(metadata=meta)
        data = re.model_dump()
        restored = RuleExecution.model_validate(data, strict=False)
        assert restored.metadata == meta

    def test_reconstructs_from_json_mode_dump(self) -> None:
        """Proves full JSON round-trip: model → JSON dict → model."""
        re = _make_rule_execution(
            evidence=[_SAMPLE_EVIDENCE],
            metadata={"nested": {"deep": True}},
        )
        json_data = re.model_dump(mode="json")
        restored = RuleExecution.model_validate(json_data, strict=False)
        assert restored == re


# ---------------------------------------------------------------------------
# Re-export from models package
# ---------------------------------------------------------------------------


class TestRuleExecutionReexport:
    def test_importable_from_models_package(self) -> None:
        from talkex.models import RuleExecution as Imported

        assert Imported is RuleExecution

    def test_evidence_item_importable_from_models_package(self) -> None:
        from talkex.models import EvidenceItem as Imported

        assert Imported is EvidenceItem
