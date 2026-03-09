"""Unit tests for domain type primitives."""

import numpy as np

from semantic_conversation_engine.models.types import (
    ConversationId,
    EmbeddingId,
    PredictionId,
    RuleId,
    Score,
    TurnId,
    Vector,
    WindowId,
)


class TestNewTypeIds:
    """NewType wrappers are assignable from their base type (str).

    These are compile-time (mypy) guarantees, not runtime checks.
    At runtime, NewType returns the base type unchanged.
    """

    def test_conversation_id_is_str(self) -> None:
        cid = ConversationId("conv_abc123")
        assert isinstance(cid, str)
        assert cid == "conv_abc123"

    def test_turn_id_is_str(self) -> None:
        tid = TurnId("turn_abc123")
        assert isinstance(tid, str)
        assert tid == "turn_abc123"

    def test_window_id_is_str(self) -> None:
        wid = WindowId("win_abc123")
        assert isinstance(wid, str)
        assert wid == "win_abc123"

    def test_embedding_id_is_str(self) -> None:
        eid = EmbeddingId("emb_abc123")
        assert isinstance(eid, str)
        assert eid == "emb_abc123"

    def test_prediction_id_is_str(self) -> None:
        pid = PredictionId("pred_abc123")
        assert isinstance(pid, str)
        assert pid == "pred_abc123"

    def test_rule_id_is_str(self) -> None:
        rid = RuleId("rule_abc123")
        assert isinstance(rid, str)
        assert rid == "rule_abc123"


class TestTypeAliases:
    """Type aliases are documentation aids, not runtime enforcement.

    Score is float, Vector is ndarray — these tests document that contract.
    """

    def test_score_is_float(self) -> None:
        assert Score is float
        score: Score = 0.95
        assert isinstance(score, float)

    def test_vector_is_ndarray(self) -> None:
        assert Vector is np.ndarray
        vec: Vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        assert isinstance(vec, np.ndarray)


class TestIdsImportFromModelsPackage:
    """Verify types are properly re-exported from the models package."""

    def test_all_ids_importable(self) -> None:
        from semantic_conversation_engine.models import (  # noqa: F401
            ConversationId,
            EmbeddingId,
            PredictionId,
            RuleId,
            TurnId,
            WindowId,
        )

    def test_aliases_importable(self) -> None:
        from semantic_conversation_engine.models import Score, Vector  # noqa: F401
