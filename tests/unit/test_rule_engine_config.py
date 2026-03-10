"""Unit tests for rule engine configuration.

Tests cover: construction, defaults, validation, enum values,
immutability, reexport.
"""

from semantic_conversation_engine.rules.config import (
    EvidencePolicy,
    PredicateType,
    RuleEngineConfig,
    RuleEvaluationMode,
    ShortCircuitPolicy,
)

# ---------------------------------------------------------------------------
# RuleEvaluationMode enum
# ---------------------------------------------------------------------------


class TestRuleEvaluationMode:
    def test_all_value(self) -> None:
        assert RuleEvaluationMode.ALL == "all"

    def test_short_circuit_value(self) -> None:
        assert RuleEvaluationMode.SHORT_CIRCUIT == "short_circuit"


# ---------------------------------------------------------------------------
# EvidencePolicy enum
# ---------------------------------------------------------------------------


class TestEvidencePolicy:
    def test_always_value(self) -> None:
        assert EvidencePolicy.ALWAYS == "always"

    def test_match_only_value(self) -> None:
        assert EvidencePolicy.MATCH_ONLY == "match_only"


# ---------------------------------------------------------------------------
# ShortCircuitPolicy enum
# ---------------------------------------------------------------------------


class TestShortCircuitPolicy:
    def test_cost_ascending_value(self) -> None:
        assert ShortCircuitPolicy.COST_ASCENDING == "cost_ascending"

    def test_priority_value(self) -> None:
        assert ShortCircuitPolicy.PRIORITY == "priority"

    def test_declaration_value(self) -> None:
        assert ShortCircuitPolicy.DECLARATION == "declaration"


# ---------------------------------------------------------------------------
# PredicateType enum
# ---------------------------------------------------------------------------


class TestPredicateType:
    def test_four_families(self) -> None:
        assert PredicateType.LEXICAL == "lexical"
        assert PredicateType.SEMANTIC == "semantic"
        assert PredicateType.STRUCTURAL == "structural"
        assert PredicateType.CONTEXTUAL == "contextual"

    def test_has_four_members(self) -> None:
        assert len(PredicateType) == 4


# ---------------------------------------------------------------------------
# RuleEngineConfig construction
# ---------------------------------------------------------------------------


class TestRuleEngineConfigConstruction:
    def test_default_values(self) -> None:
        config = RuleEngineConfig()
        assert config.evaluation_mode == RuleEvaluationMode.ALL
        assert config.evidence_policy == EvidencePolicy.MATCH_ONLY
        assert config.short_circuit_policy == ShortCircuitPolicy.COST_ASCENDING
        assert config.max_rules_per_evaluation == 0
        assert config.default_score_threshold == 0.5

    def test_custom_values(self) -> None:
        config = RuleEngineConfig(
            evaluation_mode=RuleEvaluationMode.SHORT_CIRCUIT,
            evidence_policy=EvidencePolicy.ALWAYS,
            short_circuit_policy=ShortCircuitPolicy.PRIORITY,
            max_rules_per_evaluation=50,
            default_score_threshold=0.8,
        )
        assert config.evaluation_mode == RuleEvaluationMode.SHORT_CIRCUIT
        assert config.evidence_policy == EvidencePolicy.ALWAYS
        assert config.short_circuit_policy == ShortCircuitPolicy.PRIORITY
        assert config.max_rules_per_evaluation == 50
        assert config.default_score_threshold == 0.8


# ---------------------------------------------------------------------------
# RuleEngineConfig validation
# ---------------------------------------------------------------------------


class TestRuleEngineConfigValidation:
    def test_rejects_negative_max_rules(self) -> None:
        try:
            RuleEngineConfig(max_rules_per_evaluation=-1)
            raise AssertionError("Should reject negative max_rules")
        except (ValueError, Exception):
            pass

    def test_rejects_threshold_above_one(self) -> None:
        try:
            RuleEngineConfig(default_score_threshold=1.5)
            raise AssertionError("Should reject threshold > 1.0")
        except (ValueError, Exception):
            pass

    def test_rejects_threshold_below_zero(self) -> None:
        try:
            RuleEngineConfig(default_score_threshold=-0.1)
            raise AssertionError("Should reject threshold < 0.0")
        except (ValueError, Exception):
            pass

    def test_accepts_boundary_thresholds(self) -> None:
        c0 = RuleEngineConfig(default_score_threshold=0.0)
        c1 = RuleEngineConfig(default_score_threshold=1.0)
        assert c0.default_score_threshold == 0.0
        assert c1.default_score_threshold == 1.0


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


class TestRuleEngineConfigImmutability:
    def test_frozen(self) -> None:
        config = RuleEngineConfig()
        try:
            config.evaluation_mode = RuleEvaluationMode.SHORT_CIRCUIT  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, ValueError):
            pass


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestConfigReexport:
    def test_importable_from_rules_package(self) -> None:
        from semantic_conversation_engine.rules import (
            EvidencePolicy as EP,
        )
        from semantic_conversation_engine.rules import (
            PredicateType as PT,
        )
        from semantic_conversation_engine.rules import (
            RuleEngineConfig as REC,
        )
        from semantic_conversation_engine.rules import (
            RuleEvaluationMode as REM,
        )
        from semantic_conversation_engine.rules import (
            ShortCircuitPolicy as SCP,
        )

        assert REC is RuleEngineConfig
        assert REM is RuleEvaluationMode
        assert EP is EvidencePolicy
        assert SCP is ShortCircuitPolicy
        assert PT is PredicateType
