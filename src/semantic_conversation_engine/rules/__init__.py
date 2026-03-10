"""Semantic rule engine based on Abstract Syntax Trees (AST).

Compiles a human-readable DSL into AST for safe, auditable execution.
Combines four signal families: lexical, semantic, structural, and contextual.
Supports short-circuit execution ordered by predicate cost and produces
traceable evidence for every rule evaluation.
"""

from semantic_conversation_engine.rules.ast import (
    AndNode,
    ASTNode,
    NotNode,
    OrNode,
    PredicateNode,
)
from semantic_conversation_engine.rules.compiler import SimpleRuleCompiler
from semantic_conversation_engine.rules.config import (
    EvidencePolicy,
    PredicateType,
    RuleEngineConfig,
    RuleEvaluationMode,
    ShortCircuitPolicy,
)
from semantic_conversation_engine.rules.dsl import PREDICATE_REGISTRY
from semantic_conversation_engine.rules.evaluator import (
    SimpleRuleEvaluator,
    map_to_rule_execution,
)
from semantic_conversation_engine.rules.models import (
    PredicateResult,
    RuleDefinition,
    RuleEvaluationInput,
    RuleResult,
)
from semantic_conversation_engine.rules.parser import parse_dsl

__all__ = [
    "PREDICATE_REGISTRY",
    "ASTNode",
    "AndNode",
    "EvidencePolicy",
    "NotNode",
    "OrNode",
    "PredicateNode",
    "PredicateResult",
    "PredicateType",
    "RuleDefinition",
    "RuleEngineConfig",
    "RuleEvaluationInput",
    "RuleEvaluationMode",
    "RuleResult",
    "ShortCircuitPolicy",
    "SimpleRuleCompiler",
    "SimpleRuleEvaluator",
    "map_to_rule_execution",
    "parse_dsl",
]
