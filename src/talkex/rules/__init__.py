"""Semantic rule engine based on Abstract Syntax Trees (AST).

Compiles a human-readable DSL into AST for safe, auditable execution.
Combines four signal families: lexical, semantic, structural, and contextual.
Supports short-circuit execution ordered by predicate cost and produces
traceable evidence for every rule evaluation.

Includes operational metrics, benchmark runner for comparing configurations,
and structured reporting with JSON/CSV export.
"""

from talkex.rules.ast import (
    AndNode,
    ASTNode,
    NotNode,
    OrNode,
    PredicateNode,
)
from talkex.rules.benchmark import (
    RuleBenchmarkConfig,
    RuleBenchmarkRunner,
)
from talkex.rules.compiler import SimpleRuleCompiler
from talkex.rules.config import (
    EvidencePolicy,
    PredicateType,
    RuleEngineConfig,
    RuleEvaluationMode,
    ShortCircuitPolicy,
)
from talkex.rules.dsl import PREDICATE_REGISTRY
from talkex.rules.evaluator import (
    SimpleRuleEvaluator,
    map_to_rule_execution,
)
from talkex.rules.metrics import compute_rule_metrics
from talkex.rules.models import (
    PredicateResult,
    RuleDefinition,
    RuleEvaluationInput,
    RuleResult,
)
from talkex.rules.parser import parse_dsl
from talkex.rules.report import (
    ConfigurationResult,
    RuleExperimentReport,
    RuleMetrics,
)

__all__ = [
    "PREDICATE_REGISTRY",
    "ASTNode",
    "AndNode",
    "ConfigurationResult",
    "EvidencePolicy",
    "NotNode",
    "OrNode",
    "PredicateNode",
    "PredicateResult",
    "PredicateType",
    "RuleBenchmarkConfig",
    "RuleBenchmarkRunner",
    "RuleDefinition",
    "RuleEngineConfig",
    "RuleEvaluationInput",
    "RuleEvaluationMode",
    "RuleExperimentReport",
    "RuleMetrics",
    "RuleResult",
    "ShortCircuitPolicy",
    "SimpleRuleCompiler",
    "SimpleRuleEvaluator",
    "compute_rule_metrics",
    "map_to_rule_execution",
    "parse_dsl",
]
