"""Rule engine benchmark runner — compares configurations over evaluation inputs.

Executes a set of compiled rules against a list of evaluation inputs under
different RuleEngineConfig configurations, computing operational metrics
per configuration for comparison.

The runner is evaluator-agnostic: any object satisfying the RuleEvaluator
protocol (or exposing an ``evaluate(rules, input, config)`` method) can
be benchmarked.

Results are collected into a RuleExperimentReport for comparison and
serialization (JSON/CSV).

Usage::

    runner = RuleBenchmarkRunner(
        rules=[rule1, rule2],
        inputs=[input1, input2],
    )
    report = runner.compare({
        "ALL_COST": config_all_cost,
        "SC_DECL": config_sc_decl,
    })
    report.save_json("benchmark.json")
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from talkex.rules.config import RuleEngineConfig
from talkex.rules.metrics import (
    compute_rule_metrics,
)
from talkex.rules.models import (
    RuleDefinition,
    RuleEvaluationInput,
    RuleResult,
)
from talkex.rules.report import (
    ConfigurationResult,
    RuleExperimentReport,
    RuleMetrics,
)


class _RuleEvaluator(Protocol):
    """Structural type for any rule evaluator with an evaluate method."""

    def evaluate(
        self,
        rules: list[RuleDefinition],
        evaluation_input: RuleEvaluationInput,
        config: RuleEngineConfig,
    ) -> list[RuleResult]: ...


@dataclass(frozen=True)
class RuleBenchmarkConfig:
    """Configuration for a rule benchmark run.

    Args:
        experiment_name: Name for the benchmark experiment.
        experiment_version: Version string for reproducibility.
    """

    experiment_name: str = "rule_engine_benchmark"
    experiment_version: str = "1.0"


@dataclass
class RuleBenchmarkRunner:
    """Runs rule engine configuration benchmarks.

    Evaluates a set of rules against inputs under different configurations,
    computing operational metrics per configuration for comparison.

    Args:
        rules: Compiled rule definitions to evaluate.
        inputs: Evaluation inputs to test against.
        config: Benchmark configuration.
    """

    rules: list[RuleDefinition]
    inputs: list[RuleEvaluationInput]
    config: RuleBenchmarkConfig = field(default_factory=RuleBenchmarkConfig)

    def evaluate(
        self,
        evaluator: _RuleEvaluator,
        engine_config: RuleEngineConfig,
        config_name: str,
    ) -> ConfigurationResult:
        """Evaluate rules under a single engine configuration.

        Args:
            evaluator: The rule evaluator to use.
            engine_config: Engine configuration to benchmark.
            config_name: Human-readable name for this configuration.

        Returns:
            ConfigurationResult with per-rule and aggregated metrics.
        """
        all_results: list[RuleResult] = []
        start = time.monotonic()

        for evaluation_input in self.inputs:
            results = evaluator.evaluate(self.rules, evaluation_input, engine_config)
            all_results.extend(results)

        elapsed_ms = round((time.monotonic() - start) * 1000, 2)

        # Per-rule metrics
        rule_metrics_list = [
            RuleMetrics(
                rule_id=r.rule_id,
                rule_name=r.rule_name,
                matched=r.matched,
                score=r.score,
                predicate_count=len(r.predicate_results),
                predicates_matched=sum(1 for pr in r.predicate_results if pr.matched),
                short_circuited=r.short_circuited,
                execution_time_ms=r.execution_time_ms,
            )
            for r in all_results
        ]

        # Aggregated metrics
        aggregated = compute_rule_metrics(all_results)

        # Config params for traceability
        config_params = {
            "evaluation_mode": engine_config.evaluation_mode.value,
            "evidence_policy": engine_config.evidence_policy.value,
            "short_circuit_policy": engine_config.short_circuit_policy.value,
        }

        # Extract predicate type distribution from aggregated
        per_predicate_type: dict[str, int] = {}
        dist = aggregated.get("predicate_type_distribution")
        if isinstance(dist, dict):
            per_predicate_type = {k: v for k, v in dist.items() if isinstance(v, int)}

        return ConfigurationResult(
            config_name=config_name,
            config_params=config_params,
            rule_metrics=rule_metrics_list,
            aggregated=_strip_distribution(aggregated),
            per_predicate_type=per_predicate_type,
            total_rules=len(self.rules),
            total_inputs=len(self.inputs),
            total_ms=elapsed_ms,
        )

    def compare(
        self,
        evaluator: _RuleEvaluator,
        configurations: Mapping[str, RuleEngineConfig],
    ) -> RuleExperimentReport:
        """Evaluate rules under multiple configurations and produce a comparison report.

        Args:
            evaluator: The rule evaluator to use.
            configurations: Mapping from config name to engine configuration.

        Returns:
            RuleExperimentReport comparing all configurations.
        """
        results = [
            self.evaluate(evaluator, engine_config, config_name)
            for config_name, engine_config in configurations.items()
        ]
        return RuleExperimentReport(
            experiment_name=self.config.experiment_name,
            experiment_version=self.config.experiment_version,
            results=results,
        )


def _strip_distribution(aggregated: dict[str, object]) -> dict[str, Any]:
    """Remove predicate_type_distribution from aggregated dict.

    The distribution is stored separately in ConfigurationResult.per_predicate_type,
    so we strip it from the aggregated dict to avoid duplication in CSV export.
    """
    return {k: v for k, v in aggregated.items() if k != "predicate_type_distribution"}
