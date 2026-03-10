"""Unit tests for rule engine experiment reporting.

Tests cover: RuleMetrics construction, ConfigurationResult construction,
RuleExperimentReport JSON/CSV serialization, and edge cases.
"""

import json

from semantic_conversation_engine.rules.report import (
    ConfigurationResult,
    RuleExperimentReport,
    RuleMetrics,
)


def _rule_metrics(**overrides: object) -> RuleMetrics:
    defaults: dict[str, object] = {
        "rule_id": "r1",
        "rule_name": "test_rule",
        "matched": True,
        "score": 0.85,
        "predicate_count": 3,
        "predicates_matched": 2,
        "short_circuited": False,
        "execution_time_ms": 1.5,
    }
    defaults.update(overrides)
    return RuleMetrics(**defaults)  # type: ignore[arg-type]


def _config_result(**overrides: object) -> ConfigurationResult:
    defaults: dict[str, object] = {
        "config_name": "ALL_COST",
        "config_params": {"evaluation_mode": "all", "short_circuit_policy": "cost_ascending"},
        "rule_metrics": [_rule_metrics()],
        "aggregated": {"match_rate": 1.0, "short_circuit_rate": 0.0, "avg_execution_time_ms": 1.5},
        "per_predicate_type": {"lexical": 2, "semantic": 1},
        "total_rules": 1,
        "total_inputs": 1,
        "total_ms": 1.5,
    }
    defaults.update(overrides)
    return ConfigurationResult(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# RuleMetrics
# ---------------------------------------------------------------------------


class TestRuleMetrics:
    def test_construction(self) -> None:
        rm = _rule_metrics()
        assert rm.rule_id == "r1"
        assert rm.matched is True
        assert rm.score == 0.85
        assert rm.predicate_count == 3

    def test_frozen(self) -> None:
        rm = _rule_metrics()
        try:
            rm.score = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# ConfigurationResult
# ---------------------------------------------------------------------------


class TestConfigurationResult:
    def test_construction(self) -> None:
        cr = _config_result()
        assert cr.config_name == "ALL_COST"
        assert cr.total_rules == 1
        assert len(cr.rule_metrics) == 1

    def test_frozen(self) -> None:
        cr = _config_result()
        try:
            cr.total_ms = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# RuleExperimentReport — JSON
# ---------------------------------------------------------------------------


class TestReportJson:
    def test_to_json_structure(self) -> None:
        report = RuleExperimentReport(
            experiment_name="test_bench",
            experiment_version="1.0",
            results=[_config_result()],
        )
        data = json.loads(report.to_json())
        assert data["experiment_name"] == "test_bench"
        assert len(data["results"]) == 1
        r = data["results"][0]
        assert r["config_name"] == "ALL_COST"
        assert "match_rate" in r["aggregated"]
        assert "lexical" in r["per_predicate_type"]

    def test_to_json_multiple_configs(self) -> None:
        report = RuleExperimentReport(
            experiment_name="compare",
            experiment_version="2.0",
            results=[
                _config_result(config_name="ALL_COST"),
                _config_result(config_name="SC_DECL", total_ms=0.5),
            ],
        )
        data = json.loads(report.to_json())
        assert len(data["results"]) == 2
        names = [r["config_name"] for r in data["results"]]
        assert "ALL_COST" in names
        assert "SC_DECL" in names

    def test_save_json(self, tmp_path) -> None:
        report = RuleExperimentReport(
            experiment_name="save_test",
            experiment_version="1.0",
            results=[_config_result()],
        )
        path = tmp_path / "report.json"
        report.save_json(path)
        loaded = json.loads(path.read_text())
        assert loaded["experiment_name"] == "save_test"


# ---------------------------------------------------------------------------
# RuleExperimentReport — CSV
# ---------------------------------------------------------------------------


class TestReportCsv:
    def test_to_csv_structure(self) -> None:
        report = RuleExperimentReport(
            experiment_name="csv_test",
            experiment_version="1.0",
            results=[_config_result()],
        )
        csv_text = report.to_csv()
        lines = csv_text.strip().split("\n")
        assert len(lines) == 2  # header + 1 row
        assert "config_name" in lines[0]
        assert "match_rate" in lines[0]
        assert "ALL_COST" in lines[1]

    def test_to_csv_empty(self) -> None:
        report = RuleExperimentReport(
            experiment_name="empty",
            experiment_version="1.0",
            results=[],
        )
        assert report.to_csv() == ""

    def test_save_csv(self, tmp_path) -> None:
        report = RuleExperimentReport(
            experiment_name="csv_save",
            experiment_version="1.0",
            results=[_config_result()],
        )
        path = tmp_path / "report.csv"
        report.save_csv(path)
        assert path.exists()
        content = path.read_text()
        assert "ALL_COST" in content


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestReportReexport:
    def test_importable_from_rules_package(self) -> None:
        from semantic_conversation_engine.rules import (
            ConfigurationResult,
            RuleExperimentReport,
            RuleMetrics,
        )

        assert RuleMetrics is not None
        assert ConfigurationResult is not None
        assert RuleExperimentReport is not None
