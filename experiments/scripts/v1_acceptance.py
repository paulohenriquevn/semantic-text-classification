"""V1-acceptance harness (M8 T2.1/T3.1) — RE-RUN each ship criterion, emit a readiness report.

Sources each V1 criterion from LIVE evidence, never a hard-coded constant (blueprint D3):
  - retrieval p95: read the fresh `experiments/results/m5_hybrid_bench.json` (produced by the M5 benchmark)
  - alert p95, critical-alert precision, sentiment macro-F1, purge: RE-RUN the specific pytest checks and
    treat pass/fail as the criterion result (a missing artifact / failing test → FAIL, not PASS)

The verdict is PASS only when every criterion passes. The report carries the synthetic-load caveat — no
unqualified production claim (Rule 3).

Usage:
    python experiments/scripts/v1_acceptance.py --out experiments/results/v1_readiness.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

from talkex.monitoring.domain.acceptance import Criterion, build_readiness_report

_BENCH = pathlib.Path("experiments/results/m5_hybrid_bench.json")


def _read_retrieval_p95() -> float | None:
    if not _BENCH.exists():
        return None  # missing artifact → the criterion FAILs (no silent PASS)
    return float(json.loads(_BENCH.read_text())["p95_ms"])


def _pytest_passes(path: str, keyword: str) -> bool:
    """Re-run a specific pytest check (by -k keyword); True iff it passes (exit 0)."""
    proc = subprocess.run(
        ["python", "-m", "pytest", path, "-q", "-x", "-k", keyword],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def _pytest_criterion(name: str, path: str, keyword: str) -> Criterion:
    # A passing test → measured 1.0 ≥ threshold 1.0; a failing/absent test → measured 0.0 (FAIL).
    passed = _pytest_passes(path, keyword)
    return Criterion(name=name, measured=1.0 if passed else 0.0, threshold=1.0, direction="min")


def build_criteria() -> list[Criterion]:
    rules = "tests/unit/monitoring/test_critical_rules.py"
    return [
        Criterion(name="retrieval_p95_ms < 200", measured=_read_retrieval_p95(), threshold=200.0, direction="max"),
        _pytest_criterion("alert_p95 < 2s", rules, "test_turn_to_alert_p95_under_2s"),
        _pytest_criterion("critical_alert_precision >= 0.8", rules, "test_alert_precision_meets_bar"),
        _pytest_criterion(
            "sentiment_macro_f1 >= 0.70", "tests/unit/classification/test_sentiment.py", "test_macro_f1_meets_dod"
        ),
        _pytest_criterion(
            "purge_working", "tests/integration/monitoring/test_kpi_rollup.py", "test_rollup_survives_raw_chunk_drop"
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="M8 V1-acceptance harness")
    parser.add_argument("--out", default="experiments/results/v1_readiness.json")
    args = parser.parse_args()

    report = build_readiness_report(build_criteria())
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2))
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
