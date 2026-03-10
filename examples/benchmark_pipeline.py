"""Example: Benchmark pipeline configurations.

Demonstrates how to compare different pipeline configurations
(text-only vs with-embeddings vs full) using SystemBenchmarkRunner.

Usage:
    python examples/benchmark_pipeline.py
"""

from pathlib import Path

from talkex.pipeline.benchmark import SystemBenchmarkRunner
from talkex.pipeline.config import EmbeddingConfig, PipelineConfig
from talkex.pipeline.runner import PipelineRunner
from talkex.pipeline.system_pipeline import SystemPipelineResult

TRANSCRIPT_PATH = Path(__file__).parent.parent / "tests" / "fixtures" / "sample_transcript.txt"
OUTPUT_DIR = Path(__file__).parent / "output"


def _scenario_text_only() -> SystemPipelineResult:
    """Text processing only — no embeddings, no rules."""
    runner = PipelineRunner()
    return runner.run_file(
        TRANSCRIPT_PATH,
        enable_embeddings=False,
        enable_rules=False,
    ).result


def _scenario_with_rules() -> SystemPipelineResult:
    """Text processing + rule evaluation."""
    runner = PipelineRunner()
    return runner.run_file(
        TRANSCRIPT_PATH,
        rules_text=['keyword("billing")', 'keyword("cancel")'],
        enable_embeddings=False,
    ).result


def _scenario_full() -> SystemPipelineResult:
    """Full pipeline: text + embeddings + rules."""
    config = PipelineConfig(
        embedding=EmbeddingConfig(dimensions=32),
    )
    runner = PipelineRunner(config=config)
    return runner.run_file(
        TRANSCRIPT_PATH,
        rules_text=['keyword("billing")', 'keyword("cancel")'],
    ).result


def main() -> None:
    bench_runner = SystemBenchmarkRunner()

    print("Running benchmark scenarios...")
    report = bench_runner.compare(
        {
            "text_only": _scenario_text_only,
            "with_rules": _scenario_with_rules,
            "full": _scenario_full,
        },
        scenario_params={
            "text_only": {"embeddings": "no", "rules": "no"},
            "with_rules": {"embeddings": "no", "rules": "yes"},
            "full": {"embeddings": "yes", "rules": "yes"},
        },
    )

    # Print comparison
    print(f"\nScenarios:   {report.total_runs}")
    print(f"Total time:  {report.total_ms:.2f} ms\n")

    print(f"{'Scenario':<20} {'Time (ms)':>10} {'Stages':>8} {'Skipped':>8}")
    print("-" * 50)
    for r in report.results:
        print(f"{r.scenario_name:<20} {r.total_ms:>10.2f} {r.stages_executed:>8} {r.stages_skipped:>8}")

    # Save reports
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report.save_json(OUTPUT_DIR / "benchmark.json")
    report.save_csv(OUTPUT_DIR / "benchmark.csv")
    print(f"\nReport saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
