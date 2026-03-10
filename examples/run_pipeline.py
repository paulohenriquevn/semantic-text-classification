"""Example: Run the full pipeline programmatically.

Demonstrates how to use PipelineRunner to process a transcript file
with embedding generation and rule evaluation.

Usage:
    python examples/run_pipeline.py
"""

from pathlib import Path

from talkex.pipeline.config import EmbeddingConfig, PipelineConfig
from talkex.pipeline.runner import PipelineRunner

# Use the sample transcript from test fixtures
TRANSCRIPT_PATH = Path(__file__).parent.parent / "tests" / "fixtures" / "sample_transcript.txt"
OUTPUT_DIR = Path(__file__).parent / "output"


def main() -> None:
    # Configure the pipeline
    config = PipelineConfig(
        context={"window_size": 3, "stride": 2},  # type: ignore[arg-type]
        embedding=EmbeddingConfig(dimensions=64),
        output_dir=str(OUTPUT_DIR),
    )

    runner = PipelineRunner(config=config)

    # Run with rule evaluation
    summary = runner.run_file(
        TRANSCRIPT_PATH,
        channel="voice",
        source_format="labeled",
        rules_text=[
            'keyword("billing")',
            'keyword("cancel")',
            'keyword("charged")',
        ],
    )

    # Print results
    print(f"Run ID:          {summary.run_id}")
    print(f"Turns:           {summary.turns_count}")
    print(f"Windows:         {summary.windows_count}")
    print(f"Embeddings:      {summary.embeddings_count}")
    print(f"Predictions:     {summary.predictions_count}")
    print(f"Rule executions: {summary.rule_executions_count}")
    print(f"Stages executed: {summary.stages_executed}")
    print(f"Stages skipped:  {summary.stages_skipped}")
    print(f"Total time:      {summary.total_ms:.2f} ms")

    # Persist outputs
    run_dir = PipelineRunner.save_outputs(summary)
    print(f"Output saved to: {run_dir}")

    # Access manifest
    manifest = summary.result.manifest
    if manifest:
        print("\nManifest:")
        print(f"  Timestamp:   {manifest.timestamp.isoformat()}")
        print(f"  Components:  {manifest.component_versions}")
        print(f"  Fingerprint: {manifest.config_fingerprint[:40]}...")

    # Access rule results
    if summary.result.rule_executions:
        print("\nRule Results:")
        for rule_exec in summary.result.rule_executions:
            print(f"  {rule_exec.rule_name}: matched={rule_exec.matched}, score={rule_exec.score}")


if __name__ == "__main__":
    main()
