"""CLI entrypoint for the TalkEx — Conversation Intelligence Engine.

Provides operational commands for running pipelines, benchmarks, and
exporting results. Separated from core pipeline logic (SRP).

Usage::

    # Run pipeline on a transcript file
    talkex run transcript.txt --channel voice --format labeled

    # Run pipeline with custom config
    talkex run transcript.txt --config pipeline_config.json

    # Run benchmark comparing scenarios
    talkex benchmark transcript.txt --output benchmark_output/

    # Export pipeline config template
    talkex config --export template.json

    # Show version
    talkex version
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from talkex import __version__
from talkex.pipeline.config import PipelineConfig
from talkex.pipeline.runner import PipelineRunner


@click.group()
def cli() -> None:
    """TalkEx — Conversation Intelligence Engine — CLI."""


@cli.command()
def version() -> None:
    """Show the engine version."""
    click.echo(f"talkex {__version__}")


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--config", "config_path", type=click.Path(), default=None, help="JSON config file.")
@click.option("--channel", default="voice", type=click.Choice(["voice", "chat", "email"]), help="Channel type.")
@click.option(
    "--format", "source_format", default="labeled", type=click.Choice(["labeled", "plain"]), help="Transcript format."
)
@click.option("--output", "output_dir", default="output", help="Output directory.")
@click.option("--conversation-id", default=None, help="Custom conversation ID.")
@click.option("--no-embeddings", is_flag=True, default=False, help="Skip embedding generation.")
@click.option("--no-rules", is_flag=True, default=False, help="Skip rule evaluation.")
@click.option("--rule", "rules", multiple=True, help="DSL rule string (repeatable).")
def run(
    input_path: str,
    config_path: str | None,
    channel: str,
    source_format: str,
    output_dir: str,
    conversation_id: str | None,
    no_embeddings: bool,
    no_rules: bool,
    rules: tuple[str, ...],
) -> None:
    """Run the pipeline on a transcript file."""
    config = _load_config(config_path, output_dir)
    runner = PipelineRunner(config=config)

    try:
        summary = runner.run_file(
            input_path,
            channel=channel,
            source_format=source_format,
            conversation_id=conversation_id,
            rules_text=list(rules) if rules else None,
            enable_embeddings=not no_embeddings,
            enable_rules=not no_rules,
        )
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Persist outputs
    run_dir = PipelineRunner.save_outputs(summary, output_dir)

    click.echo(f"Run ID:             {summary.run_id}")
    click.echo(f"Turns:              {summary.turns_count}")
    click.echo(f"Windows:            {summary.windows_count}")
    click.echo(f"Embeddings:         {summary.embeddings_count}")
    click.echo(f"Predictions:        {summary.predictions_count}")
    click.echo(f"Rule executions:    {summary.rule_executions_count}")
    click.echo(f"Stages executed:    {summary.stages_executed}")
    click.echo(f"Stages skipped:     {summary.stages_skipped}")
    click.echo(f"Total time:         {summary.total_ms:.2f} ms")
    click.echo(f"Output:             {run_dir}")


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--config", "config_path", type=click.Path(), default=None, help="JSON config file.")
@click.option("--output", "output_dir", default="output", help="Output directory.")
@click.option("--channel", default="voice", type=click.Choice(["voice", "chat", "email"]), help="Channel type.")
@click.option(
    "--format", "source_format", default="labeled", type=click.Choice(["labeled", "plain"]), help="Transcript format."
)
def benchmark(
    input_path: str,
    config_path: str | None,
    output_dir: str,
    channel: str,
    source_format: str,
) -> None:
    """Run benchmark comparing pipeline configurations."""
    from talkex.pipeline.benchmark import SystemBenchmarkRunner

    config = _load_config(config_path, output_dir)

    # Build scenarios: text-only vs with-embeddings
    def _text_only() -> object:
        runner = PipelineRunner(config=config)
        return runner.run_file(
            input_path,
            channel=channel,
            source_format=source_format,
            enable_embeddings=False,
            enable_rules=False,
        ).result

    def _with_embeddings() -> object:
        runner = PipelineRunner(config=config)
        return runner.run_file(
            input_path,
            channel=channel,
            source_format=source_format,
            enable_embeddings=True,
            enable_rules=False,
        ).result

    def _full() -> object:
        runner = PipelineRunner(config=config)
        return runner.run_file(
            input_path,
            channel=channel,
            source_format=source_format,
            enable_embeddings=True,
            enable_rules=True,
        ).result

    try:
        bench_runner = SystemBenchmarkRunner()
        report = bench_runner.compare(
            {
                "text_only": _text_only,  # type: ignore[dict-item]
                "with_embeddings": _with_embeddings,  # type: ignore[dict-item]
                "full": _full,  # type: ignore[dict-item]
            }
        )
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Persist report
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report.save_json(out_path / "benchmark.json")
    report.save_csv(out_path / "benchmark.csv")

    click.echo(f"Scenarios:          {report.total_runs}")
    click.echo(f"Total time:         {report.total_ms:.2f} ms")
    for r in report.results:
        click.echo(f"  {r.scenario_name}: {r.total_ms:.2f} ms ({r.stages_executed} stages)")
    click.echo(f"Report:             {out_path / 'benchmark.json'}")


@cli.command("config")
@click.option("--export", "export_path", type=click.Path(), default=None, help="Export config template to file.")
@click.option("--validate", "validate_path", type=click.Path(exists=True), default=None, help="Validate a config file.")
def config_cmd(
    export_path: str | None,
    validate_path: str | None,
) -> None:
    """Export or validate pipeline configuration."""
    if export_path:
        config = PipelineConfig()
        config.save_json(export_path)
        click.echo(f"Config template exported to: {export_path}")
    elif validate_path:
        try:
            config = PipelineConfig.from_json(validate_path)
            click.echo(f"Config valid: {validate_path}")
            click.echo(json.dumps(config.to_dict(), indent=2))
        except (ValueError, FileNotFoundError) as e:
            click.echo(f"Config invalid: {e}", err=True)
            sys.exit(1)
    else:
        # Print current defaults
        config = PipelineConfig()
        click.echo(config.to_json())


def _load_config(config_path: str | None, output_dir: str) -> PipelineConfig:
    """Load config from file or create default with output_dir override."""
    if config_path:
        config = PipelineConfig.from_json(config_path)
        # Override output_dir if specified on CLI
        if output_dir != "output":
            config = config.model_copy(update={"output_dir": output_dir})
        return config
    return PipelineConfig(output_dir=output_dir)


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
