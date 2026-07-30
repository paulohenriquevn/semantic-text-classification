"""M7 retraining pipeline entrypoint (T3.1) — export → retrain → benchmark, producing a metrics JSON.

Runs the offline lifecycle end-to-end against a real TimescaleDB: export an anonymized Parquet sample
for a window, retrain the sentiment model on its labeled rows, benchmark macro-F1 vs the deployed model,
and write a benchmark JSON. Reproducible, `main`-guarded, single entrypoint (the pattern extracted from
the peer's `churn_model/train.py`, minus its hard-coded paths / no-versioning weaknesses).

Usage:
    python experiments/scripts/run_retraining.py --days 30 --out experiments/results/m7_retrain.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib

from talkex.classification.retraining_pipeline import RetrainingPipeline
from talkex.monitoring.application.lifecycle_exporter import DataLifecycleExporter
from talkex.monitoring.config import MonitoringConfig
from talkex.monitoring.infrastructure.lifecycle_repo import TimescaleLifecycleRepository
from talkex.monitoring.infrastructure.parquet_sample_store import ParquetSampleStore, read_samples
from talkex.monitoring.infrastructure.pii_redactor import RegexRedactor
from talkex.monitoring.infrastructure.pool import MonitoringPool


async def _run(days: int, out: str, sample_dir: str, version: str, now_iso: str) -> dict[str, object]:
    from datetime import datetime, timedelta

    to_time = datetime.fromisoformat(now_iso)
    from_time = to_time - timedelta(days=days)

    pool = MonitoringPool(MonitoringConfig().dsn, min_size=1, max_size=4)
    await pool.open()
    try:
        exporter = DataLifecycleExporter(
            TimescaleLifecycleRepository(pool), RegexRedactor(), ParquetSampleStore(sample_dir)
        )
        export = await exporter.export_window(from_time, to_time, f"retrain_{version}")
    finally:
        await pool.close()

    samples = read_samples(export.path)
    labeled = [s for s in samples if s.label is not None]
    # Hold out the last 20% of labeled rows as the eval set (deterministic — ordered by created_at).
    split = max(1, int(len(labeled) * 0.8))
    train_rows, eval_rows = labeled[:split], labeled[split:] or labeled[:1]
    _, bench = RetrainingPipeline().retrain_and_benchmark(
        train_rows,
        [s.redacted_text for s in eval_rows],
        [str(s.label) for s in eval_rows],
        new_version=version,
        deployed=None,
    )
    metrics = {
        "export_path": export.path,
        "export_rows": export.row_count,
        "labeled_rows": len(labeled),
        "new_model_version": bench.new_model_version,
        "new_f1": round(bench.new_f1, 4),
        "deployed_f1": bench.deployed_f1,
        "promoted": bench.promoted,
    }
    p = pathlib.Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(metrics, indent=2))
    return metrics


async def _main() -> None:
    parser = argparse.ArgumentParser(description="M7 retraining pipeline (export → retrain → benchmark)")
    parser.add_argument("--days", type=int, default=30, help="export window size in days")
    parser.add_argument("--out", default="experiments/results/m7_retrain.json")
    parser.add_argument("--sample-dir", default="experiments/results/m7_samples")
    parser.add_argument("--version", default="sentiment-retrain")
    parser.add_argument("--now", required=True, help="ISO timestamp for the window end (pass an explicit clock)")
    args = parser.parse_args()
    metrics = await _run(args.days, args.out, args.sample_dir, args.version, args.now)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
