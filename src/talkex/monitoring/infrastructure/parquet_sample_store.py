"""Parquet SamplePort implementation (M7 D3).

Writes anonymized `RetrainingSample` rows to a Parquet file under a base directory (the local-dir stand-in
for object storage; the port lets a S3/MinIO impl swap in later — DIP). Fail-fast: a write error propagates
so the caller never purges raw data behind an un-written sample (export-before-purge).
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from talkex.monitoring.domain.lifecycle import RetrainingSample


class ParquetSampleStore:
    """Persists retraining samples as Parquet under `base_dir`."""

    def __init__(self, base_dir: str) -> None:
        self._base = Path(base_dir)

    def write(self, rows: list[RetrainingSample], name: str) -> str:
        self._base.mkdir(parents=True, exist_ok=True)
        path = self._base / f"{name}.parquet"
        table = pa.table(
            {
                "turn_id": [r.turn_id for r in rows],
                "conversation_id": [r.conversation_id for r in rows],
                "redacted_text": [r.redacted_text for r in rows],
                "label": [r.label for r in rows],
            }
        )
        pq.write_table(table, path)  # raises on an unwritable path — the caller must not purge then
        return str(path)


def read_samples(path: str) -> list[RetrainingSample]:
    """Read a Parquet sample file back into domain rows (used by the retraining pipeline + tests)."""
    table = pq.read_table(path)
    data = table.to_pydict()
    return [
        RetrainingSample(
            turn_id=data["turn_id"][i],
            conversation_id=data["conversation_id"][i],
            redacted_text=data["redacted_text"][i],
            label=data["label"][i],
        )
        for i in range(table.num_rows)
    ]
