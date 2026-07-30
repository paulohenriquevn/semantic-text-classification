"""Offline data-lifecycle value objects for M7 (blueprint D3/D4/D5).

`RetrainingSample` is one anonymized (text, label) row destined for the Parquet cold sample;
`ExportResult` records a completed export (path + row count) that gates the purge; `BenchmarkResult`
records a retrain's macro-F1 vs the deployed model and whether it was promoted.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class RetrainingSample(BaseModel):
    """One anonymized row: pseudonymous join keys + redacted text + the QA label (may be None)."""

    model_config = ConfigDict(frozen=True)

    turn_id: str
    conversation_id: str
    redacted_text: str
    label: str | None = None


class ExportResult(BaseModel):
    """The outcome of an export — a verified path + row count gates the export-before-purge drop."""

    model_config = ConfigDict(frozen=True)

    path: str
    row_count: int


class BenchmarkResult(BaseModel):
    """A retrain's macro-F1 vs the deployed model, and whether the candidate was promoted."""

    model_config = ConfigDict(frozen=True)

    new_model_version: str
    new_f1: float
    deployed_f1: float | None
    promoted: bool
