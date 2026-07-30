"""Data-lifecycle export application service (M7 D3/D4).

Reads the to-be-purged window, anonymizes each turn's text via the `Redactor`, writes an anonymized
Parquet sample via the `SamplePort`, and — only after a verified export — purges the raw chunks. The
ordering is the guarantee (export-before-purge): if the export raises, `export_then_purge` never reaches
the drop, so no un-exported data is ever lost.
"""

from __future__ import annotations

from datetime import datetime

from talkex.monitoring.domain.lifecycle import ExportResult, RetrainingSample
from talkex.monitoring.domain.ports import LifecycleReadPort, Redactor, SamplePort


class DataLifecycleExporter:
    """Exports an anonymized retraining sample, then (optionally) purges the exported window."""

    def __init__(self, reader: LifecycleReadPort, redactor: Redactor, store: SamplePort) -> None:
        self._reader = reader
        self._redactor = redactor
        self._store = store

    async def export_window(self, from_time: datetime, to_time: datetime, name: str) -> ExportResult:
        rows = await self._reader.read_export_rows(from_time, to_time)
        samples = [
            RetrainingSample(
                turn_id=turn_id,
                conversation_id=conversation_id,
                redacted_text=self._redactor.redact(raw_text),
                label=label,
            )
            for (turn_id, conversation_id, raw_text, label) in rows
        ]
        path = self._store.write(samples, name)  # raises on failure → the purge below is never reached
        return ExportResult(path=path, row_count=len(samples))

    async def export_then_purge(
        self, from_time: datetime, to_time: datetime, name: str, purge_older_than: datetime
    ) -> ExportResult:
        result = await self.export_window(from_time, to_time, name)  # export-before-purge: export first
        await self._reader.purge_before(purge_older_than)  # only reached after a successful export
        return result
