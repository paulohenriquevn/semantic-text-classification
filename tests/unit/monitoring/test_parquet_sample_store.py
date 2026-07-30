"""Unit tests for the M7 Parquet sample store (T1.1) — round-trip + fail-fast on unwritable path."""

import pytest

from talkex.monitoring.domain.lifecycle import RetrainingSample
from talkex.monitoring.infrastructure.parquet_sample_store import ParquetSampleStore, read_samples


class TestParquetSampleStore:
    def test_parquet_roundtrip(self, tmp_path) -> None:
        store = ParquetSampleStore(str(tmp_path))
        rows = [
            RetrainingSample(turn_id="t1", conversation_id="c1", redacted_text="quero cancelar", label="negative"),
            RetrainingSample(turn_id="t2", conversation_id="c1", redacted_text="obrigado", label=None),
        ]
        path = store.write(rows, "sample_2026")

        back = read_samples(path)
        assert len(back) == 2
        assert back[0].turn_id == "t1"
        assert back[0].redacted_text == "quero cancelar"
        assert back[1].label is None  # nullable label round-trips

    def test_write_fails_fast_on_unwritable_path(self) -> None:
        store = ParquetSampleStore("/proc/nonexistent-dir-cannot-write")
        with pytest.raises(OSError):
            store.write([RetrainingSample(turn_id="t", conversation_id="c", redacted_text="x")], "s")
