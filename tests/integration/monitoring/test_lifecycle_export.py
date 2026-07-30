"""Integration tests for M7 data-lifecycle export against a real TimescaleDB (T1.1 / T3.1).

Seeds turns carrying PII + QA labels, exports an anonymized Parquet, and asserts (a) the LGPD gate — no
raw PII in the exported file — and (b) export-before-purge: the raw window is dropped only after a
verified export, and a failed export leaves the raw data intact.
"""

import re

import psycopg
import pytest
from psycopg.types.json import Jsonb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from talkex.classification.retraining_pipeline import RetrainingPipeline
from talkex.classification.sentiment import SentimentDetector
from talkex.monitoring.application.lifecycle_exporter import DataLifecycleExporter
from talkex.monitoring.config import MonitoringConfig
from talkex.monitoring.infrastructure.lifecycle_repo import TimescaleLifecycleRepository
from talkex.monitoring.infrastructure.parquet_sample_store import ParquetSampleStore, read_samples
from talkex.monitoring.infrastructure.pii_redactor import RegexRedactor
from talkex.monitoring.infrastructure.pool import MonitoringPool

pytestmark = pytest.mark.integration

DSN = MonitoringConfig().dsn
_CPF = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


async def _seed_turn(conn: psycopg.AsyncConnection, turn_id: str, text: str, days_ago: int) -> None:
    await conn.execute(
        "INSERT INTO turns (turn_id, conversation_id, speaker, raw_text, normalized_text, "
        "start_offset, end_offset, metadata, created_at) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now() - make_interval(days => %s))",
        (turn_id, "conv_lc", "customer", text, None, 0, len(text), Jsonb({}), days_ago),
    )
    await conn.commit()


class TestLifecycleExport:
    async def test_export_is_anonymized(self, conn: psycopg.AsyncConnection, tmp_path) -> None:
        await _seed_turn(conn, "lc_1", "meu CPF é 123.456.789-09 e email ana@example.com", days_ago=1)
        await _seed_turn(conn, "lc_2", "quero cancelar o plano", days_ago=1)
        # a QA label on lc_1
        await conn.execute(
            "INSERT INTO labels (label_id, turn_id, conversation_id, label) VALUES (%s, %s, %s, %s)",
            ("lbl_lc", "lc_1", "conv_lc", "negative"),
        )
        await conn.commit()

        pool = MonitoringPool(DSN, min_size=1, max_size=2)
        await pool.open()
        try:
            exporter = DataLifecycleExporter(
                TimescaleLifecycleRepository(pool), RegexRedactor(), ParquetSampleStore(str(tmp_path))
            )
            from datetime import UTC, datetime, timedelta

            result = await exporter.export_window(datetime.now(UTC) - timedelta(days=2), datetime.now(UTC), "m7_sample")
        finally:
            await pool.close()

        assert result.row_count == 2
        samples = read_samples(result.path)
        joined = " ".join(s.redacted_text for s in samples)
        assert _CPF.search(joined) is None, f"CPF leaked into the cold sample: {joined}"  # LGPD gate
        assert _EMAIL.search(joined) is None, f"email leaked into the cold sample: {joined}"
        assert any(s.label == "negative" for s in samples)  # the QA label joined through

    async def test_export_then_purge_ordering(self, conn: psycopg.AsyncConnection, tmp_path) -> None:
        from datetime import UTC, datetime, timedelta

        await _seed_turn(conn, "lc_old", "quero cancelar", days_ago=40)
        pool = MonitoringPool(DSN, min_size=1, max_size=2)
        await pool.open()
        try:
            exporter = DataLifecycleExporter(
                TimescaleLifecycleRepository(pool), RegexRedactor(), ParquetSampleStore(str(tmp_path))
            )
            result = await exporter.export_then_purge(
                datetime.now(UTC) - timedelta(days=41),
                datetime.now(UTC) - timedelta(days=30),
                "m7_old",
                purge_older_than=datetime.now(UTC) - timedelta(days=30),
            )
        finally:
            await pool.close()

        # The sample was exported (the cold copy exists) BEFORE the raw chunk was dropped.
        assert read_samples(result.path)  # cold sample persisted
        await conn.commit()
        cur = await conn.execute("SELECT count(*) FROM turns WHERE turn_id = 'lc_old'")
        assert (await cur.fetchone())[0] == 0  # raw purged — only after the verified export

    async def test_failed_export_does_not_purge(self, conn: psycopg.AsyncConnection) -> None:
        from datetime import UTC, datetime, timedelta

        await _seed_turn(conn, "lc_keep", "quero cancelar", days_ago=40)
        pool = MonitoringPool(DSN, min_size=1, max_size=2)
        await pool.open()
        try:
            # An unwritable store makes the export raise → the purge must NOT run.
            exporter = DataLifecycleExporter(
                TimescaleLifecycleRepository(pool),
                RegexRedactor(),
                ParquetSampleStore("/proc/nonexistent-dir-cannot-write"),
            )
            with pytest.raises(OSError):
                await exporter.export_then_purge(
                    datetime.now(UTC) - timedelta(days=41),
                    datetime.now(UTC) - timedelta(days=30),
                    "m7_fail",
                    purge_older_than=datetime.now(UTC) - timedelta(days=30),
                )
        finally:
            await pool.close()

        await conn.commit()
        cur = await conn.execute("SELECT count(*) FROM turns WHERE turn_id = 'lc_keep'")
        assert (await cur.fetchone())[0] == 1  # raw intact — a failed export never purges

    async def test_end_to_end_export_then_retrain(self, conn: psycopg.AsyncConnection, tmp_path) -> None:
        # Seed labeled PII turns, export an anonymized Parquet, then retrain from it end-to-end.
        from datetime import UTC, datetime, timedelta

        rows = [
            ("e2e_1", "meu CPF 123.456.789-09 péssimo serviço quero cancelar", "negative"),
            ("e2e_2", "horrível atendimento ana@x.com quero cancelar", "negative"),
            ("e2e_3", "ruim demais nao aguento cancelar", "negative"),
            ("e2e_4", "ótimo atendimento muito obrigado", "positive"),
            ("e2e_5", "excelente tudo perfeito adorei", "positive"),
            ("e2e_6", "adorei o suporte obrigado", "positive"),
        ]
        for turn_id, text, label in rows:
            await _seed_turn(conn, turn_id, text, days_ago=1)
            await conn.execute(
                "INSERT INTO labels (label_id, turn_id, conversation_id, label) VALUES (%s, %s, %s, %s)",
                (f"lbl_{turn_id}", turn_id, "conv_lc", label),
            )
        await conn.commit()

        pool = MonitoringPool(DSN, min_size=1, max_size=2)
        await pool.open()
        try:
            exporter = DataLifecycleExporter(
                TimescaleLifecycleRepository(pool), RegexRedactor(), ParquetSampleStore(str(tmp_path))
            )
            result = await exporter.export_window(datetime.now(UTC) - timedelta(days=2), datetime.now(UTC), "m7_e2e")
        finally:
            await pool.close()

        samples = read_samples(result.path)
        assert _CPF.search(" ".join(s.redacted_text for s in samples)) is None  # LGPD gate end-to-end

        def _tiny(version: str) -> SentimentDetector:
            pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 1), min_df=1)), ("clf", LinearSVC(C=1.0))])
            return SentimentDetector(pipe, model_version=version)

        _, bench = RetrainingPipeline(detector_factory=_tiny).retrain_and_benchmark(
            samples,
            ["péssimo quero cancelar", "ótimo obrigado"],
            ["negative", "positive"],
            new_version="sentiment-2026.07.30",
            deployed=None,
        )
        assert bench.new_model_version == "sentiment-2026.07.30"
        assert bench.promoted is True  # first model promoted; benchmark recorded (new_f1 vs deployed None)
