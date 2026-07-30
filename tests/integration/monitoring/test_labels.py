"""Integration tests for M5 Phase 2 — label persistence + criterion-filtered search (T2.1)."""

import psycopg
import pytest

from talkex.models.enums import SpeakerRole
from talkex.models.turn import Turn
from talkex.models.types import ConversationId, TurnId
from talkex.monitoring.application.search_service import SearchService
from talkex.monitoring.config import MonitoringConfig
from talkex.monitoring.domain.search import Criterion, Label, SearchQuery
from talkex.monitoring.infrastructure.embedder import DeterministicEmbedder
from talkex.monitoring.infrastructure.label_repo import TimescaleLabelRepository
from talkex.monitoring.infrastructure.pool import MonitoringPool
from talkex.monitoring.infrastructure.read_repo import TimescaleReadRepository
from talkex.monitoring.infrastructure.timescale_repo import TimescaleTurnRepository

pytestmark = pytest.mark.integration

DSN = MonitoringConfig().dsn
EMBEDDER = DeterministicEmbedder()


def _turn(turn_id: str, text: str, speaker: SpeakerRole) -> Turn:
    return Turn(
        turn_id=TurnId(turn_id),
        conversation_id=ConversationId("conv_lbl"),
        speaker=speaker,
        raw_text=text,
        start_offset=0,
        end_offset=len(text),
    )


class TestLabelRoundTrip:
    async def test_label_round_trip(self, conn: psycopg.AsyncConnection) -> None:
        pool = MonitoringPool(DSN, min_size=1, max_size=2)
        await pool.open()
        try:
            repo = TimescaleLabelRepository(pool)
            await repo.save(
                Label(
                    label_id="lbl_1",
                    turn_id="turn_x",
                    conversation_id="conv_lbl",
                    label="compliance_violation",
                    labeled_by="qa_ana",
                )
            )
            got = await repo.get("lbl_1")
        finally:
            await pool.close()

        assert got is not None
        assert got.label == "compliance_violation"
        assert got.labeled_by == "qa_ana"

    async def test_get_missing_label_returns_none(self, conn: psycopg.AsyncConnection) -> None:
        pool = MonitoringPool(DSN, min_size=1, max_size=2)
        await pool.open()
        try:
            got = await TimescaleLabelRepository(pool).get("nope")
        finally:
            await pool.close()
        assert got is None


class TestCriterionFilteredSearch:
    async def test_criterion_narrows_results(self, conn: psycopg.AsyncConnection) -> None:
        # Same lexical text, different speakers — a speaker criterion must narrow to the customer's.
        seed = TimescaleTurnRepository(conn, embedder=EMBEDDER)
        await seed.save(_turn("lbl_cust", "quero cancelar o plano", SpeakerRole.CUSTOMER))
        await seed.save(_turn("lbl_agent", "quero cancelar o plano", SpeakerRole.AGENT))

        pool = MonitoringPool(DSN, min_size=1, max_size=2)
        await pool.open()
        try:
            service = SearchService(TimescaleReadRepository(pool), EMBEDDER)
            unfiltered = await service.search(SearchQuery(query_text="quero cancelar o plano", top_k=10))
            filtered = await service.search(
                SearchQuery(
                    query_text="quero cancelar o plano",
                    top_k=10,
                    criteria=(Criterion(field="speaker", value="customer"),),
                )
            )
        finally:
            await pool.close()

        unfiltered_ids = {h.turn_id for h in unfiltered}
        filtered_ids = {h.turn_id for h in filtered}
        assert {"lbl_cust", "lbl_agent"} <= unfiltered_ids  # both present unfiltered
        assert filtered_ids == {"lbl_cust"}  # criterion narrowed to the customer's utterance
