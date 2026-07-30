"""Unit tests for the M5 SearchService — fusion logic with a fake port + embedder (T1.1)."""

from datetime import UTC, datetime

from talkex.monitoring.application.search_service import SearchService
from talkex.monitoring.domain.search import Criterion, SearchQuery
from talkex.retrieval.models import RetrievalHit

_NOW = datetime(2026, 7, 30, tzinfo=UTC)


def _hit(object_id: str, rank: int, *, lexical: bool) -> RetrievalHit:
    return RetrievalHit(
        object_id=object_id,
        object_type="turn",
        score=1.0 / rank,
        lexical_score=1.0 / rank if lexical else None,
        semantic_score=None if lexical else 1.0 / rank,
        rank=rank,
        metadata={"conversation_id": "conv_1", "raw_text": f"text-{object_id}", "created_at": _NOW},
    )


class _FakePort:
    def __init__(self, lexical: list[RetrievalHit], semantic: list[RetrievalHit]) -> None:
        self._lexical = lexical
        self._semantic = semantic
        self.seen_vector: list[float] | None = None
        self.seen_criteria: tuple[Criterion, ...] | None = None

    async def lexical_candidates(self, query_text, top_k, window_days, criteria=()):
        self.seen_criteria = criteria
        return self._lexical

    async def semantic_candidates(self, query_vector, top_k, window_days, criteria=()):
        self.seen_vector = query_vector
        return self._semantic


class _FakeEmbedder:
    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class TestSearchServiceFusion:
    async def test_search_service_fuses_candidates(self) -> None:
        # doc_b tops the lexical list AND appears in the semantic list → RRF should rank it first.
        lexical = [_hit("doc_b", 1, lexical=True), _hit("doc_a", 2, lexical=True)]
        semantic = [_hit("doc_b", 1, lexical=False), _hit("doc_c", 2, lexical=False)]
        service = SearchService(_FakePort(lexical, semantic), _FakeEmbedder())

        hits = await service.search(SearchQuery(query_text="cancelar", top_k=10))

        assert hits[0].turn_id == "doc_b"  # present in both lists → highest RRF
        assert {h.turn_id for h in hits} == {"doc_a", "doc_b", "doc_c"}  # deduped union
        assert hits[0].raw_text == "text-doc_b"  # evidence carried through fusion

    async def test_search_service_embeds_query_and_forwards_criteria(self) -> None:
        port = _FakePort([], [])
        service = SearchService(port, _FakeEmbedder())
        criteria = (Criterion(field="sentiment", value="negative"),)

        await service.search(SearchQuery(query_text="reembolso", top_k=5, criteria=criteria))

        assert port.seen_vector == [0.1, 0.2, 0.3]  # the embedded query reached the ANN half
        assert port.seen_criteria == criteria  # criteria forwarded to the port

    async def test_top_k_truncates_fused_results(self) -> None:
        lexical = [_hit(f"d{i}", i, lexical=True) for i in range(1, 6)]
        service = SearchService(_FakePort(lexical, []), _FakeEmbedder())

        hits = await service.search(SearchQuery(query_text="x", top_k=2))

        assert len(hits) == 2  # fused list truncated to top_k
