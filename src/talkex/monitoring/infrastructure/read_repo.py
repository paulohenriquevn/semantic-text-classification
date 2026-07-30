"""Index-aligned keyset-paginated read repository (blueprint D1/Corner 4).

Reads recent turns for a conversation ordered by `created_at DESC`, aligned to the
`(conversation_id, created_at DESC)` composite index (chatwoot query-aligned precedent
conversation_finder.rb:108). Keyset pagination (no OFFSET, no N+1) via a `before` cursor.
"""

from __future__ import annotations

from datetime import datetime

from talkex.monitoring.domain.models import RecentTurn
from talkex.monitoring.domain.search import Criterion
from talkex.monitoring.infrastructure.embedder import to_vector_literal
from talkex.monitoring.infrastructure.pool import MonitoringPool
from talkex.retrieval.models import RetrievalHit

# Whitelisted criterion fields → real columns (values are always bound; the column is never from input).
_ALLOWED_COLUMNS: dict[str, str] = {
    "speaker": "speaker",
    "conversation_id": "conversation_id",
}


def _hit_from_row(row: tuple[object, ...], rank: int, *, lexical: bool) -> RetrievalHit:
    """Build a RetrievalHit from (turn_id, conversation_id, raw_text, created_at, score).

    `rank` is the 1-based position in the ORDER BY'd candidate list — RRF fuses on rank, so it
    must reflect the query ordering (not the default 1).
    """
    turn_id, conversation_id, raw_text, created_at, score = row
    score_f = float(score)  # type: ignore[arg-type]
    return RetrievalHit(
        object_id=str(turn_id),
        object_type="turn",
        score=score_f,
        lexical_score=score_f if lexical else None,
        semantic_score=None if lexical else score_f,
        rank=rank,
        metadata={"conversation_id": conversation_id, "raw_text": raw_text, "created_at": created_at},
    )


class TimescaleReadRepository:
    """Implements TurnReadPort + TurnSearchPort over a pooled TimescaleDB connection."""

    def __init__(self, pool: MonitoringPool) -> None:
        self._pool = pool

    async def recent_turns(
        self, conversation_id: str, before: datetime | None = None, limit: int = 50
    ) -> list[RecentTurn]:
        # Index-aligned WHERE + ORDER matching (conversation_id, created_at DESC); keyset via `before`.
        sql = (
            "SELECT turn_id, raw_text, created_at FROM turns "
            "WHERE conversation_id = %s "
            + ("AND created_at < %s " if before is not None else "")
            + "ORDER BY created_at DESC LIMIT %s"
        )
        params: tuple[object, ...] = (
            (conversation_id, before, limit) if before is not None else (conversation_id, limit)
        )
        async with self._pool.connection() as conn:
            cur = await conn.execute(sql, params)
            rows = await cur.fetchall()
        return [RecentTurn(turn_id=r[0], raw_text=r[1], created_at=r[2]) for r in rows]

    @staticmethod
    def _criteria_clause(criteria: tuple[Criterion, ...]) -> tuple[str, list[object]]:
        """Compile criteria to a BOUND SQL fragment + params (blueprint D2, injection-safe).

        The column comes from a fixed whitelist (never from input); the value is ALWAYS a bound
        parameter. A metadata criterion (`metadata.<key>`) binds BOTH the key and the value. An
        unknown field fails fast with a typed error — no silent pass-through (error-handling.md).
        """
        clause = ""
        params: list[object] = []
        for crit in criteria:
            if crit.field in _ALLOWED_COLUMNS:
                clause += f" AND {_ALLOWED_COLUMNS[crit.field]} = %s"
                params.append(crit.value)
            elif crit.field.startswith("metadata."):
                clause += " AND metadata->>%s = %s"
                params.extend([crit.field[len("metadata.") :], crit.value])
            else:
                raise ValueError(
                    f"unknown criterion field {crit.field!r}; allowed: {sorted(_ALLOWED_COLUMNS)} or 'metadata.<key>'"
                )
        return clause, params

    async def lexical_candidates(
        self, query_text: str, top_k: int, window_days: int, criteria: tuple[Criterion, ...] = ()
    ) -> list[RetrievalHit]:
        """BM25-adjacent lexical candidates via ts_rank over the GIN(search_vector), window-scoped."""
        extra_sql, extra_params = self._criteria_clause(criteria)
        sql = (
            "SELECT turn_id, conversation_id, raw_text, created_at, "
            "ts_rank(search_vector, plainto_tsquery('portuguese', %s)) AS score "
            "FROM turns "
            "WHERE search_vector @@ plainto_tsquery('portuguese', %s) "
            "AND created_at > now() - make_interval(days => %s)" + extra_sql + " "
            "ORDER BY score DESC LIMIT %s"
        )
        params: list[object] = [query_text, query_text, window_days, *extra_params, top_k]
        async with self._pool.connection() as conn:
            cur = await conn.execute(sql, params)
            rows = await cur.fetchall()
        return [_hit_from_row(r, i, lexical=True) for i, r in enumerate(rows, start=1)]

    async def semantic_candidates(
        self, query_vector: list[float], top_k: int, window_days: int, criteria: tuple[Criterion, ...] = ()
    ) -> list[RetrievalHit]:
        """ANN candidates via pgvector cosine (<=>) over the HNSW index, window-scoped.

        NULL embeddings are excluded (a NULL is absence, not a distance-0 match — failure scenario).
        """
        extra_sql, extra_params = self._criteria_clause(criteria)
        literal = to_vector_literal(query_vector)
        sql = (
            "SELECT turn_id, conversation_id, raw_text, created_at, "
            "1 - (embedding <=> %s::vector) AS score "
            "FROM turns "
            "WHERE embedding IS NOT NULL "
            "AND created_at > now() - make_interval(days => %s)" + extra_sql + " "
            "ORDER BY embedding <=> %s::vector LIMIT %s"
        )
        params: list[object] = [literal, window_days, *extra_params, literal, top_k]
        async with self._pool.connection() as conn:
            cur = await conn.execute(sql, params)
            rows = await cur.fetchall()
        return [_hit_from_row(r, i, lexical=False) for i, r in enumerate(rows, start=1)]
