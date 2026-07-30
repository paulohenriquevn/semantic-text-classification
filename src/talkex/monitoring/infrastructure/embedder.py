"""Embedding adapters implementing the `TurnEmbedder` port (M5 Phase 0, blueprint D5).

Two concretes behind one narrow port (DIP):

- `DeterministicEmbedder` — hash-seeded 384-dim vectors, no model download. Fast and reproducible;
  used by tests and as a no-dependency default. Reuses the same SHA-256 → PCG64 technique as
  `talkex.embeddings.NullEmbeddingGenerator` (technique reuse, not batch-API coupling).
- `SentenceTransformerEmbedder` — a real multilingual MiniLM (384-dim), lazily loaded, for production
  ingest/backfill so the ANN half carries genuine semantic signal.

The `vector(384)` column has no pgvector-python adapter registered, so callers format vectors via
`to_vector_literal` and cast `%s::vector` in SQL.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

EMBED_DIM = 384
# Multilingual MiniLM: 384-dim, CPU-friendly, PT-BR capable (matches the vector(384) column).
DEFAULT_ST_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def to_vector_literal(vector: list[float]) -> str:
    """Format a vector as the pgvector text literal `[x,y,...]` for a `%s::vector` cast."""
    return "[" + ",".join(f"{x:.6f}" for x in vector) + "]"


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0.0 else vec


@dataclass
class DeterministicEmbedder:
    """Reproducible, download-free embedder (tests + no-dependency default)."""

    dimensions: int = EMBED_DIM

    def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        seed = int(digest[:8], 16)
        rng = np.random.Generator(np.random.PCG64(seed))
        raw = rng.standard_normal(self.dimensions).astype(np.float32)
        result: list[float] = _l2_normalize(raw).tolist()
        return result


@dataclass
class SentenceTransformerEmbedder:
    """Production embedder wrapping a lazily-loaded sentence-transformers model (384-dim)."""

    model_name: str = DEFAULT_ST_MODEL
    _model: object | None = field(default=None, repr=False)

    def _ensure_model(self) -> object:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy import, deferred

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        model = self._ensure_model()
        vector = model.encode(text, normalize_embeddings=True)  # type: ignore[attr-defined]
        result: list[float] = np.asarray(vector, dtype=np.float32).tolist()
        if len(result) != EMBED_DIM:
            raise ValueError(
                f"{self.model_name} produced dim {len(result)}, expected {EMBED_DIM} "
                "(the turns.embedding column is vector(384))"
            )
        return result
