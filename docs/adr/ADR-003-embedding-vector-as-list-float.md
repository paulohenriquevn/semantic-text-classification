# ADR-003: Embedding Vectors Stored as `list[float]` in Data Models

**Status:** Accepted
**Date:** 2026-03-09
**Decision makers:** Architect Executor + NLP Engineering

## Context

`EmbeddingRecord` stores dense vectors that are produced by embedding models
(sentence-transformers) and consumed by FAISS indexes, classifiers, and the
rule engine. The natural computation type is `numpy.ndarray`, but the natural
serialization/storage type is `list[float]`.

Options considered:

1. **`numpy.ndarray` in the model**: native for computation, but requires custom
   pydantic serializers, complicates JSON round-trips, and creates numpy coupling
   in the data layer.
2. **`list[float]` in the model**: trivially serializable, pydantic-native, but
   requires conversion at computation boundaries.
3. **Both fields** (`vector: list[float]` + `array: ndarray`): violates single
   source of truth, doubles memory, adds sync complexity.

## Decision

**Option 2: `vector: list[float]` in the pydantic model. Conversion to `numpy.ndarray` happens at usage boundaries.**

```python
# In EmbeddingRecord (data layer)
vector: list[float]

# At computation boundary (embedding/retrieval layer)
import numpy as np
array = np.array(record.vector, dtype=np.float32)
```

## Consequences

### Positive
- **Clean serialization**: `model_dump()` and `model_validate()` work without custom serializers
- **No numpy coupling in data layer**: models package has zero numpy-specific code
- **Storage-friendly**: `list[float]` maps directly to JSON arrays, database columns, and message formats
- **Strict mode compatible**: `list[float]` with `strict=True` works naturally in pydantic v2

### Negative
- **Conversion cost**: `np.array(record.vector)` at every usage point. Mitigated by pre-computing at stage boundaries, not per-operation.
- **No shape enforcement in model**: the model validates `len(vector) == dimensions` but not dtype or memory layout. Acceptable — shape is the critical invariant, dtype is a computation concern.

### Rules
- `EmbeddingRecord.vector` is always `list[float]`
- Conversion to `ndarray` is done once at the boundary of the consuming module
- No numpy imports in `src/talkex/models/`
- The `dimensions` field validates vector length at construction time
