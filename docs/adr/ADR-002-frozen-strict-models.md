# ADR-002: Frozen and Strict Pydantic Models for Core Domain Types

**Status:** Accepted
**Date:** 2026-03-09
**Decision makers:** Architect Executor + NLP Engineering

## Context

The 6 core domain models (Conversation, Turn, ContextWindow, EmbeddingRecord,
Prediction, RuleExecution) are the contracts between all pipeline stages. They
flow through ingestion, embedding generation, classification, rule evaluation,
and analytics. Any mutation after creation would introduce subtle bugs in a
concurrent, multi-stage pipeline.

Pydantic v2 offers two relevant configuration flags:
- `frozen=True`: makes instances immutable (raises error on attribute assignment)
- `strict=True`: disables type coercion (e.g., `"123"` won't silently become `123`)

## Decision

**All 6 core models use `model_config = ConfigDict(frozen=True, strict=True)`.**

## Consequences

### Positive
- **Immutability**: pipeline stages cannot accidentally mutate shared state. If a stage needs a modified version, it creates a new instance. This aligns with functional data flow.
- **Strict typing**: no silent coercion means type errors surface at construction time, not downstream. A `score: float` field receiving `"0.95"` will fail immediately.
- **Debuggability**: when a model instance exists, its values are exactly what was passed at construction. No hidden transformations.
- **Test reliability**: `strict=True` forces tests to use correct types, catching integration issues early.

### Negative
- **Convenience cost**: code that passes strings where ints are expected will break. This is intentional — it surfaces bugs, but requires discipline in test factories and API boundaries.
- **Serialization round-trips**: `model_validate(data)` from JSON/dict must pass exact types. Use `model_validate(data, strict=False)` at deserialization boundaries when needed.

### Rules
- All 6 core models: `frozen=True, strict=True`
- Deserialization boundaries (API handlers, file parsers) may use `strict=False` explicitly
- Test factories must produce correctly-typed instances — no reliance on coercion
