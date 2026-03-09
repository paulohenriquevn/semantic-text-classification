# ADR-004: ContextWindow Structural Fields for Auditability

## Status

Accepted

## Context

The initial ContextWindow contract included only identity and content fields
(window_id, conversation_id, turn_ids, window_text). During Sprint 2
implementation, it became clear that a context window without positional and
parametric information is not reproducible — downstream stages (embeddings,
classification, rule evaluation) cannot audit or reconstruct the slicing
that produced a given window.

## Decision

Expand ContextWindow with four structural fields:

- **start_index / end_index** (int): 0-based position of first/last turn in the
  parent conversation. Required for reproducing which slice of the conversation
  the window covers.
- **window_size** (int): Redundant with `len(turn_ids)` by design. Exists as an
  auditable field with a `model_validator` enforcing `window_size == len(turn_ids)`,
  enabling fail-fast detection of builder inconsistencies.
- **stride** (int): The step parameter used to generate this window. Required for
  reproducing the same windowing configuration in experiments and retraining.

Two fields from the original concept were deferred:

- **role_aware_views**: Belongs to Text Processing pipeline (Theme 2).
- **embedding_id**: Belongs to Embeddings pipeline (Theme 3).

## Consequences

- ContextWindow is now a self-describing, auditable record of conversational slicing.
- The `window_size == len(turn_ids)` invariant catches builder bugs at construction time.
- No breaking changes — this is an additive expansion of the contract.
- Deferred fields will be added when their pipeline stages are implemented.
