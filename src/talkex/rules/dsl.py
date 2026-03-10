"""Rule DSL — Domain-Specific Language for declarative rule definitions.

The DSL provides a human-readable syntax for composing rule predicates
into logical expressions. Each predicate checks one of four signal
families: lexical, semantic, structural, or contextual.

Grammar (EBNF):

    rule         = expression
    expression   = or_expr
    or_expr      = and_expr ( "OR" and_expr )*
    and_expr     = not_expr ( "AND" not_expr )*
    not_expr     = "NOT" not_expr | atom
    atom         = predicate | "(" expression ")"
    predicate    = predicate_name "(" arguments ")"
    arguments    = argument ( "," argument )*
    argument     = STRING | NUMBER | IDENTIFIER

Predicate functions:

    Lexical:
        keyword(field, value)              → contains check
        regex(field, pattern)              → regex match

    Semantic:
        intent(label)                      → intent score ≥ threshold
        similarity(field, value, threshold) → embedding similarity

    Structural:
        speaker(role)                      → speaker role equality
        channel(value)                     → channel equality
        field_eq(field, value)             → generic field equality
        field_gte(field, value)            → numeric comparison ≥
        field_lte(field, value)            → numeric comparison ≤

    Contextual:
        repeated(field, value, count)      → repeated mention in window
        occurs_after(field, first, second) → sequential occurrence

Example DSL:

    keyword("text", "billing") AND speaker("customer")

    intent("cancel_subscription") AND NOT keyword("text", "upgrade")

    (keyword("text", "billing") OR keyword("text", "invoice")) AND speaker("customer")
"""

from __future__ import annotations

# Known predicate function names mapped to (PredicateType, field_name, operator, cost_hint).
# field_name may be overridden by the first argument depending on the function.
PREDICATE_REGISTRY: dict[str, tuple[str, str, str, int]] = {
    # Lexical (cost 1)
    "keyword": ("lexical", "text", "contains", 1),
    "regex": ("lexical", "text", "regex", 1),
    # Semantic (cost 4)
    "intent": ("semantic", "intent_score", "gte", 4),
    "similarity": ("semantic", "embedding_similarity", "similarity_above", 4),
    # Structural (cost 2)
    "speaker": ("structural", "speaker_role", "eq", 2),
    "channel": ("structural", "channel", "eq", 2),
    "field_eq": ("structural", "field", "eq", 2),
    "field_gte": ("structural", "field", "gte", 2),
    "field_lte": ("structural", "field", "lte", 2),
    # Contextual (cost 3)
    "repeated": ("contextual", "field", "repeated_in_window", 3),
    "occurs_after": ("contextual", "field", "occurs_after", 3),
}
"""Registry of known predicate function names and their default mappings.

Each entry maps a function name to a tuple of:
    (predicate_type, default_field_name, operator, cost_hint)

The parser uses this registry to resolve DSL function calls into
PredicateNode attributes. Unknown function names are rejected
during semantic validation.
"""
