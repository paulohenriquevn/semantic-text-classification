"""Rule DSL — Domain-Specific Language for declarative rule definitions.

The DSL provides a human-readable syntax for composing rule predicates
into logical expressions. Each predicate checks one of four signal
families: lexical, semantic, structural, or contextual.

Supports two syntax forms:

1. **Inline expressions** (original):

    keyword("billing") AND speaker("customer")

2. **Block rules** (extended):

    RULE billing_detection
    WHEN
        speaker == "customer"
        AND semantic.intent("billing") > 0.8
        AND lexical.contains_any(["cobrar", "fatura", "boleto"])
    THEN
        tag("billing")
        score(0.9)
        priority("high")

Grammar (EBNF):

    program      = rule_block | expression
    rule_block   = "RULE" IDENTIFIER "WHEN" expression "THEN" action+
    expression   = or_expr
    or_expr      = and_expr ( "OR" and_expr )*
    and_expr     = not_expr ( "AND" not_expr )*
    not_expr     = "NOT" not_expr | atom
    atom         = dotted_expr | infix_comparison | predicate | "(" expression ")"
    dotted_expr  = IDENTIFIER "." method_chain [comparison_op value]
    method_chain = IDENTIFIER "(" args ")" ( "." IDENTIFIER "(" args ")" )*
    infix_comparison = IDENTIFIER comparison_op value
    comparison_op = "==" | ">" | ">=" | "<" | "<=" | "!="
    predicate    = predicate_name "(" arguments ")"
    arguments    = argument ( "," argument )*
    argument     = STRING | NUMBER | IDENTIFIER | list_literal | keyword_arg
    list_literal = "[" (value ("," value)*)? "]"
    keyword_arg  = IDENTIFIER "=" value
    action       = IDENTIFIER "(" arguments ")"

Predicate functions (inline syntax):

    Lexical:
        keyword(field, value)              → contains check (substring)
        keyword(value)                     → contains check on "text"
        word(field, value)                 → word boundary match (whole word)
        word(value)                        → word boundary match on "text"
        stem(field, prefix)               → word-prefix match (stemming)
        stem(prefix)                      → word-prefix match on "text"
        regex(field, pattern)              → regex match (accent-normalized)
        contains_any(word_list)            → any word in list found
        contains_all(word_list)            → all words in list found
        not_contains(field, value)         → text does NOT contain value
        not_contains(value)               → text does NOT contain value on "text"
        excludes_any(word_list)            → text contains NONE of the words
        near(word1, word2, distance)       → two words within N words of each other
        starts_with(field, prefix)         → text starts with prefix
        starts_with(prefix)               → text starts with prefix on "text"
        ends_with(field, suffix)           → text ends with suffix
        ends_with(suffix)                 → text ends with suffix on "text"

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

Dotted namespace syntax (block rules):

    lexical.contains("billing")
    lexical.word("cancelar")
    lexical.stem("cancel")
    lexical.contains_any(["cancelar", "encerrar"])
    lexical.contains_all(["cancelar", "conta"])
    lexical.not_contains("teste")
    lexical.excludes_any(["teste", "debug"])
    lexical.near("cancelar", "conta", 3)
    lexical.starts_with("FAT-")
    lexical.ends_with(".pdf")
    lexical.regex("cancel|terminate")
    semantic.intent("cancelamento") > 0.82
    semantic.similarity("quero cancelar meu serviço") > 0.86
    context.turn_window(5).count(intent="insatisfacao") >= 2
    speaker == "customer"
    channel == "voice"

Example DSL (inline):

    keyword("text", "billing") AND speaker("customer")

    intent("cancel_subscription") AND NOT keyword("text", "upgrade")

Example DSL (block):

    RULE risco_cancelamento
    WHEN
        speaker == "customer"
        AND semantic.intent("cancelamento") > 0.82
        AND lexical.contains_any(["cancelar", "encerrar", "desistir"])
    THEN
        tag("cancelamento_risco") score(0.95) priority("high")
"""

from __future__ import annotations

# Known predicate function names mapped to (PredicateType, field_name, operator, cost_hint).
# field_name may be overridden by the first argument depending on the function.
PREDICATE_REGISTRY: dict[str, tuple[str, str, str, int]] = {
    # Lexical (cost 1)
    "keyword": ("lexical", "text", "contains", 1),
    "word": ("lexical", "text", "word", 1),
    "stem": ("lexical", "text", "stem", 1),
    "regex": ("lexical", "text", "regex", 1),
    "contains_any": ("lexical", "text", "contains_any", 1),
    "contains_all": ("lexical", "text", "contains_all", 1),
    "not_contains": ("lexical", "text", "not_contains", 1),
    "excludes_any": ("lexical", "text", "excludes_any", 1),
    "near": ("lexical", "text", "near", 1),
    "starts_with": ("lexical", "text", "starts_with", 1),
    "ends_with": ("lexical", "text", "ends_with", 1),
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

# Dotted namespace mapping: (namespace, method) → function_name in PREDICATE_REGISTRY.
# Used by the extended parser to resolve dotted predicates like semantic.intent()
# into the same predicate functions used by the inline syntax.
NAMESPACE_PREDICATE_MAP: dict[tuple[str, str], str] = {
    # lexical.* → lexical predicates
    ("lexical", "contains"): "keyword",
    ("lexical", "word"): "word",
    ("lexical", "stem"): "stem",
    ("lexical", "contains_any"): "contains_any",
    ("lexical", "contains_all"): "contains_all",
    ("lexical", "not_contains"): "not_contains",
    ("lexical", "excludes_any"): "excludes_any",
    ("lexical", "near"): "near",
    ("lexical", "starts_with"): "starts_with",
    ("lexical", "ends_with"): "ends_with",
    ("lexical", "regex"): "regex",
    # semantic.* → semantic predicates
    ("semantic", "intent"): "intent",
    ("semantic", "similarity"): "similarity",
}
"""Maps (namespace, method) pairs to predicate function names.

This enables the dotted syntax ``semantic.intent("x")`` to resolve to
the same predicate as ``intent("x")``.
"""

# Infix field mapping: bare field names → (predicate_type, field_name, cost_hint).
# Used when the parser encounters ``speaker == "customer"`` syntax.
INFIX_FIELD_MAP: dict[str, tuple[str, str, int]] = {
    "speaker": ("structural", "speaker_role", 2),
    "channel": ("structural", "channel", 2),
}
"""Maps bare field names to predicate metadata for infix comparison syntax."""

# Comparison operator mapping: DSL operator → internal operator string.
COMPARISON_OP_MAP: dict[str, str] = {
    "==": "eq",
    ">": "gt",
    ">=": "gte",
    "<": "lt",
    "<=": "lte",
    "!=": "neq",
}
"""Maps DSL comparison operators to internal operator strings."""
