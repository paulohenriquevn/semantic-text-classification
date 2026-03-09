"""Semantic rule engine based on Abstract Syntax Trees (AST).

Compiles a human-readable DSL into AST for safe, auditable execution.
Combines four signal families: lexical, semantic, structural, and contextual.
Supports short-circuit execution ordered by predicate cost and produces
traceable evidence for every rule evaluation.
"""
