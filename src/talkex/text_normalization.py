"""Shared text normalization for lexical matching.

Provides accent-stripping and lowercasing utilities used by both the
rule evaluator (lexical predicates) and the BM25 retrieval index.

Uses only ``unicodedata`` from the standard library — no external
dependencies. NFD decomposition separates base characters from
combining diacritical marks, which are then stripped.
"""

from __future__ import annotations

import unicodedata


def strip_accents(text: str) -> str:
    """Remove diacritical marks (accents) from text.

    Uses NFD decomposition to separate base characters from combining
    marks, then strips all combining characters (unicode category 'Mn').

    Examples:
        >>> strip_accents("não")
        'nao'
        >>> strip_accents("café")
        'cafe'
        >>> strip_accents("hello")
        'hello'

    Args:
        text: Text to normalize.

    Returns:
        Text with accents/diacritics removed.
    """
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def normalize_for_matching(text: str) -> str:
    """Normalize text for lexical matching: lowercase + strip accents.

    Composes lowercasing and accent stripping into a single operation.
    Both the search term and the target text should be passed through
    this function before comparison.

    Examples:
        >>> normalize_for_matching("NÃO")
        'nao'
        >>> normalize_for_matching("Cancelamento")
        'cancelamento'

    Args:
        text: Text to normalize.

    Returns:
        Lowercased text with accents removed.
    """
    return strip_accents(text.lower())
