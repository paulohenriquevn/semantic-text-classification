"""Unit tests for text normalization and BM25 tokenization improvements.

Tests cover: accent stripping, normalize_for_matching, and BM25 tokenizer
with punctuation removal + accent normalization.
"""

import pytest

from talkex.text_normalization import normalize_for_matching, strip_accents

# ---------------------------------------------------------------------------
# strip_accents
# ---------------------------------------------------------------------------


class TestStripAccents:
    def test_removes_acute_accent(self) -> None:
        assert strip_accents("café") == "cafe"

    def test_removes_tilde(self) -> None:
        assert strip_accents("não") == "nao"

    def test_removes_cedilla(self) -> None:
        assert strip_accents("ação") == "acao"

    def test_preserves_plain_text(self) -> None:
        assert strip_accents("hello") == "hello"

    def test_handles_empty_string(self) -> None:
        assert strip_accents("") == ""

    def test_handles_multiple_accents(self) -> None:
        assert strip_accents("informação técnica") == "informacao tecnica"

    def test_preserves_case(self) -> None:
        assert strip_accents("Ação") == "Acao"


# ---------------------------------------------------------------------------
# normalize_for_matching
# ---------------------------------------------------------------------------


class TestNormalizeForMatching:
    def test_lowercases_and_strips_accents(self) -> None:
        assert normalize_for_matching("NÃO") == "nao"

    def test_lowercases_plain_text(self) -> None:
        assert normalize_for_matching("Cancelamento") == "cancelamento"

    def test_handles_mixed_case_accented(self) -> None:
        assert normalize_for_matching("Devolução") == "devolucao"


# ---------------------------------------------------------------------------
# BM25 tokenizer
# ---------------------------------------------------------------------------


class TestBM25Tokenizer:
    def test_strips_punctuation(self) -> None:
        from talkex.retrieval.bm25 import _tokenize

        tokens = _tokenize("cancelar, encerrar! por favor.")
        assert tokens == ["cancelar", "encerrar", "por", "favor"]

    def test_normalizes_accents(self) -> None:
        from talkex.retrieval.bm25 import _tokenize

        tokens = _tokenize("não quero cancelação")
        assert tokens == ["nao", "quero", "cancelacao"]

    def test_lowercases(self) -> None:
        from talkex.retrieval.bm25 import _tokenize

        tokens = _tokenize("CANCELAR Minha Conta")
        assert tokens == ["cancelar", "minha", "conta"]

    def test_empty_string(self) -> None:
        from talkex.retrieval.bm25 import _tokenize

        assert _tokenize("") == []

    def test_punctuation_only(self) -> None:
        from talkex.retrieval.bm25 import _tokenize

        assert _tokenize("!!! ... ???") == []

    @pytest.mark.parametrize(
        ("raw", "expected_token"),
        [
            ("cancelar,", "cancelar"),
            ("cancelar.", "cancelar"),
            ("cancelar!", "cancelar"),
            ("cancelar?", "cancelar"),
            ('"cancelar"', "cancelar"),
        ],
    )
    def test_trailing_punctuation_stripped(self, raw: str, expected_token: str) -> None:
        from talkex.retrieval.bm25 import _tokenize

        tokens = _tokenize(raw)
        assert tokens == [expected_token]


class TestBM25AccentSearch:
    """Integration: BM25 search matches documents with/without accents."""

    def test_search_nao_matches_não(self) -> None:
        from talkex.retrieval.bm25 import InMemoryBM25Index

        index = InMemoryBM25Index()
        index.index(
            [
                {"doc_id": "d1", "text": "Não quero esse serviço"},
                {"doc_id": "d2", "text": "Obrigado pela ajuda"},
            ]
        )
        hits = index.search("nao quero")
        assert len(hits) >= 1
        assert hits[0].object_id == "d1"

    def test_search_accented_query_matches_plain(self) -> None:
        from talkex.retrieval.bm25 import InMemoryBM25Index

        index = InMemoryBM25Index()
        index.index(
            [
                {"doc_id": "d1", "text": "nao quero esse servico"},
            ]
        )
        hits = index.search("não quero")
        assert len(hits) >= 1
        assert hits[0].object_id == "d1"
