"""Unit tests for the M7 PT-BR PII redactor (T0.1) — LGPD negative-case: PII must be ABSENT."""

import re

from talkex.monitoring.infrastructure.pii_redactor import RegexRedactor

_redactor = RegexRedactor()

_CPF_DIGITS = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
_PHONE_DIGITS = re.compile(r"\(?\b\d{2}\)?[\s-]?9?\d{4}[\s-]?\d{4}\b")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


class TestRedactor:
    def test_redacts_cpf_phone_email(self) -> None:
        text = (
            "Meu nome é João Silva, meu CPF é 123.456.789-09, telefone (11) 98765-4321 e email joao.silva@example.com."
        )
        out = _redactor.redact(text)

        # LGPD negative-case: the raw PII must NOT survive (typed guarantee, not "it ran").
        assert _CPF_DIGITS.search(out) is None, f"CPF leaked: {out}"
        assert _PHONE_DIGITS.search(out) is None, f"phone leaked: {out}"
        assert _EMAIL.search(out) is None, f"email leaked: {out}"
        assert "João Silva" not in out, f"name leaked: {out}"
        assert "[CPF]" in out and "[TELEFONE]" in out and "[EMAIL]" in out and "[NOME]" in out

    def test_redacts_cnpj(self) -> None:
        out = _redactor.redact("A empresa tem CNPJ 12.345.678/0001-95 registrado.")
        assert "12.345.678/0001-95" not in out
        assert "[CNPJ]" in out

    def test_preserves_non_pii(self) -> None:
        text = "Quero cancelar o plano porque o serviço está ruim e caro."
        out = _redactor.redact(text)
        assert out == text  # ordinary text is untouched — the label signal survives

    def test_does_not_over_redact_sentence_start(self) -> None:
        # A capitalized word at the start of a sentence is NOT a name cue → not redacted.
        out = _redactor.redact("Cancelar agora. Preciso resolver isso hoje.")
        assert "[NOME]" not in out
