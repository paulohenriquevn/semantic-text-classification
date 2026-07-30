"""PT-BR PII redactor implementing the `Redactor` port (M7 D1, LGPD-critical).

A focused regex redactor for Brazilian PII — CPF, CNPJ, phone (DDD), email, and a conservative name
heuristic. This is a deliberate build-not-adopt call (blueprint D1): the reference project's Presidio
config targets English/US entities (SSN, spaCy `en`), and Presidio's PT support is immature. Patterns
are applied longest-first so a CNPJ is not partially eaten by the CPF pattern. Over-redaction is
preferred to a leak — a false `[NOME]` is harmless; a leaked CPF is an LGPD failure.
"""

from __future__ import annotations

import re

# Order matters: e-mail first (its digits must not be seen as a phone), then the longer numeric IDs.
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_CNPJ = re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b")
_CPF = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
# DDD phone: optional (DD), optional 9, 4+4 digits, common separators.
_PHONE = re.compile(r"\(?\b\d{2}\)?[\s-]?9?\d{4}[\s-]?\d{4}\b")
# Conservative name heuristic: a name only after an explicit self-identification / title cue, so ordinary
# capitalized words (start of sentence, place names) are NOT redacted.
_NAME = re.compile(
    r"(?i)\b(meu nome é|me chamo|aqui (?:é|e|quem fala é)|sou (?:o|a)|Sr\.?|Sra\.?|Dr\.?|Dra\.?)\s+"
    r"([A-ZÁÉÍÓÚÂÊÔÃÕÀÜÇ][\wÁÉÍÓÚÂÊÔÃÕÀÜÇáéíóúâêôãõàüç]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÀÜÇ][\wÁÉÍÓÚÂÊÔÃÕÀÜÇáéíóúâêôãõàüç]+){0,2})"
)


class RegexRedactor:
    """Redacts PT-BR PII from free text; preserves everything else (join keys are handled by the caller)."""

    def redact(self, text: str) -> str:
        text = _EMAIL.sub("[EMAIL]", text)
        text = _CNPJ.sub("[CNPJ]", text)
        text = _CPF.sub("[CPF]", text)
        text = _PHONE.sub("[TELEFONE]", text)
        text = _NAME.sub(lambda m: f"{m.group(1)} [NOME]", text)
        return text
