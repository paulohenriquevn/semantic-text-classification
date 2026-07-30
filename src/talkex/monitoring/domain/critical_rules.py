"""Critical-rule catalogue for real-time alerting (blueprint M3 D1).

A small set of compiled DSL rules whose match warrants a supervisor alert, each producing evidence.
Reuses M0's rule compiler. The negative-sentiment signal is handled separately by the orchestrator's
injected M2 SentimentDetector (blueprint D4) — it is not a lexical DSL rule.
"""

from __future__ import annotations

from talkex.rules.compiler import SimpleRuleCompiler
from talkex.rules.models import RuleDefinition

# One DSL expression per critical category, evaluated on the CUSTOMER's utterances (the intent is the
# customer's, not the agent explaining the process — this scoping is what makes the alert precise:
# a bare "cancelar" keyword over-fires at precision 0.51; this customer-intent regex reaches 0.89).
_CATALOGUE: dict[str, str] = {
    "cancellation": (
        'regex("cancelamento|cancelar (minha|meu|a|o) (conta|plano|assinatura|contrato)'
        '|(quero|desejo|gostaria|preciso|solicito|queria) cancelar")'
    ),
    "escalation": 'contains_any("supervisor", "gerente", "procon", "processo", "reclamação formal", "advogado")',
}


def build_critical_rules() -> list[RuleDefinition]:
    """Compile the critical-rule catalogue into evaluable RuleDefinitions."""
    compiler = SimpleRuleCompiler()
    return [compiler.compile(dsl_text=dsl, rule_id=f"rule_{name}", rule_name=name) for name, dsl in _CATALOGUE.items()]
