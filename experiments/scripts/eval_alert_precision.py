"""M3 alert precision — cancellation-rule precision vs the internal topic labels (blueprint D2).

The critical cancellation rule is evaluated on the CUSTOMER's utterances (intent is the customer's).
Precision = TP / (TP+FP) where a fire is a TP when the conversation's `topic` is a cancellation topic.
Target ≥ 0.80. The naive `contains_any("cancelar")` on full text scores 0.51; the customer-scoped
intent regex reaches ~0.89.
"""

import json
import re
from pathlib import Path

from talkex.monitoring.domain.critical_rules import build_critical_rules
from talkex.rules.config import RuleEngineConfig
from talkex.rules.evaluator import SimpleRuleEvaluator
from talkex.rules.models import RuleEvaluationInput

DATA = Path("experiments/data/test.jsonl")
OUT = Path("experiments/results/M3")


def customer_text(transcript: str) -> str:
    parts = re.split(r"\[(customer|agent)\]", transcript)
    out, i = [], 1
    while i < len(parts):
        if parts[i] == "customer":
            out.append(parts[i + 1] if i + 1 < len(parts) else "")
        i += 2
    return " ".join(out)


def cancellation_precision() -> tuple[float, int, int]:
    """Return (precision, fire_count, true_positives) for the cancellation rule on the test set."""
    rule = next(r for r in build_critical_rules() if r.rule_name == "cancellation")
    ev = SimpleRuleEvaluator()
    fire = tp = 0
    for line in DATA.read_text().splitlines():
        r = json.loads(line)
        cus = customer_text(r["text"])
        res = ev.evaluate(
            [rule],
            RuleEvaluationInput(source_id="w", source_type="context_window", text=cus),
            RuleEngineConfig(),
        )[0]
        if res.matched:
            fire += 1
            if "cancel" in r.get("topic", "").lower():
                tp += 1
    return (tp / fire if fire else 0.0), fire, tp


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prec, fire, tp = cancellation_precision()
    metrics = {"category": "cancellation", "precision": round(prec, 4), "fired": fire, "true_positives": tp, "target": 0.80, "pass": prec >= 0.80}
    (OUT / "alert_precision_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print("DoD (precision >= 0.80):", "PASS" if metrics["pass"] else "FAIL")


if __name__ == "__main__":
    main()
