"""Expand the Brazilian customer service dataset with controlled synthetic generation.

Generates new conversations using an LLM (Claude) to expand from ~944 to ~3,500
conversations with controlled variability:
  - Variable turn counts (4-20, log-normal distribution)
  - Realistic class imbalance (not uniform)
  - 5 customer personas (formal, informal, irritado, idoso, jovem)
  - 8 sectors, 3 sentiments
  - Compatible with existing TalkEx JSONL format

Supports resume via checkpoint file — can be interrupted and restarted safely.

Usage:
    python experiments/scripts/expand_dataset.py --output experiments/data/expanded.jsonl
    python experiments/scripts/expand_dataset.py --output experiments/data/expanded.jsonl --resume
    python experiments/scripts/expand_dataset.py --dry-run  # preview distribution without generating
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import click

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — Intent distribution, personas, sectors
# ---------------------------------------------------------------------------

INTENTS: list[str] = [
    "reclamacao",
    "duvida_produto",
    "duvida_servico",
    "suporte_tecnico",
    "compra",
    "cancelamento",
    "saudacao",
    "elogio",
    "outros",
]

# Target distribution inspired by real call center data
INTENT_DISTRIBUTION: dict[str, float] = {
    "reclamacao": 0.20,
    "duvida_produto": 0.18,
    "duvida_servico": 0.17,
    "suporte_tecnico": 0.15,
    "compra": 0.10,
    "cancelamento": 0.08,
    "saudacao": 0.05,
    "elogio": 0.04,
    "outros": 0.03,
}

SECTORS: list[str] = [
    "telecom",
    "tecnologia",
    "financeiro",
    "ecommerce",
    "saude",
    "educacao",
    "restaurante",
    "imobiliario",
]

SENTIMENTS: list[str] = ["negative", "neutral", "positive"]

# Realistic sentiment distribution (call centers skew negative)
SENTIMENT_DISTRIBUTION: dict[str, float] = {
    "negative": 0.40,
    "neutral": 0.30,
    "positive": 0.30,
}

# Intent-constrained sentiment: avoids incoherent combos (e.g. elogio+negative)
# Each intent maps to its allowed sentiments with weights.
INTENT_SENTIMENT_CONSTRAINTS: dict[str, dict[str, float]] = {
    "reclamacao": {"negative": 0.65, "neutral": 0.35},  # never positive
    "elogio": {"positive": 0.75, "neutral": 0.25},  # never negative
    "saudacao": {"positive": 0.40, "neutral": 0.60},  # never negative
    "cancelamento": {"negative": 0.50, "neutral": 0.30, "positive": 0.20},  # all ok
    "duvida_produto": {"negative": 0.25, "neutral": 0.45, "positive": 0.30},
    "duvida_servico": {"negative": 0.25, "neutral": 0.45, "positive": 0.30},
    "suporte_tecnico": {"negative": 0.40, "neutral": 0.40, "positive": 0.20},
    "compra": {"negative": 0.10, "neutral": 0.30, "positive": 0.60},
    "outros": {"negative": 0.30, "neutral": 0.40, "positive": 0.30},  # default
}

PERSONAS: list[str] = ["formal", "informal", "irritado", "idoso", "jovem"]

PERSONA_DESCRIPTIONS: dict[str, str] = {
    "formal": (
        "Cliente formal e educado. Usa linguagem correta, sem gírias. "
        "Trata o atendente por 'você' ou 'senhor/senhora'. "
        "Escreve com acentos e pontuação adequada."
    ),
    "informal": (
        "Cliente informal e descontraído. Usa abreviações como 'vc', 'td', 'tb', 'pq', 'blz'. "
        "Usa gírias como 'show', 'beleza', 'massa', 'firmeza'. "
        "Pode omitir acentos: 'nao' em vez de 'não', 'ta' em vez de 'tá'."
    ),
    "irritado": (
        "Cliente irritado e impaciente. Usa tom agressivo, pode usar letras maiúsculas "
        "para enfatizar. Faz cobranças diretas, ameaça cancelar ou procurar órgãos de defesa. "
        "Linguagem mais curta e direta. Pode usar palavras como 'absurdo', 'inadmissível', "
        "'inaceitável', 'vou cancelar', 'vou no Procon'."
    ),
    "idoso": (
        "Cliente idoso, menos familiarizado com tecnologia. Usa linguagem mais formal "
        "e detalhada. Pode ser repetitivo ao explicar o problema. Usa expressões como "
        "'meu filho/minha filha', 'por gentileza', 'se não for incômodo'. "
        "Pode ter dificuldade com termos técnicos."
    ),
    "jovem": (
        "Cliente jovem, fluente em internet. Usa muitas gírias e abreviações: "
        "'kk', 'mano', 'véi', 'tlgd', 'tmj', 'pfv', 'vlw', 'flw'. "
        "Tom descontraído, pode usar emojis textuais. "
        "Familiar com apps e tecnologia."
    ),
}

INTENT_DESCRIPTIONS: dict[str, str] = {
    "reclamacao": (
        "O cliente faz uma reclamação sobre um produto ou serviço. "
        "Houve um problema real que o incomodou (entrega atrasada, produto defeituoso, "
        "cobrança errada, mau atendimento anterior). O agente busca resolver."
    ),
    "duvida_produto": (
        "O cliente tem dúvida sobre um produto específico: preço, disponibilidade, "
        "características, especificações, comparação entre modelos. "
        "O agente fornece informações."
    ),
    "duvida_servico": (
        "O cliente tem dúvida sobre como um serviço funciona: como contratar, "
        "como usar, quais são as regras, prazos, condições. "
        "O agente explica o funcionamento."
    ),
    "suporte_tecnico": (
        "O cliente precisa de ajuda técnica: erro no app, sistema não funciona, "
        "equipamento com defeito, senha que não funciona, conexão instável. "
        "O agente faz troubleshooting."
    ),
    "compra": (
        "O cliente quer comprar/contratar algo: pedir um produto, "
        "contratar um plano, fazer uma reserva, agendar um serviço. "
        "O agente conduz a venda."
    ),
    "cancelamento": (
        "O cliente quer cancelar um serviço, plano, assinatura ou pedido. "
        "Pode estar insatisfeito ou simplesmente não precisa mais. "
        "O agente pode tentar reter ou processar o cancelamento."
    ),
    "saudacao": (
        "Interação predominantemente social: o cliente cumprimenta, "
        "pergunta como funciona o atendimento, ou faz uma consulta genérica. "
        "Conversa leve, sem problema específico a resolver."
    ),
    "elogio": (
        "O cliente faz um elogio ao atendimento, produto ou serviço. "
        "Expressa satisfação, agradecimento, ou recomendação positiva. "
        "O agente agradece e reforça o relacionamento."
    ),
    "outros": (
        "Assuntos que não se encaixam nas demais categorias: "
        "engano de número, consulta sobre horário de funcionamento, "
        "informação sobre localização, assunto pessoal com o atendente. "
        "Conversa breve e tangencial."
    ),
}

SECTOR_CONTEXTS: dict[str, str] = {
    "telecom": "Empresa de telecomunicações (internet, telefone, TV a cabo, planos de celular).",
    "tecnologia": "Empresa de tecnologia (software, SaaS, apps, suporte de TI, sistemas).",
    "financeiro": "Instituição financeira (banco, cartão de crédito, empréstimo, investimentos).",
    "ecommerce": "Loja online (marketplace, entregas, devoluções, produtos variados).",
    "saude": "Empresa de saúde (plano de saúde, clínica, farmácia, agendamento de consultas).",
    "educacao": "Instituição educacional (escola, curso online, universidade, certificação).",
    "restaurante": "Restaurante ou delivery (pedidos, cardápio, entrega, reservas).",
    "imobiliario": "Empresa do setor imobiliário (aluguel, venda, condomínio, manutenção).",
}


# ---------------------------------------------------------------------------
# Generation plan
# ---------------------------------------------------------------------------


@dataclass
class ConversationSpec:
    """Specification for a single conversation to generate."""

    intent: str
    sector: str
    sentiment: str
    persona: str
    num_turns: int
    conversation_id: str


@dataclass
class GenerationPlan:
    """Full plan of conversations to generate."""

    specs: list[ConversationSpec] = field(default_factory=list)
    seed: int = 42

    @property
    def total(self) -> int:
        return len(self.specs)

    def intent_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self.specs:
            counts[s.intent] = counts.get(s.intent, 0) + 1
        return counts


def _sample_turn_count(rng: random.Random) -> int:
    """Sample turn count from log-normal distribution, mean ~8, range 4-20.

    Uses log-normal to get a right-skewed distribution where most
    conversations have 6-10 turns but some extend to 16-20.
    """
    # Log-normal parameters calibrated for mean ~8, stdev ~4
    mu = math.log(7.5)  # ~2.015
    sigma = 0.4
    raw = rng.lognormvariate(mu, sigma)
    # Clamp to [4, 20] and round to even number (customer-agent pairs)
    turns = int(round(raw))
    turns = max(4, min(20, turns))
    # Ensure even number (each exchange = 1 customer + 1 agent)
    if turns % 2 != 0:
        turns += 1
    return min(turns, 20)


def build_generation_plan(
    target_total: int,
    existing_count: int,
    start_id: int,
    seed: int = 42,
) -> GenerationPlan:
    """Build a plan of conversations to generate.

    Args:
        target_total: Target total conversations (existing + new).
        existing_count: Number of conversations already in the dataset.
        start_id: Starting conversation ID number.
        seed: Random seed for reproducibility.

    Returns:
        GenerationPlan with specs for each conversation to generate.
    """
    rng = random.Random(seed)
    to_generate = target_total - existing_count

    if to_generate <= 0:
        logger.info("Nothing to generate: existing %d >= target %d", existing_count, target_total)
        return GenerationPlan(seed=seed)

    # Calculate per-intent counts
    intent_counts: dict[str, int] = {}
    remaining = to_generate
    for i, (intent, pct) in enumerate(INTENT_DISTRIBUTION.items()):
        if i == len(INTENT_DISTRIBUTION) - 1:
            intent_counts[intent] = remaining  # last intent gets remainder
        else:
            count = round(to_generate * pct)
            intent_counts[intent] = count
            remaining -= count

    # Build specs
    specs: list[ConversationSpec] = []
    conv_id = start_id

    for intent, count in intent_counts.items():
        for _ in range(count):
            # Intent-constrained sentiment (avoids incoherent combos)
            intent_sentiments = INTENT_SENTIMENT_CONSTRAINTS.get(intent, SENTIMENT_DISTRIBUTION)
            allowed_sentiments = list(intent_sentiments.keys())
            sentiment_weights = list(intent_sentiments.values())
            sentiment = rng.choices(allowed_sentiments, weights=sentiment_weights)[0]

            # Uniform random for sector and persona
            sector = rng.choice(SECTORS)
            persona = rng.choice(PERSONAS)
            num_turns = _sample_turn_count(rng)

            specs.append(
                ConversationSpec(
                    intent=intent,
                    sector=sector,
                    sentiment=sentiment,
                    persona=persona,
                    num_turns=num_turns,
                    conversation_id=f"conv_synth_{conv_id:05d}",
                )
            )
            conv_id += 1

    # Shuffle to avoid generating all of one intent in sequence
    rng.shuffle(specs)

    return GenerationPlan(specs=specs, seed=seed)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_prompt(
    spec: ConversationSpec,
    few_shot_examples: list[dict],
) -> str:
    """Build the generation prompt for a single conversation.

    Args:
        spec: Conversation specification (intent, sector, persona, etc.).
        few_shot_examples: 2-3 example conversations from the original dataset.

    Returns:
        Prompt string for the LLM.
    """
    examples_text = ""
    for i, ex in enumerate(few_shot_examples, 1):
        examples_text += f"\n--- Exemplo {i} (intent: {ex['topic']}, setor: {ex['domain']}) ---\n"
        examples_text += ex["text"] + "\n"

    return f"""Gere uma conversa de atendimento ao cliente em português brasileiro.

PARÂMETROS:
- Intent: {spec.intent} — {INTENT_DESCRIPTIONS[spec.intent]}
- Setor: {spec.sector} — {SECTOR_CONTEXTS[spec.sector]}
- Sentimento do cliente: {spec.sentiment}
- Persona do cliente: {spec.persona} — {PERSONA_DESCRIPTIONS[spec.persona]}
- Número de turnos: exatamente {spec.num_turns} turnos (alternando customer e agent)

REGRAS:
1. A conversa DEVE ter exatamente {spec.num_turns} turnos, alternando entre customer e agent.
2. O primeiro turno é SEMPRE do customer.
3. Use linguagem natural brasileira, consistente com a persona descrita.
4. O agent deve ser profissional mas adaptado ao tom do cliente.
5. A conversa deve ser coerente e realista para o setor e intent especificados.
6. Inclua detalhes específicos do setor (nomes de produtos, valores, procedimentos).
7. NÃO use placeholder genéricos — invente nomes, valores e detalhes concretos.

EXEMPLOS DE REFERÊNCIA (do dataset original):
{examples_text}

FORMATO DE SAÍDA — JSON puro, sem markdown:
{{
  "messages": [
    {{"role": "customer", "content": "texto do cliente"}},
    {{"role": "agent", "content": "texto do agente"}},
    ...
  ]
}}

Gere a conversa agora. Retorne APENAS o JSON, sem explicações."""


def build_system_prompt() -> str:
    """System prompt for the conversation generator."""
    return (
        "Você é um gerador de dados de treinamento para NLP. "
        "Sua tarefa é criar conversas de atendimento ao cliente realistas em português brasileiro. "
        "As conversas devem parecer autênticas, com linguagem informal quando apropriado, "
        "erros de digitação ocasionais, gírias brasileiras, e fluxo natural de diálogo. "
        "Retorne APENAS JSON válido, sem markdown, sem explicações."
    )


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------


def generate_conversation(
    spec: ConversationSpec,
    few_shot_examples: list[dict],
    client: object,
    model: str,
) -> dict | None:
    """Generate a single conversation using the Anthropic API.

    Args:
        spec: Conversation specification.
        few_shot_examples: Example conversations for few-shot prompting.
        client: Anthropic client instance.
        model: Model name (e.g. "claude-sonnet-4-20250514").

    Returns:
        Conversation record in TalkEx JSONL format, or None if generation fails.
    """
    prompt = build_prompt(spec, few_shot_examples)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=build_system_prompt(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,  # High temperature for diversity
        )

        raw_text = response.content[0].text.strip()

        # Try to parse JSON — handle common issues
        # Remove markdown code fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1]  # remove first line
            if raw_text.endswith("```"):
                raw_text = raw_text[: -len("```")]
            raw_text = raw_text.strip()

        parsed = json.loads(raw_text)
        messages = parsed.get("messages", [])

        if not messages:
            logger.warning("Empty messages for %s", spec.conversation_id)
            return None

        # Build text in TalkEx format
        turn_texts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "").strip()
            if content:
                turn_texts.append(f"[{role}] {content}")

        text = "\n".join(turn_texts)

        # Track few-shot example IDs for leakage audit
        few_shot_ids = [ex.get("conversation_id", "unknown") for ex in few_shot_examples]

        return {
            "conversation_id": spec.conversation_id,
            "text": text,
            "domain": spec.sector,
            "topic": spec.intent,
            "asr_confidence": 0.95,
            "audio_duration_seconds": len(messages) * 15,
            "word_count": len(text.split()),
            "source_file": "synthetic_expansion",
            "sentiment": spec.sentiment,
            "metadata": {
                "persona": spec.persona,
                "target_turns": spec.num_turns,
                "actual_turns": len(messages),
                "generator": "claude",
                "generator_model": model,
                "seed": 42,
                "few_shot_ids": few_shot_ids,
            },
        }

    except json.JSONDecodeError as e:
        logger.warning("JSON parse error for %s: %s", spec.conversation_id, e)
        return None
    except Exception as e:
        logger.warning("Generation error for %s: %s", spec.conversation_id, e)
        return None


# ---------------------------------------------------------------------------
# Few-shot example management
# ---------------------------------------------------------------------------


def load_original_dataset(path: str) -> dict[str, list[dict]]:
    """Load original conversations grouped by intent.

    Args:
        path: Path to the original conversations.jsonl.

    Returns:
        Dict mapping intent -> list of conversation records.
    """
    by_intent: dict[str, list[dict]] = {intent: [] for intent in INTENTS}

    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            intent = record.get("topic", "outros")
            if intent in by_intent:
                by_intent[intent].append(record)

    return by_intent


def select_few_shot(
    by_intent: dict[str, list[dict]],
    target_intent: str,
    rng: random.Random,
    n: int = 2,
) -> list[dict]:
    """Select few-shot examples for a given intent.

    Picks n examples from the target intent. If fewer than n available,
    uses what's there.
    """
    candidates = by_intent.get(target_intent, [])
    if not candidates:
        # Fallback: pick from any intent
        all_convs = [c for convs in by_intent.values() for c in convs]
        candidates = all_convs

    return rng.sample(candidates, min(n, len(candidates)))


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------


def load_checkpoint(path: Path) -> set[str]:
    """Load set of already-generated conversation IDs."""
    if not path.exists():
        return set()

    generated: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                generated.add(record["conversation_id"])
            except (json.JSONDecodeError, KeyError):
                continue

    return generated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--output",
    default="experiments/data/expanded.jsonl",
    help="Output JSONL path for generated conversations.",
)
@click.option(
    "--original",
    default="demo/data/conversations.jsonl",
    help="Path to original dataset (for few-shot examples).",
)
@click.option(
    "--target-total",
    default=3500,
    type=int,
    help="Target total conversations (existing + new).",
)
@click.option(
    "--model",
    default="claude-sonnet-4-20250514",
    help="Anthropic model to use for generation.",
)
@click.option(
    "--seed",
    default=42,
    type=int,
    help="Random seed for reproducibility.",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    help="Resume from checkpoint (default: True).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview generation plan without calling the API.",
)
@click.option(
    "--batch-delay",
    default=0.5,
    type=float,
    help="Delay between API calls in seconds (rate limiting).",
)
def main(
    output: str,
    original: str,
    target_total: int,
    model: str,
    seed: int,
    resume: bool,
    dry_run: bool,
    batch_delay: float,
) -> None:
    """Expand the Brazilian customer service dataset with synthetic generation."""
    output_path = Path(output)
    original_path = Path(original)

    if not original_path.exists():
        raise click.ClickException(f"Original dataset not found: {original_path}")

    # Load original dataset for few-shot examples
    logger.info("Loading original dataset from %s...", original_path)
    by_intent = load_original_dataset(str(original_path))
    existing_count = sum(len(v) for v in by_intent.values())
    logger.info("  Loaded %d conversations across %d intents", existing_count, len(by_intent))

    # Determine starting ID
    start_id = existing_count + 1

    # Build generation plan
    plan = build_generation_plan(
        target_total=target_total,
        existing_count=existing_count,
        start_id=start_id,
        seed=seed,
    )

    logger.info("Generation plan:")
    logger.info("  Target total: %d", target_total)
    logger.info("  Already existing: %d", existing_count)
    logger.info("  To generate: %d", plan.total)
    logger.info("  Intent distribution:")
    for intent, count in sorted(plan.intent_counts().items(), key=lambda x: -x[1]):
        pct = count / plan.total * 100 if plan.total > 0 else 0
        logger.info("    %s: %d (%.1f%%)", intent, count, pct)

    # Dry run: show plan and exit
    if dry_run:
        logger.info("\n--- DRY RUN: Turn count distribution ---")
        turn_counts: dict[int, int] = {}
        for s in plan.specs:
            turn_counts[s.num_turns] = turn_counts.get(s.num_turns, 0) + 1
        for turns in sorted(turn_counts):
            bar = "█" * (turn_counts[turns] // 5)
            logger.info("  %2d turns: %4d %s", turns, turn_counts[turns], bar)

        logger.info("\n--- DRY RUN: Persona distribution ---")
        persona_counts: dict[str, int] = {}
        for s in plan.specs:
            persona_counts[s.persona] = persona_counts.get(s.persona, 0) + 1
        for persona, count in sorted(persona_counts.items(), key=lambda x: -x[1]):
            logger.info("  %s: %d (%.1f%%)", persona, count, count / plan.total * 100)

        logger.info("\n--- DRY RUN: Sector distribution ---")
        sector_counts: dict[str, int] = {}
        for s in plan.specs:
            sector_counts[s.sector] = sector_counts.get(s.sector, 0) + 1
        for sector, count in sorted(sector_counts.items(), key=lambda x: -x[1]):
            logger.info("  %s: %d (%.1f%%)", sector, count, count / plan.total * 100)

        logger.info("\nDry run complete. Use --no-dry-run to generate.")
        return

    # Check for checkpoint / resume
    already_generated: set[str] = set()
    if resume:
        already_generated = load_checkpoint(output_path)
        if already_generated:
            logger.info("Resuming: %d conversations already generated", len(already_generated))

    # Filter plan to skip already-generated
    remaining_specs = [s for s in plan.specs if s.conversation_id not in already_generated]
    logger.info("Remaining to generate: %d", len(remaining_specs))

    if not remaining_specs:
        logger.info("Nothing to generate. All conversations already exist.")
        return

    # Initialize Anthropic client
    try:
        import anthropic

        client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
    except ImportError:
        raise click.ClickException(
            "anthropic package not installed. Run: pip install anthropic"
        )
    except Exception as e:
        raise click.ClickException(f"Failed to initialize Anthropic client: {e}")

    # Generate conversations
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    generated_count = len(already_generated)
    failed_count = 0
    start_time = time.time()

    with open(output_path, "a", encoding="utf-8") as f:
        for i, spec in enumerate(remaining_specs):
            # Select few-shot examples
            examples = select_few_shot(by_intent, spec.intent, rng, n=2)

            # Generate
            record = generate_conversation(spec, examples, client, model)

            if record is None:
                failed_count += 1
                logger.warning(
                    "[%d/%d] FAILED %s (intent=%s)",
                    i + 1,
                    len(remaining_specs),
                    spec.conversation_id,
                    spec.intent,
                )
                continue

            # Write immediately (append mode for crash safety)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            generated_count += 1

            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta_seconds = (len(remaining_specs) - i - 1) / rate if rate > 0 else 0
            eta_minutes = eta_seconds / 60

            if (i + 1) % 10 == 0 or i == 0:
                logger.info(
                    "[%d/%d] Generated %s (intent=%s, turns=%d, persona=%s) "
                    "| Total: %d | Rate: %.1f/min | ETA: %.0f min",
                    i + 1,
                    len(remaining_specs),
                    spec.conversation_id,
                    spec.intent,
                    spec.num_turns,
                    spec.persona,
                    generated_count,
                    rate * 60,
                    eta_minutes,
                )

            # Rate limiting
            if batch_delay > 0:
                time.sleep(batch_delay)

    elapsed_total = time.time() - start_time
    logger.info("─" * 60)
    logger.info("Generation complete")
    logger.info("  Generated: %d", generated_count)
    logger.info("  Failed: %d", failed_count)
    logger.info("  Time: %.1f minutes", elapsed_total / 60)
    logger.info("  Output: %s", output_path)


if __name__ == "__main__":
    main()
