"""Ingest Brazilian Portuguese customer service conversations from HuggingFace.

Downloads RichardSakaguchiMS/brazilian-customer-service-conversations,
converts multi-turn dialogues to TalkEx JSONL format.

Output: JSON Lines file with ingested conversations ready for pipeline processing.

Usage:
    python demo/scripts/ingest_br_dataset.py --output demo/data/conversations.jsonl
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ID = "RichardSakaguchiMS/brazilian-customer-service-conversations"


def _conversation_to_record(row: dict) -> dict | None:
    """Convert a single HuggingFace row to TalkEx conversation record.

    The dataset has:
      - id: "conv_00042"
      - messages: [{"role": "customer", "content": "..."}, ...]
      - metadata: {"intent": ..., "sentiment": ..., "sector": ..., "turns": ...}
    """
    messages = row.get("messages", [])
    if not messages:
        return None

    # Build full conversation text from turns
    turn_texts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "").strip()
        if content:
            turn_texts.append(f"[{role}] {content}")

    text = "\n".join(turn_texts)
    if len(text.split()) < 10:
        return None

    metadata = row.get("metadata", {})
    conv_id = row.get("id", "")

    # Map sector to domain
    sector = metadata.get("sector", "unknown")
    intent = metadata.get("intent", "unknown")
    sentiment = metadata.get("sentiment", "neutral")

    return {
        "conversation_id": conv_id,
        "text": text,
        "domain": sector,
        "topic": intent,
        "asr_confidence": 0.95,  # synthetic text, high confidence
        "audio_duration_seconds": metadata.get("turns", 6) * 15,  # estimate
        "word_count": len(text.split()),
        "source_file": REPO_ID,
        "sentiment": sentiment,
    }


@click.command()
@click.option("--output", default="demo/data/conversations.jsonl", help="Output JSONL path.")
@click.option("--limit", default=None, type=int, help="Max conversations to ingest (default: all).")
def main(output: str, limit: int | None) -> None:
    """Ingest PT-BR customer service conversations from HuggingFace."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading dataset from %s...", REPO_ID)
    ds = load_dataset(REPO_ID)

    # Combine all splits
    all_rows: list[dict] = []
    for split_name in ds:
        split = ds[split_name]
        logger.info("  Split '%s': %d rows", split_name, len(split))
        for row in split:
            all_rows.append(row)

    logger.info("Total rows across all splits: %d", len(all_rows))

    # Convert to TalkEx format
    conversations: list[dict] = []
    skipped = 0
    domains: dict[str, int] = {}
    intents: dict[str, int] = {}

    for row in all_rows:
        if limit is not None and len(conversations) >= limit:
            break

        record = _conversation_to_record(row)
        if record is None:
            skipped += 1
            continue

        conversations.append(record)
        domains[record["domain"]] = domains.get(record["domain"], 0) + 1
        intents[record["topic"]] = intents.get(record["topic"], 0) + 1

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    logger.info("─" * 60)
    logger.info("Ingestion complete")
    logger.info("  Total conversations: %d", len(conversations))
    logger.info("  Skipped: %d", skipped)
    logger.info("  Domains: %s", json.dumps(domains, indent=4, ensure_ascii=False))
    logger.info("  Intents: %s", json.dumps(intents, indent=4, ensure_ascii=False))
    logger.info("  Output: %s", output_path)


if __name__ == "__main__":
    main()
