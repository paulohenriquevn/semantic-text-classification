"""Patch expanded.jsonl to add few_shot_ids to metadata.

The initial generation run started before the few_shot_ids tracking edit was
applied. Since both the generation plan and few-shot selection use deterministic
RNG (seed=42), we can reconstruct exactly which original conversations were
used as few-shot examples for each generated record.

This script replays the exact same RNG sequence used during generation and
writes the patched file with few_shot_ids added to each record's metadata.

Usage:
    python experiments/scripts/patch_few_shot_ids.py \
        --expanded experiments/data/expanded.jsonl \
        --original demo/data/conversations_original.jsonl
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import click

# Re-use the same constants and functions from expand_dataset
sys.path.insert(0, str(Path(__file__).parent))
from expand_dataset import (
    INTENTS,
    build_generation_plan,
    load_original_dataset,
    select_few_shot,
)


@click.command()
@click.option(
    "--expanded",
    default="experiments/data/expanded.jsonl",
    help="Path to the expanded dataset to patch.",
)
@click.option(
    "--original",
    default="demo/data/conversations.jsonl",
    help="Path to original dataset (for few-shot examples).",
)
@click.option("--target-total", default=3500, help="Target total conversations.")
@click.option("--seed", default=42, help="Random seed (must match generation run).")
def main(expanded: str, original: str, target_total: int, seed: int) -> None:
    """Patch expanded.jsonl to add few_shot_ids metadata."""
    expanded_path = Path(expanded)
    original_path = Path(original)

    if not expanded_path.exists():
        raise click.ClickException(f"Expanded dataset not found: {expanded_path}")
    if not original_path.exists():
        raise click.ClickException(f"Original dataset not found: {original_path}")

    # Load original dataset (same as expand_dataset.py)
    click.echo(f"Loading original dataset from {original_path}...")
    by_intent = load_original_dataset(str(original_path))
    existing_count = sum(len(v) for v in by_intent.values())
    click.echo(f"  Loaded {existing_count} conversations across {len(by_intent)} intents")

    # Replay generation plan (deterministic)
    start_id = existing_count + 1
    plan = build_generation_plan(
        target_total=target_total,
        existing_count=existing_count,
        start_id=start_id,
        seed=seed,
    )
    click.echo(f"  Replayed plan: {plan.total} specs")

    # Replay few-shot selection RNG (same seed, same sequence as generation loop)
    rng = random.Random(seed)
    spec_to_few_shot: dict[str, list[str]] = {}

    for spec in plan.specs:
        examples = select_few_shot(by_intent, spec.intent, rng, n=2)
        few_shot_ids = [ex.get("conversation_id", "unknown") for ex in examples]
        spec_to_few_shot[spec.conversation_id] = few_shot_ids

    click.echo(f"  Computed few_shot_ids for {len(spec_to_few_shot)} specs")

    # Load expanded records
    records: list[dict] = []
    with open(expanded_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    click.echo(f"  Loaded {len(records)} expanded records")

    # Patch records
    patched = 0
    already_has = 0
    not_found = 0

    for record in records:
        conv_id = record.get("conversation_id", "")
        metadata = record.get("metadata", {})

        if "few_shot_ids" in metadata:
            already_has += 1
            continue

        if conv_id in spec_to_few_shot:
            metadata["few_shot_ids"] = spec_to_few_shot[conv_id]
            record["metadata"] = metadata
            patched += 1
        else:
            not_found += 1

    click.echo(f"  Patched: {patched}, Already had IDs: {already_has}, Not in plan: {not_found}")

    if not_found > 0:
        click.echo(f"  WARNING: {not_found} records not found in generation plan!")

    # Write patched file (atomic: write to temp, then rename)
    tmp_path = expanded_path.with_suffix(".patched.jsonl")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Backup original and rename
    backup_path = expanded_path.with_suffix(".pre_patch.jsonl")
    expanded_path.rename(backup_path)
    tmp_path.rename(expanded_path)

    click.echo(f"  Patched file: {expanded_path}")
    click.echo(f"  Backup: {backup_path}")
    click.echo("Done.")


if __name__ == "__main__":
    main()
