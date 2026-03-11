"""Build train/val/test splits from original + expanded dataset.

Combines the original conversations with the expanded synthetic conversations,
produces a unified corpus for the TalkEx pipeline AND stratified splits for
classification experiments.

Output:
    demo/data/conversations.jsonl  ← unified corpus (feeds build_index.py)
    demo/data/splits/train.jsonl   ← training set (feeds classification experiments)
    demo/data/splits/val.jsonl     ← validation set
    demo/data/splits/test.jsonl    ← test set
    demo/data/splits/manifest.json ← metadata

Usage:
    python experiments/scripts/build_splits.py \
        --original demo/data/conversations_original.jsonl \
        --expanded experiments/data/expanded.jsonl
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from pathlib import Path

import click

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def stratified_split(
    records: list[dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split records into train/val/test preserving intent distribution.

    Args:
        records: List of conversation records with 'topic' field.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train, val, test) record lists.
    """
    rng = random.Random(seed)

    # Group by intent
    by_intent: dict[str, list[dict]] = {}
    for r in records:
        intent = r.get("topic", "outros")
        by_intent.setdefault(intent, []).append(r)

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []

    for intent, group in by_intent.items():
        rng.shuffle(group)
        n = len(group)
        n_train = round(n * train_ratio)
        n_val = round(n * val_ratio)

        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])

    # Shuffle within each split
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def write_split(records: list[dict], path: Path) -> None:
    """Write records to JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def print_distribution(name: str, records: list[dict]) -> None:
    """Print intent distribution for a split."""
    counts = Counter(r.get("topic", "outros") for r in records)
    total = len(records)
    logger.info("  %s (%d conversations):", name, total)
    for intent, count in sorted(counts.items(), key=lambda x: -x[1]):
        logger.info("    %s: %d (%.1f%%)", intent, count, count / total * 100)


@click.command()
@click.option("--original", default="demo/data/conversations_original.jsonl", help="Original dataset.")
@click.option("--expanded", default="experiments/data/expanded.jsonl", help="Expanded dataset.")
@click.option("--corpus-output", default="demo/data/conversations.jsonl", help="Unified corpus for TalkEx pipeline.")
@click.option("--splits-dir", default="demo/data/splits", help="Output directory for train/val/test splits.")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option("--train-ratio", default=0.70, type=float, help="Training set fraction.")
@click.option("--val-ratio", default=0.15, type=float, help="Validation set fraction.")
def main(
    original: str,
    expanded: str,
    corpus_output: str,
    splits_dir: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> None:
    """Build unified corpus + stratified splits from combined dataset.

    Produces two complementary outputs:
    1. Unified corpus (conversations.jsonl) for the TalkEx pipeline (build_index.py)
    2. Stratified splits (train/val/test) for classification experiments (H2, H3)
    """
    original_path = Path(original)
    expanded_path = Path(expanded)
    corpus_path = Path(corpus_output)
    splits_path = Path(splits_dir)

    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.mkdir(parents=True, exist_ok=True)

    # Load datasets
    all_records: list[dict] = []

    if original_path.exists():
        with open(original_path, encoding="utf-8") as f:
            for line in f:
                all_records.append(json.loads(line))
        logger.info("Loaded %d original conversations", len(all_records))
    else:
        raise click.ClickException(f"Original dataset not found: {original_path}")

    expanded_count = 0
    if expanded_path.exists():
        with open(expanded_path, encoding="utf-8") as f:
            for line in f:
                all_records.append(json.loads(line))
                expanded_count += 1
        logger.info("Loaded %d expanded conversations", expanded_count)
    else:
        logger.warning("Expanded dataset not found: %s (using only original)", expanded_path)

    logger.info("Total: %d conversations", len(all_records))

    # 1. Write unified corpus for TalkEx pipeline
    write_split(all_records, corpus_path)
    logger.info("Unified corpus written to %s (%d conversations)", corpus_path, len(all_records))

    # 2. Build stratified splits for classification experiments
    train, val, test = stratified_split(all_records, train_ratio, val_ratio, seed)

    write_split(train, splits_path / "train.jsonl")
    write_split(val, splits_path / "val.jsonl")
    write_split(test, splits_path / "test.jsonl")

    # Report
    logger.info("─" * 60)
    logger.info("Splits created:")
    print_distribution("Train", train)
    print_distribution("Val", val)
    print_distribution("Test", test)

    # Write manifest
    manifest = {
        "original_count": len(all_records) - expanded_count,
        "expanded_count": expanded_count,
        "total_count": len(all_records),
        "train_count": len(train),
        "val_count": len(val),
        "test_count": len(test),
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": round(1 - train_ratio - val_ratio, 2),
        "intent_distribution": dict(Counter(r.get("topic", "outros") for r in all_records)),
    }
    manifest_path = splits_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info("─" * 60)
    logger.info("Pipeline integration:")
    logger.info("  Corpus: %s → feed to build_index.py", corpus_path)
    logger.info("  Splits: %s → feed to classification experiments", splits_path)
    logger.info("  Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
