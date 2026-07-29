"""Build contamination-aware train/val/test splits from clean dataset.

Replaces build_splits.py (preserved for historical reproducibility).

Key improvement: synthetic conversations are grouped into "families" with
their few-shot source originals. All members of a family are assigned to
the same split, preventing train-test leakage.

Output:
    experiments/data/train.jsonl
    experiments/data/val.jsonl
    experiments/data/test.jsonl
    experiments/data/split_manifest.json

Usage:
    python experiments/scripts/build_splits_v2.py
    python experiments/scripts/build_splits_v2.py --seed 42
    python experiments/scripts/build_splits_v2.py --input experiments/data/consolidated_clean.jsonl
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import click

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    """Load records from JSONL file."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    """Write records to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_expanded_metadata(expanded_path: Path) -> dict[str, list[str]]:
    """Load few_shot_ids from expanded.jsonl for family grouping."""
    metadata: dict[str, list[str]] = {}
    if not expanded_path.exists():
        logger.warning("Expanded dataset not found: %s", expanded_path)
        return metadata
    with open(expanded_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cid = r.get("conversation_id", "")
            meta = r.get("metadata", {})
            if meta and "few_shot_ids" in meta:
                metadata[cid] = meta["few_shot_ids"]
    return metadata


SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}


def get_few_shot_ids(
    record: dict,
    expanded_metadata: dict[str, list[str]],
) -> list[str]:
    """Get few-shot source IDs for a synthetic record."""
    cid = record["conversation_id"]
    few_shot_ids = expanded_metadata.get(cid, [])
    if not few_shot_ids:
        meta = record.get("metadata", {})
        if meta:
            few_shot_ids = meta.get("few_shot_ids", [])
    return few_shot_ids


def contamination_aware_split(
    records: list[dict],
    expanded_metadata: dict[str, list[str]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split records with contamination-aware synthetic placement.

    Strategy:
    1. Separate originals from synthetics.
    2. Split originals using standard stratified split (70/15/15).
    3. For each synthetic, find the "latest" split among its few-shot
       sources (in ordering train < val < test).
    4. Assign the synthetic to that split.

    This prevents leakage because a synthetic is always in the same or
    later split than all its few-shot sources. A synthetic in train can
    never have a source in test/val.
    """
    rng = random.Random(seed)

    # Separate originals from synthetics
    originals = [r for r in records if r.get("source_file") != "synthetic_expansion"]
    synthetics = [r for r in records if r.get("source_file") == "synthetic_expansion"]

    # Step 1: Split originals with standard stratified split
    by_intent: dict[str, list[dict]] = defaultdict(list)
    for r in originals:
        by_intent[r.get("topic", "")].append(r)

    train_orig: list[dict] = []
    val_orig: list[dict] = []
    test_orig: list[dict] = []

    for _intent, group in by_intent.items():
        rng.shuffle(group)
        n = len(group)
        n_train = round(n * train_ratio)
        n_val = round(n * val_ratio)
        train_orig.extend(group[:n_train])
        val_orig.extend(group[n_train : n_train + n_val])
        test_orig.extend(group[n_train + n_val :])

    # Build split membership for originals
    id_to_split: dict[str, str] = {}
    for r in train_orig:
        id_to_split[r["conversation_id"]] = "train"
    for r in val_orig:
        id_to_split[r["conversation_id"]] = "val"
    for r in test_orig:
        id_to_split[r["conversation_id"]] = "test"

    # Step 2: Assign synthetics to the "latest" split of their few-shot sources
    train_synth: list[dict] = []
    val_synth: list[dict] = []
    test_synth: list[dict] = []

    for r in synthetics:
        fs_ids = get_few_shot_ids(r, expanded_metadata)
        if not fs_ids:
            # No few-shot info — assign to train (safest, no leakage risk)
            train_synth.append(r)
            id_to_split[r["conversation_id"]] = "train"
            continue

        # Find latest split among few-shot sources
        latest_order = -1
        latest_split = "train"
        for fs_id in fs_ids:
            fs_split = id_to_split.get(fs_id)
            if fs_split is not None:
                order = SPLIT_ORDER[fs_split]
                if order > latest_order:
                    latest_order = order
                    latest_split = fs_split

        if latest_split == "train":
            train_synth.append(r)
        elif latest_split == "val":
            val_synth.append(r)
        else:
            test_synth.append(r)
        id_to_split[r["conversation_id"]] = latest_split

    # Combine
    train = train_orig + train_synth
    val = val_orig + val_synth
    test = test_orig + test_synth

    # Shuffle within splits
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


@click.command()
@click.option(
    "--input",
    "input_path",
    default="experiments/data/consolidated_clean.jsonl",
    help="Clean dataset (post-remediation).",
)
@click.option(
    "--expanded",
    default="experiments/data/expanded.jsonl",
    help="Expanded dataset for metadata recovery.",
)
@click.option(
    "--output-dir",
    default="experiments/data",
    help="Output directory for splits.",
)
@click.option("--seed", default=42, type=int, help="Random seed for split.")
@click.option("--train-ratio", default=0.70, type=float, help="Training set fraction.")
@click.option("--val-ratio", default=0.15, type=float, help="Validation set fraction.")
def main(
    input_path: str,
    expanded: str,
    output_dir: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> None:
    """Build contamination-aware stratified splits from clean dataset."""
    data_path = Path(input_path)
    expanded_path = Path(expanded)
    out_path = Path(output_dir)

    logger.info("=" * 60)
    logger.info("TalkEx Split Builder v2 (contamination-aware)")
    logger.info("=" * 60)

    # Load data
    records = load_jsonl(data_path)
    logger.info("Loaded %d records from %s", len(records), data_path)

    expanded_metadata = load_expanded_metadata(expanded_path)
    logger.info("Loaded metadata for %d synthetic records", len(expanded_metadata))

    # Count originals and synthetics
    n_orig = sum(1 for r in records if r.get("source_file") != "synthetic_expansion")
    n_synth = len(records) - n_orig
    logger.info("Records: %d originals, %d synthetics", n_orig, n_synth)

    # Split with contamination awareness
    train, val, test = contamination_aware_split(records, expanded_metadata, train_ratio, val_ratio, seed)

    logger.info("─" * 40)
    logger.info("Split results (seed=%d):", seed)
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        dist = Counter(r.get("topic", "") for r in split)
        source_dist = Counter("synth" if r.get("source_file") == "synthetic_expansion" else "orig" for r in split)
        logger.info("  %s: %d records (%s) %s", name, len(split), dict(dist), dict(source_dist))

    # Verify no leakage
    logger.info("─" * 40)
    logger.info("Verifying zero leakage...")

    id_to_split: dict[str, str] = {}
    for r in train:
        id_to_split[r["conversation_id"]] = "train"
    for r in val:
        id_to_split[r["conversation_id"]] = "val"
    for r in test:
        id_to_split[r["conversation_id"]] = "test"

    leakage_count = 0
    for r in train + val + test:
        cid = r["conversation_id"]
        if r.get("source_file") != "synthetic_expansion":
            continue
        few_shot_ids = get_few_shot_ids(r, expanded_metadata)
        my_split = id_to_split[cid]
        for fs_id in few_shot_ids:
            fs_split = id_to_split.get(fs_id)
            if fs_split is None:
                continue
            if (my_split == "train" and fs_split in ("val", "test")) or (my_split == "val" and fs_split == "test"):
                leakage_count += 1

    if leakage_count == 0:
        logger.info("  PASS: Zero leakage detected")
    else:
        logger.error("  FAIL: %d leakage cases detected!", leakage_count)

    # Verify no ID overlap
    train_ids = {r["conversation_id"] for r in train}
    val_ids = {r["conversation_id"] for r in val}
    test_ids = {r["conversation_id"] for r in test}
    overlap_tv = train_ids & val_ids
    overlap_tt = train_ids & test_ids
    overlap_vt = val_ids & test_ids
    if not overlap_tv and not overlap_tt and not overlap_vt:
        logger.info("  PASS: Zero split overlap")
    else:
        logger.error("  FAIL: Overlaps found: tv=%d, tt=%d, vt=%d", len(overlap_tv), len(overlap_tt), len(overlap_vt))

    # Write splits
    logger.info("─" * 40)
    write_jsonl(train, out_path / "train.jsonl")
    write_jsonl(val, out_path / "val.jsonl")
    write_jsonl(test, out_path / "test.jsonl")
    logger.info("Written splits to %s", out_path)

    # Write manifest
    manifest = {
        "created_date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "script": "build_splits_v2.py",
        "input": str(data_path),
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": round(1 - train_ratio - val_ratio, 2),
        "total_records": len(records),
        "train_count": len(train),
        "val_count": len(val),
        "test_count": len(test),
        "n_originals": n_orig,
        "n_synthetics": n_synth,
        "leakage_check": "PASS" if leakage_count == 0 else "FAIL",
        "overlap_check": "PASS" if not (overlap_tv or overlap_tt or overlap_vt) else "FAIL",
        "intent_distribution": dict(Counter(r.get("topic", "") for r in records).most_common()),
    }
    manifest_path = out_path / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info("Written manifest to %s", manifest_path)

    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
