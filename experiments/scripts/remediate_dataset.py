"""Remediate dataset based on audit findings.

Applies the following remediations in order:
1. Remove 5 hard near-duplicates (>=0.97 cosine similarity)
2. Reclassify 1 "outros" Category A record to nearest intent
3. Move 30 "outros" Category B records to abstention calibration file
4. Remove 100 "outros" Category C records from supervised training
5. Generate remediation_decisions.json with full traceability
6. Generate post_remediation_summary.json

Usage:
    python experiments/scripts/remediate_dataset.py
"""

from __future__ import annotations

import json
import logging
from collections import Counter
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
    logger.info("Written %d records to %s", len(records), path)


@click.command()
@click.option("--data-dir", default="experiments/data", help="Data directory.")
@click.option("--audit-dir", default="experiments/data/audit", help="Audit reports directory.")
def remediate(data_dir: str, audit_dir: str) -> None:
    """Apply dataset remediations based on audit findings."""
    data_path = Path(data_dir)
    audit_path = Path(audit_dir)

    logger.info("=" * 60)
    logger.info("TalkEx Dataset Remediation")
    logger.info("=" * 60)

    # Load consolidated dataset
    consolidated = load_jsonl(data_path / "consolidated.jsonl")
    logger.info("Loaded %d records from consolidated.jsonl", len(consolidated))

    # Load audit reports
    with open(audit_path / "deduplication_report.json") as f:
        dedup_report = json.load(f)
    with open(audit_path / "outros_redefinition_report.json") as f:
        outros_report = json.load(f)

    decisions: list[dict] = []
    records_by_id = {r["conversation_id"]: r for r in consolidated}

    # ─────────────────────────────────────────────────────────
    # Step 1: Remove hard near-duplicates
    # ─────────────────────────────────────────────────────────
    logger.info("─" * 40)
    logger.info("Step 1: Removing hard near-duplicates (>=0.97)")

    dupes_to_remove: set[str] = set()
    for pair in dedup_report["near_duplicates_hard_details"]:
        remove_id = pair["id_b"]  # Keep id_a (lower ID), remove id_b
        dupes_to_remove.add(remove_id)
        decisions.append(
            {
                "conversation_id": remove_id,
                "action": "removed_duplicate",
                "reason": (
                    f"Hard near-duplicate of {pair['id_a']} "
                    f"(cosine similarity={pair['similarity']}, "
                    f"intent={pair['topic_b']})"
                ),
                "source_evidence": {
                    "kept_id": pair["id_a"],
                    "removed_id": pair["id_b"],
                    "similarity": pair["similarity"],
                    "same_intent": not pair["cross_intent"],
                },
                "review_type": "automatic",
            }
        )

    logger.info("  Removing %d hard near-duplicates: %s", len(dupes_to_remove), sorted(dupes_to_remove))

    # ─────────────────────────────────────────────────────────
    # Step 2: Reclassify Category A "outros" records
    # ─────────────────────────────────────────────────────────
    logger.info("─" * 40)
    logger.info("Step 2: Reclassifying Category A 'outros' records")

    reclassified_ids: dict[str, str] = {}  # id -> new intent
    for record_info in outros_report["per_record_categorization"]:
        if record_info["category"] == "A_mislabeled":
            cid = record_info["conversation_id"]
            new_intent = record_info["nearest_intent"]
            reclassified_ids[cid] = new_intent
            decisions.append(
                {
                    "conversation_id": cid,
                    "action": "reclassified",
                    "reason": (
                        f"Mislabeled as 'outros', nearest intent is '{new_intent}' "
                        f"(cosine similarity={record_info['nearest_similarity']})"
                    ),
                    "source_evidence": {
                        "original_label": "outros",
                        "new_label": new_intent,
                        "nearest_similarity": record_info["nearest_similarity"],
                        "outros_centroid_similarity": record_info["outros_centroid_similarity"],
                    },
                    "review_type": "automatic",
                }
            )

    logger.info("  Reclassifying %d records: %s", len(reclassified_ids), reclassified_ids)

    # ─────────────────────────────────────────────────────────
    # Step 3: Move Category B to abstention calibration
    # ─────────────────────────────────────────────────────────
    logger.info("─" * 40)
    logger.info("Step 3: Moving Category B 'outros' to abstention calibration")

    abstention_ids: set[str] = set()
    abstention_records: list[dict] = []

    for record_info in outros_report["per_record_categorization"]:
        if record_info["category"] == "B_ambiguous":
            cid = record_info["conversation_id"]
            abstention_ids.add(cid)
            if cid in records_by_id:
                abstention_records.append(records_by_id[cid])
            decisions.append(
                {
                    "conversation_id": cid,
                    "action": "moved_to_abstention",
                    "reason": (
                        f"Ambiguous 'outros' record near intent boundary "
                        f"(nearest={record_info['nearest_intent']}, "
                        f"similarity={record_info['nearest_similarity']}). "
                        f"Retained for abstention threshold calibration."
                    ),
                    "source_evidence": {
                        "nearest_intent": record_info["nearest_intent"],
                        "nearest_similarity": record_info["nearest_similarity"],
                        "outros_centroid_similarity": record_info["outros_centroid_similarity"],
                    },
                    "review_type": "automatic",
                }
            )

    logger.info("  Moving %d records to abstention calibration", len(abstention_ids))

    # ─────────────────────────────────────────────────────────
    # Step 4: Remove Category C (out-of-scope)
    # ─────────────────────────────────────────────────────────
    logger.info("─" * 40)
    logger.info("Step 4: Removing Category C 'outros' (out-of-scope)")

    out_of_scope_ids: set[str] = set()
    for record_info in outros_report["per_record_categorization"]:
        if record_info["category"] == "C_out_of_scope":
            cid = record_info["conversation_id"]
            out_of_scope_ids.add(cid)
            decisions.append(
                {
                    "conversation_id": cid,
                    "action": "removed_out_of_scope",
                    "reason": (
                        f"Out-of-scope 'outros' record, not close to any intent "
                        f"(nearest={record_info['nearest_intent']}, "
                        f"similarity={record_info['nearest_similarity']})"
                    ),
                    "source_evidence": {
                        "nearest_intent": record_info["nearest_intent"],
                        "nearest_similarity": record_info["nearest_similarity"],
                        "outros_centroid_similarity": record_info["outros_centroid_similarity"],
                    },
                    "review_type": "automatic",
                }
            )

    logger.info("  Removing %d out-of-scope records", len(out_of_scope_ids))

    # ─────────────────────────────────────────────────────────
    # Apply all changes to produce clean dataset
    # ─────────────────────────────────────────────────────────
    logger.info("─" * 40)
    logger.info("Applying remediations...")

    all_remove_ids = dupes_to_remove | abstention_ids | out_of_scope_ids
    clean_records: list[dict] = []

    for r in consolidated:
        cid = r["conversation_id"]

        # Skip removed records
        if cid in all_remove_ids:
            continue

        # Apply reclassification
        if cid in reclassified_ids:
            r = {**r, "topic": reclassified_ids[cid]}

        clean_records.append(r)

    # ─────────────────────────────────────────────────────────
    # Write outputs
    # ─────────────────────────────────────────────────────────
    logger.info("─" * 40)
    logger.info("Writing outputs...")

    write_jsonl(clean_records, data_path / "consolidated_clean.jsonl")
    write_jsonl(abstention_records, data_path / "abstention_calibration.jsonl")

    # Write remediation decisions
    decisions_output = {
        "remediation_date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "source_dataset": "consolidated.jsonl",
        "output_dataset": "consolidated_clean.jsonl",
        "total_decisions": len(decisions),
        "summary": {
            "removed_duplicate": len(dupes_to_remove),
            "reclassified": len(reclassified_ids),
            "moved_to_abstention": len(abstention_ids),
            "removed_out_of_scope": len(out_of_scope_ids),
        },
        "decisions": decisions,
    }
    decisions_path = audit_path / "remediation_decisions.json"
    with open(decisions_path, "w", encoding="utf-8") as f:
        json.dump(decisions_output, f, indent=2, ensure_ascii=False)
    logger.info("Written remediation decisions to %s", decisions_path)

    # Write post-remediation summary
    clean_intent_dist = Counter(r.get("topic", "") for r in clean_records)
    clean_domain_dist = Counter(r.get("domain", "") for r in clean_records)
    clean_source_dist = Counter(
        "synthetic" if r.get("source_file") == "synthetic_expansion" else "original" for r in clean_records
    )

    post_summary = {
        "remediation_date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "original_total": len(consolidated),
        "removed_duplicates": len(dupes_to_remove),
        "removed_out_of_scope": len(out_of_scope_ids),
        "moved_to_abstention": len(abstention_ids),
        "reclassified": len(reclassified_ids),
        "final_supervised_total": len(clean_records),
        "reduction_pct": round((1 - len(clean_records) / len(consolidated)) * 100, 2),
        "final_intent_distribution": {
            intent: {"count": count, "pct": round(count / len(clean_records) * 100, 2)}
            for intent, count in clean_intent_dist.most_common()
        },
        "final_domain_distribution": {
            domain: {"count": count, "pct": round(count / len(clean_records) * 100, 2)}
            for domain, count in clean_domain_dist.most_common()
        },
        "final_source_distribution": dict(clean_source_dist),
        "final_n_intents": len(clean_intent_dist),
        "abstention_calibration_records": len(abstention_records),
    }
    summary_path = audit_path / "post_remediation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(post_summary, f, indent=2, ensure_ascii=False)
    logger.info("Written post-remediation summary to %s", summary_path)

    # Final report
    logger.info("=" * 60)
    logger.info("REMEDIATION COMPLETE")
    logger.info("=" * 60)
    logger.info("  Original: %d records", len(consolidated))
    logger.info("  Removed (duplicates): %d", len(dupes_to_remove))
    logger.info("  Removed (out-of-scope): %d", len(out_of_scope_ids))
    logger.info("  Moved to abstention: %d", len(abstention_ids))
    logger.info("  Reclassified: %d", len(reclassified_ids))
    logger.info("  Final supervised: %d records (%.1f%% reduction)", len(clean_records), post_summary["reduction_pct"])
    logger.info("  Intents: %d (was 9)", len(clean_intent_dist))
    logger.info("  Intent distribution: %s", dict(clean_intent_dist.most_common()))


if __name__ == "__main__":
    remediate()
