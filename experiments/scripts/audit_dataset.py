"""Phase 1 — Dataset Audit Protocol.

Executes all 8 audit checks defined in docs/dataset-audit-protocol.md v1.1:
  1. Schema validation
  2. Deduplication (global, two-tier thresholds)
  3. Few-shot leakage detection
  4. Split integrity
  5. Taxonomy audit (coherence + separability)
  6. "Outros" composition analysis (A/B/C categorization)
  7. Text quality
  8. Distribution analysis

Usage:
    python experiments/scripts/audit_dataset.py
    python experiments/scripts/audit_dataset.py --no-embeddings  # skip embedding checks
    python experiments/scripts/audit_dataset.py --data-dir experiments/data
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import click
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_INTENTS = {
    "cancelamento",
    "compra",
    "duvida_produto",
    "duvida_servico",
    "elogio",
    "outros",
    "reclamacao",
    "saudacao",
    "suporte_tecnico",
}

VALID_DOMAINS = {
    "telecom",
    "financeiro",
    "ecommerce",
    "saude",
    "restaurante",
    "educacao",
    "tecnologia",
    "imobiliario",
}

VALID_SENTIMENTS = {"positive", "negative", "neutral"}

NEAR_DUPLICATE_FLAG_THRESHOLD = 0.92
NEAR_DUPLICATE_HARD_THRESHOLD = 0.97
CLASS_COHESION_MIN = 0.60
CLASS_OVERLAP_MAX = 0.90
OUTROS_RECLASSIFY_SIMILARITY = 0.85

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    """Load records from JSONL file."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_expanded_metadata(expanded_path: Path) -> dict[str, list[str]]:
    """Load few_shot_ids from expanded.jsonl for leakage detection.

    Returns mapping of conversation_id → few_shot_ids.
    """
    metadata_map: dict[str, list[str]] = {}
    if not expanded_path.exists():
        logger.warning("Expanded dataset not found: %s", expanded_path)
        return metadata_map

    with open(expanded_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cid = record.get("conversation_id", "")
            meta = record.get("metadata", {})
            if meta and "few_shot_ids" in meta:
                metadata_map[cid] = meta["few_shot_ids"]
    return metadata_map


def write_report(report: dict, path: Path) -> None:
    """Write JSON report to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Report written: %s", path)


# ---------------------------------------------------------------------------
# Check 1: Schema Validation
# ---------------------------------------------------------------------------


def audit_schema(records: list[dict]) -> dict:
    """Validate every record against the expected schema."""
    violations: list[dict] = []

    for i, r in enumerate(records):
        record_violations: list[str] = []
        cid = r.get("conversation_id", f"MISSING_ID_at_index_{i}")

        # conversation_id
        if not isinstance(r.get("conversation_id"), str) or not r["conversation_id"]:
            record_violations.append("conversation_id: missing or empty")

        # text
        text = r.get("text", "")
        if not isinstance(text, str) or not text.strip():
            record_violations.append("text: missing or empty")

        # topic
        topic = r.get("topic", "")
        if topic not in VALID_INTENTS:
            record_violations.append(f"topic: invalid value '{topic}'")

        # domain
        domain = r.get("domain", "")
        if domain not in VALID_DOMAINS:
            record_violations.append(f"domain: invalid value '{domain}'")

        # sentiment
        sentiment = r.get("sentiment", "")
        if sentiment not in VALID_SENTIMENTS:
            record_violations.append(f"sentiment: invalid value '{sentiment}'")

        # word_count
        wc = r.get("word_count")
        if not isinstance(wc, int) or wc <= 0:
            record_violations.append(f"word_count: invalid value {wc}")
        elif isinstance(text, str) and text.strip():
            actual_wc = len(text.split())
            if actual_wc > 0 and abs(wc - actual_wc) / actual_wc > 0.10:
                record_violations.append(f"word_count: declared {wc} vs actual {actual_wc} (>{10}% deviation)")

        # asr_confidence
        asr = r.get("asr_confidence")
        if not isinstance(asr, (int, float)) or not (0.0 <= asr <= 1.0):
            record_violations.append(f"asr_confidence: invalid value {asr}")

        # audio_duration_seconds
        dur = r.get("audio_duration_seconds")
        if not isinstance(dur, int) or dur <= 0:
            record_violations.append(f"audio_duration_seconds: invalid value {dur}")

        # source_file
        sf = r.get("source_file", "")
        if not isinstance(sf, str) or not sf:
            record_violations.append("source_file: missing or empty")

        # metadata (optional, but if synthetic must have few_shot_ids)
        if "synth" in str(r.get("conversation_id", "")):
            meta = r.get("metadata")
            if meta is not None and not isinstance(meta, dict):
                record_violations.append("metadata: present but not a dict")

        if record_violations:
            violations.append({"conversation_id": cid, "violations": record_violations})

    status = "PASS" if len(violations) == 0 else "FAIL"
    return {
        "check": "schema_validation",
        "status": status,
        "total_records": len(records),
        "violations": len(violations),
        "details": violations[:100],  # Cap at 100 for readability
    }


# ---------------------------------------------------------------------------
# Check 2: Deduplication (global, two-tier)
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Normalize text for duplicate detection."""
    text = text.lower()
    text = re.sub(r"\[customer\]|\[agent\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def audit_deduplication(
    records: list[dict],
    embeddings: np.ndarray | None = None,
) -> dict:
    """Check for exact and near-duplicate conversations globally."""
    # Exact ID duplicates
    id_counts = Counter(r["conversation_id"] for r in records)
    id_dupes = {cid: count for cid, count in id_counts.items() if count > 1}

    # Exact text duplicates
    text_map: dict[str, list[str]] = defaultdict(list)
    for r in records:
        normalized = normalize_text(r.get("text", ""))
        text_map[normalized].append(r["conversation_id"])
    text_dupes = {ids[0]: {"duplicate_ids": ids[1:], "count": len(ids)} for ids in text_map.values() if len(ids) > 1}

    # Near-duplicates (requires embeddings)
    near_dupes_flagged: list[dict] = []
    near_dupes_hard: list[dict] = []

    if embeddings is not None:
        logger.info("Computing global pairwise cosine similarity (%d records)...", len(records))
        # Normalize embeddings for cosine similarity via dot product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        normed = embeddings / norms
        sim_matrix = normed @ normed.T

        # Extract upper triangle (avoid self-comparison and double-counting)
        n = len(records)
        id_list = [r["conversation_id"] for r in records]
        topic_list = [r.get("topic", "") for r in records]

        for i in range(n):
            for j in range(i + 1, n):
                sim = float(sim_matrix[i, j])
                if sim >= NEAR_DUPLICATE_FLAG_THRESHOLD:
                    entry = {
                        "id_a": id_list[i],
                        "id_b": id_list[j],
                        "similarity": round(sim, 4),
                        "topic_a": topic_list[i],
                        "topic_b": topic_list[j],
                        "cross_intent": topic_list[i] != topic_list[j],
                    }
                    if sim >= NEAR_DUPLICATE_HARD_THRESHOLD:
                        near_dupes_hard.append(entry)
                    else:
                        near_dupes_flagged.append(entry)

        # Sort by similarity descending
        near_dupes_hard.sort(key=lambda x: -x["similarity"])
        near_dupes_flagged.sort(key=lambda x: -x["similarity"])

    total = len(records)
    total_near = len(near_dupes_flagged) + len(near_dupes_hard)
    near_pct = (total_near / total * 100) if total > 0 else 0.0

    cross_intent_flags = [d for d in near_dupes_flagged + near_dupes_hard if d["cross_intent"]]

    has_embedding_check = embeddings is not None
    status = "PASS"
    if id_dupes:
        status = "FAIL"
    if text_dupes:
        status = "FAIL"
    if near_dupes_hard:
        status = "FAIL"
    if near_pct > 2.0:
        status = "FAIL"

    return {
        "check": "deduplication",
        "status": status,
        "embedding_check_performed": has_embedding_check,
        "exact_id_duplicates": len(id_dupes),
        "exact_id_details": id_dupes,
        "exact_text_duplicates": len(text_dupes),
        "exact_text_details": dict(list(text_dupes.items())[:20]),
        "near_duplicates_hard": len(near_dupes_hard),
        "near_duplicates_hard_details": near_dupes_hard[:50],
        "near_duplicates_flagged": len(near_dupes_flagged),
        "near_duplicates_flagged_details": near_dupes_flagged[:50],
        "near_duplicate_pct": round(near_pct, 2),
        "cross_intent_flags": len(cross_intent_flags),
        "cross_intent_details": cross_intent_flags[:30],
        "thresholds": {
            "flag": NEAR_DUPLICATE_FLAG_THRESHOLD,
            "hard": NEAR_DUPLICATE_HARD_THRESHOLD,
        },
    }


# ---------------------------------------------------------------------------
# Check 3: Few-Shot Leakage Detection
# ---------------------------------------------------------------------------


def audit_leakage(
    splits: dict[str, list[dict]],
    expanded_metadata: dict[str, list[str]],
) -> dict:
    """Detect few-shot leakage between splits."""
    # Build split membership map
    id_to_split: dict[str, str] = {}
    for split_name, records in splits.items():
        for r in records:
            id_to_split[r["conversation_id"]] = split_name

    leakage_cases: list[dict] = []
    checked = 0

    for split_name, records in splits.items():
        for r in records:
            cid = r["conversation_id"]
            if r.get("source_file") != "synthetic_expansion":
                continue

            # Get few_shot_ids from expanded metadata or record metadata
            few_shot_ids = expanded_metadata.get(cid, [])
            if not few_shot_ids:
                meta = r.get("metadata", {})
                if meta:
                    few_shot_ids = meta.get("few_shot_ids", [])

            if not few_shot_ids:
                continue

            checked += 1

            for fs_id in few_shot_ids:
                fs_split = id_to_split.get(fs_id)
                if fs_split is None:
                    continue

                # Leakage: synthetic in train, few-shot source in test/val
                if split_name == "train" and fs_split in ("test", "val"):
                    leakage_cases.append(
                        {
                            "synthetic_id": cid,
                            "synthetic_split": split_name,
                            "few_shot_source_id": fs_id,
                            "few_shot_source_split": fs_split,
                            "leakage_type": f"train←{fs_split}",
                        }
                    )
                # Leakage: synthetic in val, few-shot source in test
                elif split_name == "val" and fs_split == "test":
                    leakage_cases.append(
                        {
                            "synthetic_id": cid,
                            "synthetic_split": split_name,
                            "few_shot_source_id": fs_id,
                            "few_shot_source_split": fs_split,
                            "leakage_type": "val←test",
                        }
                    )

    # Count unique contaminated synthetic records
    contaminated_ids = {lc["synthetic_id"] for lc in leakage_cases}

    # Count synthetics without recoverable metadata
    all_synthetics = []
    for records in splits.values():
        for r in records:
            if r.get("source_file") == "synthetic_expansion":
                all_synthetics.append(r["conversation_id"])
    unrecoverable = [
        cid
        for cid in all_synthetics
        if cid not in expanded_metadata
        and not (
            isinstance(
                next(
                    (r.get("metadata", {}) for rs in splits.values() for r in rs if r["conversation_id"] == cid),
                    {},
                ),
                dict,
            )
            and next(
                (
                    r.get("metadata", {}).get("few_shot_ids")
                    for rs in splits.values()
                    for r in rs
                    if r["conversation_id"] == cid
                ),
                None,
            )
        )
    ]

    status = "PASS" if len(leakage_cases) == 0 else "FAIL"

    return {
        "check": "leakage_detection",
        "status": status,
        "total_synthetics": len(all_synthetics),
        "synthetics_checked": checked,
        "synthetics_metadata_unrecoverable": len(unrecoverable),
        "contaminated_records": len(contaminated_ids),
        "contaminated_ids": sorted(contaminated_ids),
        "leakage_cases": leakage_cases[:100],
        "recommendation": (
            "Modify build_splits.py to enforce contamination-aware splitting"
            if leakage_cases
            else "No leakage detected"
        ),
    }


# ---------------------------------------------------------------------------
# Check 4: Split Integrity
# ---------------------------------------------------------------------------


def audit_split_integrity(
    splits: dict[str, list[dict]],
    consolidated: list[dict],
) -> dict:
    """Verify split overlap, coverage, and stratification."""
    issues: list[str] = []

    # Extract IDs per split
    split_ids = {name: {r["conversation_id"] for r in recs} for name, recs in splits.items()}

    # 1. No ID overlap
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = split_ids[a] & split_ids[b]
        if overlap:
            issues.append(f"Overlap {a}∩{b}: {len(overlap)} records: {sorted(overlap)[:10]}")

    # 2. Complete coverage
    all_split_ids = set()
    for ids in split_ids.values():
        all_split_ids |= ids
    consolidated_ids = {r["conversation_id"] for r in consolidated}

    missing = consolidated_ids - all_split_ids
    extra = all_split_ids - consolidated_ids
    if missing:
        issues.append(f"Missing from splits: {len(missing)} records")
    if extra:
        issues.append(f"Extra in splits (not in consolidated): {len(extra)} records")

    # 3. Stratification check (intent)
    global_intent_dist = Counter(r.get("topic", "outros") for r in consolidated)
    global_total = len(consolidated)
    intent_deviations: dict[str, dict[str, float]] = {}

    for split_name, records in splits.items():
        split_dist = Counter(r.get("topic", "outros") for r in records)
        split_total = len(records)
        for intent in global_intent_dist:
            global_pct = global_intent_dist[intent] / global_total * 100
            split_pct = split_dist.get(intent, 0) / split_total * 100 if split_total > 0 else 0
            deviation = abs(split_pct - global_pct)
            if deviation > 3.0:
                intent_deviations.setdefault(split_name, {})[intent] = round(deviation, 2)
                issues.append(f"Stratification: {split_name}/{intent} deviates {deviation:.1f}pp from global")

    # 4. Domain balance
    global_domain_dist = Counter(r.get("domain", "") for r in consolidated)
    domain_deviations: dict[str, dict[str, float]] = {}

    for split_name, records in splits.items():
        split_dist = Counter(r.get("domain", "") for r in records)
        split_total = len(records)
        for domain in global_domain_dist:
            global_pct = global_domain_dist[domain] / global_total * 100
            split_pct = split_dist.get(domain, 0) / split_total * 100 if split_total > 0 else 0
            deviation = abs(split_pct - global_pct)
            if deviation > 5.0:
                domain_deviations.setdefault(split_name, {})[domain] = round(deviation, 2)
                issues.append(f"Domain balance: {split_name}/{domain} deviates {deviation:.1f}pp from global")

    status = "PASS" if not issues else "FAIL"
    return {
        "check": "split_integrity",
        "status": status,
        "split_sizes": {name: len(recs) for name, recs in splits.items()},
        "consolidated_size": len(consolidated),
        "overlap_count": sum(
            len(split_ids[a] & split_ids[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")]
        ),
        "missing_from_splits": len(missing),
        "extra_in_splits": len(extra),
        "intent_deviations": intent_deviations,
        "domain_deviations": domain_deviations,
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# Check 5: Taxonomy Audit
# ---------------------------------------------------------------------------


def audit_taxonomy(
    records: list[dict],
    embeddings: np.ndarray | None = None,
) -> dict:
    """Audit label consistency, class coherence, and separability."""
    # 1. Label consistency
    all_topics = {r.get("topic", "") for r in records}
    invalid_labels = all_topics - VALID_INTENTS
    label_issues = [f"Invalid label: '{label}'" for label in invalid_labels]

    # Case variation check
    topic_counter = Counter(r.get("topic", "") for r in records)

    result: dict = {
        "check": "taxonomy_audit",
        "label_consistency": {
            "status": "PASS" if not invalid_labels else "FAIL",
            "invalid_labels": sorted(invalid_labels),
            "label_distribution": dict(topic_counter.most_common()),
        },
        "class_coherence": {},
        "class_separability": {},
    }

    if embeddings is None:
        result["status"] = "PASS" if not invalid_labels else "FAIL"
        result["note"] = "Embedding checks skipped (--no-embeddings)"
        return result

    # Group embeddings by intent
    intent_groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        intent_groups[r.get("topic", "outros")].append(i)

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = embeddings / norms

    # 2. Intra-class cohesion
    cohesion: dict[str, float] = {}
    flagged_classes: list[str] = []

    for intent, indices in intent_groups.items():
        if len(indices) < 2:
            cohesion[intent] = 1.0
            continue
        group_embs = normed[indices]
        sim_matrix = group_embs @ group_embs.T
        # Mean of upper triangle (exclude diagonal)
        n = len(indices)
        upper_sum = (sim_matrix.sum() - np.trace(sim_matrix)) / 2
        n_pairs = n * (n - 1) / 2
        mean_sim = float(upper_sum / n_pairs) if n_pairs > 0 else 1.0
        cohesion[intent] = round(mean_sim, 4)
        if mean_sim < CLASS_COHESION_MIN:
            flagged_classes.append(intent)

    result["class_coherence"] = {
        "cohesion_per_class": cohesion,
        "flagged_below_threshold": flagged_classes,
        "threshold": CLASS_COHESION_MIN,
    }

    # 3. Inter-class separability
    centroids: dict[str, np.ndarray] = {}
    for intent, indices in intent_groups.items():
        centroids[intent] = normed[indices].mean(axis=0)
        centroid_norm = np.linalg.norm(centroids[intent])
        if centroid_norm > 0:
            centroids[intent] = centroids[intent] / centroid_norm

    overlapping_pairs: list[dict] = []
    intents = sorted(intent_groups.keys())
    for i_idx in range(len(intents)):
        for j_idx in range(i_idx + 1, len(intents)):
            a, b = intents[i_idx], intents[j_idx]
            sim = float(centroids[a] @ centroids[b])
            if sim > CLASS_OVERLAP_MAX:
                overlapping_pairs.append(
                    {
                        "pair": f"{a}↔{b}",
                        "centroid_similarity": round(sim, 4),
                    }
                )

    overlapping_pairs.sort(key=lambda x: -x["centroid_similarity"])

    result["class_separability"] = {
        "overlapping_pairs": overlapping_pairs,
        "threshold": CLASS_OVERLAP_MAX,
    }

    has_issues = bool(invalid_labels) or bool(flagged_classes)
    result["status"] = "FAIL" if has_issues else "PASS"
    result["issues"] = label_issues

    return result


# ---------------------------------------------------------------------------
# Check 6: "Outros" Composition Analysis
# ---------------------------------------------------------------------------


def audit_outros(
    records: list[dict],
    embeddings: np.ndarray | None = None,
) -> dict:
    """Analyze the composition of the 'outros' class for A/B/C categorization."""
    from sklearn.cluster import KMeans

    outros_indices = [i for i, r in enumerate(records) if r.get("topic") == "outros"]
    non_outros_intents = sorted({r.get("topic", "") for r in records if r.get("topic") != "outros"} - {""})

    result: dict = {
        "check": "outros_analysis",
        "total_outros": len(outros_indices),
        "non_outros_intents": non_outros_intents,
    }

    if embeddings is None or len(outros_indices) == 0:
        result["status"] = "SKIPPED"
        result["note"] = "Requires embeddings" if embeddings is None else "No 'outros' records"
        return result

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = embeddings / norms

    # Compute non-outros centroids
    intent_centroids: dict[str, np.ndarray] = {}
    for intent in non_outros_intents:
        indices = [i for i, r in enumerate(records) if r.get("topic") == intent]
        centroid = normed[indices].mean(axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm > 0:
            centroid = centroid / c_norm
        intent_centroids[intent] = centroid

    # Compute outros centroid
    outros_embs = normed[outros_indices]
    outros_centroid = outros_embs.mean(axis=0)
    oc_norm = np.linalg.norm(outros_centroid)
    if oc_norm > 0:
        outros_centroid = outros_centroid / oc_norm

    # Cluster outros records
    clustering_results: dict[int, dict] = {}
    best_k = 5  # Default

    for k in [3, 5, 7]:
        if k > len(outros_indices):
            continue
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(outros_embs)

        cluster_info = []
        for c in range(k):
            cluster_mask = labels == c
            cluster_size = int(cluster_mask.sum())
            cluster_embs = outros_embs[cluster_mask]
            cluster_centroid = cluster_embs.mean(axis=0)
            cc_norm = np.linalg.norm(cluster_centroid)
            if cc_norm > 0:
                cluster_centroid = cluster_centroid / cc_norm

            # Similarity to each non-outros intent
            intent_sims = {}
            for intent, ic in intent_centroids.items():
                sim = float(cluster_centroid @ ic)
                intent_sims[intent] = round(sim, 4)

            nearest_intent = max(intent_sims, key=intent_sims.get)  # type: ignore[arg-type]
            nearest_sim = intent_sims[nearest_intent]

            # Similarity to outros centroid
            outros_sim = float(cluster_centroid @ outros_centroid)

            cluster_info.append(
                {
                    "cluster_id": c,
                    "size": cluster_size,
                    "nearest_intent": nearest_intent,
                    "nearest_intent_similarity": round(nearest_sim, 4),
                    "outros_centroid_similarity": round(outros_sim, 4),
                    "all_intent_similarities": intent_sims,
                }
            )

        clustering_results[k] = {
            "inertia": round(float(kmeans.inertia_), 4),
            "clusters": cluster_info,
        }

    # Use k=5 for A/B/C categorization
    if best_k in clustering_results:
        categorization: list[dict] = []
        k_result = clustering_results[best_k]
        for cluster in k_result["clusters"]:
            nearest_sim = cluster["nearest_intent_similarity"]
            if nearest_sim >= OUTROS_RECLASSIFY_SIMILARITY:
                category = "A_mislabeled"
                action = f"Reclassify to {cluster['nearest_intent']}"
            elif nearest_sim >= 0.75:
                category = "B_ambiguous"
                action = "Keep for abstention calibration"
            else:
                category = "C_out_of_scope"
                action = "Remove from supervised training"
            categorization.append(
                {
                    **cluster,
                    "category": category,
                    "action": action,
                }
            )
        result["categorization_k5"] = categorization
    else:
        # Use the largest k available
        available_k = max(clustering_results.keys()) if clustering_results else 0
        if available_k > 0:
            result["categorization_fallback_k"] = available_k

    # Per-record nearest intent analysis
    per_record: list[dict] = []
    for idx in outros_indices:
        emb = normed[idx]
        sims = {intent: float(emb @ ic) for intent, ic in intent_centroids.items()}
        nearest = max(sims, key=sims.get)  # type: ignore[arg-type]
        nearest_sim = sims[nearest]
        outros_sim = float(emb @ outros_centroid)

        if nearest_sim >= OUTROS_RECLASSIFY_SIMILARITY:
            category = "A_mislabeled"
        elif nearest_sim >= 0.75:
            category = "B_ambiguous"
        else:
            category = "C_out_of_scope"

        per_record.append(
            {
                "conversation_id": records[idx]["conversation_id"],
                "nearest_intent": nearest,
                "nearest_similarity": round(nearest_sim, 4),
                "outros_centroid_similarity": round(outros_sim, 4),
                "category": category,
            }
        )

    category_counts = Counter(r["category"] for r in per_record)
    result["per_record_categorization"] = per_record
    result["category_summary"] = dict(category_counts)
    result["clustering_results"] = {str(k): v for k, v in clustering_results.items()}
    result["status"] = "COMPLETE"
    result["human_review_sample_size"] = min(50, len(outros_indices))

    return result


# ---------------------------------------------------------------------------
# Check 7: Text Quality
# ---------------------------------------------------------------------------


def audit_text_quality(records: list[dict]) -> dict:
    """Validate text structural quality."""
    failures: list[dict] = []

    for r in records:
        cid = r.get("conversation_id", "UNKNOWN")
        text = r.get("text", "")
        record_issues: list[str] = []

        # 1. Speaker markers
        has_customer = "[customer]" in text.lower() or "[cliente]" in text.lower()
        has_agent = "[agent]" in text.lower() or "[agente]" in text.lower()
        if not has_customer and not has_agent:
            record_issues.append("No speaker markers found")

        # 2. Turn structure (at least 2 turns)
        turns = re.split(r"\[(?:customer|agent|cliente|agente)\]", text, flags=re.IGNORECASE)
        non_empty_turns = [t.strip() for t in turns if t.strip()]
        if len(non_empty_turns) < 2:
            record_issues.append(f"Too few turns: {len(non_empty_turns)}")

        # 3. Language check (Latin characters + PT diacritics)
        if text:
            latin_chars = sum(1 for c in text if c.isalpha() and ord(c) < 0x0250)
            total_alpha = sum(1 for c in text if c.isalpha())
            latin_ratio = latin_chars / total_alpha if total_alpha > 0 else 0
            if latin_ratio < 0.90:
                record_issues.append(f"Low Latin character ratio: {latin_ratio:.2f}")

        # 4. Word count consistency (already checked in schema, but double-check here)
        actual_wc = len(text.split())

        # 5. Minimum length
        if actual_wc < 50:
            record_issues.append(f"Too short: {actual_wc} words (min 50)")

        # 6. Maximum length
        if actual_wc > 2000:
            record_issues.append(f"Too long: {actual_wc} words (max 2000)")

        if record_issues:
            failures.append({"conversation_id": cid, "issues": record_issues})

    failure_rate = len(failures) / len(records) * 100 if records else 0
    status = "PASS" if failure_rate <= 1.0 else "FAIL"

    return {
        "check": "text_quality",
        "status": status,
        "total_records": len(records),
        "failures": len(failures),
        "failure_rate_pct": round(failure_rate, 2),
        "details": failures[:50],
    }


# ---------------------------------------------------------------------------
# Check 8: Distribution Analysis
# ---------------------------------------------------------------------------


def audit_distributions(records: list[dict]) -> dict:
    """Compute comprehensive distribution analysis."""
    # Intent distribution
    intent_counts = Counter(r.get("topic", "outros") for r in records)
    total = len(records)

    intent_dist = {}
    for intent, count in intent_counts.most_common():
        intent_dist[intent] = {
            "count": count,
            "pct": round(count / total * 100, 2),
        }

    # Domain distribution
    domain_counts = Counter(r.get("domain", "") for r in records)
    domain_dist = {d: {"count": c, "pct": round(c / total * 100, 2)} for d, c in domain_counts.most_common()}

    # Sentiment distribution
    sentiment_counts = Counter(r.get("sentiment", "") for r in records)
    sentiment_dist = {s: {"count": c, "pct": round(c / total * 100, 2)} for s, c in sentiment_counts.most_common()}

    # Word count stats
    word_counts = [r.get("word_count", 0) for r in records]
    wc_arr = np.array(word_counts)

    # Turn count stats
    turn_counts = []
    for r in records:
        text = r.get("text", "")
        turns = re.split(r"\[(?:customer|agent|cliente|agente)\]", text, flags=re.IGNORECASE)
        non_empty = [t.strip() for t in turns if t.strip()]
        turn_counts.append(len(non_empty))
    tc_arr = np.array(turn_counts)

    # Original vs synthetic per intent
    source_intent: dict[str, dict[str, int]] = defaultdict(lambda: {"original": 0, "synthetic": 0})
    for r in records:
        intent = r.get("topic", "outros")
        source = "synthetic" if r.get("source_file") == "synthetic_expansion" else "original"
        source_intent[intent][source] += 1

    # Cross-tabulation: intent x domain
    intent_domain: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        intent_domain[r.get("topic", "outros")][r.get("domain", "")] += 1

    # Check for empty cells
    empty_cells = []
    for intent in VALID_INTENTS:
        for domain in VALID_DOMAINS:
            if intent_domain.get(intent, {}).get(domain, 0) == 0:
                empty_cells.append(f"{intent}x{domain}")

    # Cross-tabulation: intent x sentiment
    intent_sentiment: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        intent_sentiment[r.get("topic", "outros")][r.get("sentiment", "")] += 1

    # Imbalance ratio
    counts = list(intent_counts.values())
    imbalance_ratio = max(counts) / min(counts) if min(counts) > 0 else float("inf")

    return {
        "check": "distribution_analysis",
        "status": "COMPLETE",
        "total_records": total,
        "intent_distribution": intent_dist,
        "domain_distribution": domain_dist,
        "sentiment_distribution": sentiment_dist,
        "word_count_stats": {
            "mean": round(float(wc_arr.mean()), 1),
            "std": round(float(wc_arr.std()), 1),
            "min": int(wc_arr.min()),
            "max": int(wc_arr.max()),
            "q25": round(float(np.percentile(wc_arr, 25)), 1),
            "q50": round(float(np.percentile(wc_arr, 50)), 1),
            "q75": round(float(np.percentile(wc_arr, 75)), 1),
        },
        "turn_count_stats": {
            "mean": round(float(tc_arr.mean()), 1),
            "std": round(float(tc_arr.std()), 1),
            "min": int(tc_arr.min()),
            "max": int(tc_arr.max()),
            "q25": round(float(np.percentile(tc_arr, 25)), 1),
            "q50": round(float(np.percentile(tc_arr, 50)), 1),
            "q75": round(float(np.percentile(tc_arr, 75)), 1),
        },
        "source_per_intent": dict(source_intent),
        "intent_x_domain": {k: dict(v) for k, v in intent_domain.items()},
        "intent_x_sentiment": {k: dict(v) for k, v in intent_sentiment.items()},
        "empty_cells_intent_x_domain": empty_cells,
        "imbalance_ratio": round(imbalance_ratio, 2),
    }


# ---------------------------------------------------------------------------
# Embedding computation
# ---------------------------------------------------------------------------


def compute_embeddings(records: list[dict]) -> np.ndarray:
    """Compute conversation embeddings using sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [r.get("text", "") for r in records]
    logger.info("Computing embeddings for %d conversations...", len(texts))
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    return np.array(embeddings)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--data-dir",
    default="experiments/data",
    help="Directory containing dataset JSONL files.",
)
@click.option(
    "--output-dir",
    default="experiments/data/audit",
    help="Directory for audit reports.",
)
@click.option(
    "--with-embeddings/--no-embeddings",
    default=True,
    help="Include embedding-based checks (deduplication, taxonomy, outros).",
)
@click.option(
    "--expanded-path",
    default="experiments/data/expanded.jsonl",
    help="Path to expanded.jsonl for metadata recovery.",
)
def audit(
    data_dir: str,
    output_dir: str,
    with_embeddings: bool,
    expanded_path: str,
) -> None:
    """Run complete dataset audit protocol (v1.1).

    Executes all 8 checks defined in docs/dataset-audit-protocol.md.
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("=" * 60)
    logger.info("TalkEx Dataset Audit Protocol v1.1")
    logger.info("=" * 60)

    consolidated_path = data_path / "consolidated.jsonl"
    if not consolidated_path.exists():
        raise click.ClickException(f"consolidated.jsonl not found: {consolidated_path}")

    consolidated = load_jsonl(consolidated_path)
    logger.info("Loaded %d records from consolidated.jsonl", len(consolidated))

    splits: dict[str, list[dict]] = {}
    for split_name in ("train", "val", "test"):
        split_path = data_path / f"{split_name}.jsonl"
        if split_path.exists():
            splits[split_name] = load_jsonl(split_path)
            logger.info("Loaded %d records from %s.jsonl", len(splits[split_name]), split_name)
        else:
            logger.warning("Split file not found: %s", split_path)

    # Load expanded metadata for leakage detection
    expanded_metadata = load_expanded_metadata(Path(expanded_path))
    logger.info("Loaded metadata for %d synthetic records from expanded.jsonl", len(expanded_metadata))

    # Compute embeddings if needed
    embeddings: np.ndarray | None = None
    if with_embeddings:
        embeddings = compute_embeddings(consolidated)

    # Run checks
    logger.info("=" * 60)
    logger.info("Running audit checks...")
    logger.info("=" * 60)

    # 1. Schema validation
    logger.info("─" * 40)
    logger.info("Check 1/8: Schema Validation")
    schema_report = audit_schema(consolidated)
    write_report(schema_report, output_path / "schema_report.json")
    logger.info("  Status: %s (violations: %d)", schema_report["status"], schema_report["violations"])

    # 2. Deduplication
    logger.info("─" * 40)
    logger.info("Check 2/8: Deduplication (global, two-tier)")
    dedup_report = audit_deduplication(consolidated, embeddings)
    write_report(dedup_report, output_path / "deduplication_report.json")
    logger.info(
        "  Status: %s (ID dupes: %d, text dupes: %d, hard near-dupes: %d, flagged: %d)",
        dedup_report["status"],
        dedup_report["exact_id_duplicates"],
        dedup_report["exact_text_duplicates"],
        dedup_report["near_duplicates_hard"],
        dedup_report["near_duplicates_flagged"],
    )

    # 3. Few-shot leakage
    logger.info("─" * 40)
    logger.info("Check 3/8: Few-Shot Leakage Detection")
    leakage_report = audit_leakage(splits, expanded_metadata)
    write_report(leakage_report, output_path / "leakage_report.json")
    logger.info(
        "  Status: %s (contaminated: %d)",
        leakage_report["status"],
        leakage_report["contaminated_records"],
    )

    # 4. Split integrity
    logger.info("─" * 40)
    logger.info("Check 4/8: Split Integrity")
    integrity_report = audit_split_integrity(splits, consolidated)
    write_report(integrity_report, output_path / "split_integrity_report.json")
    logger.info("  Status: %s (issues: %d)", integrity_report["status"], len(integrity_report["issues"]))

    # 5. Taxonomy audit
    logger.info("─" * 40)
    logger.info("Check 5/8: Taxonomy Audit")
    taxonomy_report = audit_taxonomy(consolidated, embeddings)
    write_report(taxonomy_report, output_path / "taxonomy_report.json")
    logger.info("  Status: %s", taxonomy_report["status"])

    # 6. "Outros" analysis
    logger.info("─" * 40)
    logger.info("Check 6/8: 'Outros' Composition Analysis")
    outros_report = audit_outros(consolidated, embeddings)
    write_report(outros_report, output_path / "outros_redefinition_report.json")
    logger.info("  Status: %s", outros_report["status"])
    if "category_summary" in outros_report:
        logger.info("  Categories: %s", outros_report["category_summary"])

    # 7. Text quality
    logger.info("─" * 40)
    logger.info("Check 7/8: Text Quality")
    text_report = audit_text_quality(consolidated)
    write_report(text_report, output_path / "text_quality_report.json")
    logger.info(
        "  Status: %s (failures: %d, rate: %.2f%%)",
        text_report["status"],
        text_report["failures"],
        text_report["failure_rate_pct"],
    )

    # 8. Distribution analysis
    logger.info("─" * 40)
    logger.info("Check 8/8: Distribution Analysis")
    dist_report = audit_distributions(consolidated)
    write_report(dist_report, output_path / "distribution_report.json")
    logger.info("  Status: %s", dist_report["status"])
    if dist_report["empty_cells_intent_x_domain"]:
        logger.info("  Empty cells: %s", dist_report["empty_cells_intent_x_domain"])

    # Consolidated summary
    logger.info("=" * 60)
    logger.info("AUDIT SUMMARY")
    logger.info("=" * 60)

    blocking_issues: list[str] = []
    recommendations: list[str] = []

    checks_summary = {
        "schema_validation": {"status": schema_report["status"], "violations": schema_report["violations"]},
        "deduplication": {
            "status": dedup_report["status"],
            "exact_dupes": dedup_report["exact_id_duplicates"] + dedup_report["exact_text_duplicates"],
            "hard_near_dupes": dedup_report["near_duplicates_hard"],
            "flagged_near_dupes": dedup_report["near_duplicates_flagged"],
            "cross_intent_flags": dedup_report["cross_intent_flags"],
        },
        "leakage_detection": {
            "status": leakage_report["status"],
            "contaminated_records": leakage_report["contaminated_records"],
        },
        "split_integrity": {
            "status": integrity_report["status"],
            "overlap_count": integrity_report["overlap_count"],
        },
        "taxonomy_coherence": {
            "status": taxonomy_report["status"],
            "flagged_classes": taxonomy_report.get("class_coherence", {}).get("flagged_below_threshold", []),
        },
        "outros_analysis": {
            "status": outros_report["status"],
            "category_summary": outros_report.get("category_summary", {}),
        },
        "text_quality": {
            "status": text_report["status"],
            "failure_rate": text_report["failure_rate_pct"],
        },
        "distribution_analysis": {
            "status": dist_report["status"],
            "empty_cells": len(dist_report.get("empty_cells_intent_x_domain", [])),
        },
    }

    for check_name, info in checks_summary.items():
        if info["status"] == "FAIL":
            blocking_issues.append(check_name)

    if leakage_report["contaminated_records"] > 0:
        recommendations.append("Update build_splits.py with contamination-aware splitting")
    if dedup_report["cross_intent_flags"] > 0:
        recommendations.append(
            f"Review {dedup_report['cross_intent_flags']} cross-intent near-duplicates for mislabeling"
        )
    if outros_report.get("category_summary", {}).get("A_mislabeled", 0) > 0:
        recommendations.append("Reclassify 'outros' Category A records after human review")

    overall = "FAIL" if blocking_issues else "PASS"

    summary = {
        "audit_date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "dataset_version": "v1.0-pre-audit",
        "total_records": len(consolidated),
        "checks": checks_summary,
        "overall": overall,
        "blocking_issues": blocking_issues,
        "recommendations": recommendations,
    }

    write_report(summary, output_path / "audit_summary.json")

    logger.info("Overall: %s", overall)
    if blocking_issues:
        logger.info("Blocking issues: %s", blocking_issues)
    if recommendations:
        for rec in recommendations:
            logger.info("  → %s", rec)

    logger.info("=" * 60)
    logger.info("Reports written to: %s", output_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    audit()
