# TalkEx Dataset Audit Protocol

**Version:** 1.1
**Status:** Approved
**Phase:** 1 — Dataset Audit & "Outros" Redefinition
**Objective:** Establish acceptance criteria and methodology for validating dataset integrity before any downstream experiment.

---

## 1. Scope

This protocol defines the mandatory checks, acceptance criteria, and remediation actions for the TalkEx experimental dataset. Every check must PASS before proceeding to Phase 2 (multi-domain benchmark) or re-running any hypothesis experiment.

**Dataset under audit:**
- Source: `RichardSakaguchiMS/brazilian-customer-service-conversations` (2,022 original) + 235 synthetic expansions
- Total: 2,257 conversations, 9 intent classes, 8 domains
- Splits: train (70%) / val (15%) / test (15%), stratified by intent
- Files: `experiments/data/{train,val,test,consolidated}.jsonl`

---

## 2. Audit Checks

### 2.1 Schema Validation

**Objective:** Every record conforms to the expected schema.

**Required fields:**

| Field | Type | Constraint |
|---|---|---|
| `conversation_id` | str | Unique, non-empty, format `conv_NNNNN` or `conv_synth_NNNNN` |
| `text` | str | Non-empty, contains `[customer]` and/or `[agent]` markers |
| `topic` | str | One of the 9 known intents (pre-audit) or 8 intents (post-audit, without "outros") |
| `domain` | str | One of: telecom, financeiro, ecommerce, saude, restaurante, educacao, tecnologia, imobiliario |
| `sentiment` | str | One of: positive, negative, neutral |
| `word_count` | int | > 0, consistent with `len(text.split())` within ±10% |
| `asr_confidence` | float | 0.0–1.0 |
| `audio_duration_seconds` | int | > 0 |
| `source_file` | str | Non-empty |
| `metadata` | dict | Optional; if present and source is synthetic, must contain `few_shot_ids` |

**Acceptance criteria:** 0 violations. Any schema failure blocks the audit.

**Implementation:** Script `experiments/scripts/audit_dataset.py`, check `audit_schema()`.

---

### 2.2 Deduplication

**Objective:** No exact or near-duplicate conversations exist within or across splits.

**Checks:**

1. **Exact duplicates:** Same `conversation_id` appearing more than once → remove duplicate.
2. **Text duplicates:** Same normalized `text` with different `conversation_id` → flag for review.
3. **Near-duplicates (global):** Cosine similarity computed across the **entire dataset**, not just within classes. Two thresholds:
   - **Hard duplicate:** similarity ≥ 0.97 → near-verbatim copy, remove.
   - **Flag threshold:** similarity ≥ 0.92 → flag for review.
   - If flagged pair has **different labels** → **taxonomy issue** (potential mislabeling), escalate to taxonomy audit.

**Methodology:**
- Normalize text: lowercase, collapse whitespace, strip speaker markers for comparison.
- Compute pairwise cosine similarity across **all records** (not per-class) using `paraphrase-multilingual-MiniLM-L12-v2`.
- Hard duplicates (≥ 0.97): if both are synthetic, remove the one with higher few-shot overlap. If one is original and one is synthetic, remove the synthetic.
- Flagged pairs (≥ 0.92, < 0.97): document for manual review. Cross-intent flagged pairs feed into taxonomy audit (section 2.5).

**Acceptance criteria:**
- 0 exact duplicates (by ID or by normalized text).
- 0 hard duplicates (≥ 0.97) after remediation.
- Flagged near-duplicates (≥ 0.92) < 2% of total records. If > 2%, investigate and remediate.
- All cross-intent flagged pairs documented with disposition.

**Output:** `experiments/data/audit/deduplication_report.json` with flagged pairs and actions taken.

---

### 2.3 Few-Shot Leakage Detection

**Objective:** No synthetic conversation in the training set was generated using few-shot examples from the test or validation sets.

**Background:** 235 synthetic conversations were generated using 335 original conversations as few-shot examples. The current `build_splits.py` does NOT guard against this contamination.

**Check:**
1. For each synthetic record, extract `metadata.few_shot_ids` (list of original `conversation_id`s used as examples).
2. Determine which split each few-shot source belongs to.
3. If a synthetic record is in `train` and any of its few-shot sources is in `test` or `val` → **LEAKAGE DETECTED**.
4. If a synthetic record is in `val` and any of its few-shot sources is in `test` → **LEAKAGE DETECTED**.

**Remediation options (in priority order):**
1. **Preferred:** Modify `build_splits.py` to enforce contamination-aware splitting — synthetic records are always placed in the same split as their few-shot sources.
2. **Alternative:** Remove all synthetic records from the dataset and work with original-only (2,022 conversations).
3. **Fallback:** Move contaminated synthetic records to the same split as their few-shot sources, then rebalance.

**Acceptance criteria:** 0 leakage violations after remediation.

**Output:** `experiments/data/audit/leakage_report.json` with contamination map and remediation actions.

---

### 2.4 Split Integrity

**Objective:** Train/val/test splits have no overlap and preserve stratification.

**Checks:**
1. **No ID overlap:** `train ∩ val = ∅`, `train ∩ test = ∅`, `val ∩ test = ∅`.
2. **Complete coverage:** `train ∪ val ∪ test = consolidated`.
3. **Stratification:** Per-intent proportions in each split deviate < 3pp from global proportions.
4. **Domain balance:** Per-domain proportions in each split deviate < 5pp from global proportions.

**Acceptance criteria:** All 4 checks pass.

**Output:** `experiments/data/audit/split_integrity_report.json`.

---

### 2.5 Taxonomy Audit

**Objective:** Intent labels are coherent, mutually exclusive, and well-defined.

**Checks:**

1. **Label consistency:** All `topic` values are from the known set. No typos, no case variations.
2. **Class coherence:** For each intent, compute mean intra-class cosine similarity (using conversation embeddings). Classes with cohesion < 0.60 are flagged as potentially incoherent.
3. **Class separability:** For each pair of intents, compute mean inter-class similarity. Pairs with similarity > 0.90 are flagged as potentially overlapping.
4. **"Outros" composition analysis:**
   - Cluster the 131 "outros" conversations using k-means (k=3,5,7) on embeddings.
   - For each cluster, compute mean similarity to each of the 8 non-"outros" intents.
   - If a cluster's nearest intent similarity > 0.85 → those records may be mislabeled and should belong to that intent.
   - Remaining records (not close to any intent) are true "unknown" candidates.

**Acceptance criteria:**
- 0 label inconsistencies.
- All classes have cohesion ≥ 0.60.
- Flagged overlapping pairs documented with justification for keeping or merging.
- "Outros" composition analysis complete with per-cluster disposition.

**Output:** `experiments/data/audit/taxonomy_report.json`.

---

### 2.6 "Outros" Redefinition

**Objective:** Replace the supervised "outros" class with a confidence-based abstention mechanism.

**Rationale:** "Outros" currently mixes:
- Conversations that genuinely don't belong to any intent (true unknowns)
- Mislabeled conversations that belong to existing intents
- Ambiguous conversations at intent boundaries
- Default label for missing annotations

This creates a noisy, incoherent class (F1=0.095, cosine similarity 0.97 with "saudacao") that degrades both training signal and evaluation integrity.

**Protocol:**

1. **Analyze "outros" composition** (from taxonomy audit, section 2.5).
2. **Cluster and classify** each "outros" record into one of three categories:
   - **Category A — Mislabeled:** Embedding is closer to a specific intent centroid than to the "outros" centroid. Action: reclassify to nearest intent.
   - **Category B — Ambiguous:** Near intent boundaries, no single intent is clearly dominant. Action: keep for abstention threshold tuning (used as calibration data, not as supervised training label).
   - **Category C — Out-of-scope:** Genuinely does not belong to any of the 8 intents. Action: remove from supervised training.
3. **Mandatory human review (Phase 1.5):** Before applying any automatic disposition, manually validate a sample:
   - 50 "outros" records (covering all 3 categories proportionally)
   - 20 near-duplicate cross-intent pairs (from deduplication check)
   - 20 class-boundary cases (records near intent decision boundaries)
4. **Apply dispositions** based on human-validated categorization.
5. **Update label space:** Reduce from 9 to 8 supervised intents.
6. **Implement abstention:** In the classification pipeline, add confidence thresholding:
   ```
   if max(class_probabilities) < threshold:
       prediction = "unknown"  # Not a supervised class
   ```
7. **Tune threshold on validation set:** Use Category B records + validation set to optimize threshold balancing coverage (% of predictions made) vs accuracy (correctness of made predictions).

**Acceptance criteria:**
- "Outros" removed from supervised label space.
- Each "outros" record categorized as A (mislabeled), B (ambiguous), or C (out-of-scope).
- Human review of 90-record sample completed and documented.
- Reclassified records (Category A) documented with justification and embedding distances.
- Ambiguous records (Category B) retained for abstention calibration.
- Out-of-scope records (Category C) removed and documented.
- Abstention threshold tuned and documented.
- New label space has 8 intents.

**Output:** `experiments/data/audit/outros_redefinition_report.json`.

---

### 2.7 Text Quality

**Objective:** Conversation text is structurally valid and linguistically consistent.

**Checks:**

1. **Speaker markers:** Every conversation contains at least one `[customer]` and one `[agent]` marker.
2. **Turn structure:** Text can be parsed into ≥ 2 turns (at least one customer and one agent utterance).
3. **Language:** Text is predominantly Portuguese (spot-check using character distribution — > 90% Latin characters, presence of PT diacritics).
4. **Word count consistency:** `word_count` field matches actual `len(text.split())` within ±10%.
5. **Minimum length:** No conversation shorter than 50 words (below this, insufficient context for classification).
6. **Maximum length:** No conversation longer than 2000 words (outlier check).

**Acceptance criteria:**
- ≤ 1% of records fail any text quality check.
- Failures documented with record IDs.

**Output:** `experiments/data/audit/text_quality_report.json`.

---

### 2.8 Distribution Analysis

**Objective:** Characterize the dataset distributions for reproducibility documentation.

**Metrics to compute:**
1. Intent distribution (count, %, imbalance ratio).
2. Domain distribution (count, %).
3. Sentiment distribution (count, %).
4. Word count distribution (mean, std, min, max, quartiles).
5. Turn count distribution (mean, std, min, max, quartiles).
6. Original vs. synthetic distribution per intent.
7. Cross-tabulation: intent × domain (identify sparse cells).
8. Cross-tabulation: intent × sentiment.

**Acceptance criteria:** Report generated. No single cell in intent × domain cross-tab has 0 records.

**Output:** `experiments/data/audit/distribution_report.json`.

---

## 3. Audit Implementation

### 3.1 Audit Script

Create `experiments/scripts/audit_dataset.py` with the following structure:

```python
@click.command()
@click.option("--data-dir", default="experiments/data")
@click.option("--output-dir", default="experiments/data/audit")
@click.option("--with-embeddings/--no-embeddings", default=True)
def audit(data_dir, output_dir, with_embeddings):
    """Run complete dataset audit protocol."""
    # 1. Schema validation
    # 2. Deduplication check
    # 3. Few-shot leakage detection
    # 4. Split integrity
    # 5. Taxonomy audit (requires embeddings)
    # 6. "Outros" composition analysis (requires embeddings)
    # 7. Text quality
    # 8. Distribution analysis
    # → Write individual reports to output_dir
    # → Write consolidated audit_summary.json with PASS/FAIL per check
```

### 3.2 Output Structure

```
experiments/data/audit/
├── audit_summary.json            # Overall PASS/FAIL per check
├── schema_report.json            # Schema violations
├── deduplication_report.json     # Duplicate and near-duplicate analysis
├── leakage_report.json           # Few-shot contamination map
├── split_integrity_report.json   # Overlap and stratification checks
├── taxonomy_report.json          # Class coherence and separability
├── outros_redefinition_report.json  # "Outros" disposition decisions
├── text_quality_report.json      # Text structural checks
└── distribution_report.json      # Full distribution characterization
```

### 3.3 Audit Summary Format

```json
{
  "audit_date": "2026-03-12",
  "dataset_version": "v1.0-pre-audit",
  "total_records": 2257,
  "checks": {
    "schema_validation": {"status": "PASS|FAIL", "violations": 0},
    "deduplication": {"status": "PASS|FAIL", "exact_dupes": 0, "near_dupes": 0},
    "leakage_detection": {"status": "PASS|FAIL", "contaminated_records": 0},
    "split_integrity": {"status": "PASS|FAIL", "overlap_count": 0},
    "taxonomy_coherence": {"status": "PASS|FAIL", "flagged_classes": []},
    "outros_analysis": {"status": "COMPLETE", "reclassified": 0, "removed": 0},
    "text_quality": {"status": "PASS|FAIL", "failure_rate": 0.0},
    "distribution_analysis": {"status": "COMPLETE"}
  },
  "overall": "PASS|FAIL",
  "blocking_issues": [],
  "recommendations": []
}
```

---

## 4. Remediation Workflow

When a check fails:

```
FAIL detected
  ↓
Document the failure (record IDs, details)
  ↓
Apply remediation (remove, reclassify, fix)
  ↓
Re-run the audit
  ↓
All checks PASS?
  ├── YES → Proceed to rebuild splits
  └── NO  → Iterate
```

### Post-Audit Actions

1. **Rebuild splits** with contamination-aware splitting (`build_splits.py` v2).
2. **Regenerate validation report** (`validate_dataset.py` on clean data).
3. **Archive pre-audit data** as `deprecated_pre_audit/`.
4. **Update seeds** to `[13, 42, 123, 2024, 999]`.
5. **Log in research-log.md** with audit findings and decisions.
6. **Update CHANGELOG.md** under `[Unreleased]`.

---

## 5. Acceptance Gate

Phase 1 is COMPLETE when:

- [ ] All 8 audit checks PASS.
- [ ] Human review of 90-record sample completed (Phase 1.5).
- [ ] "Outros" records categorized (A/B/C) and dispositions applied.
- [ ] "Outros" removed from supervised label space (8 intents remain).
- [ ] Abstention mechanism implemented in classification pipeline.
- [ ] `build_splits.py` updated with few-shot contamination guards.
- [ ] Splits rebuilt with new seeds `[13, 42, 123, 2024, 999]`.
- [ ] Pre-audit results archived as `deprecated_pre_audit/`.
- [ ] Audit reports generated and stored in `experiments/data/audit/`.
- [ ] Research log updated with audit findings.
- [ ] CHANGELOG.md updated.

---

## 6. Dependencies

**Tools required:**
- `sentence-transformers` (embedding computation for deduplication, taxonomy audit)
- `scikit-learn` (k-means clustering for "outros" analysis)
- `numpy` (similarity computation)
- `click` (CLI interface)

**Upstream:** None (Phase 1 is the foundation).

**Downstream:** Phase 2 (multi-domain benchmark), Phase 3 (evaluation pipeline), all hypothesis re-runs.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| "Outros" removal reduces dataset | Only Category C is removed; Category A is reclassified (preserves data), Category B is retained for calibration |
| Near-duplicate removal may reduce dataset | Two-tier thresholds (0.92 flag, 0.97 hard) balance sensitivity vs false positives |
| Reclassifying "outros" records may introduce bias | Mandatory human review of 90-record sample before any automatic disposition |
| Few-shot leakage remediation may unbalance splits | Re-stratify after remediation; accept small imbalance if necessary |
| Cross-intent near-duplicates may indicate systematic issues | Escalate to taxonomy audit; if > 5% of records are cross-intent flags, review taxonomy |
