"""Category service — manages rule-based categories over the conversation corpus.

Each category is a named DSL rule. When applied, the rule engine evaluates the
rule against every context window in the store and records which windows (and
conversations) match.

Categories are stored in-memory and persisted to a JSON file so they survive
server restarts.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from demo.backend.services.conversation_store import ConversationStore

from talkex.rules.ast import AndNode, ASTNode, NotNode, OrNode, PredicateNode
from talkex.rules.compiler import SimpleRuleCompiler
from talkex.rules.config import PredicateType, RuleEngineConfig
from talkex.rules.evaluator import SimpleRuleEvaluator
from talkex.rules.models import RuleDefinition, RuleEvaluationInput

logger = logging.getLogger(__name__)


@dataclass
class CategoryMatch:
    """A single window that matched a category rule."""

    window_id: str
    conversation_id: str
    score: float
    matched_text: str | None = None


@dataclass
class Category:
    """A named category with its DSL rule and match results."""

    category_id: str
    name: str
    dsl_expression: str
    description: str = ""
    matches: list[CategoryMatch] = field(default_factory=list)
    match_count: int = 0
    conversation_count: int = 0
    created_at: str = ""
    applied: bool = False
    apply_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Semantic feature computation
# ---------------------------------------------------------------------------


ALL_PREDICATE_FAMILIES = {"lexical", "semantic", "structural", "contextual"}


def _collect_predicates(node: ASTNode) -> list[PredicateNode]:
    """Walk the AST and collect all predicate leaf nodes."""
    if isinstance(node, PredicateNode):
        return [node]
    if isinstance(node, AndNode | OrNode):
        result: list[PredicateNode] = []
        for child in node.children:
            result.extend(_collect_predicates(child))
        return result
    if isinstance(node, NotNode):
        return _collect_predicates(node.child)
    return []


def _compute_ast_depth(node: ASTNode) -> int:
    """Compute the maximum depth of an AST tree."""
    if isinstance(node, PredicateNode):
        return 1
    if isinstance(node, AndNode | OrNode):
        if not node.children:
            return 1
        return 1 + max(_compute_ast_depth(c) for c in node.children)
    if isinstance(node, NotNode):
        return 1 + _compute_ast_depth(node.child)
    return 1


def _analyze_pre_execution(ast: ASTNode, embedding_model: str) -> dict:
    """Analyze a compiled DSL query before execution.

    Returns pre-execution analysis with predicate coverage, threshold
    warnings, complexity, and known pitfalls.
    """
    predicates = _collect_predicates(ast)
    families_used = sorted({p.predicate_type.value for p in predicates})
    families_missing = sorted(ALL_PREDICATE_FAMILIES - set(families_used))

    count = len(predicates)
    if count <= 2:
        complexity = "simple"
    elif count <= 5:
        complexity = "moderate"
    else:
        complexity = "complex"

    threshold_warnings: list[str] = []
    pitfalls: list[str] = []

    for p in predicates:
        # Threshold warnings for semantic predicates
        if p.predicate_type == PredicateType.SEMANTIC and p.threshold is not None:
            if embedding_model == "null" and p.threshold > 0.15:
                threshold_warnings.append(
                    f"Null embedding model produces low scores. "
                    f"Threshold {p.threshold} on '{p.field_name}' may match nothing. Try < 0.10"
                )
            elif embedding_model != "null" and p.threshold > 0.95:
                threshold_warnings.append(f"Threshold {p.threshold} on '{p.field_name}' is very restrictive")

        # Pitfalls
        if p.predicate_type == PredicateType.SEMANTIC and embedding_model == "null":
            pitfalls.append(
                "Using semantic predicates with null embeddings — "
                "scores are deterministic but not semantically meaningful"
            )
        if p.predicate_type == PredicateType.LEXICAL and p.operator == "contains":
            val = str(p.value) if p.value else ""
            if len(val) <= 2:
                pitfalls.append(f"Keyword '{val}' is very short — may produce false positives")
        if p.predicate_type == PredicateType.LEXICAL and p.operator == "regex":
            val = str(p.value) if p.value else ""
            if ".*" in val:
                pitfalls.append(f"Regex '{val}' contains greedy '.*' — may match too broadly")

    # Deduplicate pitfalls
    pitfalls = list(dict.fromkeys(pitfalls))

    return {
        "predicate_families": families_used,
        "missing_families": families_missing,
        "predicate_count": count,
        "complexity": complexity,
        "threshold_warnings": threshold_warnings,
        "pitfalls": pitfalls,
    }


def _analyze_post_execution(
    all_scores: list[float],
    match_count: int,
    conversation_count: int,
    total_windows: int,
    total_conversations: int,
    pre_analysis: dict,
) -> dict:
    """Analyze query results after execution.

    Computes score distribution, coverage, concentration, signal warnings,
    and an overall quality score (0-100).
    """
    window_coverage = (match_count / total_windows * 100) if total_windows > 0 else 0.0
    conv_coverage = (conversation_count / total_conversations * 100) if total_conversations > 0 else 0.0
    concentration = (conversation_count / match_count) if match_count > 0 else 0.0

    # Score distribution
    score_dist = None
    if all_scores:
        sorted_scores = sorted(all_scores)
        n = len(sorted_scores)
        score_dist = {
            "min": round(sorted_scores[0], 4),
            "max": round(sorted_scores[-1], 4),
            "mean": round(sum(sorted_scores) / n, 4),
            "median": round(sorted_scores[n // 2], 4),
            "p90": round(sorted_scores[int(n * 0.9)], 4),
        }

    # Signal warnings
    signal_warnings: list[str] = []
    if all_scores:
        low_score_count = sum(1 for s in all_scores if s < 0.05)
        low_pct = low_score_count / len(all_scores) * 100
        if low_pct > 70:
            signal_warnings.append(
                f"{low_pct:.0f}% of matches have score < 0.05 — possible noise from permissive threshold"
            )

    if window_coverage > 80:
        signal_warnings.append(f"{window_coverage:.0f}% of corpus matched — rule may be too broad")
    elif match_count > 0 and window_coverage < 0.5:
        signal_warnings.append(f"Only {window_coverage:.1f}% of corpus matched — rule may be too restrictive")

    if match_count > 0 and concentration < 0.1:
        signal_warnings.append("Matches cluster in very few conversations")

    if match_count == 0:
        signal_warnings.append("No matches — try lowering thresholds or broadening conditions")

    # Quality score (0-100)
    quality = 50

    # Coverage sweet spot: 1-30%
    if 1 <= window_coverage <= 30:
        quality += 20
    elif 0.5 <= window_coverage < 1 or 30 < window_coverage <= 50:
        quality += 10

    # Score distribution quality
    if score_dist:
        if score_dist["mean"] > 0.5:
            quality += 15
        elif score_dist["mean"] > 0.3:
            quality += 10
        elif score_dist["mean"] > 0.1:
            quality += 5

    # Concentration
    if concentration > 0.3:
        quality += 10
    elif concentration > 0.1:
        quality += 5

    # Predicate diversity
    if len(pre_analysis.get("predicate_families", [])) >= 2:
        quality += 5

    # Penalty for no matches
    if match_count == 0:
        quality = max(quality - 30, 5)

    # Penalty for too broad
    if window_coverage > 80:
        quality = max(quality - 20, 10)

    quality = min(quality, 100)

    return {
        "score_distribution": score_dist,
        "window_coverage_pct": round(window_coverage, 2),
        "conversation_coverage_pct": round(conv_coverage, 2),
        "concentration_ratio": round(concentration, 4),
        "signal_warnings": signal_warnings,
        "quality_score": quality,
    }


SEMANTIC_FEATURE_FIELDS = {"embedding_similarity", "intent_score"}


def _extract_semantic_refs(node: ASTNode) -> list[tuple[str, str]]:
    """Walk the AST and collect (field_name, reference_text) from semantic predicates.

    Both intent_score and embedding_similarity predicates need embedding
    computation — they differ only in their feature key name.

    Returns a deduplicated list of (field_name, ref_text) tuples.
    """
    if isinstance(node, PredicateNode):
        if node.predicate_type == PredicateType.SEMANTIC and node.field_name in SEMANTIC_FEATURE_FIELDS and node.value:
            return [(node.field_name, str(node.value))]
        return []

    if isinstance(node, AndNode | OrNode):
        refs: list[tuple[str, str]] = []
        for child in node.children:
            refs.extend(_extract_semantic_refs(child))
        return list(dict.fromkeys(refs))  # deduplicate preserving order

    if isinstance(node, NotNode):
        return _extract_semantic_refs(node.child)

    return []


def _find_best_sentence(text: str, query_vector: list[float], embedding_generator: object) -> str | None:
    """Delegate to shared text highlighting utility."""
    from demo.backend.services.text_highlighting import find_best_sentence

    return find_best_sentence(text, query_vector, embedding_generator)


def _detect_speaker(text: str) -> str | None:
    """Detect the dominant speaker role from window text markers.

    The BR dataset embeds speaker markers like ``[customer]`` and ``[agent]``
    in the text.  We count occurrences and return the role that appears most
    frequently.  Returns ``None`` when no markers are found.
    """
    customer_count = text.lower().count("[customer]")
    agent_count = text.lower().count("[agent]")
    if customer_count == 0 and agent_count == 0:
        return None
    if customer_count >= agent_count:
        return "customer"
    return "agent"


class CategoryService:
    """Manages categories backed by the TalkEx rule engine.

    Args:
        store: ConversationStore with loaded windows.
        persist_path: Path to JSON file for category persistence.
        embedding_generator: Optional embedding generator for semantic features.
        vector_index: Optional vector index for retrieving window embeddings.
    """

    def __init__(
        self,
        store: ConversationStore,
        persist_path: str | Path,
        embedding_generator: object | None = None,
        vector_index: object | None = None,
        embedding_model_name: str = "null",
    ) -> None:
        self._store = store
        self._persist_path = Path(persist_path)
        self._categories: dict[str, Category] = {}
        self._compiler = SimpleRuleCompiler()
        self._evaluator = SimpleRuleEvaluator()
        self._embedding_generator = embedding_generator
        self._vector_index = vector_index
        self._embedding_model_name = embedding_model_name
        self._load()

    def _load(self) -> None:
        """Load persisted categories from disk."""
        if not self._persist_path.exists():
            return
        with open(self._persist_path) as f:
            data = json.load(f)
        for cat_data in data:
            matches = [CategoryMatch(**m) for m in cat_data.pop("matches", [])]
            cat = Category(**cat_data, matches=matches)
            self._categories[cat.category_id] = cat
        logger.info("Loaded %d categories from %s", len(self._categories), self._persist_path)

    def _persist(self) -> None:
        """Save categories to disk."""
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(cat) for cat in self._categories.values()]
        with open(self._persist_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _compute_similarity_scores(
        self, rule_def: RuleDefinition
    ) -> tuple[dict[str, dict[str, float]], list[float] | None]:
        """Compute embedding similarity scores for semantic predicates.

        For each semantic predicate (intent_score or embedding_similarity),
        embeds the reference text and uses the vector index to compute cosine
        similarity against all stored window embeddings.

        Returns:
            Tuple of:
            - Mapping of field_name → {window_id → similarity_score}.
            - Query vector (from the first semantic predicate) for sentence highlighting.
            Both are empty/None if no semantic predicates or no infrastructure.
        """
        if self._embedding_generator is None or self._vector_index is None:
            return {}, None

        semantic_refs = _extract_semantic_refs(rule_def.ast)
        if not semantic_refs:
            return {}, None

        from talkex.embeddings.inputs import EmbeddingBatch, EmbeddingInput
        from talkex.models.enums import ObjectType
        from talkex.models.types import EmbeddingId

        scores_by_field: dict[str, dict[str, float]] = {}
        first_query_vector: list[float] | None = None

        for field_name, ref_text in semantic_refs:
            # Embed the reference text
            emb_input = EmbeddingInput(
                embedding_id=EmbeddingId(f"emb_query_{uuid.uuid4().hex[:8]}"),
                object_type=ObjectType.CONTEXT_WINDOW,
                object_id="query",
                text=ref_text,
            )
            batch = EmbeddingBatch(items=[emb_input])
            records = self._embedding_generator.generate(batch)
            if not records:
                continue

            query_vector = records[0].vector
            if first_query_vector is None:
                first_query_vector = query_vector

            # Use vector index to search against all windows
            window_count = self._store.window_count
            hits = self._vector_index.search_by_vector(query_vector, top_k=window_count)

            # Build lookup: window_id → similarity score
            score_map: dict[str, float] = {}
            for hit in hits:
                score_map[hit.object_id] = hit.score

            scores_by_field[field_name] = score_map
            logger.debug(
                "Computed %s scores for ref='%s': %d windows scored",
                field_name,
                ref_text[:50],
                len(score_map),
            )

        return scores_by_field, first_query_vector

    def _build_features(
        self,
        window_id: str,
        similarity_scores: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """Build the features dict for a window's rule evaluation.

        Injects pre-computed similarity scores so semantic predicates
        can read them during evaluation.

        Args:
            window_id: The window being evaluated.
            similarity_scores: Mapping field_name → {window_id → score}.

        Returns:
            Features dict with intent_score and/or embedding_similarity.
        """
        features: dict[str, float] = {}

        for field_name, score_map in similarity_scores.items():
            features[field_name] = score_map.get(window_id, 0.0)

        return features

    def list_categories(self) -> list[Category]:
        """Return all categories, sorted by name."""
        return sorted(self._categories.values(), key=lambda c: c.name)

    def get_category(self, category_id: str) -> Category | None:
        """Get a category by ID."""
        return self._categories.get(category_id)

    def create_category(self, name: str, dsl_expression: str, description: str = "") -> Category:
        """Create a new category and validate its DSL expression.

        Args:
            name: Human-readable category name.
            dsl_expression: TalkEx rule engine DSL expression.
            description: Optional description.

        Returns:
            The created Category (not yet applied).

        Raises:
            ValueError: If a category with this name already exists.
            talkex.exceptions.RuleError: If the DSL expression is invalid.
        """
        # Check duplicate names
        for cat in self._categories.values():
            if cat.name.lower() == name.lower():
                raise ValueError(f"Category with name '{name}' already exists")

        # Validate DSL by compiling (raises RuleError on invalid syntax)
        category_id = f"cat_{uuid.uuid4().hex[:12]}"
        self._compiler.compile(dsl_expression, category_id, name)

        cat = Category(
            category_id=category_id,
            name=name,
            dsl_expression=dsl_expression,
            description=description,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self._categories[category_id] = cat
        self._persist()
        return cat

    def delete_category(self, category_id: str) -> bool:
        """Delete a category by ID. Returns True if deleted."""
        if category_id not in self._categories:
            return False
        del self._categories[category_id]
        self._persist()
        return True

    def preview_dsl(self, dsl_expression: str, *, max_samples: int = 10) -> dict[str, object]:
        """Dry-run a DSL expression against all windows without persisting.

        Returns a dict with match_count, conversation_count, sample_matches
        (with full text and per-predicate evidence), and latency_ms.
        """
        rule_def = self._compiler.compile(dsl_expression, "preview", "preview")
        config = RuleEngineConfig()

        start = time.monotonic()

        # Pre-compute semantic similarity scores if needed
        similarity_scores, query_vector = self._compute_similarity_scores(rule_def)

        matches: list[dict[str, object]] = []
        matched_conversations: set[str] = set()

        # Cache best-sentence results per window to avoid re-embedding
        best_sentence_cache: dict[str, str | None] = {}

        for window_id, window_meta in self._store.iter_windows():
            text = window_meta.get("window_text", "")
            conversation_id = window_meta.get("conversation_id", "")

            features = self._build_features(window_id, similarity_scores)

            eval_input = RuleEvaluationInput(
                source_id=window_id,
                source_type="context_window",
                text=text,
                features=features,
                speaker_role=_detect_speaker(text),
                metadata={
                    "domain": window_meta.get("domain", ""),
                    "topic": window_meta.get("topic", ""),
                    "channel": "voice",
                },
            )

            results = self._evaluator.evaluate([rule_def], eval_input, config)
            if results and results[0].matched:
                evidence = []
                for pr in results[0].predicate_results:
                    if not pr.matched:
                        continue
                    matched_text = pr.matched_text
                    # For semantic predicates, find the best-matching sentence
                    if (
                        matched_text is None
                        and pr.predicate_type == PredicateType.SEMANTIC
                        and query_vector is not None
                        and self._embedding_generator is not None
                        and len(matches) < max_samples  # only for displayed matches
                    ):
                        if window_id not in best_sentence_cache:
                            best_sentence_cache[window_id] = _find_best_sentence(
                                text, query_vector, self._embedding_generator
                            )
                        matched_text = best_sentence_cache[window_id]

                    evidence.append(
                        {
                            "predicate_type": pr.predicate_type.value,
                            "field_name": pr.field_name,
                            "operator": pr.operator,
                            "score": pr.score,
                            "threshold": pr.threshold,
                            "matched_text": matched_text,
                        }
                    )

                matches.append(
                    {
                        "window_id": window_id,
                        "conversation_id": conversation_id,
                        "score": results[0].score,
                        "window_text": text,
                        "evidence": evidence,
                    }
                )
                matched_conversations.add(conversation_id)

        elapsed_ms = (time.monotonic() - start) * 1000

        # Compute query evaluation
        pre_analysis = _analyze_pre_execution(rule_def.ast, self._embedding_model_name)
        all_scores = [m["score"] for m in matches]
        post_analysis = _analyze_post_execution(
            all_scores=all_scores,
            match_count=len(matches),
            conversation_count=len(matched_conversations),
            total_windows=self._store.window_count,
            total_conversations=self._store.conversation_count,
            pre_analysis=pre_analysis,
        )

        return {
            "match_count": len(matches),
            "conversation_count": len(matched_conversations),
            "sample_matches": matches[:max_samples],
            "latency_ms": round(elapsed_ms, 2),
            "evaluation": {
                "pre_execution": pre_analysis,
                "post_execution": post_analysis,
            },
        }

    def apply_category(self, category_id: str) -> Category:
        """Apply a category's rule against all context windows.

        Compiles the DSL expression, evaluates it against every window text,
        and stores the matches.

        Args:
            category_id: ID of the category to apply.

        Returns:
            Updated Category with match results.

        Raises:
            KeyError: If category not found.
        """
        cat = self._categories.get(category_id)
        if cat is None:
            raise KeyError(f"Category not found: {category_id}")

        # Compile rule
        rule_def = self._compiler.compile(cat.dsl_expression, cat.category_id, cat.name)
        config = RuleEngineConfig()

        start = time.monotonic()

        # Pre-compute semantic similarity scores if needed
        similarity_scores, _query_vector = self._compute_similarity_scores(rule_def)

        matches: list[CategoryMatch] = []
        matched_conversations: set[str] = set()

        # Evaluate against every window
        for window_id, window_meta in self._store.iter_windows():
            text = window_meta.get("window_text", "")
            conversation_id = window_meta.get("conversation_id", "")

            features = self._build_features(window_id, similarity_scores)

            eval_input = RuleEvaluationInput(
                source_id=window_id,
                source_type="context_window",
                text=text,
                features=features,
                speaker_role=_detect_speaker(text),
                metadata={
                    "domain": window_meta.get("domain", ""),
                    "topic": window_meta.get("topic", ""),
                    "channel": "voice",
                },
            )

            results = self._evaluator.evaluate([rule_def], eval_input, config)
            if results and results[0].matched:
                # Extract matched text from predicate results
                matched_text = None
                for pr in results[0].predicate_results:
                    if pr.matched and pr.matched_text:
                        matched_text = pr.matched_text
                        break

                matches.append(
                    CategoryMatch(
                        window_id=window_id,
                        conversation_id=conversation_id,
                        score=results[0].score,
                        matched_text=matched_text,
                    )
                )
                matched_conversations.add(conversation_id)

        elapsed_ms = (time.monotonic() - start) * 1000

        # Update category
        cat.matches = matches
        cat.match_count = len(matches)
        cat.conversation_count = len(matched_conversations)
        cat.applied = True
        cat.apply_time_ms = round(elapsed_ms, 2)

        self._persist()
        logger.info(
            "Category '%s' applied: %d window matches across %d conversations in %.2f ms",
            cat.name,
            cat.match_count,
            cat.conversation_count,
            elapsed_ms,
        )
        return cat
