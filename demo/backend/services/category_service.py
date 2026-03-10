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

import numpy as np
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


def _extract_similarity_refs(node: ASTNode) -> list[str]:
    """Walk the AST and collect reference texts from similarity predicates.

    Returns a deduplicated list of reference texts that need embedding
    computation for semantic evaluation.
    """
    if isinstance(node, PredicateNode):
        if node.predicate_type == PredicateType.SEMANTIC and node.field_name == "embedding_similarity" and node.value:
            return [str(node.value)]
        return []

    if isinstance(node, (AndNode, OrNode)):
        refs: list[str] = []
        for child in node.children:
            refs.extend(_extract_similarity_refs(child))
        return list(dict.fromkeys(refs))  # deduplicate preserving order

    if isinstance(node, NotNode):
        return _extract_similarity_refs(node.child)

    return []


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


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
    ) -> None:
        self._store = store
        self._persist_path = Path(persist_path)
        self._categories: dict[str, Category] = {}
        self._compiler = SimpleRuleCompiler()
        self._evaluator = SimpleRuleEvaluator()
        self._embedding_generator = embedding_generator
        self._vector_index = vector_index
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

    def _compute_similarity_scores(self, rule_def: RuleDefinition) -> dict[str, dict[str, float]]:
        """Compute embedding similarity scores for semantic predicates.

        For each reference text found in similarity predicates, embeds the
        reference text and uses the vector index to compute cosine similarity
        against all stored window embeddings.

        Returns:
            Mapping of reference_text → {window_id → similarity_score}.
            Empty dict if no semantic predicates or no embedding infrastructure.
        """
        if self._embedding_generator is None or self._vector_index is None:
            return {}

        ref_texts = _extract_similarity_refs(rule_def.ast)
        if not ref_texts:
            return {}

        from talkex.embeddings.inputs import EmbeddingBatch, EmbeddingInput
        from talkex.models.enums import ObjectType
        from talkex.models.types import EmbeddingId

        scores_by_ref: dict[str, dict[str, float]] = {}

        for ref_text in ref_texts:
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

            # Use vector index to search against all windows
            window_count = self._store.window_count
            hits = self._vector_index.search_by_vector(query_vector, top_k=window_count)

            # Build lookup: window_id → similarity score
            score_map: dict[str, float] = {}
            for hit in hits:
                score_map[hit.object_id] = hit.score

            scores_by_ref[ref_text] = score_map
            logger.debug(
                "Computed similarity scores for ref='%s': %d windows scored",
                ref_text[:50],
                len(score_map),
            )

        return scores_by_ref

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
            similarity_scores: Mapping ref_text → {window_id → score}.

        Returns:
            Features dict with embedding_similarity populated.
        """
        features: dict[str, float] = {}

        if similarity_scores:
            # Use the first reference text's score (most common case: one
            # similarity predicate per rule). For multiple, the rule engine
            # reads features["embedding_similarity"] which gets the score
            # for the first reference.
            for _ref_text, score_map in similarity_scores.items():
                features["embedding_similarity"] = score_map.get(window_id, 0.0)
                break

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
        similarity_scores = self._compute_similarity_scores(rule_def)

        matches: list[dict[str, object]] = []
        matched_conversations: set[str] = set()

        for window_id, window_meta in self._store.iter_windows():
            text = window_meta.get("window_text", "")
            conversation_id = window_meta.get("conversation_id", "")

            features = self._build_features(window_id, similarity_scores)

            eval_input = RuleEvaluationInput(
                source_id=window_id,
                source_type="context_window",
                text=text,
                features=features,
                metadata={
                    "domain": window_meta.get("domain", ""),
                    "topic": window_meta.get("topic", ""),
                },
            )

            results = self._evaluator.evaluate([rule_def], eval_input, config)
            if results and results[0].matched:
                evidence = [
                    {
                        "predicate_type": pr.predicate_type.value,
                        "field_name": pr.field_name,
                        "operator": pr.operator,
                        "score": pr.score,
                        "threshold": pr.threshold,
                        "matched_text": pr.matched_text,
                    }
                    for pr in results[0].predicate_results
                    if pr.matched
                ]

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

        return {
            "match_count": len(matches),
            "conversation_count": len(matched_conversations),
            "sample_matches": matches[:max_samples],
            "latency_ms": round(elapsed_ms, 2),
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
        similarity_scores = self._compute_similarity_scores(rule_def)

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
                metadata={
                    "domain": window_meta.get("domain", ""),
                    "topic": window_meta.get("topic", ""),
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
