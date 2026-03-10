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

from talkex.rules.compiler import SimpleRuleCompiler
from talkex.rules.config import RuleEngineConfig
from talkex.rules.evaluator import SimpleRuleEvaluator
from talkex.rules.models import RuleEvaluationInput

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


class CategoryService:
    """Manages categories backed by the TalkEx rule engine.

    Args:
        store: ConversationStore with loaded windows.
        persist_path: Path to JSON file for category persistence.
    """

    def __init__(self, store: ConversationStore, persist_path: str | Path) -> None:
        self._store = store
        self._persist_path = Path(persist_path)
        self._categories: dict[str, Category] = {}
        self._compiler = SimpleRuleCompiler()
        self._evaluator = SimpleRuleEvaluator()
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
        matches: list[CategoryMatch] = []
        matched_conversations: set[str] = set()

        # Evaluate against every window
        for window_id, window_meta in self._store.iter_windows():
            text = window_meta.get("window_text", "")
            conversation_id = window_meta.get("conversation_id", "")

            eval_input = RuleEvaluationInput(
                source_id=window_id,
                source_type="context_window",
                text=text,
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
