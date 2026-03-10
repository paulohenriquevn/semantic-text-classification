"""Unit tests for rule engine AST nodes.

Tests cover: PredicateNode construction, composite node composition,
immutability, nesting, ASTNode type alias, reexport.
"""

from semantic_conversation_engine.rules.ast import (
    AndNode,
    ASTNode,
    NotNode,
    OrNode,
    PredicateNode,
)
from semantic_conversation_engine.rules.config import PredicateType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lexical_predicate(**overrides: object) -> PredicateNode:
    defaults: dict[str, object] = {
        "predicate_type": PredicateType.LEXICAL,
        "field_name": "text",
        "operator": "contains",
        "value": "billing",
    }
    defaults.update(overrides)
    return PredicateNode(**defaults)  # type: ignore[arg-type]


def _make_semantic_predicate(**overrides: object) -> PredicateNode:
    defaults: dict[str, object] = {
        "predicate_type": PredicateType.SEMANTIC,
        "field_name": "intent_score",
        "operator": "gte",
        "value": 0.8,
        "threshold": 0.7,
        "cost_hint": 4,
        "metadata": {"model_name": "e5-base"},
    }
    defaults.update(overrides)
    return PredicateNode(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# PredicateNode
# ---------------------------------------------------------------------------


class TestPredicateNode:
    def test_lexical_predicate(self) -> None:
        p = _make_lexical_predicate()
        assert p.predicate_type == PredicateType.LEXICAL
        assert p.field_name == "text"
        assert p.operator == "contains"
        assert p.value == "billing"

    def test_semantic_predicate(self) -> None:
        p = _make_semantic_predicate()
        assert p.predicate_type == PredicateType.SEMANTIC
        assert p.threshold == 0.7
        assert p.cost_hint == 4
        assert p.metadata["model_name"] == "e5-base"

    def test_structural_predicate(self) -> None:
        p = PredicateNode(
            predicate_type=PredicateType.STRUCTURAL,
            field_name="speaker_role",
            operator="eq",
            value="customer",
            cost_hint=2,
        )
        assert p.predicate_type == PredicateType.STRUCTURAL
        assert p.value == "customer"

    def test_contextual_predicate(self) -> None:
        p = PredicateNode(
            predicate_type=PredicateType.CONTEXTUAL,
            field_name="repeated_keyword",
            operator="occurs_after",
            value="billing",
            cost_hint=3,
        )
        assert p.predicate_type == PredicateType.CONTEXTUAL

    def test_default_cost_hint(self) -> None:
        p = _make_lexical_predicate()
        assert p.cost_hint == 1

    def test_default_threshold_none(self) -> None:
        p = _make_lexical_predicate()
        assert p.threshold is None

    def test_default_metadata_empty(self) -> None:
        p = _make_lexical_predicate()
        assert p.metadata == {}

    def test_frozen(self) -> None:
        p = _make_lexical_predicate()
        try:
            p.value = "cancel"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# AndNode
# ---------------------------------------------------------------------------


class TestAndNode:
    def test_construction_with_predicates(self) -> None:
        p1 = _make_lexical_predicate()
        p2 = _make_semantic_predicate()
        node = AndNode(children=[p1, p2])
        assert len(node.children) == 2

    def test_children_preserved_in_order(self) -> None:
        p1 = _make_lexical_predicate(value="first")
        p2 = _make_lexical_predicate(value="second")
        node = AndNode(children=[p1, p2])
        assert node.children[0].value == "first"  # type: ignore[union-attr]
        assert node.children[1].value == "second"  # type: ignore[union-attr]

    def test_frozen(self) -> None:
        node = AndNode(children=[_make_lexical_predicate()])
        try:
            node.children = []  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# OrNode
# ---------------------------------------------------------------------------


class TestOrNode:
    def test_construction_with_predicates(self) -> None:
        p1 = _make_lexical_predicate(value="billing")
        p2 = _make_lexical_predicate(value="invoice")
        node = OrNode(children=[p1, p2])
        assert len(node.children) == 2

    def test_frozen(self) -> None:
        node = OrNode(children=[_make_lexical_predicate()])
        try:
            node.children = []  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# NotNode
# ---------------------------------------------------------------------------


class TestNotNode:
    def test_construction(self) -> None:
        p = _make_lexical_predicate()
        node = NotNode(child=p)
        assert node.child is p

    def test_frozen(self) -> None:
        node = NotNode(child=_make_lexical_predicate())
        try:
            node.child = _make_lexical_predicate()  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Nesting / Composition
# ---------------------------------------------------------------------------


class TestASTComposition:
    def test_and_or_composition(self) -> None:
        """AND(OR(p1, p2), p3) — mixed nesting."""
        p1 = _make_lexical_predicate(value="billing")
        p2 = _make_lexical_predicate(value="invoice")
        p3 = _make_semantic_predicate()

        or_node = OrNode(children=[p1, p2])
        and_node = AndNode(children=[or_node, p3])

        assert len(and_node.children) == 2
        assert isinstance(and_node.children[0], OrNode)
        assert isinstance(and_node.children[1], PredicateNode)

    def test_not_and_composition(self) -> None:
        """NOT(AND(p1, p2)) — negated conjunction."""
        p1 = _make_lexical_predicate()
        p2 = _make_semantic_predicate()
        and_node = AndNode(children=[p1, p2])
        not_node = NotNode(child=and_node)

        assert isinstance(not_node.child, AndNode)

    def test_deep_nesting(self) -> None:
        """AND(OR(p1, NOT(p2)), p3) — three levels deep."""
        p1 = _make_lexical_predicate(value="billing")
        p2 = _make_lexical_predicate(value="cancel")
        p3 = _make_structural_predicate()

        not_p2 = NotNode(child=p2)
        or_node = OrNode(children=[p1, not_p2])
        and_node = AndNode(children=[or_node, p3])

        assert len(and_node.children) == 2
        or_child = and_node.children[0]
        assert isinstance(or_child, OrNode)
        assert isinstance(or_child.children[1], NotNode)

    def test_ast_node_type_alias(self) -> None:
        """ASTNode union accepts all node types."""
        nodes: list[ASTNode] = [
            _make_lexical_predicate(),
            AndNode(children=[_make_lexical_predicate()]),
            OrNode(children=[_make_lexical_predicate()]),
            NotNode(child=_make_lexical_predicate()),
        ]
        assert len(nodes) == 4


def _make_structural_predicate() -> PredicateNode:
    return PredicateNode(
        predicate_type=PredicateType.STRUCTURAL,
        field_name="speaker_role",
        operator="eq",
        value="customer",
        cost_hint=2,
    )


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestASTReexport:
    def test_importable_from_rules_package(self) -> None:
        from semantic_conversation_engine.rules import (
            AndNode as And,
        )
        from semantic_conversation_engine.rules import (
            ASTNode as AN,
        )
        from semantic_conversation_engine.rules import (
            NotNode as Not,
        )
        from semantic_conversation_engine.rules import (
            OrNode as Or,
        )
        from semantic_conversation_engine.rules import (
            PredicateNode as PN,
        )

        assert PN is PredicateNode
        assert And is AndNode
        assert Or is OrNode
        assert Not is NotNode
        # ASTNode is a type alias, just verify it exists
        assert AN is not None
