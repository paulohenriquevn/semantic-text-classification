"""Unit tests for rule DSL tokenizer and parser.

Tests cover: tokenization, basic predicate parsing, logical operators (AND, OR, NOT),
operator precedence, parenthesized groups, all predicate functions, error handling,
and the parse_dsl public API.
"""

import pytest

from talkex.exceptions import RuleError
from talkex.rules.ast import (
    AndNode,
    NotNode,
    OrNode,
    PredicateNode,
)
from talkex.rules.config import PredicateType
from talkex.rules.parser import TokenType, parse_dsl, tokenize

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class TestTokenizer:
    def test_simple_predicate(self) -> None:
        tokens = tokenize('keyword("billing")')
        types = [t.token_type for t in tokens]
        assert types == [
            TokenType.IDENTIFIER,
            TokenType.LPAREN,
            TokenType.STRING,
            TokenType.RPAREN,
            TokenType.EOF,
        ]

    def test_and_or_not_keywords(self) -> None:
        tokens = tokenize("a AND b OR NOT c")
        types = [t.token_type for t in tokens]
        assert types == [
            TokenType.IDENTIFIER,
            TokenType.AND,
            TokenType.IDENTIFIER,
            TokenType.OR,
            TokenType.NOT,
            TokenType.IDENTIFIER,
            TokenType.EOF,
        ]

    def test_case_insensitive_keywords(self) -> None:
        tokens = tokenize("and or not And Or Not AND OR NOT")
        keyword_tokens = [t for t in tokens if t.token_type in (TokenType.AND, TokenType.OR, TokenType.NOT)]
        assert len(keyword_tokens) == 9

    def test_string_double_quotes(self) -> None:
        tokens = tokenize('"hello world"')
        assert tokens[0].value == "hello world"
        assert tokens[0].token_type == TokenType.STRING

    def test_string_single_quotes(self) -> None:
        tokens = tokenize("'hello'")
        assert tokens[0].value == "hello"

    def test_string_with_escape(self) -> None:
        tokens = tokenize('"say \\"hello\\"" ')
        assert tokens[0].value == 'say "hello"'

    def test_integer_literal(self) -> None:
        tokens = tokenize("42")
        assert tokens[0].value == 42
        assert tokens[0].token_type == TokenType.NUMBER

    def test_float_literal(self) -> None:
        tokens = tokenize("0.85")
        assert tokens[0].value == 0.85

    def test_negative_number(self) -> None:
        tokens = tokenize("-1")
        assert tokens[0].value == -1

    def test_parentheses_and_comma(self) -> None:
        tokens = tokenize("f(a, b)")
        types = [t.token_type for t in tokens]
        assert types == [
            TokenType.IDENTIFIER,
            TokenType.LPAREN,
            TokenType.IDENTIFIER,
            TokenType.COMMA,
            TokenType.IDENTIFIER,
            TokenType.RPAREN,
            TokenType.EOF,
        ]

    def test_whitespace_ignored(self) -> None:
        tokens = tokenize("  keyword  (  )  ")
        types = [t.token_type for t in tokens if t.token_type != TokenType.EOF]
        assert types == [TokenType.IDENTIFIER, TokenType.LPAREN, TokenType.RPAREN]

    def test_position_tracking(self) -> None:
        tokens = tokenize('keyword("billing")')
        assert tokens[0].position == 0
        assert tokens[1].position == 7
        assert tokens[2].position == 8

    def test_unterminated_string_raises(self) -> None:
        with pytest.raises(RuleError, match="Unterminated string"):
            tokenize('"hello')

    def test_unexpected_character_raises(self) -> None:
        with pytest.raises(RuleError, match="Unexpected character"):
            tokenize("keyword @ billing")

    def test_underscore_in_identifier(self) -> None:
        tokens = tokenize("field_eq")
        assert tokens[0].token_type == TokenType.IDENTIFIER
        assert tokens[0].value == "field_eq"


# ---------------------------------------------------------------------------
# Parser — basic predicates
# ---------------------------------------------------------------------------


class TestParserPredicates:
    def test_keyword_single_arg(self) -> None:
        ast = parse_dsl('keyword("billing")')
        assert isinstance(ast, PredicateNode)
        assert ast.predicate_type == PredicateType.LEXICAL
        assert ast.field_name == "text"
        assert ast.operator == "contains"
        assert ast.value == "billing"
        assert ast.cost_hint == 1

    def test_keyword_two_args(self) -> None:
        ast = parse_dsl('keyword("subject", "billing")')
        assert isinstance(ast, PredicateNode)
        assert ast.field_name == "subject"
        assert ast.value == "billing"

    def test_regex_single_arg(self) -> None:
        ast = parse_dsl('regex("bill(ing|ed)")')
        assert isinstance(ast, PredicateNode)
        assert ast.operator == "regex"
        assert ast.value == "bill(ing|ed)"

    def test_regex_two_args(self) -> None:
        ast = parse_dsl('regex("notes", "\\\\d{3}")')
        assert isinstance(ast, PredicateNode)
        assert ast.field_name == "notes"

    def test_intent_single_arg(self) -> None:
        ast = parse_dsl('intent("cancel_subscription")')
        assert isinstance(ast, PredicateNode)
        assert ast.predicate_type == PredicateType.SEMANTIC
        assert ast.field_name == "intent_score"
        assert ast.operator == "gte"
        assert ast.value == "cancel_subscription"
        assert ast.threshold is None
        assert ast.cost_hint == 4

    def test_intent_with_threshold(self) -> None:
        ast = parse_dsl('intent("cancel", 0.8)')
        assert isinstance(ast, PredicateNode)
        assert ast.threshold == 0.8

    def test_similarity(self) -> None:
        ast = parse_dsl('similarity("embedding", "billing_cluster", 0.7)')
        assert isinstance(ast, PredicateNode)
        assert ast.predicate_type == PredicateType.SEMANTIC
        assert ast.field_name == "embedding"
        assert ast.value == "billing_cluster"
        assert ast.threshold == 0.7

    def test_speaker(self) -> None:
        ast = parse_dsl('speaker("customer")')
        assert isinstance(ast, PredicateNode)
        assert ast.predicate_type == PredicateType.STRUCTURAL
        assert ast.field_name == "speaker_role"
        assert ast.operator == "eq"
        assert ast.value == "customer"
        assert ast.cost_hint == 2

    def test_channel(self) -> None:
        ast = parse_dsl('channel("voice")')
        assert isinstance(ast, PredicateNode)
        assert ast.field_name == "channel"
        assert ast.value == "voice"

    def test_field_eq(self) -> None:
        ast = parse_dsl('field_eq("status", "active")')
        assert isinstance(ast, PredicateNode)
        assert ast.field_name == "status"
        assert ast.operator == "eq"
        assert ast.value == "active"

    def test_field_gte_with_number(self) -> None:
        ast = parse_dsl('field_gte("turn_count", 5)')
        assert isinstance(ast, PredicateNode)
        assert ast.operator == "gte"
        assert ast.value == 5

    def test_field_lte(self) -> None:
        ast = parse_dsl('field_lte("duration", 300)')
        assert isinstance(ast, PredicateNode)
        assert ast.operator == "lte"
        assert ast.value == 300

    def test_repeated(self) -> None:
        ast = parse_dsl('repeated("text", "billing", 3)')
        assert isinstance(ast, PredicateNode)
        assert ast.predicate_type == PredicateType.CONTEXTUAL
        assert ast.operator == "repeated_in_window"
        assert ast.value == "billing"
        assert ast.metadata == {"count": 3}
        assert ast.cost_hint == 3

    def test_occurs_after(self) -> None:
        ast = parse_dsl('occurs_after("text", "complaint", "resolution")')
        assert isinstance(ast, PredicateNode)
        assert ast.operator == "occurs_after"
        assert ast.value == "complaint"
        assert ast.metadata == {"second": "resolution"}


# ---------------------------------------------------------------------------
# Parser — logical operators
# ---------------------------------------------------------------------------


class TestParserLogicalOperators:
    def test_and_two_predicates(self) -> None:
        ast = parse_dsl('keyword("billing") AND speaker("customer")')
        assert isinstance(ast, AndNode)
        assert len(ast.children) == 2
        assert isinstance(ast.children[0], PredicateNode)
        assert isinstance(ast.children[1], PredicateNode)

    def test_or_two_predicates(self) -> None:
        ast = parse_dsl('keyword("billing") OR keyword("invoice")')
        assert isinstance(ast, OrNode)
        assert len(ast.children) == 2

    def test_not_predicate(self) -> None:
        ast = parse_dsl('NOT keyword("upgrade")')
        assert isinstance(ast, NotNode)
        assert isinstance(ast.child, PredicateNode)

    def test_and_chains_flat(self) -> None:
        """a AND b AND c → single AndNode with 3 children."""
        ast = parse_dsl('keyword("a") AND keyword("b") AND keyword("c")')
        assert isinstance(ast, AndNode)
        assert len(ast.children) == 3

    def test_or_chains_flat(self) -> None:
        """a OR b OR c → single OrNode with 3 children."""
        ast = parse_dsl('keyword("a") OR keyword("b") OR keyword("c")')
        assert isinstance(ast, OrNode)
        assert len(ast.children) == 3


# ---------------------------------------------------------------------------
# Parser — precedence
# ---------------------------------------------------------------------------


class TestParserPrecedence:
    def test_and_binds_tighter_than_or(self) -> None:
        """a OR b AND c → OR(a, AND(b, c))."""
        ast = parse_dsl('keyword("a") OR keyword("b") AND keyword("c")')
        assert isinstance(ast, OrNode)
        assert len(ast.children) == 2
        assert isinstance(ast.children[0], PredicateNode)
        assert isinstance(ast.children[1], AndNode)

    def test_not_binds_tightest(self) -> None:
        """NOT a AND b → AND(NOT(a), b)."""
        ast = parse_dsl('NOT keyword("a") AND keyword("b")')
        assert isinstance(ast, AndNode)
        assert isinstance(ast.children[0], NotNode)
        assert isinstance(ast.children[1], PredicateNode)

    def test_parentheses_override_precedence(self) -> None:
        """(a OR b) AND c → AND(OR(a, b), c)."""
        ast = parse_dsl('(keyword("a") OR keyword("b")) AND keyword("c")')
        assert isinstance(ast, AndNode)
        assert isinstance(ast.children[0], OrNode)
        assert isinstance(ast.children[1], PredicateNode)

    def test_nested_parentheses(self) -> None:
        """((a AND b)) → AND(a, b)."""
        ast = parse_dsl('((keyword("a") AND keyword("b")))')
        assert isinstance(ast, AndNode)

    def test_not_parenthesized_group(self) -> None:
        """NOT (a OR b) → NOT(OR(a, b))."""
        ast = parse_dsl('NOT (keyword("a") OR keyword("b"))')
        assert isinstance(ast, NotNode)
        assert isinstance(ast.child, OrNode)

    def test_double_not(self) -> None:
        """NOT NOT a → NOT(NOT(a))."""
        ast = parse_dsl('NOT NOT keyword("a")')
        assert isinstance(ast, NotNode)
        assert isinstance(ast.child, NotNode)
        assert isinstance(ast.child.child, PredicateNode)


# ---------------------------------------------------------------------------
# Parser — complex compositions
# ---------------------------------------------------------------------------


class TestParserCompositions:
    def test_architect_example_1(self) -> None:
        """keyword("text", "billing") AND speaker("customer")."""
        ast = parse_dsl('keyword("text", "billing") AND speaker("customer")')
        assert isinstance(ast, AndNode)
        assert len(ast.children) == 2

    def test_architect_example_2(self) -> None:
        """intent("cancel_subscription") AND NOT keyword("text", "upgrade")."""
        ast = parse_dsl('intent("cancel_subscription") AND NOT keyword("text", "upgrade")')
        assert isinstance(ast, AndNode)
        assert isinstance(ast.children[0], PredicateNode)
        assert isinstance(ast.children[1], NotNode)

    def test_architect_example_3(self) -> None:
        """(keyword("text", "billing") OR keyword("text", "invoice")) AND speaker("customer")."""
        dsl = '(keyword("text", "billing") OR keyword("text", "invoice")) AND speaker("customer")'
        ast = parse_dsl(dsl)
        assert isinstance(ast, AndNode)
        assert isinstance(ast.children[0], OrNode)
        assert isinstance(ast.children[1], PredicateNode)

    def test_three_levels_deep(self) -> None:
        """AND(OR(a, NOT(b)), c)."""
        dsl = '(keyword("a") OR NOT keyword("b")) AND keyword("c")'
        ast = parse_dsl(dsl)
        assert isinstance(ast, AndNode)
        or_node = ast.children[0]
        assert isinstance(or_node, OrNode)
        assert isinstance(or_node.children[1], NotNode)

    def test_mixed_predicate_types(self) -> None:
        """All four predicate families in one rule."""
        dsl = 'keyword("billing") AND intent("cancel") AND speaker("customer") AND repeated("text", "billing", 3)'
        ast = parse_dsl(dsl)
        assert isinstance(ast, AndNode)
        assert len(ast.children) == 4
        types = [child.predicate_type for child in ast.children]  # type: ignore[union-attr]
        assert PredicateType.LEXICAL in types
        assert PredicateType.SEMANTIC in types
        assert PredicateType.STRUCTURAL in types
        assert PredicateType.CONTEXTUAL in types


# ---------------------------------------------------------------------------
# Parser — error handling
# ---------------------------------------------------------------------------


class TestParserErrors:
    def test_empty_expression(self) -> None:
        with pytest.raises(RuleError, match="Empty DSL expression"):
            parse_dsl("")

    def test_whitespace_only(self) -> None:
        with pytest.raises(RuleError, match="Empty DSL expression"):
            parse_dsl("   ")

    def test_unknown_predicate(self) -> None:
        with pytest.raises(RuleError, match="Unknown predicate function 'foobar'"):
            parse_dsl('foobar("x")')

    def test_missing_lparen(self) -> None:
        with pytest.raises(RuleError, match="Expected '\\('"):
            parse_dsl('keyword "billing"')

    def test_missing_rparen(self) -> None:
        with pytest.raises(RuleError, match="Expected '\\)'"):
            parse_dsl('keyword("billing"')

    def test_missing_rparen_group(self) -> None:
        with pytest.raises(RuleError, match="Expected '\\)'"):
            parse_dsl('(keyword("a") AND keyword("b")')

    def test_trailing_tokens(self) -> None:
        with pytest.raises(RuleError, match="Unexpected token"):
            parse_dsl('keyword("a") keyword("b")')

    def test_keyword_wrong_arg_count(self) -> None:
        with pytest.raises(RuleError, match="expects 2 argument"):
            parse_dsl('keyword("a", "b", "c")')

    def test_speaker_wrong_arg_count(self) -> None:
        with pytest.raises(RuleError, match="expects 1 argument"):
            parse_dsl('speaker("a", "b")')

    def test_similarity_wrong_arg_count(self) -> None:
        with pytest.raises(RuleError, match="expects 3 argument"):
            parse_dsl('similarity("a", "b")')

    def test_intent_too_many_args(self) -> None:
        with pytest.raises(RuleError, match="expects 1-2 argument"):
            parse_dsl('intent("a", 0.8, "extra")')

    def test_repeated_wrong_arg_count(self) -> None:
        with pytest.raises(RuleError, match="expects 3 argument"):
            parse_dsl('repeated("a", "b")')

    def test_occurs_after_wrong_arg_count(self) -> None:
        with pytest.raises(RuleError, match="expects 3 argument"):
            parse_dsl('occurs_after("a")')

    def test_unexpected_token_in_atom(self) -> None:
        with pytest.raises(RuleError, match="Expected predicate"):
            parse_dsl("AND")


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestParserReexport:
    def test_parse_dsl_from_rules_package(self) -> None:
        from talkex.rules import parse_dsl as imported

        assert imported is parse_dsl

    def test_predicate_registry_from_rules_package(self) -> None:
        from talkex.rules import PREDICATE_REGISTRY

        assert "keyword" in PREDICATE_REGISTRY
