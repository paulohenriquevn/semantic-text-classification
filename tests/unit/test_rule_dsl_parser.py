"""Unit tests for rule DSL tokenizer and parser.

Tests cover: tokenization, basic predicate parsing, logical operators (AND, OR, NOT),
operator precedence, parenthesized groups, all predicate functions, error handling,
the parse_dsl public API, extended dotted namespace syntax, infix comparisons,
list literals, RULE...WHEN...THEN block syntax, and the parse_rule_block API.
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
from talkex.rules.models import ParsedRuleBlock, RuleAction
from talkex.rules.parser import TokenType, parse_dsl, parse_rule_block, tokenize

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

    def test_parse_rule_block_from_rules_package(self) -> None:
        from talkex.rules import parse_rule_block as imported

        assert imported is parse_rule_block

    def test_namespace_map_from_rules_package(self) -> None:
        from talkex.rules import NAMESPACE_PREDICATE_MAP

        assert ("semantic", "intent") in NAMESPACE_PREDICATE_MAP


# ---------------------------------------------------------------------------
# Extended tokenizer — new token types
# ---------------------------------------------------------------------------


class TestExtendedTokenizer:
    def test_dot_token(self) -> None:
        tokens = tokenize("semantic.intent")
        types = [t.token_type for t in tokens if t.token_type != TokenType.EOF]
        assert types == [TokenType.IDENTIFIER, TokenType.DOT, TokenType.IDENTIFIER]

    def test_comparison_operators(self) -> None:
        tokens = tokenize('speaker == "customer"')
        types = [t.token_type for t in tokens if t.token_type != TokenType.EOF]
        assert types == [TokenType.IDENTIFIER, TokenType.EQ_OP, TokenType.STRING]

    def test_gt_gte_tokens(self) -> None:
        tokens = tokenize("> >=")
        types = [t.token_type for t in tokens if t.token_type != TokenType.EOF]
        assert types == [TokenType.GT, TokenType.GTE]

    def test_lt_lte_tokens(self) -> None:
        tokens = tokenize("< <=")
        types = [t.token_type for t in tokens if t.token_type != TokenType.EOF]
        assert types == [TokenType.LT, TokenType.LTE]

    def test_neq_token(self) -> None:
        tokens = tokenize("!=")
        assert tokens[0].token_type == TokenType.NEQ

    def test_assign_token(self) -> None:
        tokens = tokenize("intent=")
        types = [t.token_type for t in tokens if t.token_type != TokenType.EOF]
        assert types == [TokenType.IDENTIFIER, TokenType.ASSIGN]

    def test_brackets(self) -> None:
        tokens = tokenize('["a", "b"]')
        types = [t.token_type for t in tokens if t.token_type != TokenType.EOF]
        assert types == [
            TokenType.LBRACKET,
            TokenType.STRING,
            TokenType.COMMA,
            TokenType.STRING,
            TokenType.RBRACKET,
        ]

    def test_rule_when_then_keywords(self) -> None:
        tokens = tokenize("RULE test WHEN x THEN y")
        types = [t.token_type for t in tokens if t.token_type != TokenType.EOF]
        assert types == [
            TokenType.RULE,
            TokenType.IDENTIFIER,
            TokenType.WHEN,
            TokenType.IDENTIFIER,
            TokenType.THEN,
            TokenType.IDENTIFIER,
        ]

    def test_rule_when_then_case_insensitive(self) -> None:
        tokens = tokenize("rule x when y then z")
        types = [t.token_type for t in tokens if t.token_type != TokenType.EOF]
        assert types == [
            TokenType.RULE,
            TokenType.IDENTIFIER,
            TokenType.WHEN,
            TokenType.IDENTIFIER,
            TokenType.THEN,
            TokenType.IDENTIFIER,
        ]


# ---------------------------------------------------------------------------
# Infix comparison syntax
# ---------------------------------------------------------------------------


class TestInfixComparison:
    def test_speaker_eq(self) -> None:
        ast = parse_dsl('speaker == "customer"')
        assert isinstance(ast, PredicateNode)
        assert ast.predicate_type == PredicateType.STRUCTURAL
        assert ast.field_name == "speaker_role"
        assert ast.operator == "eq"
        assert ast.value == "customer"

    def test_channel_eq(self) -> None:
        ast = parse_dsl('channel == "voice"')
        assert isinstance(ast, PredicateNode)
        assert ast.field_name == "channel"
        assert ast.operator == "eq"
        assert ast.value == "voice"

    def test_speaker_neq(self) -> None:
        ast = parse_dsl('speaker != "bot"')
        assert isinstance(ast, PredicateNode)
        assert ast.operator == "neq"

    def test_unknown_infix_field_raises(self) -> None:
        with pytest.raises(RuleError, match="Unknown field 'foobar'"):
            parse_dsl('foobar == "x"')

    def test_infix_in_and_expression(self) -> None:
        ast = parse_dsl('speaker == "customer" AND keyword("billing")')
        assert isinstance(ast, AndNode)
        assert isinstance(ast.children[0], PredicateNode)
        assert ast.children[0].operator == "eq"
        assert isinstance(ast.children[1], PredicateNode)
        assert ast.children[1].operator == "contains"


# ---------------------------------------------------------------------------
# Dotted namespace syntax
# ---------------------------------------------------------------------------


class TestDottedNamespace:
    def test_semantic_intent_with_threshold(self) -> None:
        ast = parse_dsl('semantic.intent("cancelamento") > 0.82')
        assert isinstance(ast, PredicateNode)
        assert ast.predicate_type == PredicateType.SEMANTIC
        assert ast.field_name == "intent_score"
        assert ast.value == "cancelamento"
        assert ast.threshold == 0.82

    def test_semantic_intent_gte(self) -> None:
        ast = parse_dsl('semantic.intent("billing") >= 0.7')
        assert isinstance(ast, PredicateNode)
        assert ast.operator == "gte"
        assert ast.threshold == 0.7

    def test_semantic_intent_without_comparison(self) -> None:
        ast = parse_dsl('semantic.intent("cancel")')
        assert isinstance(ast, PredicateNode)
        assert ast.predicate_type == PredicateType.SEMANTIC
        assert ast.value == "cancel"
        assert ast.threshold is None

    def test_semantic_similarity(self) -> None:
        ast = parse_dsl('semantic.similarity("quero cancelar meu serviço") > 0.86')
        assert isinstance(ast, PredicateNode)
        assert ast.predicate_type == PredicateType.SEMANTIC
        assert ast.field_name == "embedding_similarity"
        assert ast.operator == "gt"
        assert ast.value == "quero cancelar meu serviço"
        assert ast.threshold == 0.86

    def test_semantic_similarity_respects_comparison_operators(self) -> None:
        """Similarity predicates should use the actual comparison operator."""
        # Less-than
        ast = parse_dsl('semantic.similarity("saudações") < 0.79')
        assert isinstance(ast, PredicateNode)
        assert ast.operator == "lt"
        assert ast.threshold == 0.79

        # Greater-equal
        ast = parse_dsl('semantic.similarity("reclamação") >= 0.90')
        assert isinstance(ast, PredicateNode)
        assert ast.operator == "gte"
        assert ast.threshold == 0.90

        # Less-equal
        ast = parse_dsl('semantic.similarity("elogio") <= 0.5')
        assert isinstance(ast, PredicateNode)
        assert ast.operator == "lte"
        assert ast.threshold == 0.5

    def test_semantic_similarity_no_operator_defaults_to_similarity_above(self) -> None:
        """Without explicit operator, similarity defaults to similarity_above (>=)."""
        ast = parse_dsl('semantic.similarity("teste")')
        assert isinstance(ast, PredicateNode)
        assert ast.operator == "similarity_above"
        assert ast.threshold is None

    def test_lexical_contains(self) -> None:
        ast = parse_dsl('lexical.contains("billing")')
        assert isinstance(ast, PredicateNode)
        assert ast.predicate_type == PredicateType.LEXICAL
        assert ast.operator == "contains"
        assert ast.value == "billing"

    def test_lexical_contains_any_with_list(self) -> None:
        ast = parse_dsl('lexical.contains_any(["cancelar", "encerrar", "desistir"])')
        assert isinstance(ast, PredicateNode)
        assert ast.predicate_type == PredicateType.LEXICAL
        assert ast.operator == "contains_any"
        assert ast.value == ["cancelar", "encerrar", "desistir"]

    def test_lexical_regex(self) -> None:
        ast = parse_dsl('lexical.regex("cancel|terminate")')
        assert isinstance(ast, PredicateNode)
        assert ast.operator == "regex"
        assert ast.value == "cancel|terminate"

    def test_unknown_namespace_method_raises(self) -> None:
        with pytest.raises(RuleError, match="Unknown predicate"):
            parse_dsl('foobar.baz("x")')

    def test_dotted_in_and_or(self) -> None:
        dsl = 'semantic.intent("cancel") > 0.8 AND lexical.contains("billing")'
        ast = parse_dsl(dsl)
        assert isinstance(ast, AndNode)
        assert len(ast.children) == 2


# ---------------------------------------------------------------------------
# Context namespace with method chaining
# ---------------------------------------------------------------------------


class TestContextChaining:
    def test_turn_window_count(self) -> None:
        ast = parse_dsl('context.turn_window(5).count(intent="insatisfacao") >= 2')
        assert isinstance(ast, PredicateNode)
        assert ast.predicate_type == PredicateType.CONTEXTUAL
        assert ast.field_name == "window_count"
        assert ast.operator == "gte"
        assert ast.value == 2
        assert ast.metadata["window_size"] == 5
        assert ast.metadata["intent"] == "insatisfacao"

    def test_turn_window_count_gt(self) -> None:
        ast = parse_dsl('context.turn_window(3).count(intent="reclamacao") > 1')
        assert isinstance(ast, PredicateNode)
        assert ast.operator == "gt"
        assert ast.value == 1
        assert ast.metadata["window_size"] == 3

    def test_turn_window_without_chain_raises(self) -> None:
        with pytest.raises(RuleError, match="requires a chained method"):
            parse_dsl("context.turn_window(5)")

    def test_unknown_context_method_raises(self) -> None:
        with pytest.raises(RuleError, match="Unknown context method"):
            parse_dsl("context.foobar(5)")


# ---------------------------------------------------------------------------
# contains_any predicate (inline function-call syntax)
# ---------------------------------------------------------------------------


class TestContainsAny:
    def test_inline_contains_any(self) -> None:
        ast = parse_dsl('contains_any("cancelar", "encerrar")')
        assert isinstance(ast, PredicateNode)
        assert ast.operator == "contains_any"
        assert ast.value == ["cancelar", "encerrar"]
        assert ast.predicate_type == PredicateType.LEXICAL

    def test_contains_any_single_word(self) -> None:
        ast = parse_dsl('contains_any("billing")')
        assert isinstance(ast, PredicateNode)
        assert ast.value == ["billing"]

    def test_contains_any_no_args_raises(self) -> None:
        with pytest.raises(RuleError, match="expects at least 1"):
            parse_dsl("contains_any()")


# ---------------------------------------------------------------------------
# RULE...WHEN...THEN block syntax
# ---------------------------------------------------------------------------


class TestRuleBlock:
    def test_simple_rule_block(self) -> None:
        dsl = """
        RULE billing_detection
        WHEN keyword("billing") AND speaker("customer")
        THEN tag("billing")
        """
        block = parse_rule_block(dsl)
        assert isinstance(block, ParsedRuleBlock)
        assert block.rule_name == "billing_detection"
        assert isinstance(block.ast, AndNode)
        assert len(block.actions) == 1
        assert block.actions[0].action_type == "tag"
        assert block.actions[0].value == "billing"

    def test_rule_block_multiple_actions(self) -> None:
        dsl = """
        RULE test_rule
        WHEN keyword("x")
        THEN tag("label") score(0.95) priority("high")
        """
        block = parse_rule_block(dsl)
        assert len(block.actions) == 3
        assert block.actions[0] == RuleAction(action_type="tag", value="label")
        assert block.actions[1] == RuleAction(action_type="score", value=0.95)
        assert block.actions[2] == RuleAction(action_type="priority", value="high")

    def test_rule_block_with_dotted_syntax(self) -> None:
        dsl = """
        RULE risco_cancelamento
        WHEN
            speaker == "customer"
            AND semantic.intent("cancelamento") > 0.82
            AND lexical.contains_any(["cancelar", "encerrar", "desistir"])
        THEN
            tag("cancelamento_risco") score(0.95) priority("high")
        """
        block = parse_rule_block(dsl)
        assert block.rule_name == "risco_cancelamento"
        assert isinstance(block.ast, AndNode)
        assert len(block.ast.children) == 3

        # First child: speaker == "customer"
        speaker = block.ast.children[0]
        assert isinstance(speaker, PredicateNode)
        assert speaker.operator == "eq"
        assert speaker.value == "customer"

        # Second child: semantic.intent(...)
        intent = block.ast.children[1]
        assert isinstance(intent, PredicateNode)
        assert intent.predicate_type == PredicateType.SEMANTIC
        assert intent.threshold == 0.82

        # Third child: lexical.contains_any(...)
        contains = block.ast.children[2]
        assert isinstance(contains, PredicateNode)
        assert contains.operator == "contains_any"
        assert contains.value == ["cancelar", "encerrar", "desistir"]

        # Actions
        assert len(block.actions) == 3

    def test_users_full_example(self) -> None:
        """Test the exact DSL from the user's request."""
        dsl = """
        RULE risco_cancelamento_alto
        WHEN
            speaker == "customer"
            AND semantic.intent("cancelamento") > 0.82
            AND (
                lexical.contains_any(["cancelar", "encerrar", "desistir"])
                OR semantic.similarity("quero cancelar meu serviço") > 0.86
            )
            AND context.turn_window(5).count(intent="insatisfacao") >= 2
        THEN
            tag("cancelamento_risco_alto")
            score(0.95)
            priority("high")
        """
        block = parse_rule_block(dsl)
        assert block.rule_name == "risco_cancelamento_alto"
        assert isinstance(block.ast, AndNode)
        assert len(block.ast.children) == 4
        assert len(block.actions) == 3

        # Verify nested OR group
        or_group = block.ast.children[2]
        assert isinstance(or_group, OrNode)
        assert len(or_group.children) == 2

    def test_rule_block_without_then(self) -> None:
        dsl = 'RULE simple WHEN keyword("test")'
        block = parse_rule_block(dsl)
        assert block.rule_name == "simple"
        assert isinstance(block.ast, PredicateNode)
        assert block.actions == []

    def test_parse_dsl_detects_block_syntax(self) -> None:
        """parse_dsl should extract AST from RULE block."""
        dsl = 'RULE test WHEN keyword("billing") THEN tag("x")'
        ast = parse_dsl(dsl)
        assert isinstance(ast, PredicateNode)
        assert ast.value == "billing"

    def test_empty_rule_block_raises(self) -> None:
        with pytest.raises(RuleError, match="Empty rule block"):
            parse_rule_block("")

    def test_non_rule_block_raises(self) -> None:
        with pytest.raises(RuleError, match="must start with 'RULE'"):
            parse_rule_block('keyword("x")')

    def test_unknown_action_raises(self) -> None:
        with pytest.raises(RuleError, match="Unknown action 'foobar'"):
            parse_rule_block('RULE test WHEN keyword("x") THEN foobar("y")')


# ---------------------------------------------------------------------------
# Compiler integration with RULE blocks
# ---------------------------------------------------------------------------


class TestCompilerRuleBlock:
    def test_compiler_handles_rule_block(self) -> None:
        from talkex.rules.compiler import SimpleRuleCompiler

        compiler = SimpleRuleCompiler()
        dsl = """
        RULE billing_detection
        WHEN keyword("billing") AND speaker("customer")
        THEN tag("billing_issue") score(0.9) priority("high")
        """
        rule = compiler.compile(dsl, "r1", "r1")
        assert rule.rule_name == "billing_detection"
        assert "billing_issue" in rule.tags
        assert rule.priority == 10  # "high" maps to 10
        assert rule.metadata.get("score_override") == 0.9

    def test_compiler_preserves_inline_syntax(self) -> None:
        from talkex.rules.compiler import SimpleRuleCompiler

        compiler = SimpleRuleCompiler()
        rule = compiler.compile('keyword("billing")', "r1", "billing_check")
        assert rule.rule_name == "billing_check"
        assert isinstance(rule.ast, PredicateNode)


# ---------------------------------------------------------------------------
# Evaluator integration with contains_any
# ---------------------------------------------------------------------------


class TestEvaluatorContainsAny:
    def test_contains_any_matches(self) -> None:
        from talkex.rules.config import RuleEngineConfig
        from talkex.rules.evaluator import SimpleRuleEvaluator
        from talkex.rules.models import RuleDefinition, RuleEvaluationInput

        ast = parse_dsl('contains_any("cancelar", "encerrar", "desistir")')
        rule = RuleDefinition(
            rule_id="r1",
            rule_name="test",
            rule_version="1.0",
            description="test",
            ast=ast,
        )
        eval_input = RuleEvaluationInput(
            source_id="w1",
            source_type="context_window",
            text="eu quero cancelar minha conta agora",
        )
        evaluator = SimpleRuleEvaluator()
        results = evaluator.evaluate([rule], eval_input, RuleEngineConfig())
        assert results[0].matched is True
        assert results[0].predicate_results[0].matched_text == "cancelar"

    def test_contains_any_no_match(self) -> None:
        from talkex.rules.config import RuleEngineConfig
        from talkex.rules.evaluator import SimpleRuleEvaluator
        from talkex.rules.models import RuleDefinition, RuleEvaluationInput

        ast = parse_dsl('contains_any("cancelar", "encerrar")')
        rule = RuleDefinition(
            rule_id="r1",
            rule_name="test",
            rule_version="1.0",
            description="test",
            ast=ast,
        )
        eval_input = RuleEvaluationInput(
            source_id="w1",
            source_type="context_window",
            text="obrigado pela ajuda",
        )
        evaluator = SimpleRuleEvaluator()
        results = evaluator.evaluate([rule], eval_input, RuleEngineConfig())
        assert results[0].matched is False

    def test_contains_any_multiple_matches(self) -> None:
        from talkex.rules.config import RuleEngineConfig
        from talkex.rules.evaluator import SimpleRuleEvaluator
        from talkex.rules.models import RuleDefinition, RuleEvaluationInput

        ast = parse_dsl('contains_any("cancelar", "encerrar", "desistir")')
        rule = RuleDefinition(
            rule_id="r1",
            rule_name="test",
            rule_version="1.0",
            description="test",
            ast=ast,
        )
        eval_input = RuleEvaluationInput(
            source_id="w1",
            source_type="context_window",
            text="quero cancelar e desistir de tudo",
        )
        evaluator = SimpleRuleEvaluator()
        results = evaluator.evaluate([rule], eval_input, RuleEngineConfig())
        assert results[0].matched is True
        # Score should reflect proportion of matched words
        pr = results[0].predicate_results[0]
        assert pr.score == pytest.approx(2.0 / 3.0)
        assert pr.metadata["matched_words"] == ["cancelar", "desistir"]
