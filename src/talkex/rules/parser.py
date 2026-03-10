"""Rule DSL parser — recursive-descent parser producing typed AST nodes.

Transforms DSL text into an AST tree using a two-phase process:

    1. Tokenization: DSL text → list of tokens
    2. Parsing: token stream → ASTNode tree

The parser is a recursive-descent parser implementing the grammar defined
in ``dsl.py``. It delegates semantic validation to the ``compiler`` module.

Supports two syntax modes:

    1. **Inline** (original): ``keyword("billing") AND speaker("customer")``
    2. **Block** (extended): ``RULE name WHEN expr THEN actions``

Token types:
    AND, OR, NOT     — logical operators (case-insensitive)
    RULE, WHEN, THEN — block rule keywords (case-insensitive)
    LPAREN, RPAREN   — grouping parentheses
    LBRACKET, RBRACKET — list literal brackets
    COMMA            — argument separator
    DOT              — dotted namespace separator
    EQ_OP            — equality comparison (==)
    NEQ              — not-equal comparison (!=)
    GT, GTE          — greater-than comparisons (>, >=)
    LT, LTE          — less-than comparisons (<, <=)
    ASSIGN           — keyword argument assignment (=)
    STRING           — quoted string literal (single or double quotes)
    NUMBER           — integer or float literal
    IDENTIFIER       — function name or bare word
    EOF              — end of input
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from talkex.exceptions import RuleError
from talkex.rules.ast import (
    AndNode,
    ASTNode,
    NotNode,
    OrNode,
    PredicateNode,
)
from talkex.rules.config import PredicateType
from talkex.rules.dsl import (
    COMPARISON_OP_MAP,
    INFIX_FIELD_MAP,
    NAMESPACE_PREDICATE_MAP,
    PREDICATE_REGISTRY,
)
from talkex.rules.models import ParsedRuleBlock, RuleAction

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class TokenType(StrEnum):
    """Token types produced by the DSL tokenizer."""

    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    RULE = "RULE"
    WHEN = "WHEN"
    THEN = "THEN"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    LBRACKET = "LBRACKET"
    RBRACKET = "RBRACKET"
    COMMA = "COMMA"
    DOT = "DOT"
    EQ_OP = "EQ_OP"
    NEQ = "NEQ"
    GTE = "GTE"
    GT = "GT"
    LTE = "LTE"
    LT = "LT"
    ASSIGN = "ASSIGN"
    STRING = "STRING"
    NUMBER = "NUMBER"
    IDENTIFIER = "IDENTIFIER"
    EOF = "EOF"


# Comparison token types for quick membership checks
_COMPARISON_TOKENS = frozenset(
    {
        TokenType.EQ_OP,
        TokenType.NEQ,
        TokenType.GT,
        TokenType.GTE,
        TokenType.LT,
        TokenType.LTE,
    }
)

# Map token types to operator strings
_TOKEN_TO_OP: dict[TokenType, str] = {
    TokenType.EQ_OP: "==",
    TokenType.NEQ: "!=",
    TokenType.GT: ">",
    TokenType.GTE: ">=",
    TokenType.LT: "<",
    TokenType.LTE: "<=",
}


@dataclass(frozen=True)
class Token:
    """A single token from the DSL tokenizer.

    Args:
        token_type: The type of this token.
        value: The literal value (string content, number, identifier name).
        position: Character offset in the original DSL text.
    """

    token_type: TokenType
    value: str | float | int
    position: int


def tokenize(dsl_text: str) -> list[Token]:
    """Tokenize DSL text into a list of tokens.

    Args:
        dsl_text: The DSL rule text.

    Returns:
        Ordered list of tokens ending with EOF.

    Raises:
        RuleError: If the text contains invalid characters or malformed strings.
    """
    tokens: list[Token] = []
    i = 0
    length = len(dsl_text)

    while i < length:
        ch = dsl_text[i]

        # Skip whitespace
        if ch in " \t\n\r":
            i += 1
            continue

        # Two-character operators (must check before single-char)
        if ch == "=" and i + 1 < length and dsl_text[i + 1] == "=":
            tokens.append(Token(TokenType.EQ_OP, "==", i))
            i += 2
            continue
        if ch == "!" and i + 1 < length and dsl_text[i + 1] == "=":
            tokens.append(Token(TokenType.NEQ, "!=", i))
            i += 2
            continue
        if ch == ">" and i + 1 < length and dsl_text[i + 1] == "=":
            tokens.append(Token(TokenType.GTE, ">=", i))
            i += 2
            continue
        if ch == "<" and i + 1 < length and dsl_text[i + 1] == "=":
            tokens.append(Token(TokenType.LTE, "<=", i))
            i += 2
            continue

        # Single-character operators
        if ch == ">":
            tokens.append(Token(TokenType.GT, ">", i))
            i += 1
            continue
        if ch == "<":
            tokens.append(Token(TokenType.LT, "<", i))
            i += 1
            continue
        if ch == "=":
            tokens.append(Token(TokenType.ASSIGN, "=", i))
            i += 1
            continue

        # Parentheses, brackets, comma, dot
        if ch == "(":
            tokens.append(Token(TokenType.LPAREN, "(", i))
            i += 1
            continue
        if ch == ")":
            tokens.append(Token(TokenType.RPAREN, ")", i))
            i += 1
            continue
        if ch == "[":
            tokens.append(Token(TokenType.LBRACKET, "[", i))
            i += 1
            continue
        if ch == "]":
            tokens.append(Token(TokenType.RBRACKET, "]", i))
            i += 1
            continue
        if ch == ",":
            tokens.append(Token(TokenType.COMMA, ",", i))
            i += 1
            continue
        if ch == ".":
            tokens.append(Token(TokenType.DOT, ".", i))
            i += 1
            continue

        # String literals (single or double quotes)
        if ch in ('"', "'"):
            start = i
            quote_char = ch
            i += 1
            string_chars: list[str] = []
            while i < length and dsl_text[i] != quote_char:
                if dsl_text[i] == "\\" and i + 1 < length:
                    # Escaped character
                    i += 1
                    string_chars.append(dsl_text[i])
                else:
                    string_chars.append(dsl_text[i])
                i += 1
            if i >= length:
                raise RuleError(f"Unterminated string starting at position {start}")
            i += 1  # consume closing quote
            tokens.append(Token(TokenType.STRING, "".join(string_chars), start))
            continue

        # Numbers (integers and floats)
        if ch.isdigit() or (ch == "-" and i + 1 < length and dsl_text[i + 1].isdigit()):
            start = i
            if ch == "-":
                i += 1
            while i < length and (dsl_text[i].isdigit() or dsl_text[i] == "."):
                i += 1
            num_str = dsl_text[start:i]
            if "." in num_str:
                tokens.append(Token(TokenType.NUMBER, float(num_str), start))
            else:
                tokens.append(Token(TokenType.NUMBER, int(num_str), start))
            continue

        # Identifiers and keywords (AND, OR, NOT, RULE, WHEN, THEN)
        if ch.isalpha() or ch == "_":
            start = i
            while i < length and (dsl_text[i].isalnum() or dsl_text[i] == "_"):
                i += 1
            word = dsl_text[start:i]
            upper = word.upper()
            if upper == "AND":
                tokens.append(Token(TokenType.AND, "AND", start))
            elif upper == "OR":
                tokens.append(Token(TokenType.OR, "OR", start))
            elif upper == "NOT":
                tokens.append(Token(TokenType.NOT, "NOT", start))
            elif upper == "RULE":
                tokens.append(Token(TokenType.RULE, "RULE", start))
            elif upper == "WHEN":
                tokens.append(Token(TokenType.WHEN, "WHEN", start))
            elif upper == "THEN":
                tokens.append(Token(TokenType.THEN, "THEN", start))
            else:
                tokens.append(Token(TokenType.IDENTIFIER, word, start))
            continue

        raise RuleError(f"Unexpected character '{ch}' at position {i}")

    tokens.append(Token(TokenType.EOF, "", length))
    return tokens


# ---------------------------------------------------------------------------
# Recursive-descent parser
# ---------------------------------------------------------------------------


class _Parser:
    """Recursive-descent parser for the rule DSL.

    Produces an ASTNode tree from a token stream. The parser handles
    operator precedence via recursive descent:

        lowest  → OR  (evaluated last, binds least tightly)
        middle  → AND
        highest → NOT, atoms (predicates, parenthesized groups)

    Extended atoms handle dotted namespace predicates (``semantic.intent()``)
    and infix comparisons (``speaker == "customer"``).
    """

    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    def parse(self) -> ASTNode:
        """Parse the token stream into an ASTNode.

        Returns:
            The root ASTNode of the parsed expression.

        Raises:
            RuleError: If the token stream contains syntax errors.
        """
        node = self._or_expr()
        if self._current().token_type != TokenType.EOF:
            tok = self._current()
            raise RuleError(f"Unexpected token '{tok.value}' at position {tok.position}, expected end of expression")
        return node

    def parse_until_then(self) -> ASTNode:
        """Parse expression tokens until THEN keyword or EOF.

        Used by rule block parsing to extract the WHEN clause.

        Returns:
            The root ASTNode of the WHEN expression.
        """
        node = self._or_expr()
        return node

    # -- Recursive descent levels --

    def _or_expr(self) -> ASTNode:
        """Parse: or_expr = and_expr ( "OR" and_expr )*"""
        left = self._and_expr()
        children = [left]
        while self._current().token_type == TokenType.OR:
            self._advance()  # consume OR
            children.append(self._and_expr())
        if len(children) == 1:
            return children[0]
        return OrNode(children=children)

    def _and_expr(self) -> ASTNode:
        """Parse: and_expr = not_expr ( "AND" not_expr )*"""
        left = self._not_expr()
        children = [left]
        while self._current().token_type == TokenType.AND:
            self._advance()  # consume AND
            children.append(self._not_expr())
        if len(children) == 1:
            return children[0]
        return AndNode(children=children)

    def _not_expr(self) -> ASTNode:
        """Parse: not_expr = "NOT" not_expr | atom"""
        if self._current().token_type == TokenType.NOT:
            self._advance()  # consume NOT
            child = self._not_expr()
            return NotNode(child=child)
        return self._atom()

    def _atom(self) -> ASTNode:
        """Parse: atom = dotted_expr | infix_comparison | predicate | "(" expression ")"

        Extended to handle:
        - Dotted predicates: ``semantic.intent("x") > 0.82``
        - Infix comparisons: ``speaker == "customer"``
        - Legacy function calls: ``keyword("billing")``
        - Parenthesized groups: ``(expr)``
        """
        tok = self._current()

        # Parenthesized group
        if tok.token_type == TokenType.LPAREN:
            self._advance()  # consume (
            node = self._or_expr()
            if self._current().token_type != TokenType.RPAREN:
                raise RuleError(f"Expected ')' at position {self._current().position}, got '{self._current().value}'")
            self._advance()  # consume )
            return node

        # Identifier-led expressions (predicates, dotted, infix)
        if tok.token_type == TokenType.IDENTIFIER:
            return self._identifier_expr()

        raise RuleError(f"Expected predicate or '(' at position {tok.position}, got '{tok.value}' ({tok.token_type})")

    def _identifier_expr(self) -> PredicateNode:
        """Parse an identifier-led expression.

        Dispatches based on what follows the identifier:
        - DOT → dotted expression (e.g., ``semantic.intent(...)``)
        - comparison operator → infix comparison (e.g., ``speaker == "customer"``)
        - LPAREN → function call predicate (e.g., ``keyword("billing")``)
        """
        tok = self._current()
        next_tok = self._peek()

        # Dotted namespace: semantic.intent(...), context.turn_window(...)
        if next_tok is not None and next_tok.token_type == TokenType.DOT:
            return self._dotted_expression()

        # Infix comparison: speaker == "customer"
        if next_tok is not None and next_tok.token_type in _COMPARISON_TOKENS:
            return self._infix_comparison()

        # Legacy function call: keyword("billing")
        if next_tok is not None and next_tok.token_type == TokenType.LPAREN:
            return self._predicate()

        raise RuleError(f"Expected '(', '.', or comparison operator after '{tok.value}' at position {tok.position}")

    def _dotted_expression(self) -> PredicateNode:
        """Parse a dotted namespace expression.

        Handles:
        - ``semantic.intent("cancelamento") > 0.82``
        - ``lexical.contains_any(["cancelar", "encerrar"])``
        - ``context.turn_window(5).count(intent="insatisfacao") >= 2``
        """
        namespace_tok = self._current()
        namespace = str(namespace_tok.value)
        self._advance()  # consume namespace identifier

        self._expect(TokenType.DOT, "Expected '.' after namespace")

        # Parse first method call
        method_tok = self._current()
        if method_tok.token_type != TokenType.IDENTIFIER:
            raise RuleError(f"Expected method name after '{namespace}.' at position {method_tok.position}")
        method = str(method_tok.value)
        self._advance()  # consume method name

        self._expect(TokenType.LPAREN, f"Expected '(' after '{namespace}.{method}'")
        args, kwargs = self._extended_arguments()
        self._expect(TokenType.RPAREN, f"Expected ')' in '{namespace}.{method}(...)'")

        # Handle method chaining: context.turn_window(5).count(...)
        chains: list[tuple[str, list[str | int | float | list[str]], dict[str, str | int | float]]] = []
        while self._current().token_type == TokenType.DOT:
            self._advance()  # consume DOT
            chain_tok = self._current()
            if chain_tok.token_type != TokenType.IDENTIFIER:
                raise RuleError(f"Expected method name after '.' at position {chain_tok.position}")
            chain_method = str(chain_tok.value)
            self._advance()

            self._expect(TokenType.LPAREN, f"Expected '(' after '.{chain_method}'")
            chain_args, chain_kwargs = self._extended_arguments()
            self._expect(TokenType.RPAREN, f"Expected ')' in '.{chain_method}(...)'")
            chains.append((chain_method, chain_args, chain_kwargs))

        # Check for trailing comparison operator
        comparison_op: str | None = None
        comparison_value: str | int | float | None = None
        if self._current().token_type in _COMPARISON_TOKENS:
            op_tok = self._advance()
            comparison_op = _TOKEN_TO_OP[op_tok.token_type]
            val_tok = self._current()
            if val_tok.token_type not in (TokenType.STRING, TokenType.NUMBER):
                raise RuleError(f"Expected value after '{comparison_op}' at position {val_tok.position}")
            comparison_value = val_tok.value
            self._advance()

        return _build_dotted_predicate(
            namespace,
            method,
            args,
            kwargs,
            chains,
            comparison_op,
            comparison_value,
            namespace_tok.position,
        )

    def _infix_comparison(self) -> PredicateNode:
        """Parse an infix comparison: ``speaker == "customer"``.

        Maps bare field names to structural predicates via INFIX_FIELD_MAP.
        """
        field_tok = self._current()
        field_name = str(field_tok.value)
        self._advance()  # consume field name

        op_tok = self._advance()  # consume comparison operator
        op_str = _TOKEN_TO_OP[op_tok.token_type]

        val_tok = self._current()
        if val_tok.token_type not in (TokenType.STRING, TokenType.NUMBER):
            raise RuleError(f"Expected value after '{op_str}' at position {val_tok.position}")
        value = val_tok.value
        self._advance()

        return _build_infix_predicate(field_name, op_str, value, field_tok.position)

    def _predicate(self) -> PredicateNode:
        """Parse a predicate function call: name(arg1, arg2, ...)."""
        name_tok = self._current()
        func_name = str(name_tok.value)
        self._advance()  # consume function name

        # Expect opening parenthesis
        if self._current().token_type != TokenType.LPAREN:
            raise RuleError(f"Expected '(' after '{func_name}' at position {self._current().position}")
        self._advance()  # consume (

        # Parse arguments
        args: list[str | int | float] = []
        if self._current().token_type != TokenType.RPAREN:
            args.append(self._argument())
            while self._current().token_type == TokenType.COMMA:
                self._advance()  # consume ,
                args.append(self._argument())

        # Expect closing parenthesis
        if self._current().token_type != TokenType.RPAREN:
            raise RuleError(f"Expected ')' at position {self._current().position}, got '{self._current().value}'")
        self._advance()  # consume )

        return _build_predicate_node(func_name, args, name_tok.position)

    def _argument(self) -> str | int | float:
        """Parse a single predicate argument (string, number, or identifier)."""
        tok = self._current()
        if tok.token_type in (TokenType.STRING, TokenType.NUMBER):
            self._advance()
            return tok.value
        if tok.token_type == TokenType.IDENTIFIER:
            self._advance()
            return str(tok.value)
        raise RuleError(f"Expected argument at position {tok.position}, got '{tok.value}' ({tok.token_type})")

    def _extended_arguments(
        self,
    ) -> tuple[list[str | int | float | list[str]], dict[str, str | int | float]]:
        """Parse extended arguments supporting lists and keyword args.

        Returns:
            Tuple of (positional_args, keyword_args).
        """
        args: list[str | int | float | list[str]] = []
        kwargs: dict[str, str | int | float] = {}

        if self._current().token_type == TokenType.RPAREN:
            return args, kwargs

        self._parse_one_extended_arg(args, kwargs)
        while self._current().token_type == TokenType.COMMA:
            self._advance()  # consume ,
            self._parse_one_extended_arg(args, kwargs)

        return args, kwargs

    def _parse_one_extended_arg(
        self,
        args: list[str | int | float | list[str]],
        kwargs: dict[str, str | int | float],
    ) -> None:
        """Parse one extended argument (positional, keyword, or list)."""
        tok = self._current()

        # List literal: ["a", "b", "c"]
        if tok.token_type == TokenType.LBRACKET:
            args.append(self._list_literal())
            return

        # Keyword argument: intent="cancelamento"
        if tok.token_type == TokenType.IDENTIFIER:
            next_tok = self._peek()
            if next_tok is not None and next_tok.token_type == TokenType.ASSIGN:
                key = str(tok.value)
                self._advance()  # consume identifier
                self._advance()  # consume =
                val_tok = self._current()
                if val_tok.token_type in (TokenType.STRING, TokenType.NUMBER):
                    kwargs[key] = val_tok.value
                    self._advance()
                    return
                raise RuleError(f"Expected value after '{key}=' at position {val_tok.position}")

        # Regular argument: string, number, identifier
        if tok.token_type in (TokenType.STRING, TokenType.NUMBER):
            self._advance()
            args.append(tok.value)
            return
        if tok.token_type == TokenType.IDENTIFIER:
            self._advance()
            args.append(str(tok.value))
            return

        raise RuleError(f"Expected argument at position {tok.position}, got '{tok.value}' ({tok.token_type})")

    def _list_literal(self) -> list[str]:
        """Parse a list literal: ["a", "b", "c"]."""
        self._advance()  # consume [
        items: list[str] = []

        if self._current().token_type != TokenType.RBRACKET:
            tok = self._current()
            if tok.token_type not in (TokenType.STRING, TokenType.NUMBER):
                raise RuleError(f"Expected string or number in list at position {tok.position}")
            items.append(str(tok.value))
            self._advance()

            while self._current().token_type == TokenType.COMMA:
                self._advance()  # consume ,
                tok = self._current()
                if tok.token_type not in (TokenType.STRING, TokenType.NUMBER):
                    raise RuleError(f"Expected string or number in list at position {tok.position}")
                items.append(str(tok.value))
                self._advance()

        self._expect(TokenType.RBRACKET, "Expected ']' to close list literal")
        return items

    def parse_actions(self) -> list[tuple[str, list[str | int | float]]]:
        """Parse THEN action calls until EOF.

        Actions are function-call syntax: tag("x"), score(0.95), priority("high").

        Returns:
            List of (action_name, args) tuples.
        """
        actions: list[tuple[str, list[str | int | float]]] = []

        while self._current().token_type == TokenType.IDENTIFIER:
            name = str(self._current().value)
            self._advance()

            self._expect(TokenType.LPAREN, f"Expected '(' after action '{name}'")
            args: list[str | int | float] = []
            if self._current().token_type != TokenType.RPAREN:
                args.append(self._argument())
                while self._current().token_type == TokenType.COMMA:
                    self._advance()
                    args.append(self._argument())
            self._expect(TokenType.RPAREN, f"Expected ')' in action '{name}(...)'")

            actions.append((name, args))

        return actions

    # -- Token stream helpers --

    def _current(self) -> Token:
        """Return the current token without consuming it."""
        return self._tokens[self._pos]

    def _peek(self) -> Token | None:
        """Look ahead one token without consuming."""
        if self._pos + 1 < len(self._tokens):
            return self._tokens[self._pos + 1]
        return None

    def _advance(self) -> Token:
        """Consume the current token and return it."""
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, expected_type: TokenType, error_msg: str) -> Token:
        """Consume a token, raising if it's not the expected type."""
        tok = self._current()
        if tok.token_type != expected_type:
            raise RuleError(f"{error_msg}, got '{tok.value}' at position {tok.position}")
        return self._advance()


# ---------------------------------------------------------------------------
# Predicate node construction — legacy function-call syntax
# ---------------------------------------------------------------------------


def _build_predicate_node(func_name: str, args: list[str | int | float], position: int) -> PredicateNode:
    """Build a PredicateNode from a parsed function call.

    Uses the PREDICATE_REGISTRY to look up the function and construct
    the appropriate node.

    Args:
        func_name: The predicate function name.
        args: Parsed argument values.
        position: Character position in DSL text for error reporting.

    Returns:
        A PredicateNode configured for this predicate.

    Raises:
        RuleError: If the function name is unknown or arguments are invalid.
    """
    if func_name not in PREDICATE_REGISTRY:
        known = ", ".join(sorted(PREDICATE_REGISTRY.keys()))
        raise RuleError(f"Unknown predicate function '{func_name}' at position {position}. Known predicates: {known}")

    pred_type_str, default_field, operator, cost_hint = PREDICATE_REGISTRY[func_name]
    predicate_type = PredicateType(pred_type_str)

    # Dispatch to specific argument parsers
    if func_name == "keyword":
        return _build_keyword(args, predicate_type, operator, cost_hint, position)
    if func_name == "regex":
        return _build_regex(args, predicate_type, operator, cost_hint, position)
    if func_name == "intent":
        return _build_intent(args, predicate_type, operator, cost_hint, position)
    if func_name == "similarity":
        return _build_similarity(args, predicate_type, operator, cost_hint, position)
    if func_name in ("speaker", "channel"):
        return _build_simple_eq(func_name, args, predicate_type, default_field, operator, cost_hint, position)
    if func_name in ("field_eq", "field_gte", "field_lte"):
        return _build_field_comparison(func_name, args, predicate_type, operator, cost_hint, position)
    if func_name == "repeated":
        return _build_repeated(args, predicate_type, operator, cost_hint, position)
    if func_name == "occurs_after":
        return _build_occurs_after(args, predicate_type, operator, cost_hint, position)
    if func_name == "contains_any":
        return _build_contains_any(args, predicate_type, operator, cost_hint, position)

    # Should not reach here if PREDICATE_REGISTRY is consistent
    raise RuleError(f"No builder for predicate '{func_name}'")  # pragma: no cover


# -- Individual predicate builders --


def _expect_arg_count(func_name: str, args: list[str | int | float], expected: int, position: int) -> None:
    """Validate argument count for a predicate function."""
    if len(args) != expected:
        raise RuleError(f"'{func_name}' expects {expected} argument(s), got {len(args)} at position {position}")


def _expect_min_args(func_name: str, args: list[str | int | float], minimum: int, position: int) -> None:
    """Validate minimum argument count for a predicate function."""
    if len(args) < minimum:
        raise RuleError(f"'{func_name}' expects at least {minimum} argument(s), got {len(args)} at position {position}")


def _build_keyword(
    args: list[str | int | float],
    predicate_type: PredicateType,
    operator: str,
    cost_hint: int,
    position: int,
) -> PredicateNode:
    """keyword(field, value) or keyword(value)."""
    if len(args) == 1:
        return PredicateNode(
            predicate_type=predicate_type,
            field_name="text",
            operator=operator,
            value=str(args[0]),
            cost_hint=cost_hint,
        )
    _expect_arg_count("keyword", args, 2, position)
    return PredicateNode(
        predicate_type=predicate_type,
        field_name=str(args[0]),
        operator=operator,
        value=str(args[1]),
        cost_hint=cost_hint,
    )


def _build_regex(
    args: list[str | int | float],
    predicate_type: PredicateType,
    operator: str,
    cost_hint: int,
    position: int,
) -> PredicateNode:
    """regex(field, pattern) or regex(pattern)."""
    if len(args) == 1:
        return PredicateNode(
            predicate_type=predicate_type,
            field_name="text",
            operator=operator,
            value=str(args[0]),
            cost_hint=cost_hint,
        )
    _expect_arg_count("regex", args, 2, position)
    return PredicateNode(
        predicate_type=predicate_type,
        field_name=str(args[0]),
        operator=operator,
        value=str(args[1]),
        cost_hint=cost_hint,
    )


def _build_intent(
    args: list[str | int | float],
    predicate_type: PredicateType,
    operator: str,
    cost_hint: int,
    position: int,
) -> PredicateNode:
    """intent(label) or intent(label, threshold)."""
    _expect_min_args("intent", args, 1, position)
    if len(args) > 2:
        raise RuleError(f"'intent' expects 1-2 argument(s), got {len(args)} at position {position}")
    label = str(args[0])
    threshold = float(args[1]) if len(args) == 2 else None
    return PredicateNode(
        predicate_type=predicate_type,
        field_name="intent_score",
        operator=operator,
        value=label,
        threshold=threshold,
        cost_hint=cost_hint,
    )


def _build_similarity(
    args: list[str | int | float],
    predicate_type: PredicateType,
    operator: str,
    cost_hint: int,
    position: int,
) -> PredicateNode:
    """similarity(field, value, threshold)."""
    _expect_arg_count("similarity", args, 3, position)
    return PredicateNode(
        predicate_type=predicate_type,
        field_name=str(args[0]),
        operator=operator,
        value=str(args[1]),
        threshold=float(args[2]),
        cost_hint=cost_hint,
    )


def _build_simple_eq(
    func_name: str,
    args: list[str | int | float],
    predicate_type: PredicateType,
    field_name: str,
    operator: str,
    cost_hint: int,
    position: int,
) -> PredicateNode:
    """speaker(value) or channel(value)."""
    _expect_arg_count(func_name, args, 1, position)
    return PredicateNode(
        predicate_type=predicate_type,
        field_name=field_name,
        operator=operator,
        value=str(args[0]),
        cost_hint=cost_hint,
    )


def _build_field_comparison(
    func_name: str,
    args: list[str | int | float],
    predicate_type: PredicateType,
    operator: str,
    cost_hint: int,
    position: int,
) -> PredicateNode:
    """field_eq(field, value), field_gte(field, value), field_lte(field, value)."""
    _expect_arg_count(func_name, args, 2, position)
    value: str | int | float = args[1]
    return PredicateNode(
        predicate_type=predicate_type,
        field_name=str(args[0]),
        operator=operator,
        value=value,
        cost_hint=cost_hint,
    )


def _build_repeated(
    args: list[str | int | float],
    predicate_type: PredicateType,
    operator: str,
    cost_hint: int,
    position: int,
) -> PredicateNode:
    """repeated(field, value, count)."""
    _expect_arg_count("repeated", args, 3, position)
    return PredicateNode(
        predicate_type=predicate_type,
        field_name=str(args[0]),
        operator=operator,
        value=str(args[1]),
        cost_hint=cost_hint,
        metadata={"count": int(args[2])},
    )


def _build_occurs_after(
    args: list[str | int | float],
    predicate_type: PredicateType,
    operator: str,
    cost_hint: int,
    position: int,
) -> PredicateNode:
    """occurs_after(field, first, second)."""
    _expect_arg_count("occurs_after", args, 3, position)
    return PredicateNode(
        predicate_type=predicate_type,
        field_name=str(args[0]),
        operator=operator,
        value=str(args[1]),
        cost_hint=cost_hint,
        metadata={"second": str(args[2])},
    )


def _build_contains_any(
    args: list[str | int | float],
    predicate_type: PredicateType,
    operator: str,
    cost_hint: int,
    position: int,
) -> PredicateNode:
    """contains_any("word1", "word2", ...) — inline function-call syntax."""
    if not args:
        raise RuleError(f"'contains_any' expects at least 1 argument at position {position}")
    word_list = [str(a) for a in args]
    return PredicateNode(
        predicate_type=predicate_type,
        field_name="text",
        operator=operator,
        value=word_list,
        cost_hint=cost_hint,
    )


# ---------------------------------------------------------------------------
# Predicate node construction — dotted namespace syntax
# ---------------------------------------------------------------------------


def _build_dotted_predicate(
    namespace: str,
    method: str,
    args: list[str | int | float | list[str]],
    kwargs: dict[str, str | int | float],
    chains: list[tuple[str, list[str | int | float | list[str]], dict[str, str | int | float]]],
    comparison_op: str | None,
    comparison_value: str | int | float | None,
    position: int,
) -> PredicateNode:
    """Build a PredicateNode from dotted namespace syntax.

    Maps dotted expressions like ``semantic.intent("x") > 0.82`` into
    the same PredicateNode structure as the inline function-call syntax.

    Args:
        namespace: The namespace (e.g., "semantic", "lexical", "context").
        method: The method name (e.g., "intent", "contains_any").
        args: Positional arguments.
        kwargs: Keyword arguments.
        chains: Chained method calls [(method, args, kwargs), ...].
        comparison_op: Trailing comparison operator (e.g., ">", ">=") or None.
        comparison_value: Right-hand side of comparison or None.
        position: Character position for error reporting.

    Returns:
        A PredicateNode.

    Raises:
        RuleError: If the namespace/method combination is unknown.
    """
    # Handle context namespace with method chaining
    if namespace == "context":
        return _build_context_predicate(
            method,
            args,
            kwargs,
            chains,
            comparison_op,
            comparison_value,
            position,
        )

    # Look up in namespace map
    key = (namespace, method)
    if key not in NAMESPACE_PREDICATE_MAP:
        known_ns = sorted({ns for ns, _ in NAMESPACE_PREDICATE_MAP})
        raise RuleError(
            f"Unknown predicate '{namespace}.{method}' at position {position}. Known namespaces: {', '.join(known_ns)}"
        )

    func_name = NAMESPACE_PREDICATE_MAP[key]

    # Resolve comparison operator into threshold or adjust operator
    if func_name == "intent":
        # semantic.intent("cancelamento") > 0.82
        label = str(args[0]) if args else ""
        threshold = float(comparison_value) if comparison_value is not None else None
        internal_op = COMPARISON_OP_MAP.get(comparison_op, "gte") if comparison_op else "gte"
        return PredicateNode(
            predicate_type=PredicateType.SEMANTIC,
            field_name="intent_score",
            operator=internal_op,
            value=label,
            threshold=threshold,
            cost_hint=4,
        )

    if func_name == "similarity":
        # semantic.similarity("quero cancelar meu serviço") > 0.86
        ref_text = str(args[0]) if args else ""
        threshold = float(comparison_value) if comparison_value is not None else None
        internal_op = COMPARISON_OP_MAP.get(comparison_op, "similarity_above") if comparison_op else "similarity_above"
        return PredicateNode(
            predicate_type=PredicateType.SEMANTIC,
            field_name="embedding_similarity",
            operator=internal_op,
            value=ref_text,
            threshold=threshold,
            cost_hint=4,
        )

    if func_name == "keyword":
        # lexical.contains("billing")
        value = str(args[0]) if args else ""
        return PredicateNode(
            predicate_type=PredicateType.LEXICAL,
            field_name="text",
            operator="contains",
            value=value,
            cost_hint=1,
        )

    if func_name == "contains_any":
        # lexical.contains_any(["cancelar", "encerrar", "desistir"])
        word_list: list[str] = []
        for arg in args:
            if isinstance(arg, list):
                word_list.extend(arg)
            else:
                word_list.append(str(arg))
        return PredicateNode(
            predicate_type=PredicateType.LEXICAL,
            field_name="text",
            operator="contains_any",
            value=word_list,
            cost_hint=1,
        )

    if func_name == "regex":
        # lexical.regex("cancel|terminate")
        pattern = str(args[0]) if args else ""
        return PredicateNode(
            predicate_type=PredicateType.LEXICAL,
            field_name="text",
            operator="regex",
            value=pattern,
            cost_hint=1,
        )

    raise RuleError(f"Unhandled dotted predicate '{namespace}.{method}' at position {position}")  # pragma: no cover


def _build_context_predicate(
    method: str,
    args: list[str | int | float | list[str]],
    kwargs: dict[str, str | int | float],
    chains: list[tuple[str, list[str | int | float | list[str]], dict[str, str | int | float]]],
    comparison_op: str | None,
    comparison_value: str | int | float | None,
    position: int,
) -> PredicateNode:
    """Build a PredicateNode for context namespace with method chaining.

    Handles expressions like:
        ``context.turn_window(5).count(intent="insatisfacao") >= 2``

    Maps to a contextual predicate with metadata carrying the window
    configuration and chain parameters.
    """
    if method != "turn_window":
        raise RuleError(f"Unknown context method '{method}' at position {position}. Known: turn_window")

    window_size = int(args[0]) if args and not isinstance(args[0], list) else 5

    if not chains:
        raise RuleError(
            f"context.turn_window({window_size}) requires a chained method (e.g., .count(...)) at position {position}"
        )

    chain_method, chain_args, chain_kwargs = chains[0]
    internal_op = COMPARISON_OP_MAP.get(comparison_op, "gte") if comparison_op else "gte"
    threshold_value = comparison_value if comparison_value is not None else 1

    if chain_method == "count":
        # context.turn_window(5).count(intent="insatisfacao") >= 2
        return PredicateNode(
            predicate_type=PredicateType.CONTEXTUAL,
            field_name="window_count",
            operator=internal_op,
            value=threshold_value,
            cost_hint=3,
            metadata={
                "window_size": window_size,
                **{k: v for k, v in chain_kwargs.items()},
                **({"pattern": str(chain_args[0])} if chain_args else {}),
            },
        )

    if chain_method == "any":
        # context.turn_window(5).any(keyword="cancelar")
        return PredicateNode(
            predicate_type=PredicateType.CONTEXTUAL,
            field_name="window_any",
            operator="eq",
            value=True,
            cost_hint=3,
            metadata={
                "window_size": window_size,
                **{k: v for k, v in chain_kwargs.items()},
            },
        )

    raise RuleError(f"Unknown chain method '{chain_method}' at position {position}. Known: count, any")


# ---------------------------------------------------------------------------
# Predicate node construction — infix comparison syntax
# ---------------------------------------------------------------------------


def _build_infix_predicate(
    field_name: str,
    op_str: str,
    value: str | int | float,
    position: int,
) -> PredicateNode:
    """Build a PredicateNode from infix comparison syntax.

    Maps ``speaker == "customer"`` to the same PredicateNode as
    ``speaker("customer")``.

    Args:
        field_name: Bare field name (e.g., "speaker", "channel").
        op_str: Comparison operator string (e.g., "==", ">").
        value: Right-hand value.
        position: Character position for error reporting.

    Returns:
        A PredicateNode.

    Raises:
        RuleError: If the field name is unknown for infix syntax.
    """
    if field_name not in INFIX_FIELD_MAP:
        known = ", ".join(sorted(INFIX_FIELD_MAP.keys()))
        raise RuleError(
            f"Unknown field '{field_name}' for infix comparison at position {position}. "
            f"Known fields: {known}. Use dotted syntax for other predicates."
        )

    pred_type_str, internal_field, cost_hint = INFIX_FIELD_MAP[field_name]
    internal_op = COMPARISON_OP_MAP.get(op_str, "eq")

    return PredicateNode(
        predicate_type=PredicateType(pred_type_str),
        field_name=internal_field,
        operator=internal_op,
        value=value,
        cost_hint=cost_hint,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_dsl(dsl_text: str) -> ASTNode:
    """Parse a DSL text expression into an ASTNode.

    This is the main entry point for DSL parsing. It tokenizes the input
    and runs the recursive-descent parser. Supports both inline and block
    syntax — for block syntax (``RULE...WHEN...THEN``), only the WHEN
    clause AST is returned. Use ``parse_rule_block()`` for full block parsing.

    Args:
        dsl_text: Rule expression in DSL syntax (inline or block).

    Returns:
        The root ASTNode of the parsed expression.

    Raises:
        RuleError: If the DSL text has syntax errors, unknown predicates,
            or invalid arguments.
    """
    if not dsl_text or not dsl_text.strip():
        raise RuleError("Empty DSL expression")
    tokens = tokenize(dsl_text)
    parser = _Parser(tokens)

    # Detect block syntax: starts with RULE keyword
    if tokens and tokens[0].token_type == TokenType.RULE:
        block = _parse_block(parser)
        return block.ast

    return parser.parse()


def parse_rule_block(dsl_text: str) -> ParsedRuleBlock:
    """Parse a RULE...WHEN...THEN block into a ParsedRuleBlock.

    Args:
        dsl_text: Full rule block text starting with ``RULE``.

    Returns:
        ParsedRuleBlock with rule_name, ast, and actions.

    Raises:
        RuleError: If the text is not a valid rule block.
    """
    if not dsl_text or not dsl_text.strip():
        raise RuleError("Empty rule block")

    tokens = tokenize(dsl_text)
    parser = _Parser(tokens)

    if tokens[0].token_type != TokenType.RULE:
        raise RuleError("Rule block must start with 'RULE' keyword")

    return _parse_block(parser)


def _parse_block(parser: _Parser) -> ParsedRuleBlock:
    """Internal: parse a RULE...WHEN...THEN block.

    Args:
        parser: Parser positioned at the RULE token.

    Returns:
        ParsedRuleBlock with extracted name, AST, and actions.
    """
    # Consume RULE
    parser._expect(TokenType.RULE, "Expected 'RULE'")

    # Rule name
    name_tok = parser._current()
    if name_tok.token_type != TokenType.IDENTIFIER:
        raise RuleError(f"Expected rule name after RULE at position {name_tok.position}")
    rule_name = str(name_tok.value)
    parser._advance()

    # WHEN
    parser._expect(TokenType.WHEN, f"Expected 'WHEN' after rule name '{rule_name}'")

    # Parse WHEN expression (stops at THEN or EOF)
    ast = parser.parse_until_then()

    # THEN (optional)
    actions: list[RuleAction] = []
    if parser._current().token_type == TokenType.THEN:
        parser._advance()  # consume THEN

        raw_actions = parser.parse_actions()
        for action_name, action_args in raw_actions:
            if action_name not in ("tag", "score", "priority"):
                raise RuleError(f"Unknown action '{action_name}'. Known actions: tag, score, priority")
            if not action_args:
                raise RuleError(f"Action '{action_name}' requires at least 1 argument")
            value: str | float = float(action_args[0]) if action_name == "score" else str(action_args[0])
            actions.append(RuleAction(action_type=action_name, value=value))

    # Verify we consumed everything
    if parser._current().token_type != TokenType.EOF:
        tok = parser._current()
        raise RuleError(f"Unexpected token '{tok.value}' at position {tok.position} after rule block")

    return ParsedRuleBlock(rule_name=rule_name, ast=ast, actions=actions)
