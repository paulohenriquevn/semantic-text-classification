"""Rule DSL parser — recursive-descent parser producing typed AST nodes.

Transforms DSL text into an AST tree using a two-phase process:

    1. Tokenization: DSL text → list of tokens
    2. Parsing: token stream → ASTNode tree

The parser is a recursive-descent parser implementing the grammar defined
in ``dsl.py``. It delegates semantic validation to the ``compiler`` module.

Token types:
    AND, OR, NOT    — logical operators (case-insensitive)
    LPAREN, RPAREN  — grouping parentheses
    COMMA           — argument separator
    STRING          — quoted string literal (single or double quotes)
    NUMBER          — integer or float literal
    IDENTIFIER      — function name or bare word
    EOF             — end of input
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
from talkex.rules.dsl import PREDICATE_REGISTRY

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class TokenType(StrEnum):
    """Token types produced by the DSL tokenizer."""

    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    COMMA = "COMMA"
    STRING = "STRING"
    NUMBER = "NUMBER"
    IDENTIFIER = "IDENTIFIER"
    EOF = "EOF"


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

        # Parentheses and comma
        if ch == "(":
            tokens.append(Token(TokenType.LPAREN, "(", i))
            i += 1
            continue
        if ch == ")":
            tokens.append(Token(TokenType.RPAREN, ")", i))
            i += 1
            continue
        if ch == ",":
            tokens.append(Token(TokenType.COMMA, ",", i))
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

        # Identifiers and keywords (AND, OR, NOT)
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
        """Parse: atom = predicate | "(" expression ")" """
        tok = self._current()

        # Parenthesized group
        if tok.token_type == TokenType.LPAREN:
            self._advance()  # consume (
            node = self._or_expr()
            if self._current().token_type != TokenType.RPAREN:
                raise RuleError(f"Expected ')' at position {self._current().position}, got '{self._current().value}'")
            self._advance()  # consume )
            return node

        # Predicate function call
        if tok.token_type == TokenType.IDENTIFIER:
            return self._predicate()

        raise RuleError(f"Expected predicate or '(' at position {tok.position}, got '{tok.value}' ({tok.token_type})")

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
            return tok.value  # type: ignore[return-value]
        if tok.token_type == TokenType.IDENTIFIER:
            self._advance()
            return str(tok.value)
        raise RuleError(f"Expected argument at position {tok.position}, got '{tok.value}' ({tok.token_type})")

    # -- Token stream helpers --

    def _current(self) -> Token:
        """Return the current token without consuming it."""
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        """Consume the current token and return it."""
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok


# ---------------------------------------------------------------------------
# Predicate node construction
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_dsl(dsl_text: str) -> ASTNode:
    """Parse a DSL text expression into an ASTNode.

    This is the main entry point for DSL parsing. It tokenizes the input
    and runs the recursive-descent parser.

    Args:
        dsl_text: Rule expression in DSL syntax.

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
    return parser.parse()
