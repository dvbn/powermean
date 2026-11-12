"""Bounded coefficient algebra for coefficient combinations.

Only real constants, coefficient identifiers, unary signs, arithmetic and exp
are interpreted. Python evaluation, attributes, indexing and caller namespaces
are never used. Affine expressions are reduced symbolically; nonlinear values
carry a full analytic gradient at the supplied reported coefficient point.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass

from powermean.data.request_values import invalid
from powermean.errors import SpecificationError


@dataclass(frozen=True, slots=True)
class Affine:
    """A constant plus weights in the exact reported coefficient order."""

    constant: float
    weights: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ExpressionValue:
    """A scalar value and full analytic coefficient gradient."""

    value: float
    gradient: tuple[float, ...]
    depends: bool


def _refusal(expression: object) -> SpecificationError:
    return invalid(
        "expression",
        expression,
        "real coefficient algebra (+, -, *, /, **), with exp for nlcom",
    )


def _tree(expression: str) -> ast.expr:
    if not isinstance(expression, str) or not expression.strip():
        raise _refusal(expression)
    try:
        return ast.parse(expression.strip(), mode="eval").body
    except (SyntaxError, ValueError, RecursionError) as error:
        raise _refusal(expression) from error


def _constant(node: ast.Constant, expression: str) -> float:
    if isinstance(node.value, bool) or not isinstance(node.value, int | float):
        raise _refusal(expression)
    try:
        value = float(node.value)
    except OverflowError as error:
        raise _refusal(expression) from error
    if not math.isfinite(value):
        raise _refusal(expression)
    return value


def affine(expression: str, names: tuple[str, ...]) -> Affine:
    """Reduce an affine expression without consulting a coefficient value."""
    try:
        result = _affine(_tree(expression), names, expression)
    except (ArithmeticError, RecursionError) as error:
        raise _refusal(expression) from error
    if not all(math.isfinite(x) for x in (result.constant, *result.weights)):
        raise _refusal(expression)
    return result


def _affine(node: ast.expr, names: tuple[str, ...], expression: str) -> Affine:
    zeros = (0.0,) * len(names)
    if isinstance(node, ast.Constant):
        return Affine(_constant(node, expression), zeros)
    if isinstance(node, ast.Name) and node.id in names:
        return Affine(0.0, tuple(float(name == node.id) for name in names))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
        term = _affine(node.operand, names, expression)
        return _scale(term, -1.0 if isinstance(node.op, ast.USub) else 1.0)
    if isinstance(node, ast.BinOp):
        left = _affine(node.left, names, expression)
        right = _affine(node.right, names, expression)
        return _affine_binary(node.op, left, right, expression)
    raise _refusal(expression)


def _scale(term: Affine, multiplier: float) -> Affine:
    return Affine(
        multiplier * term.constant, tuple(multiplier * x for x in term.weights)
    )


def _affine_binary(
    operation: ast.operator, left: Affine, right: Affine, expression: str
) -> Affine:
    if isinstance(operation, ast.Add | ast.Sub):
        signed = _scale(right, -1.0 if isinstance(operation, ast.Sub) else 1.0)
        return Affine(
            left.constant + signed.constant,
            tuple(a + b for a, b in zip(left.weights, signed.weights, strict=True)),
        )
    if isinstance(operation, ast.Mult):
        if not any(left.weights):
            return _scale(right, left.constant)
        if not any(right.weights):
            return _scale(left, right.constant)
    if isinstance(operation, ast.Div) and not any(right.weights):
        return _scale(left, 1.0 / right.constant)
    if isinstance(operation, ast.Pow) and not any(right.weights):
        if right.constant == 1.0:
            return left
        if right.constant == 0.0:
            return Affine(1.0, (0.0,) * len(left.weights))
        if not any(left.weights):
            return Affine(math.pow(left.constant, right.constant), left.weights)
    raise _refusal(expression)


def equality(expression: str, names: tuple[str, ...]) -> Affine:
    """Reduce one affine equality to constant + weights'beta = 0."""
    if not isinstance(expression, str) or expression.count("=") != 1:
        raise _refusal(expression)
    left, right = expression.split("=")
    return _affine_binary(
        ast.Sub(), affine(left, names), affine(right, names), expression
    )


def nonlinear(
    expression: str, names: tuple[str, ...], beta: tuple[float, ...]
) -> ExpressionValue:
    """Evaluate smooth algebra and exp with analytic forward differentiation.

    Powers with a coefficient-dependent exponent require a positive base. A
    constant exponent also allows the real differentiable polynomial domain.
    Undefined values, nonsmooth points and unrepresentable gradients are refused.
    """
    try:
        result = _nonlinear(_tree(expression), names, beta, expression)
    except (ArithmeticError, ValueError, RecursionError) as error:
        raise _refusal(expression) from error
    if not all(math.isfinite(x) for x in (result.value, *result.gradient)):
        raise _refusal(expression)
    return result


def _nonlinear(
    node: ast.expr, names: tuple[str, ...], beta: tuple[float, ...], expression: str
) -> ExpressionValue:
    if isinstance(node, ast.Constant):
        return ExpressionValue(_constant(node, expression), (0.0,) * len(names), False)
    if isinstance(node, ast.Name) and node.id in names:
        return ExpressionValue(
            beta[names.index(node.id)],
            tuple(float(name == node.id) for name in names),
            True,
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
        term = _nonlinear(node.operand, names, beta, expression)
        sign = -1.0 if isinstance(node.op, ast.USub) else 1.0
        return ExpressionValue(
            sign * term.value, tuple(sign * x for x in term.gradient), term.depends
        )
    if isinstance(node, ast.BinOp):
        left = _nonlinear(node.left, names, beta, expression)
        right = _nonlinear(node.right, names, beta, expression)
        return _nonlinear_binary(node.op, left, right, expression)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "exp"
        and len(node.args) == 1
        and not node.keywords
    ):
        term = _nonlinear(node.args[0], names, beta, expression)
        value = math.exp(term.value)
        if value == 0.0:
            raise _refusal(expression)
        return ExpressionValue(
            value, tuple(value * x for x in term.gradient), term.depends
        )
    raise _refusal(expression)


def _nonlinear_binary(
    operation: ast.operator,
    left: ExpressionValue,
    right: ExpressionValue,
    expression: str,
) -> ExpressionValue:
    if isinstance(operation, ast.Add | ast.Sub):
        sign = -1.0 if isinstance(operation, ast.Sub) else 1.0
        value, d_left, d_right = left.value + sign * right.value, 1.0, sign
    elif isinstance(operation, ast.Mult):
        value, d_left, d_right = left.value * right.value, right.value, left.value
    elif isinstance(operation, ast.Div):
        value = left.value / right.value
        d_left, d_right = 1.0 / right.value, -value / right.value
    elif isinstance(operation, ast.Pow):
        value, d_left, d_right = _power(left, right)
    else:
        raise _refusal(expression)
    gradient = tuple(
        d_left * a + d_right * b
        for a, b in zip(left.gradient, right.gradient, strict=True)
    )
    if not all(math.isfinite(x) for x in (value, *gradient)):
        raise _refusal(expression)
    return ExpressionValue(value, gradient, left.depends or right.depends)


def _power(left: ExpressionValue, right: ExpressionValue) -> tuple[float, float, float]:
    if left.depends and left.value == 0.0 and not right.value.is_integer():
        raise ValueError("a real nonintegral power is not smooth through zero")
    value = math.pow(left.value, right.value)
    if not left.depends and not right.depends:
        return value, 0.0, 0.0
    if not right.depends:
        derivative = (
            0.0
            if right.value == 0.0
            else right.value * math.pow(left.value, right.value - 1.0)
        )
        return value, derivative, 0.0
    return value, right.value * value / left.value, value * math.log(left.value)
