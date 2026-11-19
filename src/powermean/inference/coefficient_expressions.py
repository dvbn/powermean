"""Bounded coefficient algebra for design/04 Section 6 combinations.

Only real constants, coefficient identifiers, unary signs, arithmetic and exp
are interpreted. Python evaluation, attributes, indexing and caller namespaces
are never used. Affine expressions are reduced symbolically; nonlinear values
carry a full analytic gradient at the supplied reported coefficient point.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from fractions import Fraction

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
    return _affine_operand(expression, names, expression)


def _affine_operand(source: str, names: tuple[str, ...], expression: str) -> Affine:
    """Reduce one operand, retaining the full request for a power-domain error."""
    try:
        result = _affine(_tree(source), names, expression)
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
        return Affine(
            left.constant / right.constant,
            tuple(weight / right.constant for weight in left.weights),
        )
    if isinstance(operation, ast.Pow) and not any(right.weights):
        if right.constant == 1.0:
            return left
        if right.constant == 0.0:
            return Affine(1.0, (0.0,) * len(left.weights))
        if not any(left.weights):
            try:
                value = math.pow(left.constant, right.constant)
            except ValueError as error:
                raise _refusal(expression) from error
            return Affine(value, left.weights)
    raise _refusal(expression)


def equality(expression: str, names: tuple[str, ...]) -> Affine:
    """Reduce one affine equality to constant + weights'beta = 0."""
    if not isinstance(expression, str) or expression.count("=") != 1:
        raise _refusal(expression)
    left, right = expression.split("=")
    return _affine_binary(
        ast.Sub(),
        _affine_operand(left, names, expression),
        _affine_operand(right, names, expression),
        expression,
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
    terms = zip(left.gradient, right.gradient, strict=True)
    if isinstance(operation, ast.Add | ast.Sub):
        sign = -1.0 if isinstance(operation, ast.Sub) else 1.0
        value = left.value + sign * right.value
        gradient = tuple(a + sign * b for a, b in terms)
    elif isinstance(operation, ast.Mult):
        value = left.value * right.value
        gradient = _product_gradient(left, right)
    elif isinstance(operation, ast.Div):
        value = left.value / right.value
        if right.depends:
            # Round (a - (x/y)*b)/y once: even a finite rounded quotient
            # or product can erase a small derivative or invent one at zero.
            denominator = Fraction(right.value)
            quotient = Fraction(left.value) / denominator
            gradient = tuple(
                float((Fraction(a) - quotient * Fraction(b)) / denominator)
                if b
                else a / right.value
                for a, b in terms
            )
        else:
            gradient = tuple(a / right.value for a in left.gradient)
    elif isinstance(operation, ast.Pow):
        value, d_left, d_right = _power(left, right)
        gradient = tuple(d_left * a + d_right * b for a, b in terms)
    else:
        raise _refusal(expression)
    if not math.isfinite(value):
        raise _refusal(expression)
    if not all(map(math.isfinite, gradient)):
        if not isinstance(operation, ast.Pow):
            raise _refusal(expression)
        gradient = _recover_power_gradient(left, right, value, gradient)
    return ExpressionValue(value, gradient, left.depends or right.depends)


def _product_gradient(
    left: ExpressionValue, right: ExpressionValue
) -> tuple[float, ...]:
    """Round each product-rule coordinate after adding both contributions."""
    # A shared coefficient can leave a finite derivative even when the two
    # rounded products coincide or overflow. A single contribution already
    # gets the correctly rounded binary64 multiplication.
    return tuple(
        float(Fraction(right.value) * Fraction(a) + Fraction(left.value) * Fraction(b))
        if a and b
        else right.value * a + left.value * b
        for a, b in zip(left.gradient, right.gradient, strict=True)
    )


def _recover_power_gradient(
    left: ExpressionValue,
    right: ExpressionValue,
    value: float,
    gradient: tuple[float, ...],
) -> tuple[float, ...]:
    """Round a failed chain-rule coordinate only after combining both terms."""
    x, y = Fraction(left.value), Fraction(right.value)
    powered = Fraction(value)
    d_left = powered * y / x if left.depends else Fraction(0)
    d_right = powered * Fraction(math.log(left.value)) if right.depends else Fraction(0)
    return tuple(
        g if math.isfinite(g) else float(d_left * Fraction(a) + d_right * Fraction(b))
        for g, a, b in zip(gradient, left.gradient, right.gradient, strict=True)
    )


def _power(left: ExpressionValue, right: ExpressionValue) -> tuple[float, float, float]:
    if left.depends and left.value == 0.0 and not right.value.is_integer():
        raise ValueError("a real nonintegral power is not smooth through zero")
    value = math.pow(left.value, right.value)
    if not left.depends and not right.depends:
        return value, 0.0, 0.0
    if not right.depends:
        try:
            derivative = (
                0.0
                if right.value == 0.0
                else right.value * math.pow(left.value, right.value - 1.0)
            )
        except OverflowError:
            # The partial may overflow even when its chain-rule product is finite.
            derivative = math.inf
        return value, derivative, 0.0
    d_left = right.value * value / left.value if left.depends else 0.0
    return value, d_left, value * math.log(left.value)
