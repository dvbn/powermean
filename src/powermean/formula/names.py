"""The names that a Python factor reads from the mapping Formulaic evaluates it in.

:mod:`powermean.formula.frame` attributes to each factor of a fitted spec the data
columns that its evaluation reads.  Formulaic evaluates a factor of Python code as
``eval(compiled, {}, mapping)``: the compiled expression, an empty dictionary as its
globals and, as its locals, a mapping that layers the data, the context (empty) and
Formulaic's transforms (``formulaic.utils.stateful_transforms``).  The data columns that
the evaluation reads are the names that Python loads from that mapping, and Python
loads a name from it only in the block of the expression, where a load by name tries
the mapping first, then the empty globals, then the builtins:

- a name of the expression outside any lambda or generator expression is loaded from
  the mapping; so is a name in a list, set or dictionary comprehension, which Python
  3.12 inlines into the enclosing block (PEP 709), except the comprehension's own
  variables, locals of the comprehension that hide the mapping within it;
- a name in the body of a lambda or of a generator expression, a function scope of its
  own, is a parameter, or a global load that reaches the empty globals and the
  builtins, never the mapping: ``x.map(lambda v: abs(v))`` calls the builtin ``abs``,
  whatever column is named ``abs``, and a column read there is an error of Formulaic's;
  the defaults of a lambda's parameters and the first iterable of a generator
  expression are evaluated in the enclosing scope;
- an assignment expression (``:=``) in the block of the expression stores its target
  in the mapping, where a later load finds it in place of the data: ``(z := x) + z``
  reads ``x`` alone, while ``(z > 0) + (z := x)`` reads the column ``z`` before the
  store, since Python evaluates left to right.  A conditional expression evaluates one
  of its branches; ``and`` and ``or`` evaluate their later operands, and a chained
  comparison its comparators after the first, only while the result is undecided
  (``0 > 1 > (z := 0)`` never stores ``z``), so a store there has not surely happened
  for a load after it, which reads the column.  An assignment expression within a
  comprehension binds its
  target as a global of the block (PEP 572), so every load of that name in the block
  is a global load that never reaches the mapping.

The expression is read, never evaluated: a name loaded in a branch that the data would
not take is a name read, as it is for a conditional expression without a store.  The
block's declared globals are those of Python's own symbol table (:mod:`symtable`); the
rest follows Python's syntax tree (:mod:`ast`) in the order of evaluation, so that a
name is classified by where Python loads it from, not by whether the expression binds
it somewhere.  Formulaic binds each name within backticks to a variable before Python
evaluates the expression, so such a name is read wherever it stands.

The raw-reference collector can also request literal ``Q("name")`` reads in this
same traversal.  Only a call whose ``Q`` loads from the mapping contributes the
literal name; a caller whose data shadows the transform must leave this disabled.
Dynamic string reads remain unknown, without discarding independently known reads.
"""

from __future__ import annotations

import ast
import re
import symtable
from collections.abc import Container
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from powermean.formula.formulaic_api import Spec

# A Python string, or a name within backticks, in the expression of a Python factor,
# read as Formulaic's tokenizer reads them: a backslash keeps the next character from
# closing either.  Formulaic binds each name within backticks, the group, to a variable
# before Python evaluates the expression.
_QUOTED = re.compile(
    r"""'(?:\\.|[^\\'])*'|"(?:\\.|[^\\"])*"|`((?:\\.|[^\\`])*)`""", re.DOTALL
)


@dataclass(frozen=True)
class _Scope:
    """What is known at a point of the evaluation of the block of an expression.

    Attributes
    ----------
    never : frozenset of str
        The names that the block declares global: the targets of the assignment
        expressions within its comprehensions, loaded as globals wherever they stand.
    literal_q : bool
        Whether an unshadowed mapping call ``Q("name")`` also reads the literal name.
    loaded : set of str
        The names loaded from the mapping so far.
    stored : set of str
        The names that an assignment expression has surely stored in the mapping
        before this point of the evaluation.
    hidden : list of str
        The variables of the comprehensions being read, innermost last: locals that
        hide the mapping within them.
    """

    never: frozenset[str]
    loaded: set[str] = field(default_factory=set)
    stored: set[str] = field(default_factory=set)
    hidden: list[str] = field(default_factory=list)
    literal_q: bool = False


def python_names(expression: str, *, literal_q: bool = False) -> set[str]:
    """The names that the evaluation of a Python factor reads, columns or not.

    Parameters
    ----------
    expression : str
        The expression of a factor that Formulaic evaluates as Python, as written.
    literal_q : bool, default False
        Include the string of a one-positional-string-literal ``Q`` call when the
        callable is loaded from the evaluation mapping, not locally bound.  The
        caller must exclude data-shadowed ``Q``.  This reuses the same parse and
        scope traversal; unknown dynamic lookups add no inferred column.

    Returns
    -------
    set of str
        The names within backticks, and the names that Python loads from the mapping
        of the evaluation (module docstring).  Each name within backticks has a
        stand-in of its own, a name longer than the expression, so that Python reads
        the expression and takes none of its own names, nor two names within
        backticks, for one name.  With ``literal_q``, the known literal column names
        are included as well.
    """
    quoted, stand_ins, code = _python_code(expression)
    looked_up = _names_loaded(code, literal_q=literal_q)
    return quoted | (looked_up - stand_ins)


# The builtins that read the namespace of the frame that calls them, here the mapping
# of the evaluation: ``locals()`` and ``vars()`` return it, ``dir()`` lists its names,
# ``eval`` and ``exec`` evaluate code in it.
NAMESPACE_BUILTINS = frozenset({"dir", "eval", "exec", "locals", "vars"})


def string_lookups(expression: str) -> set[str] | None:
    """The columns that a Python factor reads by a string, or ``None`` if not known.

    A Python factor reads a column by its name as Python loads it
    (:func:`python_names`), or by a string: Formulaic's ``Q("name")`` returns the
    column of the string, and a builtin of :data:`NAMESPACE_BUILTINS` reaches the
    mapping of the evaluation, the data among it.  The strings of every ``Q`` called
    with one string literal are returned; ``None`` when the expression uses ``Q`` in
    any other way or loads such a builtin from the mapping, since the column then read
    is known only from the evaluation.
    """
    loaded = python_names(expression)
    if loaded & NAMESPACE_BUILTINS:
        return None
    if "Q" not in loaded:
        return set()
    tree = ast.parse(_python_code(expression)[2], mode="eval")
    used = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "Q"]
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id != "Q":
                continue
            if node.keywords or len(node.args) != 1:
                return None
            name = node.args[0]
            if not (isinstance(name, ast.Constant) and isinstance(name.value, str)):
                return None
            found.add(name.value)
            used.remove(node.func)
    return None if used else found


def _python_code(expression: str) -> tuple[set[str], set[str], str]:
    """The names within backticks, their stand-ins and the code that Python reads.

    The stand-ins are those of :func:`python_names`.
    """
    quoted = {
        found[1] for found in _QUOTED.finditer(expression) if found[1] is not None
    }
    stand_ins = {
        name: "_" * (len(expression) + at) for at, name in enumerate(quoted, start=1)
    }
    code = _QUOTED.sub(
        lambda found: found[0] if found[1] is None else stand_ins[found[1]], expression
    )
    return quoted, set(stand_ins.values()), code


def _names_loaded(code: str, *, literal_q: bool = False) -> set[str]:
    """The names that Python loads from the mapping while it evaluates ``code``.

    The block of the expression, in Python's symbol table, says which names it
    declares global; its syntax tree is then read in the order of evaluation.
    """
    block = symtable.symtable(code, "<factor>", "eval")
    declared = {
        symbol.get_name()
        for symbol in block.get_symbols()
        if symbol.is_declared_global()
    }
    tree = ast.parse(code, mode="eval")
    scope = _Scope(never=frozenset(declared), literal_q=literal_q)
    _read(tree.body, scope)
    return scope.loaded


def _read(node: ast.AST, scope: _Scope) -> None:
    """Read ``node`` in the order of its evaluation, within ``scope``.

    The nodes read apart are those that bind a name or open a scope (an assignment
    expression, a lambda, a generator expression, a comprehension), those that
    evaluate a part of themselves conditionally (a conditional expression, a Boolean
    operation, a chained comparison) and the one whose children are not listed in the
    order of evaluation (a dictionary display, each key before its value).  Every
    other node evaluates its children in their order.
    """
    match node:
        case ast.Name(id=name, ctx=ast.Load()):
            _name(name, scope)
        case ast.Call(
            func=ast.Name(id="Q"), args=[ast.Constant(value=name)], keywords=[]
        ) if scope.literal_q and isinstance(name, str):
            if _name("Q", scope):
                scope.loaded.add(name)
        case ast.NamedExpr(target=ast.Name(id=name), value=value):
            _read(value, scope)
            scope.stored.add(name)
        case ast.Lambda(args=arguments):
            _defaults(arguments, scope)
        case ast.GeneratorExp(generators=[first, *_]):
            _read(first.iter, scope)
        case (
            ast.ListComp(generators=generators, elt=element)
            | ast.SetComp(generators=generators, elt=element)
        ):
            _comprehension(generators, [element], scope)
        case ast.DictComp(generators=generators, key=key, value=value):
            _comprehension(generators, [key, value], scope)
        case ast.IfExp():
            _conditional(node, scope)
        case ast.BoolOp(values=[first, *rest]):
            _undecided([first], rest, scope)
        case ast.Compare(left=left, comparators=[first, *rest]):
            _undecided([left, first], rest, scope)
        case ast.Dict(keys=keys, values=values):
            _dictionary(keys, values, scope)
        case _:
            _children(node, scope)


def _name(name: str, scope: _Scope) -> bool:
    """A load of ``name``: from the mapping, unless Python finds it elsewhere.

    A global of the block is loaded from the empty globals and the builtins; a variable
    of an enclosing comprehension is a local; a name that an assignment expression has
    surely stored is found in the mapping before the data.  Return whether this
    load reaches the original mapping, for the literal-Q call attribution.
    """
    elsewhere = scope.never | set(scope.hidden) | scope.stored
    if name not in elsewhere:
        scope.loaded.add(name)
        return True
    return False


def _defaults(arguments: ast.arguments, scope: _Scope) -> None:
    """The defaults of a lambda's parameters, evaluated in the enclosing scope.

    The body of the lambda is a function scope of its own, whose names never reach the
    mapping (module docstring); it is not read.
    """
    for default in [*arguments.defaults, *arguments.kw_defaults]:
        if default is not None:
            _read(default, scope)


def _comprehension(
    generators: list[ast.comprehension], body: list[ast.expr], scope: _Scope
) -> None:
    """A list, set or dictionary comprehension, inlined into the enclosing block.

    Its first iterable is evaluated before its variables exist; the later iterables,
    the conditions and the body (an element, or a key and a value) with the variables
    of the generators so far hidden, which the end of the comprehension uncovers.
    """
    _read(generators[0].iter, scope)
    outside = len(scope.hidden)
    for at, generator in enumerate(generators):
        if at > 0:
            _read(generator.iter, scope)
        _read(generator.target, scope)
        scope.hidden.extend(_bound(generator.target))
        for condition in generator.ifs:
            _read(condition, scope)
    for part in body:
        _read(part, scope)
    del scope.hidden[outside:]


def _conditional(node: ast.IfExp, scope: _Scope) -> None:
    """A conditional expression: its test, then one of its branches.

    A name is surely stored after it only where both branches store it.
    """
    _read(node.test, scope)
    before = set(scope.stored)
    _read(node.body, scope)
    taken = set(scope.stored)
    scope.stored.difference_update(taken - before)
    _read(node.orelse, scope)
    scope.stored.intersection_update(taken)


def _undecided(
    reached: list[ast.expr], conditional: list[ast.expr], scope: _Scope
) -> None:
    """An operation that stops once its result is decided, operand by operand.

    A Boolean operation (``and``, ``or``) surely evaluates its first operand, a chained
    comparison (``a < b < c``) its left side and its first comparator; the
    ``conditional`` operands follow only while the result is undecided.  Only what the
    operands surely ``reached`` store is surely stored after the operation.
    """
    for operand in reached:
        _read(operand, scope)
    sure = set(scope.stored)
    for operand in conditional:
        _read(operand, scope)
    scope.stored.intersection_update(sure)


def _dictionary(
    keys: list[ast.expr | None], values: list[ast.expr], scope: _Scope
) -> None:
    """A dictionary display: each key, then its value; ``**mapping`` has no key."""
    for key, value in zip(keys, values, strict=True):
        if key is not None:
            _read(key, scope)
        _read(value, scope)


def _children(node: ast.AST, scope: _Scope) -> None:
    """Any other node: its children, listed in the order of their evaluation."""
    for child in ast.iter_child_nodes(node):
        _read(child, scope)


def _bound(target: ast.expr) -> list[str]:
    """The names that the target of a generator binds, however nested in the target."""
    return [
        node.id
        for node in ast.walk(target)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    ]


def columns_read(spec: Spec, columns: Container[str]) -> tuple[str, ...]:
    """The data columns that the factors of ``spec`` read, sorted."""
    read: set[str] = set()
    literal_q = "Q" not in columns
    for term in spec.term_slices:
        for factor in term.factors:
            if factor.eval_method.value == "lookup":
                read.add(factor.expr)
            elif factor.eval_method.value == "python":
                read |= python_names(factor.expr, literal_q=literal_q)
    return tuple(sorted(name for name in read if name in columns))
