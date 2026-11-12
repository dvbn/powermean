"""Typed views of the parts of Formulaic that the formula package uses.

Formulaic has no ``py.typed`` marker and no stub package is a dependency, so the type
checker reads none of its inline annotations, and a static ``import formulaic`` fails
the strict type check.  This module imports ``formulaic``, ``formulaic.errors``,
``formulaic.materializers``, ``formulaic.materializers.pandas`` and ``interface_meta``
with ``importlib.import_module`` and casts them to protocols or typed views of the
objects used, each typed as :mod:`powermean.formula` uses it, and no further.  Nothing
is wrapped: ``formulaic.model_matrix`` is Formulaic's function, ``formulaic.ModelSpec``
its class, ``formulaic_errors.FormulaSyntaxError`` its exception and
``PandasMaterializer`` its native pandas materializer class, which the tests check.  A
context is typed as a mapping, so that a frame depth, with which Formulaic would read
the variables of a caller, is a type error, and ``model_matrix`` accepts the explicit
``materializer`` keyword.  The import is dynamic, so import-linter does not see the
edge to Formulaic, which the formula package may import; the contracts that forbid
Formulaic concern the stages that may not import this package.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, TypedDict, Unpack, cast, overload

import pandas as pd


class EvalMethod(Protocol):
    """A member of ``Factor.EvalMethod``: how Formulaic evaluates a factor."""

    @property
    def value(self) -> str:
        """The name of the method: ``"lookup"``, ``"python"`` or ``"literal"``."""


class FactorKind(Protocol):
    """A member of Formulaic's factor-kind enumeration."""

    @property
    def value(self) -> str:
        """The fitted encoding kind."""


class Factor(Protocol):
    """A factor of a term of a fitted spec."""

    @property
    def expr(self) -> str:
        """The expression of the factor: a name, Python code or a literal."""

    @property
    def eval_method(self) -> EvalMethod:
        """How Formulaic evaluates the expression."""

    @property
    def kind(self) -> FactorKind:
        """The declared kind, which may be unknown before evaluation."""

    @kind.setter
    def kind(self, value: FactorKind) -> None:
        """Set the kind of an owned factor copy."""


class EvaluatedFactor(Protocol):
    """One evaluated factor at the materializer's encoding boundary."""

    @property
    def expr(self) -> str:
        """The fitted factor expression."""

    @property
    def values(self) -> object:
        """Formulaic's wrapped evaluated values."""


class Term(Protocol):
    """A term of a fitted spec, whose ``str`` is Formulaic's name of the term."""

    @property
    def factors(self) -> Sequence[Factor]:
        """The factors whose product the term is."""


class SpecKeywords(TypedDict, total=False):
    """The one keyword of a replay that the package may set."""

    na_action: str


class MaterializerKeywords(TypedDict, total=False):
    """The keywords of ``PandasMaterializer.get_model_matrix`` that the package sets."""

    output: str
    na_action: str


class Spec(Protocol):
    """A fitted ``formulaic.ModelSpec``."""

    @property
    def term_slices(self) -> Mapping[Term, slice]:
        """The columns of each term, in Formulaic's order."""

    @property
    def encoder_state(self) -> Mapping[str, tuple[FactorKind, dict[str, object]]]:
        """The fitted factor kinds and encoding state."""

    @property
    def transform_state(self) -> Mapping[str, object]:
        """The fitted stateful-transform state."""

    def update(
        self,
        *,
        encoder_state: dict[str, tuple[FactorKind, dict[str, object]]],
        transform_state: dict[str, object],
    ) -> Spec:
        """Shallowly replace explicitly owned state, without evaluating rows."""

    def get_model_matrix(
        self,
        data: pd.DataFrame,
        *,
        context: Mapping[str, object],
        **keywords: Unpack[SpecKeywords],
    ) -> object:
        """The model matrix of ``data``, with the state that the spec learned."""


if TYPE_CHECKING:

    class PandasMaterializer:
        """Formulaic's native ``PandasMaterializer``."""

        REGISTER_NAME: str

        def __init__(
            self, data: pd.DataFrame, *, context: Mapping[str, object]
        ) -> None:
            """Bind ``data`` and the explicit evaluation ``context``."""

        def _is_categorical(self, values: object) -> bool:
            """Whether ``values`` is encoded as a categorical factor."""

        def _encode_categorical(
            self, values: object, *args: object, **kwargs: object
        ) -> object:
            """Encode ``values`` into categorical contrast columns."""

        def _evaluate_factor(
            self, factor: Factor, spec: Spec, drop_rows: set[int]
        ) -> EvaluatedFactor:
            """Evaluate a factor once through Formulaic."""

        def _encode_evaled_factor(
            self,
            factor: EvaluatedFactor,
            spec: Spec,
            drop_rows: Sequence[int],
            reduced_rank: bool = False,
        ) -> dict[str, object]:
            """Encode one evaluated factor with the fitted state."""

        def get_model_matrix(
            self,
            spec: str | Spec,
            **keywords: Unpack[MaterializerKeywords],
        ) -> object:
            """The model matrix or matrices of ``spec`` evaluated on ``data``."""

else:
    PandasMaterializer = importlib.import_module(
        "formulaic.materializers.pandas"
    ).PandasMaterializer


class FormulaMaterializerType(Protocol):
    """Formulaic's ``FormulaMaterializer`` registry and resolution class."""

    def for_materializer(
        self, materializer: str | type[PandasMaterializer]
    ) -> type[PandasMaterializer]:
        """Resolve ``materializer`` to its ``FormulaMaterializer`` subclass."""

    def for_data(self, data: object) -> type[PandasMaterializer]:
        """Resolve ``data`` to its registered ``FormulaMaterializer`` subclass."""


class Override(Protocol):
    """The ``interface_meta.override`` decorator."""

    @overload
    def __call__[**P, R](self, func: Callable[P, R], /) -> Callable[P, R]: ...

    @overload
    def __call__[**P, R](
        self, *, force: bool = ...
    ) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


class Formulaic(Protocol):
    """The module ``formulaic``."""

    ModelSpec: type[Spec]

    def model_matrix(
        self,
        spec: str,
        data: pd.DataFrame,
        *,
        context: Mapping[str, object],
        materializer: str = ...,
        output: str,
        na_action: str,
    ) -> object:
        """The model matrices of the formula ``spec``, evaluated on ``data``."""


class FormulaicErrors(Protocol):
    """The module ``formulaic.errors``."""

    FormulaSyntaxError: type[Exception]


formulaic = cast("Formulaic", importlib.import_module("formulaic"))
formulaic_errors = cast("FormulaicErrors", importlib.import_module("formulaic.errors"))
FormulaMaterializer = cast(
    "FormulaMaterializerType",
    importlib.import_module("formulaic.materializers").FormulaMaterializer,
)
override = cast(
    "Override",
    importlib.import_module("interface_meta").override,
)
