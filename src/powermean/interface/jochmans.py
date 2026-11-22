"""Formula entry point for the two exogenous Jochmans equations."""

from __future__ import annotations

from hashlib import sha256
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from powermean import messages
from powermean.errors import SampleError, SpecificationError
from powermean.formula.formulaic_api import formulaic, formulaic_errors
from powermean.formula.frame import Design, prediction_template
from powermean.formula.grammar import FormulaParts, collapsed_powers, parse_formula
from powermean.formula.materialization import materialize_formula
from powermean.formula.names import columns_read
from powermean.formula.validity import columns_invalid
from powermean.results.jochmans import JochmansResult, _Sample, covariance_convention
from powermean.results.two_way_prediction import GMMIndex, TwoWayPrediction
from powermean.solve.jochmans import DifferenceMoments, FloatArray, _failure, solve

IntArray = NDArray[np.intp]


def _invalid(argument: str, value: object, valid: str) -> SpecificationError:
    """The standard public option error."""
    return SpecificationError(
        "specification.invalid_value",
        {"argument": argument, "value": value, "valid": valid},
    )


def _sample_error(detail: str) -> SampleError:
    """Explain which required part of the observed array failed."""
    return SampleError("sample.jochmans", {"detail": detail})


def _numeric(value: ArrayLike) -> FloatArray:
    """Read a real numerical column or matrix without discarding imaginary values."""
    array = np.asarray(value)
    if np.iscomplexobj(array) or (
        array.dtype.kind == "O"
        and any(isinstance(item, (complex, np.complexfloating)) for item in array.flat)
    ):
        raise ValueError("complex values")
    return np.asarray(array, dtype=np.float64)


def _start(start: ArrayLike | None, columns: int) -> FloatArray:
    """Read a coefficient start in displayed design order."""
    if start is None:
        return np.zeros(columns)
    requirement = f"a finite real vector of length {columns} in coefficient order"
    try:
        values = _numeric(start)
    except (TypeError, ValueError, OverflowError) as error:
        raise _invalid("start", start, requirement) from error
    if values.shape != (columns,) or not np.isfinite(values).all():
        raise _invalid("start", start, requirement)
    return values


def _layout(
    data: pd.DataFrame, effects: tuple[str, str], *, directed: bool
) -> tuple[IntArray, IntArray, tuple[pd.Index, pd.Index]]:
    """Encode observed pairs without sorting labels or introducing new rows."""
    if columns_invalid(data, effects).any():
        raise _sample_error("missing or non-finite fixed-effect identifiers")
    left, right = (data[name].to_numpy() for name in effects)
    try:
        if directed:
            both = np.concatenate(
                [data[name].to_numpy(dtype=object) for name in effects]
            )
            codes, levels = pd.factorize(both, sort=False)
            rows, cols = np.split(codes, 2)
            row_levels = col_levels = levels
        else:
            rows, row_levels = pd.factorize(left, sort=False)
            cols, col_levels = pd.factorize(right, sort=False)
    except TypeError as error:
        raise _sample_error(
            "fixed-effect identifiers must be hashable scalars"
        ) from error
    nr, nc = len(row_levels), len(col_levels)
    if directed and np.any(rows == cols):
        raise _sample_error("the directed layout excludes self-links")
    if len(np.unique(rows * nc + cols)) != len(data):
        raise _sample_error("duplicate row and column pairs")
    if directed:
        if nr < 4 or len(data) != nr * (nr - 1):
            raise _sample_error(
                "expected every non-self link among at least four common agents"
            )
    elif min(nr, nc) < 2 or len(data) != nr * nc:
        raise _sample_error(
            "expected every cell of a rectangle with at least two levels per block"
        )
    labels = (
        pd.Index(
            row_levels, dtype=object, name=effects[0], tupleize_cols=False, copy=True
        ),
        pd.Index(
            col_levels, dtype=object, name=effects[1], tupleize_cols=False, copy=True
        ),
    )
    return rows, cols, labels


def _design(
    parts: FormulaParts, data: pd.DataFrame
) -> tuple[FloatArray, FloatArray, Design]:
    """Materialize the Formulaic right-hand side once, preserving every row."""
    working = data.copy(deep=False)
    working.index = pd.RangeIndex(len(data))
    try:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            matrix = materialize_formula(parts.first_part, working)
    except formulaic_errors.FormulaicError as error:
        raise _invalid(
            "formula", parts.formula, "a materializable Formulaic formula"
        ) from error
    rhs = getattr(matrix, "rhs", None)
    spec = getattr(rhs, "model_spec", None)
    if not isinstance(rhs, pd.DataFrame) or not isinstance(spec, formulaic.ModelSpec):
        raise _invalid("formula", parts.formula, "one Formulaic right-hand side")
    names = tuple(str(name) for name in rhs.columns)
    if not names:
        raise _invalid("formula", parts.formula, "at least one slope")
    referenced = columns_read(spec, working.columns)
    if columns_invalid(working, (parts.outcome, *referenced)).any():
        raise _sample_error("missing or non-finite outcomes or regressors")
    try:
        y = _numeric(working[parts.outcome].to_numpy())
        x = _numeric(rhs.to_numpy())
    except (TypeError, ValueError, OverflowError) as error:
        raise _sample_error(
            "outcomes and evaluated regressors must be real numbers"
        ) from error
    if len(x) != len(y) or not np.isfinite(y).all() or not np.isfinite(x).all():
        raise _sample_error("missing or non-finite outcomes or regressors")
    count = int(np.count_nonzero(y < 0))
    if count:
        raise SampleError("sample.negative_outcome", {"count": count})
    return y, x, prediction_template(rhs, spec, tuple(referenced))


def _scale(
    x: FloatArray,
) -> tuple[
    FloatArray, FloatArray, tuple[NDArray[np.int32], FloatArray, FloatArray, FloatArray]
]:
    """Center and scale columns without overflowing or losing a common level."""
    _, exponent = np.frexp(np.max(np.abs(x), axis=0))
    with np.errstate(over="ignore", under="ignore"):
        scaled = np.ldexp(x, -exponent)
        shifted = scaled - scaled[0]
        mean = shifted.mean(axis=0)
        centered = shifted - mean
        amplitude = np.max(np.abs(centered), axis=0)
        amplitude[amplitude == 0] = 1.0
        scale = np.ldexp(amplitude, exponent)
    if (
        not np.isfinite(scale).all()
        or np.any(scale == 0)
        or np.any((x != 0) & (scaled == 0))
    ):
        raise _failure("numerical_domain", None)
    return centered / amplitude, scale, (exponent, scaled[0].copy(), mean, amplitude)


def _sample(
    data: pd.DataFrame,
    names: tuple[str, ...],
    y: FloatArray,
    x: FloatArray,
    *,
    rows: IntArray,
    cols: IntArray,
    layout: str,
) -> _Sample:
    """Record ordered sample inputs and cell memberships in a compact digest."""
    digest = sha256(repr((layout, names, x.shape)).encode("utf-8"))
    for array, dtype in ((rows, "<i8"), (cols, "<i8"), (y, "<f8"), (x, "<f8")):
        digest.update(np.ascontiguousarray(array, dtype=dtype))
    return _Sample(
        data.index.copy(deep=True),
        int(np.count_nonzero(y == 0)),
        digest.hexdigest(),
        names,
    )


def _formula(formula: str, data: pd.DataFrame) -> tuple[FormulaParts, tuple[str, str]]:
    """Resolve the exogenous formula and its two simple identifier columns."""
    if not isinstance(formula, str):
        raise _invalid("formula", formula, "a formula string")
    parts = parse_formula(formula)
    blocks = parts.fixed_effects
    if (
        parts.instruments is not None
        or len(blocks) != 2
        or any(len(block) != 1 for block in blocks)
        or blocks[0] == blocks[1]
    ):
        raise _invalid("formula", formula, "outcome ~ slopes | row + column")
    if collapsed_powers(parts.regressors):
        raise _invalid("formula", formula, "use I(x**p) for powers")
    effects = (blocks[0][0], blocks[1][0])
    if any(name not in data for name in (parts.outcome, *effects)):
        raise _invalid("formula", formula, "outcome and fixed-effect columns in data")
    return parts, effects


def fit_difference_gmm(
    formula: str,
    data: pd.DataFrame,
    *,
    estimator: Literal["gmm1", "gmm2"],
    fe_layout: Literal["rectangular", "directed"] = "rectangular",
    vcov: Literal["paper", "stata"] = "paper",
    start: ArrayLike | None = None,
    max_iter: int = 100,
    **unsupported: object,
) -> JochmansResult:
    """Fit an exogenous two-way exponential regression by difference moments.

    Parameters
    ----------
    formula : str
        For example, ``"y ~ 0 + x1 + x2 | exporter + importer"``.
        Supply two distinct fixed-effect identifiers after ``|`` and omit the
        intercept. The identifiers can be unit and period or the two agents
        in an ordered pair.
    data : pandas.DataFrame
        One row per observed cell. Outcomes are finite and nonnegative.
    estimator : {"gmm1", "gmm2"}
        The two distinct Jochmans and Verardi moment equations.
    fe_layout : {"rectangular", "directed"}, default "rectangular"
        Use ``"rectangular"`` for a complete crossing with at least two levels
        per axis (Stata's ``twexp``). A single rectangle cannot support covariance.
        Use ``"directed"`` for all ordered non-self
        links among at least four common agents (Stata's ``twgravity``).
    vcov : {"paper", "stata"}, default "paper"
        Use observed-cell normalization or apply the additional (n-1)/n
        multiplier in the directed layout used by Stata's twgravity.
    start : array_like, optional
        Finite slopes in Formulaic design order. Defaults to zero. To initialize
        GMM2 from GMM1, pass the first fit's ``coef()`` with the same formula.
    max_iter : int, default 100
        Maximum accepted Newton updates before the fit refuses.

    Returns
    -------
    JochmansResult
        The selected local root, sample, and full cell-projection covariance.

    Raises
    ------
    SpecificationError
        An invalid formula, option or start.
    SampleError
        Invalid values, duplicate cells or incomplete geometry.
    IdentificationError
        Insufficient resolved identifying design, support or Jacobian.
    ConvergenceError
        No certified local solution with usable covariance within numerical limits.

    Notes
    -----
    Observed zeros are retained. To restrict the sample, filter the data frame
    before fitting and retain a complete array on the selected levels.

    The moments assume exogeneity with respect to the full regressor array
    and the fixed effects, including the unit conditional products of errors
    used in the differences. Inference assumes conditionally independent
    observed cells and growing dimensions. See the Jochmans methods chapter.
    """
    if unsupported:
        raise SpecificationError(
            "specification.unknown_option",
            {
                "kind": "keyword of fit with a two-way GMM estimator",
                "key": next(iter(unsupported)),
                "valid": messages.format_keys(
                    ("estimator", "fe_layout", "vcov", "start", "max_iter")
                ),
            },
        )
    if not isinstance(estimator, str) or estimator not in ("gmm1", "gmm2"):
        raise _invalid("estimator", estimator, "'gmm1' or 'gmm2'")
    if not isinstance(fe_layout, str) or fe_layout not in ("rectangular", "directed"):
        raise _invalid("fe_layout", fe_layout, "'rectangular' or 'directed'")
    convention = covariance_convention(vcov)
    if type(max_iter) is not int or max_iter < 1:
        raise _invalid("max_iter", max_iter, "a positive integer")
    if not isinstance(data, pd.DataFrame) or not data.columns.is_unique:
        raise _invalid(
            "data", type(data).__name__, "a DataFrame with unique column names"
        )
    parts, effects = _formula(formula, data)
    if data.empty:
        raise SampleError("sample.empty", {"count": len(data)})
    directed = fe_layout == "directed"
    rows, cols, levels = _layout(data, effects, directed=directed)
    shape = (len(levels[0]), len(levels[1]))
    y, x, design = _design(parts, data)
    names = design.columns
    initial = _start(start, len(names))
    scaled, scale, coordinates = _scale(x)
    mask = np.zeros(shape, dtype=bool)
    mask[rows, cols] = True
    yg = np.zeros(shape)
    yg[rows, cols] = y
    xg = np.zeros((*shape, len(names)))
    xg[rows, cols] = scaled
    moments = DifferenceMoments(yg, xg, mask, gmm1=estimator == "gmm1")
    theta, state, root = solve(moments, initial, scale, max_iter)
    with np.errstate(over="ignore", under="ignore"):
        beta = theta / scale
    if not np.isfinite(beta).all() or np.any((theta != 0) & (beta == 0)):
        raise _failure("numerical_domain", beta)
    try:
        paper = moments.covariance(state, scale)
    except FloatingPointError as error:
        raise _failure("unusable_covariance", beta, certificate=root) from error
    factor = (shape[0] - 1) / shape[0] if directed else 1.0
    sample = _sample(data, names, y, x, rows=rows, cols=cols, layout=fe_layout)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        index = scaled @ theta
        common = float(x[0] @ beta - index[0])
    projection = GMMIndex(*coordinates, theta)
    prediction = TwoWayPrediction(
        y, index, rows, cols, levels, design, projection, common
    )
    return JochmansResult(
        beta,
        paper,
        factor,
        convention,
        sample,
        root,
        formula,
        estimator,
        fe_layout,
        shape,
        effects,
        prediction,
    )
