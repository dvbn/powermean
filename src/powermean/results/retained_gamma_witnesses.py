"""Detached snapshots of GPML recession raw checks, identities, and witnesses."""

from __future__ import annotations

from dataclasses import dataclass

from powermean.model.variance import FloatArray
from powermean.results.absorbed_partitions import PartitionIdentityMap
from powermean.results.owned import _optional, owned_array
from powermean.results.retained_diagnostic_problems import (
    MarginalTuple,
    RetainedGammaProblem,
    _owned_marginals,
)
from powermean.results.retained_input_values import ProblemKey, _tuple_arrays
from powermean.results.retained_inputs import ProblemIdentityMap
from powermean.results.retained_support_witnesses import RetainedColumnSums
from powermean.solve.support_gamma_lp import RawCheck, RawValues, Stage
from powermean.solve.support_gamma_witness import (
    GammaDual,
    GammaWitness,
    Identities,
    Multipliers,
)
from powermean.solve.support_lp import LPControls

__all__ = [
    "RetainedGammaDual",
    "RetainedGammaIdentities",
    "RetainedGammaMultipliers",
    "RetainedGammaWitness",
    "RetainedRawCheck",
    "RetainedRawValues",
    "_detach_gamma_dual",
    "_detach_gamma_witness",
    "_detach_raw_check",
    "_detach_raw_values",
]


@dataclass(frozen=True)
class RetainedRawValues:
    """Detached snapshot of ``RawValues``."""

    activity: float | None
    balance: float | None
    objective: float
    scale: float
    agreement: float

    @classmethod
    def from_values(cls, v: RawValues) -> RetainedRawValues:
        """Detach a ``RawValues`` record."""
        act = None if v.activity is None else float(v.activity)
        bal = None if v.balance is None else float(v.balance)
        return cls(act, bal, float(v.objective), float(v.scale), float(v.agreement))


def _detach_raw_values(v: RawValues | None) -> RetainedRawValues | None:
    """Detach an optional ``RawValues`` record."""
    return None if v is None else RetainedRawValues.from_values(v)


@dataclass(frozen=True)
class RetainedRawCheck:
    """Detached snapshot of ``RawCheck``."""

    failure: str
    box: float
    values: RetainedRawValues | None = None

    @classmethod
    def from_check(cls, c: RawCheck) -> RetainedRawCheck:
        """Detach a ``RawCheck`` record."""
        return cls(str(c.failure), float(c.box), _detach_raw_values(c.values))


def _detach_raw_check(c: RawCheck | None) -> RetainedRawCheck | None:
    """Detach an optional ``RawCheck`` record."""
    return None if c is None else RetainedRawCheck.from_check(c)


@dataclass(frozen=True)
class RetainedGammaIdentities:
    """Detached snapshot of ``Identities`` with owned arrays on every access."""

    _lower: FloatArray
    _upper: FloatArray
    cone: RetainedColumnSums
    stationarity: RetainedColumnSums

    def __post_init__(self) -> None:
        """Own all stored numeric arrays."""
        object.__setattr__(self, "_lower", owned_array(self._lower))
        object.__setattr__(self, "_upper", owned_array(self._upper))

    @classmethod
    def from_identities(cls, ident: Identities) -> RetainedGammaIdentities:
        """Detach an ``Identities`` record."""
        c = RetainedColumnSums.from_sums(ident.cone)
        st = RetainedColumnSums.from_sums(ident.stationarity)
        return cls(ident.lower, ident.upper, c, st)

    @property
    def lower(self) -> FloatArray:
        """Return a fresh owned copy of ``lower``."""
        return owned_array(self._lower)

    @property
    def upper(self) -> FloatArray:
        """Return a fresh owned copy of ``upper``."""
        return owned_array(self._upper)


@dataclass(frozen=True)
class RetainedGammaMultipliers:
    """Detached snapshot of ``Multipliers`` with owned arrays on every access."""

    _cone: FloatArray
    nu: float
    _lower: FloatArray
    _upper: FloatArray

    def __post_init__(self) -> None:
        """Own all stored numeric arrays."""
        object.__setattr__(self, "_cone", owned_array(self._cone))
        object.__setattr__(self, "_lower", owned_array(self._lower))
        object.__setattr__(self, "_upper", owned_array(self._upper))

    @classmethod
    def from_multipliers(cls, mult: Multipliers) -> RetainedGammaMultipliers:
        """Detach a ``Multipliers`` record."""
        return cls(mult.cone, float(mult.nu), mult.lower, mult.upper)

    @property
    def cone(self) -> FloatArray:
        """Return a fresh owned copy of ``cone``."""
        return owned_array(self._cone)

    @property
    def lower(self) -> FloatArray:
        """Return a fresh owned copy of ``lower``."""
        return owned_array(self._lower)

    @property
    def upper(self) -> FloatArray:
        """Return a fresh owned copy of ``upper``."""
        return owned_array(self._upper)


@dataclass(frozen=True)
class RetainedGammaWitness:
    """Detached snapshot of ``GammaWitness`` preserving optional failed/raw fields."""

    problem_key: ProblemKey
    problem: RetainedGammaProblem
    controls: LPControls
    stage: Stage
    _boxed: FloatArray
    reported: float
    failure: tuple[str, ...]
    check: RetainedRawCheck | None = None
    _mapped_slopes: FloatArray | None = None
    _mapped_effects: tuple[FloatArray, ...] | None = None
    _raw: FloatArray | None = None
    scale: float | None = None
    _slopes: FloatArray | None = None
    _effects: tuple[FloatArray, ...] | None = None
    _h: FloatArray | None = None
    total: float | None = None
    positive_min: float | None = None
    positive_max: float | None = None

    def __post_init__(self) -> None:
        """Own all stored numeric arrays."""
        object.__setattr__(self, "_boxed", owned_array(self._boxed))
        for name in ("_mapped_slopes", "_raw", "_slopes", "_h"):
            object.__setattr__(self, name, _optional(getattr(self, name)))
        for name in ("_mapped_effects", "_effects"):
            val = getattr(self, name)
            if val is not None:
                object.__setattr__(self, name, _tuple_arrays(val))

    @classmethod
    def from_witness(
        cls,
        w: GammaWitness,
        *,
        problem_snapshot: RetainedGammaProblem | None = None,
        problem_map: ProblemIdentityMap | None = None,
        partition_map: PartitionIdentityMap | None = None,
    ) -> RetainedGammaWitness:
        """Detach a ``GammaWitness`` without re-running witness checks."""
        if problem_map is not None or problem_snapshot is None:
            rp = None if problem_map is None else problem_map.register_gamma
            dg = None if partition_map is None else partition_map.register_grouping
            prob = RetainedGammaProblem.from_problem(
                w.problem, register_problem=rp, detach_group=dg
            )
        else:
            prob = problem_snapshot
        sc = None if w.scale is None else float(w.scale)
        tot = None if w.total is None else float(w.total)
        pmin = None if w.positive_min is None else float(w.positive_min)
        pmax = None if w.positive_max is None else float(w.positive_max)
        fails, chk = tuple(str(f) for f in w.failure), _detach_raw_check(w.check)
        ms, me, rw = w.mapped_slopes, w.mapped_effects, w.raw
        tail = (ms, me, rw, sc, w.slopes, w.effects, w.h, tot, pmin, pmax)
        return cls(
            prob.key,
            prob,
            w.controls,
            w.stage,
            w.boxed,
            float(w.reported),
            fails,
            chk,
            *tail,
        )

    @property
    def boxed(self) -> FloatArray:
        """Return a fresh owned copy of ``boxed``."""
        return owned_array(self._boxed)

    @property
    def mapped_slopes(self) -> FloatArray | None:
        """Return a fresh owned copy of ``mapped_slopes``, or ``None``."""
        return _optional(self._mapped_slopes)

    @property
    def mapped_effects(self) -> tuple[FloatArray, ...] | None:
        """Return fresh owned copies of ``mapped_effects``, or ``None``."""
        me = self._mapped_effects
        return None if me is None else _tuple_arrays(me)

    @property
    def raw(self) -> FloatArray | None:
        """Return a fresh owned copy of ``raw``, or ``None``."""
        return _optional(self._raw)

    @property
    def slopes(self) -> FloatArray | None:
        """Return a fresh owned copy of ``slopes``, or ``None``."""
        return _optional(self._slopes)

    @property
    def beta(self) -> FloatArray | None:
        """Alias for ``slopes``."""
        return self.slopes

    @property
    def effects(self) -> tuple[FloatArray, ...] | None:
        """Return fresh owned copies of ``effects``, or ``None``."""
        return None if self._effects is None else _tuple_arrays(self._effects)

    @property
    def coefficients(self) -> tuple[FloatArray, ...] | None:
        """Alias for ``effects``."""
        return self.effects

    @property
    def h(self) -> FloatArray | None:
        """Return a fresh owned copy of ``h``, or ``None``."""
        return _optional(self._h)

    @property
    def sum_h(self) -> float | None:
        """Alias for ``total``."""
        return self.total

    @property
    def valid(self) -> bool:
        """Return True if the witness has no failures."""
        return not self.failure

    @property
    def kind(self) -> str | None:
        """Return the recession witness kind when valid, else ``None``."""
        if not self.valid:
            return None
        neg = self.stage == "negative"
        return "negative_sum_recession" if neg else "zero_sum_recession"


def _detach_gamma_witness(
    w: GammaWitness | None,
    *,
    problem_snapshot: RetainedGammaProblem | None = None,
    problem_map: ProblemIdentityMap | None = None,
    partition_map: PartitionIdentityMap | None = None,
) -> RetainedGammaWitness | None:
    """Detach an optional ``GammaWitness`` record."""
    if w is None:
        return None
    return RetainedGammaWitness.from_witness(
        w,
        problem_snapshot=problem_snapshot,
        problem_map=problem_map,
        partition_map=partition_map,
    )


@dataclass(frozen=True)
class RetainedGammaDual:
    """Detached snapshot of ``GammaDual`` preserving optional failed/raw fields."""

    problem_key: ProblemKey
    problem: RetainedGammaProblem
    controls: LPControls
    stage: Stage
    _boxed: FloatArray
    reported: float
    _marginals: MarginalTuple
    failure: tuple[str, ...]
    check: RetainedRawCheck | None = None
    multipliers: RetainedGammaMultipliers | None = None
    bound: float | None = None
    gap: float | None = None
    ratios: tuple[float, float] | None = None
    columns: tuple[RetainedGammaIdentities, ...] = ()

    def __post_init__(self) -> None:
        """Own all stored numeric arrays."""
        object.__setattr__(self, "_boxed", owned_array(self._boxed))
        owned_m = _owned_marginals(self._marginals)
        if owned_m is not None:
            object.__setattr__(self, "_marginals", owned_m)

    @classmethod
    def from_dual(
        cls,
        d: GammaDual,
        *,
        problem_snapshot: RetainedGammaProblem | None = None,
        problem_map: ProblemIdentityMap | None = None,
        partition_map: PartitionIdentityMap | None = None,
    ) -> RetainedGammaDual:
        """Detach a ``GammaDual`` without re-running dual checks."""
        if problem_map is not None or problem_snapshot is None:
            rp = None if problem_map is None else problem_map.register_gamma
            dg = None if partition_map is None else partition_map.register_grouping
            prob = RetainedGammaProblem.from_problem(
                d.problem, register_problem=rp, detach_group=dg
            )
        else:
            prob = problem_snapshot
        m_raw = d.multipliers
        mult = (
            None if m_raw is None else RetainedGammaMultipliers.from_multipliers(m_raw)
        )
        cols = tuple(RetainedGammaIdentities.from_identities(c) for c in d.columns)
        rats = None if d.ratios is None else (float(d.ratios[0]), float(d.ratios[1]))
        bnd = None if d.bound is None else float(d.bound)
        gp = None if d.gap is None else float(d.gap)
        head = (prob.key, prob, d.controls, d.stage, d.boxed, float(d.reported))
        fails = tuple(str(f) for f in d.failure)
        tail = (d.marginals, fails, _detach_raw_check(d.check))
        return cls(*head, *tail, mult, bnd, gp, rats, cols)

    @property
    def boxed(self) -> FloatArray:
        """Return a fresh owned copy of ``boxed``."""
        return owned_array(self._boxed)

    @property
    def marginals(self) -> MarginalTuple:
        """Return fresh owned copies of ``marginals``."""
        owned_m = _owned_marginals(self._marginals)
        assert owned_m is not None
        return owned_m

    @property
    def passed(self) -> bool:
        """Return True if the dual check has no failures."""
        return not self.failure


def _detach_gamma_dual(
    d: GammaDual | None,
    *,
    problem_snapshot: RetainedGammaProblem | None = None,
    problem_map: ProblemIdentityMap | None = None,
    partition_map: PartitionIdentityMap | None = None,
) -> RetainedGammaDual | None:
    """Detach an optional ``GammaDual`` record."""
    if d is None:
        return None
    return RetainedGammaDual.from_dual(
        d,
        problem_snapshot=problem_snapshot,
        problem_map=problem_map,
        partition_map=partition_map,
    )
