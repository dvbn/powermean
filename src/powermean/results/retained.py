"""The state that a result retains, bound to one point, equation, sample and basis.

A :class:`RetainedFit` is formed once by :func:`powermean.results.handoff.bind` from the
records that the earlier stages made, and never changed afterwards.  What it holds, and
why:

- the estimator (mean order, working variance, moment, formulation) and the requested
  inference, as records; the resolved numerical settings as the storage-only snapshot
  of :class:`~powermean.data.computation.Computation`;
- the sample of the fit (:class:`~powermean.data.retained.OrderSample`): the selected
  rows with their original positions and accounting, the fingerprint, the numeric
  record of the retained rows (the original outcome, the unscaled trained design, the
  raw log-power-mean offset and the typed weights with their kind), the transformed
  problem, the scaled design with its column scale, and the requested classifications
  as raw typed columns; the user's row labels of every input row, owned, kept apart
  from the positional keys, and the columns that the preparation read;
- the parsed formula and the owned fitted right-hand side (the
  :class:`~powermean.formula.frame.Design` that the prepared frame hands out, with its
  fitted ``ModelSpec``, its terms, the columns it read and the Formulaic version), for
  the frozen replay of a prediction; the design is never handed out;
- the authentic certificate as supplied, with its fresh evaluation at the point (the
  row residuals, the observed curvature and the expected information, the raw
  weights, the sums ``S``, ``C`` and ``H``, the contributions and the units), the
  solver's candidate and native reason when a conclusion was handed over, and the
  status composed by the status rules; on the linear route, the candidate of the
  least-squares solve with its residuals and factorization and
  the fresh :class:`~powermean.solve.linear_certificate.LinearCertificate` of its
  point with the status of :mod:`powermean.solve.linear_status` where a conclusion
  was handed over (a bare candidate keeps no certificate and no status);
  or, on the power route, the :class:`~powermean.solve.no_point.StartFailure` of an
  automatic start that had no finite value, with its status and no point;
- on the linear route, the
  :class:`~powermean.inference.hc.LinearSolve` that the dispatch formed before its
  solve (the owned equation as the solve received it, paired with the candidate that
  solve returned), where one was handed over and the influence is kept: the owner
  that the named HC covariances of the linear path read
  (:mod:`powermean.results.retained_hc`); never formed after the fit;
- the start requested on the reported scale, where the caller gave one, beside the
  solver's start in its own coordinates in the candidate; and the fitted design of
  an offset or exposure expression, for its frozen replay on prediction rows;
- the influence record of the point (:class:`~powermean.inference.contributions.
  Influence`): the primitive scores, the actual sum-moment Jacobian and its map, the
  raw weights, the units, the kind of the weights and the statistical count, in the
  coordinates of the solver; discarded when ``keep_influence`` is false, after the
  default covariance was formed, and never reconstructed from the certificate's
  evaluation, which keeps those quantities as diagnostics of the certificate;
- the default covariance (:class:`CovarianceRecord`): the one requested at the fit,
  in the coordinates of the solver and on the reported scale, with its method, bread,
  factor, counts, rank, diagnostics and convention; for a named HC method, the
  convention applied to the counts (requested and in force, its source, the factor
  with its owner, the reference with its degrees of freedom) and the leverage
  geometry, and no row array, so that ``keep_influence=False`` discards the rows; for
  a typed cluster request of the expanded menu, the same applied convention and the
  typed answer of the convention engine (:mod:`powermean.results.retained_inference`):
  the bound request, the multiway components with their signs, cluster counts and
  factors, the raw port estimate with its diagnostics, the ``psd`` treatment and,
  where it ran, its adjustment and frame records; no row array either.

Counts are kept apart: the input rows, the retained rows that enter the estimating
equation, the statistical count ``N`` (the exact total of frequency counts, else the
rows of positive weight) and the clusters ``G`` of a covariance.  The arrays of the
records are owned and read-only; the record compares by identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from powermean._prepare.compose import Columns
from powermean.data.computation import Computation
from powermean.data.inference_conventions import ConventionValues
from powermean.data.inference_requests import HCMethod
from powermean.data.request_values import Omitted
from powermean.data.retained import OrderSample
from powermean.data.specs import Estimator, Inference
from powermean.formula.frame import Design
from powermean.formula.grammar import FormulaParts
from powermean.inference.contributions import Influence
from powermean.inference.convention import Kind
from powermean.inference.hc import LinearSolve
from powermean.inference.ssc import AppliedConvention, ReferenceDistribution
from powermean.model.variance import FloatArray
from powermean.solve.certificate import Certificate, Level
from powermean.solve.iols.candidate import IolsCandidate
from powermean.solve.joint_root_records import JointRootCandidate
from powermean.solve.linear import LinearCandidate
from powermean.solve.linear_certificate import LinearCertificate
from powermean.solve.newton_trace import NativeReason, NewtonCandidate
from powermean.solve.no_point import StartFailure
from powermean.solve.status import Status

if TYPE_CHECKING:
    from powermean.results.retained_inference import RetainedInference

AnyCertificate = Certificate | LinearCertificate
AnyCandidate = NewtonCandidate | IolsCandidate | JointRootCandidate

Coordinate = Literal["reported"]
Bread = Literal["observed", "expected"]
Convention = str | ConventionValues
Reference = Literal["normal", "t"]


@dataclass(frozen=True, eq=False)
class CovarianceRecord:
    """One covariance of the fitted point, in both coordinate systems.

    Attributes
    ----------
    kind : {"robust", "cluster", "hc"}
        The method: the robust estimator, a cluster covariance, or a named HC
        method.
    cluster : str or None
        The name of the retained classification clustered on, the requested
        dimensions joined by ``" + "`` for several, or ``None``.
    theta : array
        ``V`` of the coefficients of the solver, exactly symmetric, with the factor of
        the convention applied once (:mod:`powermean.inference.convention`), or the
        method's own factor once for a named HC method.
    reported : array
        ``V`` on the reported scale, each entry mapped once by the range-safe
        order scale: ``V_jk / (d_j d_k) / omega^2``; for a signed multiway request
        whose ``psd`` treatment is ``"cgm"``, the final matrix of that operation.
    standard_errors : array
        ``sqrt`` of the reported variances, NaN where a variance is negative and
        zero where it is zero.
    factor : float or tuple of float
        ``N/(N - 1)`` or ``G/(G - 1)``; a named method's own factor, ``N/(N - K)``
        for HC1 and one otherwise; the convention's one factor of a cluster request,
        or, for a multiway request under ``cluster_count="each"``, the factor of each
        component, in the order of the port's components.
    count : int
        ``N``, the statistical count of the convention.
    clusters : int or None
        ``G``, for one cluster dimension; ``G_min``, the smallest count over the
        requested dimensions, for several.
    rank : int
        The rank of the primitive's estimate, by its own tolerance.
    diagnostics : tuple of str
        The diagnostics of the primitive: singular, zero, indefinite or negative
        diagonal, as it reported them; nothing was repaired.  For a signed multiway
        request whose ``psd`` adjustment was active, the notes that describe the
        reported matrix (``covariance.psd_adjusted``, and ``covariance.indefinite``
        where the final matrix is not positive semidefinite); the raw estimate's
        diagnostics stay in ``answer``.
    bread, coordinate, convention, reference
        The bread, the reported coordinate of ``reported``, the convention in force
        (a preset name, or the custom record of a named HC request or of a typed
        cluster request) and the reference: the observed bread, ``"stata"`` and the
        normal reference for the default robust and cluster covariances.
    method : HC method or None
        The named HC method, or ``None``.
    applied : AppliedConvention or None
        For a named HC or a recomputed conventional request, its applied convention:
        and the resolved convention with its source, ``N`` and the exact rank ``K``,
        the factor with its owner, the reference with its degrees of freedom;
        a record of integers and strings, without an array.
    hat_geometry : str or None
        For a named HC method, the geometry of its leverage: the expected
        information, beside the bread of the influence rows.
    answer : RetainedInference or None
        For a request answered by the convention engine
        (:mod:`powermean.results.retained_inference`), its typed answer: the bound
        request, the multiway components with their signs, counts and factors, the
        raw port estimate with its diagnostics, the ``psd`` treatment and, where it
        ran, its adjustment and frame records; ``None`` otherwise.
    """

    kind: Kind
    cluster: str | None
    theta: FloatArray
    reported: FloatArray
    standard_errors: FloatArray
    factor: float | tuple[float, ...]
    count: int
    clusters: int | None
    rank: int
    diagnostics: tuple[str, ...]
    bread: Bread = "observed"
    coordinate: Coordinate = "reported"
    convention: Convention = "stata"
    reference: Reference = "normal"
    method: HCMethod | None = None
    applied: AppliedConvention | None = None
    hat_geometry: str | None = None
    answer: RetainedInference | None = None

    def __post_init__(self) -> None:
        """Named HC records carry method and hats; conventional records never do.

        A typed answer belongs to a robust or cluster record with its applied
        convention.
        """
        named = (self.method, self.hat_geometry)
        present = [value is not None for value in named]
        if present != [self.kind == "hc"] * len(present) or (
            self.kind == "hc" and self.applied is None
        ):
            raise ValueError(
                "a named HC covariance records its method, its applied convention "
                "and its leverage geometry; other covariances name no HC method"
            )
        if self.answer is not None and (self.kind == "hc" or self.applied is None):
            raise ValueError(
                "a typed answer belongs to a robust or cluster covariance with its "
                "applied convention"
            )

    @property
    def degenerate(self) -> bool:
        """Whether a reported standard error is exactly zero."""
        return bool((self.standard_errors == 0).any())

    @property
    def distribution(self) -> ReferenceDistribution:
        """The reference distribution, with the degrees of freedom of a ``t``."""
        if self.applied is not None:
            return self.applied.reference
        return ReferenceDistribution(self.reference, None)

    def to_dict(self) -> dict[str, object]:
        """The method, factors, counts and convention, without the matrices.

        An applied convention adds its records: the method, the convention in force (its
        preset name, or ``"custom"``) with its values, the requested ``ssc`` (``None``
        where the request gave none) and the source of the convention in force, the
        factor's formula and owner, the exact rank ``K``, the degrees of freedom of a
        ``t`` reference and the leverage geometry for named HC.  A multiway answer adds
        its form, the cluster count of each requested dimension and the ``psd``
        treatment resolved for it (``None`` where it does not apply).  Unchanged default
        records retain their schema.
        """
        record: dict[str, object] = {
            "kind": self.kind,
            "cluster": self.cluster,
            "factor": self.factor,
            "count": self.count,
            "clusters": self.clusters,
            "rank": self.rank,
            "diagnostics": list(self.diagnostics),
            "bread": self.bread,
            "coordinate": self.coordinate,
            "convention": self.convention,
            "reference": self.reference,
            "degenerate": self.degenerate,
        }
        if self.applied is not None:
            record.update(self._named(self.applied))
        answer = self.answer
        if answer is not None and answer.multiway is not None:
            record.update(
                multiway=answer.multiway.form,
                cluster_counts=list(answer.applied.counts.clusters),
                psd=answer.psd,
            )
        return record

    def _named(self, applied: AppliedConvention) -> dict[str, object]:
        """The applied convention's records, as plain data.

        The factor is the convention's one factor or, for a multiway request whose
        components take their own counts, the factor of its requested dimensions,
        which share one formula and one owner.
        """
        resolved, factor = applied.convention, applied.factor
        if factor is None and applied.components:
            factor = applied.components[0]
        if factor is None:
            raise ValueError("a recorded covariance applies its factors")
        requested = resolved.requested
        requested_ssc: object = requested
        if isinstance(requested, Omitted):
            requested_ssc = None
        elif isinstance(requested, ConventionValues):
            requested_ssc = requested.to_dict()
        return {
            "method": self.method,
            "convention": resolved.preset if resolved.origin == "preset" else "custom",
            "convention_values": resolved.values.to_dict(),
            "requested_ssc": requested_ssc,
            "ssc_source": resolved.source,
            "factor_formula": factor.formula,
            "factor_owner": factor.owner,
            "design_rank": applied.counts.rank,
            "reference_df": applied.reference.df,
            "hat_geometry": self.hat_geometry,
        }


@dataclass(frozen=True, eq=False)
class RetainedFit:
    """What a result holds, bound once (module docstring).

    ``theta`` is the point in the coordinates of the solver, or ``None`` where the
    linear path formed no coefficients or the automatic start failed; ``status`` is the
    status of the solve, or of the supplied point by the gates of its certificate, that
    of the linear conclusion or of the start failure, and ``None`` for a bare linear
    candidate handed over without its conclusion; ``reason`` is the native reason of the
    retained attempt of a Newton conclusion (of the target stage of a joint root
    conclusion outside ``[1, 2]``, whose candidate also records its initializer, merit
    metric and stages).  ``certificate`` is the joint certificate on the power route and
    the fresh linear certificate on the linear route, ``failure`` the record of an
    automatic start that had no finite value, ``requested_start`` the start on the
    reported scale where one was given, and ``offset_design`` the fitted design of an
    offset or exposure expression.  ``influence`` is ``None`` where it was discarded or
    could not be formed, ``default`` where no covariance was formed.  ``linear_solve``
    is the captured equation and solve of the linear route, ``None`` on the power route,
    for a candidate handed over without it, or after ``keep_influence=False``.
    """

    estimator: Estimator
    inference: Inference
    computation: Computation
    sample: OrderSample
    labels: pd.Index[Any]
    columns: Columns
    subset_missing: int
    parts: FormulaParts
    design: Design
    certificate: AnyCertificate | None
    candidate: AnyCandidate | None
    linear: LinearCandidate | None
    reason: NativeReason | None
    status: Status | None
    theta: FloatArray | None
    influence: Influence | None
    default: CovarianceRecord | None
    keep_influence: bool
    failure: StartFailure | None = None
    requested_start: FloatArray | None = None
    offset_design: Design | None = None
    linear_solve: LinearSolve | None = None

    @property
    def level(self) -> Level | None:
        """The level at which the certificate passes, if one does."""
        return None if self.certificate is None else self.certificate.level

    @property
    def certified(self) -> bool:
        """Whether an authentic certificate accepts the point at either level."""
        return self.certificate is not None and self.certificate.accepted

    @property
    def names(self) -> tuple[str, ...]:
        """The names of the coefficients: the columns of the design, in order."""
        return self.sample.data.column_names
