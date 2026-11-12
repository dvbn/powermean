"""Owned copies of the records that a result retains and of those it hands out.

Owned immutable arrays may be shared across records.  The records of the earlier stages
keep their arrays in bytes of their own, which NumPy refuses to write or to make
writable again.  Immutable bytes do not freeze the metadata of the array object that
views them: ``array.shape = (1, 1)`` succeeds on such an array, and so does an
assignment to its ``dtype`` or ``strides``, and every holder of that object sees the
change.  A record is bound to its arrays by identity, so a caller that keeps the object
it handed to the result, or one that the result handed out, can change what the result
reads afterwards: the labels of a sample through the shape of its positions, the
coefficients through the shape of the certificate's candidate.

So the result layer never retains an array object that another holder has, and never
hands one out: :func:`owned_array` forms, for an array, a fresh object of the same
dtype, shape and bytes. Complete immutable byte storage may be shared; mutable,
noncontiguous and subclass inputs retain their original copying path. The functions
below re-form each
record of the handoff with such arrays, field by field, through the record's own
constructor, with every scalar, tuple, mapping and nested record kept as it is.  The
values, the types and the diagnostics of a record are unchanged; its identity, and
that of each of its arrays, is new.  :func:`powermean.results.handoff.bind` re-forms
the sample, the certificate, the candidate of the solve and the linear candidate once,
before it checks and retains them, and :class:`~powermean.results.result.Result`
re-forms the certificate and the last state on every access and in every refusal that
carries them, so that an edit of the caller's record, or of a record handed out,
reaches nothing retained.  Object columns of the sample's metadata, whose elements
are no bytes, are re-formed by the owned selection of
:class:`~powermean.data.raw.AlignedColumns`.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from powermean._prepare.compose import _owned_index, _owned_table
from powermean.data.accounting import MissingAccount
from powermean.data.inputs import RegressionData
from powermean.data.numeric_fit import _immutable_bytes
from powermean.data.retained import OrderSample
from powermean.data.weights import ObservationWeights
from powermean.fe._cancel_storage import FloatStorage
from powermean.formula.frame import Design
from powermean.model.order import Selection
from powermean.model.order_scale import OrderScale
from powermean.results.retained import AnyCandidate, AnyCertificate
from powermean.solve.certificate import Certificate
from powermean.solve.iols.adapter import (
    IOLSAdapterResult,
    TransformedNormalizationState,
)
from powermean.solve.iols.candidate import IolsCandidate
from powermean.solve.iols.direct_route import IOLSDirectNoFEResult
from powermean.solve.iols.records import owned_value
from powermean.solve.joint_root_records import JointRootCandidate
from powermean.solve.linear import LinearCandidate
from powermean.solve.linear_certificate import LinearCertificate
from powermean.solve.linear_evaluation import LinearEvaluation
from powermean.solve.newton_trace import AcceptedState, NewtonAttempt, NewtonCandidate
from powermean.solve.reported import CoefficientCheck, ReportedScale
from powermean.solve.starts import Start
from powermean.systems.joint import JointEvaluation, SamplingUnits

AnyArray = NDArray[Any]


def owned_array(array: AnyArray) -> AnyArray:
    """A fresh read-only array with the same dtype, shape and immutable byte values.

    Complete byte-owned storage can be shared, never the source array header.
    Every other storage/subclass takes the original ``tobytes`` copying path.

    Parameters
    ----------
    array : numpy.ndarray
        An array of a plain dtype (numbers, Booleans, dates); an object array holds
        references, not values, and is refused.

    Returns
    -------
    numpy.ndarray
        A new array object over immutable bytes of its own, read-only, whose shape
        and dtype are those of ``array`` at the call.

    Raises
    ------
    TypeError
        An object array.
    """
    if array.dtype.kind == "O":
        raise TypeError("an object array has no bytes of its own to copy")
    owner = _immutable_bytes(array)
    stored: AnyArray = np.frombuffer(
        array.tobytes() if owner is None else owner, dtype=array.dtype
    )
    return stored.reshape(array.shape)


def _optional(array: AnyArray | None) -> AnyArray | None:
    """:func:`owned_array` of an array that may be absent."""
    return None if array is None else owned_array(array)


# ------------------------------------------------------------------- the sample


def _owned_data(data: RegressionData) -> RegressionData:
    """The numeric record of the retained rows, re-formed through its constructor.

    The record forms the offset of an exposure once, from the exposure; re-formed
    from the same exposure bytes, it is the same offset.
    """
    weights = None
    if data.weights is not None:
        weights = ObservationWeights(
            owned_array(data.weights.values), data.weights.kind
        )
    if data.exposure is not None:
        return RegressionData(
            owned_array(data.y),
            owned_array(data.design),
            exposure=owned_array(data.exposure),
            weights=weights,
            row_labels=data.row_labels,
            column_names=data.column_names,
        )
    return RegressionData(
        owned_array(data.y),
        owned_array(data.design),
        offset=owned_array(data.offset),
        weights=weights,
        row_labels=data.row_labels,
        column_names=data.column_names,
    )


def _owned_missing(missing: MissingAccount) -> MissingAccount:
    """The diagnostics of step 2 with owned masks."""
    return MissingAccount._owned(
        {group: owned_array(mask) for group, mask in missing.groups.items()},
        owned_array(missing.excluded),
    )


def _owned_selection(selection: Selection) -> Selection:
    """The selection of a transformed problem with owned arrays."""
    return replace(
        selection,
        mask=owned_array(selection.mask),
        positions=owned_array(selection.positions),
        removed=owned_array(selection.removed),
    )


def owned_sample(sample: OrderSample) -> OrderSample:
    """The sample of a mean order with every array, column and nested record owned.

    The positions, the inverse map, the codes, the masks of step 2, the numeric
    record, the transformed problem with its selection, the scaled design and the
    aligned metadata (by the owned selection of every retained row, so that the
    elements of an object column are fresh copies as well) are re-formed; the
    order, the flags and the fingerprint are kept.
    """
    problem = sample.problem
    count = np.arange(sample.count, dtype=np.intp)
    return replace(
        sample,
        positions=owned_array(sample.positions),
        retained_index=owned_array(sample.retained_index),
        codes=owned_array(sample.codes),
        missing=_owned_missing(sample.missing),
        data=_owned_data(sample.data),
        problem=replace(
            problem,
            response=owned_array(problem.response),
            offset=owned_array(problem.offset),
            selection=_owned_selection(problem.selection),
        ),
        design=replace(
            sample.design,
            matrix=owned_array(sample.design.matrix),
            scale=owned_array(sample.design.scale),
        ),
        metadata=sample.metadata.take(count),
    )


def owned_design(design: Design) -> Design:
    """The fitted right-hand side of the frame with its two masks owned.

    The matrix, the labels and the fitted spec of the design that
    :attr:`~powermean._prepare.compose.PreparedFrame.frame` hands out are copies of
    its own already; the masks of the input rows are shared with the caller's frame.
    A design that no prepared frame copied is owned by :func:`owned_expression_design`.
    """
    return replace(
        design,
        referenced_invalid=owned_array(design.referenced_invalid),
        evaluated_invalid=owned_array(design.evaluated_invalid),
    )


def owned_expression_design(design: Design) -> Design:
    """A fitted design handed over directly, every mutable part of it owned.

    The design of an offset or exposure expression reaches the result from the
    interface as the object the preparation evaluated, which the caller may hold:
    no prepared frame copied it.  Its fitted ``ModelSpec`` is mutable (the states of
    its stateful transforms, a learned center say, decide every replay), and so are
    its matrix, the axes and labels of the matrix, and its row labels.  Each is
    re-formed by the mechanisms that the prepared frame uses for the right-hand
    side (:func:`powermean._prepare.compose._owned_frame`): the spec by
    ``copy.deepcopy``, with the states it learned on the full input and nothing
    learned again; the matrix by the owned table, its two axes included; the labels
    by the owned Index; the masks by :func:`owned_array`.  The terms, the variables
    and the version are immutable and shared.  A later edit of the caller's design,
    through the object it kept, reaches nothing retained, and the replay of a
    prediction reads the states as they were at the fit.
    """
    return Design(
        matrix=_owned_table(design.matrix),
        spec=copy.deepcopy(design.spec),
        terms=design.terms,
        variables=design.variables,
        labels=_owned_index(design.labels),
        referenced_invalid=owned_array(design.referenced_invalid),
        evaluated_invalid=owned_array(design.evaluated_invalid),
        formulaic_version=design.formulaic_version,
        _contrast_spec=copy.deepcopy(design._contrast_spec),
    )


# -------------------------------------------------------------- the certificate


def owned_units(units: SamplingUnits) -> SamplingUnits:
    """The sampling units with owned position arrays."""
    return SamplingUnits(
        owned_array(units.identifiers), owned_array(units.inactive), units.nobs
    )


def owned_evaluation(evaluation: JointEvaluation) -> JointEvaluation:
    """An evaluation of the joint system with every array and its units owned."""
    return JointEvaluation(
        gamma=owned_array(evaluation.gamma),
        log_mean=owned_array(evaluation.log_mean),
        residual=owned_array(evaluation.residual),
        curvature=owned_array(evaluation.curvature),
        expected_curvature=owned_array(evaluation.expected_curvature),
        weights=owned_array(evaluation.weights),
        moment=owned_array(evaluation.moment),
        bread=owned_array(evaluation.bread),
        information=owned_array(evaluation.information),
        contributions=owned_array(evaluation.contributions),
        units=owned_units(evaluation.units),
    )


def _owned_reported(reported: ReportedScale) -> ReportedScale:
    """The reported scale of a certificate, re-formed through its constructor.

    The map is re-formed in the form that the record holds: the dense ``M`` as an
    owned array, or the direct order scale on an owned copy of its column scale,
    without materializing a dense map that may not be representable.
    """
    order = reported.order
    if order is not None:
        return ReportedScale(
            gamma=owned_array(reported.gamma),
            standard_errors=owned_array(reported.standard_errors),
            order=OrderScale(order.omega, owned_array(order.scale)),
        )
    return ReportedScale(
        gamma=owned_array(reported.gamma),
        transform=owned_array(reported.transform),
        standard_errors=owned_array(reported.standard_errors),
    )


def _owned_coefficients(check: CoefficientCheck) -> CoefficientCheck:
    """The coefficient gate of a certificate with owned arrays."""
    return replace(
        check,
        corrections=owned_array(check.corrections),
        standard_errors=owned_array(check.standard_errors),
        passed=owned_array(check.passed),
        ratios=owned_array(check.ratios),
    )


def owned_certificate(certificate: Certificate) -> Certificate:
    """A certificate with its candidate, evaluation, quantities and gates owned.

    The errors keep their texts in a fresh read-only mapping; the gates that failed
    at each level are the same tuples of names.
    """
    evaluation = certificate.evaluation
    reported = certificate.reported
    coefficients = certificate.coefficients
    return replace(
        certificate,
        gamma=owned_array(certificate.gamma),
        evaluation=None if evaluation is None else owned_evaluation(evaluation),
        correction=_optional(certificate.correction),
        reported=None if reported is None else _owned_reported(reported),
        coefficients=None
        if coefficients is None
        else _owned_coefficients(coefficients),
        errors=MappingProxyType(dict(certificate.errors)),
    )


def owned_linear_evaluation(evaluation: LinearEvaluation) -> LinearEvaluation:
    """A fresh evaluation of the linear equation with its arrays and units owned."""
    return LinearEvaluation(
        gamma=owned_array(evaluation.gamma),
        residual=owned_array(evaluation.residual),
        weights=owned_array(evaluation.weights),
        moment=owned_array(evaluation.moment),
        bread=owned_array(evaluation.bread),
        contributions=owned_array(evaluation.contributions),
        units=owned_units(evaluation.units),
    )


def owned_linear_certificate(certificate: LinearCertificate) -> LinearCertificate:
    """A linear certificate with its point, evaluation, quantities and gates owned."""
    evaluation = certificate.evaluation
    reported = certificate.reported
    coefficients = certificate.coefficients
    return replace(
        certificate,
        gamma=owned_array(certificate.gamma),
        evaluation=None if evaluation is None else owned_linear_evaluation(evaluation),
        correction=_optional(certificate.correction),
        reported=None if reported is None else _owned_reported(reported),
        coefficients=None
        if coefficients is None
        else _owned_coefficients(coefficients),
        errors=MappingProxyType(dict(certificate.errors)),
    )


def owned_any_certificate(certificate: AnyCertificate) -> AnyCertificate:
    """The owned copy of a joint or of a linear certificate, by its type."""
    if isinstance(certificate, Certificate):
        return owned_certificate(certificate)
    return owned_linear_certificate(certificate)


# ----------------------------------------------------------------- the candidates


def _owned_state(state: AcceptedState) -> AcceptedState:
    """The last accepted state of an attempt with owned arrays."""
    return AcceptedState(
        coefficients=owned_array(state.coefficients),
        raw_prediction=_optional(state.raw_prediction),
        loss=state.loss,
        gradient=_optional(state.gradient),
    )


def _owned_attempt(attempt: NewtonAttempt) -> NewtonAttempt:
    """One attempt of the solve with its start, its state and its warnings owned."""
    return replace(
        attempt,
        start=owned_array(attempt.start),
        state=_owned_state(attempt.state),
        warnings=tuple(attempt.warnings),
    )


def _owned_direct_result(direct: IOLSDirectNoFEResult) -> IOLSDirectNoFEResult:
    """The direct IOLS kernel result with owned arrays and read-only mappings."""
    return replace(
        direct,
        beta=owned_array(direct.beta),
        _input_endog=owned_array(direct._input_endog),
        _input_slopes=owned_array(direct._input_slopes),
        linear=_optional(direct.linear),
        iteration=owned_value(direct.iteration),
        diagnostics=owned_value(direct.diagnostics),
        control=owned_value(direct.control),
        backend_details=owned_value(direct.backend_details),
    )


def _owned_adapter_result(adapter: IOLSAdapterResult) -> IOLSAdapterResult:
    """The IOLS adapter result with owned arrays and read-only mappings."""
    norm = adapter.normalization
    owned_norm = TransformedNormalizationState(
        requested=norm.requested,
        enabled=norm.enabled,
        status=norm.status,
        basis=norm.basis,
        scale=_optional(norm.scale),
        condition_before=norm.condition_before,
        condition_after=norm.condition_after,
        scale_min=norm.scale_min,
        scale_max=norm.scale_max,
        scale_floor=norm.scale_floor,
        transformed_rank=norm.transformed_rank,
    )
    return replace(
        adapter,
        theta=_optional(adapter.theta),
        slopes=owned_array(adapter.slopes),
        slopes_star=owned_array(adapter.slopes_star),
        linear=_optional(adapter.linear),
        normalization=owned_norm,
        diagnostics=owned_value(adapter.diagnostics),
        direct_result=None
        if adapter.direct_result is None
        else _owned_direct_result(adapter.direct_result),
    )


def _owned_start(start: Start) -> Start:
    """A start with its coefficients owned, its origin and metadata kept."""
    return Start(
        owned_array(start.coefficients),
        start.source,
        intercept=start.intercept,
        mean=start.mean,
    )


def _owned_newton(candidate: NewtonCandidate) -> NewtonCandidate:
    """A convex Newton candidate with its start, attempts and states owned."""
    owned_attempts = tuple(_owned_attempt(attempt) for attempt in candidate.attempts)
    return replace(
        candidate,
        start=_owned_start(candidate.start),
        attempts=owned_attempts,
    )


def _owned_root(candidate: JointRootCandidate) -> JointRootCandidate:
    """A joint root candidate with its start and its initializer's records owned.

    The initializer's convex candidate and certificate are re-formed as the records of
    a convex fit are; the points, the merit factor and the attempts of the stages are
    held as immutable bytes handed out as fresh views, and the stage summaries are
    scalars with read-only error texts, kept as they are.
    """
    initializer = candidate.initializer
    newton, certificate = initializer.newton, initializer.certificate
    owned_initializer = replace(
        initializer,
        _point=FloatStorage.store(initializer.point),
        newton=None if newton is None else _owned_newton(newton),
        certificate=None if certificate is None else owned_certificate(certificate),
    )
    stages = tuple(
        replace(
            stage,
            attempts=tuple(
                replace(
                    attempt,
                    _opening=FloatStorage.store(attempt.opening),
                    _point=FloatStorage.store(attempt.point),
                )
                for attempt in stage.attempts
            ),
            summary=replace(
                stage.summary, errors=MappingProxyType(dict(stage.summary.errors))
            ),
        )
        for stage in candidate.stages
    )
    merit = candidate.merit_cholesky
    return replace(
        candidate,
        start=_owned_start(candidate.start),
        initializer=owned_initializer,
        stages=stages,
        _merit_cholesky=None if merit is None else FloatStorage.store(merit),
        _point=FloatStorage.store(candidate.coefficients),
    )


def owned_candidate(candidate: AnyCandidate) -> AnyCandidate:
    """The candidate of a solve with its start, attempts and states owned.

    The options, the provider, the counts, the assessments and the timing are
    scalars or frozen records of scalars, kept as they are.
    """
    if isinstance(candidate, IolsCandidate):
        return replace(
            candidate,
            adapter_result=_owned_adapter_result(candidate.adapter_result),
            direct_result=_owned_direct_result(candidate.direct_result),
        )
    if isinstance(candidate, JointRootCandidate):
        return _owned_root(candidate)
    return _owned_newton(candidate)


def owned_linear(linear: LinearCandidate) -> LinearCandidate:
    """The linear candidate with its units, weights, factorization and fit owned."""
    factorization = linear.factorization
    return replace(
        linear,
        units=owned_units(linear.units),
        weights=owned_array(linear.weights),
        factorization=None
        if factorization is None
        else replace(
            factorization, singular_values=owned_array(factorization.singular_values)
        ),
        coefficients=_optional(linear.coefficients),
        residuals=_optional(linear.residuals),
    )
