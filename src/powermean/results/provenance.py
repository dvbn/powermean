"""The provenance of a result: what actually ran, recorded beside the estimates.

Each result records the package version; hashes of the compiled kernels and of the
carried scikit-learn files, and the provider used; the resolved estimator, solver,
options and start; the trace summary (iterations, line-search trials, retries,
absorption sweeps, wall time by stage); a fingerprint of the estimation sample; and
which implementation ran.

The record is formed once from the retained state and reports, as values that JSON
holds, each nested value owned and read-only (a mapping as a read-only view, a list or
a tuple as a tuple, recursively, so that no nested value is a mutable escape hatch
into the record), and ``to_dict`` returns a fresh plain copy of every record on every
call (dicts and lists, recursively, with the meaning that JSON gives them), so that
an edit of a copy reaches nothing retained:

- the versions of the package, of Python, of the platform, of NumPy and pandas as
  imported, and of the installed SciPy and Formulaic distributions;
- the implementation of the row operations and of the iteration;
  the Newton provider and its version; the
  SHA-256 of the carried scikit-learn files that ran (the Newton iteration of the
  bundled provider, the linear loss of both providers), read from the
  installed files, which is a record of what ran and no claim of parity with any
  other implementation or of a whole-fit comparison with a source;
- the solver, the resolved settings as stored, the start with its origin, and the
  trace of the solve: the attempts with their native reasons, iterations, line
  searches, trials, evaluations, warnings and timing, the retained attempt, the
  assessments of the retry, and whether a retry made no measurable progress; for a
  joint root solve outside ``[1, 2]``, the initializer (the caller's start, or the
  nearest convex fit with its own trace, status and certificate summary), the merit
  metric, the root-Newton options, every stage with its attempts and certificate
  summary, the target stage, the continuation, the counts and the counted work; the
  start that the caller requested on the reported scale, where one was given, beside
  the solver's start in its own coordinates (both coordinate meanings);
  on the linear route the least-squares driver, its cutoff, rank and singular values,
  and the requested start mapped to the coordinates of the solver by the
  forward map at the order zero, recorded as not used, since the least-squares solve
  reads no start and has no iterative trace;
  for a supplied point without a solve, no trace; for an automatic start that had no
  finite value, the record of that failure and no trace;
- the status of the solve by the status rules, the level of the certificate,
  whether the point is certified, whether its inference is degenerate, the
  coverage label of the theory and its basis;
- the estimator record, the kind of the weights and the columns that the preparation
  read, an offset or exposure expression being recorded as the text it was given;
  the formula's first part, its terms and the columns of the data that they read,
  the Formulaic version and the names of the coefficients;
- the fingerprint of the sample and its accounting: input rows, retained rows, the
  statistical count, the counts by reason and by variable group, the conditioning on
  positive outcomes and the rows whose subset condition was missing;
- the convention of the default covariance: method, cluster, bread, coordinate,
  factor, counts, rank and diagnostics;
- the references retained with the result (:mod:`powermean.results.citations`).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from powermean.results import citations, naming
from powermean.results.retained import AnyCandidate, RetainedFit
from powermean.solve.cancelled_records import RootAttemptRecord
from powermean.solve.iols.candidate import IolsCandidate
from powermean.solve.joint_root import summary_of
from powermean.solve.joint_root_records import (
    JointRootCandidate,
    JointRootInitializer,
    StageSummary,
)
from powermean.solve.newton_trace import NewtonAttempt, NewtonCandidate
from powermean.solve.starts import Start

# The carried scikit-learn files, by the provider that runs them: the Newton iteration
# of the bundled provider, the linear loss of both.  Read as files by path, so that this
# module imports no vendor code: the folder beside this package's result layer, located
# from this module and not through the root package, which a lower layer does not import
# (the root names the public API for the type checker).
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
VENDOR = PACKAGE_ROOT / "_vendor"
NATIVE_IOLS = PACKAGE_ROOT / "_native" / "iols"
SOLVE_IOLS = PACKAGE_ROOT / "solve" / "iols"
NEWTON_FILE = "vendor_newton.py"
LINEAR_LOSS_FILE = "vendor_linear_loss.py"
IOLS_BIND_CPP_FILE = "bind.cpp"
IOLS_KERNEL_PY_FILE = "kernel.py"
IMPLEMENTATION = "NumPy row operations and iteration; no compiled kernel"
IOLS_CPP_IMPLEMENTATION = (
    "NumPy row operations; compiled pybind11/Eigen IOLS direct fixed-tau kernel"
)
IOLS_NUMPY_IMPLEMENTATION = (
    "NumPy row operations and NumPy reference IOLS direct fixed-tau kernel"
)
Record = Mapping[str, object]


def _frozen_value(value: object) -> object:
    """A read-only form of a value: mappings to read-only views, sequences to tuples.

    Recursive, so that a list inside a mapping inside a tuple is a tuple as well;
    strings, numbers, Booleans and ``None`` are themselves.
    """
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _frozen_value(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_frozen_value(item) for item in value)
    return value


def _frozen(mapping: Mapping[str, object]) -> Record:
    """A read-only view of a mapping, every nested value frozen as well."""
    return MappingProxyType(
        {key: _frozen_value(value) for key, value in mapping.items()}
    )


def _thawed(value: object) -> object:
    """A fresh plain copy of a frozen value: mappings to dicts, tuples to lists.

    Recursive; the copy shares no container with the record, so that an edit of it
    reaches nothing retained.
    """
    if isinstance(value, Mapping):
        return {key: _thawed(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_thawed(item) for item in value]
    return value


def file_digest(name: str) -> str:
    """The SHA-256 of a carried file of ``powermean._vendor``, as installed."""
    return hashlib.sha256((VENDOR / name).read_bytes()).hexdigest()


def _powermean_version() -> str:
    """Installed version, or the non-version sentinel ``uninstalled`` if absent."""
    try:
        return importlib.metadata.version("powermean")
    except importlib.metadata.PackageNotFoundError as error:
        if error.args == ("powermean",):
            return "uninstalled"
        raise


def versions() -> Record:
    """The versions of the package and of the libraries, as imported."""
    return _frozen(
        {
            "powermean": _powermean_version(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "formulaic": importlib.metadata.version("formulaic"),
        }
    )


def _attempt(attempt: NewtonAttempt) -> Record:
    """The trace of one attempt."""
    assessment = attempt.assessment
    return _frozen(
        {
            "reason": attempt.reason,
            "message": attempt.message,
            "iterations": attempt.iterations,
            "line_searches": attempt.line_searches,
            "trials": attempt.trials,
            "refused_trials": attempt.refused_trials,
            "evaluations": {
                "loss": attempt.evaluations.loss,
                "loss_gradient": attempt.evaluations.loss_gradient,
                "gradient_hessian": attempt.evaluations.gradient_hessian,
                "gradient": attempt.evaluations.gradient,
            },
            "warnings": tuple(attempt.warnings),
            "wall": attempt.timing.wall,
            "cpu": attempt.timing.cpu,
            "assessment": None
            if assessment is None
            else {
                "accepted": assessment.accepted,
                "score": assessment.score,
                "error": assessment.error,
            },
        }
    )


def _start(start: Start) -> dict[str, object]:
    """A start in the coordinates of the solver, with its origin."""
    return {
        "coefficients": tuple(start.coefficients.tolist()),
        "source": start.source,
        "intercept": start.intercept,
        "mean": start.mean,
        "coordinates": "solver",
    }


def _newton(candidate: NewtonCandidate) -> dict[str, object]:
    """The trace of a convex Newton solve, as plain values."""
    return {
        "provider": candidate.provider,
        "provider_version": candidate.provider_version,
        "start": _start(candidate.start),
        "attempts": tuple(_attempt(attempt) for attempt in candidate.attempts),
        "retained": candidate.retained,
        "retries": len(candidate.attempts) - 1,
        "iterations": candidate.iterations,
        "evaluations": candidate.evaluations.total,
        "fallback_triggers": candidate.fallback_triggers,
        "no_measurable_progress": candidate.no_measurable_progress,
        "assessments": candidate.assessments,
        "assessment_wall": candidate.assessment_timing.wall,
        "assessment_cpu": candidate.assessment_timing.cpu,
        "wall": candidate.timing.wall,
        "cpu": candidate.timing.cpu,
    }


def _summary(summary: StageSummary) -> dict[str, object]:
    """The gates and quantities of a certificate summary, as plain values."""
    return {
        "level": summary.level,
        "strict": tuple(summary.strict),
        "secondary": tuple(summary.secondary),
        "scaled_score": summary.scaled_score,
        "index_correction": summary.index_correction,
        "rcond": summary.rcond,
        "positive_definite": summary.positive_definite,
        "conditioned": summary.conditioned,
        "score_metric": summary.score_metric,
        "errors": dict(summary.errors),
    }


def _root_attempt(attempt: RootAttemptRecord) -> dict[str, object]:
    """One root-Newton attempt: its native reason, counts and merit values."""
    return {
        "reason": attempt.native,
        "message": attempt.message,
        "iterations": attempt.iterations,
        "evaluations": attempt.evaluations,
        "backtracks": attempt.backtracks,
        "initial_merit": attempt.initial_merit,
        "final_merit": attempt.final_merit,
    }


def _initializer(initializer: JointRootInitializer) -> dict[str, object]:
    """The start of a root solve: the caller's, or the nearest convex fit's record."""
    newton, certificate = initializer.newton, initializer.certificate
    return {
        "kind": initializer.kind,
        "tau": None if initializer.variance is None else initializer.variance.tau,
        "certified": initializer.certified,
        "status": initializer.status,
        "reason": initializer.reason,
        "point": tuple(initializer.point.tolist()),
        "coordinates": "solver",
        "certificate": None
        if certificate is None
        else _summary(summary_of(certificate)),
        "trace": None if newton is None else _newton(newton),
    }


def _root_trace(candidate: JointRootCandidate) -> Record:
    """The trace of a joint root solve outside ``[1, 2]`` (module docstring)."""
    stages = tuple(
        {
            "tau": stage.variance.tau,
            "attempts": tuple(_root_attempt(attempt) for attempt in stage.attempts),
            "retained": stage.retained,
            "status": stage.status,
            "iterations": stage.iterations,
            "evaluations": stage.evaluations,
            "certificate": _summary(stage.summary),
        }
        for stage in candidate.stages
    )
    work = {
        count.name: {"begun": count.begun, "completed": count.completed}
        for count in candidate.work.counts
    }
    return _frozen(
        {
            "route": "joint root Newton outside tau in [1, 2]",
            "start": _start(candidate.start),
            "initializer": _initializer(candidate.initializer),
            "merit_metric": candidate.merit_metric,
            "merit_formed": candidate.merit_cholesky is not None,
            "root_options": dict(candidate.root_options.to_dict()),
            "stages": stages,
            "target_stage": candidate.target_stage,
            "continuation_used": candidate.continuation_used,
            "reason": candidate.reason,
            "iterations": candidate.iterations,
            "evaluations": candidate.evaluations,
            "root_iterations": candidate.root_iterations,
            "root_evaluations": candidate.root_evaluations,
            "work": work,
            "wall": candidate.timing.wall,
            "cpu": candidate.timing.cpu,
        }
    )


def trace(candidate: AnyCandidate) -> Record:
    """The trace of a solve: provider, start, attempts, retry, counts and timing."""
    if isinstance(candidate, JointRootCandidate):
        return _root_trace(candidate)
    if isinstance(candidate, IolsCandidate):
        return _frozen(
            {
                "requested_backend": candidate.requested_backend,
                "preferred_backend": candidate.preferred_backend,
                "actual_backend": candidate.actual_backend,
                "backend_explicit": candidate.iols_options.backend_explicit,
                "fallback_used": candidate.fallback_used,
                "availability_reason": candidate.availability_reason,
                "reason": candidate.reason,
                "start_policy": "kernel_internal_log1p_ols_on_demeaned_x",
                "iterations": candidate.iterations,
                "phase1_evaluations": candidate.phase1_iterations,
                "phase2_evaluations": candidate.phase2_iterations,
                "rho_restarts": candidate.rho_restarts,
                "iteration": candidate.direct_result.iteration,
                "diagnostics": candidate.direct_result.diagnostics,
                "backend_details": candidate.direct_result.backend_details,
                "adapter_diagnostics": candidate.diagnostics,
                "native_status": candidate.adapter_result.native_status,
                "failure_reason": candidate.adapter_result.failure_reason,
                "omega": candidate.omega,
                "column_scale": tuple(candidate.design_scale.tolist()),
                "intercept_column": candidate.intercept,
                "wall": candidate.timing.wall,
                "cpu": candidate.timing.cpu,
            }
        )
    return _frozen(_newton(candidate))


def carried_files(candidate: AnyCandidate | None) -> Record:
    """The digests of the carried files that ran: none on the linear route.

    A joint root solve runs no carried file itself; its nearest convex initializer
    ran the convex Newton iteration, whose files are recorded, and a caller's start
    ran none.
    """
    if candidate is None:
        return _frozen({})
    if isinstance(candidate, JointRootCandidate):
        return carried_files(candidate.initializer.newton)
    if isinstance(candidate, IolsCandidate):
        sources = candidate.direct_result.backend_details["source_files"]
        return _frozen({name: metadata["sha256"] for name, metadata in sources.items()})
    files = {LINEAR_LOSS_FILE: file_digest(LINEAR_LOSS_FILE)}
    if candidate.provider == "bundled":
        files[NEWTON_FILE] = file_digest(NEWTON_FILE)
    return _frozen(files)


def _linear_start(retained: RetainedFit) -> dict[str, object] | None:
    """The requested start in the coordinates of the solver, on the linear route.

    The accepted forward map of the order scale at the order zero, ``theta_j = d_j
    beta_j``, of the start that the caller gave; the least-squares solve read no
    start, which ``used`` says, and no trace exists for it.
    """
    start = retained.requested_start
    if start is None:
        return None
    mapped = retained.sample.order_scale.solver_coefficients(start)
    return {
        "coefficients": tuple(mapped.tolist()),
        "source": "user",
        "coordinates": "solver",
        "used": False,
    }


def _linear(retained: RetainedFit) -> Record | None:
    """The least-squares solve of the linear route, as the candidate reports it."""
    linear = retained.linear
    if linear is None:
        return None
    factorization = linear.factorization
    return _frozen(
        {
            "reason": linear.reason,
            "message": linear.message,
            "driver": None if factorization is None else factorization.driver,
            "cutoff": None if factorization is None else factorization.cutoff,
            "rank": None if factorization is None else factorization.rank,
            "singular_values": None
            if factorization is None
            else tuple(factorization.singular_values.tolist()),
            "coefficients_formed": linear.coefficients is not None,
            "certified_afresh": retained.certificate is not None,
            "start": _linear_start(retained),
        }
    )


def _requested_start(retained: RetainedFit) -> Record | None:
    """The start given on the reported scale, where the caller gave one."""
    start = retained.requested_start
    if start is None:
        return None
    return _frozen({"coefficients": tuple(start.tolist()), "coordinates": "reported"})


def _failure(retained: RetainedFit) -> Record | None:
    """The automatic start that had no finite value, where the solve ended there."""
    failure = retained.failure
    if failure is None:
        return None
    return _frozen(
        {
            "source": failure.source,
            "message": failure.message,
            "mean": failure.mean,
            "intercept": failure.intercept,
        }
    )


def _sample(retained: RetainedFit) -> Record:
    """The fingerprint and the accounting of the sample."""
    sample = retained.sample
    return _frozen(
        {
            "fingerprint": sample.fingerprint,
            "nobs": sample.nobs,
            "count": sample.count,
            "effective_n": sample.effective_n,
            "counts": dict(sample.counts),
            "missing": dict(sample.missing.counts),
            "subset_missing": retained.subset_missing,
            "positive_only": sample.positive_only,
            "conditioned": sample.conditioned,
            "weight_kind": None
            if sample.data.weights is None
            else sample.data.weights.kind,
            "weights_column": retained.columns.weights,
            "offset_column": retained.columns.offset,
            "exposure_column": retained.columns.exposure,
            "offset_expression": retained.offset_design is not None,
            "cluster_column": retained.columns.cluster,
            "retained_classifications": tuple(sample.metadata),
        }
    )


def _formula(retained: RetainedFit) -> Record:
    """The fitted right-hand side: terms, columns read, version, coefficient names."""
    design = retained.design
    return _frozen(
        {
            "terms": tuple((term, tuple(columns)) for term, columns in design.terms),
            "variables": tuple(design.variables),
            "formulaic": design.formulaic_version,
            "names": tuple(retained.names),
        }
    )


@dataclass(frozen=True, eq=False)
class Provenance:
    """What ran, as read-only records (module docstring).

    ``to_dict`` returns a plain copy that :mod:`json` holds, with ``allow_nan``: the
    timing and the quantities are floats, the trace may hold NaN assessments.
    """

    versions: Record
    implementation: str
    solver: str | None
    carried_files: Record
    options: Record
    trace: Record | None
    linear: Record | None
    status: str | None
    level: str | None
    certified: bool
    degenerate: bool
    coverage: Record
    estimator: Record
    sample: Record
    formula: Record
    covariance: Record | None
    references: tuple[citations.Reference, ...]
    requested_start: Record | None = None
    failure: Record | None = None

    def to_dict(self) -> dict[str, object]:
        """A plain, JSON-compatible copy of every record."""
        return {
            "versions": _thawed(self.versions),
            "implementation": self.implementation,
            "solver": self.solver,
            "carried_files": _thawed(self.carried_files),
            "options": _thawed(self.options),
            "trace": _thawed(self.trace),
            "linear": _thawed(self.linear),
            "status": self.status,
            "level": self.level,
            "certified": self.certified,
            "degenerate": self.degenerate,
            "coverage": _thawed(self.coverage),
            "estimator": _thawed(self.estimator),
            "sample": _thawed(self.sample),
            "formula": _thawed(self.formula),
            "covariance": _thawed(self.covariance),
            "references": [reference.to_dict() for reference in self.references],
            "requested_start": _thawed(self.requested_start),
            "failure": _thawed(self.failure),
        }


def provenance(retained: RetainedFit) -> Provenance:
    """The provenance of a retained fit (module docstring)."""
    candidate, certificate = retained.candidate, retained.certificate
    label = naming.coverage(retained.estimator)
    solver: str | None = None
    implementation = IMPLEMENTATION
    if retained.estimator.route == "linear":
        solver = "linear least squares (scipy.linalg.lstsq)"
    elif candidate is not None:
        solver = retained.computation.solver
        if isinstance(candidate, IolsCandidate):
            implementation = (
                IOLS_CPP_IMPLEMENTATION
                if candidate.actual_backend == "cpp"
                else IOLS_NUMPY_IMPLEMENTATION
            )
    return Provenance(
        versions=versions(),
        implementation=implementation,
        solver=solver,
        carried_files=carried_files(candidate),
        options=_frozen(dict(retained.computation.options)),
        trace=None if candidate is None else trace(candidate),
        linear=_linear(retained),
        status=retained.status,
        level=retained.level,
        certified=retained.certified,
        degenerate=certificate is not None and certificate.degenerate,
        coverage=_frozen({"label": label.label, "basis": label.basis}),
        estimator=_frozen(retained.estimator.to_dict()),
        sample=_sample(retained),
        formula=_formula(retained),
        covariance=None
        if retained.default is None
        else _frozen(retained.default.to_dict()),
        references=citations.references(retained.estimator),
        requested_start=_requested_start(retained),
        failure=_failure(retained),
    )
