"""Result backends for joint absorbed fixed-effects and selected-cancellation fits."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import asdict, replace
from typing import Unpack

import numpy as np
import pandas as pd

from powermean._prepare.compose import _owned_index
from powermean.data.inference_conventions import ConventionValues
from powermean.data.inference_requests import (
    ClusterRequest,
    HCRequest,
)
from powermean.data.raw import _object_array
from powermean.data.request_values import OMITTED, Omitted
from powermean.effects.targets import ReportingContext, ReportingPoint
from powermean.errors import ConvergenceError, SpecificationError
from powermean.fe.groups import Grouping, encode_block
from powermean.interface.fe_options import (
    FEBread,
    FEVcovRequest,
)
from powermean.interface.fe_result_backends import (
    AbsorbedState,
    FEView,
    _bind_absorbed_cluster_codes,
    _parse_vcov_spec_for_absorbed,
    _target_reporting_context,
    _tidy_frame,
    _vcov_attrs_from_retained,
    _vcov_frame_from_retained,
)
from powermean.interface.retained_result import RetainedAbsorbedResult
from powermean.model.order_scale import OrderScale
from powermean.results import citations, naming
from powermean.results.absorbed_owned import RetainedCertificateSnapshot
from powermean.results.absorbed_retained import RetainedAbsorbedEnvelope
from powermean.results.fe_helpers import (
    PredictionOptions,
    detached_record,
    normal_quantile,
    prediction_kind,
)
from powermean.results.fe_sample import FESampleView
from powermean.results.owned import owned_array
from powermean.results.provenance import (
    IMPLEMENTATION,
    Provenance,
    _frozen,
    carried_files,
    versions,
)
from powermean.results.retained_hc import RetainedHC
from powermean.results.retained_hc import (
    retained_request as compute_retained_request,
)
from powermean.results.retained_hc_records import HCInferenceRequest
from powermean.results.retained_inference import (
    InferenceRequest as RetainedInferenceRequest,
)
from powermean.results.retained_inference import (
    RetainedInference,
)
from powermean.results.summary import (
    WIDTH,
    _table,
)
from powermean.results.wz_inference import joint_wz
from powermean.solve.absorbed_direction import normalize
from powermean.solve.absorbed_scale_details import owned_record


class AbsorbedResultBackend:
    """Backend adapting a joint absorbed fixed-effects fit to the public ``Result``."""

    def __init__(self, view: FEView, state: AbsorbedState) -> None:
        """Bind the authentic retained state to owned normal-call views."""
        self._inference_override: RetainedInference | RetainedHC | None = None
        estimator = view.estimator
        order_sample = view.order_sample
        input_labels = view.input_labels
        columns = view.columns
        subset_missing = view.subset_missing
        parts = view.parts
        design = view.design
        offset_design = view.offset_design
        solver_options = view.solver_options
        fit_vcov_spec = view.fit_vcov_spec
        requested_start = view.requested_start
        retained_result = state.retained_result
        initial_inference = state.initial_inference
        unsolved_outcome = state.unsolved_outcome
        start_failure = state.start_failure
        status = state.status
        absorbed_options = state.absorbed_options
        organization = state.organization
        self.retained_result = retained_result
        self.initial_inference = initial_inference
        self.unsolved_outcome = unsolved_outcome
        self.start_failure = start_failure
        self.status = status
        self.estimator = estimator
        self.order_sample = order_sample
        self.input_labels = _owned_index(input_labels)
        self.columns = columns
        self.subset_missing = int(subset_missing)
        self.parts = parts
        self.design = design
        self.offset_design = offset_design
        self.solver_options = solver_options
        self.absorbed_options = absorbed_options
        self.organization = organization
        self.operation = state.operation
        self.root_options = state.root_options
        self.max_dense_bytes = state.max_dense_bytes
        self._bias = state.bias_correction
        self.fit_vcov_spec = owned_record(fit_vcov_spec)
        self.requested_start = (
            None if requested_start is None else owned_array(requested_start)
        )
        self._sample_view = FESampleView(
            order_sample,
            self.input_labels,
            columns,
            subset_missing,
            dependence=(
                ()
                if retained_result is None or retained_result.dependence is None
                else retained_result.dependence.retained_names
            ),
            fe_snapshot=(None if retained_result is None else retained_result.sample),
            outcome_snapshot=unsolved_outcome,
        )
        self._provenance = self._build_provenance()

    @property
    def certified(self) -> bool:
        """Return the certified from the actual retained state, detached outward."""
        return (
            self.retained_result is not None
            and bool(self.retained_result.accepted)
            and self.status in ("verified", "verified_secondary")
        )

    @property
    def certificate(self) -> RetainedCertificateSnapshot | None:
        """Return the certificate from the actual retained state, detached outward."""
        if self.retained_result is None:
            return None
        return detached_record(self.retained_result.certificate)

    @property
    def sample(self) -> FESampleView:
        """Return the sample from the actual retained state, detached outward."""
        return self._sample_view

    @property
    def provenance(self) -> Provenance:
        """Return the provenance from the actual retained state, detached outward."""
        return self._provenance

    def _names(self) -> pd.Index[str]:
        return pd.Index(list(self._sample_view.column_names), name="term")

    def _withheld(self) -> ConvergenceError:
        cert = self.certificate
        last_state: object = cert
        if self.retained_result is not None:
            if self.retained_result.history.final is not None:
                last_state = self.retained_result.history.final
            elif self.retained_result.selected_point is not None:
                last_state = self.retained_result.selected_point
        elif self.unsolved_outcome is not None:
            last_state = self.unsolved_outcome
        elif self.start_failure is not None:
            last_state = replace(self.start_failure)
        return ConvergenceError(
            "convergence.not_certified",
            status=self.status,
            last_state=detached_record(last_state),
            certificate=cert,
        )

    def coef(self) -> pd.Series[float]:
        """Return the coef from the actual retained state, detached outward."""
        if self.retained_result is None or self.retained_result.selected_point is None:
            raise self._withheld()
        if self._bias is not None and not self.certified:
            raise self._withheld()
        beta = (
            self.retained_result.coef()
            if self._bias is None
            else self._bias.coefficients()
        )
        if not np.isfinite(beta).all():
            raise self._withheld()
        return pd.Series(
            np.array(beta, dtype=np.float64, copy=True),
            index=self._names(),
            name="estimate",
        )

    def se(self) -> pd.Series[float]:
        """Return the se from the actual retained state, detached outward."""
        if not self.certified or self.retained_result is None:
            raise self._withheld()
        if self.retained_result.empty:
            return pd.Series(
                np.empty(0, dtype=np.float64),
                index=self._names(),
                name="std_error",
            )
        record = self._reported_inference()
        if record is None:
            raise self._withheld()
        return pd.Series(
            np.array(
                record.standard_errors,
                dtype=np.float64,
                copy=True,
            ),
            index=self._names(),
            name="std_error",
        )

    def _reported_inference(self) -> RetainedInference | RetainedHC | None:
        """Read this view's covariance while retaining the original fit cache."""
        return (
            self.initial_inference
            if self._inference_override is None
            else self._inference_override
        )

    def _reporting_context(self) -> ReportingContext:
        """Bind the reported point and cached active covariance without new work."""
        if not self.certified or self.retained_result is None:
            raise self._withheld()
        record = self._reported_inference()
        if record is None:
            raise self._withheld()
        coefficient_point = (
            "certified uncorrected root"
            if self._bias is None
            else f"{self._bias.label} corrected reported coefficients"
        )
        return _target_reporting_context(
            self._sample_view.column_names,
            np.asarray(self.coef().to_numpy(), dtype=np.float64),
            record,
            self.estimator.mean.omega,
            ReportingPoint(self.status, coefficient_point),
        )

    def _matches_initial(
        self,
        fe_vcov: FEVcovRequest,
        ssc_val: str | ConventionValues | Omitted,
        bread_val: FEBread,
    ) -> bool:
        init_vcov, init_ssc, init_bread = self.fit_vcov_spec
        return (
            fe_vcov == init_vcov
            and (isinstance(ssc_val, Omitted) or ssc_val == init_ssc)
            and bread_val == init_bread
        )

    def vcov(self, spec: object = None) -> pd.DataFrame:
        """Return detached covariance from the retained uncorrected state."""
        record = self._inference(spec)
        if isinstance(record, pd.DataFrame):
            return record
        return _vcov_frame_from_retained(record, self._sample_view.column_names)

    def with_vcov(self, spec: object) -> AbsorbedResultBackend:
        """Return a covariance view without refitting or changing certification."""
        record = self._inference(spec)
        if isinstance(record, pd.DataFrame):
            raise SpecificationError(
                "specification.unsupported_inference",
                {"method": "with_vcov", "case": "an empty slope result"},
            )
        result = copy.copy(self)
        result._inference_override = record
        result._provenance = result._build_provenance()
        return result

    def _inference(
        self, spec: object = None
    ) -> RetainedInference | RetainedHC | pd.DataFrame:
        """Return the vcov from the actual retained state, detached outward."""
        if not self.certified or self.retained_result is None:
            raise self._withheld()
        init_vcov, init_ssc, init_bread = self.fit_vcov_spec
        if spec is None and self._inference_override is not None:
            return self._inference_override
        if spec is None:
            fe_vcov, ssc_val, bread_val = init_vcov, init_ssc, init_bread
            is_initial = True
        else:
            fe_vcov, ssc_val, bread_val = _parse_vcov_spec_for_absorbed(
                spec, OMITTED, init_bread
            )
            is_initial = self._matches_initial(fe_vcov, ssc_val, bread_val)

        if self.retained_result.empty:
            if not is_initial and not self.retained_result.keep_influence:
                raise SpecificationError(
                    "specification.influence_discarded",
                    {"request": "a covariance other than the fit's"},
                )
            idx = self._names()
            empty_df = pd.DataFrame(
                np.empty((0, 0), dtype=np.float64), index=idx, columns=idx.copy()
            )
            empty_df.attrs["method"] = (
                fe_vcov.method
                if isinstance(fe_vcov, HCRequest)
                else ("cluster" if isinstance(fe_vcov, ClusterRequest) else "robust")
            )
            empty_df.attrs["bread"] = bread_val
            empty_df.attrs["coordinate"] = "reported"
            return empty_df

        if is_initial and self.initial_inference is not None:
            return self.initial_inference

        if isinstance(fe_vcov, HCRequest):
            hc_req = HCInferenceRequest.from_hc_request(
                fe_vcov,
                ssc=ssc_val,
                bread=bread_val,
            )
            return compute_retained_request(self.retained_result.envelope, hc_req)

        if isinstance(fe_vcov, ClusterRequest):
            clusters = _bind_absorbed_cluster_codes(self.retained_result, fe_vcov)
            inf_req = RetainedInferenceRequest.from_cluster_request(
                fe_vcov,
                clusters,
                ssc=ssc_val,
                bread=bread_val,
            )
        else:
            inf_req = RetainedInferenceRequest(
                "robust",
                (),
                ssc_val,
                bread_val,
            )
        if inf_req.adjustment == "wz":
            return joint_wz(
                self.retained_result.envelope,
                inf_req,
                self.parts.fixed_effects,
                self.absorbed_options,
                self.solver_options,
                max_dense_bytes=self.max_dense_bytes,
            )
        return compute_retained_request(self.retained_result.envelope, inf_req)

    def tidy(self, level: float = 0.95) -> pd.DataFrame:
        """Return the tidy from the actual retained state, detached outward."""
        normal_quantile(level)
        if not self.certified or self.retained_result is None:
            raise self._withheld()
        names = self._sample_view.column_names
        if self.retained_result.empty:
            return _tidy_frame(
                names,
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                level,
                None,
            )
        record = self._reported_inference()
        if record is None:
            raise self._withheld()
        beta = (
            self.retained_result.coef()
            if self._bias is None
            else self._bias.coefficients()
        )
        errors = record.standard_errors
        ref = record.applied.reference
        table = _tidy_frame(names, beta, errors, level, ref)
        if self._bias is not None:
            assert self._bias.point is not None
            table["uncorrected"] = self._bias.point.uncorrected.view()
            table["bias"] = self._bias.point.bias.view()
            table.attrs["coefficient_point"] = "WZ corrected reported coefficients"
            table.attrs["covariance_point"] = "certified uncorrected root"
        return table

    def confint(self, level: float = 0.95) -> pd.DataFrame:
        """Return the confint from the actual retained state, detached outward."""
        table = self.tidy(level)
        if len(table) == 0:
            return pd.DataFrame(
                np.empty((0, 2), dtype=np.float64),
                index=self._names(),
                columns=["lower", "upper"],
            )
        arr = table[["conf_low", "conf_high"]].to_numpy(dtype=np.float64)
        return pd.DataFrame(arr, index=self._names(), columns=["lower", "upper"])

    def influence(self) -> pd.DataFrame:
        """Return the influence from the actual retained state, detached outward."""
        if not self.certified or self.retained_result is None:
            raise self._withheld()
        if not self.retained_result.keep_influence:
            raise SpecificationError(
                "specification.influence_discarded",
                {"request": "influence"},
            )
        pos = self.retained_result.sample.positions
        if self.retained_result.empty:
            return pd.DataFrame(
                np.empty((pos.size, 0), dtype=np.float64),
                index=pd.Index(pos.tolist(), name="position"),
                columns=self._names(),
            )
        psi_theta = self.retained_result.envelope.influence()
        if psi_theta is None:
            raise SpecificationError(
                "specification.influence_discarded",
                {"request": "influence"},
            )
        order_scale = self.retained_result.order
        assert order_scale is not None
        rows = order_scale.influences(psi_theta)
        return pd.DataFrame(
            np.array(rows, dtype=np.float64, copy=True),
            index=pd.Index(pos.tolist(), name="position"),
            columns=self._names(),
        )

    def fixef(self) -> dict[str, pd.Series[float]]:
        """Return normalized absorbed fixed effects per block on the reported scale."""
        if (
            not self.certified
            or self.retained_result is None
            or self.retained_result.selected_point is None
        ):
            raise self._withheld()
        point = self.retained_result.selected_point
        grouping = self.retained_result.sample.grouping
        if grouping is None:
            raise self._withheld()
        norm_effects = point.normalized_effects
        if norm_effects is None:
            raw_effects = point.raw_effects
            if raw_effects is None:
                raise self._withheld()
            rebuilt = Grouping(
                tuple(
                    encode_block(
                        _object_array(block.levels)[block.codes],
                        name=block.name,
                    )
                    for block in grouping.blocks
                ),
                grouping.positions,
            )
            norm_effects = normalize(rebuilt, raw_effects).effects
        omega = self.estimator.mean.omega
        out: dict[str, pd.Series[float]] = {}
        for block, eff in zip(grouping.blocks, norm_effects, strict=True):
            scale = OrderScale(omega, np.ones(eff.size, dtype=np.float64))
            scaled = scale.coefficients(eff)
            out[block.name] = pd.Series(
                np.array(scaled, dtype=np.float64, copy=True),
                index=pd.Index(
                    list(block.levels),
                    dtype=object,
                    name=block.name,
                    tupleize_cols=False,
                ),
                name=block.name,
            )
            if self._bias is not None:
                out[block.name].attrs["evaluation_point"] = (
                    "original uncorrected estimating root"
                )
                out[block.name].attrs["nuisance_correction"] = "not_supplied"
        return out

    def predict(
        self,
        data: pd.DataFrame | None = None,
        **options: Unpack[PredictionOptions],
    ) -> pd.Series[float]:
        """Return the predict from the actual retained state, detached outward."""
        kind = prediction_kind(options)
        if (
            not self.certified
            or self.retained_result is None
            or self.retained_result.selected_point is None
        ):
            raise self._withheld()
        if data is not None:
            raise SpecificationError(
                "specification.invalid_value",
                {
                    "argument": "data",
                    "value": "out-of-sample data",
                    "valid": "None (in-sample absorbed fitted values)",
                },
            )
        q = self.retained_result.selected_point.q
        omega = self.estimator.mean.omega
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            index = (
                np.array(q, dtype=np.float64, copy=True) / omega
                if omega != 0.0
                else np.array(q, dtype=np.float64, copy=True)
            )
            fitted = np.exp(index)
        if not np.isfinite(index).all():
            raise FloatingPointError(
                "a prediction index is not representable in float64"
            )
        if not np.isfinite(fitted).all():
            raise FloatingPointError(
                "a fitted power mean is not representable in float64"
            )
        if np.any(fitted == 0.0):
            raise FloatingPointError(
                "a fitted power mean is not representable in float64: it rounds to zero"
            )
        series = pd.Series(fitted, index=self._sample_view.labels, name=kind)
        series.attrs["label"] = naming.prediction_label(omega)
        series.attrs["omega"] = omega
        if self._bias is not None:
            series.attrs["evaluation_point"] = "original uncorrected estimating root"
            series.attrs["nuisance_correction"] = "not_supplied"
        return series

    def summary(self, level: float = 0.95) -> str:
        """Render actual sample, formulation, coverage and retained inference."""
        normal_quantile(level)
        lines = [
            "powermean",
            "=" * WIDTH,
            naming.estimator_name(self.estimator),
            f"Fixed effects: {self.estimator.fixed_effects.to_dict()}",
            f"Observations: {self.sample.count} of {self.sample.nobs}",
            f"Status: {self.status}; certified: {self.certified}",
            f"Coverage: {self.provenance.coverage}",
        ]
        if self.certified:
            table = self.tidy(level)
            if self._bias is not None:
                lines.append(
                    "WZ-corrected reported coefficients; certificate and first-order "
                    "inference at the uncorrected root."
                )
                lines.append(
                    "Fixed effects and predictions remain original, uncorrected "
                    "estimating-root outputs."
                )
                lines.append(table[["uncorrected", "bias", "estimate"]].to_string())
            lines.extend(_table(table, level))
        else:
            lines.append("Inference withheld: the retained solution is not certified.")
            if self._bias is not None:
                lines.append("WZ correction withheld: no certified uncorrected root.")
        return "\n".join(lines)

    def _root_trace(self) -> dict[str, object]:
        """The selected-root record's trace fields; none on another route.

        The operation of the root stages and the organization requested at the fit
        stay apart; the initializer, its certification, the path and the work are
        read from the retained record, never re-formed.
        """
        if self.retained_result is None:
            return {}
        root = self.retained_result.envelope.root
        if root is None:
            return {}
        initializer = root.initializer
        return {
            "operation": root.operation,
            "initializer": initializer.kind,
            "initializer_variance": (
                None if initializer.variance is None else initializer.variance.tau
            ),
            "initializer_certified": initializer.certified,
            "initializer_status": initializer.status,
            "initializer_gap": initializer.gap,
            "merit_metric": root.root_options.merit_metric,
            "merit_formed": root.merit is not None,
            "stages": root.stage_variances,
            "target_stage": root.target_stage,
            "continuation_used": root.continuation_used,
            "iterations": root.iterations,
            "evaluations": root.evaluations,
            "root_iterations": root.root_iterations,
            "root_evaluations": root.root_evaluations,
        }

    def _build_provenance(self) -> Provenance:
        cov_label = naming.coverage(self.estimator)
        # Ordinary scalar provenance is not a request for the canonical certificate
        # or historical presentation graphs.  Rich access remains detached outward.
        compact = (
            RetainedAbsorbedEnvelope._certificate_data(self.retained_result.envelope)
            if type(self.retained_result) is RetainedAbsorbedResult else None
        )
        cert = (
            None if self.retained_result is None else
            self.retained_result.certificate if compact is None else compact
        )
        outcome = (
            self.unsolved_outcome
            if self.retained_result is None
            else self.retained_result.outcome
        )
        opts_dict: dict[str, object] = {
            "sample_request": (None if outcome is None else asdict(outcome.request)),
            "solver_options": self.solver_options.to_dict(),
            "absorbed": self.organization,
            "absorbed_options": self.absorbed_options.to_dict(),
            "max_dense_bytes": self.max_dense_bytes,
        }
        if self._bias is not None:
            opts_dict["bias_correction"] = self._bias.to_dict()
        if self.operation is not None and self.root_options is not None:
            opts_dict["operation"] = self.operation
            opts_dict["root_options"] = self.root_options.to_dict()
        cov_dict: Mapping[str, object] | None = None
        record = self._reported_inference()
        if self.certified and record is not None:
            cov_dict = _frozen(
                {str(k): v for k, v in _vcov_attrs_from_retained(record).items()}
            )
        sample_rec = _frozen(
            {
                "fingerprint": self._sample_view.fingerprint,
                "nobs": self._sample_view.nobs,
                "count": self._sample_view.count,
                "effective_n": self._sample_view.effective_n,
                "counts": dict(self._sample_view.counts),
                "missing": dict(self._sample_view.missing),
                "subset_missing": self.subset_missing,
                "positive_only": self._sample_view.positive_only,
                "conditioned": self._sample_view.conditioned,
                "weight_kind": self._sample_view.weight_kind,
                "weights_column": self.columns.weights,
                "offset_column": self.columns.offset,
                "exposure_column": self.columns.exposure,
                "offset_expression": self.offset_design is not None,
                "retained_classifications": self._sample_view.cluster_names,
                "dropped_singleton": self._sample_view.dropped_singleton,
                "dropped_separated": self._sample_view.dropped_separated,
                "dropped_collinear": self._sample_view.dropped_collinear,
            }
        )
        formula_rec = _frozen(
            {
                "terms": tuple((term, tuple(cols)) for term, cols in self.design.terms),
                "variables": tuple(self.design.variables),
                "formulaic": self.design.formulaic_version,
                "names": self._sample_view.column_names,
                "fixed_effects": tuple(
                    "^".join(term) for term in self.parts.fixed_effects
                ),
            }
        )
        req_start = (
            None
            if self.requested_start is None
            else _frozen(
                {
                    "coefficients": tuple(self.requested_start.tolist()),
                    "coordinates": "reported",
                }
            )
        )
        history = (
            RetainedAbsorbedEnvelope._history_data(self.retained_result.envelope)
            if type(self.retained_result) is RetainedAbsorbedResult else None
        )
        trace_rec = (
            _frozen(
                {
                    "organization": self.organization,
                    "n_attempts": (
                        len(self.retained_result.history.attempts)
                        if history is None else len(history.attempts)
                    ),
                    "reason": (
                        self.retained_result.history.reason
                        if history is None else history.reason
                    ),
                    **self._root_trace(),
                }
            )
            if self.retained_result is not None
            else None
        )
        solver = (
            "linear absorbed least squares"
            if self.organization == "linear"
            else f"newton ({self.organization})"
        )
        if self.operation is not None:
            solver = (
                f"root newton ({self.operation}; requested organization "
                f"{self.organization!r})"
            )
        return Provenance(
            versions=versions(),
            implementation=IMPLEMENTATION,
            solver=solver,
            carried_files=carried_files(None),
            options=_frozen(opts_dict),
            trace=trace_rec,
            linear=None,
            status=self.status,
            level=None if cert is None else cert.level,
            certified=self.certified,
            degenerate=(
                bool(cert is not None and cert.degenerate)
                if self._inference_override is None
                else bool(np.any(self._inference_override.standard_errors == 0))
            ),
            coverage=_frozen({"label": cov_label.label, "basis": cov_label.basis}),
            estimator=_frozen(self.estimator.to_dict()),
            sample=sample_rec,
            formula=formula_rec,
            covariance=cov_dict,
            references=citations.references(self.estimator),
            requested_start=req_start,
            failure=None
            if self.certified
            else _frozen(
                {
                    "status": self.status,
                    "message": None
                    if self.retained_result is None
                    else self.retained_result.history.message
                    if history is None else history.message,
                }
            ),
        )
