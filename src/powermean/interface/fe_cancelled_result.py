"""Result backends for joint absorbed fixed-effects and selected-cancellation fits."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import asdict
from typing import Unpack

import numpy as np
import pandas as pd

from powermean._prepare.compose import _owned_index
from powermean.data.inference_requests import (
    ClusterRequest,
    HCRequest,
)
from powermean.effects.targets import ReportingContext, ReportingPoint
from powermean.errors import ConvergenceError, SpecificationError
from powermean.inference.convention import ClusterCodes
from powermean.inference.wz_binding import WZSource
from powermean.interface.fe_result_backends import (
    CancelledState,
    FEView,
    _bind_cancelled_cluster_codes,
    _parse_vcov_spec_for_absorbed,
    _target_reporting_context,
    _tidy_frame,
    _vcov_frame_from_retained,
)
from powermean.results import citations, naming
from powermean.results.cancelled_recovery import cancelled_effect
from powermean.results.fe_helpers import (
    PredictionOptions,
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
from powermean.results.wz_bias import WZBias, nuisance_policy
from powermean.results.wz_inference import cancelled_wz
from powermean.solve.cancelled_records import CancelledCertificate


class CancelledResultBackend:
    """Backend adapting a selected-cancellation fit to the public ``Result``."""

    def __init__(self, view: FEView, state: CancelledState) -> None:
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
        envelope = state.envelope
        dependence = state.dependence
        grouping = state.grouping
        controls = state.controls
        self.envelope = envelope
        self.dependence = dependence
        self.grouping = grouping
        self.estimator = estimator
        self.order_sample = order_sample
        self.input_labels = _owned_index(input_labels)
        self.columns = columns
        self.subset_missing = int(subset_missing)
        self.parts = parts
        self.design = design
        self.offset_design = offset_design
        self.solver_options = solver_options
        self.controls = controls
        self.requested_absorbed_options = state.requested_absorbed_options
        self._bias = state.bias_correction
        self.fit_vcov_spec = fit_vcov_spec
        self.requested_start = (
            None if requested_start is None else owned_array(requested_start)
        )
        self._sample_view = FESampleView(
            order_sample,
            self.input_labels,
            columns,
            subset_missing,
            dependence=dependence.retained_names,
            cancelled_envelope=envelope,
        )
        self.status = envelope.status
        self._provenance = self._build_provenance()

    @property
    def certified(self) -> bool:
        """Return the certified from the actual retained state, detached outward."""
        return bool(self.envelope.accepted)

    @property
    def certificate(self) -> CancelledCertificate:
        """Return the certificate from the actual retained state, detached outward."""
        return self.envelope.certificate

    @property
    def sample(self) -> FESampleView:
        """Return the sample from the actual retained state, detached outward."""
        return self._sample_view

    @property
    def provenance(self) -> Provenance:
        """Return the provenance from the actual retained state, detached outward."""
        return self._provenance

    def _names(self) -> pd.Index[str]:
        return pd.Index(list(self.envelope.slope_names), name="term")

    def _withheld(self) -> ConvergenceError:
        cert = self.certificate
        last_state: object = cert
        if self.envelope.attempts is not None:
            last_state = self.envelope.attempts
        elif self.envelope.tau1_result is not None:
            last_state = self.envelope.tau1_result
        elif len(self.envelope.root_stages) > 0:
            last_state = self.envelope.root_stages
        return ConvergenceError(
            "convergence.not_certified",
            status=self.status,
            last_state=last_state,
            certificate=cert,
        )

    def _has_point(self) -> bool:
        """Distinguish an evaluated candidate from a diagnostic fallback point."""
        trace = self.envelope.tau1_result
        if trace is None:
            return True
        candidate = trace.candidate
        return (
            candidate is not None
            and candidate.attempts[candidate.retained].point is not None
        )

    def coef(self) -> pd.Series[float]:
        """Return the reported point; certification stays at the uncorrected root."""
        if not self._has_point():
            raise self._withheld()
        if self._bias is not None and not self.certified:
            raise self._withheld()
        try:
            beta = (
                self.envelope.coef()
                if self._bias is None
                else self._bias.coefficients()
            )
        except FloatingPointError as exc:
            raise self._withheld() from exc
        if not np.isfinite(beta).all():
            raise self._withheld()
        return pd.Series(
            np.array(beta, dtype=np.float64, copy=True),
            index=self._names(),
            name="estimate",
        )

    def se(self) -> pd.Series[float]:
        """Return the se from the actual retained state, detached outward."""
        if not self.certified:
            raise self._withheld()
        return pd.Series(
            np.array(
                self.envelope.se()
                if self._inference_override is None
                else self._inference_override.standard_errors,
                dtype=np.float64,
                copy=True,
            ),
            index=self._names(),
            name="std_error",
        )

    def _initial_scale(self) -> RetainedInference | RetainedHC | None:
        """Read the view covariance; the original fit certificate stays intact."""
        return (
            self.envelope.initial_scale
            if self._inference_override is None
            else self._inference_override
        )

    def _reporting_context(self) -> ReportingContext:
        """Bind actual BBP/WZ beta and the active first-order root covariance once."""
        if not self.certified:
            raise self._withheld()
        record = self._initial_scale()
        if record is None:
            raise self._withheld()
        coefficient_point = (
            "certified uncorrected root"
            if self._bias is None
            else f"{self._bias.label} corrected reported coefficients"
        )
        return _target_reporting_context(
            self.envelope.slope_names,
            np.asarray(self.coef().to_numpy(), dtype=np.float64),
            record,
            self.estimator.mean.omega,
            ReportingPoint(self.status, coefficient_point),
        )

    def vcov(self, spec: object = None) -> pd.DataFrame:
        """Return detached covariance from the retained uncorrected state."""
        record = self._inference(spec)
        return _vcov_frame_from_retained(record, self.envelope.slope_names)

    def with_vcov(self, spec: object) -> CancelledResultBackend:
        """Return a covariance view without refitting or changing certification."""
        record = self._inference(spec)
        result = copy.copy(self)
        result._inference_override = record
        result._provenance = result._build_provenance()
        return result

    def _inference(self, spec: object = None) -> RetainedInference | RetainedHC:
        """Return the vcov from the actual retained state, detached outward."""
        if not self.certified:
            raise self._withheld()
        _init_vcov, init_ssc, init_bread = self.fit_vcov_spec
        if spec is None:
            scale = self._initial_scale()
            if scale is None:
                raise self._withheld()
            return scale

        fe_vcov, ssc_val, bread_val = _parse_vcov_spec_for_absorbed(
            spec, init_ssc, init_bread
        )
        if isinstance(fe_vcov, HCRequest):
            hc_req = HCInferenceRequest.from_hc_request(
                fe_vcov,
                ssc=ssc_val,
                bread=bread_val,
            )
            return compute_retained_request(self.envelope, hc_req)

        if fe_vcov == "cluster_cancelled":
            default_c = ClusterCodes(
                self.envelope.cancelled_name,
                self.envelope.geometry.structure.cancel_codes,
            )
            inf_req = RetainedInferenceRequest(
                "cluster",
                (default_c,),
                ssc_val,
                bread_val,
            )
        elif isinstance(fe_vcov, ClusterRequest):
            clusters = _bind_cancelled_cluster_codes(
                self.envelope, self.dependence, self.grouping, fe_vcov
            )
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
            return cancelled_wz(
                self.envelope,
                inf_req,
                WZSource(
                    self.parts.fixed_effects,
                    self.grouping,
                    tuple(self.input_labels.take(self.order_sample.positions).tolist()),
                ),
                self.requested_absorbed_options,
            )
        return compute_retained_request(self.envelope, inf_req)

    def tidy(self, level: float = 0.95) -> pd.DataFrame:
        """Return the tidy from the actual retained state, detached outward."""
        normal_quantile(level)
        if not self.certified:
            raise self._withheld()
        scale = self._initial_scale()
        if scale is None:
            raise self._withheld()
        table = _tidy_frame(
            self.envelope.slope_names,
            self._bias.coefficients()
            if self._bias is not None
            else self.envelope.coef(),
            self.envelope.se()
            if self._inference_override is None
            else self._inference_override.standard_errors,
            level,
            scale.applied.reference,
        )
        if self._bias is not None:
            assert self._bias.point is not None
            table["uncorrected"] = self._bias.point.uncorrected.view()
            table["bias"] = self._bias.point.bias.view()
            table.attrs["coefficient_point"] = (
                f"{self._bias.label} corrected reported coefficients"
            )
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
        if not self.certified:
            raise self._withheld()
        inf_arr = self.envelope.influence(rows=False)
        return pd.DataFrame(
            np.array(inf_arr, dtype=np.float64, copy=True),
            index=pd.Index(range(inf_arr.shape[0]), name="group"),
            columns=self._names(),
        )

    def fixef(self) -> dict[str, pd.Series[float]]:
        """Return retained and recovered cancelled effects at one coherent point."""
        if not self.certified:
            raise self._withheld()
        block_map = {b.name: b for b in self.grouping.blocks}
        ret_eff = (
            self.envelope.retained_effects()
            if self._bias is None or isinstance(self._bias, WZBias)
            else self._bias.retained_effects()
        )
        sizes = self.envelope.geometry.structure.block_sizes
        out: dict[str, pd.Series[float]] = {}
        offset = 0
        for r_name, b_size in zip(self.envelope.retained_names, sizes, strict=True):
            block = block_map[r_name]
            vals = np.array(
                ret_eff[offset : offset + b_size], dtype=np.float64, copy=True
            )
            offset += b_size
            out[r_name] = pd.Series(
                vals,
                index=pd.Index(
                    list(block.levels),
                    dtype=object,
                    name=r_name,
                    tupleize_cols=False,
                ),
                name=r_name,
            )
            if isinstance(self._bias, WZBias):
                out[r_name].attrs.update(
                    evaluation_point="original uncorrected estimating root",
                    nuisance_correction="not_supplied",
                )
            elif self._bias is not None:
                out[r_name].attrs.update(
                    point="BBP algebraic shift under the original T normalization",
                    certificate_point="uncorrected estimating root",
                    incidental_effect_consistency="not asserted",
                )
        out[self.envelope.cancelled_name] = cancelled_effect(
            self.order_sample, self.envelope, self._bias
        )
        return {name: out[name] for name in self.envelope.block_order}

    def predict(
        self,
        data: pd.DataFrame | None = None,
        **options: Unpack[PredictionOptions],
    ) -> pd.Series[float]:
        """Return the predict from the actual retained state, detached outward."""
        prediction_kind(options)
        if data is not None and not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not self.certified:
            raise self._withheld()
        if isinstance(self._bias, WZBias):
            raise nuisance_policy("predict")
        raise SpecificationError(
            "specification.invalid_value",
            {
                "argument": "predict",
                "value": "power_mean",
                "valid": (
                    "a model supporting prediction "
                    "(prediction after cancellation remains unavailable)"
                ),
            },
        )

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
                    f"{self._bias.label}-corrected reported coefficients; "
                    "certificate and first-order "
                    "inference at the uncorrected estimating root."
                )
                lines.append(table[["uncorrected", "bias", "estimate"]].to_string())
            lines.extend(_table(table, level))
        else:
            lines.append("Inference withheld: the retained solution is not certified.")
            if self._bias is not None:
                lines.append(
                    f"{self._bias.label} correction withheld: "
                    "no certified uncorrected root."
                )
        return "\n".join(lines)

    def _build_provenance(self) -> Provenance:
        cov_label = naming.coverage(self.estimator)
        root_opts = self.envelope.root_options
        opts_dict = {
            "controls": asdict(self.controls),
            "solver_options": self.solver_options.to_dict(),
            "cancel": self.envelope.cancelled_name,
            "route": self.envelope.route,
            "root_selection": self.envelope.root_selection,
            "root_options": (None if root_opts is None else root_opts.to_dict()),
            "requested_absorbed_options": self.requested_absorbed_options.to_dict(),
            "absorbed": (
                self.requested_absorbed_options.absorbed
                if self.envelope.route == "absorbed_tau1"
                else None
            ),
            "absorbed_options": (
                self.requested_absorbed_options.to_dict()
                if self.envelope.route == "absorbed_tau1"
                else None
            ),
        }
        if self._bias is not None:
            opts_dict["bias_correction"] = self._bias.to_dict()
        cov_dict: Mapping[str, object] | None = None
        scale = self._initial_scale()
        if self.certified and scale is not None:
            vcov_df = _vcov_frame_from_retained(
                scale,
                self.envelope.slope_names,
            )
            cov_dict = _frozen({str(k): v for k, v in vcov_df.attrs.items()})
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
                "cancelled": self.envelope.cancelled_name,
                "retained_fixed_effects": self.envelope.retained_names,
                "n_groups": self.envelope.count,
                "omitted_slopes": self.envelope.omitted_slopes,
            }
        )
        formula_rec = _frozen(
            {
                "terms": tuple((term, tuple(cols)) for term, cols in self.design.terms),
                "variables": tuple(self.design.variables),
                "formulaic": self.design.formulaic_version,
                "names": self.envelope.slope_names,
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
        trace_rec = _frozen(
            {
                "route": self.envelope.route,
                "root_selection": self.envelope.root_selection,
                "continuation_used": self.envelope.continuation_used,
                "n_root_stages": len(self.envelope.root_stages),
                "has_evaluated_point": self._has_point(),
            }
        )
        return Provenance(
            versions=versions(),
            implementation=IMPLEMENTATION,
            solver=f"newton ({self.envelope.route})",
            carried_files=carried_files(None),
            options=_frozen(opts_dict),
            trace=trace_rec,
            linear=None,
            status=self.status,
            level=self.envelope.level,
            certified=self.certified,
            degenerate=bool(scale is not None and np.any(scale.standard_errors == 0)),
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
                    "no_point": not self._has_point(),
                    "native_message": None
                    if self.envelope.tau1_result is None
                    else self.envelope.tau1_result.message,
                }
            ),
        )
