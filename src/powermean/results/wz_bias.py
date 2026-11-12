"""Published WZ reporting triplet; no corrected nuisance vector is supplied."""

from dataclasses import asdict, dataclass
from typing import Literal

from powermean.errors import SpecificationError
from powermean.fe._cancel_storage import FloatStorage
from powermean.inference.wz_coefficient import CoefficientBias
from powermean.model.variance import FloatArray


def nuisance_policy(consumer: str) -> SpecificationError:
    """Refuse unavailable nuisance output without inventing corrected effects."""
    return SpecificationError(
        "specification.unsupported_inference",
        {
            "method": consumer,
            "case": (
                "WZ supplies only a slope correction; corrected nuisance "
                "effects and predictions are not supplied"
            ),
        },
    )


@dataclass(frozen=True, slots=True)
class WZBiasPoint:
    """Owned reported slopes; no fictitious corrected fixed effects."""

    uncorrected: FloatStorage
    bias: FloatStorage
    corrected: FloatStorage


@dataclass(frozen=True, slots=True)
class WZBias:
    """Discriminated WZ result evaluated once at the certified original root."""

    sampling_name: str | None
    point: WZBiasPoint | None
    operator: CoefficientBias | None = None
    method: Literal["wz"] = "wz"
    request: Literal["wz"] = "wz"
    label: str = "WZ"

    def coefficients(self) -> FloatArray:
        """Detached corrected slopes, without changing the root or its V/SE."""
        if self.point is None:
            raise ValueError("WZ correction withheld for an uncertified root")
        return self.point.corrected.view()

    def to_dict(self) -> dict[str, object]:
        """Detached method, factor, root/first-order ownership and work metadata."""
        result: dict[str, object] = {
            "request": self.request,
            "method": self.method,
            "sampling_name": self.sampling_name,
            "status": "applied" if self.point is not None else "withheld_uncertified",
            "evaluation_point": "certified_uncorrected_root"
            if self.point is not None
            else None,
            "covariance_point": "uncorrected_root" if self.point is not None else None,
            "covariance": "original_requested_first_order",
            "nuisance_correction": "not_supplied",
            "report_mapping": "OrderScale_once_from_input_slopes",
            "evaluations_begun": int(self.point is not None),
            "evaluations_completed": int(self.point is not None),
        }
        if self.operator is not None:
            op = self.operator
            result.update(
                source="WZ CWP11/21 Section 3.4, Appendix A.2.1; formal G=-DH",
                binding_sha256=op.binding_sha256,
                route=op.route,
                nodes=op.nodes,
                periods=op.periods,
                slope_names=op.names,
                factor_numerator=op.nodes,
                factor_denominator=op.nodes - 1,
                factor_owner="WZ_coefficient_only",
                rcond_min=op.rcond_min,
                nuisance_rcond=op.nuisance_rcond,
                slope_rcond=op.slope_rcond,
                minimum_group_rcond=op.minimum_group_rcond,
                reservation=asdict(op.reservation),
            )
        if self.point is not None:
            result.update(
                uncorrected=tuple(self.point.uncorrected.view().tolist()),
                bias=tuple(self.point.bias.view().tolist()),
                corrected=tuple(self.point.corrected.view().tolist()),
            )
        return result
