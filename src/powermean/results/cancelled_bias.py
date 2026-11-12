"""Small owned BBP reporting records, separate from the uncorrected root state."""

from dataclasses import dataclass
from typing import Literal

from powermean.fe._cancel_storage import FloatStorage
from powermean.model.variance import FloatArray


@dataclass(frozen=True, slots=True)
class BiasPoint:
    """Only reporting vectors are retained, not the dense correction fit graph."""

    uncorrected: FloatStorage
    bias: FloatStorage
    corrected: FloatStorage
    effects: FloatStorage


@dataclass(frozen=True, slots=True)
class CancelledBias:
    """Requested sampling domain and conditional correction at the certified root."""

    request: str
    method: Literal["short_group", "cell"]
    sampling_name: str
    n_units: int
    point: BiasPoint | None
    workspace: tuple[tuple[str, int], ...] = ()

    @property
    def label(self) -> str:
        """The existing general equation correction, distinct from published WZ."""
        return "BBP"

    def coefficients(self) -> FloatArray:
        """Fresh coefficient metadata over immutable corrected numeric storage."""
        if self.point is None:
            raise ValueError("BBP correction withheld for an uncertified root")
        return self.point.corrected.view()

    def retained_effects(self) -> FloatArray:
        """Retained effects corrected in the same identified T coordinates."""
        if self.point is None:
            raise ValueError("BBP correction withheld for an uncertified root")
        return self.point.effects.view()

    def to_dict(self) -> dict[str, object]:
        """Detached triplet and accounting; the coefficient certificate is unchanged."""
        result: dict[str, object] = {
            "request": self.request,
            "method": self.method,
            "sampling_name": self.sampling_name,
            "n_units": self.n_units,
            "status": "applied" if self.point is not None else "withheld_uncertified",
            "evaluation_point": (
                "certified_uncorrected_root" if self.point is not None else None
            ),
            "covariance_point": "uncorrected_root" if self.point is not None else None,
            "covariance": "original_requested_first_order",
            "workspace": dict(self.workspace),
            "evaluations_begun": int(self.point is not None),
            "evaluations_completed": int(self.point is not None),
        }
        if self.point is not None:
            result.update(
                uncorrected=tuple(self.point.uncorrected.view().tolist()),
                bias=tuple(self.point.bias.view().tolist()),
                corrected=tuple(self.point.corrected.view().tolist()),
            )
        return result
