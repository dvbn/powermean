"""Absorbed covariance requests: legacy constructor and full request identity."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from powermean.data.inference_conventions import (
    PSD_TREATMENTS,
    ConventionValues,
    PsdTreatment,
)
from powermean.data.inference_requests import (
    HC_METHODS,
    MULTIWAY_FORMS,
    ClusterRequest,
    HCMethod,
    Multiway,
    PairAdjustment,
)
from powermean.data.request_values import OMITTED, Omitted

IntegerArray = NDArray[np.integer[Any]]
Kind = Literal["robust", "cluster", "hc"]
Bread = Literal["observed", "expected"]
BREAD_VALUES: tuple[str, ...] = ("observed", "expected")
PRESET_NAMES: tuple[str, ...] = ("stata", "none", "regression")
MULTIWAY_VALUES: tuple[str, ...] = MULTIWAY_FORMS
PSD_VALUES: tuple[str, ...] = PSD_TREATMENTS


def _integers(values: object, name: str) -> IntegerArray:
    """An owned read-only copy of a one-dimensional NumPy integer array."""
    if not isinstance(values, np.ndarray) or values.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a NumPy integer array")
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return cast("IntegerArray", np.frombuffer(values.tobytes(), dtype=values.dtype))


def _check_ssc_value(value: object, *, allow_omitted: bool) -> None:
    """Validate ``convention`` or ``ssc`` on :class:`CovarianceRequest`."""
    if allow_omitted and value is OMITTED:
        return
    if type(value) is ConventionValues:
        return
    if isinstance(value, str) and value in PRESET_NAMES:
        return
    raise ValueError("ssc convention must be a preset name or ConventionValues")


def _ssc_token(value: str | ConventionValues | Omitted) -> object:
    """Return an immutable value token for ``value``."""
    if value is OMITTED:
        return ("omitted",)
    if type(value) is ConventionValues:
        return (
            "values",
            value.robust,
            value.cluster,
            value.cluster_count,
            value.reference,
            value.psd,
        )
    return ("preset", str(value))


@dataclass(frozen=True, eq=False)
class CovarianceRequest:
    """The requested fit-time or post-fit covariance for a joint absorbed FE model.

    Supports ``"robust"``, one-way or multi-way ``"cluster"`` (``"cgm"`` or ``"ddg"``
    with ``psd="cgm" | "none"``), and ``"hc"`` (``HC0``..``HC5``, ``HC4m``) under
    ``bread="observed" | "expected"`` and presets/custom ``ssc``.  Calling
    ``CovarianceRequest("robust")`` or
    ``CovarianceRequest("cluster", name, codes, positions)`` preserves the legacy
    positional signature and identity.
    """

    kind: Kind = "robust"
    name: str | None = None
    codes: IntegerArray | None = None
    positions: IntegerArray | None = None
    convention: str | ConventionValues = "stata"
    clusters: tuple[tuple[str, IntegerArray], ...] = ()
    method: HCMethod | None = None
    bread: Bread = "observed"
    multiway: Multiway | Omitted = OMITTED
    psd: PsdTreatment | Omitted = OMITTED
    ssc: str | ConventionValues | Omitted = OMITTED
    ssc_inherited: bool = False
    adjustment: PairAdjustment | Omitted = OMITTED

    def __post_init__(self) -> None:
        """Check the request and own its arrays."""
        self._options()
        if self.kind == "robust":
            self._robust()
        elif self.kind == "hc":
            self._hc()
        else:
            self._clustered()
        if self.adjustment is not OMITTED:
            if self.kind != "cluster":
                raise ValueError("adjustment belongs to a cluster request")
            ClusterRequest(
                tuple(name for name, _ in self.clusters),
                self.multiway,
                self.psd,
                self.adjustment,
            )

    def _options(self) -> None:
        """Validate common options and own custom convention wrappers."""
        if self.kind not in ("robust", "cluster", "hc"):
            raise ValueError("a request kind must be 'robust', 'cluster', or 'hc'")
        if self.bread not in BREAD_VALUES:
            raise ValueError("bread must be 'observed' or 'expected'")
        if not isinstance(self.ssc_inherited, bool):
            raise ValueError("ssc_inherited must be a bool")
        _check_ssc_value(self.convention, allow_omitted=False)
        _check_ssc_value(self.ssc, allow_omitted=True)
        if (
            self.ssc is OMITTED
            and self.convention != "stata"
            and not self.ssc_inherited
        ):
            object.__setattr__(self, "ssc", self.convention)
        elif (
            self.ssc is not OMITTED
            and self.convention == "stata"
            and self.adjustment != "wz"
            and self.ssc != "stata"
        ):
            object.__setattr__(self, "convention", self.ssc)
        if self.multiway is not OMITTED and self.multiway not in MULTIWAY_VALUES:
            raise ValueError("multiway must be 'cgm' or 'ddg'")
        if self.psd is not OMITTED and self.psd not in PSD_VALUES:
            raise ValueError("psd must be 'cgm' or 'none'")
        if type(self.convention) is ConventionValues:
            object.__setattr__(self, "convention", replace(self.convention))
        if type(self.ssc) is ConventionValues:
            object.__setattr__(self, "ssc", replace(self.ssc))

    def _robust(self) -> None:
        """A robust request has neither cluster nor HC options."""
        if (self.name, self.codes, self.positions, self.method) != (
            None,
            None,
            None,
            None,
        ) or self.clusters:
            raise ValueError("a robust request names no cluster or HC method")
        if self.multiway is not OMITTED or self.psd is not OMITTED:
            raise ValueError("a robust request takes no multiway or psd option")

    def _hc(self) -> None:
        """A named HC request owns its method and no cluster options."""
        if self.method not in HC_METHODS:
            raise ValueError("an hc request requires a valid HC method")
        if (self.name, self.codes, self.positions) != (
            None,
            None,
            None,
        ) or self.clusters:
            raise ValueError("an hc request names no cluster")
        if self.multiway is not OMITTED or self.psd is not OMITTED:
            raise ValueError("an hc request takes no multiway or psd option")

    def _normalized_clusters(
        self, positions: IntegerArray
    ) -> tuple[tuple[str, IntegerArray], ...]:
        """Own an explicitly supplied tuple of aligned cluster dimensions."""
        if not isinstance(self.clusters, tuple):
            raise ValueError("clusters must be a tuple of (name, codes) pairs")
        norm_list: list[tuple[str, IntegerArray]] = []
        for item in self.clusters:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("each cluster dimension must be (name, codes)")
            cname, ccodes = item
            if not isinstance(cname, str) or not cname:
                raise ValueError("a cluster request needs a nonempty name")
            owned_c = _integers(ccodes, "codes")
            if owned_c.shape != positions.shape:
                raise ValueError("one cluster code per bound row")
            norm_list.append((cname, owned_c))
        return tuple(norm_list)

    def _clustered(self) -> None:
        """Own legacy one-way or explicit multiway cluster arrays."""
        if self.method is not None:
            raise ValueError("a cluster request takes no HC method")
        positions = _integers(self.positions, "positions")
        if self.clusters:
            norm_clusters = self._normalized_clusters(positions)
            if len(norm_clusters) == 1:
                cname0, ccodes0 = norm_clusters[0]
                if self.name is not None and self.name != cname0:
                    raise ValueError("name does not match single cluster dimension")
                if self.codes is not None and not np.array_equal(self.codes, ccodes0):
                    raise ValueError("codes does not match single cluster dimension")
                object.__setattr__(self, "name", cname0)
                object.__setattr__(self, "codes", ccodes0)
            elif self.name is not None or self.codes is not None:
                raise ValueError(
                    "multiway cluster request uses clusters, not name/codes"
                )
            object.__setattr__(self, "clusters", norm_clusters)
            object.__setattr__(self, "positions", positions)
            return
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("a cluster request needs a nonempty name")
        codes = _integers(self.codes, "codes")
        if codes.shape != positions.shape:
            raise ValueError("one cluster code per bound row")
        object.__setattr__(self, "codes", codes)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "clusters", ((self.name, codes),))

    @property
    def is_default_stata(self) -> bool:
        """Whether this request uses the default observed Stata path."""
        return (
            self.kind in ("robust", "cluster")
            and len(self.clusters) <= 1
            and self.bread == "observed"
            and self.method is None
            and self.multiway is OMITTED
            and self.psd is OMITTED
            and self.ssc is OMITTED
            and not self.ssc_inherited
            and self.convention == "stata"
            and self.adjustment is OMITTED
        )

    @property
    def uses_multiway(self) -> bool:
        """Whether this cluster request uses the multiway kernel."""
        return self.kind == "cluster" and (
            len(self.clusters) >= 2 or self.multiway is not OMITTED
        )

    def identity(self) -> tuple[object, ...]:
        """The request by value: kind, name, codes and positions bytes, convention."""
        if self.is_default_stata:
            arrays = tuple(
                None if a is None else (a.dtype.str, a.tobytes())
                for a in (self.codes, self.positions)
            )
            return (self.kind, self.name, *arrays, self.convention)
        clusters_id = tuple(
            (cname, ccodes.dtype.str, ccodes.tobytes())
            for cname, ccodes in self.clusters
        )
        pos_id = (
            None
            if self.positions is None
            else (self.positions.dtype.str, self.positions.tobytes())
        )
        mw_id = ("omitted",) if self.multiway is OMITTED else ("given", self.multiway)
        psd_id = ("omitted",) if self.psd is OMITTED else ("given", self.psd)
        ssc_id = (
            ("omitted", _ssc_token(self.convention))
            if self.ssc is OMITTED
            else ("given", _ssc_token(self.ssc))
        )
        identity = (
            self.kind,
            self.method,
            self.bread,
            clusters_id,
            pos_id,
            mw_id,
            psd_id,
            ssc_id,
            self.ssc_inherited,
        )
        return (
            identity
            if self.adjustment is OMITTED
            else (*identity, ("adjustment", self.adjustment))
        )
