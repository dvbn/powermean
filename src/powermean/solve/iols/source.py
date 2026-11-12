"""Runtime source receipts for IOLS; digests record bytes, not numerical acceptance."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import ModuleType

SOURCE_FILES = (
    "solve/iols/__init__.py",
    "solve/iols/_numeric.py",
    "solve/iols/fe.py",
    "solve/iols/kernel_records.py",
    "solve/iols/kernel_state.py",
    "solve/iols/kernel_grid.py",
    "solve/iols/kernel_step.py",
    "solve/iols/kernel_attempt.py",
    "solve/iols/kernel_loop.py",
    "solve/iols/kernel_phase.py",
    "solve/iols/kernel.py",
    "solve/iols/reference.py",
    "solve/iols/options.py",
    "solve/iols/direct_route.py",
    "solve/iols/normalization.py",
    "solve/iols/adapter.py",
    "solve/iols/candidate.py",
    "solve/iols/records.py",
    "solve/iols/source.py",
    "interface/options.py",
    "interface/preparation.py",
    "interface/dispatch.py",
    "data/computation.py",
    "model/rows.py",
    "model/variance.py",
    "model/order_scale.py",
    "systems/joint.py",
    "solve/certificate.py",
    "solve/assessment.py",
    "solve/reported.py",
    "inference/provider.py",
    "inference/contributions.py",
    "inference/bread.py",
    "inference/vcov.py",
    "inference/convention.py",
    "inference/ssc.py",
    "inference/numeric.py",
    "results/point.py",
    "results/covariance.py",
    "solve/status.py",
    "results/handoff.py",
    "results/owned.py",
    "results/provenance.py",
    "results/retained.py",
    "api.py",
)


def source_receipt(native: ModuleType | None) -> dict[str, object]:
    """Pin installed implementation/caller files and the actual imported extension."""
    root = Path(__file__).resolve().parents[2]
    paths = {name: root / name for name in SOURCE_FILES}
    if native is not None:
        filename = getattr(native, "__file__", None)
        if not isinstance(filename, str):
            raise ImportError("native IOLS module has no file identity")
        paths["_kernel_cpp"] = Path(filename)
        paths["_native/iols/bind.cpp"] = root / "_native/iols/bind.cpp"
    receipt: dict[str, object] = {}
    for name, path in paths.items():
        contents = path.read_bytes()
        receipt[name] = {
            "bytes": len(contents),
            "sha256": hashlib.sha256(contents).hexdigest(),
        }
    return receipt
