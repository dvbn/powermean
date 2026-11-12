"""Explicit BLAS/kernel thread control and provenance for absorbed Newton solves.

``threads``: default ``1``, valid values ``int >= 1``, controlling worker threads for
multi-RHS absorption and BLAS calls; single-threaded by default with ``threads=1``,
while ``threads=k`` controls BLAS and multi-column absorption.  BLAS/OpenMP thread
control is explicit when parallelism is disabled, and thread and machine provenance
is recorded in benchmarks.  Internal module: ``powermean`` exports nothing of it, and
the ``SolverOptions`` and ``AbsorbedOptions`` key sets stay unchanged.

Control without an undeclared third-party dependency.  ``pyproject.toml`` does not
declare ``threadpoolctl`` as a runtime dependency, so this module discovers loaded
BLAS, LAPACK and OpenMP shared libraries directly through the standard library's
:mod:`ctypes` dynamic-loader tables and the ``numpy.libs`` / ``scipy.libs`` wheel
directories, deduplicating by resolved C entry-point address, and calls their thread
getter/setter entry points (OpenBLAS LP64 and ILP64 with and without the ``scipy_``
symbol prefix, Intel MKL, BLIS and OpenMP), while setting and restoring the standard
thread environment variables inside a context manager.  The accepted fixed-effect
absorption and profiling kernels of :mod:`powermean.fe` run in single-threaded NumPy
(``python_kernel_threads = 1``), recorded explicitly alongside the active BLAS library
thread counts and machine/cgroup topology.

Participating providers.  Every discovered thread pool
is set, read back and restored, but numeric control is verified over the providers
that take part in the numerical work, not over every image that happens to be loaded.
The numerical entry points are the loaded third-party native extension modules
(those of NumPy, SciPy and pandas, and of any other loaded third-party package such
as Formulaic's dependency wrapt; the interpreter and its standard-library extension
modules make no BLAS, LAPACK or OpenMP call for the package).  Their closure is read
from each loaded image's own dynamic-linking records in memory (ELF ``DT_NEEDED``
and ``DT_SONAME`` through ``dl_iterate_phdr``; Mach-O ``LC_*_DYLIB``, ``LC_RPATH``
and ``LC_UUID`` through the dyld image table) and resolved to loaded images as the
loader resolves them; every dependency of an extension module or non-umbrella
library is traversed in full, while the non-provider ``reexport`` sub-libraries of
the loaded Darwin ``libSystem`` C runtime umbrella and of the loaded
``Accelerate``/``vecLib`` umbrella framework record their own ``reexport``
sub-libraries and any direct provider dependencies without expanding the OS IPC/GUI
shared-cache graph when reached solely as umbrella ``reexport`` sub-libraries.  An
image of the closure participates when a thread-control handle's getter lies in it or
when it defines a BLAS, LAPACK, OpenMP or thread-control entry point
(:data:`_PROVIDER_SYMBOLS`); an image that services such an entry point in the
default symbol scope participates too, and Apple ``vecLib``'s loaded system
``libLAPACK.dylib`` is governed through its sibling ``libBLAS.dylib`` in
``vecLib.framework`` under ``<vecLib/thread_api.h>`` (``BLASSetThreading`` scopes
BLAS and LAPACK in ``vecLib``), whereas arbitrary separate LAPACK images or private
binaries declaring nominal ``LC_ID_DYLIB`` metadata do not inherit a linked BLAS
dependency's handles.  A thread pool outside a complete closure is incidental and
stays recorded.  Any unresolved dependency, unreadable image, unmapped handle or
failed probe leaves the membership incomplete, so numeric control is not verified.
This bounds the evidence to the reachable entry points; it is not a proof that a
system framework is never used.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import importlib.machinery
import json
import os
import platform
import site
import stat
import struct
import sys
import sysconfig
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from powermean.data.sample_fe_state import FESample
from powermean.solve import newton_absorbed, newton_profiled
from powermean.solve.absorbed import (
    POSITIVE_INTEGER,
    AbsorbedOptions,
    Setting,
    changes,
    resolved,
)
from powermean.solve.newton_absorbed import IndexAssessor, IndexCandidate
from powermean.solve.newton_profiled import (
    Family,
    FullAssessor,
    ProfiledCandidate,
)
from powermean.solve.options import SolverOptions
from powermean.solve.profiled_trigger import Admission, Trigger

THREAD_DOMAINS: Mapping[str, Setting] = MappingProxyType({"threads": POSITIVE_INTEGER})

THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

_OPENBLAS_AFFIXES: tuple[tuple[str, str], ...] = (
    ("scipy_", "64_"),
    ("scipy_", ""),
    ("", "64_"),
    ("", ""),
)

_DEFAULT_CPU_ROOT = Path("/sys/devices/system/cpu")
_DARWIN_DYLD_PREFIXES: tuple[str, ...] = (
    "/System/Library/",
    "/usr/lib/",
)
_DARWIN_VECLIB_BLAS = (
    "/System/Library/Frameworks/Accelerate.framework/Versions/A/"
    "Frameworks/vecLib.framework/Versions/A/libBLAS.dylib"
)
_DARWIN_VECLIB_LAPACK = (
    "/System/Library/Frameworks/Accelerate.framework/Versions/A/"
    "Frameworks/vecLib.framework/Versions/A/libLAPACK.dylib"
)
_DARWIN_UMBRELLA_INSTALL_NAMES: frozenset[str] = frozenset(
    {
        "/usr/lib/libSystem.B.dylib",
        "/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate",
        (
            "/System/Library/Frameworks/Accelerate.framework/Versions/A/"
            "Frameworks/vecLib.framework/Versions/A/vecLib"
        ),
    }
)
_ACCELERATE_CANDIDATE_PATHS: tuple[str, ...] = (
    _DARWIN_VECLIB_BLAS,
    "/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate",
)


def _parse_positive_int_env(name: str) -> int | None:
    """Parse a positive integer environment variable if present and valid."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        parsed = int(raw.strip())
    except ValueError:
        return None
    return parsed if parsed > 0 else None


_IMPORT_TIME_VECLIB_MAX_THREADS: int | None = _parse_positive_int_env(
    "VECLIB_MAXIMUM_THREADS"
)
_STARTUP_VECLIB_MAX_THREADS: int | None = _IMPORT_TIME_VECLIB_MAX_THREADS


# --------------------------------------------------------------- option record


@dataclass(frozen=True, kw_only=True)
class ThreadOptions:
    """Explicit thread control option record.

    Keyword arguments only.  Validated against :data:`THREAD_DOMAINS` with the
    domain rules of :mod:`powermean.solve.options`.

    Parameters
    ----------
    threads : int, default 1
        Worker threads for BLAS/LAPACK calls and multi-column absorption; a
        positive integer.
    """

    threads: int = 1

    def __post_init__(self) -> None:
        """Validate ``threads`` and store the resolved exact ``int``."""
        for option in fields(self):
            value = getattr(self, option.name)
            stored = resolved(THREAD_DOMAINS, option.name, value)
            object.__setattr__(self, option.name, stored)

    def to_dict(self) -> dict[str, object]:
        """The resolved thread options by name as a new dictionary."""
        return {"threads": self.threads}

    @classmethod
    def from_dict(cls, options: object) -> ThreadOptions:
        """The record of a flat mapping of thread options over the default."""
        return cls(**changes(options, tuple(THREAD_DOMAINS)))


def resolve_thread_options(
    options: ThreadOptions | int | np.integer[Any] | Mapping[str, object] | None = None,
    **overrides: object,
) -> ThreadOptions:
    """Resolve ``options`` and keyword ``overrides`` into a :class:`ThreadOptions`."""
    if options is None:
        base = ThreadOptions()
    elif isinstance(options, ThreadOptions):
        base = options
    elif not isinstance(options, bool | np.bool_) and isinstance(
        options, int | np.integer
    ):
        base = ThreadOptions(threads=int(options))
    else:
        base = ThreadOptions.from_dict(options)
    if not overrides:
        return base
    merged: dict[str, object] = {**base.to_dict(), **overrides}
    return ThreadOptions.from_dict(merged)


# ------------------------------------------------------------ machine topology


@dataclass(frozen=True)
class CPUTopology:
    """Observed CPU, affinity and cgroup topology without reading ``/proc``.

    Attributes
    ----------
    platform_system : str
        Operating system name from :func:`platform.system`.
    platform_release : str
        Kernel release string from :func:`platform.release`.
    platform_machine : str
        Architecture string from :func:`platform.machine`.
    cpu_model : str
        Processor or microarchitecture string reported by the platform or BLAS.
    logical_cpus : int
        Logical CPU count reported by :func:`os.cpu_count`.
    affinity_cpus : tuple of int
        Sorted CPU indices in the current process affinity mask.
    sysfs_physical_cores : int or None
        Physical core count read from sysfs ``thread_siblings_list``, or ``None``
        when sysfs topology files are absent.
    physical_cores : int
        Detected physical core count on the current system.
    cgroup_cpu_max : str or None
        Raw cgroup v2 ``/sys/fs/cgroup/cpu.max`` (or v1 quota/period) string.
    cgroup_quota_cores : float or None
        Effective CPU bandwidth quota in cores (``quota / period``), or ``None``
        when unlimited or unavailable.
    cgroup_effective_cpus : str or None
        Raw ``/sys/fs/cgroup/cpuset.cpus.effective`` string when present.
    virtualized_kernel : bool
        Whether the kernel release or hypervisor flag identifies a VM guest
        whose topology does not expose bare-metal host physical cores.
    quota_throttled : bool
        Whether ``cgroup_quota_cores`` is strictly less than ``physical_cores``.
    host_physical_cores_verifiable : bool
        Whether an unthrottled all-physical-host-core run is directly verifiable
        in the current execution environment.
    effective_core_limit : float
        Minimum of ``physical_cores``, ``len(affinity_cpus)`` and any finite
        ``cgroup_quota_cores``.
    """

    platform_system: str
    platform_release: str
    platform_machine: str
    cpu_model: str
    logical_cpus: int
    affinity_cpus: tuple[int, ...]
    sysfs_physical_cores: int | None
    physical_cores: int
    cgroup_cpu_max: str | None
    cgroup_quota_cores: float | None
    cgroup_effective_cpus: str | None
    virtualized_kernel: bool
    quota_throttled: bool
    host_physical_cores_verifiable: bool
    effective_core_limit: float

    def to_dict(self) -> dict[str, object]:
        """Serialize the topology record to plain JSON-compatible types."""
        return {
            "platform_system": self.platform_system,
            "platform_release": self.platform_release,
            "platform_machine": self.platform_machine,
            "cpu_model": self.cpu_model,
            "logical_cpus": self.logical_cpus,
            "affinity_cpus": list(self.affinity_cpus),
            "sysfs_physical_cores": self.sysfs_physical_cores,
            "physical_cores": self.physical_cores,
            "cgroup_cpu_max": self.cgroup_cpu_max,
            "cgroup_quota_cores": self.cgroup_quota_cores,
            "cgroup_effective_cpus": self.cgroup_effective_cpus,
            "virtualized_kernel": self.virtualized_kernel,
            "quota_throttled": self.quota_throttled,
            "host_physical_cores_verifiable": self.host_physical_cores_verifiable,
            "effective_core_limit": self.effective_core_limit,
        }


def _read_text_if_exists(path: Path) -> str | None:
    """Read and strip a single sysfs/cgroup file if it exists."""
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").strip()


def _parse_quota_pair(quota_str: str, period_str: str) -> float | None:
    """Parse integer ``quota`` and ``period`` strings into a positive core ratio."""
    if quota_str == "max":
        return None
    try:
        quota = int(quota_str)
        period = int(period_str)
    except ValueError:
        return None
    if quota > 0 and period > 0:
        return float(quota) / float(period)
    return None


def _parse_cgroup_quota() -> tuple[str | None, float | None]:
    """Read cgroup v2 or v1 CPU bandwidth quota without touching ``/proc``."""
    v2_raw = _read_text_if_exists(Path("/sys/fs/cgroup/cpu.max"))
    if v2_raw is not None:
        parts = v2_raw.split()
        if len(parts) == 2:
            return v2_raw, _parse_quota_pair(parts[0], parts[1])
        return v2_raw, None
    v1_quota = _read_text_if_exists(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"))
    v1_period = _read_text_if_exists(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us"))
    if v1_quota is not None and v1_period is not None:
        raw = f"{v1_quota} {v1_period}"
        return raw, _parse_quota_pair(v1_quota, v1_period)
    return None, None


def _linux_sysfs_physical_cores(
    affinity: set[int], cpu_root: Path = _DEFAULT_CPU_ROOT
) -> int | None:
    """Count distinct ``thread_siblings_list`` groups across online affinity CPUs."""
    if not cpu_root.is_dir():
        return None
    groups: set[str] = set()
    for entry in sorted(cpu_root.iterdir()):
        name = entry.name
        if not (name.startswith("cpu") and name[3:].isdigit()):
            continue
        if affinity and int(name[3:]) not in affinity:
            continue
        siblings = _read_text_if_exists(entry / "topology" / "thread_siblings_list")
        if siblings:
            groups.add(siblings)
    return len(groups) if groups else None


def _darwin_sysctl_int(name: str) -> int | None:
    """Read a 32-bit or 64-bit integer sysctl value via ``libc.sysctlbyname``."""
    try:
        libc = ctypes.CDLL(None)
    except OSError:
        return None
    if not hasattr(libc, "sysctlbyname"):
        return None
    fn = libc.sysctlbyname
    fn.argtypes = [
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    fn.restype = ctypes.c_int
    buf = ctypes.c_int64(0)
    size = ctypes.c_size_t(ctypes.sizeof(buf))
    rc = int(fn(name.encode("utf-8"), ctypes.byref(buf), ctypes.byref(size), None, 0))
    if rc != 0:
        return None
    if size.value == ctypes.sizeof(ctypes.c_int32):
        return int(ctypes.c_int32.from_buffer(buf).value)
    return int(buf.value)


def _darwin_sysctl_str(name: str) -> str | None:
    """Read a NUL-terminated string sysctl value via ``libc.sysctlbyname``."""
    try:
        libc = ctypes.CDLL(None)
    except OSError:
        return None
    if not hasattr(libc, "sysctlbyname"):
        return None
    fn = libc.sysctlbyname
    fn.argtypes = [
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    fn.restype = ctypes.c_int
    buf = ctypes.create_string_buffer(256)
    size = ctypes.c_size_t(ctypes.sizeof(buf))
    rc = int(fn(name.encode("utf-8"), buf, ctypes.byref(size), None, 0))
    if rc != 0 or size.value == 0:
        return None
    value = buf.value.decode("utf-8", errors="replace").strip()
    return value if value else None


def _resolve_cpu_model(system: str, machine: str, physical: int) -> str:
    """Resolve processor or microarchitecture name across Linux and macOS."""
    if system == "Darwin":
        brand = _darwin_sysctl_str("machdep.cpu.brand_string")
        if brand:
            return brand
    model = platform.processor()
    if not model or model in ("arm", "arm64", "x86_64", "i386"):
        for handle in _discover_library_handles(physical_cores=physical):
            if handle.corename:
                model = handle.corename
                break
    return model if model else machine


def inspect_cpu_topology(
    cpu_root: Path = _DEFAULT_CPU_ROOT,
) -> CPUTopology:
    """Inspect machine, affinity and cgroup CPU limits honestly."""
    system = platform.system()
    release = platform.release()
    machine = platform.machine()
    logical = os.cpu_count() or 1
    affinity = (
        tuple(sorted(os.sched_getaffinity(0)))
        if hasattr(os, "sched_getaffinity")
        else tuple(range(logical))
    )

    sysfs_phys = _linux_sysfs_physical_cores(set(affinity), cpu_root=cpu_root)
    darwin_phys: int | None = None
    darwin_vmm: int | None = None
    if sysfs_phys is None and system == "Darwin" and cpu_root == _DEFAULT_CPU_ROOT:
        darwin_phys = _darwin_sysctl_int("hw.physicalcpu")
        darwin_vmm = _darwin_sysctl_int("kern.hv_vmm_present")

    detected_phys = (
        sysfs_phys
        if sysfs_phys is not None and sysfs_phys > 0
        else (darwin_phys if darwin_phys is not None and darwin_phys > 0 else None)
    )
    fallback_cores = max(1, len(affinity))
    physical = detected_phys if detected_phys is not None else fallback_cores

    cgroup_max, cgroup_quota = _parse_cgroup_quota()
    cgroup_cpus = _read_text_if_exists(Path("/sys/fs/cgroup/cpuset.cpus.effective"))
    virtualized = ("linuxkit" in release.lower()) or bool(
        darwin_vmm is not None and darwin_vmm != 0
    )
    throttled = cgroup_quota is not None and cgroup_quota < float(physical)
    verifiable = (
        detected_phys is not None
        and not virtualized
        and not throttled
        and len(affinity) >= detected_phys
    )
    effective = min(
        float(physical),
        float(len(affinity)) if affinity else float(physical),
        cgroup_quota if cgroup_quota is not None else float(physical),
    )
    model = _resolve_cpu_model(system, machine, physical)

    return CPUTopology(
        platform_system=system,
        platform_release=release,
        platform_machine=machine,
        cpu_model=model,
        logical_cpus=logical,
        affinity_cpus=affinity,
        sysfs_physical_cores=sysfs_phys,
        physical_cores=physical,
        cgroup_cpu_max=cgroup_max,
        cgroup_quota_cores=cgroup_quota,
        cgroup_effective_cpus=cgroup_cpus,
        virtualized_kernel=virtualized,
        quota_throttled=throttled,
        host_physical_cores_verifiable=verifiable,
        effective_core_limit=effective,
    )


# --------------------------------------------------- dynamic library discovery


class _DlPhdrInfo(ctypes.Structure):
    """POSIX ``dl_iterate_phdr`` structure on 64-bit Linux."""

    _fields_ = [
        ("dlpi_addr", ctypes.c_void_p),
        ("dlpi_name", ctypes.c_char_p),
        ("dlpi_phdr", ctypes.c_void_p),
        ("dlpi_phnum", ctypes.c_uint16),
    ]


@dataclass(frozen=True)
class _LibraryHandle:
    """Callable getter/setter binding for one loaded native thread pool."""

    library_path: str
    backend: str
    symbol_prefix: str
    symbol_suffix: str
    config: str
    corename: str
    parallel_mode: int | None
    getter_address: int
    _getter: Any
    _setter: Any
    is_mode_only: bool = False
    configured_ceiling: int | None = None
    import_time_veclib_max_threads: int | None = None
    _raw_mode_getter: Any = None
    _raw_mode_setter: Any = None

    def get_threads(self) -> int | None:
        """Return verified numeric thread count, or ``None`` if only mode is known."""
        val = self._getter()
        if val is None:
            return None
        return max(1, int(val))

    def set_threads(self, count: int) -> None:
        """Set the thread count or threading mode of this library for ``count``."""
        self._setter(int(count))

    def get_raw_mode(self) -> int | None:
        """Return raw native mode enum when supported (e.g. Accelerate ``0``/``1``)."""
        if self._raw_mode_getter is not None:
            return int(self._raw_mode_getter())
        return None

    def get_mode_label(self, raw_mode: int | None = None) -> str:
        """Return human-readable threading mode label for ``raw_mode``."""
        if not self.is_mode_only:
            return "numeric"
        mode = self.get_raw_mode() if raw_mode is None else raw_mode
        if mode == 1:
            return "single_threaded"
        if mode == 0:
            return "multi_threaded_automatic"
        return f"unknown_mode_{mode}"

    def check_mode_matched(
        self,
        requested_threads: int,
        threads_active: int | None,
        raw_mode_active: int | None,
    ) -> bool:
        """Return whether the native mode matches the requested regime."""
        if not self.is_mode_only:
            return threads_active is not None and threads_active == int(
                requested_threads
            )
        if int(requested_threads) == 1:
            return raw_mode_active == 1 and threads_active == 1
        if self.configured_ceiling is not None and self.configured_ceiling < int(
            requested_threads
        ):
            return False
        return raw_mode_active == 0

    def restore_state(self, state: LibraryThreadState) -> None:
        """Restore the library to its pre-entry numeric thread count or raw mode."""
        if self._raw_mode_setter is not None and state.raw_mode_before is not None:
            self._raw_mode_setter(int(state.raw_mode_before))
            return
        if state.threads_before is not None:
            self._setter(int(state.threads_before))


@dataclass(frozen=True)
class LibraryThreadState:
    """Recorded thread state of one native BLAS/OpenMP library across a control block.

    Attributes
    ----------
    library_path : str
        Resolved path of the shared library.
    backend : str
        Backend family (``'scipy-openblas'``, ``'openblas'``, ``'accelerate'``,
        ``'mkl'``, ``'blis'`` or ``'openmp'``).
    symbol_prefix : str
        Symbol prefix (``'scipy_'`` or ``''``).
    symbol_suffix : str
        Symbol suffix (``'64_'`` or ``''``).
    config : str
        Library configuration string when provided by the runtime.
    corename : str
        Target microarchitecture name when provided by the runtime.
    parallel_mode : int or None
        Build-time threading model of OpenBLAS (``openblas_get_parallel``: ``0``
        sequential, ``1`` platform threads, ``2`` OpenMP), part of the stable build
        identity; ``None`` for backends without one.  The mutable Accelerate
        per-thread mode is recorded in ``raw_mode_before``, ``raw_mode_active`` and
        ``raw_mode_restored`` instead.
    threads_before : int or None
        Verified numeric thread limit before entering the control block, or
        ``None`` when the native getter reports only an automatic multi-thread
        mode (such as Apple Accelerate ``BLAS_THREADING_MULTI_THREADED = 0``).
    threads_active : int or None
        Verified numeric thread limit read back immediately after applying the
        control, or ``None`` when the native API reports only an automatic
        multi-thread mode without a numeric worker count.
    threads_restored : int or None
        Verified numeric thread limit read back after exiting the control block,
        or ``None`` before exit or when restored to an automatic multi-thread
        mode without a numeric getter.
    numeric_threads_verifiable : bool
        Whether ``threads_active`` is a genuine numeric native readback rather
        than a mode-only state.
    raw_mode_before : int or None
        Raw native mode enum before entry when applicable (e.g. Accelerate ``0``
        for ``BLAS_THREADING_MULTI_THREADED``, ``1`` for
        ``BLAS_THREADING_SINGLE_THREADED``).
    raw_mode_active : int or None
        Raw native mode enum while active when applicable.
    raw_mode_restored : int or None
        Raw native mode enum after exit when applicable.
    threading_mode_before : str
        Observed mode label before entry (``'numeric'``, ``'single_threaded'``
        or ``'multi_threaded_automatic'``).
    threading_mode_active : str
        Observed mode label while active.
    threading_mode_restored : str or None
        Observed mode label after exit.
    mode_matched : bool
        Whether the active native mode matches the requested regime (exact
        numeric match on numeric backends; mode ``1`` for ``threads=1`` or mode
        ``0`` for ``threads>1`` on mode-only backends).
    configured_ceiling : int or None
        Configured/derived ceiling (e.g. from ``physical_cores`` and
        ``import_time_veclib_max_threads``), explicitly distinct from
        ``threads_active``.
    import_time_veclib_max_threads : int or None
        Value of ``VECLIB_MAXIMUM_THREADS`` observed in ``os.environ`` at Python
        module import time (not claimed as verified native library startup state).
    participation : str
        ``'participating'``, ``'incidental'`` or ``'unknown'``: the role of the
        library's image in the provider membership (:class:`NativeMembership`).
    """

    library_path: str
    backend: str
    symbol_prefix: str
    symbol_suffix: str
    config: str
    corename: str
    parallel_mode: int | None
    threads_before: int | None
    threads_active: int | None
    threads_restored: int | None = None
    numeric_threads_verifiable: bool = True
    raw_mode_before: int | None = None
    raw_mode_active: int | None = None
    raw_mode_restored: int | None = None
    threading_mode_before: str = "numeric"
    threading_mode_active: str = "numeric"
    threading_mode_restored: str | None = None
    mode_matched: bool = False
    configured_ceiling: int | None = None
    import_time_veclib_max_threads: int | None = None
    participation: str = "unknown"

    def to_dict(self) -> dict[str, object]:
        """Serialize the library state to plain JSON-compatible types."""
        return {
            "library_path": self.library_path,
            "backend": self.backend,
            "symbol_prefix": self.symbol_prefix,
            "symbol_suffix": self.symbol_suffix,
            "config": self.config,
            "corename": self.corename,
            "parallel_mode": self.parallel_mode,
            "threads_before": self.threads_before,
            "threads_active": self.threads_active,
            "threads_restored": self.threads_restored,
            "numeric_threads_verifiable": self.numeric_threads_verifiable,
            "raw_mode_before": self.raw_mode_before,
            "raw_mode_active": self.raw_mode_active,
            "raw_mode_restored": self.raw_mode_restored,
            "threading_mode_before": self.threading_mode_before,
            "threading_mode_active": self.threading_mode_active,
            "threading_mode_restored": self.threading_mode_restored,
            "mode_matched": self.mode_matched,
            "configured_ceiling": self.configured_ceiling,
            "import_time_veclib_max_threads": self.import_time_veclib_max_threads,
            "participation": self.participation,
        }


def _enumerate_loader_paths(libc: Any) -> list[str]:
    """Enumerate loaded shared-library paths via ``dl_iterate_phdr`` or ``dyld``."""
    raw_paths: list[str] = []
    if hasattr(libc, "dl_iterate_phdr"):
        cb_type = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.POINTER(_DlPhdrInfo),
            ctypes.c_size_t,
            ctypes.c_void_p,
        )

        def _visit(info: Any, _size: int, _data: int | None) -> int:
            name = info.contents.dlpi_name
            if name:
                raw_paths.append(name.decode("utf-8", errors="replace"))
            return 0

        libc.dl_iterate_phdr(cb_type(_visit), None)
    if hasattr(libc, "_dyld_image_count") and hasattr(libc, "_dyld_get_image_name"):
        fn_count = libc._dyld_image_count
        fn_count.argtypes = []
        fn_count.restype = ctypes.c_uint32
        fn_name = libc._dyld_get_image_name
        fn_name.argtypes = [ctypes.c_uint32]
        fn_name.restype = ctypes.c_char_p
        for idx in range(int(fn_count())):
            raw_name = fn_name(idx)
            if raw_name:
                raw_paths.append(raw_name.decode("utf-8", errors="replace"))
    return raw_paths


def _loaded_shared_object_paths() -> list[Path]:
    """Enumerate loaded shared libraries and bundled wheel library folders."""
    scipy_mod: Any = importlib.import_module("scipy")
    scipy_linalg: Any = importlib.import_module("scipy.linalg")
    _ = (np.__version__, scipy_mod.__version__, scipy_linalg.__name__)
    raw_paths = _enumerate_loader_paths(ctypes.CDLL(None))

    np_root = Path(np.__file__).resolve().parent
    sp_root = Path(str(scipy_mod.__file__)).resolve().parent
    for folder in (
        np_root.parent / "numpy.libs",
        sp_root.parent / "scipy.libs",
        np_root / ".dylibs",
        sp_root / ".dylibs",
    ):
        if folder.is_dir():
            for child in sorted(folder.iterdir()):
                if child.is_file():
                    raw_paths.append(str(child))

    if platform.system() == "Darwin":
        raw_paths.extend(_ACCELERATE_CANDIDATE_PATHS)

    keywords = (
        "openblas",
        "accelerate",
        "veclib",
        "libblas",
        "liblapack",
        "mkl",
        "blis",
        "gomp",
        "iomp",
        "omp",
    )
    seen: set[Path] = set()
    ordered: list[Path] = []
    for raw in raw_paths:
        candidate = Path(raw)
        base = candidate.name.lower()
        if ".cpython-" in base or not any(k in base for k in keywords):
            continue
        is_dyld_cache = raw.startswith(_DARWIN_DYLD_PREFIXES)
        if not candidate.exists() and not is_dyld_cache:
            continue
        resolved_path = candidate.resolve() if candidate.exists() else candidate
        if resolved_path not in seen:
            seen.add(resolved_path)
            ordered.append(resolved_path)
    return ordered


def _function_address(func: Any) -> int:
    """Return the integer address of a ``ctypes`` function pointer, or ``id(func)``."""
    try:
        addr = ctypes.cast(func, ctypes.c_void_p).value
        return int(addr) if addr is not None else id(func)
    except (TypeError, ctypes.ArgumentError):
        return id(func)


def _probe_openblas(dll: Any, path_str: str) -> list[_LibraryHandle]:
    """Probe an OpenBLAS shared library for LP64 and ILP64 thread controls."""
    handles: list[_LibraryHandle] = []
    for prefix, suffix in _OPENBLAS_AFFIXES:
        get_name = f"{prefix}openblas_get_num_threads{suffix}"
        set_name = f"{prefix}openblas_set_num_threads{suffix}"
        if not (hasattr(dll, get_name) and hasattr(dll, set_name)):
            continue
        fn_get = getattr(dll, get_name)
        fn_get.argtypes = []
        fn_get.restype = ctypes.c_int
        fn_set = getattr(dll, set_name)
        fn_set.argtypes = [ctypes.c_int]
        fn_set.restype = None

        cfg_str = ""
        cfg_name = f"{prefix}openblas_get_config{suffix}"
        if hasattr(dll, cfg_name):
            fn_cfg = getattr(dll, cfg_name)
            fn_cfg.argtypes = []
            fn_cfg.restype = ctypes.c_char_p
            raw_cfg = fn_cfg()
            if raw_cfg:
                cfg_str = raw_cfg.decode("utf-8", errors="replace").strip()

        core_str = ""
        core_name = f"{prefix}openblas_get_corename{suffix}"
        if hasattr(dll, core_name):
            fn_core = getattr(dll, core_name)
            fn_core.argtypes = []
            fn_core.restype = ctypes.c_char_p
            raw_core = fn_core()
            if raw_core:
                core_str = raw_core.decode("utf-8", errors="replace").strip()

        par_mode: int | None = None
        par_name = f"{prefix}openblas_get_parallel{suffix}"
        if hasattr(dll, par_name):
            fn_par = getattr(dll, par_name)
            fn_par.argtypes = []
            fn_par.restype = ctypes.c_int
            par_mode = int(fn_par())

        backend = "scipy-openblas" if prefix == "scipy_" else "openblas"
        handles.append(
            _LibraryHandle(
                library_path=path_str,
                backend=backend,
                symbol_prefix=prefix,
                symbol_suffix=suffix,
                config=cfg_str,
                corename=core_str,
                parallel_mode=par_mode,
                getter_address=_function_address(fn_get),
                _getter=fn_get,
                _setter=fn_set,
            )
        )
    return handles


def _probe_accelerate(
    dll: Any,
    path_str: str,
    *,
    physical_cores: int = 1,
    startup_veclib_max: int | None = _IMPORT_TIME_VECLIB_MAX_THREADS,
) -> list[_LibraryHandle]:
    """Probe Apple Accelerate / vecLib for ``BLASGetThreading``/``BLASSetThreading``.

    In Apple's ``<vecLib/thread_api.h>``, ``BLASGetThreading`` and ``BLASSetThreading``
    toggle between ``BLAS_THREADING_MULTI_THREADED`` (``0``, automatic multi-threading
    where Accelerate decides how many threads to use) and
    ``BLAS_THREADING_SINGLE_THREADED`` (``1``, single-threaded execution).
    ``BLASGetThreading`` returns the per-thread enum mode (``0`` or ``1``), NOT a
    numeric worker count.  In single-threaded mode (``1``), the active thread limit is
    genuinely ``1``.  In multi-threaded mode (``0``), ``get_threads()`` returns ``None``
    rather than fabricating a numeric count from ``multithread_ceiling``; the derived
    ceiling and import-time ``VECLIB_MAXIMUM_THREADS`` observation are recorded
    separately on :class:`LibraryThreadState`.  The mode is per-thread control state
    that the controller itself changes, so the handle's ``parallel_mode`` (stable build
    identity) is ``None``; the mode is read through ``get_raw_mode``.
    """
    if not (hasattr(dll, "BLASGetThreading") and hasattr(dll, "BLASSetThreading")):
        return []
    fn_get = dll.BLASGetThreading
    fn_get.argtypes = []
    fn_get.restype = ctypes.c_int
    fn_set = dll.BLASSetThreading
    fn_set.argtypes = [ctypes.c_int]
    fn_set.restype = ctypes.c_int

    multithread_ceiling = (
        max(1, min(physical_cores, startup_veclib_max))
        if startup_veclib_max is not None
        else max(1, physical_cores)
    )

    def _get_raw_mode() -> int:
        return int(fn_get())

    def _set_raw_mode(target_mode: int) -> None:
        rc = int(fn_set(int(target_mode)))
        if rc != 0:
            raise RuntimeError(
                f"BLASSetThreading({target_mode}) failed with status {rc}"
            )

    def _get_threads() -> int | None:
        mode = int(fn_get())
        if mode == 1:
            return 1
        return None

    def _set_threads(count: int) -> None:
        target_mode = 1 if int(count) <= 1 else 0
        _set_raw_mode(target_mode)

    cfg = (
        "Apple Accelerate vecLib "
        "(BLASSetThreading/BLASGetThreading mode_api, "
        f"configured_ceiling={multithread_ceiling})"
    )
    return [
        _LibraryHandle(
            library_path=path_str,
            backend="accelerate",
            symbol_prefix="",
            symbol_suffix="",
            config=cfg,
            corename=platform.machine(),
            parallel_mode=None,
            getter_address=_function_address(fn_get),
            _getter=_get_threads,
            _setter=_set_threads,
            is_mode_only=True,
            configured_ceiling=multithread_ceiling,
            import_time_veclib_max_threads=startup_veclib_max,
            _raw_mode_getter=_get_raw_mode,
            _raw_mode_setter=_set_raw_mode,
        )
    ]


def _probe_other_backends(
    dll: Any,
    path_str: str,
    *,
    physical_cores: int = 1,
    startup_veclib_max: int | None = _IMPORT_TIME_VECLIB_MAX_THREADS,
) -> list[_LibraryHandle]:
    """Probe a shared library for MKL, BLIS, OpenMP or Apple Accelerate controls."""
    handles: list[_LibraryHandle] = []
    lower = Path(path_str).name.lower()
    specs = (
        ("mkl", "Intel MKL", "MKL_Get_Max_Threads", "MKL_Set_Num_Threads", True),
        (
            "blis",
            "BLIS",
            "bli_thread_get_num_threads",
            "bli_thread_set_num_threads",
            True,
        ),
        (
            "openmp",
            "OpenMP",
            "omp_get_max_threads",
            "omp_set_num_threads",
            any(tag in lower for tag in ("gomp", "iomp", "omp")),
        ),
    )
    for backend, label, get_sym, set_sym, enabled in specs:
        if enabled and hasattr(dll, get_sym) and hasattr(dll, set_sym):
            fn_get = getattr(dll, get_sym)
            fn_get.argtypes = []
            fn_get.restype = ctypes.c_int
            fn_set = getattr(dll, set_sym)
            fn_set.argtypes = [ctypes.c_int]
            fn_set.restype = None
            handles.append(
                _LibraryHandle(
                    library_path=path_str,
                    backend=backend,
                    symbol_prefix="",
                    symbol_suffix="",
                    config=label,
                    corename="",
                    parallel_mode=None,
                    getter_address=_function_address(fn_get),
                    _getter=fn_get,
                    _setter=fn_set,
                )
            )
    if not handles:
        handles.extend(
            _probe_accelerate(
                dll,
                path_str,
                physical_cores=physical_cores,
                startup_veclib_max=startup_veclib_max,
            )
        )
    return handles


def _discover_library_handles(
    paths: Sequence[Path] | None = None,
    *,
    physical_cores: int | None = None,
) -> list[_LibraryHandle]:
    """Discover loaded BLAS/OpenMP library thread controls in deterministic order."""
    if physical_cores is None:
        darwin_phys = (
            _darwin_sysctl_int("hw.physicalcpu")
            if platform.system() == "Darwin"
            else None
        )
        sysfs_phys = _linux_sysfs_physical_cores(set())
        physical_cores = (
            sysfs_phys
            if sysfs_phys is not None and sysfs_phys > 0
            else (
                darwin_phys
                if darwin_phys is not None and darwin_phys > 0
                else (os.cpu_count() or 1)
            )
        )
    handles: list[_LibraryHandle] = []
    seen_addresses: set[int] = set()
    candidates = _loaded_shared_object_paths() if paths is None else list(paths)
    for path in candidates:
        path_str = str(path)
        try:
            dll = ctypes.CDLL(path_str)
        except OSError:
            continue
        found = _probe_openblas(dll, path_str) or _probe_other_backends(
            dll, path_str, physical_cores=physical_cores
        )
        for handle in found:
            if handle.getter_address not in seen_addresses:
                seen_addresses.add(handle.getter_address)
                handles.append(handle)
    return handles


def _blas_build_metadata() -> tuple[str, str, str, str]:
    """Read NumPy and SciPy build-time BLAS dependency names and versions."""
    np_cfg_mod: Any = getattr(np, "__config__", None)
    np_cfg: Any = np_cfg_mod.show(mode="dicts") if np_cfg_mod is not None else {}
    np_blas = np_cfg.get("Build Dependencies", {}).get("blas", {})
    scipy_mod: Any = importlib.import_module("scipy")
    sp_cfg_mod: Any = getattr(scipy_mod, "__config__", None)
    sp_cfg: Any = sp_cfg_mod.show(mode="dicts") if sp_cfg_mod is not None else {}
    sp_blas = sp_cfg.get("Build Dependencies", {}).get("blas", {})
    return (
        str(np_blas.get("name", "unknown")),
        str(np_blas.get("version", "unknown")),
        str(sp_blas.get("name", "unknown")),
        str(sp_blas.get("version", "unknown")),
    )


# ------------------------------------------------- native provider membership

# The NumPy and SciPy modules that the package source imports (``import numpy``,
# ``numpy.typing`` and the ``importlib`` imports of ``scipy.linalg``,
# ``scipy.linalg.lapack``, ``scipy.optimize``, ``scipy.sparse`` and
# ``scipy.sparse.linalg``).  They are imported before membership is derived, so their
# native extension modules are entry points from the first control block.  pandas is
# never imported here; its extension modules count once loaded.
_NUMERICAL_SOURCE_MODULES: tuple[str, ...] = (
    "numpy",
    "numpy.typing",
    "scipy",
    "scipy.linalg",
    "scipy.linalg.lapack",
    "scipy.optimize",
    "scipy.sparse",
    "scipy.sparse.linalg",
)

# Exported entry points that identify a native numerical provider image: BLAS
# ``dgemm`` in its Fortran and CBLAS forms and LAPACK ``dgetrf``, with the OpenBLAS
# symbol prefixes and ILP64 suffixes, and the thread-control or runtime entry points
# of OpenBLAS, Intel MKL, BLIS, Apple Accelerate and OpenMP.
_PROVIDER_SYMBOLS: tuple[str, ...] = (
    *(f"{prefix}dgemm_{suffix}" for prefix, suffix in _OPENBLAS_AFFIXES),
    *(f"{prefix}cblas_dgemm{suffix}" for prefix, suffix in _OPENBLAS_AFFIXES),
    *(f"{prefix}dgetrf_{suffix}" for prefix, suffix in _OPENBLAS_AFFIXES),
    *(
        f"{prefix}openblas_get_num_threads{suffix}"
        for prefix, suffix in _OPENBLAS_AFFIXES
    ),
    "MKL_Get_Max_Threads",
    "bli_thread_get_num_threads",
    "BLASGetThreading",
    "omp_get_max_threads",
    "GOMP_parallel",
    "__kmpc_fork_call",
)
_LAPACK_SYMBOLS: frozenset[str] = frozenset(
    f"{prefix}dgetrf_{suffix}" for prefix, suffix in _OPENBLAS_AFFIXES
)

_PT_LOAD = 1
_PT_DYNAMIC = 2
_PF_X = 1
_DT_NULL = 0
_DT_NEEDED = 1
_DT_STRTAB = 5
_DT_STRSZ = 10
_DT_SONAME = 14
_ELF64_PHDR = struct.Struct("=IIQQQQQQ")
_ELF64_DYN = struct.Struct("=qQ")
_ELF_STRING_LIMIT = 4096

_MH_MAGIC_64 = 0xFEEDFACF
_MH_EXECUTE = 0x2
_MH_TWOLEVEL = 0x80
_MH_DYLIB_IN_CACHE = 0x80000000
_LC_REQ_DYLD = 0x80000000
_LC_SEGMENT_64 = 0x19
_LC_ID_DYLIB = 0xD
_LC_UUID = 0x1B
_LC_RPATH = 0x1C | _LC_REQ_DYLD
_VM_PROT_EXECUTE = 0x4
_MACHO_DEPENDENCY_KINDS: Mapping[int, str] = MappingProxyType(
    {
        0xC: "needed",
        0x18 | _LC_REQ_DYLD: "weak",
        0x1F | _LC_REQ_DYLD: "reexport",
        0x20: "lazy",
        0x23 | _LC_REQ_DYLD: "upward",
    }
)
_MACH_HEADER_64 = struct.Struct("=IiiIIIII")
_LOAD_COMMAND = struct.Struct("=II")
_SEGMENT_64 = struct.Struct("=16sQQQQiiII")
_LC_STRING_OFFSET = struct.Struct("=I")

_MemoryReader = Callable[[int, int], bytes]


def _read_memory(address: int, size: int) -> bytes:
    """Copy ``size`` bytes of this process's memory at ``address``."""
    return ctypes.string_at(address, size)


def _canonical_path(name: str) -> str:
    """The resolved form of a loader or module path, ``''`` for an unnamed image."""
    return str(Path(name).resolve()) if name else ""


@dataclass(frozen=True)
class _ImageRecord:
    """One loaded native image and the dynamic-linking records read from its memory.

    ``key`` is the load identity (format, loader name, load address and header
    location), fixed while the image stays loaded.  ``dependencies`` holds
    ``(kind, name)`` pairs: ELF ``DT_NEEDED`` names (kind ``'needed'``) or Mach-O
    dylib load commands (``'needed'``, ``'weak'``, ``'reexport'``, ``'lazy'``,
    ``'upward'``).  ``detail`` is empty when the records were read and otherwise says
    why they were not.
    """

    key: tuple[object, ...]
    name: str
    canonical: str
    image_format: str
    code_ranges: tuple[tuple[int, int], ...] = ()
    dependencies: tuple[tuple[str, str], ...] = ()
    install_name: str = ""
    rpaths: tuple[str, ...] = ()
    binary_id: str = ""
    main_program: bool = False
    two_level: bool = True
    in_shared_cache: bool = False
    detail: str = ""

    def owns(self, address: int) -> bool:
        """Whether ``address`` lies in an executable segment of this image."""
        return any(low <= address < high for low, high in self.code_ranges)


def _inside(ranges: Sequence[tuple[int, int]], address: int, size: int) -> bool:
    """Whether ``[address, address + size)`` lies in one of ``ranges``."""
    return any(low <= address and address + size <= high for low, high in ranges)


def _read_exact(read: _MemoryReader, address: int, size: int) -> bytes:
    """``size`` bytes at ``address``; a short read is an error, never empty records."""
    data = read(address, size)
    if len(data) != size:
        raise ValueError(f"read {len(data)} of {size} bytes at {address:#x}")
    return data


def _elf_dynamic_tags(
    raw: bytes,
) -> tuple[list[int], int | None, int | None, int | None]:
    """``DT_NEEDED`` offsets, ``DT_SONAME``, ``DT_STRTAB`` and ``DT_STRSZ`` values."""
    needed: list[int] = []
    values: dict[int, int] = {}
    for tag, value in _ELF64_DYN.iter_unpack(raw):
        if tag == _DT_NULL:
            break
        if tag == _DT_NEEDED:
            needed.append(value)
        elif tag in (_DT_SONAME, _DT_STRTAB, _DT_STRSZ):
            values[tag] = value
    return (
        needed,
        values.get(_DT_SONAME),
        values.get(_DT_STRTAB),
        values.get(_DT_STRSZ),
    )


def _elf_dynamic_names(
    dynamic: tuple[int, int],
    loads: Sequence[tuple[int, int]],
    base: int,
    read: _MemoryReader,
) -> tuple[tuple[str, ...], str]:
    """The ``DT_NEEDED`` names and the ``DT_SONAME`` of an in-memory ``_DYNAMIC``.

    The loader may or may not have relocated ``DT_STRTAB`` in place; the string
    table is the candidate address that lies, whole, in a loaded segment.
    """
    address, size = dynamic
    if not _inside(loads, address, size):
        raise ValueError("PT_DYNAMIC lies outside every PT_LOAD segment")
    needed, soname, strtab, strsz = _elf_dynamic_tags(
        _read_exact(read, address, size - size % _ELF64_DYN.size)
    )
    if not needed and soname is None:
        return (), ""
    if strtab is None or strsz is None:
        raise ValueError("DT_NEEDED or DT_SONAME without a string table")
    table = strtab if _inside(loads, strtab, strsz) else strtab + base
    if not _inside(loads, table, strsz):
        raise ValueError("DT_STRTAB lies outside every PT_LOAD segment")

    def text(offset: int) -> str:
        if offset >= strsz:
            raise ValueError(f"string offset {offset} beyond DT_STRSZ {strsz}")
        chunk = read(table + offset, min(strsz - offset, _ELF_STRING_LIMIT))
        end = chunk.find(b"\x00")
        if end < 0:
            raise ValueError(f"unterminated string at offset {offset}")
        return chunk[:end].decode("utf-8")

    return tuple(text(offset) for offset in needed), (
        "" if soname is None else text(soname)
    )


def _parse_elf_image(
    row: tuple[int, str, int, int], read: _MemoryReader = _read_memory
) -> _ImageRecord:
    """Read the ELF records of one ``dl_iterate_phdr`` row ``(base, name, phdr, n)``."""
    base, name, phdr, phnum = row
    record = _ImageRecord(
        key=("elf", base, name, phdr, phnum),
        name=name,
        canonical=_canonical_path(name),
        image_format="elf",
    )
    try:
        table = _read_exact(read, phdr, phnum * _ELF64_PHDR.size)
        loads: list[tuple[int, int]] = []
        code: list[tuple[int, int]] = []
        dynamic: tuple[int, int] | None = None
        for p_type, p_flags, _, vaddr, _, _, memsz, _ in _ELF64_PHDR.iter_unpack(table):
            start = base + vaddr
            if p_type == _PT_LOAD:
                loads.append((start, start + memsz))
                if p_flags & _PF_X:
                    code.append((start, start + memsz))
            elif p_type == _PT_DYNAMIC:
                dynamic = (start, memsz)
        needed, soname = (
            ((), "")
            if dynamic is None
            else _elf_dynamic_names(dynamic, loads, base, read)
        )
    except (struct.error, ValueError) as error:
        return replace(record, detail=f"unreadable ELF dynamic records ({error})")
    return replace(
        record,
        code_ranges=tuple(code),
        dependencies=tuple(("needed", dep) for dep in needed),
        install_name=soname,
        main_program=not name,
    )


def _load_command_string(body: bytes) -> str:
    """The ``lc_str`` of a Mach-O load command whose offset field is at byte 8."""
    (offset,) = _LC_STRING_OFFSET.unpack_from(body, _LOAD_COMMAND.size)
    end = body.find(b"\x00", offset)
    if offset >= len(body) or end < 0:
        raise ValueError("load command string outside its command")
    return body[offset:end].decode("utf-8")


def _parse_macho_image(
    row: tuple[str, int, int], read: _MemoryReader = _read_memory
) -> _ImageRecord:
    """Read the Mach-O load commands of one dyld row ``(name, header, slide)``."""
    name, header, slide = row
    record = _ImageRecord(
        key=("macho", name, header, slide),
        name=name,
        canonical=_canonical_path(name),
        image_format="macho",
    )
    try:
        magic, _, _, filetype, ncmds, sizeofcmds, flags, _ = _MACH_HEADER_64.unpack(
            _read_exact(read, header, _MACH_HEADER_64.size)
        )
        if magic != _MH_MAGIC_64:
            raise ValueError(f"magic {magic:#x} is not a 64-bit Mach-O header")
        commands = _read_exact(read, header + _MACH_HEADER_64.size, sizeofcmds)
        code: list[tuple[int, int]] = []
        dependencies: list[tuple[str, str]] = []
        rpaths: list[str] = []
        install_name = ""
        uuid = ""
        offset = 0
        for _ in range(ncmds):
            cmd, cmdsize = _LOAD_COMMAND.unpack_from(commands, offset)
            if cmdsize < _LOAD_COMMAND.size or offset + cmdsize > len(commands):
                raise ValueError("load command outside sizeofcmds")
            body = commands[offset : offset + cmdsize]
            offset += cmdsize
            if cmd == _LC_SEGMENT_64:
                fields_64 = _SEGMENT_64.unpack_from(body, _LOAD_COMMAND.size)
                vmaddr, vmsize, initprot = fields_64[1], fields_64[2], fields_64[6]
                if vmsize and initprot & _VM_PROT_EXECUTE:
                    code.append((vmaddr + slide, vmaddr + slide + vmsize))
            elif cmd in _MACHO_DEPENDENCY_KINDS:
                kind = _MACHO_DEPENDENCY_KINDS[cmd]
                dependencies.append((kind, _load_command_string(body)))
            elif cmd == _LC_ID_DYLIB:
                install_name = _load_command_string(body)
            elif cmd == _LC_RPATH:
                rpaths.append(_load_command_string(body))
            elif cmd == _LC_UUID:
                uuid = body[_LOAD_COMMAND.size : _LOAD_COMMAND.size + 16].hex()
    except (struct.error, ValueError) as error:
        return replace(record, detail=f"unreadable Mach-O load commands ({error})")
    return replace(
        record,
        code_ranges=tuple(code),
        dependencies=tuple(dependencies),
        install_name=install_name,
        rpaths=tuple(rpaths),
        binary_id=uuid,
        main_program=filetype == _MH_EXECUTE,
        two_level=bool(flags & _MH_TWOLEVEL),
        in_shared_cache=bool(flags & _MH_DYLIB_IN_CACHE),
    )


def _elf_image_rows(libc: Any) -> list[tuple[int, str, int, int]]:
    """``(base, name, phdr, phnum)`` of every object in the ELF link map."""
    rows: list[tuple[int, str, int, int]] = []
    cb_type = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.POINTER(_DlPhdrInfo),
        ctypes.c_size_t,
        ctypes.c_void_p,
    )

    def _visit(info: Any, _size: int, _data: int | None) -> int:
        entry = info.contents
        raw_name = entry.dlpi_name or b""
        rows.append(
            (
                int(entry.dlpi_addr or 0),
                raw_name.decode("utf-8", errors="replace"),
                int(entry.dlpi_phdr or 0),
                int(entry.dlpi_phnum),
            )
        )
        return 0

    libc.dl_iterate_phdr(cb_type(_visit), None)
    return rows


def _macho_image_rows(libc: Any) -> list[tuple[str, int, int]]:
    """``(name, header, slide)`` of every image in the dyld image table."""
    count = libc._dyld_image_count
    count.argtypes = []
    count.restype = ctypes.c_uint32
    get_name = libc._dyld_get_image_name
    get_name.argtypes = [ctypes.c_uint32]
    get_name.restype = ctypes.c_char_p
    get_header = libc._dyld_get_image_header
    get_header.argtypes = [ctypes.c_uint32]
    get_header.restype = ctypes.c_void_p
    get_slide = libc._dyld_get_image_vmaddr_slide
    get_slide.argtypes = [ctypes.c_uint32]
    get_slide.restype = ctypes.c_ssize_t
    rows: list[tuple[str, int, int]] = []
    for index in range(int(count())):
        header = get_header(index)
        if not header:
            continue
        raw_name = get_name(index) or b""
        rows.append(
            (
                raw_name.decode("utf-8", errors="replace"),
                int(header),
                int(get_slide(index)),
            )
        )
    return rows


_IMAGE_RECORDS: dict[tuple[object, ...], _ImageRecord] = {}


def _loaded_images() -> tuple[_ImageRecord, ...]:
    """Every loaded native image with the dynamic-linking records of its memory.

    The records of an image are read once per load identity, which is fixed while it
    stays loaded (CPython never unloads an extension module and :mod:`ctypes` never
    closes a library).  Without an ELF link map or a dyld image table the result is
    empty, and membership is then incomplete.
    """
    libc = ctypes.CDLL(None)
    records: list[_ImageRecord] = []
    if hasattr(libc, "dl_iterate_phdr"):
        for elf_row in _elf_image_rows(libc):
            key: tuple[object, ...] = ("elf", *elf_row)
            record = _IMAGE_RECORDS.get(key)
            if record is None:
                record = _IMAGE_RECORDS.setdefault(key, _parse_elf_image(elf_row))
            records.append(record)
    elif hasattr(libc, "_dyld_image_count") and hasattr(libc, "_dyld_get_image_header"):
        for macho_row in _macho_image_rows(libc):
            key = ("macho", *macho_row)
            record = _IMAGE_RECORDS.get(key)
            if record is None:
                record = _IMAGE_RECORDS.setdefault(key, _parse_macho_image(macho_row))
            records.append(record)
    return tuple(records)


_RESOLVED_MODULE_FILES: dict[str, str] = {}


def _within(path: str, root: str) -> bool:
    """Whether the resolved ``path`` is ``root`` or lies below it."""
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _standard_library_roots() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The resolved standard-library directories and site-package directories."""
    paths = sysconfig.get_paths()
    library = {_canonical_path(paths[key]) for key in ("stdlib", "platstdlib")}
    sites = {
        _canonical_path(path)
        for path in (
            paths["purelib"],
            paths["platlib"],
            *site.getsitepackages(),
            site.getusersitepackages(),
        )
    }
    return tuple(sorted(library)), tuple(sorted(sites))


def _entry_module_files() -> tuple[str, ...]:
    """Resolved files of the loaded third-party native extension modules.

    These are the numerical entry points: every loaded module whose file has an
    extension-module suffix and does not lie in the standard library outside a
    site-packages directory.  :data:`_NUMERICAL_SOURCE_MODULES` is imported first.
    """
    for module_name in _NUMERICAL_SOURCE_MODULES:
        importlib.import_module(module_name)
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    library, sites = _standard_library_roots()
    files: set[str] = set()
    for module in list(sys.modules.values()):
        origin = getattr(module, "__file__", None)
        if not isinstance(origin, str) or not origin.endswith(suffixes):
            continue
        path = _RESOLVED_MODULE_FILES.get(origin)
        if path is None:
            path = _RESOLVED_MODULE_FILES.setdefault(origin, _canonical_path(origin))
        standard = any(_within(path, root) for root in library) and not any(
            _within(path, root) for root in sites
        )
        if not standard:
            files.add(path)
    return tuple(sorted(files))


def _symbol_address(library: Any, symbol: str) -> int | None:
    """The address that ``dlsym`` gives for ``symbol`` through ``library``."""
    try:
        function = getattr(library, symbol)
    except AttributeError:
        return None
    address = ctypes.cast(function, ctypes.c_void_p).value
    return int(address) if address else None


_IMAGE_EXPORTS: dict[tuple[object, ...], tuple[str, ...] | None] = {}


def _image_exports(image: _ImageRecord) -> tuple[str, ...] | None:
    """The :data:`_PROVIDER_SYMBOLS` defined in ``image``, ``None`` if not probed.

    The image is opened with ``RTLD_NOLOAD`` (never loading anything) and a symbol
    counts only when its address lies in an executable segment of ``image`` itself,
    not of a dependency or a re-exported library.
    """
    if image.key in _IMAGE_EXPORTS:
        return _IMAGE_EXPORTS[image.key]
    try:
        library = ctypes.CDLL(image.name or None, mode=os.RTLD_NOLOAD | os.RTLD_LAZY)
    except OSError:
        found: tuple[str, ...] | None = None
    else:
        found = tuple(
            symbol
            for symbol in _PROVIDER_SYMBOLS
            if (address := _symbol_address(library, symbol)) is not None
            and image.owns(address)
        )
    _IMAGE_EXPORTS[image.key] = found
    return found


def _default_scope_address(symbol: str) -> int | None:
    """The address of ``symbol`` in the process's default (global) symbol scope."""
    return _symbol_address(ctypes.CDLL(None), symbol)


def _file_sha256(name: str) -> str | None:
    """SHA-256 of the regular file behind a loader path, ``None`` without one.

    The current bytes are read at every call, in 64 KiB chunks so no large buffer
    is held.  Nothing is memoized: a same-size rewrite can keep the path, device,
    inode, size and every timestamp of the file, so no metadata proves its bytes
    unchanged.
    """
    if not name:
        return None
    target = Path(name)
    try:
        status = target.stat()
    except OSError:
        return None
    if not stat.S_ISREG(status.st_mode):
        return None
    hasher = hashlib.sha256()
    try:
        with target.open("rb") as stream:
            while chunk := stream.read(1 << 16):
                hasher.update(chunk)
    except OSError:
        return None
    return hasher.hexdigest()


def _image_sha256(image: _ImageRecord) -> str | None:
    """SHA-256 of the file backing ``image``, ``None`` when mapped from shared cache."""
    return None if image.in_shared_cache else _file_sha256(image.name)


def _handle_key(handle: Any) -> tuple[object, ...]:
    """The identity of a thread-control handle: path, backend, affixes, getter."""
    return (
        str(getattr(handle, "library_path", "")),
        str(getattr(handle, "backend", "")),
        str(getattr(handle, "symbol_prefix", "")),
        str(getattr(handle, "symbol_suffix", "")),
        getattr(handle, "getter_address", None),
    )


def _resolve_elf_dependency(
    name: str, images: Sequence[_ImageRecord]
) -> list[_ImageRecord]:
    """The loaded images that a ``DT_NEEDED`` ``name`` can denote.

    As the dynamic loader does, the first image in link-map order whose
    ``DT_SONAME`` or loader name equals ``name``; otherwise every image whose file
    name equals ``name``.
    """
    named = [image for image in images if name in (image.install_name, image.name)]
    if named:
        return named[:1]
    return [image for image in images if image.name and Path(image.name).name == name]


def _expand_rpath(entry: str, declaring: _ImageRecord, main_dir: str | None) -> str:
    """One ``LC_RPATH`` entry of ``declaring`` with its loader prefixes expanded."""
    for prefix, base in (
        ("@loader_path", str(Path(declaring.canonical).parent)),
        ("@executable_path", main_dir),
    ):
        if entry == prefix or entry.startswith(prefix + "/"):
            rest = entry.removeprefix(prefix).lstrip("/")
            return "" if base is None else str(Path(base) / rest)
    return entry


def _resolve_macho_dependency(
    name: str,
    image: _ImageRecord,
    images: Sequence[_ImageRecord],
    rpath_dirs: Sequence[str],
    main_dir: str | None,
) -> list[_ImageRecord]:
    """The loaded images that the dylib load command ``name`` of ``image`` denotes.

    ``@loader_path`` and ``@executable_path`` are expanded, ``@rpath`` is tried in
    the run-path order of ``rpath_dirs``, and the first candidate that is a loaded
    image is taken; otherwise, for non-system install names, the images whose
    ``LC_ID_DYLIB`` equals ``name``.
    """
    if name.startswith("@rpath/"):
        rest = name.removeprefix("@rpath/")
        candidates = [str(Path(path) / rest) for path in rpath_dirs if path]
    else:
        expanded = _expand_rpath(name, image, main_dir)
        candidates = [expanded] if expanded else []
    for candidate in candidates:
        canonical = _canonical_path(candidate)
        hits = [other for other in images if other.canonical == canonical]
        if hits:
            return hits
    if name.startswith(_DARWIN_DYLD_PREFIXES):
        return []
    return [other for other in images if other.install_name == name]


@dataclass(frozen=True)
class _Closure:
    """The dependency closure of the numerical entry points over loaded images."""

    entries: tuple[_ImageRecord, ...]
    via: Mapping[tuple[object, ...], tuple[str, ...]]
    edges: Mapping[tuple[object, ...], tuple[tuple[str, str, str | None], ...]]
    reasons: tuple[str, ...]


def _is_provider_image(
    image: _ImageRecord, pooled_keys: AbstractSet[tuple[object, ...]]
) -> bool:
    """Whether ``image`` owns a thread-control handle or defines a provider symbol."""
    return image.key in pooled_keys or bool(_image_exports(image))


def _entry_images(
    images: Sequence[_ImageRecord], entry_files: Sequence[str]
) -> tuple[list[_ImageRecord], list[str]]:
    """Resolve ``entry_files`` to loaded images and collect any missing reasons."""
    reasons: list[str] = []
    entries: list[_ImageRecord] = []
    for path in entry_files:
        found = [image for image in images if image.canonical == path]
        if len(found) == 1:
            entries.append(found[0])
        else:
            reasons.append(f"entry module {path}: {len(found)} loaded images")
    if not entry_files:
        reasons.append("no third-party native extension module is loaded")
    return entries, reasons


def _main_rpaths(images: Sequence[_ImageRecord]) -> tuple[str | None, tuple[str, ...]]:
    """The main executable's directory and expanded ``LC_RPATH`` entries."""
    main = next((image for image in images if image.main_program), None)
    if main is None:
        return None, ()
    main_dir = str(Path(main.canonical).parent) if main.canonical else None
    return main_dir, tuple(
        _expand_rpath(entry, main, main_dir) for entry in main.rpaths
    )


def _is_darwin_system_image(image: _ImageRecord, system_path: str) -> bool:
    """Whether ``image`` is the loaded Mach-O system image at ``system_path``."""
    return (
        image.image_format == "macho"
        and image.two_level
        and not image.detail
        and image.name == system_path
        and image.canonical == system_path
        and image.install_name == system_path
    )


def _is_darwin_umbrella_image(image: _ImageRecord) -> bool:
    """Whether ``image`` is a loaded Darwin ``libSystem``/``Accelerate`` umbrella."""
    return image.name in _DARWIN_UMBRELLA_INSTALL_NAMES and _is_darwin_system_image(
        image, image.name
    )


def _is_apple_veclib_image(image: _ImageRecord, install_name: str) -> bool:
    """Whether ``image`` is the loaded Mach-O ``vecLib`` image at ``install_name``."""
    return _is_darwin_system_image(image, install_name)


def _dependency_closure(
    images: Sequence[_ImageRecord],
    entry_files: Sequence[str],
    pooled_keys: AbstractSet[tuple[object, ...]] = frozenset(),
) -> _Closure:
    """Close the entry images under their resolved dynamic dependencies.

    ``via`` gives, for each image of the closure, the loader paths from an entry
    point to it; ``edges`` the resolved target of each dependency.  Every dependency
    of an entry point or non-umbrella library (``needed``, ``reexport``, ``weak``,
    ``lazy``, ``upward``) is expanded in full; the non-provider ``reexport``
    sub-libraries of the Darwin ``libSystem`` C runtime umbrella and of the
    ``Accelerate``/``vecLib`` umbrella framework record their own ``reexport``
    sub-libraries and any direct provider dependencies without expanding the OS
    IPC/GUI shared-cache graph when reached solely as umbrella ``reexport``
    sub-libraries.  Every entry file without exactly one loaded image, unreadable
    image, flat-namespace Mach-O image and traversed dependency that resolves to no
    or several images (an absent weak dylib excepted) is a reason that the closure
    is incomplete.
    """
    entries, reasons = _entry_images(images, entry_files)
    main_dir, main_rpaths = _main_rpaths(images)
    via: dict[tuple[object, ...], tuple[str, ...]] = {}
    edges: dict[tuple[object, ...], tuple[tuple[str, str, str | None], ...]] = {}
    full_expanded: set[tuple[object, ...]] = set()
    queue: deque[tuple[_ImageRecord, tuple[str, ...], tuple[str, ...], bool]] = deque(
        (entry, (entry.name,), (), True) for entry in entries
    )
    while queue:
        image, chain, inherited, direct = queue.popleft()
        follow_all = (
            direct
            or bool(image.detail)
            or image.image_format != "macho"
            or not image.two_level
            or _is_provider_image(image, pooled_keys)
        )
        already_visited = image.key in via
        if already_visited and (not follow_all or image.key in full_expanded):
            continue
        if not already_visited:
            via[image.key] = chain
        if follow_all:
            full_expanded.add(image.key)
        if image.detail:
            reasons.append(f"{image.name}: {image.detail}")
            continue
        if image.image_format == "macho" and not image.two_level:
            reasons.append(f"{image.name}: flat-namespace image")
        own = tuple(_expand_rpath(entry, image, main_dir) for entry in image.rpaths)
        resolved: list[tuple[str, str, str | None]] = []
        for kind, name in image.dependencies:
            targets = (
                _resolve_elf_dependency(name, images)
                if image.image_format == "elf"
                else _resolve_macho_dependency(
                    name, image, images, (*own, *inherited, *main_rpaths), main_dir
                )
            )
            if not follow_all and kind != "reexport":
                if len(targets) == 1 and _is_provider_image(targets[0], pooled_keys):
                    resolved.append((kind, name, targets[0].name))
                    queue.append(
                        (
                            targets[0],
                            (*chain, targets[0].name),
                            (*own, *inherited),
                            False,
                        )
                    )
                continue
            if len(targets) == 1:
                resolved.append((kind, name, targets[0].name))
                next_direct = follow_all and not (
                    kind == "reexport" and _is_darwin_umbrella_image(image)
                )
                queue.append(
                    (
                        targets[0],
                        (*chain, targets[0].name),
                        (*own, *inherited),
                        next_direct,
                    )
                )
                continue
            resolved.append((kind, name, None))
            if (targets or kind != "weak") and not (
                already_visited and kind == "reexport"
            ):
                reasons.append(
                    f"{image.name}: {kind} dependency {name!r} resolves to "
                    f"{len(targets)} loaded images"
                )
        edges[image.key] = tuple(resolved)
    return _Closure(
        entries=tuple(entries), via=via, edges=edges, reasons=tuple(reasons)
    )


@dataclass(frozen=True)
class NativeProvider:
    """One native numerical provider image and the evidence of its participation.

    Attributes
    ----------
    library_path : str
        Loader path of the image.
    role : str
        ``'participating'`` when the image is in the dependency closure of a
        numerical entry point or services a provider entry point in the default
        symbol scope; ``'incidental'`` for a thread pool outside a complete
        closure; ``'unknown'`` for a thread pool when the closure is incomplete.
    control : str
        ``'numeric'`` when a numeric thread getter and setter map to the image,
        ``'mode_only'`` when only a threading mode does (Apple Accelerate), and
        ``'unsupported'`` when it defines provider entry points but no recognized
        thread control maps to it.
    backends : tuple of str
        Backends of the thread-control handles governing the image.
    exports : tuple of str
        The :data:`_PROVIDER_SYMBOLS` that the image itself defines.
    sha256 : str or None
        SHA-256 of the file behind the image, ``None`` when no regular file backs
        it (an image of the dyld shared cache).
    binary_id : str
        Mach-O ``LC_UUID`` read from the loaded image, ``''`` for an ELF image
        (whose file digest identifies it).
    via : tuple of str
        Loader paths from a numerical entry point to the image, or the default
        scope symbol that it services.
    """

    library_path: str
    role: str
    control: str
    backends: tuple[str, ...]
    exports: tuple[str, ...]
    sha256: str | None
    binary_id: str
    via: tuple[str, ...]

    def identity(self) -> dict[str, object]:
        """The stable build identity and classification of the provider."""
        return {
            "library_path": self.library_path,
            "role": self.role,
            "control": self.control,
            "backends": list(self.backends),
            "exports": list(self.exports),
            "sha256": self.sha256,
            "binary_id": self.binary_id,
        }

    def to_dict(self) -> dict[str, object]:
        """Serialize the provider, with its dependency path, to plain JSON types."""
        return {**self.identity(), "via": list(self.via)}


@dataclass(frozen=True, eq=False)
class NativeMembership:
    """Evidence-bound participation of native numerical providers in one process state.

    Attributes
    ----------
    complete : bool
        Whether every entry point, dependency, handle and probe was resolved, so
        that a thread pool outside the closure is demonstrably incidental.
    entry_points : int
        Number of loaded third-party native extension modules.
    closure_images : int
        Number of loaded images in their dependency closure.
    loaded_images : int
        Number of loaded native images of the process.
    providers : tuple of NativeProvider
        Participating providers, each with its control kind.
    other_pools : tuple of NativeProvider
        Thread pools outside the closure: incidental, or unknown when incomplete.
    unresolved : tuple of str
        Why the membership is incomplete; empty when it is complete.
    evidence_sha256 : str
        SHA-256 of the canonical JSON of ``evidence``.
    evidence : Mapping of str to object
        Entry points, closure images with their resolved dependencies and binary
        identities, handles, providers and the loaded image names.
    participation : Mapping
        Role of each thread-control handle by :func:`_handle_key`.
    """

    complete: bool
    entry_points: int
    closure_images: int
    loaded_images: int
    providers: tuple[NativeProvider, ...]
    other_pools: tuple[NativeProvider, ...]
    unresolved: tuple[str, ...]
    evidence_sha256: str
    evidence: Mapping[str, object]
    participation: Mapping[tuple[object, ...], str]

    def participation_of(self, handle: Any) -> str:
        """``'participating'``, ``'incidental'`` or ``'unknown'`` for ``handle``."""
        return self.participation.get(_handle_key(handle), "unknown")

    @property
    def provider_keys(self) -> tuple[tuple[object, ...], ...]:
        """Path, binary identity and control kind of every participating provider."""
        return tuple(
            (p.library_path, p.sha256, p.binary_id, p.control) for p in self.providers
        )

    def identity(self) -> dict[str, object]:
        """The stable part: completeness, providers, other pools and reasons."""
        return {
            "complete": self.complete,
            "providers": [p.identity() for p in self.providers],
            "other_pools": [p.identity() for p in self.other_pools],
            "unresolved": list(self.unresolved),
        }

    def to_dict(self) -> dict[str, object]:
        """Serialize the membership summary to plain JSON-compatible types."""
        return {
            "complete": self.complete,
            "entry_points": self.entry_points,
            "closure_images": self.closure_images,
            "loaded_images": self.loaded_images,
            "providers": [p.to_dict() for p in self.providers],
            "other_pools": [p.to_dict() for p in self.other_pools],
            "unresolved": list(self.unresolved),
            "evidence_sha256": self.evidence_sha256,
        }


def _provider(
    image: _ImageRecord, role: str, handles: Sequence[Any], via: tuple[str, ...]
) -> NativeProvider:
    """The provider record of ``image`` with the handles governing its pool."""
    control = (
        "unsupported"
        if not handles
        else (
            "mode_only"
            if any(getattr(h, "is_mode_only", False) for h in handles)
            else "numeric"
        )
    )
    return NativeProvider(
        library_path=image.name,
        role=role,
        control=control,
        backends=tuple(sorted({str(getattr(h, "backend", "")) for h in handles})),
        exports=_image_exports(image) or (),
        sha256=_image_sha256(image),
        binary_id=image.binary_id,
        via=via,
    )


def _handle_owners(
    handles: Sequence[Any], images: Sequence[_ImageRecord]
) -> tuple[dict[tuple[object, ...], tuple[object, ...] | None], list[str]]:
    """The image whose executable segment holds each handle's getter address."""
    owners: dict[tuple[object, ...], tuple[object, ...] | None] = {}
    reasons: list[str] = []
    for handle in handles:
        address = getattr(handle, "getter_address", None)
        hosts = (
            [image for image in images if image.owns(address)]
            if isinstance(address, int)
            else []
        )
        owners[_handle_key(handle)] = hosts[0].key if len(hosts) == 1 else None
        if len(hosts) != 1:
            reasons.append(
                f"thread-control handle {getattr(handle, 'backend', '?')} at "
                f"{getattr(handle, 'library_path', '?')}: getter lies in "
                f"{len(hosts)} loaded images"
            )
    return owners, reasons


def _participating_images(
    closure: _Closure,
    by_key: Mapping[tuple[object, ...], _ImageRecord],
    pooled: Mapping[tuple[object, ...], Sequence[Any]],
    images: Sequence[_ImageRecord],
) -> tuple[dict[tuple[object, ...], tuple[str, ...]], list[str]]:
    """Providers of the closure and the images serving their default-scope symbols.

    An image of the closure is a provider when a handle maps to it or it defines a
    provider entry point.  A provider symbol that the default scope resolves to
    another loaded image makes that image a provider too (it can service that
    entry point).  A closure image without a binary identity, a failed probe or an
    unattributable default-scope address is a reason of incompleteness.
    """
    reasons: list[str] = []
    providers: dict[tuple[object, ...], tuple[str, ...]] = {}
    for key, chain in closure.via.items():
        image = by_key[key]
        exports = _image_exports(image)
        if exports is None:
            reasons.append(f"{image.name}: provider probe failed (RTLD_NOLOAD open)")
        elif exports or pooled.get(key):
            providers[key] = chain
        if not image.binary_id and _image_sha256(image) is None:
            reasons.append(f"{image.name}: no binary identity")
    pending = list(providers)
    while pending:
        image = by_key[pending.pop()]
        for symbol in _image_exports(image) or ():
            address = _default_scope_address(symbol)
            if address is None or image.owns(address):
                continue
            hosts = [other for other in images if other.owns(address)]
            if len(hosts) != 1:
                reasons.append(
                    f"default scope resolves {symbol} into {len(hosts)} loaded images"
                )
            elif hosts[0].key not in providers:
                providers[hosts[0].key] = ("default scope", symbol)
                pending.append(hosts[0].key)
    return providers, reasons


def _bound_provider_handles(
    key: tuple[object, ...],
    closure: _Closure,
    by_key: Mapping[tuple[object, ...], _ImageRecord],
    participating: Mapping[tuple[object, ...], tuple[str, ...]],
    pooled: Mapping[tuple[object, ...], Sequence[Any]],
) -> Sequence[Any]:
    """Thread-control handles governing ``key``, including Apple ``vecLib`` LAPACK.

    A provider image is governed by the handles whose getter lies in it.  Across
    images, Apple's ``<vecLib/thread_api.h>`` documents ``BLASSetThreading`` as
    governing both BLAS and LAPACK in ``vecLib.framework``, so ``vecLib``'s loaded
    system ``libLAPACK.dylib`` (defining only :data:`_LAPACK_SYMBOLS` and having no
    unresolved required dependency) inherits the ``accelerate`` handle of its
    participating sibling ``libBLAS.dylib`` when that is the only provider
    dependency it links and both share the same shared-cache residency.  Arbitrary
    LAPACK-only images, including private binaries that declare ``vecLib``'s
    ``LC_ID_DYLIB`` metadata, do not inherit a linked BLAS dependency's handles.
    """
    direct = pooled.get(key, ())
    if direct:
        return direct
    image = by_key[key]
    if not _is_apple_veclib_image(image, _DARWIN_VECLIB_LAPACK):
        return ()
    exports = _image_exports(image) or ()
    if not exports or not set(exports).issubset(_LAPACK_SYMBOLS):
        return ()
    edges = closure.edges.get(key, ())
    if any(target_name is None and kind != "weak" for kind, _, target_name in edges):
        return ()
    dep_providers: list[tuple[object, ...]] = []
    for _, _, target_name in edges:
        if target_name is None:
            continue
        matches = [
            img.key
            for img in by_key.values()
            if img.name == target_name and img.key in participating and img.key != key
        ]
        if len(matches) == 1:
            dep_providers.append(matches[0])
    if not dep_providers or not all(
        _is_apple_veclib_image(by_key[dep], _DARWIN_VECLIB_BLAS)
        and by_key[dep].in_shared_cache == image.in_shared_cache
        and bool(pooled.get(dep))
        and all(getattr(h, "backend", None) == "accelerate" for h in pooled[dep])
        for dep in dep_providers
    ):
        return ()
    seen: set[tuple[object, ...]] = set()
    bound: list[Any] = []
    for dep in dep_providers:
        for handle in pooled[dep]:
            hkey = _handle_key(handle)
            if hkey not in seen:
                seen.add(hkey)
                bound.append(handle)
    return tuple(bound)


def _membership_evidence(
    images: Sequence[_ImageRecord],
    closure: _Closure,
    *,
    providers: Sequence[NativeProvider],
    other_pools: Sequence[NativeProvider],
    handles: Mapping[tuple[object, ...], str],
    reasons: Sequence[str],
) -> dict[str, object]:
    """The full evidence of one membership derivation, in plain JSON types."""
    by_key = {image.key: image for image in images}
    provider_paths = {provider.library_path for provider in providers}
    entry_paths = {entry.name for entry in closure.entries}
    roles = {
        path: "+".join(
            role
            for role, member in (
                ("entry", path in entry_paths),
                ("provider", path in provider_paths),
            )
            if member
        )
        for path in provider_paths | entry_paths
    }
    return {
        "scope": (
            "loaded third-party native extension modules and their in-memory "
            "dynamic-linking dependency closure"
        ),
        "source_modules": list(_NUMERICAL_SOURCE_MODULES),
        "provider_symbols": list(_PROVIDER_SYMBOLS),
        "closure": [
            {
                "path": by_key[key].name,
                "role": roles.get(by_key[key].name, "dependency"),
                "sha256": _image_sha256(by_key[key]),
                "binary_id": by_key[key].binary_id,
                "dependencies": [list(edge) for edge in closure.edges.get(key, ())],
            }
            for key in closure.via
        ],
        "handles": [
            {"handle": [str(part) for part in key], "participation": role}
            for key, role in handles.items()
        ],
        "providers": [provider.to_dict() for provider in providers],
        "other_pools": [pool.to_dict() for pool in other_pools],
        "unresolved": list(reasons),
        "loaded_images": [image.name for image in images],
    }


def _derive_membership(
    images: Sequence[_ImageRecord],
    entry_files: Sequence[str],
    handles: Sequence[Any],
) -> NativeMembership:
    """Classify the providers and thread pools of one loaded-image state."""
    owners, owner_reasons = _handle_owners(handles, images)
    by_key = {image.key: image for image in images}
    pooled: dict[tuple[object, ...], list[Any]] = {}
    for handle in handles:
        owner = owners[_handle_key(handle)]
        if owner is not None:
            pooled.setdefault(owner, []).append(handle)
    closure = _dependency_closure(images, entry_files, frozenset(pooled))
    participating, provider_reasons = _participating_images(
        closure, by_key, pooled, images
    )
    reasons = (*closure.reasons, *owner_reasons, *provider_reasons)
    complete = not reasons
    providers = tuple(
        sorted(
            (
                _provider(
                    by_key[key],
                    "participating",
                    _bound_provider_handles(
                        key, closure, by_key, participating, pooled
                    ),
                    chain,
                )
                for key, chain in participating.items()
            ),
            key=lambda p: p.library_path,
        )
    )
    other_pools = tuple(
        sorted(
            (
                _provider(
                    by_key[key], "incidental" if complete else "unknown", pool, ()
                )
                for key, pool in pooled.items()
                if key not in participating
            ),
            key=lambda p: p.library_path,
        )
    )
    participation: dict[tuple[object, ...], str] = {}
    for handle in handles:
        owner = owners[_handle_key(handle)]
        participation[_handle_key(handle)] = (
            "participating"
            if owner in participating
            else ("incidental" if complete and owner is not None else "unknown")
        )
    evidence = _membership_evidence(
        images,
        closure,
        providers=providers,
        other_pools=other_pools,
        handles=participation,
        reasons=reasons,
    )
    canonical = json.dumps(evidence, sort_keys=True, separators=(",", ":"))
    return NativeMembership(
        complete=complete,
        entry_points=len(closure.entries),
        closure_images=len(closure.via),
        loaded_images=len(images),
        providers=providers,
        other_pools=other_pools,
        unresolved=reasons,
        evidence_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        evidence=MappingProxyType(evidence),
        participation=MappingProxyType(participation),
    )


_MEMBERSHIP_CACHE: dict[tuple[object, ...], NativeMembership] = {}


def native_membership(handles: Sequence[Any]) -> NativeMembership:
    """The evidence-bound provider membership of the process for ``handles``.

    Derived from the loaded images, the loaded entry modules and the handles; the
    last derivation is reused only while all three are unchanged, so a later load,
    an unload or a different handle set is classified afresh.
    """
    images = _loaded_images()
    entry_files = _entry_module_files()
    key = (
        tuple(image.key for image in images),
        entry_files,
        tuple(
            (_handle_key(h), bool(getattr(h, "is_mode_only", False))) for h in handles
        ),
    )
    membership = _MEMBERSHIP_CACHE.get(key)
    if membership is None:
        membership = _derive_membership(images, entry_files, handles)
        _MEMBERSHIP_CACHE.clear()
        _MEMBERSHIP_CACHE[key] = membership
    return membership


def describe_native_membership(
    handles: Sequence[Any] | None = None,
) -> dict[str, object]:
    """The full membership evidence of the process, for host probes and review."""
    if handles is None:
        handles = _discover_library_handles()
    membership = native_membership(handles)
    return {
        **membership.to_dict(),
        "evidence": dict(membership.evidence),
    }


# ----------------------------------------------------- provenance & controller


@dataclass(frozen=True)
class ThreadProvenance:
    """Complete thread-control and machine-topology provenance of a controlled block.

    Attributes
    ----------
    options : ThreadOptions
        Resolved thread option record.
    requested_threads : int
        Requested thread count (``options.threads``).
    python_kernel_threads : int
        Active worker threads in the Python/NumPy fixed-effect absorption and
        profiling kernels (always ``1`` in the accepted NumPy implementation).
    libraries : tuple of LibraryThreadState
        Discovered native BLAS/OpenMP libraries and their thread counts / modes
        before, during and after the controlled block.
    env_before : Mapping of str to str or None
        Thread environment variables before entering the controlled block.
    env_active : Mapping of str to str
        Thread environment variables set while inside the controlled block.
    env_restored : Mapping of str to str or None, or None
        Thread environment variables read back after exiting the controlled block.
    topology : CPUTopology
        Observed machine, CPU affinity and cgroup topology.
    numpy_blas_name : str
        NumPy build-dependency BLAS name.
    numpy_blas_version : str
        NumPy build-dependency BLAS version.
    scipy_blas_name : str
        SciPy build-dependency BLAS name.
    scipy_blas_version : str
        SciPy build-dependency BLAS version.
    all_libraries_matched : bool
        Observed provenance: whether at least one native library was discovered
        and every discovered native library, participating or not, reported a
        verified numeric ``threads_active == requested_threads`` inside the block.
    all_libraries_restored : bool or None
        Whether at least one native library was discovered and every discovered
        native library restored its pre-entry numeric thread count or native
        threading mode (and environment variables matched ``env_before``) after
        exiting the block.
    all_libraries_mode_matched : bool
        Observed provenance: whether at least one native library was discovered
        and every discovered native library matched the requested threading mode
        (exact numeric match on numeric backends; mode ``1`` for ``threads=1`` or
        mode ``0`` for ``threads>1`` on mode-only backends such as Apple
        Accelerate).
    numeric_control_verified : bool
        Whether the provider membership is complete, at least one provider
        participates, no participating provider lacks a numeric control and every
        thread-control handle of a participating provider reported a genuine
        numeric ``threads_active == requested_threads``; after exit, also whether
        the membership stayed unchanged through the block.
    membership : NativeMembership or None
        The provider membership derived when the block was entered.
    membership_complete : bool
        Whether ``membership`` was complete.
    membership_unchanged : bool or None
        Whether the membership derived on exit was complete with the same
        participating providers, ``None`` before exit.
    membership_exit : NativeMembership or None
        The provider membership derived on exit, ``None`` before exit.
    """

    options: ThreadOptions
    requested_threads: int
    python_kernel_threads: int
    libraries: tuple[LibraryThreadState, ...]
    env_before: Mapping[str, str | None]
    env_active: Mapping[str, str]
    env_restored: Mapping[str, str | None] | None
    topology: CPUTopology
    numpy_blas_name: str
    numpy_blas_version: str
    scipy_blas_name: str
    scipy_blas_version: str
    all_libraries_matched: bool
    all_libraries_restored: bool | None = None
    all_libraries_mode_matched: bool = False
    numeric_control_verified: bool = False
    membership: NativeMembership | None = None
    membership_complete: bool = False
    membership_unchanged: bool | None = None
    membership_exit: NativeMembership | None = None

    def membership_record(self) -> dict[str, object]:
        """The entry membership summary and, when it differs, the exit summary."""
        exit_summary = (
            None
            if self.membership_exit is None
            or (
                self.membership is not None
                and self.membership_exit.evidence_sha256
                == self.membership.evidence_sha256
            )
            else self.membership_exit.to_dict()
        )
        return {
            "membership_complete": self.membership_complete,
            "membership_unchanged": self.membership_unchanged,
            "membership": (
                None if self.membership is None else self.membership.to_dict()
            ),
            "membership_exit": exit_summary,
        }

    def to_dict(self) -> dict[str, object]:
        """Serialize the provenance record to plain JSON-compatible types."""
        return {
            "options": self.options.to_dict(),
            "requested_threads": self.requested_threads,
            "python_kernel_threads": self.python_kernel_threads,
            "libraries": [lib.to_dict() for lib in self.libraries],
            "env_before": dict(self.env_before),
            "env_active": dict(self.env_active),
            "env_restored": (
                None if self.env_restored is None else dict(self.env_restored)
            ),
            "topology": self.topology.to_dict(),
            "numpy_blas_name": self.numpy_blas_name,
            "numpy_blas_version": self.numpy_blas_version,
            "scipy_blas_name": self.scipy_blas_name,
            "scipy_blas_version": self.scipy_blas_version,
            "all_libraries_matched": self.all_libraries_matched,
            "all_libraries_restored": self.all_libraries_restored,
            "all_libraries_mode_matched": self.all_libraries_mode_matched,
            "numeric_control_verified": self.numeric_control_verified,
            **self.membership_record(),
        }


class ThreadControlSession:
    """Mutable handle yielded by :func:`thread_control` holding live provenance."""

    def __init__(self, provenance: ThreadProvenance) -> None:
        self._provenance = provenance

    @property
    def provenance(self) -> ThreadProvenance:
        """The current :class:`ThreadProvenance` (updated with restoration on exit)."""
        return self._provenance


def _state_restored_ok(state: LibraryThreadState) -> bool:
    """Check whether ``state`` restored its pre-entry numeric count or raw mode."""
    if state.threads_before is not None:
        if state.threads_restored != state.threads_before:
            return False
        if state.raw_mode_before is not None:
            return state.raw_mode_restored == state.raw_mode_before
        return True
    if state.raw_mode_before is not None:
        return (
            state.raw_mode_restored == state.raw_mode_before
            and state.threading_mode_restored == state.threading_mode_before
        )
    return False


def _restore_handles_and_env(
    acquired: Sequence[tuple[Any, LibraryThreadState]],
    env_before: Mapping[str, str | None],
    total_handles: int,
) -> tuple[
    tuple[LibraryThreadState, ...],
    dict[str, str | None],
    bool,
    Exception | None,
]:
    """Restore all modified library handles and environment variables safely."""
    restored_states: list[LibraryThreadState] = []
    first_error: Exception | None = None
    for handle, state in acquired:
        try:
            if hasattr(handle, "restore_state"):
                handle.restore_state(state)
            elif state.threads_before is not None:
                handle.set_threads(state.threads_before)
            restored = handle.get_threads()
            raw_mode_restored = getattr(handle, "get_raw_mode", lambda: None)()
            mode_restored = getattr(
                handle, "get_mode_label", lambda _m=None: "numeric"
            )(raw_mode_restored)
            restored_states.append(
                replace(
                    state,
                    threads_restored=restored,
                    raw_mode_restored=raw_mode_restored,
                    threading_mode_restored=mode_restored,
                )
            )
        except Exception as exc:
            if first_error is None:
                first_error = exc
            restored_states.append(
                replace(
                    state,
                    threads_restored=None,
                    raw_mode_restored=None,
                    threading_mode_restored=None,
                )
            )
    for key, previous in env_before.items():
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous
    env_restored: dict[str, str | None] = {
        key: os.environ.get(key) for key in THREAD_ENV_VARS
    }
    all_restored = (
        bool(restored_states)
        and len(restored_states) == total_handles
        and first_error is None
        and all(_state_restored_ok(s) for s in restored_states)
        and (env_restored == dict(env_before))
    )
    return tuple(restored_states), env_restored, all_restored, first_error


@contextmanager
def thread_control(
    threads: ThreadOptions | int | Mapping[str, object] | None = None,
) -> Iterator[ThreadControlSession]:
    """Apply explicit BLAS/OpenMP thread control and restore prior state on exit.

    Parameters
    ----------
    threads : ThreadOptions, int, Mapping or None, default None
        Requested thread configuration (defaults to ``ThreadOptions(threads=1)``).

    Yields
    ------
    ThreadControlSession
        Session whose ``.provenance`` attribute records the active and restored
        library thread counts, environment variables, CPU topology and the
        provider membership derived on entry and on exit.

    Notes
    -----
    Every discovered library is set, read back and restored.  Numeric control is
    verified over the participating providers of :func:`native_membership` only;
    an incidental thread pool stays recorded without deciding it.  The membership
    is derived again on exit: a provider that a later load added, a changed
    provider identity or newly incomplete evidence leaves the block unverified.
    """
    opts = resolve_thread_options(threads)
    target = opts.threads
    topology = inspect_cpu_topology()
    np_name, np_ver, sp_name, sp_ver = _blas_build_metadata()
    handles = _discover_library_handles(physical_cores=topology.physical_cores)
    membership = native_membership(handles)

    env_before: dict[str, str | None] = {
        key: os.environ.get(key) for key in THREAD_ENV_VARS
    }
    acquired: list[tuple[Any, LibraryThreadState]] = []
    session = ThreadControlSession(
        ThreadProvenance(
            options=opts,
            requested_threads=target,
            python_kernel_threads=1,
            libraries=(),
            env_before=MappingProxyType(dict(env_before)),
            env_active=MappingProxyType({}),
            env_restored=None,
            topology=topology,
            numpy_blas_name=np_name,
            numpy_blas_version=np_ver,
            scipy_blas_name=sp_name,
            scipy_blas_version=sp_ver,
            all_libraries_matched=False,
            all_libraries_restored=None,
            all_libraries_mode_matched=False,
            numeric_control_verified=False,
            membership=membership,
            membership_complete=membership.complete,
        )
    )
    body_raised = False
    try:
        for key in THREAD_ENV_VARS:
            os.environ[key] = str(target)
        env_active: dict[str, str] = {key: os.environ[key] for key in THREAD_ENV_VARS}
        for handle in handles:
            raw_before = getattr(handle, "get_raw_mode", lambda: None)()
            mode_before = getattr(handle, "get_mode_label", lambda _m=None: "numeric")(
                raw_before
            )
            before = handle.get_threads()
            cfg_ceil = getattr(handle, "configured_ceiling", None)
            imp_veclib = getattr(handle, "import_time_veclib_max_threads", None)
            pending_state = LibraryThreadState(
                library_path=handle.library_path,
                backend=handle.backend,
                symbol_prefix=handle.symbol_prefix,
                symbol_suffix=handle.symbol_suffix,
                config=handle.config,
                corename=handle.corename,
                parallel_mode=handle.parallel_mode,
                threads_before=before,
                threads_active=before,
                threads_restored=None,
                numeric_threads_verifiable=(before is not None),
                raw_mode_before=raw_before,
                raw_mode_active=raw_before,
                raw_mode_restored=None,
                threading_mode_before=mode_before,
                threading_mode_active=mode_before,
                threading_mode_restored=None,
                mode_matched=False,
                configured_ceiling=cfg_ceil,
                import_time_veclib_max_threads=imp_veclib,
                participation=membership.participation_of(handle),
            )
            acquired.append((handle, pending_state))
            handle.set_threads(target)
            raw_active = getattr(handle, "get_raw_mode", lambda: None)()
            mode_active = getattr(handle, "get_mode_label", lambda _m=None: "numeric")(
                raw_active
            )
            active = handle.get_threads()
            num_verifiable = bool(active is not None)
            mode_ok = bool(
                getattr(
                    handle,
                    "check_mode_matched",
                    lambda t, a, _rm: a is not None and a == t,
                )(target, active, raw_active)
            )
            acquired[-1] = (
                handle,
                replace(
                    pending_state,
                    threads_active=active,
                    numeric_threads_verifiable=num_verifiable,
                    raw_mode_active=raw_active,
                    threading_mode_active=mode_active,
                    mode_matched=mode_ok,
                ),
            )
        lib_states = tuple(state for _, state in acquired)
        matched = (
            bool(lib_states)
            and len(lib_states) == len(handles)
            and all(
                state.numeric_threads_verifiable
                and state.threads_active is not None
                and state.threads_active == target
                for state in lib_states
            )
        )
        mode_matched = (
            bool(lib_states)
            and len(lib_states) == len(handles)
            and all(state.mode_matched for state in lib_states)
        )
        participating = [s for s in lib_states if s.participation == "participating"]
        numeric_verified = (
            membership.complete
            and bool(membership.providers)
            and all(p.control != "unsupported" for p in membership.providers)
            and bool(participating)
            and len(lib_states) == len(handles)
            and all(
                state.numeric_threads_verifiable
                and state.threads_active is not None
                and state.threads_active == target
                for state in participating
            )
        )
        session._provenance = ThreadProvenance(
            options=opts,
            requested_threads=target,
            python_kernel_threads=1,
            libraries=lib_states,
            env_before=MappingProxyType(dict(env_before)),
            env_active=MappingProxyType(dict(env_active)),
            env_restored=None,
            topology=topology,
            numpy_blas_name=np_name,
            numpy_blas_version=np_ver,
            scipy_blas_name=sp_name,
            scipy_blas_version=sp_ver,
            all_libraries_matched=matched,
            all_libraries_restored=None,
            all_libraries_mode_matched=mode_matched,
            numeric_control_verified=numeric_verified,
            membership=membership,
            membership_complete=membership.complete,
        )
        yield session
    except BaseException:
        body_raised = True
        raise
    finally:
        restored_states, env_restored, all_restored, restore_error = (
            _restore_handles_and_env(acquired, env_before, len(handles))
        )
        exit_membership = native_membership(handles)
        unchanged = (
            exit_membership.complete
            and exit_membership.provider_keys == membership.provider_keys
        )
        session._provenance = replace(
            session.provenance,
            libraries=restored_states,
            env_restored=MappingProxyType(env_restored),
            all_libraries_restored=all_restored,
            numeric_control_verified=(
                session.provenance.numeric_control_verified and unchanged
            ),
            membership_unchanged=unchanged,
            membership_exit=exit_membership,
        )
        if restore_error is not None and not body_raised:
            raise restore_error


# --------------------------------------------------------- solver wrappers


@dataclass(frozen=True, eq=False)
class ThreadControlledSolve[CandidateT: (ProfiledCandidate, IndexCandidate)]:
    """A solver candidate paired with the :class:`ThreadProvenance` of its execution."""

    candidate: CandidateT
    provenance: ThreadProvenance


def solve_profiled_with_threads(
    sample: FESample,
    variance: Family,
    options: SolverOptions,
    absorbed: AbsorbedOptions,
    *,
    threads: ThreadOptions | int | Mapping[str, object] | None = None,
    trigger: Trigger,
    admit: Admission,
    assess: FullAssessor | None = None,
) -> ThreadControlledSolve[ProfiledCandidate]:
    """Run :func:`powermean.solve.newton_profiled.solve` under thread control."""
    with thread_control(threads) as session:
        candidate = newton_profiled.solve(
            sample,
            variance,
            options,
            absorbed,
            trigger=trigger,
            admit=admit,
            assess=assess,
        )
    return ThreadControlledSolve(candidate=candidate, provenance=session.provenance)


def solve_index_with_threads(
    sample: FESample,
    variance: Family,
    options: SolverOptions,
    absorbed: AbsorbedOptions,
    *,
    threads: ThreadOptions | int | Mapping[str, object] | None = None,
    trigger: Trigger,
    admit: Admission,
    assess: IndexAssessor | None = None,
) -> ThreadControlledSolve[IndexCandidate]:
    """Run :func:`powermean.solve.newton_absorbed.solve` under thread control."""
    with thread_control(threads) as session:
        candidate = newton_absorbed.solve(
            sample,
            variance,
            options,
            absorbed,
            trigger=trigger,
            admit=admit,
            assess=assess,
        )
    return ThreadControlledSolve(candidate=candidate, provenance=session.provenance)
