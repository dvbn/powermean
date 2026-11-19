# Install and troubleshoot no-FE IOLS

Powermean provides a compiled implementation of the complete no-fixed-effects
IOLS iteration loop and a corresponding NumPy implementation. Newton remains
the default solver; installing an IOLS extension does not change that default.

## Choose the artifact before installing

Choose a wheel for your Python version, operating system and architecture.
Use a compiled wheel if you want to require the C++ IOLS backend. The wheel
name identifies the Python and platform tags; check the artifact description
to distinguish a compiled build from a NumPy-only build.

Start in a separate, compatible Python environment with the dependencies declared
by the package already available. A local, offline wheel-installation recipe is:

```sh
python -m pip install --no-index --no-deps --no-compile \
    ./artifacts/powermean-0.1.0-*.whl
```

Put only the selected wheel in `artifacts`. This offline command uses the
dependencies already in your environment. For normal dependency resolution,
follow the [installation guide](install.md).

## Check the installed IOLS implementation

Run the [example script](examples/iols_first_fit.py) using the Python
environment where you installed Powermean:

```sh
python docs/examples/iols_first_fit.py --backend auto
```

The script creates a fixed-seed, 240-row simulated sample, requests no-FE PPML
with IOLS and robust covariance, and prints the coefficient table and actual implementation.
On a compiled build, the solver information includes:

```text
Solver: iols
requested_backend: auto
actual_backend: cpp
fallback_used: False
```

The script's `--backend auto` means that it omits `iols_backend` from the package
request. It does not pass `iols_backend="auto"`, which is not a public option
value. With automatic selection, the example reports `numpy` on a
NumPy-only build. Use the reported backend to confirm which one ran.

## Select the implementation deliberately

For an IOLS fit, `solver="iols"` selects IOLS. Within `solver_options`, the only
public `iols_backend` values are `"cpp"` and `"numpy"`:

- Omit the backend to prefer C++. Automatic NumPy selection is allowed only when
  the exact `powermean._native.iols._kernel_cpp` module is genuinely absent.
- Supply `iols_backend="cpp"` to require C++. Its absence or failure is an error,
  not permission to switch implementations.
- Supply `iols_backend="numpy"` to select NumPy intentionally.

`pm.solver_defaults("iols")` returns a typed options record. Its default backend
value is `"cpp"`, but the record separately retains that the caller omitted the
backend. Passing that typed record preserves omission. Passing its complete
`.to_dict()` mapping supplies the backend explicitly and therefore requires C++.
This distinction matters on a NumPy-only installation.

Inspect the returned result's provenance rather than inferring which
implementation ran from the package name or an iteration count. A broken native
extension, exhausted budget or failed numerical assessment does not cause a retry
with NumPy or Newton.

## Source-build requirements

A source build is distinct from installing a compatible wheel. Its declared build
requirements are installed by pip when it builds in isolation. The CMake
configuration uses installed native libraries and headers.

The compiled target uses C++17, Python development-module support, pybind11 and
Eigen headers supplied by `cmeel-eigen` or a local installation. On Linux
and Windows it uses its Eigen JacobiSVD least-squares
branch and links no LAPACK library unless an OpenBLAS library is explicitly
supplied through `POWERMEAN_IOLS_OPENBLAS64_LIBRARY`. On macOS
the build links the build NumPy's scipy-openblas64 library when it
finds one in NumPy's `numpy.libs` folder, and otherwise requires an existing
LAPACK implementation. Set `EIGEN3_INCLUDE_DIR` if CMake cannot find the
existing Eigen headers.

`POWERMEAN_BUILD_IOLS` is the CMake switch for including the IOLS extension and is
`ON` by default. Selecting `OFF` omits that target; it does not establish a
compiler-free build or a platform-independent wheel. The current root CMake
project still declares the C++ language.

## Troubleshooting without changing the requested model

| Observation | Meaning and appropriate next step |
| --- | --- |
| The example reports NumPy when selection was omitted | On a NumPy-only artifact this is expected. Check the supplied artifact manifest and the recorded actual backend. |
| Explicit C++ fails because its exact module is absent | The installed artifact cannot satisfy that explicit request. Use a compatible compiled artifact, or explicitly request NumPy if that is what you intend. |
| Import reports a missing transitive dependency, incompatible ABI or missing symbol | This is not extension absence. Correct the installation or library compatibility; automatic fallback is not applied. |
| The module lacks a required entry point | Install the wheel for your Python version and platform. |
| `iols_backend="auto"` is refused | Omit the option for automatic availability-based selection. `"auto"` is an internal request label, not a public literal. |
| A typed default works but its `.to_dict()` mapping fails without C++ | The complete mapping made C++ explicit. Pass the typed record to preserve omission, or explicitly request NumPy. |
| A budget of one constructs successfully but is refused when fitting | Construction and execution have distinct domains: `max_phase1_iterations` and `stopcrit` require at least two for execution; `max_phase2_evaluations=1` remains valid. Values are not silently clamped. |
| IOLS refuses FE, weights, offsets, exposure, instruments or a supplied start | This no-FE solver does not support those requests. This is a capability refusal, not a broken installation or a reason for hidden solver substitution. |
| A fitted solution fails its convergence or inference checks | An installed extension is not a guarantee of a verified fit. Keep the failure visible; changing implementation or weakening a check is not an automatic recovery step. |

The supported IOLS domain uses positive `omega`, power working variance with
`tau` in `[1, 2]`, a unit intercept, at least one slope and more observations than
design columns. NB2, log OLS, cancellation and bias correction are not supported
by this route. The package's separate FE/cancellation work does not expand this
solver's domain.

Use the [IOLS option reference](reference/iols_options.md) for all nine
controls and their domains. Report the small public call, the relevant exception
and the implementation provenance when asking for help; do not share credentials,
private observations or a complete environment dump.

<!-- guide-navigation:start -->
[Previous: IOLS options](reference/iols_options.md) · [Guide index](README.md) · [Next: Troubleshooting](help/troubleshooting.md)
<!-- guide-navigation:end -->
