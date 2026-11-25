# Install Powermean

Use Python 3.12 or later. The examples in this guide require Powermean 0.1.1 or
later.

## Install in your analysis environment

Install the package from PyPI in the environment where you run your analysis.

```bash
python -m pip install "powermean>=0.1.1"
python -c "from importlib.metadata import version; print(version('powermean'))"
```

A local wheel or source archive installs with `python -m pip install`
followed by the filename. From a source checkout, run `python -m pip install .`.
A wheel name gives its Python version and platform.
[IOLS installation](iols_installation.md) explains the compiled and NumPy-only wheels.

The release workflow builds separate wheels for CPython 3.12, 3.13 and 3.14
on Linux, macOS and Windows for x86-64 and ARM64. The macOS
wheels target macOS 15 or later.

The second command prints the installed version. Keep that version with your analysis.

## Know what the installation needs

| Dependency | Declared minimum |
| --- | --- |
| Python | 3.12 |
| NumPy | 2.3.1 |
| SciPy | 1.15.0 |
| pandas | 2.2.2 |
| Formulaic | 1.0.0 |
| within-py | 0.4, below 0.5 |

The source distribution uses the `scikit_build_core.build` backend, with declared
build requirements `scikit-build-core>=0.10`, `pybind11>=2.13`, `numpy>=2.3.1`
and `cmeel-eigen==3.4.1`.
Normal pip build isolation resolves these separately from runtime dependencies.
Save the dependency versions alongside the distribution filename and digest. The
[version-report recipe](help/report-a-problem.md#record-versions-without-sharing-credentials)
shows how.

The CMake source build enables the optional IOLS C++ extension by default.
That build needs C++17, Python development-module support and Eigen headers.
Pip's isolated build environment supplies Eigen. On macOS the build also links
an existing LAPACK implementation. Installing a compatible compiled wheel
instead uses its included extension. A NumPy-only wheel omits that extension.
The source-build switch `-Ccmake.define.POWERMEAN_BUILD_IOLS=OFF` explicitly
omits the IOLS target; it does not establish a compiler-free build or a
platform-independent wheel, since the root CMake project still declares C++.

The optional IOLS solver has compiled and NumPy implementations. Automatic
selection uses NumPy only when the exact extension module is genuinely absent;
a broken extension or a failed calculation is an error, not permission to retry
with NumPy or Newton. An explicit compiled request is refused when the extension
is absent. Each IOLS result records which implementation ran.
[Availability](availability.md) identifies the supported request scope, not a
checked matrix of platforms or dependency versions.

For the checked wheel formats, strict C++ selection, NumPy-only installations
and build troubleshooting, continue to [Install and troubleshoot IOLS](iols_installation.md).

## Run the first example

Start with the self-contained [conditional-mean example](methods/model.md#fit-one-conditional-mean-three-ways).
For the gravity example, run the script from the unpacked source root.

```bash
python examples/first_fit.py
```

The [gravity walkthrough](get-started.md) explains which trade flows identify
the agreement coefficient and how to report its percentage effect.

<!-- guide-navigation:start -->
[Guide index](README.md) · [Next: Three worked examples](quick-tour.md)
<!-- guide-navigation:end -->
