# Native extension notices and source

Powermean's own source is licensed as stated in `LICENSE`. Third-party
source covered by the notices below retains its own terms. The exact source
references identify the inputs of the checked Linux ARM64 CPython 3.12 build;
they do not describe every build from this source distribution.

## Eigen

The checked native IOLS extension used headers from Eigen 3.4.1, commit
`d71c30c47858effcbd39967097a2d99ee48db464`. Obtain the corresponding
[Eigen source archive](https://gitlab.com/libeigen/eigen/-/archive/d71c30c47858effcbd39967097a2d99ee48db464/eigen-d71c30c47858effcbd39967097a2d99ee48db464.tar.gz)
with its original notices. The MPL-covered Eigen source is offered under
the [Mozilla Public License 2.0](eigen/COPYING.MPL2). See the upstream
[license overview](https://gitlab.com/libeigen/eigen/-/raw/d71c30c47858effcbd39967097a2d99ee48db464/COPYING.README),
[Intel notice](eigen/COPYING.BSD) and
[additional header notices](eigen/HEADER_NOTICES.txt). The checked basis
extension did not use Eigen.

## pybind11

Both checked native extensions used pybind11 3.1.0, commit
`97bf890db679505a14dfe547a5e77bb2bd05dc90`. Obtain its
[source archive](https://codeload.github.com/pybind/pybind11/tar.gz/97bf890db679505a14dfe547a5e77bb2bd05dc90)
and see the [pybind11 license](PYBIND11_LICENSE).

That source identifies
[`detail/function_ref.h`](https://github.com/pybind/pybind11/blob/97bf890db679505a14dfe547a5e77bb2bd05dc90/include/pybind11/detail/function_ref.h)
as adapted from ExecuTorch's `FunctionRef`, itself adapted from LLVM. The
header carries Meta Platforms, Inc. and affiliates' copyright and the
LLVM Apache 2.0 with LLVM-exception notice. The
[ExecuTorch BSD terms](pybind11/EXECUTORCH_LICENSE) and
[LLVM license and exceptions](pybind11/LLVM_LICENSE) are included.
The immutable pybind11 archive above contains the actual modified header.

See the [corresponding upstream ExecuTorch notice](https://github.com/pytorch/executorch/blob/3755d3bd41cccddd65746b90b5d1fd40e104addf/LICENSE)
and the [LLVM notice reference](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/LICENSE.TXT).
