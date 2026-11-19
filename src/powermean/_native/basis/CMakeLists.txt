# D19: installed libraries or explicit inputs only; no network downloads.
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# The exact row reduction of the FE incidence (fe/certificate_basis.py, the
# operational echelon) in checked 64-bit rational arithmetic.  Integer-only: no
# floating-point flag can change a result; every overflow declines to Python.
pybind11_add_module(_echelon_cpp MODULE NO_EXTRAS echelon.cpp)
target_compile_features(_echelon_cpp PRIVATE cxx_std_17)
if(NOT MSVC)
  target_compile_options(_echelon_cpp PRIVATE -fno-strict-overflow)
endif()
install(TARGETS _echelon_cpp LIBRARY DESTINATION powermean/_native/basis
        RUNTIME DESTINATION powermean/_native/basis)
