# D19: installed libraries or explicit inputs only; no network downloads.
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_path(EIGEN3_INCLUDE_DIR NAMES Eigen/Core PATH_SUFFIXES eigen3)
if(NOT EIGEN3_INCLUDE_DIR)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
      "from importlib.metadata import distribution; print(distribution('cmeel-eigen').locate_file('cmeel.prefix/include/eigen3'))"
    OUTPUT_VARIABLE eigen_headers
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  find_path(EIGEN3_INCLUDE_DIR NAMES Eigen/Core HINTS "${eigen_headers}" REQUIRED)
endif()

# Match the paper's least-squares selection when explicitly supplied for replay;
# ordinary Linux builds use _kernel_cpp's existing Eigen::JacobiSVD branch so
# wheels do not bind the build NumPy's private hashed OpenBLAS SONAME.
set(POWERMEAN_IOLS_OPENBLAS64_LIBRARY "" CACHE FILEPATH
    "Existing NumPy scipy-openblas64 library (empty: Eigen::JacobiSVD on Linux/Windows; detect NumPy, then LAPACK on macOS)")
if(NOT POWERMEAN_IOLS_OPENBLAS64_LIBRARY AND APPLE)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
      "import glob, os, numpy; libs=glob.glob(os.path.join(os.path.dirname(numpy.__file__), '..', 'numpy.libs', 'libscipy_openblas64_*.so')); print(libs[0] if libs else '')"
    RESULT_VARIABLE numpy_probe_status
    OUTPUT_VARIABLE POWERMEAN_IOLS_OPENBLAS64_LIBRARY
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT numpy_probe_status EQUAL 0)
    message(FATAL_ERROR "NumPy build-library discovery failed")
  endif()
endif()

pybind11_add_module(_kernel_cpp MODULE NO_EXTRAS bind.cpp)
target_compile_features(_kernel_cpp PRIVATE cxx_std_17)
target_include_directories(_kernel_cpp PRIVATE "${EIGEN3_INCLUDE_DIR}")
if(POWERMEAN_IOLS_OPENBLAS64_LIBRARY)
  target_compile_definitions(_kernel_cpp PRIVATE PYIOLS_HAVE_SCIPY_OPENBLAS64=1)
  target_link_libraries(_kernel_cpp PRIVATE "${POWERMEAN_IOLS_OPENBLAS64_LIBRARY}")
  if(NOT MSVC AND NOT APPLE)
    get_filename_component(openblas64_dir "${POWERMEAN_IOLS_OPENBLAS64_LIBRARY}" DIRECTORY)
    set_target_properties(_kernel_cpp PROPERTIES
      BUILD_RPATH "${openblas64_dir}"
      INSTALL_RPATH "$ORIGIN/../../../numpy.libs")
  endif()
elseif(APPLE)
  find_package(LAPACK REQUIRED)
  target_compile_definitions(_kernel_cpp PRIVATE PYIOLS_HAVE_LAPACK=1)
  target_link_libraries(_kernel_cpp PRIVATE ${LAPACK_LIBRARIES})
endif()
if(MSVC)
  target_compile_options(_kernel_cpp PRIVATE /fp:precise)
  target_compile_definitions(_kernel_cpp PRIVATE
    POWERMEAN_IOLS_BUILD_FP_FLAGS="/fp:precise")
else()
  target_compile_options(_kernel_cpp PRIVATE
    -fno-fast-math -fno-finite-math-only -ffp-contract=off)
  target_compile_definitions(_kernel_cpp PRIVATE
    POWERMEAN_IOLS_BUILD_FP_FLAGS="-fno-fast-math,-fno-finite-math-only,-ffp-contract=off")
endif()
install(TARGETS _kernel_cpp LIBRARY DESTINATION powermean/_native/iols
        RUNTIME DESTINATION powermean/_native/iols)
