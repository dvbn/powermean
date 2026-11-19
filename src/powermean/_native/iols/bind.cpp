#include <Python.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/SVD>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

#if defined(PYIOLS_HAVE_SCIPY_OPENBLAS64)
using PyiolsLapackInt = std::int64_t;
extern "C" {
void scipy_dgelsd_64_(
    const PyiolsLapackInt* m,
    const PyiolsLapackInt* n,
    const PyiolsLapackInt* nrhs,
    double* a,
    const PyiolsLapackInt* lda,
    double* b,
    const PyiolsLapackInt* ldb,
    double* s,
    const double* rcond,
    PyiolsLapackInt* rank,
    double* work,
    const PyiolsLapackInt* lwork,
    PyiolsLapackInt* iwork,
    PyiolsLapackInt* info);
}
#define PYIOLS_DGELSD scipy_dgelsd_64_
#elif defined(PYIOLS_HAVE_LAPACK)
using PyiolsLapackInt = int;
extern "C" {
void dgelsd_(
    const PyiolsLapackInt* m,
    const PyiolsLapackInt* n,
    const PyiolsLapackInt* nrhs,
    double* a,
    const PyiolsLapackInt* lda,
    double* b,
    const PyiolsLapackInt* ldb,
    double* s,
    const double* rcond,
    PyiolsLapackInt* rank,
    double* work,
    const PyiolsLapackInt* lwork,
    PyiolsLapackInt* iwork,
    PyiolsLapackInt* info);
}
#define PYIOLS_DGELSD dgelsd_
#endif

namespace pyiols::kernel {

using RowMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

double stable_l2_norm(
    const Eigen::Ref<const Eigen::VectorXd>& values) {
    double norm = 0.0;
    for (Eigen::Index idx = 0; idx < values.size(); ++idx) {
        norm = std::hypot(norm, std::abs(values(idx)));
    }
    return norm;
}

Eigen::MatrixXd min_norm_lstsq_eigen(
    const Eigen::Ref<const Eigen::MatrixXd>& a,
    const Eigen::Ref<const Eigen::MatrixXd>& b) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        Eigen::MatrixXd(a),
        Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd singular = svd.singularValues();
    const double max_sv = singular.size() == 0 ? 0.0 : singular.maxCoeff();
    const double cutoff =
        std::numeric_limits<double>::epsilon() *
        static_cast<double>(std::max(a.rows(), a.cols())) * max_sv;
    Eigen::MatrixXd projected = svd.matrixU().transpose() * b;
    for (Eigen::Index row = 0; row < projected.rows(); ++row) {
        const double sv = singular(row);
        if (sv > cutoff) {
            projected.row(row) /= sv;
        } else {
            projected.row(row).setZero();
        }
    }
    return svd.matrixV() * projected;
}

Eigen::MatrixXd min_norm_lstsq(
    const Eigen::Ref<const Eigen::MatrixXd>& a,
    const Eigen::Ref<const Eigen::MatrixXd>& b) {
    if (a.rows() == 0 || a.cols() == 0) {
        throw std::invalid_argument("least-squares matrix must be non-empty");
    }
    if (a.rows() != b.rows()) {
        throw std::invalid_argument("least-squares row dimensions do not match");
    }
#if defined(PYIOLS_HAVE_SCIPY_OPENBLAS64) || defined(PYIOLS_HAVE_LAPACK)
    if (a.rows() >
            static_cast<Eigen::Index>(std::numeric_limits<PyiolsLapackInt>::max()) ||
        a.cols() >
            static_cast<Eigen::Index>(std::numeric_limits<PyiolsLapackInt>::max()) ||
        b.cols() >
            static_cast<Eigen::Index>(std::numeric_limits<PyiolsLapackInt>::max())) {
        throw std::overflow_error("least-squares dimensions exceed LAPACK int range");
    }
    const PyiolsLapackInt m = static_cast<PyiolsLapackInt>(a.rows());
    const PyiolsLapackInt n = static_cast<PyiolsLapackInt>(a.cols());
    const PyiolsLapackInt nrhs = static_cast<PyiolsLapackInt>(b.cols());
    const PyiolsLapackInt lda = std::max<PyiolsLapackInt>(1, m);
    const PyiolsLapackInt ldb = std::max<PyiolsLapackInt>(1, std::max(m, n));
    const PyiolsLapackInt minmn = std::min(m, n);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> a_lapack =
        a;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> b_lapack =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>::Zero(
            ldb,
            nrhs);
    b_lapack.topRows(m) = b;
    Eigen::VectorXd singular(minmn);

    const double rcond =
        std::numeric_limits<double>::epsilon() *
        static_cast<double>(std::max(a.rows(), a.cols()));
    PyiolsLapackInt rank = 0;
    PyiolsLapackInt info = 0;

    const PyiolsLapackInt smlsiz = 25;
    PyiolsLapackInt nlvl = 0;
    if (minmn > smlsiz + 1) {
        nlvl = static_cast<PyiolsLapackInt>(
            std::log(static_cast<double>(minmn) / static_cast<double>(smlsiz + 1)) /
            std::log(2.0)) +
               1;
    }
    const PyiolsLapackInt liwork =
        std::max<PyiolsLapackInt>(
            1,
            3 * minmn * std::max<PyiolsLapackInt>(0, nlvl) + 11 * minmn);
    std::vector<PyiolsLapackInt> iwork(static_cast<std::size_t>(liwork), 0);

    double work_query = 0.0;
    PyiolsLapackInt lwork = -1;
    PYIOLS_DGELSD(
        &m,
        &n,
        &nrhs,
        a_lapack.data(),
        &lda,
        b_lapack.data(),
        &ldb,
        singular.data(),
        &rcond,
        &rank,
        &work_query,
        &lwork,
        iwork.data(),
        &info);
    if (info != 0 || !std::isfinite(work_query) || work_query <= 0.0) {
        throw std::runtime_error("LAPACK DGELSD workspace query failed");
    }
    lwork = static_cast<PyiolsLapackInt>(std::ceil(work_query));
    std::vector<double> work(
        static_cast<std::size_t>(std::max<PyiolsLapackInt>(1, lwork)),
        0.0);
    PYIOLS_DGELSD(
        &m,
        &n,
        &nrhs,
        a_lapack.data(),
        &lda,
        b_lapack.data(),
        &ldb,
        singular.data(),
        &rcond,
        &rank,
        work.data(),
        &lwork,
        iwork.data(),
        &info);
    if (info != 0) {
        throw std::runtime_error("LAPACK DGELSD least-squares solve failed");
    }
    return Eigen::MatrixXd(b_lapack.topRows(n));
#else
    return min_norm_lstsq_eigen(a, b);
#endif
}

Eigen::MatrixXd min_norm_lstsq_anderson(
    const Eigen::Ref<const Eigen::MatrixXd>& a,
    const Eigen::Ref<const Eigen::MatrixXd>& b) {
    return min_norm_lstsq(a, b);
}

}  // namespace pyiols::kernel

namespace {

using pyiols::kernel::RowMatrixXd;

constexpr double kFloat64Max = std::numeric_limits<double>::max();
const double kFloat64LogMax = std::log(kFloat64Max);

Eigen::VectorXd vector_from_array(
    py::array_t<double, py::array::c_style | py::array::forcecast> array,
    const char* name) {
    const py::buffer_info info = array.request();
    if (info.ndim != 1) {
        throw std::invalid_argument(std::string(name) + " must be one-dimensional");
    }
    Eigen::VectorXd out(info.shape[0]);
    const auto* ptr = static_cast<const double*>(info.ptr);
    if (info.shape[0] > 0) {
        std::memcpy(
            out.data(),
            ptr,
            static_cast<std::size_t>(info.shape[0]) * sizeof(double));
    }
    return out;
}

RowMatrixXd matrix_from_array(
    py::array_t<double, py::array::c_style | py::array::forcecast> array,
    const char* name) {
    const py::buffer_info info = array.request();
    if (info.ndim != 2) {
        throw std::invalid_argument(std::string(name) + " must be two-dimensional");
    }
    RowMatrixXd out(info.shape[0], info.shape[1]);
    const auto* ptr = static_cast<const double*>(info.ptr);
    if (info.shape[0] > 0 && info.shape[1] > 0) {
        std::memcpy(
            out.data(),
            ptr,
            static_cast<std::size_t>(info.shape[0] * info.shape[1]) *
                sizeof(double));
    }
    return out;
}

std::vector<std::int64_t> groups_from_array(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> groups,
    Eigen::Index expected_rows,
    std::int64_t n_groups) {
    const py::buffer_info info = groups.request();
    if (info.ndim != 1) {
        throw std::invalid_argument("groups must be one-dimensional");
    }
    if (info.shape[0] != expected_rows) {
        throw std::invalid_argument("values and groups must have the same rows");
    }
    const auto* ptr = static_cast<const std::int64_t*>(info.ptr);
    std::vector<std::int64_t> out(static_cast<std::size_t>(info.shape[0]));
    for (py::ssize_t row = 0; row < info.shape[0]; ++row) {
        if (ptr[row] < 0 || ptr[row] >= n_groups) {
            throw std::invalid_argument("group labels must be in [0, n_groups)");
        }
        out[static_cast<std::size_t>(row)] = ptr[row];
    }
    return out;
}

py::array_t<double> vector_to_array(const Eigen::Ref<const Eigen::VectorXd>& vec) {
    py::array_t<double> out({vec.size()});
    if (vec.size() > 0) {
        std::memcpy(
            out.mutable_data(),
            vec.data(),
            static_cast<std::size_t>(vec.size()) * sizeof(double));
    }
    return out;
}

py::array_t<double> matrix_to_array(const Eigen::Ref<const Eigen::MatrixXd>& mat) {
    py::array_t<double> out({mat.rows(), mat.cols()});
    auto unchecked = out.mutable_unchecked<2>();
    for (py::ssize_t row = 0; row < unchecked.shape(0); ++row) {
        for (py::ssize_t col = 0; col < unchecked.shape(1); ++col) {
            unchecked(row, col) = mat(row, col);
        }
    }
    return out;
}

py::object matrix_or_vector_to_array(
    const Eigen::Ref<const Eigen::MatrixXd>& mat,
    bool as_vector) {
    if (!as_vector) {
        return matrix_to_array(mat);
    }
    return vector_to_array(mat.col(0));
}

Eigen::MatrixXd matrix_from_values(
    py::array_t<double, py::array::c_style | py::array::forcecast> values,
    bool& was_vector) {
    const py::buffer_info info = values.request();
    if (info.ndim == 1) {
        was_vector = true;
        Eigen::MatrixXd out(info.shape[0], 1);
        const auto* ptr = static_cast<const double*>(info.ptr);
        for (py::ssize_t row = 0; row < info.shape[0]; ++row) {
            out(row, 0) = ptr[row];
        }
        return out;
    }
    if (info.ndim == 2) {
        was_vector = false;
        return matrix_from_array(values, "values");
    }
    throw std::invalid_argument("values must be one- or two-dimensional");
}

struct DemeanWorkspaceCore {
    Eigen::MatrixXd scratch;
    Eigen::MatrixXd previous;
    Eigen::MatrixXd sums;
    Eigen::VectorXd denom;
};

void demean_oneway_matrix_into_core(
    const Eigen::Ref<const Eigen::MatrixXd>& values,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups,
    const Eigen::VectorXd* weights,
    Eigen::MatrixXd& out,
    DemeanWorkspaceCore& workspace) {
    workspace.sums.resize(static_cast<Eigen::Index>(n_groups), values.cols());
    workspace.sums.setZero();
    workspace.denom.resize(static_cast<Eigen::Index>(n_groups));
    workspace.denom.setZero();
    for (Eigen::Index row = 0; row < values.rows(); ++row) {
        const auto group = static_cast<Eigen::Index>(
            groups[static_cast<std::size_t>(row)]);
        const double weight = weights != nullptr ? (*weights)(row) : 1.0;
        workspace.sums.row(group) += weight * values.row(row);
        workspace.denom(group) += weight;
    }

    for (Eigen::Index group = 0; group < workspace.sums.rows(); ++group) {
        if (weights != nullptr) {
            workspace.sums.row(group) /=
                std::max(workspace.denom(group), 1.0e-300);
        } else {
            const double divisor = std::max(workspace.denom(group), 1.0);
            workspace.sums.row(group) /= divisor;
        }
    }

    out.resize(values.rows(), values.cols());
    for (Eigen::Index row = 0; row < values.rows(); ++row) {
        out.row(row) = values.row(row) -
                       workspace.sums.row(groups[static_cast<std::size_t>(row)]);
    }
}

Eigen::MatrixXd demean_oneway_matrix(
    const Eigen::Ref<const Eigen::MatrixXd>& values,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups,
    const std::optional<Eigen::VectorXd>& weights) {
    Eigen::MatrixXd out;
    DemeanWorkspaceCore workspace;
    const Eigen::VectorXd* weights_ptr = weights.has_value() ? &(*weights) : nullptr;
    demean_oneway_matrix_into_core(
        values,
        groups,
        n_groups,
        weights_ptr,
        out,
        workspace);
    return out;
}

py::array_t<double> exp_guard(
    py::array_t<double, py::array::c_style | py::array::forcecast> values) {
    const py::buffer_info info = values.request();
    py::array_t<double> out(info.shape);
    const auto* src = static_cast<const double*>(info.ptr);
    auto* dst = static_cast<double*>(out.mutable_data());
    for (py::ssize_t idx = 0; idx < info.size; ++idx) {
        const double value = src[idx];
        if (std::isinf(value) && value > 0.0) {
            dst[idx] = kFloat64Max;
        } else if (value > kFloat64LogMax) {
            dst[idx] = kFloat64Max;
        } else {
            dst[idx] = std::exp(value);
        }
    }
    return out;
}

py::array_t<double> group_sum(
    py::array_t<double, py::array::c_style | py::array::forcecast> values,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> groups,
    std::int64_t n_groups) {
    const Eigen::VectorXd values_vec = vector_from_array(values, "values");
    const auto group_vec = groups_from_array(groups, values_vec.size(), n_groups);
    Eigen::VectorXd out = Eigen::VectorXd::Zero(n_groups);
    for (Eigen::Index row = 0; row < values_vec.size(); ++row) {
        out(group_vec[row]) += values_vec(row);
    }
    return vector_to_array(out);
}

py::array_t<double> group_logsumexp(
    py::array_t<double, py::array::c_style | py::array::forcecast> values,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> groups,
    std::int64_t n_groups) {
    const Eigen::VectorXd values_vec = vector_from_array(values, "values");
    const auto group_vec = groups_from_array(groups, values_vec.size(), n_groups);
    Eigen::VectorXd max_by_group =
        Eigen::VectorXd::Constant(n_groups, -std::numeric_limits<double>::infinity());
    std::vector<bool> has_nan(static_cast<std::size_t>(n_groups), false);
    for (Eigen::Index row = 0; row < values_vec.size(); ++row) {
        const auto group = static_cast<Eigen::Index>(group_vec[row]);
        const double value = values_vec(row);
        if (std::isnan(value)) {
            has_nan[static_cast<std::size_t>(group)] = true;
            continue;
        }
        if (value > max_by_group(group)) {
            max_by_group(group) = value;
        }
    }

    Eigen::VectorXd out =
        Eigen::VectorXd::Constant(n_groups, -std::numeric_limits<double>::infinity());
    Eigen::VectorXd sums = Eigen::VectorXd::Zero(n_groups);
    for (Eigen::Index row = 0; row < values_vec.size(); ++row) {
        const auto group = static_cast<Eigen::Index>(group_vec[row]);
        const double max_value = max_by_group(group);
        const double value = values_vec(row);
        if (has_nan[static_cast<std::size_t>(group)] || !std::isfinite(max_value)) {
            continue;
        }
        if (std::isfinite(value)) {
            sums(group) += std::exp(value - max_value);
        }
    }
    for (Eigen::Index group = 0; group < n_groups; ++group) {
        if (has_nan[static_cast<std::size_t>(group)]) {
            continue;
        }
        if (std::isinf(max_by_group(group)) && max_by_group(group) > 0.0) {
            out(group) = std::numeric_limits<double>::infinity();
        } else if (std::isfinite(max_by_group(group)) && sums(group) > 0.0) {
            out(group) = max_by_group(group) + std::log(sums(group));
        }
    }
    return vector_to_array(out);
}

py::object demean_oneway(
    py::array_t<double, py::array::c_style | py::array::forcecast> values,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> groups,
    std::int64_t n_groups,
    py::object weights_obj) {
    bool was_vector = false;
    const Eigen::MatrixXd values_mat = matrix_from_values(values, was_vector);
    const auto group_vec = groups_from_array(groups, values_mat.rows(), n_groups);
    std::optional<Eigen::VectorXd> weights;
    if (!weights_obj.is_none()) {
        weights = vector_from_array(
            py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(
                weights_obj),
            "weights");
        if (weights->size() != values_mat.rows()) {
            throw std::invalid_argument("weights and groups must have the same shape");
        }
    }
    const Eigen::MatrixXd demeaned =
        demean_oneway_matrix(values_mat, group_vec, n_groups, weights);
    return matrix_or_vector_to_array(demeaned, was_vector);
}

py::object lstsq_min_norm(
    py::array_t<double, py::array::c_style | py::array::forcecast> a,
    py::array_t<double, py::array::c_style | py::array::forcecast> b) {
    const Eigen::MatrixXd a_mat = matrix_from_array(a, "a");
    bool b_was_vector = false;
    const Eigen::MatrixXd b_mat = matrix_from_values(b, b_was_vector);
    const Eigen::MatrixXd solution = pyiols::kernel::min_norm_lstsq(a_mat, b_mat);
    return matrix_or_vector_to_array(solution, b_was_vector);
}

py::object anderson_lstsq(
    py::array_t<double, py::array::c_style | py::array::forcecast> f_mat,
    py::array_t<double, py::array::c_style | py::array::forcecast> residual) {
    const Eigen::MatrixXd f_mat_core = matrix_from_array(f_mat, "f_mat");
    bool residual_was_vector = false;
    const Eigen::MatrixXd residual_core =
        matrix_from_values(residual, residual_was_vector);
    const Eigen::MatrixXd solution =
        pyiols::kernel::min_norm_lstsq_anderson(f_mat_core, residual_core);
    return matrix_or_vector_to_array(solution, residual_was_vector);
}

struct EigenVectorPairRingCore {
    std::vector<Eigen::VectorXd> x;
    std::vector<Eigen::VectorXd> f;
    std::size_t capacity = 0;
    std::size_t start = 0;
    std::size_t count = 0;

    // Callers consume references immediately; do not hold x_at/f_at/f_back
    // references across another push because a full ring overwrites slots.
    void reset(std::size_t new_capacity) {
        capacity = new_capacity;
        start = 0;
        count = 0;
        x.clear();
        f.clear();
        x.reserve(capacity);
        f.reserve(capacity);
    }

    void clear() {
        start = 0;
        count = 0;
        x.clear();
        f.clear();
    }

    // runtime_v2 (direct fixed-tau route only): enlarge the ring in place when
    // the Anderson memory adapts upward, keeping the logical order of the
    // retained pairs. The NumPy reference trims its history lists to the
    // adapted aa_m + 1, so after adaptation it uses up to aa_m lags; a fixed
    // ring silently kept the initial memory. No-op unless new_capacity is larger.
    void grow(std::size_t new_capacity) {
        if (new_capacity <= capacity) {
            return;
        }
        std::vector<Eigen::VectorXd> x_ordered;
        std::vector<Eigen::VectorXd> f_ordered;
        x_ordered.reserve(new_capacity);
        f_ordered.reserve(new_capacity);
        for (std::size_t logical = 0; logical < count; ++logical) {
            x_ordered.emplace_back(x_at(logical));
            f_ordered.emplace_back(f_at(logical));
        }
        x = std::move(x_ordered);
        f = std::move(f_ordered);
        capacity = new_capacity;
        start = 0;
    }

    void push(const Eigen::VectorXd& x_value, const Eigen::VectorXd& f_value) {
        if (capacity == 0) {
            return;
        }
        if (count < capacity) {
            x.emplace_back(x_value);
            f.emplace_back(f_value);
            ++count;
            return;
        }
        x[start] = x_value;
        f[start] = f_value;
        start = (start + 1) % capacity;
    }

    std::size_t size() const {
        return count;
    }

    std::size_t physical_index(std::size_t logical_index) const {
        assert(capacity > 0);
        assert(logical_index < count);
        return (start + logical_index) % capacity;
    }

    const Eigen::VectorXd& x_at(std::size_t logical_index) const {
        return x[physical_index(logical_index)];
    }

    const Eigen::VectorXd& f_at(std::size_t logical_index) const {
        return f[physical_index(logical_index)];
    }

    const Eigen::VectorXd& f_back() const {
        assert(count > 0);
        return f_at(count - 1);
    }
};

py::dict backend_info() {
    py::dict out;
    out["kernel_backend"] = "cpp";
    out["python_reference_backend"] = "python_reference";
    out["direct_nofe_status"] = "available";
#ifdef PYIOLS_HAVE_SCIPY_OPENBLAS64
    out["linear_solver"] =
        "NumPy OpenBLAS64 LAPACK DGELSD minimum-norm least squares";
#elif defined(PYIOLS_HAVE_LAPACK)
    out["linear_solver"] = "LAPACK DGELSD minimum-norm least squares";
#else
    out["linear_solver"] = "Eigen::JacobiSVD minimum-norm least squares";
#endif
#if defined(POWERMEAN_IOLS_BUILD_FP_FLAGS)
    out["fp_flags"] = POWERMEAN_IOLS_BUILD_FP_FLAGS;
#else
    out["fp_flags"] = "unrecorded";
#endif
    out["eigen_version"] = std::to_string(EIGEN_WORLD_VERSION) + "." +
        std::to_string(EIGEN_MAJOR_VERSION) + "." + std::to_string(EIGEN_MINOR_VERSION);
    out["input_ownership"] = "snapshot_copy";
    out["gil_policy"] = "release_around_core_solve";
    return out;
}

struct FullrecoverControlCore {
    int anderson_m = 5;
    double eps = 1e-6;
    int stopcrit = 10000;
    int tau_path_steps = 20;
    bool has_min_dtau = false;
    double min_dtau = 0.0;
    int max_path_evaluations = 10000;
    int max_phase1_iterations = 10000;
    int max_phase2_evaluations = 50000;
    int fe_demean_maxiter = 10000;
    double fe_demean_tol = 1e-10;
    double delta_log_min = -5.0;
    double delta_log_max = 5.0;
    int delta_grid_size = 5;
    std::optional<std::vector<double>> pilot_deltas;
    double rho_log_min = -7.0;
    double rho_log_max = 14.0;
    int rho_grid_size = 150;
    // runtime_v3: Phase 2 restart start policy, read ONLY by the direct no-FE
    // fixed-tau route ("phase1_start" = runtime_v2 behaviour,
    // "best_score_iterate" = restart from / return the best preconditioned
    // score iterate). Legacy callers never read it.
    std::string direct_restart_start = "phase1_start";
    bool research_tiered_fe_tol = false;
    double research_inner_fe_demean_tol = 1e-8;
    bool speed_fullrecover_joint_anderson = false;
    bool speed_fullrecover_accept_score_warning_path_steps = false;
    bool speed_small_manyway_exact_demean = false;
    bool speed_compact_fixed_tau_details = false;
    int speed_small_manyway_exact_demean_max_rows = 1200;
    int speed_small_manyway_exact_demean_max_levels = 512;
};

// runtime_v3: one Phase 2 rho attempt of the direct no-FE fixed-tau route
// (twin of the NumPy phase2_attempt_trace entries).
struct Phase2AttemptTraceCore {
    int attempt = 0;
    double rho = 0.0;
    int iterations = 0;
    int evaluations = 0;
    std::string failure_reason;  // empty = converged
    bool has_final_criterion = false;
    double final_criterion = 0.0;
    double best_criterion = std::numeric_limits<double>::quiet_NaN();
    double kappa_last = std::numeric_limits<double>::quiet_NaN();
    std::string start_policy = "phase1_start";
    // runtime_v4 progress-monitor fields (twins of the NumPy trace keys).
    double final_progress_residual = std::numeric_limits<double>::quiet_NaN();
    double best_progress_residual = std::numeric_limits<double>::quiet_NaN();
    int anderson_m_start = 0;
    int anderson_m_end = 0;
    int anderson_accepted = 0;
};

struct FullrecoverIterationCore {
    bool converged = false;
    std::string phase = "not_started";
    int iterations = 0;
    int restarts = 0;
    bool has_final_rho = false;
    double final_rho = 0.0;
    std::vector<double> rho_path;
    bool has_final_criterion = false;
    double final_criterion = 0.0;
    int anderson_m_initial = 5;
    int anderson_m_final = 5;
    std::string failure_reason;
    int phase1_attempts = 0;
    int phase1_accepted = 0;
    int phase1_rejected = 0;
    int phase1_evaluations = 0;  // runtime_v2: profiled core only
    int phase2_evaluations = 0;
    int max_phase2_evaluations = 0;
    double last_finite_criterion = std::numeric_limits<double>::quiet_NaN();
    double best_finite_criterion = std::numeric_limits<double>::quiet_NaN();
    std::string rho_policy;
    double fullcancel_initial_scale = std::numeric_limits<double>::quiet_NaN();
    int rho_candidate_count = 0;
    bool has_rho_hint = false;
    double rho_hint = std::numeric_limits<double>::quiet_NaN();
    double rho_log_min = std::numeric_limits<double>::quiet_NaN();
    double rho_log_max = std::numeric_limits<double>::quiet_NaN();
    int rho_grid_size = 0;
    bool has_rho_min = false;
    double rho_min = std::numeric_limits<double>::quiet_NaN();
    bool has_rho_max = false;
    double rho_max = std::numeric_limits<double>::quiet_NaN();
    std::string algorithm;
    std::string index_policy;
    std::string fe_policy;
    std::string fe_state_policy;
    std::string anderson_policy;
    double qhat_last = std::numeric_limits<double>::quiet_NaN();
    int qhat_invalid_count = 0;
    double kappa_monitor_last = std::numeric_limits<double>::quiet_NaN();
    double kappa_monitor_max = std::numeric_limits<double>::quiet_NaN();
    int kappa_monitor_checks = 0;
    double post_convergence_residual = std::numeric_limits<double>::quiet_NaN();
    double post_convergence_kappa_est = std::numeric_limits<double>::quiet_NaN();
    double post_convergence_score_residual = std::numeric_limits<double>::quiet_NaN();
    double post_convergence_fe_score_residual = std::numeric_limits<double>::quiet_NaN();
    double score_tolerance = std::numeric_limits<double>::quiet_NaN();
    std::string score_validation_policy = "diagnostic";
    std::string score_validation_status = "not_checked";
    std::vector<std::string> score_warning_codes;
    std::vector<std::string> score_warning_messages;
    bool post_convergence_score_exceeds_tolerance = false;
    bool post_convergence_fe_score_exceeds_tolerance = false;
    int joint_hdfe_phi_calls = 0;
    int joint_hdfe_phi_demean_sweeps = 0;
    bool research_tiered_fe_tol = false;
    double research_inner_fe_demean_tol =
        std::numeric_limits<double>::quiet_NaN();
    int phase2_rho_restart_events = 0;
    int phase2_rho_attempts = 0;
    // runtime_v3 direct no-FE fixed-tau route only (profiled core with
    // direct_nofe_fixed_tau = true); defaults are untouched on legacy routes.
    std::string direct_restart_start;
    std::string nonconverged_iterate_policy;
    double kappa_coordinate_row_max = std::numeric_limits<double>::quiet_NaN();
    double best_score_norm = std::numeric_limits<double>::infinity();
    double best_score_rho = std::numeric_limits<double>::quiet_NaN();
    int phase2_warm_restarts = 0;
    int phase2_attempts = 0;
    std::vector<Phase2AttemptTraceCore> phase2_attempt_trace;
    int post_convergence_invalid_map_count = 0;  // runtime_v4, Phase 2
    int phase1_post_convergence_invalid_map_count = 0;  // runtime_v4, Phase 1
};

struct FullrecoverConcentratedCore {
    Eigen::VectorXd beta;
    Eigen::VectorXd linear;
    bool has_linear = false;
    FullrecoverIterationCore iteration;
};

bool all_finite_vec(const Eigen::Ref<const Eigen::VectorXd>& values) {
    for (Eigen::Index idx = 0; idx < values.size(); ++idx) {
        if (!std::isfinite(values(idx))) {
            return false;
        }
    }
    return true;
}

double exp_guard_scalar(double value) {
    if (std::isinf(value) && value > 0.0) {
        return kFloat64Max;
    }
    if (value > kFloat64LogMax) {
        return kFloat64Max;
    }
    return std::exp(value);
}

double logsumexp_all_core(const Eigen::Ref<const Eigen::VectorXd>& values) {
    if (values.size() == 0) {
        return -std::numeric_limits<double>::infinity();
    }
    bool has_posinf = false;
    bool has_finite = false;
    double max_value = -std::numeric_limits<double>::infinity();
    for (Eigen::Index idx = 0; idx < values.size(); ++idx) {
        const double value = values(idx);
        if (std::isinf(value) && value > 0.0) {
            has_posinf = true;
        }
        if (std::isfinite(value)) {
            has_finite = true;
            max_value = std::max(max_value, value);
        }
    }
    if (has_posinf) {
        return std::numeric_limits<double>::infinity();
    }
    if (!has_finite) {
        return -std::numeric_limits<double>::infinity();
    }
    double sum = 0.0;
    for (Eigen::Index idx = 0; idx < values.size(); ++idx) {
        const double value = values(idx);
        if (std::isfinite(value)) {
            sum += std::exp(value - max_value);
        }
    }
    return max_value + std::log(sum);
}

bool is_singleton_intercept_group(
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups) {
    if (n_groups != 1) {
        return false;
    }
    for (const std::int64_t group : groups) {
        if (group != 0) {
            return false;
        }
    }
    return true;
}

double profile_intercept_adjustment_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const Eigen::VectorXd>& linear,
    double tau) {
    Eigen::VectorXd numerator =
        Eigen::VectorXd::Constant(linear.size(), -std::numeric_limits<double>::infinity());
    Eigen::VectorXd denominator(linear.size());
    for (Eigen::Index row = 0; row < linear.size(); ++row) {
        if (endog(row) > 0.0) {
            numerator(row) = std::log(endog(row)) + (1.0 - tau) * linear(row);
        }
        denominator(row) = (2.0 - tau) * linear(row);
    }
    const double log_num = logsumexp_all_core(numerator);
    const double log_den = logsumexp_all_core(denominator);
    if (std::isfinite(log_num) && std::isfinite(log_den)) {
        return log_num - log_den;
    }
    return 0.0;
}

Eigen::VectorXd group_logsumexp_core(
    const Eigen::Ref<const Eigen::VectorXd>& values,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups) {
    Eigen::VectorXd max_by_group =
        Eigen::VectorXd::Constant(n_groups, -std::numeric_limits<double>::infinity());
    std::vector<bool> has_nan(static_cast<std::size_t>(n_groups), false);
    for (Eigen::Index row = 0; row < values.size(); ++row) {
        const auto group = static_cast<Eigen::Index>(groups[static_cast<std::size_t>(row)]);
        const double value = values(row);
        if (std::isnan(value)) {
            has_nan[static_cast<std::size_t>(group)] = true;
            continue;
        }
        if (value > max_by_group(group)) {
            max_by_group(group) = value;
        }
    }
    Eigen::VectorXd out =
        Eigen::VectorXd::Constant(n_groups, -std::numeric_limits<double>::infinity());
    Eigen::VectorXd sums = Eigen::VectorXd::Zero(n_groups);
    for (Eigen::Index row = 0; row < values.size(); ++row) {
        const auto group = static_cast<Eigen::Index>(groups[static_cast<std::size_t>(row)]);
        const double max_value = max_by_group(group);
        const double value = values(row);
        if (has_nan[static_cast<std::size_t>(group)] || !std::isfinite(max_value)) {
            continue;
        }
        if (std::isfinite(value)) {
            sums(group) += std::exp(value - max_value);
        }
    }
    for (Eigen::Index group = 0; group < n_groups; ++group) {
        if (has_nan[static_cast<std::size_t>(group)]) {
            continue;
        }
        if (std::isinf(max_by_group(group)) && max_by_group(group) > 0.0) {
            out(group) = std::numeric_limits<double>::infinity();
        } else if (std::isfinite(max_by_group(group)) && sums(group) > 0.0) {
            out(group) = max_by_group(group) + std::log(sums(group));
        }
    }
    return out;
}

Eigen::VectorXd profile_one_effect_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const Eigen::VectorXd>& partial_linear,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups,
    double tau) {
    Eigen::VectorXd numerator =
        Eigen::VectorXd::Constant(partial_linear.size(), -std::numeric_limits<double>::infinity());
    Eigen::VectorXd denominator(partial_linear.size());
    for (Eigen::Index row = 0; row < partial_linear.size(); ++row) {
        if (endog(row) > 0.0) {
            numerator(row) =
                std::log(endog(row)) + (1.0 - tau) * partial_linear(row);
        }
        denominator(row) = (2.0 - tau) * partial_linear(row);
    }
    const Eigen::VectorXd log_num = group_logsumexp_core(numerator, groups, n_groups);
    const Eigen::VectorXd log_den = group_logsumexp_core(denominator, groups, n_groups);
    Eigen::VectorXd effect = Eigen::VectorXd::Zero(n_groups);
    for (Eigen::Index group = 0; group < effect.size(); ++group) {
        if (std::isfinite(log_num(group)) && std::isfinite(log_den(group))) {
            effect(group) = log_num(group) - log_den(group);
        }
    }
    return effect;
}

Eigen::VectorXd profile_linear_profiled_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const RowMatrixXd>& exog,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups,
    double tau) {
    Eigen::VectorXd base = exog * beta;
    if (is_singleton_intercept_group(groups, n_groups)) {
        const double adjustment = profile_intercept_adjustment_core(endog, base, tau);
        return Eigen::VectorXd(base.array() + adjustment);
    }
    const Eigen::VectorXd effect =
        profile_one_effect_core(endog, base, groups, n_groups, tau);
    Eigen::VectorXd linear(base.size());
    for (Eigen::Index row = 0; row < base.size(); ++row) {
        linear(row) = base(row) + effect(groups[static_cast<std::size_t>(row)]);
    }
    return linear;
}

Eigen::MatrixXd demean_singleton_intercept_core(
    const Eigen::Ref<const Eigen::MatrixXd>& values) {
    if (values.size() == 0) {
        return Eigen::MatrixXd(values);
    }
    Eigen::RowVectorXd mean = values.colwise().mean();
    return values.rowwise() - mean;
}

Eigen::MatrixXd demean_profiled_core(
    const Eigen::Ref<const Eigen::MatrixXd>& values,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups) {
    if (is_singleton_intercept_group(groups, n_groups)) {
        return demean_singleton_intercept_core(values);
    }
    return demean_oneway_matrix(values, groups, n_groups, std::nullopt);
}

struct FullrecoverInvariantCore {
    Eigen::MatrixXd x_demeaned;
    Eigen::MatrixXd projection;
    Eigen::MatrixXd manyway_q_basis;
};

FullrecoverInvariantCore fullrecover_invariant_core(
    const Eigen::Ref<const RowMatrixXd>& exog,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups) {
    FullrecoverInvariantCore state;
    state.x_demeaned = demean_profiled_core(exog, groups, n_groups);
    if (state.x_demeaned.cols() == 0) {
        state.projection = Eigen::MatrixXd(0, state.x_demeaned.rows());
        return state;
    }
    const Eigen::MatrixXd& x = state.x_demeaned;
    const Eigen::MatrixXd gram = x.transpose() * x;
    bool range_loss = !gram.allFinite();
    for (Eigen::Index j = 0; !range_loss && j < x.cols(); ++j) {
        range_loss = gram(j, j) < std::numeric_limits<double>::min() &&
                     (x.col(j).array() != 0.0).any();
    }
    if (range_loss) {
        const double largest = x.cwiseAbs().maxCoeff();
        if (largest > 0.0 && std::isfinite(largest) && x.allFinite()) {
            int shift;
            std::frexp(largest, &shift);
            const auto scale = [&shift](double v) { return std::scalbn(v, -shift); };
            const auto reversible = [&x](int k) {
                return (x.unaryExpr([k](double v) {
                    return std::scalbn(std::scalbn(v, -k), k);
                }).array() == x.array()).all();
            };
            // Keep the smallest entries exact if a full common shift loses bits.
            if (shift > 0 && !reversible(shift)) {
                int low = 0, high = shift;
                while (high - low > 1) {
                    const int mid = (low + high) / 2;
                    if (reversible(mid)) low = mid;
                    else high = mid;
                }
                shift = low;
            }
            const Eigen::MatrixXd z = x.unaryExpr(scale);
            const Eigen::MatrixXd scaled_gram = z.transpose() * z;
            if (shift && scaled_gram.allFinite()) {
                const Eigen::MatrixXd scaled = pyiols::kernel::min_norm_lstsq(
                    scaled_gram, z.transpose());
                state.projection = scaled.unaryExpr(scale);
                if (state.projection.allFinite()) {
                    return state;
                }
            }
        }
    }
    state.projection = pyiols::kernel::min_norm_lstsq(gram, x.transpose());
    return state;
}

Eigen::VectorXd initial_linear_profiled_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const RowMatrixXd>& exog,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups,
    double tau) {
    return profile_linear_profiled_core(endog, exog, beta, groups, n_groups, tau);
}

bool phi_step_profiled_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const RowMatrixXd>& exog,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups,
    double tau,
    const Eigen::Ref<const Eigen::MatrixXd>& projection,
    int phase,
    double param,
    Eigen::VectorXd& beta_new) {
    const Eigen::VectorXd linear_current =
        profile_linear_profiled_core(endog, exog, beta, groups, n_groups, tau);
    if (!all_finite_vec(linear_current)) {
        beta_new = beta;
        return false;
    }
    const double phase1_log_denom = phase == 1 ? std::log1p(param) : 0.0;
    const double phase1_inv_denom =
        phase == 1 ? 1.0 / (1.0 + param) : std::numeric_limits<double>::quiet_NaN();
    const bool tau_is_one = std::abs(tau - 1.0) <= 1e-12;
    const bool tau_is_two = std::abs(tau - 2.0) <= 1e-12;
    Eigen::VectorXd pseudo(linear_current.size());
    for (Eigen::Index row = 0; row < linear_current.size(); ++row) {
        const double linear = linear_current(row);
        if (phase == 0) {
            const double mu = exp_guard_scalar(linear);
            const double u_ratio = endog(row) / mu;
            const double weight = exp_guard_scalar((2.0 - tau) * linear);
            const double delta = param;
            const double delta_tilde = delta / std::max(weight, 1e-300);
            pseudo(row) =
                linear + std::log1p((u_ratio - 1.0) / (1.0 + delta_tilde));
        } else {
            double positive_component = 0.0;
            if (endog(row) > 0.0) {
                if (tau_is_one) {
                    positive_component = endog(row) * phase1_inv_denom;
                } else {
                    const double positive_log =
                        std::log(endog(row)) + (1.0 - tau) * linear -
                        phase1_log_denom;
                    positive_component = exp_guard_scalar(positive_log);
                }
            }
            const double negative_component =
                tau_is_two ? phase1_inv_denom
                           : exp_guard_scalar((2.0 - tau) * linear -
                                              phase1_log_denom);
            pseudo(row) = linear + positive_component - negative_component;
        }
    }
    if (!all_finite_vec(pseudo)) {
        beta_new = beta;
        return false;
    }
    const Eigen::MatrixXd pseudo_mat = pseudo;
    const Eigen::VectorXd demeaned =
        demean_profiled_core(pseudo_mat, groups, n_groups).col(0);
    beta_new = projection * demeaned;
    if (!all_finite_vec(beta_new)) {
        beta_new = beta;
        return false;
    }
    return true;
}

double profiled_score_row_core(double y, double linear, double tau) {
    const double log_weight = (1.0 - tau) * linear;
    const double log_base = (2.0 - tau) * linear;
    const double base = std::exp(log_base);
    if (y == 0.0) {
        return -base;
    }
    const double weight = std::exp(log_weight);
    const double score = y * weight - base;
    if (weight >= std::numeric_limits<double>::min() && std::isfinite(score)) {
        return score;
    }
    // Factor the difference by its larger term if the direct product lost range.
    const double log_y = std::log(y);
    const double d = log_y - linear;
    if (d == 0.0) {
        return 0.0;
    }
    const double larger = d > 0.0 ? log_y + log_weight : log_base;
    return std::copysign(std::exp(larger + std::log(-std::expm1(-std::abs(d)))), d);
}

double profiled_slope_score_residual_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const RowMatrixXd>& exog,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups,
    const Eigen::Ref<const Eigen::MatrixXd>& x_demeaned,
    double tau) {
    const Eigen::VectorXd linear =
        profile_linear_profiled_core(endog, exog, beta, groups, n_groups, tau);
    Eigen::VectorXd score_obs(linear.size());
    for (Eigen::Index row = 0; row < linear.size(); ++row) {
        score_obs(row) = profiled_score_row_core(endog(row), linear(row), tau);
    }
    if (!all_finite_vec(score_obs)) {
        return std::numeric_limits<double>::infinity();
    }
    const Eigen::VectorXd score = x_demeaned.transpose() * score_obs;
    if (!all_finite_vec(score)) {
        return std::numeric_limits<double>::infinity();
    }
    if (score.size() == 0) {
        return 0.0;
    }
    return score.cwiseAbs().maxCoeff() / std::max<double>(endog.size(), 1.0);
}

double profiled_fe_score_residual_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const RowMatrixXd>& exog,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups,
    double tau) {
    const Eigen::VectorXd linear =
        profile_linear_profiled_core(endog, exog, beta, groups, n_groups, tau);
    Eigen::VectorXd score_obs(linear.size());
    for (Eigen::Index row = 0; row < linear.size(); ++row) {
        score_obs(row) = profiled_score_row_core(endog(row), linear(row), tau);
    }
    if (!all_finite_vec(score_obs)) {
        return std::numeric_limits<double>::infinity();
    }
    if (is_singleton_intercept_group(groups, n_groups)) {
        return std::abs(score_obs.sum()) / std::max<double>(endog.size(), 1.0);
    }
    Eigen::VectorXd sums = Eigen::VectorXd::Zero(n_groups);
    for (Eigen::Index row = 0; row < score_obs.size(); ++row) {
        sums(groups[static_cast<std::size_t>(row)]) += score_obs(row);
    }
    return sums.cwiseAbs().maxCoeff() / std::max<double>(endog.size(), 1.0);
}

std::vector<double> exp_linspace(double log_min, double log_max, int size) {
    std::vector<double> out;
    const int n = std::max(size, 0);
    if (n == 0) {
        return out;
    }
    if (n == 1) {
        out.push_back(std::exp(log_min));
        return out;
    }
    out.reserve(static_cast<std::size_t>(n));
    for (int idx = 0; idx < n; ++idx) {
        const double frac = static_cast<double>(idx) / static_cast<double>(n - 1);
        out.push_back(std::exp(log_min + frac * (log_max - log_min)));
    }
    return out;
}

std::vector<double> unique_positive_finite(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    std::vector<double> out;
    for (const double value : values) {
        if (!std::isfinite(value) || value <= 0.0) {
            continue;
        }
        if (out.empty() || value != out.back()) {
            out.push_back(value);
        }
    }
    return out;
}

std::vector<double> focus_rho_path_core(
    const std::vector<double>& path,
    std::optional<double> rho_hint) {
    if (path.empty() || !rho_hint.has_value() || !std::isfinite(*rho_hint) ||
        *rho_hint <= 0.0) {
        return path;
    }
    const double low = path.front();
    const double high = path.back();
    const double hint = std::min(std::max(*rho_hint, low), high);
    const double mults[] = {0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0};
    std::vector<double> ordered;
    auto push_unique = [&ordered](double value) {
        for (const double seen : ordered) {
            if (seen == value) {
                return;
            }
        }
        ordered.push_back(value);
    };
    for (const double mult : mults) {
        const double value = std::min(std::max(hint * mult, low), high);
        if (std::isfinite(value) && value > 0.0) {
            push_unique(value);
        }
    }
    for (const double value : path) {
        push_unique(value);
    }
    return ordered;
}

std::vector<double> rho_path_core(
    const FullrecoverControlCore& control,
    std::optional<double> rho_hint) {
    if (!(std::isfinite(control.rho_log_min) &&
          std::isfinite(control.rho_log_max)) ||
        control.rho_log_max <= control.rho_log_min) {
        throw std::invalid_argument("rho_log_max must be greater than rho_log_min");
    }
    std::vector<double> path = unique_positive_finite(exp_linspace(
        control.rho_log_min,
        control.rho_log_max,
        std::max(control.rho_grid_size, 2)));
    return focus_rho_path_core(path, rho_hint);
}

std::vector<double> delta_grid_core(const FullrecoverControlCore& control) {
    // Internal package adapter; absent input preserves the paper path below.
    if (control.pilot_deltas.has_value()) {
        return *control.pilot_deltas;
    }
    const int grid_size = std::max(control.delta_grid_size, 0);
    if (grid_size == 0) {
        return {};
    }
    if (!(std::isfinite(control.delta_log_min) &&
          std::isfinite(control.delta_log_max)) ||
        control.delta_log_max <= control.delta_log_min) {
        throw std::invalid_argument("delta_log_max must be greater than delta_log_min");
    }
    return exp_linspace(control.delta_log_min, control.delta_log_max, grid_size);
}

std::vector<double> direct_step_rho_grid_core(
    double alpha0, double alpha_min, bool& clipped);

bool phase2_initialization_usable_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const RowMatrixXd>& exog,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups,
    double tau,
    const Eigen::Ref<const Eigen::MatrixXd>& projection,
    const FullrecoverControlCore& control,
    bool policy_grid) {
    if (!all_finite_vec(beta)) {
        return false;
    }
    std::vector<double> rho_path;
    if (policy_grid && control.rho_log_max <= control.rho_log_min) {
        // Both step policies clip to the upper rho bound in this case.
        const double alpha_min = 1.0 / (1.0 + std::exp(control.rho_log_max));
        bool clipped = false;
        rho_path = direct_step_rho_grid_core(alpha_min, alpha_min, clipped);
    } else {
        rho_path = rho_path_core(control, std::nullopt);
    }
    for (const double rho : rho_path) {
        Eigen::VectorXd beta_test(beta.size());
        if (phi_step_profiled_core(
                endog,
                exog,
                beta,
                groups,
                n_groups,
                tau,
                projection,
                1,
                rho,
                beta_test)) {
            return true;
        }
    }
    return false;
}

double max_abs_change_core(
    const Eigen::Ref<const Eigen::VectorXd>& left,
    const Eigen::Ref<const Eigen::VectorXd>& right) {
    if (left.size() == 0 && right.size() == 0) {
        return 0.0;
    }
    return (left - right).cwiseAbs().maxCoeff();
}

bool criterion_stagnated_core(
    const std::vector<double>& history,
    double eps,
    int window = 50,
    int min_iterations = 100,
    double min_relative_drop = 0.01) {
    if (history.size() < static_cast<std::size_t>(std::max(window * 2, min_iterations))) {
        return false;
    }
    const double current = history.back();
    const double previous = history[history.size() - static_cast<std::size_t>(window)];
    if (!(std::isfinite(current) && std::isfinite(previous))) {
        return false;
    }
    if (current <= eps * 10.0) {
        return false;
    }
    const double relative_drop = (previous - current) / std::max(std::abs(previous), eps);
    return relative_drop < min_relative_drop;
}

bool record_criterion_progress_core(
    std::vector<double>& history,
    double criterion,
    int phase,
    double tau,
    double eps,
    int max_phase2_evaluations,
    double& last_finite_criterion,
    double& best_finite_criterion,
    std::string& failure_reason,
    const std::vector<double>* stagnation_history = nullptr) {
    // runtime_v4: the direct route passes its plain-map residual history as
    // stagnation_history; every other caller keeps the criterion history.
    if (!std::isfinite(criterion)) {
        return false;
    }
    history.push_back(criterion);
    last_finite_criterion = criterion;
    best_finite_criterion =
        std::isfinite(best_finite_criterion)
            ? std::min(best_finite_criterion, criterion)
            : criterion;
    if (phase == 1 && tau > 2.0 && history.size() >= 20) {
        const double criterion_ref = history[history.size() - 20];
        if (std::isfinite(criterion_ref) && criterion_ref > eps &&
            criterion > std::max(criterion_ref * 100.0, 1e6)) {
            failure_reason = "criterion_explosion";
            return true;
        }
    }
    const int no_progress_min_iterations =
        std::min(std::max(max_phase2_evaluations / 2500, 20), 200);
    const int no_progress_window =
        std::min(std::max(no_progress_min_iterations / 2, 10), 50);
    if (phase == 1 &&
        criterion_stagnated_core(
            stagnation_history == nullptr ? history : *stagnation_history,
            eps,
            no_progress_window,
            no_progress_min_iterations,
            1e-4)) {
        failure_reason = "criterion_stagnation";
        return true;
    }
    return false;
}

int adapt_anderson_m_core(int aa_m, double kappa_val) {
    if (!std::isfinite(kappa_val)) {
        return aa_m;
    }
    if (kappa_val >= 0.95) {
        return std::max(aa_m, 20);
    }
    if (kappa_val >= 0.85) {
        return std::max(aa_m, 12);
    }
    if (kappa_val >= 0.70) {
        return std::max(aa_m, 8);
    }
    return aa_m;
}

// runtime_v3 direct-route kappa statistic (twin of the NumPy
// _windowed_contraction_rate): geometric-mean per-iteration step contraction
// over the last `window` recorded finite criteria,
// (c_last / c_{last-window})^(1/window). NaN when not computable.
constexpr int kDirectKappaWindow = 10;

double windowed_contraction_rate_core(
    const std::vector<double>& history,
    int window) {
    if (window < 1 ||
        history.size() < static_cast<std::size_t>(window) + 1) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double current = history.back();
    const double previous =
        history[history.size() - 1 - static_cast<std::size_t>(window)];
    if (!(std::isfinite(current) && std::isfinite(previous))) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (previous <= 0.0 || current < 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::pow(current / previous, 1.0 / static_cast<double>(window));
}

// runtime_v4 per-attempt progress-monitor fields (twin of the NumPy
// _direct_progress_trace_fields); the v3 criterion fields keep their meaning.
void set_direct_progress_trace_fields(
    Phase2AttemptTraceCore& record,
    const std::vector<double>& progress_history,
    int aa_m_start,
    int aa_m_end,
    int aa_accepted) {
    if (!progress_history.empty()) {
        record.final_progress_residual = progress_history.back();
        record.best_progress_residual =
            *std::min_element(progress_history.begin(), progress_history.end());
    }
    record.anderson_m_start = aa_m_start;
    record.anderson_m_end = aa_m_end;
    record.anderson_accepted = aa_accepted;
}

#if defined(PYIOLS_DIRECT_TEST_HOOKS)
// runtime_v4 TEST-ONLY fault injection, compiled only with
// -DPYIOLS_DIRECT_TEST_HOOKS=1 (never in the pinned campaign extension).
// While armed, the direct-route final verification map of the given phase
// (0 = Phase 1, 1 = Phase 2) is reported invalid. Single-threaded tests only.
int g_direct_test_fault_phase = -1;
int g_direct_test_fault_remaining = 0;  // -1 = every matching post-check
int g_direct_test_fault_fired = 0;

bool direct_test_hook_invalidate_post_check(int phase) {
    if (phase != g_direct_test_fault_phase || g_direct_test_fault_remaining == 0) {
        return false;
    }
    if (g_direct_test_fault_remaining > 0) {
        g_direct_test_fault_remaining -= 1;
    }
    g_direct_test_fault_fired += 1;
    return true;
}
#endif

constexpr const char* kDirectRestartStartPhase1 = "phase1_start";
constexpr const char* kDirectRestartStartBestScore = "best_score_iterate";
// runtime_v4: renamed because the windowed rate now reads the plain-map
// residual history instead of accepted moves.
constexpr const char* kDirectKappaMonitorPolicy = "windowed_plain_residual_ratio_10";
constexpr const char* kDirectProgressMonitorPolicy =
    "plain_map_residual_maxabs_star_at_current_iterate";
constexpr const char* kDirectPostCheckPolicy =
    "invalid_verification_map_clears_convergence";
constexpr const char* kDirectNofeRuntimeVersion = "direct_nofe_runtime_v4_2026_10_03";

FullrecoverIterationCore not_started_iteration_core(
    const FullrecoverControlCore& control) {
    FullrecoverIterationCore iteration;
    iteration.anderson_m_initial = control.anderson_m;
    iteration.anderson_m_final = control.anderson_m;
    iteration.failure_reason = "not_started";
    return iteration;
}

constexpr std::size_t kProfiledBetaPathRetained = 4;
constexpr std::size_t kProfiledKappaMinEntriesLegacy = 3;
constexpr std::size_t kProfiledKappaMinEntriesNumpyParity = 4;

void push_profiled_beta_path_core(
    std::vector<Eigen::VectorXd>& beta_path,
    const Eigen::VectorXd& beta,
    std::size_t& entries_seen,
    double& kappa_row_max,
    std::size_t min_entries = kProfiledKappaMinEntriesLegacy) {
    beta_path.push_back(beta);
    entries_seen += 1;
    if (entries_seen >= min_entries) {
        const std::size_t row = beta_path.size() - 1;
        const Eigen::VectorXd d_now =
            (beta_path[row] - beta_path[row - 1]).cwiseAbs();
        const Eigen::VectorXd d_prev =
            (beta_path[row - 1] - beta_path[row - 2]).cwiseAbs();
        double mean_ratio = 0.0;
        int count = 0;
        for (Eigen::Index idx = 0; idx < d_now.size(); ++idx) {
            const double ratio = d_now(idx) / d_prev(idx);
            if (!std::isnan(ratio)) {
                mean_ratio += ratio;
                count += 1;
            }
        }
        if (count > 0) {
            kappa_row_max = std::max(kappa_row_max, mean_ratio / count);
        }
    }
    if (beta_path.size() > kProfiledBetaPathRetained) {
        beta_path.erase(beta_path.begin());
    }
}

// rho_grid_hook_v1 (research diagnostic, twin of the NumPy hook): opt-in
// Phase-2 rho list for the direct no-FE route. A null hook or the "legacy"
// policy keeps rho_path_core untouched; no other control flow changes.
constexpr const char* kDirectRhoGridHookVersion = "rho_grid_hook_v1_2026_10_05";
constexpr const char* kDirectRhoGridLegacy = "legacy";
constexpr const char* kDirectRhoGridStep = "step_grid";
constexpr const char* kDirectRhoGridCurvatureStep = "curvature_step_grid";

struct DirectRhoGridHookCore {
    std::string policy = kDirectRhoGridLegacy;
    // Outputs, filled once at the Phase-2 handoff.
    bool reached = false;
    int grid_count = 0;
    std::vector<double> values;
    double alpha0 = std::numeric_limits<double>::quiet_NaN();
    double alpha_min = std::numeric_limits<double>::quiet_NaN();
    bool clipped = false;
    bool has_curvature = false;
    double curvature_L0 = std::numeric_limits<double>::quiet_NaN();
    double curvature_lambda_min = std::numeric_limits<double>::quiet_NaN();
    std::int64_t curvature_wall_ns = 0;
};

void validate_direct_rho_grid_policy_core(const std::string& policy) {
    if (policy != kDirectRhoGridLegacy && policy != kDirectRhoGridStep &&
        policy != kDirectRhoGridCurvatureStep) {
        throw std::invalid_argument(
            "rho_grid_policy must be one of ('legacy', 'step_grid', "
            "'curvature_step_grid')");
    }
}

// Halving alpha grid in plain double arithmetic (libm exp only in callers).
std::vector<double> direct_step_rho_grid_core(
    double alpha0,
    double alpha_min,
    bool& clipped) {
    if (!(std::isfinite(alpha0) && alpha0 > 0.0)) {
        throw std::runtime_error("direct_rho_grid_invalid: alpha0");
    }
    if (!(std::isfinite(alpha_min) && alpha_min > 0.0)) {
        throw std::runtime_error("direct_rho_grid_invalid: alpha_min");
    }
    std::vector<double> alphas;
    clipped = alpha0 <= alpha_min;
    if (!clipped) {
        double a = alpha0;
        while (a > alpha_min) {
            alphas.push_back(a);
            a = a * 0.5;
        }
    }
    alphas.push_back(alpha_min);
    std::vector<double> rho;
    rho.reserve(alphas.size());
    for (const double a : alphas) {
        const double value = 1.0 / a - 1.0;
        if (!(std::isfinite(value) && value > 0.0)) {
            throw std::runtime_error("direct_rho_grid_invalid: rho");
        }
        if (!rho.empty() && !(value > rho.back())) {
            throw std::runtime_error(
                "direct_rho_grid_invalid: not strictly increasing");
        }
        rho.push_back(value);
    }
    return rho;
}

// Anchor the weighted mean at a heaviest row to retain small displacements.
Eigen::MatrixXd anchored_curvature_core(
    Eigen::MatrixXd centered, const Eigen::VectorXd& h,
    const Eigen::VectorXd& mean_weights, Eigen::Index peak) {
    const Eigen::RowVectorXd anchor = centered.row(peak);
    centered.rowwise() -= anchor;
    const Eigen::RowVectorXd shift = mean_weights.transpose() * centered;
    centered.rowwise() -= shift;
    centered.array().colwise() *= h.array().sqrt();
    return centered.transpose() * centered;
}

// Form the same generalized spectrum in orthonormal slope coordinates when
// an unscaled intermediate overflows. Only the weighted mean uses relative
// weights; small independent directions retain their original signed curvature.
Eigen::MatrixXd equilibrated_curvature_core(
    const Eigen::MatrixXd& z, const Eigen::VectorXd& h, Eigen::Index anchor) {
    Eigen::MatrixXd u = z;
    for (Eigen::Index j = 0; j < u.cols(); ++j) {
        const double largest = u.col(j).cwiseAbs().maxCoeff();
        if (largest != 0.0) {
            u.col(j) /= std::scalbn(1.0, std::ilogb(largest));
        }
    }
    const Eigen::LLT<Eigen::MatrixXd> llt(u.transpose() * u);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error("direct_rho_curvature_failed: gram_cholesky_failed");
    }
    // A positive rounded pivot is not proof of independence. Use the same
    // exact dyadic rank certificate as the NumPy fallback on the original rows.
    if (z.cols() > 1) {
        py::gil_scoped_acquire acquire;
        const py::object full_rank =
            py::module_::import("powermean.data.design").attr("_has_full_column_rank");
        if (!full_rank(matrix_to_array(z)).cast<bool>()) {
            throw std::runtime_error("direct_rho_curvature_failed: gram_cholesky_failed");
        }
    }
    Eigen::MatrixXd q = llt.matrixL().solve(u.transpose()).transpose();
    const Eigen::VectorXd relative = h / h.cwiseAbs().maxCoeff();
    const double total = relative.sum();
    if (!(total > 0.0)) {
        throw std::runtime_error("direct_rho_curvature_failed: nonpositive_curvature_sum");
    }
    if (anchor >= 0) {
        return anchored_curvature_core(std::move(q), h, relative / total, anchor);
    }
    const Eigen::RowVectorXd mean = (relative.transpose() * q) / total;
    q.rowwise() -= mean;
    q.array().colwise() *= h.cwiseAbs().cwiseSqrt().array();
    return q.transpose() * (q.array().colwise() * h.array().sign()).matrix();
}

Eigen::MatrixXd direct_curvature_matrix_core(
    const Eigen::MatrixXd& x_demeaned, const Eigen::VectorXd& h, double total) {
    const double n_double = static_cast<double>(x_demeaned.rows());
    Eigen::Index peak;
    const double largest = h.maxCoeff(&peak);
    const double smallest = h.minCoeff();
    // On each centered contrast the raw moment / profiled variance is at
    // most max(h)/min(h). Switch when its n*eps error bound can cost half
    // the significand. Keep the original signed Schur form for signed weights.
    const double relative_floor = n_double * std::sqrt(std::numeric_limits<double>::epsilon());
    const Eigen::Index anchor =
        smallest >= 0.0 && smallest < largest && smallest / largest < relative_floor
            ? peak
            : -1;
    if (!std::isfinite(total)) {
        return equilibrated_curvature_core(x_demeaned, h, anchor);
    }
    const Eigen::VectorXd g = x_demeaned.transpose() * h;
    Eigen::MatrixXd hess_raw;
    if (anchor >= 0 && (g.array() != 0.0).any()) {
        if (!g.allFinite() ||
            g.cwiseAbs().maxCoeff() > std::sqrt(std::numeric_limits<double>::max())) {
            return equilibrated_curvature_core(x_demeaned, h, anchor);
        }
        hess_raw = anchored_curvature_core(x_demeaned, h, h / total, anchor) / n_double;
    } else {
        const Eigen::MatrixXd weighted =
            (x_demeaned.array().colwise() * h.array()).matrix();
        hess_raw =
            (x_demeaned.transpose() * weighted - (g * g.transpose()) / total) / n_double;
    }
    const Eigen::MatrixXd hess = 0.5 * (hess_raw + hess_raw.transpose());
    const Eigen::MatrixXd gram_raw =
        (x_demeaned.transpose() * x_demeaned) / n_double;
    const Eigen::MatrixXd gram = 0.5 * (gram_raw + gram_raw.transpose());
    // Detect finite subnormal loss as well as overflow, including g*g
    // before division by the signed total in the Schur correction.
    constexpr double tiny = std::numeric_limits<double>::min();
    if (!hess.allFinite() || !gram.allFinite() ||
        (gram.diagonal().array() < tiny).any() ||
        (hess.diagonal().cwiseAbs().array() < tiny).any() ||
        ((g.array() != 0.0) && (g.cwiseAbs().array() < std::sqrt(tiny))).any()) {
        return equilibrated_curvature_core(x_demeaned, h, anchor);
    }
    const Eigen::LLT<Eigen::MatrixXd> llt(gram);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error(
            "direct_rho_curvature_failed: gram_cholesky_failed");
    }
    const Eigen::MatrixXd lower = llt.matrixL();
    const Eigen::MatrixXd left =
        lower.triangularView<Eigen::Lower>().solve(hess);
    const Eigen::MatrixXd cmat_raw =
        lower.triangularView<Eigen::Lower>().solve(left.transpose());
    const Eigen::MatrixXd cmat = 0.5 * (cmat_raw + cmat_raw.transpose());
    return cmat.allFinite() ? cmat : equilibrated_curvature_core(x_demeaned, h, anchor);
}

// Largest generalized eigenvalue of the observed intercept-profiled negative
// Hessian H against Q = Z'Z/n at beta (Z = centered slope design). Diagnostic
// only: the projection and the update direction are not touched.
void direct_profiled_curvature_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const RowMatrixXd>& exog,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups,
    double tau,
    const Eigen::MatrixXd& x_demeaned,
    DirectRhoGridHookCore& hook) {
    const auto start = std::chrono::steady_clock::now();
    const Eigen::VectorXd linear =
        initial_linear_profiled_core(endog, exog, beta, groups, n_groups, tau);
    if (!all_finite_vec(linear)) {
        throw std::runtime_error("direct_rho_curvature_failed: nonfinite_index");
    }
    const Eigen::Index n = x_demeaned.rows();
    const double c1 = tau - 1.0;
    const double c2 = 2.0 - tau;
    Eigen::VectorXd h = Eigen::VectorXd::Zero(n);
    for (Eigen::Index row = 0; row < n; ++row) {
        double value = 0.0;
        if (c1 != 0.0 && endog(row) > 0.0) {
            value += c1 * exp_guard_scalar(
                              std::log(endog(row)) + (1.0 - tau) * linear(row));
        }
        if (c2 != 0.0) {
            value += c2 * exp_guard_scalar((2.0 - tau) * linear(row));
        }
        h(row) = value;
    }
    if (!all_finite_vec(h)) {
        throw std::runtime_error(
            "direct_rho_curvature_failed: nonfinite_row_curvature");
    }
    const double total = h.sum();
    if (std::isfinite(total) && total <= 0.0) {
        throw std::runtime_error(
            "direct_rho_curvature_failed: nonpositive_curvature_sum");
    }
    const Eigen::MatrixXd cmat = direct_curvature_matrix_core(x_demeaned, h, total);
    if (!cmat.allFinite()) {
        throw std::runtime_error("direct_rho_curvature_failed: nonfinite_hessian");
    }
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(
        cmat, Eigen::EigenvaluesOnly);
    if (spectrum.info() != Eigen::Success) {
        throw std::runtime_error("direct_rho_curvature_failed: eigensolver_failed");
    }
    if (!spectrum.eigenvalues().allFinite()) {
        throw std::runtime_error(
            "direct_rho_curvature_failed: nonfinite_eigenvalue");
    }
    hook.has_curvature = true;
    hook.curvature_L0 = spectrum.eigenvalues().maxCoeff();
    hook.curvature_lambda_min = spectrum.eigenvalues().minCoeff();
    hook.curvature_wall_ns = static_cast<std::int64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
}

// Non-legacy rho list at the Phase-2 handoff (curvature computed once here).
std::vector<double> direct_policy_rho_path_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const RowMatrixXd>& exog,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups,
    double tau,
    const FullrecoverControlCore& control,
    const FullrecoverInvariantCore& state,
    DirectRhoGridHookCore& hook) {
    const double alpha_max = 1.0 / (1.0 + std::exp(control.rho_log_min));
    const double alpha_min = 1.0 / (1.0 + std::exp(control.rho_log_max));
    double alpha0 = alpha_max;
    if (hook.policy == kDirectRhoGridCurvatureStep) {
        direct_profiled_curvature_core(
            endog, exog, beta, groups, n_groups, tau, state.x_demeaned, hook);
        alpha0 = 1.0 / std::max(1.0 + std::exp(control.rho_log_min), hook.curvature_L0);
    } else if (hook.policy != kDirectRhoGridStep) {
        validate_direct_rho_grid_policy_core(hook.policy);
        throw std::invalid_argument("rho_grid_policy legacy has no policy path");
    }
    bool clipped = false;
    std::vector<double> rho = direct_step_rho_grid_core(alpha0, alpha_min, clipped);
    hook.values = rho;
    hook.alpha0 = alpha0;
    hook.alpha_min = alpha_min;
    hook.clipped = clipped;
    return rho;
}

FullrecoverConcentratedCore concentrated_fullrecover_profiled_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const RowMatrixXd>& exog,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups,
    double tau,
    const FullrecoverControlCore& control,
    const FullrecoverInvariantCore& state,
    std::optional<Eigen::VectorXd> beta_init,
    std::optional<Eigen::VectorXd> linear_init,
    std::optional<double> rho_hint,
    bool direct_nofe_fixed_tau = false,
    DirectRhoGridHookCore* rho_hook = nullptr) {
    // rho_grid_hook_v1: opt-in policy is direct-route only.
    if (rho_hook != nullptr) {
        validate_direct_rho_grid_policy_core(rho_hook->policy);
        if (rho_hook->policy != kDirectRhoGridLegacy && !direct_nofe_fixed_tau) {
            throw std::invalid_argument(
                "rho_grid_policy other than 'legacy' requires direct_nofe_fixed_tau.");
        }
    }
    const bool policy_grid = direct_nofe_fixed_tau && rho_hook != nullptr &&
                             rho_hook->policy != kDirectRhoGridLegacy;
    if (direct_nofe_fixed_tau &&
        (beta_init.has_value() || !is_singleton_intercept_group(groups, n_groups))) {
        throw std::invalid_argument(
            "Direct fixed-tau Phase 1 + Phase 2 is authorized only for the "
            "noFE singleton-intercept route without beta_init");
    }
    if (!beta_init.has_value() && std::abs(tau - 2.0) > 1e-10 &&
        !direct_nofe_fixed_tau) {
        throw std::invalid_argument(
            "Non-anchor iOLS path steps require beta_init from tau=2");
    }
    const std::vector<double> delta_grid =
        beta_init.has_value() ? std::vector<double>{} : delta_grid_core(control);
    Eigen::VectorXd beta;
    Eigen::VectorXd linear;
    int n_outer = 0;
    if (beta_init.has_value()) {
        beta = *beta_init;
        linear = linear_init.has_value()
                     ? *linear_init
                     : initial_linear_profiled_core(
                           endog, exog, beta, groups, n_groups, tau);
        n_outer = 1;
    } else {
        Eigen::VectorXd log_y(endog.size());
        for (Eigen::Index row = 0; row < endog.size(); ++row) {
            log_y(row) = std::log(endog(row) + 1.0);
        }
        const Eigen::MatrixXd log_y_mat = log_y;
        const Eigen::VectorXd y_init =
            demean_profiled_core(log_y_mat, groups, n_groups).col(0);
        beta = state.projection * y_init;
        linear = initial_linear_profiled_core(endog, exog, beta, groups, n_groups, tau);
        n_outer = static_cast<int>(delta_grid.size()) + 1;
    }

    const Eigen::VectorXd beta_initial = beta;
    const Eigen::VectorXd linear_initial = linear;
    Eigen::VectorXd phase1_accepted_beta = beta;
    Eigen::VectorXd phase1_accepted_linear = linear;
    int phase1_attempts = 0;
    int phase1_accepted = 0;
    int phase1_rejected = 0;
    // runtime_v2: total Phase 1 map evaluations (plain steps plus Anderson
    // trial maps over every delta attempt); twin of the NumPy counter.
    int phase1_evaluations = 0;
    // runtime_v2 direct route: NumPy-parity kappa window and Anderson history
    // that grows with the adapted memory. Legacy callers keep their semantics.
    const std::size_t kappa_min_entries = direct_nofe_fixed_tau
                                              ? kProfiledKappaMinEntriesNumpyParity
                                              : kProfiledKappaMinEntriesLegacy;
    FullrecoverIterationCore final_diag = not_started_iteration_core(control);
    // runtime_v3 direct-route state (declared at function scope for C++
    // scoping; read and written only inside direct_nofe_fixed_tau blocks).
    bool v3_restart_from_best = false;
    double v3_best_score_norm = std::numeric_limits<double>::infinity();
    double v3_best_score_rho = std::numeric_limits<double>::quiet_NaN();
    bool v3_has_best_score_beta = false;
    Eigen::VectorXd v3_best_score_beta;
    int v3_phase2_warm_restarts = 0;
    std::vector<Phase2AttemptTraceCore> v3_phase2_attempt_trace;
    double v3_kappa_coordinate_row_max = std::numeric_limits<double>::quiet_NaN();
    double v3_kappa_monitor_last = std::numeric_limits<double>::quiet_NaN();
    double v3_kappa_monitor_max = std::numeric_limits<double>::quiet_NaN();
    int v3_kappa_monitor_checks = 0;
    int v3_attempt_evaluations_start = 0;
    std::string v3_attempt_start_policy = kDirectRestartStartPhase1;
    // runtime_v4: final verification maps that came back invalid, by
    // internal phase (0 = Phase 1 delta grid, 1 = Phase 2 rho grid).
    int v4_post_convergence_invalid_map_counts[2] = {0, 0};
    if (direct_nofe_fixed_tau) {
        v3_restart_from_best =
            control.direct_restart_start == kDirectRestartStartBestScore;
    }

    for (int outer_index = 0; outer_index < n_outer; ++outer_index) {
        if (!all_finite_vec(beta)) {
            beta = beta_initial;
            linear = linear_initial;
        }
        Eigen::VectorXd beta_outer_initial = beta;
        Eigen::VectorXd linear_outer_initial = linear;
        int phase = 0;
        double param = 0.0;
        double eps_current = control.eps;
        int stopcrit_current = control.stopcrit;
        std::vector<double> rho_path;
        if (outer_index < static_cast<int>(delta_grid.size())) {
            phase = 0;
            param = delta_grid[static_cast<std::size_t>(outer_index)];
            eps_current = std::max(control.eps * 10.0, 1e-5);
            stopcrit_current = std::min(control.stopcrit, control.max_phase1_iterations);
        } else {
            phase = 1;
            if (!beta_init.has_value()) {
                beta = phase1_accepted_beta;
                linear = phase1_accepted_linear;
            }
            if (policy_grid) {
                // rho_grid_hook_v1: step grid; curvature L0 at the Phase-2 pilot.
                rho_path = direct_policy_rho_path_core(
                    endog, exog, beta, groups, n_groups, tau, control, state, *rho_hook);
            } else {
                rho_path = rho_path_core(control, rho_hint);
            }
            if (direct_nofe_fixed_tau && rho_hook != nullptr) {
                rho_hook->reached = true;
                rho_hook->grid_count = static_cast<int>(rho_path.size());
            }
            if (rho_path.empty()) {
                FullrecoverConcentratedCore out;
                out.beta = beta;
                out.linear = linear;
                out.has_linear = all_finite_vec(linear);
                out.iteration = not_started_iteration_core(control);
                out.iteration.phase = "P2";
                out.iteration.failure_reason = "empty_finite_rho_path";
                out.iteration.phase1_attempts = phase1_attempts;
                out.iteration.phase1_accepted = phase1_accepted;
                out.iteration.phase1_rejected = phase1_rejected;
                out.iteration.phase1_evaluations = phase1_evaluations;
                return out;
            }
            param = rho_path.front();
            beta_outer_initial = beta;
            linear_outer_initial = linear;
        }

        int restart_index = 0;
        bool verified = false;
        int phase2_evaluations = 0;
        bool phase2_budget_exhausted = false;
        int iteration_count = 0;
        bool has_final_criterion = false;
        double final_criterion = 0.0;
        int aa_m_final = control.anderson_m;
        std::string failure_reason;
        double last_finite_criterion = std::numeric_limits<double>::quiet_NaN();
        double best_finite_criterion = std::numeric_limits<double>::quiet_NaN();
        double post_convergence_residual =
            std::numeric_limits<double>::quiet_NaN();
        double post_convergence_kappa_est =
            std::numeric_limits<double>::quiet_NaN();
        double post_convergence_score_residual =
            std::numeric_limits<double>::quiet_NaN();
        double post_convergence_fe_score_residual =
            std::numeric_limits<double>::quiet_NaN();
        double score_tol = std::numeric_limits<double>::quiet_NaN();
        std::string score_validation_status = "not_checked";
        std::vector<std::string> score_warning_codes;
        std::vector<std::string> score_warning_messages;
        bool score_exceeds_tolerance = false;
        bool fe_score_exceeds_tolerance = false;

        while (!verified) {
            int k = 1;
            double criterion = 100.0;
            std::vector<double> criterion_history;
            int kappa_bad_count = 0;
            std::vector<Eigen::VectorXd> beta_path;
            EigenVectorPairRingCore aa_hist;
            beta_path.reserve(kProfiledBetaPathRetained + 1);
            beta_path.push_back(beta);
            std::size_t beta_path_entries = 1;
            double kappa_row_max = -std::numeric_limits<double>::infinity();
            int aa_m = control.anderson_m;
            if (direct_nofe_fixed_tau && phase == 1) {
                v3_attempt_evaluations_start = phase2_evaluations;
            }
            criterion_history.reserve(
                static_cast<std::size_t>(std::max(stopcrit_current, 0)));
            if (aa_m > 0) {
                aa_hist.reset(static_cast<std::size_t>(aa_m + 1));
            }
            // runtime_v4 direct-route Phase 2 progress monitor (twin of the
            // NumPy progress_history): max-abs plain-map residual at the
            // current iterate, recorded at the same iterations as the
            // criterion. It drives stagnation, kappa and the Anderson memory
            // adaptation; the eps stop still uses `criterion`.
            std::vector<double> progress_history;
            double plain_residual_maxabs = std::numeric_limits<double>::quiet_NaN();
            int attempt_aa_m_start = aa_m;
            int attempt_aa_accepted = 0;

            while (criterion > eps_current && k < stopcrit_current &&
                   !std::isnan(criterion)) {
                const Eigen::VectorXd beta0 = beta;
                if (phase == 1) {
                    if (phase2_evaluations >= control.max_phase2_evaluations) {
                        criterion = std::numeric_limits<double>::quiet_NaN();
                        failure_reason = "phase2_evaluation_budget_exhausted";
                        phase2_budget_exhausted = true;
                        break;
                    }
                    phase2_evaluations += 1;
                } else {
                    phase1_evaluations += 1;
                }
                Eigen::VectorXd beta_phi(beta.size());
                const bool valid = phi_step_profiled_core(
                    endog,
                    exog,
                    beta,
                    groups,
                    n_groups,
                    tau,
                    state.projection,
                    phase,
                    param,
                    beta_phi);
                if (!valid) {
                    criterion = std::numeric_limits<double>::quiet_NaN();
                    failure_reason = "invalid_phi_step";
                    break;
                }
                if (direct_nofe_fixed_tau && phase == 1) {
                    // runtime_v3: rho-free preconditioned score norm of the
                    // current iterate, (1+rho)*||phi(beta)-beta||_2 =
                    // ||P score_obs(beta)||_2 (twin of the NumPy tracker).
                    const Eigen::VectorXd v3_plain_step = beta_phi - beta;
                    const double v3_step_score_norm =
                        pyiols::kernel::stable_l2_norm(v3_plain_step) *
                        (1.0 + param);
                    if (std::isfinite(v3_step_score_norm) &&
                        v3_step_score_norm < v3_best_score_norm) {
                        v3_best_score_norm = v3_step_score_norm;
                        v3_best_score_rho = param;
                        v3_best_score_beta = beta;
                        v3_has_best_score_beta = true;
                    }
                    // runtime_v4: same operands as the plain-step criterion,
                    // so it equals `criterion` whenever no Anderson step is
                    // accepted; no extra map evaluation.
                    plain_residual_maxabs = max_abs_change_core(beta_phi, beta);
                }

                bool accepted_aa_step = false;
                if (aa_m > 0 && k > aa_m + 1) {
                    const Eigen::VectorXd f_k = beta_phi - beta;
                    aa_hist.push(beta, f_k);
                    const int m_cur = static_cast<int>(aa_hist.size()) - 1;
                    if (m_cur >= 1) {
                        Eigen::MatrixXd f_mat(beta.size(), m_cur);
                        for (int j = 1; j <= m_cur; ++j) {
                            f_mat.col(j - 1) =
                                aa_hist.f_back() -
                                aa_hist.f_at(aa_hist.size() - static_cast<std::size_t>(j + 1));
                        }
                        Eigen::VectorXd gamma =
                            pyiols::kernel::min_norm_lstsq_anderson(
                                f_mat,
                                aa_hist.f_back())
                                .col(0);
                        Eigen::VectorXd beta_aa = beta_phi;
                        for (int j = 0; j < m_cur; ++j) {
                            const std::size_t idx =
                                aa_hist.size() - static_cast<std::size_t>(j + 2);
                            const Eigen::VectorXd phi_prev =
                                aa_hist.x_at(idx) + aa_hist.f_at(idx);
                            beta_aa += gamma(j) * (phi_prev - beta_phi);
                        }
                        if (all_finite_vec(beta_aa)) {
                            if (phase == 1) {
                                if (phase2_evaluations >=
                                    control.max_phase2_evaluations) {
                                    criterion =
                                        std::numeric_limits<double>::quiet_NaN();
                                    failure_reason =
                                        "phase2_evaluation_budget_exhausted";
                                    phase2_budget_exhausted = true;
                                    break;
                                }
                                phase2_evaluations += 1;
                            } else {
                                phase1_evaluations += 1;
                            }
                            Eigen::VectorXd beta_aa_phi(beta.size());
                            const bool aa_valid = phi_step_profiled_core(
                                endog,
                                exog,
                                beta_aa,
                                groups,
                                n_groups,
                                tau,
                                state.projection,
                                phase,
                                param,
                                beta_aa_phi);
                            if (aa_valid) {
                                const Eigen::VectorXd plain_residual = beta_phi - beta;
                                const Eigen::VectorXd aa_residual =
                                    beta_aa_phi - beta_aa;
                                const double res_plain =
                                    pyiols::kernel::stable_l2_norm(plain_residual);
                                const double res_aa =
                                    pyiols::kernel::stable_l2_norm(aa_residual);
                                if (res_aa < res_plain * 1.1) {
                                    beta = beta_aa;
                                    push_profiled_beta_path_core(
                                        beta_path,
                                        beta,
                                        beta_path_entries,
                                        kappa_row_max,
                                        kappa_min_entries);
                                    if (k >= 2) {
                                        criterion = max_abs_change_core(beta, beta0);
                                    }
                                    accepted_aa_step = true;
                                    attempt_aa_accepted += 1;
                                }
                            }
                        }
                    }
                } else if (aa_m > 0) {
                    aa_hist.push(beta, beta_phi - beta);
                }

                if (!accepted_aa_step) {
                    beta = beta_phi;
                    push_profiled_beta_path_core(
                        beta_path,
                        beta,
                        beta_path_entries,
                        kappa_row_max,
                        kappa_min_entries);
                }
                if (k >= 2) {
                    if (!accepted_aa_step) {
                        criterion = max_abs_change_core(beta, beta0);
                    }
                    const bool direct_progress = direct_nofe_fixed_tau && phase == 1;
                    if (direct_progress && std::isfinite(plain_residual_maxabs)) {
                        progress_history.push_back(plain_residual_maxabs);
                    }
                    if (record_criterion_progress_core(
                            criterion_history,
                            criterion,
                            phase,
                            tau,
                            eps_current,
                            control.max_phase2_evaluations,
                            last_finite_criterion,
                            best_finite_criterion,
                            failure_reason,
                            direct_progress ? &progress_history : nullptr)) {
                        criterion = std::numeric_limits<double>::quiet_NaN();
                    }
                }

                if (phase == 1 && std::abs(tau - 2.0) > 1e-10 && k >= 30 &&
                    k % 10 == 0 && beta_path_entries >= kappa_min_entries) {
                    if (direct_nofe_fixed_tau) {
                        // runtime_v3: the lifetime coordinate-mean ratio
                        // maximum is kept as a diagnostic only (+inf rows
                        // included: they are the pollution it exists to show).
                        if (!std::isnan(kappa_row_max) &&
                            kappa_row_max > -std::numeric_limits<double>::infinity()) {
                            v3_kappa_coordinate_row_max =
                                std::isnan(v3_kappa_coordinate_row_max)
                                    ? kappa_row_max
                                    : std::max(v3_kappa_coordinate_row_max,
                                               kappa_row_max);
                        }
                    }
                    // runtime_v3 direct route: windowed geometric-mean
                    // contraction (twin of _windowed_contraction_rate);
                    // legacy routes keep the lifetime coordinate statistic.
                    // runtime_v4: taken over the plain-map residual history,
                    // not accepted moves.
                    const double kappa_val = direct_nofe_fixed_tau
                                                 ? windowed_contraction_rate_core(
                                                       progress_history,
                                                       kDirectKappaWindow)
                                                 : kappa_row_max;
                    if (direct_nofe_fixed_tau) {
                        v3_kappa_monitor_last = kappa_val;
                        v3_kappa_monitor_checks += 1;
                        if (std::isfinite(kappa_val)) {
                            v3_kappa_monitor_max =
                                std::isnan(v3_kappa_monitor_max)
                                    ? kappa_val
                                    : std::max(v3_kappa_monitor_max, kappa_val);
                        }
                    }
                    const int aa_m_new = adapt_anderson_m_core(aa_m, kappa_val);
                    if (aa_m_new > aa_m) {
                        aa_m = aa_m_new;
                        kappa_bad_count = 0;
                        if (direct_nofe_fixed_tau && aa_m > 0) {
                            // NumPy parity: the history now holds up to aa_m lags.
                            aa_hist.grow(static_cast<std::size_t>(aa_m + 1));
                        }
                    } else if (std::isfinite(kappa_val) && kappa_val >= 1.0) {
                        kappa_bad_count += 1;
                    } else {
                        kappa_bad_count = 0;
                    }
                    if (kappa_bad_count >= 3) {
                        criterion = std::numeric_limits<double>::quiet_NaN();
                        failure_reason = "noncontractive";
                    }
                }

                k += 1;
                iteration_count = k;

                if (std::isnan(criterion)) {
                    restart_index += 1;
                    if (phase == 1 && !phase2_budget_exhausted &&
                        restart_index < static_cast<int>(rho_path.size())) {
                        if (direct_nofe_fixed_tau) {
                            Phase2AttemptTraceCore v3_record;
                            // restart_index was already advanced: the attempt
                            // that just ended is the previous one.
                            v3_record.attempt = restart_index - 1;
                            v3_record.rho = param;
                            v3_record.iterations = k;
                            v3_record.evaluations =
                                phase2_evaluations - v3_attempt_evaluations_start;
                            v3_record.failure_reason = failure_reason;
                            v3_record.has_final_criterion =
                                std::isfinite(last_finite_criterion);
                            v3_record.final_criterion = last_finite_criterion;
                            v3_record.best_criterion =
                                criterion_history.empty()
                                    ? std::numeric_limits<double>::quiet_NaN()
                                    : *std::min_element(criterion_history.begin(),
                                                        criterion_history.end());
                            v3_record.kappa_last = v3_kappa_monitor_last;
                            v3_record.start_policy = v3_attempt_start_policy;
                            set_direct_progress_trace_fields(
                                v3_record,
                                progress_history,
                                attempt_aa_m_start,
                                aa_m,
                                attempt_aa_accepted);
                            v3_phase2_attempt_trace.push_back(v3_record);
                        }
                        param = rho_path[static_cast<std::size_t>(restart_index)];
                        criterion = 100.0;
                        beta = beta_outer_initial;
                        if (direct_nofe_fixed_tau) {
                            // runtime_v3 restart start policy.
                            v3_attempt_start_policy = kDirectRestartStartPhase1;
                            if (v3_restart_from_best && v3_has_best_score_beta) {
                                beta = v3_best_score_beta;
                                v3_phase2_warm_restarts += 1;
                                v3_attempt_start_policy = kDirectRestartStartBestScore;
                            }
                            v3_attempt_evaluations_start = phase2_evaluations;
                            v3_kappa_monitor_last = std::numeric_limits<double>::quiet_NaN();
                        }
                        k = 1;
                        criterion_history.clear();
                        kappa_bad_count = 0;
                        beta_path.clear();
                        beta_path.push_back(beta);
                        beta_path_entries = 1;
                        kappa_row_max = -std::numeric_limits<double>::infinity();
                        aa_hist.clear();
                        // runtime_v4: aa_m itself is still carried over (v3).
                        progress_history.clear();
                        plain_residual_maxabs = std::numeric_limits<double>::quiet_NaN();
                        attempt_aa_m_start = aa_m;
                        attempt_aa_accepted = 0;
                    } else if (phase == 0) {
                        break;
                    }
                }
            }

            bool converged = !std::isnan(criterion) && criterion <= eps_current;
            has_final_criterion = !std::isnan(criterion);
            final_criterion = criterion;
            if (converged) {
                Eigen::VectorXd beta_check(beta.size());
                bool valid = phi_step_profiled_core(
                    endog,
                    exog,
                    beta,
                    groups,
                    n_groups,
                    tau,
                    state.projection,
                    phase,
                    param,
                    beta_check);
#if defined(PYIOLS_DIRECT_TEST_HOOKS)
                if (direct_nofe_fixed_tau && direct_test_hook_invalidate_post_check(phase)) {
                    valid = false;
                }
#endif
                if (valid) {
                    const double residual = max_abs_change_core(beta_check, beta);
                    post_convergence_residual = residual;
                    if (residual > eps_current * 10.0) {
                        converged = false;
                        failure_reason = "post_convergence_residual";
                    }
                } else if (direct_nofe_fixed_tau) {
                    // runtime_v4: an invalid final verification map clears
                    // native convergence (v3 kept it). No residual exists, so
                    // an earlier attempt's post-check values are not reported.
                    converged = false;
                    failure_reason = "post_convergence_invalid_map";
                    v4_post_convergence_invalid_map_counts[phase] += 1;
                    post_convergence_residual = std::numeric_limits<double>::quiet_NaN();
                    post_convergence_kappa_est = std::numeric_limits<double>::quiet_NaN();
                }
                if (converged && phase == 1) {
                    const double score_residual = profiled_slope_score_residual_core(
                        endog,
                        exog,
                        beta,
                        groups,
                        n_groups,
                        state.x_demeaned,
                        tau);
                    const double fe_score_residual = profiled_fe_score_residual_core(
                        endog,
                        exog,
                        beta,
                        groups,
                        n_groups,
                        tau);
                    score_tol = std::max(eps_current * 10.0, 1e-8);
                    post_convergence_score_residual = score_residual;
                    score_exceeds_tolerance =
                        !std::isfinite(score_residual) || score_residual > score_tol;
                    if (score_exceeds_tolerance) {
                        score_warning_codes.push_back(
                            "post_convergence_score_residual");
                        score_warning_messages.push_back(
                            "Slope score residual exceeds tolerance at this "
                            "tau-path state.");
                        score_validation_status = "diagnostic_warning";
                    } else if (score_validation_status == "not_checked") {
                        score_validation_status = "passed";
                    }
                    post_convergence_fe_score_residual = fe_score_residual;
                    fe_score_exceeds_tolerance =
                        !std::isfinite(fe_score_residual) ||
                        fe_score_residual > score_tol;
                    if (fe_score_exceeds_tolerance) {
                        score_warning_codes.push_back(
                            "post_convergence_fe_score_residual");
                        score_warning_messages.push_back(
                            "Recovered-FE score residual exceeds tolerance at "
                            "this tau-path state.");
                        score_validation_status = "diagnostic_warning";
                    } else if (score_validation_status == "not_checked") {
                        score_validation_status = "passed";
                    }
                }
                if (converged && beta_path.size() >= 4 &&
                    !(phase == 1 && std::abs(tau - 2.0) <= 1e-10)) {
                    const double d0 =
                        max_abs_change_core(beta_path[beta_path.size() - 1],
                                            beta_path[beta_path.size() - 2]);
                    const double d1 =
                        max_abs_change_core(beta_path[beta_path.size() - 2],
                                            beta_path[beta_path.size() - 3]);
                    const double d2 =
                        max_abs_change_core(beta_path[beta_path.size() - 3],
                                            beta_path[beta_path.size() - 4]);
                    if (d1 > 1e-15 && d2 > 1e-15) {
                        const double kappa_est = d0 / d1;
                        post_convergence_kappa_est = kappa_est;
                        if (kappa_est >= 1.0) {
                            converged = false;
                            failure_reason = "post_convergence_noncontractive";
                        }
                    }
                }
            }

            if (direct_nofe_fixed_tau && phase == 1) {
                Phase2AttemptTraceCore v3_record;
                // restart_index can equal rho_path.size() when the last
                // attempt was killed in-loop with no rho left.
                v3_record.attempt =
                    std::min(restart_index, static_cast<int>(rho_path.size()) - 1);
                v3_record.rho = param;
                // k (not iteration_count) so an attempt that ended on its
                // first map does not inherit the previous count.
                v3_record.iterations = k;
                v3_record.evaluations =
                    phase2_evaluations - v3_attempt_evaluations_start;
                // a finite criterion above eps here means the per-rho cap
                // (stopcrit) ended the attempt; failure_reason is per phase
                // and would be stale.
                v3_record.failure_reason =
                    converged ? std::string()
                              : ((!std::isnan(criterion) && criterion > eps_current)
                                     ? std::string("stopcrit_reached")
                                     : failure_reason);
                v3_record.has_final_criterion = has_final_criterion;
                v3_record.final_criterion = final_criterion;
                v3_record.best_criterion =
                    criterion_history.empty()
                        ? std::numeric_limits<double>::quiet_NaN()
                        : *std::min_element(criterion_history.begin(),
                                            criterion_history.end());
                v3_record.kappa_last = v3_kappa_monitor_last;
                v3_record.start_policy = v3_attempt_start_policy;
                set_direct_progress_trace_fields(
                    v3_record,
                    progress_history,
                    attempt_aa_m_start,
                    aa_m,
                    attempt_aa_accepted);
                v3_phase2_attempt_trace.push_back(v3_record);
            }
            if (!converged && phase == 1 && !phase2_budget_exhausted &&
                restart_index < static_cast<int>(rho_path.size()) - 1) {
                restart_index = std::min(
                    restart_index + std::max(restart_index, 5),
                    static_cast<int>(rho_path.size()) - 1);
                param = rho_path[static_cast<std::size_t>(restart_index)];
                beta = beta_outer_initial;
                if (direct_nofe_fixed_tau) {
                    // runtime_v3 restart start policy.
                    v3_attempt_start_policy = kDirectRestartStartPhase1;
                    if (v3_restart_from_best && v3_has_best_score_beta) {
                        beta = v3_best_score_beta;
                        v3_phase2_warm_restarts += 1;
                        v3_attempt_start_policy = kDirectRestartStartBestScore;
                    }
                    v3_kappa_monitor_last = std::numeric_limits<double>::quiet_NaN();
                }
            } else {
                verified = true;
            }
            aa_m_final = aa_m;

            if (phase == 0) {
                phase1_attempts += 1;
                const bool candidate_usable =
                    converged &&
                    phase2_initialization_usable_core(
                        endog,
                        exog,
                        beta,
                        groups,
                        n_groups,
                        tau,
                        state.projection,
                        control,
                        policy_grid);
                if (candidate_usable) {
                    phase1_accepted += 1;
                    phase1_accepted_beta = beta;
                    phase1_accepted_linear = initial_linear_profiled_core(
                        endog, exog, beta, groups, n_groups, tau);
                } else {
                    phase1_rejected += 1;
                    beta = phase1_accepted_beta;
                    if (converged && failure_reason.empty()) {
                        failure_reason = "phase1_candidate_unusable";
                    }
                }
                final_diag.converged = candidate_usable;
            } else {
                if (!converged && !direct_nofe_fixed_tau) {
                    // Legacy tau-path semantics: hand the anchor/previous
                    // state back to the caller. The runtime_v2 direct route
                    // keeps the final iterate of the last rho attempt so that
                    // the external exact-score audit can still accept it.
                    beta = beta_outer_initial;
                }
                if (direct_nofe_fixed_tau) {
                    final_diag.nonconverged_iterate_policy =
                        "last_attempt_final_iterate";
                    if (v3_restart_from_best) {
                        final_diag.nonconverged_iterate_policy =
                            kDirectRestartStartBestScore;
                        if (!converged && v3_has_best_score_beta) {
                            // runtime_v3: hand back the smallest
                            // preconditioned-score iterate of the whole
                            // Phase 2; the external audit decides.
                            beta = v3_best_score_beta;
                        }
                    }
                    final_diag.direct_restart_start = control.direct_restart_start;
                    final_diag.kappa_coordinate_row_max = v3_kappa_coordinate_row_max;
                    final_diag.kappa_monitor_last = v3_kappa_monitor_last;
                    final_diag.kappa_monitor_max = v3_kappa_monitor_max;
                    final_diag.kappa_monitor_checks = v3_kappa_monitor_checks;
                    final_diag.best_score_norm = v3_best_score_norm;
                    final_diag.best_score_rho = v3_best_score_rho;
                    final_diag.phase2_warm_restarts = v3_phase2_warm_restarts;
                    final_diag.phase2_attempts =
                        static_cast<int>(v3_phase2_attempt_trace.size());
                    final_diag.phase2_attempt_trace = v3_phase2_attempt_trace;
                    final_diag.post_convergence_invalid_map_count =
                        v4_post_convergence_invalid_map_counts[1];
                    final_diag.phase1_post_convergence_invalid_map_count =
                        v4_post_convergence_invalid_map_counts[0];
                }
                final_diag.converged = converged;
            }

            final_diag.phase = phase == 0 ? "P1" : "P2";
            final_diag.iterations = iteration_count;
            final_diag.restarts = restart_index;
            final_diag.has_final_rho = phase != 0;
            final_diag.final_rho = param;
            final_diag.rho_path.clear();
            if (phase != 0) {
                const int upto =
                    std::min(restart_index + 1, static_cast<int>(rho_path.size()));
                for (int idx = 0; idx < upto; ++idx) {
                    final_diag.rho_path.push_back(rho_path[static_cast<std::size_t>(idx)]);
                }
            }
            final_diag.has_final_criterion = has_final_criterion;
            final_diag.final_criterion = final_criterion;
            final_diag.anderson_m_initial = control.anderson_m;
            final_diag.anderson_m_final = aa_m_final;
            final_diag.failure_reason =
                final_diag.converged
                    ? ""
                    : (!failure_reason.empty() ? failure_reason : "not_converged");
            final_diag.phase1_attempts = phase1_attempts;
            final_diag.phase1_accepted = phase1_accepted;
            final_diag.phase1_rejected = phase1_rejected;
            final_diag.phase1_evaluations = phase1_evaluations;
            final_diag.phase2_evaluations = phase2_evaluations;
            final_diag.max_phase2_evaluations = control.max_phase2_evaluations;
            final_diag.last_finite_criterion = last_finite_criterion;
            final_diag.best_finite_criterion = best_finite_criterion;
            final_diag.fe_state_policy = "analytical_profile";
            final_diag.anderson_policy = "beta_anderson";
            final_diag.post_convergence_residual = post_convergence_residual;
            final_diag.post_convergence_kappa_est = post_convergence_kappa_est;
            final_diag.post_convergence_score_residual =
                post_convergence_score_residual;
            final_diag.post_convergence_fe_score_residual =
                post_convergence_fe_score_residual;
            final_diag.score_tolerance = score_tol;
            final_diag.score_validation_policy = "diagnostic";
            final_diag.score_validation_status = score_validation_status;
            final_diag.score_warning_codes = score_warning_codes;
            final_diag.score_warning_messages = score_warning_messages;
            final_diag.post_convergence_score_exceeds_tolerance =
                score_exceeds_tolerance;
            final_diag.post_convergence_fe_score_exceeds_tolerance =
                fe_score_exceeds_tolerance;
        }
        if (phase == 1) {
            break;
        }
    }

    FullrecoverConcentratedCore out;
    out.beta = beta;
    out.linear = initial_linear_profiled_core(endog, exog, beta, groups, n_groups, tau);
    out.has_linear = all_finite_vec(out.linear);
    out.iteration = final_diag;
    return out;
}

FullrecoverControlCore control_from_mapping(py::object control_obj) {
    FullrecoverControlCore control;
    if (control_obj.is_none()) {
        return control;
    }
    const py::dict mapping = py::cast<py::dict>(control_obj);
    auto get_int = [&mapping](const char* key, int current) {
        if (mapping.contains(py::str(key))) {
            return py::cast<int>(mapping[py::str(key)]);
        }
        return current;
    };
    auto get_double = [&mapping](const char* key, double current) {
        if (mapping.contains(py::str(key))) {
            return py::cast<double>(mapping[py::str(key)]);
        }
        return current;
    };
    control.anderson_m = get_int("anderson_m", control.anderson_m);
    control.eps = get_double("eps", control.eps);
    control.stopcrit = get_int("stopcrit", control.stopcrit);
    control.tau_path_steps = get_int("tau_path_steps", control.tau_path_steps);
    if (mapping.contains(py::str("min_dtau")) && !mapping[py::str("min_dtau")].is_none()) {
        control.has_min_dtau = true;
        control.min_dtau = py::cast<double>(mapping[py::str("min_dtau")]);
    }
    control.max_path_evaluations =
        get_int("max_path_evaluations", control.max_path_evaluations);
    control.max_phase1_iterations =
        get_int("max_phase1_iterations", control.max_phase1_iterations);
    control.max_phase2_evaluations =
        get_int("max_phase2_evaluations", control.max_phase2_evaluations);
    control.fe_demean_maxiter =
        get_int("fe_demean_maxiter", control.fe_demean_maxiter);
    control.fe_demean_tol = get_double("fe_demean_tol", control.fe_demean_tol);
    control.delta_log_min = get_double("delta_log_min", control.delta_log_min);
    control.delta_log_max = get_double("delta_log_max", control.delta_log_max);
    control.delta_grid_size = get_int("delta_grid_size", control.delta_grid_size);
    if (mapping.contains(py::str("_pilot_deltas"))) {
        const py::sequence values = py::cast<py::sequence>(mapping["_pilot_deltas"]);
        std::vector<double> deltas;
        for (const auto value : values) {
            const double delta = py::cast<double>(value);
            if (!std::isfinite(delta) || delta <= 0.0) {
                throw std::invalid_argument("_pilot_deltas must be a nonempty positive finite vector");
            }
            deltas.push_back(delta);
        }
        if (deltas.empty()) {
            throw std::invalid_argument("_pilot_deltas must be a nonempty positive finite vector");
        }
        control.pilot_deltas = std::move(deltas);
    }
    control.rho_log_min = get_double("rho_log_min", control.rho_log_min);
    control.rho_log_max = get_double("rho_log_max", control.rho_log_max);
    control.rho_grid_size = get_int("rho_grid_size", control.rho_grid_size);
    if (mapping.contains(py::str("direct_restart_start")) &&
        !mapping[py::str("direct_restart_start")].is_none()) {
        // runtime_v3: read only by the direct no-FE fixed-tau route.
        control.direct_restart_start =
            py::cast<std::string>(mapping[py::str("direct_restart_start")]);
    }
    return control;
}

// Direct fixed-tau noFE route, twin of the NumPy
// _solve_nofe_direct_fixed_tau. Messages and check order match the NumPy
// validators so both backends reject the same inputs the same way.
void validate_direct_nofe_inputs_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const RowMatrixXd>& exog_slopes,
    double tau) {
    if (exog_slopes.rows() != endog.size()) {
        throw std::invalid_argument(
            "endog and exog_slopes must have the same number of rows");
    }
    if (exog_slopes.cols() < 1) {
        throw std::invalid_argument(
            "exog_slopes must have at least one slope column; the intercept is "
            "profiled");
    }
    bool any_positive = false;
    for (Eigen::Index row = 0; row < endog.size(); ++row) {
        if (!std::isfinite(endog(row)) || endog(row) < 0.0) {
            throw std::invalid_argument("endog must be finite and nonnegative");
        }
        any_positive = any_positive || endog(row) > 0.0;
    }
    if (!any_positive) {
        throw std::invalid_argument("endog must contain at least one positive value");
    }
    for (Eigen::Index row = 0; row < exog_slopes.rows(); ++row) {
        for (Eigen::Index col = 0; col < exog_slopes.cols(); ++col) {
            if (!std::isfinite(exog_slopes(row, col))) {
                throw std::invalid_argument("exog_slopes must be finite");
            }
        }
    }
    for (Eigen::Index col = 0; col < exog_slopes.cols(); ++col) {
        if (exog_slopes.col(col).maxCoeff() == exog_slopes.col(col).minCoeff()) {
            throw std::invalid_argument(
                "exog_slopes column " + std::to_string(col) +
                " is constant; pass slopes only, the intercept is profiled");
        }
    }
    if (!std::isfinite(tau) || tau < 0.0 || tau > 3.0) {
        throw std::invalid_argument("tau must be finite and in [0, 3]");
    }
}

void validate_direct_nofe_control_core(const FullrecoverControlCore& control) {
    if (control.delta_grid_size < 1) {
        throw std::invalid_argument(
            "direct fixed-tau route requires delta_grid_size >= 1 so Phase 1 runs");
    }
    if (control.anderson_m < 0) {
        throw std::invalid_argument("anderson_m must be nonnegative");
    }
    if (!std::isfinite(control.eps) || control.eps <= 0.0) {
        throw std::invalid_argument("eps must be finite and positive");
    }
    if (control.stopcrit < 2 || control.max_phase1_iterations < 2) {
        throw std::invalid_argument(
            "stopcrit and max_phase1_iterations must be at least 2");
    }
    if (control.max_phase2_evaluations < 1) {
        throw std::invalid_argument("max_phase2_evaluations must be at least 1");
    }
    if (control.direct_restart_start != kDirectRestartStartPhase1 &&
        control.direct_restart_start != kDirectRestartStartBestScore) {
        throw std::invalid_argument(
            "direct_restart_start must be one of ('phase1_start', "
            "'best_score_iterate')");
    }
}

// Runs the existing profiled Phase 1 (delta grid) and Phase 2 (rho grid) once
// at the requested tau: no tau=2 anchor and no tau continuation. Validation,
// the fixed-design invariant state, both phases and the profiled index all run
// in C++ with the GIL released.
py::dict solve_nofe_direct_fixed_tau_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> endog,
    py::array_t<double, py::array::c_style | py::array::forcecast> exog_slopes,
    double tau,
    py::object control_obj,
    const std::string& rho_grid_policy) {
    // rho_grid_hook_v1: validate the opt-in policy before any work.
    validate_direct_rho_grid_policy_core(rho_grid_policy);
    if (control_obj.is_none()) {
        throw std::invalid_argument(
            "direct fixed-tau route requires an explicit control mapping");
    }
    const Eigen::VectorXd endog_vec = vector_from_array(endog, "endog");
    const RowMatrixXd exog_mat = matrix_from_array(exog_slopes, "exog_slopes");
    FullrecoverControlCore control = control_from_mapping(control_obj);
    // Env-derived research switches are not read by the profiled core; reset
    // them so this route never depends on the process environment.
    control.research_tiered_fe_tol = false;
    control.speed_fullrecover_joint_anderson = false;
    control.speed_fullrecover_accept_score_warning_path_steps = false;
    control.speed_small_manyway_exact_demean = false;
    control.speed_compact_fixed_tau_details = false;
    const std::vector<std::int64_t> groups(
        static_cast<std::size_t>(endog_vec.size()), 0);
    FullrecoverConcentratedCore result;
    double intercept = std::numeric_limits<double>::quiet_NaN();
    DirectRhoGridHookCore rho_hook;
    rho_hook.policy = rho_grid_policy;
    {
        py::gil_scoped_release release;
        validate_direct_nofe_inputs_core(endog_vec, exog_mat, tau);
        validate_direct_nofe_control_core(control);
        const FullrecoverInvariantCore state =
            fullrecover_invariant_core(exog_mat, groups, 1);
        result = concentrated_fullrecover_profiled_core(
            endog_vec,
            exog_mat,
            groups,
            1,
            tau,
            control,
            state,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            /*direct_nofe_fixed_tau=*/true,
            &rho_hook);
        if (all_finite_vec(result.beta)) {
            const Eigen::VectorXd base = exog_mat * result.beta;
            intercept = profile_intercept_adjustment_core(endog_vec, base, tau);
        }
    }

    const FullrecoverIterationCore& diag = result.iteration;
    const py::object failure_reason =
        diag.failure_reason.empty() ? py::none() : py::cast(diag.failure_reason);
    py::dict iteration;
    iteration["converged"] = diag.converged;
    iteration["phase"] = diag.phase;
    iteration["iterations"] = diag.iterations;
    iteration["restarts"] = diag.restarts;
    iteration["final_rho"] = diag.has_final_rho ? py::cast(diag.final_rho) : py::none();
    py::list rho_path;
    for (const double value : diag.rho_path) {
        rho_path.append(value);
    }
    iteration["rho_path"] = py::tuple(rho_path);
    iteration["final_criterion"] =
        diag.has_final_criterion ? py::cast(diag.final_criterion) : py::none();
    iteration["anderson_m_initial"] = diag.anderson_m_initial;
    iteration["anderson_m_final"] = diag.anderson_m_final;
    iteration["failure_reason"] = failure_reason;

    py::dict diagnostics;
    diagnostics["phase1_attempts"] = diag.phase1_attempts;
    diagnostics["phase1_accepted"] = diag.phase1_accepted;
    diagnostics["phase1_rejected"] = diag.phase1_rejected;
    diagnostics["phase2_evaluations"] = diag.phase2_evaluations;
    diagnostics["last_finite_criterion"] = diag.last_finite_criterion;
    diagnostics["best_finite_criterion"] = diag.best_finite_criterion;
    diagnostics["post_convergence_residual"] = diag.post_convergence_residual;
    diagnostics["post_convergence_kappa_est"] = diag.post_convergence_kappa_est;
    diagnostics["post_convergence_score_residual"] =
        diag.post_convergence_score_residual;
    diagnostics["post_convergence_fe_score_residual"] =
        diag.post_convergence_fe_score_residual;
    diagnostics["score_tolerance"] = diag.score_tolerance;
    diagnostics["score_validation_status"] = diag.score_validation_status;
    py::list score_codes;
    for (const std::string& code : diag.score_warning_codes) {
        score_codes.append(code);
    }
    diagnostics["score_warning_codes"] = py::tuple(score_codes);

    py::dict control_out;
    control_out["anderson_m"] = control.anderson_m;
    control_out["eps"] = control.eps;
    control_out["stopcrit"] = control.stopcrit;
    control_out["max_phase1_iterations"] = control.max_phase1_iterations;
    control_out["max_phase2_evaluations"] = control.max_phase2_evaluations;
    control_out["delta_log_min"] = control.delta_log_min;
    control_out["delta_log_max"] = control.delta_log_max;
    control_out["delta_grid_size"] = control.delta_grid_size;
    control_out["rho_log_min"] = control.rho_log_min;
    control_out["rho_log_max"] = control.rho_log_max;
    control_out["rho_grid_size"] = control.rho_grid_size;
    // runtime_v3: direct_restart_start is reported in backend_details only
    // (the harness casts non-float control echoes with int()).

    py::dict backend_details;
    if (control.pilot_deltas.has_value()) {
        py::list deltas;
        for (const double delta : *control.pilot_deltas) {
            deltas.append(delta);
        }
        backend_details["pilot_deltas"] = py::tuple(deltas);
        backend_details["pilot_grid_input"] = "exact_vector";
    }
    backend_details["runtime_version"] = kDirectNofeRuntimeVersion;
    backend_details["gil_policy"] = "released_around_validation_and_solve";
    backend_details["anderson_history_capacity_initial"] =
        control.anderson_m > 0 ? control.anderson_m + 1 : 0;
    backend_details["anderson_history_policy"] = "grows_with_adapted_m";
    backend_details["kappa_window_first_entry"] = 4;
    backend_details["qhat_check"] = "absent_in_profiled_cpp_core";
    backend_details["qhat_guard_policy"] = "absent_in_profiled_cpp_core";
    backend_details["nonconverged_iterate_policy"] =
        diag.nonconverged_iterate_policy.empty()
            ? std::string("last_attempt_final_iterate")
            : diag.nonconverged_iterate_policy;
    backend_details["phase1_evaluations"] = diag.phase1_evaluations;
    backend_details["max_phase2_evaluations"] = diag.max_phase2_evaluations;
    // runtime_v3 direct-route diagnostics (twins of the NumPy rho_details keys).
    backend_details["direct_restart_start"] = control.direct_restart_start;
    backend_details["kappa_monitor_policy"] = kDirectKappaMonitorPolicy;
    backend_details["kappa_monitor_last"] = diag.kappa_monitor_last;
    backend_details["kappa_monitor_max"] = diag.kappa_monitor_max;
    backend_details["kappa_monitor_checks"] = diag.kappa_monitor_checks;
    backend_details["kappa_coordinate_row_max"] = diag.kappa_coordinate_row_max;
    backend_details["phase2_attempts"] = diag.phase2_attempts;
    backend_details["phase2_warm_restarts"] = diag.phase2_warm_restarts;
    backend_details["best_score_norm"] = diag.best_score_norm;
    backend_details["best_score_rho"] = diag.best_score_rho;
    // runtime_v4 direct-route keys (twins of the NumPy rho_details keys).
    backend_details["progress_monitor_policy"] = kDirectProgressMonitorPolicy;
    backend_details["post_check_policy"] = kDirectPostCheckPolicy;
    backend_details["post_convergence_invalid_map_count"] =
        diag.post_convergence_invalid_map_count;
    backend_details["phase1_post_convergence_invalid_map_count"] =
        diag.phase1_post_convergence_invalid_map_count;
    py::list attempt_trace;
    for (const Phase2AttemptTraceCore& record : diag.phase2_attempt_trace) {
        py::dict entry;
        entry["attempt"] = record.attempt;
        entry["rho"] = record.rho;
        entry["iterations"] = record.iterations;
        entry["evaluations"] = record.evaluations;
        entry["failure_reason"] = record.failure_reason.empty()
                                       ? py::object(py::none())
                                       : py::object(py::cast(record.failure_reason));
        entry["final_criterion"] = record.has_final_criterion
                                        ? py::object(py::cast(record.final_criterion))
                                        : py::object(py::none());
        entry["best_criterion"] = record.best_criterion;
        entry["kappa_last"] = record.kappa_last;
        entry["start_policy"] = record.start_policy;
        entry["final_progress_residual"] = record.final_progress_residual;
        entry["best_progress_residual"] = record.best_progress_residual;
        entry["anderson_m_start"] = record.anderson_m_start;
        entry["anderson_m_end"] = record.anderson_m_end;
        entry["anderson_accepted"] = record.anderson_accepted;
        attempt_trace.append(entry);
    }
    backend_details["phase2_attempt_trace"] = py::tuple(attempt_trace);
    // rho_grid_hook_v1 diagnostics (twins of the NumPy keys; absent when the
    // Phase-2 handoff was never reached).
    if (rho_hook.reached) {
        const bool policy_grid = rho_hook.policy != kDirectRhoGridLegacy;
        backend_details["rho_grid_policy"] = rho_hook.policy;
        backend_details["rho_grid_hook_version"] = kDirectRhoGridHookVersion;
        backend_details["rho_grid_count"] = rho_hook.grid_count;
        if (policy_grid) {
            py::list grid_values;
            for (const double value : rho_hook.values) {
                grid_values.append(value);
            }
            backend_details["rho_grid_values"] = py::tuple(grid_values);
            backend_details["rho_grid_alpha0"] = rho_hook.alpha0;
            backend_details["rho_grid_alpha_min"] = rho_hook.alpha_min;
            backend_details["rho_grid_clipped"] = rho_hook.clipped;
        } else {
            backend_details["rho_grid_values"] = py::none();
            backend_details["rho_grid_alpha0"] = py::none();
            backend_details["rho_grid_alpha_min"] = py::none();
            backend_details["rho_grid_clipped"] = py::none();
        }
        if (rho_hook.has_curvature) {
            backend_details["rho_curvature_L0"] = rho_hook.curvature_L0;
            backend_details["rho_curvature_lambda_min"] = rho_hook.curvature_lambda_min;
            backend_details["rho_curvature_wall_ns"] = rho_hook.curvature_wall_ns;
            backend_details["rho_curvature_pilot"] = "phase1_accepted_beta";
        } else {
            backend_details["rho_curvature_L0"] = py::none();
            backend_details["rho_curvature_lambda_min"] = py::none();
            backend_details["rho_curvature_wall_ns"] = py::none();
            backend_details["rho_curvature_pilot"] = py::none();
        }
    }

    py::dict out;
    out["schema"] = "pyiols.direct_nofe_fixed_tau.v1";
    out["route"] = "direct_nofe_fixed_tau";
    out["backend"] = "cpp";
    out["kernel_backend"] = "cpp";
    out["algorithm"] = "iols_phase1_phase2_at_requested_tau";
    out["intercept_policy"] = "profiled_singleton_intercept";
    out["tau_requested"] = tau;
    out["tau_fit"] = tau;
    out["phase1_tau"] = tau;
    out["phase2_tau"] = tau;
    out["tau2_anchor_used"] = false;
    out["tau_continuation_used"] = false;
    out["converged"] = diag.converged;
    out["failure_reason"] = failure_reason;
    out["beta"] = vector_to_array(result.beta);
    out["intercept"] = intercept;
    out["linear"] = result.has_linear ? py::object(vector_to_array(result.linear))
                                      : py::object(py::none());
    out["iteration"] = iteration;
    out["diagnostics"] = diagnostics;
    out["control"] = control_out;
    out["backend_details"] = backend_details;
    return out;
}

}  // namespace

PYBIND11_MODULE(_kernel_cpp, m) {
    m.doc() = "Restricted pybind11/Eigen backend for the direct no-FE IOLS loop.";
    m.def("backend_info", &backend_info);
    m.def("exp_guard", &exp_guard, py::arg("values"));
    m.def("group_sum", &group_sum, py::arg("values"), py::arg("groups"),
          py::arg("n_groups"));
    m.def("group_logsumexp", &group_logsumexp, py::arg("values"),
          py::arg("groups"), py::arg("n_groups"));
    m.def("demean_oneway", &demean_oneway, py::arg("values"),
          py::arg("groups"), py::arg("n_groups"), py::arg("weights") = py::none());
    m.def("lstsq_min_norm", &lstsq_min_norm, py::arg("a"), py::arg("b"));
    m.def("anderson_lstsq", &anderson_lstsq, py::arg("f_mat"),
          py::arg("residual"));
    m.def("solve_nofe_direct_fixed_tau", &solve_nofe_direct_fixed_tau_cpp,
          py::arg("endog"), py::arg("exog_slopes"), py::arg("tau"),
          py::arg("control"), py::arg("rho_grid_policy") = "legacy");
    // runtime_v4: provenance flag; the pinned campaign build must report false.
#if defined(PYIOLS_DIRECT_TEST_HOOKS)
    m.attr("direct_test_hooks_compiled") = true;
    m.def("_test_arm_direct_post_check_fault", [](int phase, int count) {
        g_direct_test_fault_phase = phase;
        g_direct_test_fault_remaining = count;
        g_direct_test_fault_fired = 0;
    }, py::arg("phase"), py::arg("count"));
    m.def("_test_direct_post_check_fault_fired", []() {
        return g_direct_test_fault_fired;
    });
#else
    m.attr("direct_test_hooks_compiled") = false;
#endif
}
