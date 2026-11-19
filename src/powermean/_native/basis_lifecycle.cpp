// Exact fresh-validation and proposal lifecycle for certificate_basis.
// CPython 3.12 C API, standard C++17, arbitrary-precision integer/rational fallback
// via exact PyLong operations, no fixed-width narrowing, GIL held throughout.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace {

struct PyError : public std::exception {
    const char* what() const noexcept override { return "Python C-API exception"; }
};

struct PyObjRef {
    PyObject* ptr = nullptr;
    PyObjRef() noexcept = default;
    explicit PyObjRef(PyObject* p) noexcept : ptr(p) {}
    ~PyObjRef() { Py_XDECREF(ptr); }
    PyObjRef(const PyObjRef& o) noexcept : ptr(o.ptr) { Py_XINCREF(ptr); }
    PyObjRef(PyObjRef&& o) noexcept : ptr(o.ptr) { o.ptr = nullptr; }
    PyObjRef& operator=(const PyObjRef& o) noexcept {
        if (this != &o) {
            Py_XINCREF(o.ptr);
            Py_XDECREF(ptr);
            ptr = o.ptr;
        }
        return *this;
    }
    PyObjRef& operator=(PyObjRef&& o) noexcept {
        if (this != &o) {
            Py_XDECREF(ptr);
            ptr = o.ptr;
            o.ptr = nullptr;
        }
        return *this;
    }
    PyObject* get() const noexcept { return ptr; }
    PyObject* release() noexcept {
        PyObject* p = ptr;
        ptr = nullptr;
        return p;
    }
    explicit operator bool() const noexcept { return ptr != nullptr; }
};

// Module-level cached Python types and functions (initialized in PyInit_basis_lifecycle)
PyTypeObject* g_ndarray_type = nullptr;
PyObject* g_intp_dtype = nullptr;
PyObject* g_one_long = nullptr;

// ---------------------------------------------------------------------------
// ExactInt: inline int64_t within [-kInlineMax, kInlineMax], promoting to
// exact CPython PyLongObject* outside that range. Never narrows any value.
// ---------------------------------------------------------------------------
class ExactInt {
public:
    static constexpr int64_t kInlineMax = (1LL << 62) - 1;

    ExactInt() noexcept : small_(0), big_(nullptr) {}

    explicit ExactInt(int64_t v) : small_(0), big_(nullptr) {
        if (v >= -kInlineMax && v <= kInlineMax) {
            small_ = v;
        } else {
            big_ = PyLong_FromLongLong(v);
            if (!big_) {
                throw PyError();
            }
        }
    }

    ~ExactInt() { Py_XDECREF(big_); }

    ExactInt(const ExactInt& o) noexcept : small_(o.small_), big_(o.big_) {
        Py_XINCREF(big_);
    }

    ExactInt(ExactInt&& o) noexcept : small_(o.small_), big_(o.big_) {
        o.small_ = 0;
        o.big_ = nullptr;
    }

    ExactInt& operator=(const ExactInt& o) noexcept {
        if (this != &o) {
            Py_XINCREF(o.big_);
            Py_XDECREF(big_);
            small_ = o.small_;
            big_ = o.big_;
        }
        return *this;
    }

    ExactInt& operator=(ExactInt&& o) noexcept {
        if (this != &o) {
            Py_XDECREF(big_);
            small_ = o.small_;
            big_ = o.big_;
            o.small_ = 0;
            o.big_ = nullptr;
        }
        return *this;
    }

    static ExactInt take_pyobj(PyObject* owned) {
        if (!owned) {
            throw PyError();
        }
        int overflow = 0;
        long long v = PyLong_AsLongLongAndOverflow(owned, &overflow);
        if (v == -1 && PyErr_Occurred()) {
            Py_DECREF(owned);
            throw PyError();
        }
        if (overflow == 0 && v >= -kInlineMax && v <= kInlineMax) {
            Py_DECREF(owned);
            ExactInt res;
            res.small_ = static_cast<int64_t>(v);
            return res;
        }
        ExactInt res;
        res.big_ = owned;
        return res;
    }

    static bool from_pylong_exact(PyObject* obj, ExactInt& out) {
        if (!PyLong_CheckExact(obj)) {
            return false;
        }
        int overflow = 0;
        long long v = PyLong_AsLongLongAndOverflow(obj, &overflow);
        if (v == -1 && PyErr_Occurred()) {
            throw PyError();
        }
        Py_XDECREF(out.big_);
        if (overflow == 0 && v >= -kInlineMax && v <= kInlineMax) {
            out.small_ = static_cast<int64_t>(v);
            out.big_ = nullptr;
        } else {
            Py_INCREF(obj);
            out.small_ = 0;
            out.big_ = obj;
        }
        return true;
    }

    PyObjRef to_pyobj() const {
        if (big_) {
            Py_INCREF(big_);
            return PyObjRef(big_);
        }
        PyObject* p = PyLong_FromLongLong(small_);
        if (!p) {
            throw PyError();
        }
        return PyObjRef(p);
    }

    bool is_zero() const noexcept { return big_ == nullptr && small_ == 0; }
    bool is_one() const noexcept { return big_ == nullptr && small_ == 1; }
    bool is_minus_one() const noexcept { return big_ == nullptr && small_ == -1; }

    int sgn() const {
        if (!big_) {
            return (small_ > 0) - (small_ < 0);
        }
        PyObjRef zero(PyLong_FromLong(0));
        if (!zero) {
            throw PyError();
        }
        int gt = PyObject_RichCompareBool(big_, zero.get(), Py_GT);
        if (gt < 0) {
            throw PyError();
        }
        return gt ? 1 : -1;
    }

    ExactInt operator-() const {
        if (!big_) {
            ExactInt res;
            res.small_ = -small_;
            return res;
        }
        return take_pyobj(PyNumber_Negative(big_));
    }

    ExactInt abs() const { return sgn() < 0 ? -(*this) : *this; }

    bool operator==(const ExactInt& o) const {
        if (!big_ && !o.big_) {
            return small_ == o.small_;
        }
        if ((!big_) != (!o.big_)) {
            return false;
        }
        int eq = PyObject_RichCompareBool(big_, o.big_, Py_EQ);
        if (eq < 0) {
            throw PyError();
        }
        return eq == 1;
    }

    bool operator!=(const ExactInt& o) const { return !(*this == o); }

    ExactInt operator+(const ExactInt& o) const {
        if (!big_ && !o.big_) {
            int64_t s = small_ + o.small_;
            if (s >= -kInlineMax && s <= kInlineMax) {
                ExactInt res;
                res.small_ = s;
                return res;
            }
            return take_pyobj(PyLong_FromLongLong(s));
        }
        PyObjRef a = to_pyobj();
        PyObjRef b = o.to_pyobj();
        return take_pyobj(PyNumber_Add(a.get(), b.get()));
    }

    ExactInt operator-(const ExactInt& o) const {
        if (!big_ && !o.big_) {
            int64_t d = small_ - o.small_;
            if (d >= -kInlineMax && d <= kInlineMax) {
                ExactInt res;
                res.small_ = d;
                return res;
            }
            return take_pyobj(PyLong_FromLongLong(d));
        }
        PyObjRef a = to_pyobj();
        PyObjRef b = o.to_pyobj();
        return take_pyobj(PyNumber_Subtract(a.get(), b.get()));
    }

    ExactInt operator*(const ExactInt& o) const {
        if (!big_ && !o.big_) {
            int64_t p = 0;
            if (!__builtin_mul_overflow(small_, o.small_, &p) &&
                p >= -kInlineMax && p <= kInlineMax) {
                ExactInt res;
                res.small_ = p;
                return res;
            }
        }
        PyObjRef a = to_pyobj();
        PyObjRef b = o.to_pyobj();
        return take_pyobj(PyNumber_Multiply(a.get(), b.get()));
    }

    static ExactInt divexact(const ExactInt& a, const ExactInt& b) {
        if (!a.big_ && !b.big_) {
            ExactInt res;
            res.small_ = a.small_ / b.small_;
            return res;
        }
        PyObjRef pa = a.to_pyobj();
        PyObjRef pb = b.to_pyobj();
        return take_pyobj(PyNumber_FloorDivide(pa.get(), pb.get()));
    }

    static ExactInt gcd(const ExactInt& a, const ExactInt& b) {
        if (!a.big_ && !b.big_) {
            uint64_t ua = static_cast<uint64_t>(a.small_ < 0 ? -a.small_ : a.small_);
            uint64_t ub = static_cast<uint64_t>(b.small_ < 0 ? -b.small_ : b.small_);
            ExactInt res;
            res.small_ = static_cast<int64_t>(std::gcd(ua, ub));
            return res;
        }
        // Exact PyLong remainder cannot run a monkeypatched Python math.gcd
        // callback while grouping buffers are held.
        ExactInt left = a.abs();
        ExactInt right = b.abs();
        while (!right.is_zero()) {
            PyObjRef pa = left.to_pyobj();
            PyObjRef pb = right.to_pyobj();
            ExactInt remainder = take_pyobj(PyNumber_Remainder(pa.get(), pb.get()));
            left = std::move(right);
            right = std::move(remainder);
        }
        return left;
    }

    static ExactInt lcm(const ExactInt& a, const ExactInt& b) {
        if (a.is_zero() || b.is_zero()) {
            return ExactInt(0);
        }
        ExactInt g = gcd(a, b);
        return divexact(a.abs(), g) * b.abs();
    }

private:
    int64_t small_;
    PyObject* big_;
};

// ---------------------------------------------------------------------------
// ExactRat: canonical reduced rational (num, den) with den > 0, gcd == 1.
// ---------------------------------------------------------------------------
class ExactRat {
public:
    ExactInt num;
    ExactInt den;

    ExactRat() noexcept : num(0), den(1) {}
    explicit ExactRat(int64_t n) : num(n), den(1) {}
    explicit ExactRat(ExactInt n) : num(std::move(n)), den(1) {}

    ExactRat(ExactInt n, ExactInt d) {
        int ds = d.sgn();
        if (ds < 0) {
            n = -n;
            d = -d;
        }
        if (n.is_zero()) {
            num = ExactInt(0);
            den = ExactInt(1);
        } else if (d.is_one()) {
            num = std::move(n);
            den = ExactInt(1);
        } else {
            ExactInt g = ExactInt::gcd(n, d);
            if (g.is_one()) {
                num = std::move(n);
                den = std::move(d);
            } else {
                num = ExactInt::divexact(n, g);
                den = ExactInt::divexact(d, g);
            }
        }
    }

    bool is_zero() const noexcept { return num.is_zero(); }

    ExactRat operator-() const {
        ExactRat r;
        r.num = -num;
        r.den = den;
        return r;
    }

    bool operator==(const ExactRat& o) const {
        return num == o.num && den == o.den;
    }

    bool operator!=(const ExactRat& o) const { return !(*this == o); }

    ExactRat operator+(const ExactRat& o) const {
        if (num.is_zero()) return o;
        if (o.num.is_zero()) return *this;
        if (den.is_one() && o.den.is_one()) {
            return ExactRat(num + o.num);
        }
        if (den == o.den) {
            return ExactRat(num + o.num, den);
        }
        return ExactRat(num * o.den + o.num * den, den * o.den);
    }

    ExactRat operator-(const ExactRat& o) const {
        if (o.num.is_zero()) return *this;
        if (num.is_zero()) return -o;
        if (den.is_one() && o.den.is_one()) {
            return ExactRat(num - o.num);
        }
        if (den == o.den) {
            return ExactRat(num - o.num, den);
        }
        return ExactRat(num * o.den - o.num * den, den * o.den);
    }

    ExactRat operator*(const ExactRat& o) const {
        if (num.is_zero() || o.num.is_zero()) {
            return ExactRat(0);
        }
        if (den.is_one() && o.den.is_one()) {
            return ExactRat(num * o.num);
        }
        return ExactRat(num * o.num, den * o.den);
    }

    ExactRat operator/(const ExactRat& o) const {
        if (num.is_zero()) {
            return ExactRat(0);
        }
        if (den.is_one() && o.den.is_one()) {
            return ExactRat(num, o.num);
        }
        return ExactRat(num * o.den, den * o.num);
    }
};

// ---------------------------------------------------------------------------
// RAII Py_buffer guard for exact 1-D C-contiguous np.intp arrays
// ---------------------------------------------------------------------------
struct IntpBuffer {
    Py_buffer view{};
    bool active = false;
    std::vector<Py_ssize_t> snapshot;
    const Py_ssize_t* data = nullptr;
    Py_ssize_t size = 0;

    IntpBuffer() noexcept = default;
    ~IntpBuffer() {
        if (active) {
            PyBuffer_Release(&view);
        }
    }
    IntpBuffer(const IntpBuffer&) = delete;
    IntpBuffer& operator=(const IntpBuffer&) = delete;
    IntpBuffer(IntpBuffer&& o) noexcept
        : view(o.view), active(o.active), snapshot(std::move(o.snapshot)),
          data(snapshot.data()), size(o.size) {
        o.active = false;
        o.data = nullptr;
        o.size = 0;
    }

    bool acquire(PyObject* obj) {
        if (!obj || !Py_IS_TYPE(obj, g_ndarray_type)) {
            return false;
        }
        PyObjRef dt(PyObject_GetAttrString(obj, "dtype"));
        if (!dt) {
            PyErr_Clear();
            return false;
        }
        int eq = PyObject_RichCompareBool(dt.get(), g_intp_dtype, Py_EQ);
        if (eq <= 0) {
            if (eq < 0) PyErr_Clear();
            return false;
        }
        if (PyObject_GetBuffer(obj, &view, PyBUF_STRIDES) != 0) {
            PyErr_Clear();
            return false;
        }
        active = true;
        if (view.ndim != 1 || view.itemsize != static_cast<Py_ssize_t>(sizeof(Py_ssize_t)) ||
            !view.shape) {
            return false;
        }
        size = view.shape[0];
        if (size < 0 || size > PY_SSIZE_T_MAX / view.itemsize ||
            (size > 0 && (!view.buf ||
             reinterpret_cast<uintptr_t>(view.buf) % alignof(Py_ssize_t) != 0))) {
            return false;
        }
        if (size > 1 && (!view.strides || view.strides[0] != view.itemsize)) {
            return false;
        }
        if (view.len != size * view.itemsize) {
            return false;
        }
        // Subsequent Python object allocation can run cyclic-GC finalizers.
        // Never revisit caller-owned storage after bounds validation: a callback
        // must not turn a previously checked code into an out-of-bounds index.
        snapshot.resize(static_cast<size_t>(size));
        if (size > 0) {
            std::memcpy(snapshot.data(), view.buf, static_cast<size_t>(view.len));
        }
        data = snapshot.data();
        PyBuffer_Release(&view);
        active = false;
        return true;
    }
};

bool parse_bounded_int(PyObject* obj, int64_t lo, int64_t hi, int64_t& out) {
    if (!PyLong_CheckExact(obj)) {
        return false;
    }
    int overflow = 0;
    long long v = PyLong_AsLongLongAndOverflow(obj, &overflow);
    if (v == -1 && PyErr_Occurred()) {
        throw PyError();
    }
    if (overflow != 0 || v < lo || v > hi) {
        return false;
    }
    out = static_cast<int64_t>(v);
    return true;
}

bool same_structure(PyObject* a, PyObject* b) {
    if (Py_TYPE(a) != Py_TYPE(b)) {
        return false;
    }
    if (PyTuple_CheckExact(a)) {
        Py_ssize_t n = PyTuple_GET_SIZE(a);
        if (n != PyTuple_GET_SIZE(b)) {
            return false;
        }
        for (Py_ssize_t i = 0; i < n; ++i) {
            if (!same_structure(PyTuple_GET_ITEM(a, i), PyTuple_GET_ITEM(b, i))) {
                return false;
            }
        }
        return true;
    }
    // Every expected leaf here is an exact integer, string or byte string.
    // Do not invoke arbitrary equality while raw array pointers are held.
    if (!PyLong_CheckExact(a) && !PyUnicode_CheckExact(a) && !PyBytes_CheckExact(a)) {
        return false;
    }
    int eq = PyObject_RichCompareBool(a, b, Py_EQ);
    if (eq < 0) {
        throw PyError();
    }
    return eq == 1;
}

bool plain_content(PyObject* content) {
    if (!PyTuple_CheckExact(content) || PyTuple_GET_SIZE(content) < 1 ||
        !PyBytes_CheckExact(PyTuple_GET_ITEM(content, 0))) {
        return false;
    }
    for (Py_ssize_t k = 1; k < PyTuple_GET_SIZE(content); ++k) {
        PyObject* item = PyTuple_GET_ITEM(content, k);
        if (!PyTuple_CheckExact(item) || PyTuple_GET_SIZE(item) != 3 ||
            !PyUnicode_CheckExact(PyTuple_GET_ITEM(item, 0)) ||
            !PyBytes_CheckExact(PyTuple_GET_ITEM(item, 1)) ||
            !PyBytes_CheckExact(PyTuple_GET_ITEM(item, 2))) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Extracted Grouping Data and Topology Computations (components, rank, distinct)
// ---------------------------------------------------------------------------
struct GroupingData {
    int G = 0;
    int64_t n_rows = 0;
    std::vector<IntpBuffer> codes_bufs;
    std::vector<IntpBuffer> counts_bufs;
    std::vector<int64_t> n_levels;
    std::vector<int64_t> offsets;
    int64_t total_vertices = 0;
};

bool extract_grouping(PyObject* codes_tuple, PyObject* counts_tuple, GroupingData& gd) {
    if (!PyTuple_CheckExact(codes_tuple) || !PyTuple_CheckExact(counts_tuple)) {
        return false;
    }
    Py_ssize_t G = PyTuple_GET_SIZE(codes_tuple);
    if (G < 1 || G != PyTuple_GET_SIZE(counts_tuple) || G > 10000) {
        return false;
    }
    gd.G = static_cast<int>(G);
    gd.codes_bufs.resize(static_cast<size_t>(G));
    gd.counts_bufs.resize(static_cast<size_t>(G));
    gd.n_levels.resize(static_cast<size_t>(G));
    gd.offsets.resize(static_cast<size_t>(G));

    int64_t offset = 0;
    for (Py_ssize_t k = 0; k < G; ++k) {
        if (!gd.codes_bufs[static_cast<size_t>(k)].acquire(PyTuple_GET_ITEM(codes_tuple, k)) ||
            !gd.counts_bufs[static_cast<size_t>(k)].acquire(PyTuple_GET_ITEM(counts_tuple, k))) {
            return false;
        }
        int64_t nr = gd.codes_bufs[static_cast<size_t>(k)].size;
        int64_t nl = gd.counts_bufs[static_cast<size_t>(k)].size;
        if (nr <= 0 || nl <= 0) {
            return false;
        }
        if (k == 0) {
            gd.n_rows = nr;
        } else if (nr != gd.n_rows) {
            return false;
        }
        const Py_ssize_t* cptr = gd.codes_bufs[static_cast<size_t>(k)].data;
        std::vector<uint8_t> seen(static_cast<size_t>(nl), 0);
        int64_t seen_count = 0;
        for (int64_t i = 0; i < nr; ++i) {
            Py_ssize_t c = cptr[i];
            if (c < 0 || c >= nl) {
                return false;
            }
            if (!seen[static_cast<size_t>(c)]) {
                seen[static_cast<size_t>(c)] = 1;
                ++seen_count;
            }
        }
        if (seen_count != nl) {
            return false;
        }
        gd.n_levels[static_cast<size_t>(k)] = nl;
        gd.offsets[static_cast<size_t>(k)] = offset;
        // All index vectors and serialized index bytes must fit Py_ssize_t.
        if (nl > PY_SSIZE_T_MAX / static_cast<int64_t>(sizeof(Py_ssize_t)) - offset) {
            return false;
        }
        offset += nl;
    }
    gd.total_vertices = offset;
    return true;
}

struct TopologyInfo {
    int64_t n_components = 0;
    std::vector<int64_t> of_rows;
    std::vector<std::vector<int64_t>> of_levels;
    std::vector<std::tuple<int, int, int64_t>> pairwise;
    int64_t rank_lower = 0;
    int64_t rank_upper = 0;
    bool has_exact_rank = false;
    int64_t exact_rank = 0;

    std::vector<std::vector<Py_ssize_t>> references;
    std::vector<std::vector<Py_ssize_t>> columns;
    int64_t count = 0;
    std::vector<int64_t> distinct_rows;
};

int64_t dsu_find(std::vector<int64_t>& parent, int64_t x) noexcept {
    int64_t root = x;
    while (parent[static_cast<size_t>(root)] != root) {
        root = parent[static_cast<size_t>(root)];
    }
    while (parent[static_cast<size_t>(x)] != root) {
        int64_t nxt = parent[static_cast<size_t>(x)];
        parent[static_cast<size_t>(x)] = root;
        x = nxt;
    }
    return root;
}

bool compute_topology(const GroupingData& gd, TopologyInfo& topo) {
    const int G = gd.G;
    const int64_t n_rows = gd.n_rows;
    const int64_t V = gd.total_vertices;

    std::vector<int64_t> parent(static_cast<size_t>(V));
    std::iota(parent.begin(), parent.end(), int64_t{0});

    const Py_ssize_t* c0 = gd.codes_bufs[0].data;
    for (int k = 1; k < G; ++k) {
        const Py_ssize_t* ck = gd.codes_bufs[static_cast<size_t>(k)].data;
        const int64_t off_k = gd.offsets[static_cast<size_t>(k)];
        for (int64_t i = 0; i < n_rows; ++i) {
            int64_t ra = dsu_find(parent, c0[i]);
            int64_t rb = dsu_find(parent, off_k + ck[i]);
            if (ra != rb) {
                if (ra < rb) {
                    parent[static_cast<size_t>(rb)] = ra;
                } else {
                    parent[static_cast<size_t>(ra)] = rb;
                }
            }
        }
    }

    std::vector<int64_t> comp_id(static_cast<size_t>(V), -1);
    topo.of_rows.resize(static_cast<size_t>(n_rows));
    int64_t n_comp = 0;
    for (int64_t i = 0; i < n_rows; ++i) {
        int64_t r = dsu_find(parent, c0[i]);
        int64_t cid = comp_id[static_cast<size_t>(r)];
        if (cid < 0) {
            cid = n_comp++;
            comp_id[static_cast<size_t>(r)] = cid;
        }
        topo.of_rows[static_cast<size_t>(i)] = cid;
    }
    topo.n_components = n_comp;

    topo.of_levels.resize(static_cast<size_t>(G));
    for (int k = 0; k < G; ++k) {
        int64_t nl = gd.n_levels[static_cast<size_t>(k)];
        int64_t off_k = gd.offsets[static_cast<size_t>(k)];
        topo.of_levels[static_cast<size_t>(k)].resize(static_cast<size_t>(nl));
        for (int64_t l = 0; l < nl; ++l) {
            int64_t r = dsu_find(parent, off_k + l);
            int64_t cid = comp_id[static_cast<size_t>(r)];
            if (cid < 0) {
                return false;
            }
            topo.of_levels[static_cast<size_t>(k)][static_cast<size_t>(l)] = cid;
        }
    }

    // IndicatorRank
    if (G == 1) {
        topo.rank_lower = gd.n_levels[0];
        topo.rank_upper = gd.n_levels[0];
    } else if (G == 2) {
        int64_t r = gd.n_levels[0] + gd.n_levels[1] - n_comp;
        topo.rank_lower = r;
        topo.rank_upper = r;
    } else {
        int64_t lower = 0;
        int64_t sum_levels = 0;
        for (int k = 0; k < G; ++k) {
            sum_levels += gd.n_levels[static_cast<size_t>(k)];
        }
        std::vector<int64_t> pair_parent;
        std::vector<uint8_t> root_seen;
        for (int a = 0; a < G; ++a) {
            int64_t la = gd.n_levels[static_cast<size_t>(a)];
            const Py_ssize_t* ca = gd.codes_bufs[static_cast<size_t>(a)].data;
            for (int b = a + 1; b < G; ++b) {
                int64_t lb = gd.n_levels[static_cast<size_t>(b)];
                const Py_ssize_t* cb = gd.codes_bufs[static_cast<size_t>(b)].data;
                int64_t v_ab = la + lb;
                pair_parent.resize(static_cast<size_t>(v_ab));
                std::iota(pair_parent.begin(), pair_parent.end(), int64_t{0});
                for (int64_t i = 0; i < n_rows; ++i) {
                    int64_t ra = dsu_find(pair_parent, ca[i]);
                    int64_t rb = dsu_find(pair_parent, la + cb[i]);
                    if (ra != rb) {
                        if (ra < rb) {
                            pair_parent[static_cast<size_t>(rb)] = ra;
                        } else {
                            pair_parent[static_cast<size_t>(ra)] = rb;
                        }
                    }
                }
                root_seen.assign(static_cast<size_t>(v_ab), 0);
                int64_t c_ab = 0;
                for (int64_t i = 0; i < n_rows; ++i) {
                    int64_t r = dsu_find(pair_parent, ca[i]);
                    if (!root_seen[static_cast<size_t>(r)]) {
                        root_seen[static_cast<size_t>(r)] = 1;
                        ++c_ab;
                    }
                }
                topo.pairwise.emplace_back(a, b, c_ab);
                int64_t pair_rank = la + lb - c_ab;
                if (pair_rank > lower) {
                    lower = pair_rank;
                }
            }
        }
        topo.rank_lower = lower;
        topo.rank_upper = sum_levels - static_cast<int64_t>(G - 1) * n_comp;
    }
    topo.has_exact_rank = (topo.rank_lower == topo.rank_upper);
    topo.exact_rank = topo.rank_lower;

    // _references and _columns
    topo.references.resize(static_cast<size_t>(G - 1));
    for (int k = 1; k < G; ++k) {
        int64_t nl = gd.n_levels[static_cast<size_t>(k)];
        auto& refs = topo.references[static_cast<size_t>(k - 1)];
        refs.assign(static_cast<size_t>(n_comp), static_cast<Py_ssize_t>(nl));
        const auto& of_lvl = topo.of_levels[static_cast<size_t>(k)];
        for (int64_t l = 0; l < nl; ++l) {
            int64_t comp = of_lvl[static_cast<size_t>(l)];
            if (l < refs[static_cast<size_t>(comp)]) {
                refs[static_cast<size_t>(comp)] = static_cast<Py_ssize_t>(l);
            }
        }
    }

    topo.columns.resize(static_cast<size_t>(G));
    int64_t col_offset = 0;
    for (int k = 0; k < G; ++k) {
        int64_t nl = gd.n_levels[static_cast<size_t>(k)];
        std::vector<uint8_t> is_kept(static_cast<size_t>(nl), 1);
        if (k > 0) {
            for (Py_ssize_t ref : topo.references[static_cast<size_t>(k - 1)]) {
                is_kept[static_cast<size_t>(ref)] = 0;
            }
        }
        auto& col = topo.columns[static_cast<size_t>(k)];
        col.assign(static_cast<size_t>(nl), -1);
        for (int64_t l = 0; l < nl; ++l) {
            if (is_kept[static_cast<size_t>(l)]) {
                col[static_cast<size_t>(l)] = static_cast<Py_ssize_t>(col_offset++);
            }
        }
    }
    topo.count = col_offset;

    // _distinct: first-occurrence distinct code rows in row order
    size_t cap = 16;
    while (cap < static_cast<size_t>(n_rows) * 2) {
        cap <<= 1;
    }
    size_t mask = cap - 1;
    std::vector<int64_t> table(cap, -1);
    topo.distinct_rows.clear();
    topo.distinct_rows.reserve(static_cast<size_t>(n_rows));

    for (int64_t i = 0; i < n_rows; ++i) {
        uint64_t h = 0x9e3779b97f4a7c15ULL;
        for (int k = 0; k < G; ++k) {
            uint64_t v = static_cast<uint64_t>(gd.codes_bufs[static_cast<size_t>(k)].data[i]);
            h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        size_t slot = static_cast<size_t>(h) & mask;
        bool duplicate = false;
        while (table[slot] != -1) {
            int64_t j = table[slot];
            bool same = true;
            for (int k = 0; k < G; ++k) {
                if (gd.codes_bufs[static_cast<size_t>(k)].data[j] !=
                    gd.codes_bufs[static_cast<size_t>(k)].data[i]) {
                    same = false;
                    break;
                }
            }
            if (same) {
                duplicate = true;
                break;
            }
            slot = (slot + 1) & mask;
        }
        if (!duplicate) {
            table[slot] = i;
            topo.distinct_rows.push_back(i);
        }
    }
    return true;
}

PyObjRef build_rank_tuple(const GroupingData& gd, const TopologyInfo& topo) {
    PyObjRef levels(PyTuple_New(gd.G));
    if (!levels) throw PyError();
    for (int k = 0; k < gd.G; ++k) {
        PyObject* v = PyLong_FromLongLong(gd.n_levels[static_cast<size_t>(k)]);
        if (!v) throw PyError();
        PyTuple_SET_ITEM(levels.get(), k, v);
    }

    PyObjRef pairwise(PyTuple_New(static_cast<Py_ssize_t>(topo.pairwise.size())));
    if (!pairwise) throw PyError();
    for (size_t idx = 0; idx < topo.pairwise.size(); ++idx) {
        auto [i, j, c] = topo.pairwise[idx];
        PyObjRef item(PyTuple_New(3));
        if (!item) throw PyError();
        PyObject* pi = PyLong_FromLong(i);
        PyObject* pj = PyLong_FromLong(j);
        PyObject* pc = PyLong_FromLongLong(c);
        if (!pi || !pj || !pc) {
            Py_XDECREF(pi);
            Py_XDECREF(pj);
            Py_XDECREF(pc);
            throw PyError();
        }
        PyTuple_SET_ITEM(item.get(), 0, pi);
        PyTuple_SET_ITEM(item.get(), 1, pj);
        PyTuple_SET_ITEM(item.get(), 2, pc);
        PyTuple_SET_ITEM(pairwise.get(), static_cast<Py_ssize_t>(idx), item.release());
    }

    PyObjRef res(PyTuple_New(5));
    if (!res) throw PyError();
    PyObject* n_comp = PyLong_FromLongLong(topo.n_components);
    PyObject* lower = PyLong_FromLongLong(topo.rank_lower);
    PyObject* upper = PyLong_FromLongLong(topo.rank_upper);
    if (!n_comp || !lower || !upper) {
        Py_XDECREF(n_comp);
        Py_XDECREF(lower);
        Py_XDECREF(upper);
        throw PyError();
    }
    PyTuple_SET_ITEM(res.get(), 0, levels.release());
    PyTuple_SET_ITEM(res.get(), 1, n_comp);
    PyTuple_SET_ITEM(res.get(), 2, pairwise.release());
    PyTuple_SET_ITEM(res.get(), 3, lower);
    PyTuple_SET_ITEM(res.get(), 4, upper);
    return res;
}

bool check_bytes_tuple(PyObject* obj, const std::vector<std::vector<Py_ssize_t>>& expected) {
    if (!PyTuple_CheckExact(obj) ||
        PyTuple_GET_SIZE(obj) != static_cast<Py_ssize_t>(expected.size())) {
        return false;
    }
    for (size_t k = 0; k < expected.size(); ++k) {
        PyObject* item = PyTuple_GET_ITEM(obj, static_cast<Py_ssize_t>(k));
        if (!PyBytes_CheckExact(item)) {
            return false;
        }
        Py_ssize_t exp_bytes = static_cast<Py_ssize_t>(expected[k].size() * sizeof(Py_ssize_t));
        if (PyBytes_GET_SIZE(item) != exp_bytes) {
            return false;
        }
        if (exp_bytes > 0 &&
            std::memcmp(PyBytes_AS_STRING(item), expected[k].data(), static_cast<size_t>(exp_bytes)) != 0) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Sparse rational/integer helpers for validation and proposal
// ---------------------------------------------------------------------------
using SparseIntRow = std::vector<std::pair<int64_t, ExactInt>>;
using SparseRatRow = std::vector<std::pair<int64_t, ExactRat>>;

struct RatWorkspace {
    std::vector<ExactRat> val;
    std::vector<uint8_t> in;
    std::vector<int64_t> cols;

    explicit RatWorkspace(int64_t n)
        : val(static_cast<size_t>(n)), in(static_cast<size_t>(n), 0) {
        cols.reserve(32);
    }

    void add(int64_t c, const ExactRat& v) {
        if (v.is_zero()) return;
        size_t uc = static_cast<size_t>(c);
        if (!in[uc]) {
            in[uc] = 1;
            val[uc] = v;
            cols.push_back(c);
        } else {
            val[uc] = val[uc] + v;
            if (val[uc].is_zero()) {
                in[uc] = 0;
            }
        }
    }

    SparseRatRow extract_sorted() {
        std::sort(cols.begin(), cols.end());
        SparseRatRow out;
        int64_t prev = -1;
        for (int64_t c : cols) {
            if (c == prev) continue;
            prev = c;
            size_t uc = static_cast<size_t>(c);
            if (in[uc]) {
                in[uc] = 0;
                if (!val[uc].is_zero()) {
                    out.emplace_back(c, std::move(val[uc]));
                }
            }
        }
        cols.clear();
        return out;
    }

    void clear() noexcept {
        for (int64_t c : cols) {
            in[static_cast<size_t>(c)] = 0;
        }
        cols.clear();
    }
};

struct IntWorkspace {
    std::vector<ExactInt> val;
    std::vector<uint8_t> in;
    std::vector<int64_t> cols;

    explicit IntWorkspace(int64_t n)
        : val(static_cast<size_t>(n)), in(static_cast<size_t>(n), 0) {
        cols.reserve(32);
    }

    void add(int64_t c, const ExactInt& v) {
        if (v.is_zero()) return;
        size_t uc = static_cast<size_t>(c);
        if (!in[uc]) {
            in[uc] = 1;
            val[uc] = v;
            cols.push_back(c);
        } else {
            val[uc] = val[uc] + v;
            if (val[uc].is_zero()) {
                in[uc] = 0;
            }
        }
    }

    bool all_zero_and_clear() noexcept {
        bool ok = true;
        for (int64_t c : cols) {
            size_t uc = static_cast<size_t>(c);
            if (in[uc]) {
                if (!val[uc].is_zero()) {
                    ok = false;
                }
                in[uc] = 0;
            }
        }
        cols.clear();
        return ok;
    }
};

SparseIntRow make_primitive(const SparseRatRow& found) {
    ExactInt scale(1);
    for (const auto& [c, r] : found) {
        if (!r.den.is_one()) {
            scale = ExactInt::lcm(scale, r.den);
        }
    }
    SparseIntRow row;
    row.reserve(found.size());
    bool has_unit = false;
    for (const auto& [c, r] : found) {
        ExactInt v = scale.is_one() ? r.num : (r.num * ExactInt::divexact(scale, r.den));
        if (v.is_one() || v.is_minus_one()) {
            has_unit = true;
        }
        row.emplace_back(c, std::move(v));
    }
    ExactInt div(1);
    if (!has_unit) {
        div = row[0].second.abs();
        for (size_t i = 1; i < row.size() && !div.is_one(); ++i) {
            div = ExactInt::gcd(div, row[i].second);
        }
    }
    if (row[0].second.sgn() < 0) {
        div = -div;
    }
    if (!div.is_one()) {
        for (auto& [c, v] : row) {
            v = ExactInt::divexact(v, div);
        }
    }
    return row;
}

bool parse_echelon(
    PyObject* echelon_obj,
    int64_t count,
    std::vector<SparseIntRow>& pivot_rows,
    std::vector<uint8_t>& has_pivot,
    std::vector<Py_ssize_t>& kept
) {
    if (!PyTuple_CheckExact(echelon_obj)) {
        return false;
    }
    Py_ssize_t n_pivs = PyTuple_GET_SIZE(echelon_obj);
    pivot_rows.assign(static_cast<size_t>(count), SparseIntRow{});
    has_pivot.assign(static_cast<size_t>(count), 0);
    kept.clear();
    kept.reserve(static_cast<size_t>(n_pivs));

    int64_t prev_p = -1;
    for (Py_ssize_t i = 0; i < n_pivs; ++i) {
        PyObject* row_tuple = PyTuple_GET_ITEM(echelon_obj, i);
        if (!PyTuple_CheckExact(row_tuple)) {
            return false;
        }
        Py_ssize_t row_len = PyTuple_GET_SIZE(row_tuple);
        if (row_len <= 0) {
            return false;
        }
        SparseIntRow row;
        row.reserve(static_cast<size_t>(row_len));
        int64_t prev_c = -1;
        bool has_unit = false;
        for (Py_ssize_t j = 0; j < row_len; ++j) {
            PyObject* pair = PyTuple_GET_ITEM(row_tuple, j);
            if (!PyTuple_CheckExact(pair) || PyTuple_GET_SIZE(pair) != 2) {
                return false;
            }
            int64_t c = -1;
            if (!parse_bounded_int(PyTuple_GET_ITEM(pair, 0), prev_c + 1, count - 1, c)) {
                return false;
            }
            ExactInt val;
            if (!ExactInt::from_pylong_exact(PyTuple_GET_ITEM(pair, 1), val) || val.is_zero()) {
                return false;
            }
            if (val.is_one() || val.is_minus_one()) {
                has_unit = true;
            }
            prev_c = c;
            row.emplace_back(c, std::move(val));
        }
        int64_t p = row[0].first;
        if (!(prev_p < p && p < count) || row[0].second.sgn() <= 0) {
            return false;
        }
        if (!has_unit) {
            ExactInt g = row[0].second;
            for (size_t j = 1; j < row.size() && !g.is_one(); ++j) {
                g = ExactInt::gcd(g, row[j].second);
            }
            if (!g.is_one()) {
                return false;
            }
        }
        prev_p = p;
        has_pivot[static_cast<size_t>(p)] = 1;
        pivot_rows[static_cast<size_t>(p)] = std::move(row);
        kept.push_back(static_cast<Py_ssize_t>(p));
    }
    return true;
}

bool check_unit_witness(PyObject* w, int64_t expected_pos) {
    if (!PyTuple_CheckExact(w) || PyTuple_GET_SIZE(w) != 1) {
        return false;
    }
    PyObject* item = PyTuple_GET_ITEM(w, 0);
    if (!PyTuple_CheckExact(item) || PyTuple_GET_SIZE(item) != 3) {
        return false;
    }
    int64_t idx = -1, num = 0, den = 0;
    if (!parse_bounded_int(PyTuple_GET_ITEM(item, 0), expected_pos, expected_pos, idx) ||
        !parse_bounded_int(PyTuple_GET_ITEM(item, 1), 1, 1, num) ||
        !parse_bounded_int(PyTuple_GET_ITEM(item, 2), 1, 1, den)) {
        return false;
    }
    return true;
}

bool check_witness_matches_rat_row(PyObject* w, const SparseRatRow& expected) {
    if (!PyTuple_CheckExact(w) ||
        PyTuple_GET_SIZE(w) != static_cast<Py_ssize_t>(expected.size())) {
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        PyObject* item = PyTuple_GET_ITEM(w, static_cast<Py_ssize_t>(i));
        if (!PyTuple_CheckExact(item) || PyTuple_GET_SIZE(item) != 3) {
            return false;
        }
        int64_t idx = -1;
        int64_t exp_pos = expected[i].first;
        if (!parse_bounded_int(PyTuple_GET_ITEM(item, 0), exp_pos, exp_pos, idx)) {
            return false;
        }
        ExactInt num, den;
        if (!ExactInt::from_pylong_exact(PyTuple_GET_ITEM(item, 1), num) ||
            !ExactInt::from_pylong_exact(PyTuple_GET_ITEM(item, 2), den)) {
            return false;
        }
        if (num != expected[i].second.num || den != expected[i].second.den) {
            return false;
        }
    }
    return true;
}

bool validate_witnesses_cpp(
    PyObject* witnesses_obj,
    const GroupingData& gd,
    const TopologyInfo& topo,
    const std::vector<SparseIntRow>& pivot_rows,
    const std::vector<Py_ssize_t>& kept,
    const std::vector<Py_ssize_t>& place,
    std::vector<SparseIntRow>& parts
) {
    const int G = gd.G;
    if (!PyTuple_CheckExact(witnesses_obj) || PyTuple_GET_SIZE(witnesses_obj) != G) {
        return false;
    }
    const int64_t count = topo.count;
    const int64_t n_kept = static_cast<int64_t>(kept.size());

    // Split pivot_rows into col_rows (for kept columns) and dep_rhs (for dependent columns)
    std::vector<SparseIntRow> col_rows(static_cast<size_t>(count));
    std::vector<SparseIntRow> dep_rhs(static_cast<size_t>(count));
    for (Py_ssize_t p_ss : kept) {
        int64_t p = static_cast<int64_t>(p_ss);
        for (const auto& [c, val] : pivot_rows[static_cast<size_t>(p)]) {
            if (place[static_cast<size_t>(c)] >= 0) {
                col_rows[static_cast<size_t>(c)].emplace_back(p, val);
            } else {
                dep_rhs[static_cast<size_t>(c)].emplace_back(p, val);
            }
        }
    }

    std::vector<SparseRatRow> of_column(static_cast<size_t>(count));
    parts.assign(static_cast<size_t>(count + 1), SparseIntRow{});
    IntWorkspace lhs_ws(count);

    for (int k = 0; k < G; ++k) {
        PyObject* w_blk = PyTuple_GET_ITEM(witnesses_obj, k);
        int64_t nl = gd.n_levels[static_cast<size_t>(k)];
        if (!PyTuple_CheckExact(w_blk) || PyTuple_GET_SIZE(w_blk) != nl) {
            return false;
        }
        const auto& col_k = topo.columns[static_cast<size_t>(k)];
        for (int64_t level = 0; level < nl; ++level) {
            PyObject* w = PyTuple_GET_ITEM(w_blk, static_cast<Py_ssize_t>(level));
            Py_ssize_t c = col_k[static_cast<size_t>(level)];
            if (c < 0) {
                continue;
            }
            Py_ssize_t pos = place[static_cast<size_t>(c)];
            if (pos >= 0) {
                if (!check_unit_witness(w, static_cast<int64_t>(pos))) {
                    return false;
                }
                of_column[static_cast<size_t>(c)] = {{static_cast<int64_t>(pos), ExactRat(1)}};
                continue;
            }
            // Dependent column check
            if (!PyTuple_CheckExact(w)) {
                return false;
            }
            Py_ssize_t w_len = PyTuple_GET_SIZE(w);
            SparseRatRow rat;
            rat.reserve(static_cast<size_t>(w_len));
            ExactInt lcm_d(1);
            int64_t prev_idx = -1;
            for (Py_ssize_t m = 0; m < w_len; ++m) {
                PyObject* item = PyTuple_GET_ITEM(w, m);
                if (!PyTuple_CheckExact(item) || PyTuple_GET_SIZE(item) != 3) {
                    return false;
                }
                int64_t idx = -1;
                if (!parse_bounded_int(PyTuple_GET_ITEM(item, 0), prev_idx + 1, n_kept - 1, idx)) {
                    return false;
                }
                ExactInt num, den;
                if (!ExactInt::from_pylong_exact(PyTuple_GET_ITEM(item, 1), num) ||
                    !ExactInt::from_pylong_exact(PyTuple_GET_ITEM(item, 2), den)) {
                    return false;
                }
                if (num.is_zero() || den.sgn() <= 0 || !ExactInt::gcd(num, den).is_one()) {
                    return false;
                }
                prev_idx = idx;
                if (!den.is_one()) {
                    lcm_d = ExactInt::lcm(lcm_d, den);
                }
                ExactRat r;
                r.num = std::move(num);
                r.den = std::move(den);
                rat.emplace_back(idx, std::move(r));
            }
            // Compute scaled and verify R_J * scaled == lcm_d * R_c
            lhs_ws.all_zero_and_clear();
            for (const auto& [idx, r] : rat) {
                ExactInt s_val = r.den.is_one()
                                     ? (r.num * lcm_d)
                                     : (r.num * ExactInt::divexact(lcm_d, r.den));
                int64_t kept_col = static_cast<int64_t>(kept[static_cast<size_t>(idx)]);
                parts[static_cast<size_t>(kept_col)].emplace_back(static_cast<int64_t>(c), s_val);
                for (const auto& [p, r_val] : col_rows[static_cast<size_t>(kept_col)]) {
                    lhs_ws.add(p, r_val * s_val);
                }
            }
            for (const auto& [p, rhs_val] : dep_rhs[static_cast<size_t>(c)]) {
                lhs_ws.add(p, -(rhs_val * lcm_d));
            }
            if (!lhs_ws.all_zero_and_clear()) {
                return false;
            }
            parts[static_cast<size_t>(c)] = {{static_cast<int64_t>(c), -lcm_d}};
            of_column[static_cast<size_t>(c)] = std::move(rat);
        }
    }

    // Reference witnesses check
    std::vector<std::vector<int64_t>> heads_by_comp(static_cast<size_t>(topo.n_components));
    for (int64_t h = 0; h < gd.n_levels[0]; ++h) {
        heads_by_comp[static_cast<size_t>(topo.of_levels[0][static_cast<size_t>(h)])].push_back(h);
    }
    RatWorkspace ref_ws(n_kept);
    std::vector<std::vector<int64_t>> same_by_comp(static_cast<size_t>(topo.n_components));

    for (int k = 0; k < G; ++k) {
        int64_t nl = gd.n_levels[static_cast<size_t>(k)];
        const auto& col_k = topo.columns[static_cast<size_t>(k)];
        const auto& of_lvl_k = topo.of_levels[static_cast<size_t>(k)];

        bool has_ref = false;
        for (int64_t l = 0; l < nl; ++l) {
            if (col_k[static_cast<size_t>(l)] < 0) {
                has_ref = true;
                break;
            }
        }
        if (!has_ref) continue;

        for (auto& v : same_by_comp) v.clear();
        for (int64_t s = 0; s < nl; ++s) {
            same_by_comp[static_cast<size_t>(of_lvl_k[static_cast<size_t>(s)])].push_back(s);
        }

        PyObject* w_blk = PyTuple_GET_ITEM(witnesses_obj, k);
        for (int64_t level = 0; level < nl; ++level) {
            if (col_k[static_cast<size_t>(level)] >= 0) {
                continue;
            }
            int64_t comp = of_lvl_k[static_cast<size_t>(level)];
            ref_ws.clear();
            for (int64_t h : heads_by_comp[static_cast<size_t>(comp)]) {
                Py_ssize_t hc = topo.columns[0][static_cast<size_t>(h)];
                for (const auto& [p, val] : of_column[static_cast<size_t>(hc)]) {
                    ref_ws.add(p, val);
                }
            }
            for (int64_t s : same_by_comp[static_cast<size_t>(comp)]) {
                if (s == level) continue;
                Py_ssize_t sc = col_k[static_cast<size_t>(s)];
                for (const auto& [p, val] : of_column[static_cast<size_t>(sc)]) {
                    ref_ws.add(p, -val);
                }
            }
            SparseRatRow expected = ref_ws.extract_sorted();
            PyObject* w = PyTuple_GET_ITEM(w_blk, static_cast<Py_ssize_t>(level));
            if (!check_witness_matches_rat_row(w, expected)) {
                return false;
            }
        }
    }
    return true;
}

void update_current_reduced(
    int64_t q,
    int64_t now,
    std::vector<SparseRatRow>& reduced,
    const std::vector<uint8_t>& is_seen,
    std::vector<int64_t>& stamps,
    RatWorkspace& ws
) {
    std::vector<int64_t> stack = {q};
    while (!stack.empty()) {
        int64_t x = stack.back();
        auto& row = reduced[static_cast<size_t>(x)];
        if (row.empty() || stamps[static_cast<size_t>(x)] == now) {
            stack.pop_back();
            continue;
        }
        std::vector<int64_t> hits;
        std::vector<int64_t> stale;
        for (const auto& [h, val] : row) {
            if (is_seen[static_cast<size_t>(h)]) {
                hits.push_back(h);
                if (!reduced[static_cast<size_t>(h)].empty() &&
                    stamps[static_cast<size_t>(h)] != now) {
                    stale.push_back(h);
                }
            }
        }
        if (!stale.empty()) {
            stack.insert(stack.end(), stale.begin(), stale.end());
            continue;
        }
        if (!hits.empty()) {
            ws.clear();
            std::vector<std::pair<int64_t, ExactRat>> hit_factors;
            hit_factors.reserve(hits.size());
            for (auto& [c, val] : row) {
                if (is_seen[static_cast<size_t>(c)]) {
                    hit_factors.emplace_back(c, std::move(val));
                } else {
                    ws.add(c, val);
                }
            }
            for (const auto& [h, factor] : hit_factors) {
                for (const auto& [c, val] : reduced[static_cast<size_t>(h)]) {
                    ws.add(c, -(factor * val));
                }
            }
            row = ws.extract_sorted();
        }
        stamps[static_cast<size_t>(x)] = now;
        stack.pop_back();
    }
}

bool valid_elimination_cpp(
    const GroupingData& gd,
    const TopologyInfo& topo,
    const std::vector<SparseIntRow>& pivot_rows,
    const std::vector<uint8_t>& has_pivot,
    const std::vector<Py_ssize_t>& kept,
    const std::vector<Py_ssize_t>& place,
    const std::vector<SparseIntRow>& parts
) {
    const int G = gd.G;
    const int64_t count = topo.count;
    const size_t n_distinct = topo.distinct_rows.size();

    // 1. _spanned
    IntWorkspace span_ws(count);
    for (size_t r = 0; r < n_distinct; ++r) {
        int64_t row_idx = topo.distinct_rows[r];
        for (int k = 0; k < G; ++k) {
            Py_ssize_t code = gd.codes_bufs[static_cast<size_t>(k)].data[row_idx];
            Py_ssize_t c = topo.columns[static_cast<size_t>(k)][static_cast<size_t>(code)];
            size_t part_idx = (c < 0) ? static_cast<size_t>(count) : static_cast<size_t>(c);
            for (const auto& [d, val] : parts[part_idx]) {
                span_ws.add(d, val);
            }
        }
        if (!span_ws.all_zero_and_clear()) {
            return false;
        }
    }

    // 2. _ordered
    std::vector<SparseRatRow> reduced(static_cast<size_t>(count));
    std::vector<uint8_t> is_seen(static_cast<size_t>(count), 0);
    std::vector<int64_t> stamps(static_cast<size_t>(count), 0);
    int64_t num_seen = 0;

    RatWorkspace part_ws(count);
    RatWorkspace cur_ws(count);

    for (size_t r = 0; r < n_distinct; ++r) {
        int64_t row_idx = topo.distinct_rows[r];
        part_ws.clear();
        for (int k = 0; k < G; ++k) {
            Py_ssize_t code = gd.codes_bufs[static_cast<size_t>(k)].data[row_idx];
            Py_ssize_t c_ss = topo.columns[static_cast<size_t>(k)][static_cast<size_t>(code)];
            if (c_ss < 0 || place[static_cast<size_t>(c_ss)] < 0) {
                continue;
            }
            int64_t c = static_cast<int64_t>(c_ss);
            if (!is_seen[static_cast<size_t>(c)]) {
                part_ws.add(c, ExactRat(1));
            } else if (!reduced[static_cast<size_t>(c)].empty()) {
                if (stamps[static_cast<size_t>(c)] != num_seen) {
                    update_current_reduced(c, num_seen, reduced, is_seen, stamps, cur_ws);
                }
                for (const auto& [d, val] : reduced[static_cast<size_t>(c)]) {
                    part_ws.add(d, -val);
                }
            }
        }
        SparseRatRow part = part_ws.extract_sorted();
        if (!part.empty()) {
            int64_t p = part[0].first;
            if (!has_pivot[static_cast<size_t>(p)]) {
                return false;
            }
            const auto& piv_row = pivot_rows[static_cast<size_t>(p)];
            const ExactInt& lead = piv_row[0].second;
            const ExactRat& own = part[0].second;
            ExactRat lead_rat(lead);

            size_t part_idx = 0;
            SparseRatRow new_red;
            for (const auto& [c, val] : piv_row) {
                if (place[static_cast<size_t>(c)] < 0) {
                    continue;
                }
                if (part_idx >= part.size() || part[part_idx].first != c) {
                    return false;
                }
                if (part[part_idx].second * lead_rat != ExactRat(val) * own) {
                    return false;
                }
                ++part_idx;
                if (c != p) {
                    new_red.emplace_back(c, ExactRat(val, lead));
                }
            }
            if (part_idx != part.size()) {
                return false;
            }
            reduced[static_cast<size_t>(p)] = std::move(new_red);
            is_seen[static_cast<size_t>(p)] = 1;
            ++num_seen;
            stamps[static_cast<size_t>(p)] = num_seen;
        }
    }
    return num_seen == static_cast<int64_t>(kept.size());
}

// ---------------------------------------------------------------------------
// Python Entry Point 1: valid_from_buffers(held_tuple, codes_tuple, counts_tuple, grouping_id, content)
// Returns True / False (exact validation result), or None if buffers are ineligible.
// ---------------------------------------------------------------------------
PyObject* py_valid_from_buffers(PyObject* /*self*/, PyObject* args) {
    PyObject* held_obj = nullptr;
    PyObject* codes_tuple = nullptr;
    PyObject* counts_tuple = nullptr;
    PyObject* grouping_id_obj = nullptr;
    PyObject* content_obj = nullptr;
    if (!PyArg_ParseTuple(
            args,
            "OOOOO",
            &held_obj,
            &codes_tuple,
            &counts_tuple,
            &grouping_id_obj,
            &content_obj)) {
        return nullptr;
    }

    try {
        if (!PyTuple_Check(held_obj) || PyTuple_GET_SIZE(held_obj) != 11) {
            Py_RETURN_NONE;
        }
        if (!PyLong_CheckExact(grouping_id_obj) || !plain_content(content_obj)) {
            Py_RETURN_NONE;
        }
        GroupingData gd;
        if (!extract_grouping(codes_tuple, counts_tuple, gd)) {
            Py_RETURN_NONE;
        }

        // 1. _same_structure((held.grouping, held.content), (id(grouping), content))
        if (!same_structure(PyTuple_GET_ITEM(held_obj, 0), grouping_id_obj) ||
            !same_structure(PyTuple_GET_ITEM(held_obj, 1), content_obj)) {
            Py_RETURN_FALSE;
        }

        // 2. Compute topology (components, rank, references, columns, distinct)
        TopologyInfo topo;
        if (!compute_topology(gd, topo)) {
            Py_RETURN_NONE;
        }

        // 3. Check rows, rank, references, columns
        PyObject* held_rows = PyTuple_GET_ITEM(held_obj, 10);
        int64_t rows_val = -1;
        int64_t n_distinct = static_cast<int64_t>(topo.distinct_rows.size());
        if (!parse_bounded_int(held_rows, n_distinct, n_distinct, rows_val)) {
            Py_RETURN_FALSE;
        }
        PyObjRef exp_rank = build_rank_tuple(gd, topo);
        if (!same_structure(PyTuple_GET_ITEM(held_obj, 3), exp_rank.get())) {
            Py_RETURN_FALSE;
        }
        if (!check_bytes_tuple(PyTuple_GET_ITEM(held_obj, 4), topo.references) ||
            !check_bytes_tuple(PyTuple_GET_ITEM(held_obj, 5), topo.columns)) {
            Py_RETURN_FALSE;
        }

        // 4. _valid_kept
        PyObject* held_route = PyTuple_GET_ITEM(held_obj, 2);
        PyObject* held_echelon = PyTuple_GET_ITEM(held_obj, 9);
        std::vector<SparseIntRow> pivot_rows;
        std::vector<uint8_t> has_pivot;
        std::vector<Py_ssize_t> kept;

        if (topo.has_exact_rank && topo.exact_rank == topo.count) {
            if (!PyUnicode_CheckExact(held_route) ||
                PyUnicode_CompareWithASCIIString(held_route, "connectivity") != 0 ||
                !PyTuple_CheckExact(held_echelon) ||
                PyTuple_GET_SIZE(held_echelon) != 0) {
                Py_RETURN_FALSE;
            }
            pivot_rows.assign(static_cast<size_t>(topo.count), SparseIntRow{});
            has_pivot.assign(static_cast<size_t>(topo.count), 0);
            kept.resize(static_cast<size_t>(topo.count));
            std::iota(kept.begin(), kept.end(), Py_ssize_t{0});
        } else {
            if (!PyUnicode_CheckExact(held_route) ||
                PyUnicode_CompareWithASCIIString(held_route, "elimination") != 0) {
                Py_RETURN_FALSE;
            }
            if (!parse_echelon(held_echelon, topo.count, pivot_rows, has_pivot, kept)) {
                Py_RETURN_FALSE;
            }
        }

        int64_t n_kept = static_cast<int64_t>(kept.size());
        if (!(topo.rank_lower <= n_kept && n_kept <= topo.rank_upper)) {
            Py_RETURN_FALSE;
        }

        PyObject* held_selected = PyTuple_GET_ITEM(held_obj, 6);
        Py_ssize_t exp_sel_bytes = static_cast<Py_ssize_t>(kept.size() * sizeof(Py_ssize_t));
        if (!PyBytes_CheckExact(held_selected) ||
            PyBytes_GET_SIZE(held_selected) != exp_sel_bytes ||
            (exp_sel_bytes > 0 &&
             std::memcmp(PyBytes_AS_STRING(held_selected), kept.data(), static_cast<size_t>(exp_sel_bytes)) != 0)) {
            Py_RETURN_FALSE;
        }

        std::vector<Py_ssize_t> place(static_cast<size_t>(topo.count), -1);
        for (size_t i = 0; i < kept.size(); ++i) {
            place[static_cast<size_t>(kept[i])] = static_cast<Py_ssize_t>(i);
        }

        std::vector<std::vector<Py_ssize_t>> exp_positions(static_cast<size_t>(gd.G));
        for (int k = 0; k < gd.G; ++k) {
            const auto& col_k = topo.columns[static_cast<size_t>(k)];
            auto& pos_k = exp_positions[static_cast<size_t>(k)];
            pos_k.resize(col_k.size());
            for (size_t l = 0; l < col_k.size(); ++l) {
                Py_ssize_t c = col_k[l];
                pos_k[l] = (c >= 0) ? place[static_cast<size_t>(c)] : -1;
            }
        }
        if (!check_bytes_tuple(PyTuple_GET_ITEM(held_obj, 7), exp_positions)) {
            Py_RETURN_FALSE;
        }

        // 5. _valid_witnesses
        std::vector<SparseIntRow> parts;
        if (!validate_witnesses_cpp(
                PyTuple_GET_ITEM(held_obj, 8),
                gd,
                topo,
                pivot_rows,
                kept,
                place,
                parts)) {
            Py_RETURN_FALSE;
        }

        if (topo.has_exact_rank && topo.exact_rank == topo.count) {
            Py_RETURN_TRUE;
        }

        // 6. _valid_elimination (_spanned + _ordered)
        if (!valid_elimination_cpp(gd, topo, pivot_rows, has_pivot, kept, place, parts)) {
            Py_RETURN_FALSE;
        }
        Py_RETURN_TRUE;
    } catch (const PyError&) {
        return nullptr;
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

// ---------------------------------------------------------------------------
// Helpers to build Python objects for proposal output
// ---------------------------------------------------------------------------
PyObjRef make_bytes_tuple(const std::vector<std::vector<Py_ssize_t>>& vecs) {
    PyObjRef tup(PyTuple_New(static_cast<Py_ssize_t>(vecs.size())));
    if (!tup) throw PyError();
    for (size_t k = 0; k < vecs.size(); ++k) {
        PyObject* b = PyBytes_FromStringAndSize(
            reinterpret_cast<const char*>(vecs[k].data()),
            static_cast<Py_ssize_t>(vecs[k].size() * sizeof(Py_ssize_t)));
        if (!b) throw PyError();
        PyTuple_SET_ITEM(tup.get(), static_cast<Py_ssize_t>(k), b);
    }
    return tup;
}

PyObjRef make_witness_tuple(const SparseRatRow& row) {
    PyObjRef tup(PyTuple_New(static_cast<Py_ssize_t>(row.size())));
    if (!tup) throw PyError();
    for (size_t i = 0; i < row.size(); ++i) {
        PyObjRef item(PyTuple_New(3));
        if (!item) throw PyError();
        PyObjRef p(PyLong_FromLongLong(row[i].first));
        if (!p) throw PyError();
        PyObjRef num = row[i].second.num.to_pyobj();
        PyObjRef den = row[i].second.den.to_pyobj();
        PyTuple_SET_ITEM(item.get(), 0, p.release());
        PyTuple_SET_ITEM(item.get(), 1, num.release());
        PyTuple_SET_ITEM(item.get(), 2, den.release());
        PyTuple_SET_ITEM(tup.get(), static_cast<Py_ssize_t>(i), item.release());
    }
    return tup;
}

PyObjRef make_unit_witness_tuple(Py_ssize_t pos) {
    PyObjRef tup(PyTuple_New(1));
    if (!tup) throw PyError();
    PyObjRef item(PyTuple_New(3));
    if (!item) throw PyError();
    PyObject* p = PyLong_FromSsize_t(pos);
    if (!p) throw PyError();
    Py_INCREF(g_one_long);
    Py_INCREF(g_one_long);
    PyTuple_SET_ITEM(item.get(), 0, p);
    PyTuple_SET_ITEM(item.get(), 1, g_one_long);
    PyTuple_SET_ITEM(item.get(), 2, g_one_long);
    PyTuple_SET_ITEM(tup.get(), 0, item.release());
    return tup;
}

// ---------------------------------------------------------------------------
// Python Entry Point 2: proposal_from_buffers(codes_tuple, counts_tuple)
// Returns 9-tuple (route, rank_tuple, refs_bytes, cols_bytes, sel_bytes,
// pos_bytes, witnesses, echelon, rows), or False if proposal refused/budget,
// or None if buffers are ineligible.
// ---------------------------------------------------------------------------
PyObject* py_proposal_from_buffers(PyObject* /*self*/, PyObject* args) {
    PyObject* codes_tuple = nullptr;
    PyObject* counts_tuple = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &codes_tuple, &counts_tuple)) {
        return nullptr;
    }

    try {
        GroupingData gd;
        if (!extract_grouping(codes_tuple, counts_tuple, gd)) {
            Py_RETURN_NONE;
        }
        TopologyInfo topo;
        if (!compute_topology(gd, topo)) {
            Py_RETURN_NONE;
        }

        const int G = gd.G;
        const int64_t count = topo.count;
        const size_t n_distinct = topo.distinct_rows.size();

        // Check row components consistency
        for (int k = 1; k < G; ++k) {
            const Py_ssize_t* ck = gd.codes_bufs[static_cast<size_t>(k)].data;
            const auto& of_lvl_k = topo.of_levels[static_cast<size_t>(k)];
            for (int64_t i = 0; i < gd.n_rows; ++i) {
                if (of_lvl_k[static_cast<size_t>(ck[i])] != topo.of_rows[static_cast<size_t>(i)]) {
                    Py_RETURN_FALSE;
                }
            }
        }

        const char* route_str = nullptr;
        std::vector<SparseIntRow> pivot_rows(static_cast<size_t>(count));
        std::vector<uint8_t> has_pivot(static_cast<size_t>(count), 0);
        std::vector<Py_ssize_t> kept;
        std::vector<SparseRatRow> of_column(static_cast<size_t>(count));

        if (topo.has_exact_rank && topo.exact_rank == count) {
            route_str = "connectivity";
            kept.resize(static_cast<size_t>(count));
            std::iota(kept.begin(), kept.end(), Py_ssize_t{0});
            for (int64_t c = 0; c < count; ++c) {
                of_column[static_cast<size_t>(c)] = {{c, ExactRat(1)}};
            }
        } else {
            route_str = "elimination";
            const ExactInt budget = ExactInt(32) *
                (ExactInt(static_cast<int64_t>(n_distinct)) * ExactInt(G) + ExactInt(count));
            ExactInt work(0);

            std::vector<SparseRatRow> tail_rows(static_cast<size_t>(count));
            std::vector<uint8_t> has_tail(static_cast<size_t>(count), 0);
            std::vector<std::vector<int64_t>> holders(static_cast<size_t>(count));

            RatWorkspace found_ws(count);
            RatWorkspace other_ws(count);

            for (size_t r = 0; r < n_distinct; ++r) {
                int64_t row_idx = topo.distinct_rows[r];
                int64_t head = static_cast<int64_t>(
                    topo.columns[0][static_cast<size_t>(gd.codes_bufs[0].data[row_idx])]);

                found_ws.clear();
                bool head_is_piv = (has_pivot[static_cast<size_t>(head)] != 0);
                if (!head_is_piv) {
                    for (int k = 1; k < G; ++k) {
                        Py_ssize_t code = gd.codes_bufs[static_cast<size_t>(k)].data[row_idx];
                        Py_ssize_t c = topo.columns[static_cast<size_t>(k)][static_cast<size_t>(code)];
                        if (c >= 0 && !found_ws.in[static_cast<size_t>(c)]) {
                            found_ws.add(static_cast<int64_t>(c), ExactRat(1));
                        }
                    }
                } else {
                    const auto& u = pivot_rows[static_cast<size_t>(head)];
                    const ExactInt& u_head = u[0].second;
                    for (size_t j = 1; j < u.size(); ++j) {
                        found_ws.add(u[j].first, ExactRat(-u[j].second));
                    }
                    ExactRat u_head_rat(u_head);
                    for (int k = 1; k < G; ++k) {
                        Py_ssize_t code = gd.codes_bufs[static_cast<size_t>(k)].data[row_idx];
                        Py_ssize_t c = topo.columns[static_cast<size_t>(k)][static_cast<size_t>(code)];
                        if (c >= 0) {
                            found_ws.add(static_cast<int64_t>(c), u_head_rat);
                        }
                    }
                }

                // _reduce_by(found, tails)
                std::vector<std::pair<int64_t, ExactRat>> to_elim;
                for (int64_t c : found_ws.cols) {
                    size_t uc = static_cast<size_t>(c);
                    if (found_ws.in[uc] && has_tail[uc]) {
                        to_elim.emplace_back(c, std::move(found_ws.val[uc]));
                        found_ws.in[uc] = 0;
                    }
                }
                for (const auto& [q, factor] : to_elim) {
                    const auto& tq = tail_rows[static_cast<size_t>(q)];
                    work = work + ExactInt(static_cast<int64_t>(tq.size()));
                    for (const auto& [c, val] : tq) {
                        found_ws.add(c, -(factor * val));
                    }
                }

                if (!head_is_piv) {
                    found_ws.add(head, ExactRat(1));
                    SparseRatRow f_sorted = found_ws.extract_sorted();
                    pivot_rows[static_cast<size_t>(head)] = make_primitive(f_sorted);
                    has_pivot[static_cast<size_t>(head)] = 1;
                } else {
                    SparseRatRow f_sorted = found_ws.extract_sorted();
                    if (!f_sorted.empty()) {
                        int64_t p = f_sorted[0].first;
                        pivot_rows[static_cast<size_t>(p)] = make_primitive(f_sorted);
                        has_pivot[static_cast<size_t>(p)] = 1;

                        ExactRat lead = std::move(f_sorted[0].second);
                        SparseRatRow row;
                        row.reserve(f_sorted.size() - 1);
                        for (size_t j = 1; j < f_sorted.size(); ++j) {
                            row.emplace_back(f_sorted[j].first, f_sorted[j].second / lead);
                        }

                        // _new_tail(p, row, tails, holders)
                        std::vector<int64_t> qs = std::move(holders[static_cast<size_t>(p)]);
                        holders[static_cast<size_t>(p)].clear();
                        for (int64_t q : qs) {
                            auto& other = tail_rows[static_cast<size_t>(q)];
                            ExactRat factor(0);
                            bool found_p = false;
                            other_ws.clear();
                            for (auto& [c, val] : other) {
                                if (c == p) {
                                    factor = std::move(val);
                                    found_p = true;
                                } else {
                                    other_ws.add(c, val);
                                }
                            }
                            if (!found_p) continue;
                            work = work + ExactInt(static_cast<int64_t>(row.size()));
                            for (const auto& [c, val] : row) {
                                bool had = (other_ws.in[static_cast<size_t>(c)] != 0);
                                other_ws.add(c, -(factor * val));
                                bool now_in = (other_ws.in[static_cast<size_t>(c)] != 0);
                                if (now_in && !had) {
                                    holders[static_cast<size_t>(c)].push_back(q);
                                } else if (had && !now_in) {
                                    auto& hc = holders[static_cast<size_t>(c)];
                                    hc.erase(std::remove(hc.begin(), hc.end(), q), hc.end());
                                }
                            }
                            other = other_ws.extract_sorted();
                        }
                        for (const auto& [c, _] : row) {
                            holders[static_cast<size_t>(c)].push_back(p);
                        }
                        tail_rows[static_cast<size_t>(p)] = std::move(row);
                        has_tail[static_cast<size_t>(p)] = 1;
                    }
                }
                if ((work - budget).sgn() > 0) {
                    Py_RETURN_FALSE;
                }
            }

            for (int64_t c = 0; c < count; ++c) {
                if (has_pivot[static_cast<size_t>(c)]) {
                    kept.push_back(static_cast<Py_ssize_t>(c));
                }
            }
            int64_t n_kept = static_cast<int64_t>(kept.size());
            if (!(topo.rank_lower <= n_kept && n_kept <= topo.rank_upper)) {
                Py_RETURN_FALSE;
            }

            // _proposed_witnesses
            for (size_t i = 0; i < kept.size(); ++i) {
                int64_t p = static_cast<int64_t>(kept[i]);
                of_column[static_cast<size_t>(p)] = {{static_cast<int64_t>(i), ExactRat(1)}};
                SparseRatRow row;
                if (has_tail[static_cast<size_t>(p)]) {
                    row = tail_rows[static_cast<size_t>(p)];
                } else {
                    const auto& u = pivot_rows[static_cast<size_t>(p)];
                    const ExactInt& u_p = u[0].second;
                    found_ws.clear();
                    for (size_t j = 1; j < u.size(); ++j) {
                        int64_t c = u[j].first;
                        ExactRat coeff(u[j].second);
                        if (has_tail[static_cast<size_t>(c)]) {
                            for (const auto& [d, entry] : tail_rows[static_cast<size_t>(c)]) {
                                found_ws.add(d, -(coeff * entry));
                            }
                        } else {
                            found_ws.add(c, coeff);
                        }
                    }
                    SparseRatRow red = found_ws.extract_sorted();
                    ExactRat u_p_rat(u_p);
                    row.reserve(red.size());
                    for (auto& [c, val] : red) {
                        row.emplace_back(c, val / u_p_rat);
                    }
                }
                for (auto& [c, val] : row) {
                    of_column[static_cast<size_t>(c)].emplace_back(
                        static_cast<int64_t>(i), std::move(val));
                }
            }
        }

        int64_t n_kept = static_cast<int64_t>(kept.size());
        if (!(topo.rank_lower <= n_kept && n_kept <= topo.rank_upper)) {
            Py_RETURN_FALSE;
        }

        std::vector<Py_ssize_t> place(static_cast<size_t>(count), -1);
        for (size_t i = 0; i < kept.size(); ++i) {
            place[static_cast<size_t>(kept[i])] = static_cast<Py_ssize_t>(i);
        }

        std::vector<std::vector<Py_ssize_t>> positions_vecs(static_cast<size_t>(G));
        for (int k = 0; k < G; ++k) {
            const auto& col_k = topo.columns[static_cast<size_t>(k)];
            auto& pos_k = positions_vecs[static_cast<size_t>(k)];
            pos_k.resize(col_k.size());
            for (size_t l = 0; l < col_k.size(); ++l) {
                Py_ssize_t c = col_k[l];
                pos_k[l] = (c >= 0) ? place[static_cast<size_t>(c)] : -1;
            }
        }

        // Build witnesses tuple
        std::vector<std::vector<int64_t>> heads_by_comp(static_cast<size_t>(topo.n_components));
        for (int64_t h = 0; h < gd.n_levels[0]; ++h) {
            heads_by_comp[static_cast<size_t>(topo.of_levels[0][static_cast<size_t>(h)])].push_back(h);
        }
        std::vector<std::vector<int64_t>> same_by_comp(static_cast<size_t>(topo.n_components));
        RatWorkspace ref_ws(n_kept);

        PyObjRef witnesses_tup(PyTuple_New(G));
        if (!witnesses_tup) throw PyError();
        for (int k = 0; k < G; ++k) {
            int64_t nl = gd.n_levels[static_cast<size_t>(k)];
            const auto& col_k = topo.columns[static_cast<size_t>(k)];
            const auto& of_lvl_k = topo.of_levels[static_cast<size_t>(k)];

            if (k > 0) {
                for (auto& v : same_by_comp) v.clear();
                for (int64_t s = 0; s < nl; ++s) {
                    same_by_comp[static_cast<size_t>(of_lvl_k[static_cast<size_t>(s)])].push_back(s);
                }
            }

            PyObjRef w_blk(PyTuple_New(static_cast<Py_ssize_t>(nl)));
            if (!w_blk) throw PyError();
            for (int64_t level = 0; level < nl; ++level) {
                Py_ssize_t c = col_k[static_cast<size_t>(level)];
                PyObjRef w_item;
                if (c >= 0) {
                    Py_ssize_t pos = place[static_cast<size_t>(c)];
                    if (pos >= 0) {
                        w_item = make_unit_witness_tuple(pos);
                    } else {
                        w_item = make_witness_tuple(of_column[static_cast<size_t>(c)]);
                    }
                } else {
                    int64_t comp = of_lvl_k[static_cast<size_t>(level)];
                    ref_ws.clear();
                    for (int64_t h : heads_by_comp[static_cast<size_t>(comp)]) {
                        Py_ssize_t hc = topo.columns[0][static_cast<size_t>(h)];
                        for (const auto& [p, val] : of_column[static_cast<size_t>(hc)]) {
                            ref_ws.add(p, val);
                        }
                    }
                    for (int64_t s : same_by_comp[static_cast<size_t>(comp)]) {
                        if (s == level) continue;
                        Py_ssize_t sc = col_k[static_cast<size_t>(s)];
                        for (const auto& [p, val] : of_column[static_cast<size_t>(sc)]) {
                            ref_ws.add(p, -val);
                        }
                    }
                    SparseRatRow ref_row = ref_ws.extract_sorted();
                    w_item = make_witness_tuple(ref_row);
                }
                PyTuple_SET_ITEM(w_blk.get(), static_cast<Py_ssize_t>(level), w_item.release());
            }
            PyTuple_SET_ITEM(witnesses_tup.get(), k, w_blk.release());
        }

        // Build echelon tuple
        bool is_conn = (topo.has_exact_rank && topo.exact_rank == count);
        PyObjRef echelon_tup(PyTuple_New(is_conn ? 0 : static_cast<Py_ssize_t>(kept.size())));
        if (!echelon_tup) throw PyError();
        if (!is_conn) {
            for (size_t i = 0; i < kept.size(); ++i) {
                const auto& prow = pivot_rows[static_cast<size_t>(kept[i])];
                PyObjRef r_tup(PyTuple_New(static_cast<Py_ssize_t>(prow.size())));
                if (!r_tup) throw PyError();
                for (size_t j = 0; j < prow.size(); ++j) {
                    PyObjRef pair(PyTuple_New(2));
                    if (!pair) throw PyError();
                    PyObjRef pc(PyLong_FromLongLong(prow[j].first));
                    if (!pc) throw PyError();
                    PyObjRef pv = prow[j].second.to_pyobj();
                    PyTuple_SET_ITEM(pair.get(), 0, pc.release());
                    PyTuple_SET_ITEM(pair.get(), 1, pv.release());
                    PyTuple_SET_ITEM(r_tup.get(), static_cast<Py_ssize_t>(j), pair.release());
                }
                PyTuple_SET_ITEM(echelon_tup.get(), static_cast<Py_ssize_t>(i), r_tup.release());
            }
        }

        // Build 9-tuple result
        PyObjRef route_obj(PyUnicode_FromString(route_str));
        PyObjRef rank_tup = build_rank_tuple(gd, topo);
        PyObjRef refs_bytes = make_bytes_tuple(topo.references);
        PyObjRef cols_bytes = make_bytes_tuple(topo.columns);
        PyObjRef sel_bytes(PyBytes_FromStringAndSize(
            reinterpret_cast<const char*>(kept.data()),
            static_cast<Py_ssize_t>(kept.size() * sizeof(Py_ssize_t))));
        PyObjRef pos_bytes = make_bytes_tuple(positions_vecs);
        PyObjRef rows_obj(PyLong_FromSize_t(n_distinct));
        if (!route_obj || !sel_bytes || !rows_obj) throw PyError();

        PyObjRef out(PyTuple_New(9));
        if (!out) throw PyError();
        PyTuple_SET_ITEM(out.get(), 0, route_obj.release());
        PyTuple_SET_ITEM(out.get(), 1, rank_tup.release());
        PyTuple_SET_ITEM(out.get(), 2, refs_bytes.release());
        PyTuple_SET_ITEM(out.get(), 3, cols_bytes.release());
        PyTuple_SET_ITEM(out.get(), 4, sel_bytes.release());
        PyTuple_SET_ITEM(out.get(), 5, pos_bytes.release());
        PyTuple_SET_ITEM(out.get(), 6, witnesses_tup.release());
        PyTuple_SET_ITEM(out.get(), 7, echelon_tup.release());
        PyTuple_SET_ITEM(out.get(), 8, rows_obj.release());
        return out.release();
    } catch (const PyError&) {
        return nullptr;
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

PyMethodDef g_methods[] = {
    {"valid_from_buffers",
     py_valid_from_buffers,
     METH_VARARGS,
     "Validate held _Held tuple against grouping code/count buffers in C++."},
    {"proposal_from_buffers",
     py_proposal_from_buffers,
     METH_VARARGS,
     "Construct proposed CertificateBasis fields from grouping code/count buffers in C++."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef g_module = {
    PyModuleDef_HEAD_INIT,
    "basis_lifecycle",
    "Exact C++17 fresh-validation and proposal lifecycle for certificate_basis.",
    -1,
    g_methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit_basis_lifecycle(void) {
    PyObjRef np_mod(PyImport_ImportModule("numpy"));
    if (!np_mod) return nullptr;
    PyObjRef ndarray_t(PyObject_GetAttrString(np_mod.get(), "ndarray"));
    if (!ndarray_t || !PyType_Check(ndarray_t.get())) return nullptr;
    PyObjRef intp_t(PyObject_GetAttrString(np_mod.get(), "intp"));
    if (!intp_t) return nullptr;
    PyObjRef intp_dt(PyObject_CallMethod(np_mod.get(), "dtype", "O", intp_t.get()));
    if (!intp_dt) return nullptr;

    PyObjRef one_l(PyLong_FromLong(1));
    if (!one_l) return nullptr;

    PyObject* m = PyModule_Create(&g_module);
    if (!m) return nullptr;

    g_ndarray_type = reinterpret_cast<PyTypeObject*>(ndarray_t.release());
    g_intp_dtype = intp_dt.release();
    g_one_long = one_l.release();
    return m;
}
