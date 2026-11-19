// Compiled port of the per-row loop of powermean.fe.certificate_basis._operational_echelon.
//
// The Python loop is the specification.  This kernel performs the SAME exact
// operations (_add / _reduce_by / _new_tail / _over) in exact rational arithmetic
// (lowest terms, positive denominators, as fractions.Fraction) on 64-bit numerators
// and denominators with 128-bit intermediates where the compiler offers them, and
// counts the SAME ``work``.  Whenever a value cannot be represented in 64 bits, or
// the budget is exceeded, the kernel declines by returning None and the caller runs
// the unchanged Python loop, so no result of the package ever depends on this
// extension.  The insertion order of a sparse row (a dict in Python) never enters a
// result: every elimination of _reduce_by subtracts a tail without pivot columns,
// so the eliminated factors do not depend on the order, and _new_tail updates each
// holder independently.
//
// ``tails_from_mapped`` is the fit-path producer: the first source row of every head
// and the reduced tails of the later pivots (the final RREF of the later-block
// difference rows).  ``audit_from_mapped`` is the same loop over every row, with the
// canonical echelon row of each pivot emitted at its creation: for a head, the
// anchor row reduced by the later pivots existing at that moment; for a later pivot,
// the reduced difference row.  Both are the primitive integer rows with a positive
// leading entry that the canonical kernel (certificate_basis._reduced) produces,
// because both are the unique element of (row + span of the rows seen before) that
// vanishes on the pivot columns seen before.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

// The caller runs the Python loop instead; nothing of this object survives.
struct Decline {};

constexpr std::int64_t I64_MAX = std::numeric_limits<std::int64_t>::max();
constexpr std::int64_t I64_MIN = std::numeric_limits<std::int64_t>::min();

inline std::int64_t checked_add(std::int64_t a, std::int64_t b) {
    if ((b > 0 && a > I64_MAX - b) || (b < 0 && a < I64_MIN - b)) throw Decline();
    return a + b;
}

inline std::int64_t checked_neg(std::int64_t a) {
    if (a == I64_MIN) throw Decline();
    return -a;
}

inline std::int64_t checked_mul(std::int64_t a, std::int64_t b) {
    if (a == 0 || b == 0) return 0;
    if (a > 0) {
        if (b > 0) {
            if (a > I64_MAX / b) throw Decline();
        } else if (b < I64_MIN / a) {
            throw Decline();
        }
    } else if (b > 0) {
        if (a < I64_MIN / b) throw Decline();
    } else if (a == I64_MIN || b == I64_MIN || -a > I64_MAX / -b) {
        throw Decline();
    }
    return a * b;
}

inline std::int64_t positive_gcd(std::int64_t a, std::int64_t b) {
    if (a == I64_MIN || b == I64_MIN) throw Decline();
    return std::gcd(a < 0 ? -a : a, b < 0 ? -b : b);
}

// An exact rational in lowest terms with a positive denominator (Fraction's form).
struct Q {
    std::int64_t n;
    std::int64_t d;
};

#if defined(__SIZEOF_INT128__)
using Wide = __int128;
using UWide = unsigned __int128;

inline UWide wide_gcd(UWide a, UWide b) {
    while (b != 0) {
        const UWide r = a % b;
        a = b;
        b = r;
    }
    return a;
}

// n / d in lowest terms; declines unless both fit 64 bits
inline Q normalized_wide(Wide n, Wide d) {
    if (d < 0) {
        n = -n;
        d = -d;
    }
    if (n == 0) return {0, 1};
    const UWide g = wide_gcd(n < 0 ? static_cast<UWide>(-n) : static_cast<UWide>(n),
                             static_cast<UWide>(d));
    n /= static_cast<Wide>(g);
    d /= static_cast<Wide>(g);
    if (n > static_cast<Wide>(I64_MAX) || n < static_cast<Wide>(I64_MIN)
            || d > static_cast<Wide>(I64_MAX)) {
        throw Decline();
    }
    return {static_cast<std::int64_t>(n), static_cast<std::int64_t>(d)};
}

inline Q q_add(Q a, Q b) {
    if (a.d == 1 && b.d == 1) return {checked_add(a.n, b.n), 1};
    return normalized_wide(static_cast<Wide>(a.n) * b.d + static_cast<Wide>(b.n) * a.d,
                           static_cast<Wide>(a.d) * b.d);
}

inline Q q_mul(Q a, Q b) {
    if (a.d == 1 && b.d == 1) return {checked_mul(a.n, b.n), 1};
    return normalized_wide(static_cast<Wide>(a.n) * b.n, static_cast<Wide>(a.d) * b.d);
}

// value / lead exactly (_over): an integer when it divides, else the fraction.
inline Q q_div(Q a, Q b) {
    return normalized_wide(static_cast<Wide>(a.n) * b.d, static_cast<Wide>(a.d) * b.n);
}
#else
inline Q normalized(std::int64_t n, std::int64_t d) {
    if (d < 0) {
        n = checked_neg(n);
        d = checked_neg(d);
    }
    if (n == 0) return {0, 1};
    const std::int64_t g = positive_gcd(n, d);
    return {n / g, d / g};
}

inline Q q_add(Q a, Q b) {
    if (a.d == 1 && b.d == 1) return {checked_add(a.n, b.n), 1};
    return normalized(checked_add(checked_mul(a.n, b.d), checked_mul(b.n, a.d)),
                      checked_mul(a.d, b.d));
}

inline Q q_mul(Q a, Q b) {
    if (a.d == 1 && b.d == 1) return {checked_mul(a.n, b.n), 1};
    return normalized(checked_mul(a.n, b.n), checked_mul(a.d, b.d));
}

inline Q q_div(Q a, Q b) {
    return normalized(checked_mul(a.n, b.d), checked_mul(a.d, b.n));
}
#endif

inline Q q_neg(Q a) { return {checked_neg(a.n), a.d}; }

using Entry = std::pair<std::int64_t, Q>;
using Row = std::vector<Entry>;  // a sparse row: column -> nonzero value (dict semantics)

// O(1) column lookup for ONE loaded row at a time through a column-indexed table.
struct Scratch {
    std::vector<std::int32_t> pos;  // column -> index in the loaded row, -1 elsewhere

    explicit Scratch(std::int64_t width) : pos(static_cast<std::size_t>(width), -1) {}

    void load(const Row& row) {
        for (std::size_t i = 0; i < row.size(); ++i) {
            pos[static_cast<std::size_t>(row[i].first)] = static_cast<std::int32_t>(i);
        }
    }

    void unload(const Row& row) {
        for (const auto& e : row) pos[static_cast<std::size_t>(e.first)] = -1;
    }

    bool has(std::int64_t column) const { return pos[static_cast<std::size_t>(column)] >= 0; }

    void erase_at(Row& row, std::int32_t i) {
        pos[static_cast<std::size_t>(row[static_cast<std::size_t>(i)].first)] = -1;
        const std::size_t last = row.size() - 1;
        if (static_cast<std::size_t>(i) != last) {
            row[static_cast<std::size_t>(i)] = row[last];
            pos[static_cast<std::size_t>(row[static_cast<std::size_t>(i)].first)] = i;
        }
        row.pop_back();
    }

    // _add: row[column] += value, dropping an exact zero.
    void add(Row& row, std::int64_t column, Q value) {
        const std::int32_t i = pos[static_cast<std::size_t>(column)];
        if (i < 0) {
            if (value.n != 0) {
                pos[static_cast<std::size_t>(column)] = static_cast<std::int32_t>(row.size());
                row.emplace_back(column, value);
            }
            return;
        }
        const Q total = q_add(row[static_cast<std::size_t>(i)].second, value);
        if (total.n != 0) {
            row[static_cast<std::size_t>(i)].second = total;
        } else {
            erase_at(row, i);
        }
    }

    // dict.pop(column): the value, which must be present.
    Q pop(Row& row, std::int64_t column) {
        const std::int32_t i = pos[static_cast<std::size_t>(column)];
        if (i < 0) throw Decline();  // never reached: a present pivot was found
        const Q value = row[static_cast<std::size_t>(i)].second;
        erase_at(row, i);
        return value;
    }
};

struct State {
    std::int64_t width;
    std::vector<char> has_tail;
    std::vector<Row> tails;                          // indexed by pivot column
    std::vector<std::unordered_set<std::int64_t>> holders;  // column -> pivots holding it
    Scratch scratch;
    explicit State(std::int64_t w)
        : width(w), has_tail(static_cast<std::size_t>(w), 0),
          tails(static_cast<std::size_t>(w)), holders(static_cast<std::size_t>(w)), scratch(w) {}
};

// _reduce_by: eliminate every later pivot present in ``found`` (loaded); the work done.
std::int64_t reduce_by(Row& found, State& st, std::vector<std::int64_t>& present) {
    std::int64_t work = 0;
    present.clear();
    for (const auto& e : found) {
        if (st.has_tail[static_cast<std::size_t>(e.first)]) present.push_back(e.first);
    }
    for (const std::int64_t q : present) {
        const Q minus = q_neg(st.scratch.pop(found, q));
        const Row& tail = st.tails[static_cast<std::size_t>(q)];
        work = checked_add(work, static_cast<std::int64_t>(tail.size()));
        for (const auto& e : tail) st.scratch.add(found, e.first, q_mul(minus, e.second));
    }
    return work;
}

// _new_tail: store the reduced row of pivot ``p``; eliminate ``p`` from every other
// reduced row (the holders index them by column); the work done.  Nothing is loaded.
std::int64_t new_tail(std::int64_t p, Row row, State& st) {
    std::int64_t work = 0;
    std::unordered_set<std::int64_t> held;
    held.swap(st.holders[static_cast<std::size_t>(p)]);
    for (const std::int64_t q : held) {
        Row& other = st.tails[static_cast<std::size_t>(q)];
        st.scratch.load(other);
        const Q minus = q_neg(st.scratch.pop(other, p));
        work = checked_add(work, static_cast<std::int64_t>(row.size()));
        for (const auto& e : row) {
            const bool had = st.scratch.has(e.first);
            st.scratch.add(other, e.first, q_mul(minus, e.second));
            const bool has = st.scratch.has(e.first);
            auto& hold = st.holders[static_cast<std::size_t>(e.first)];
            if (has && !had) {
                hold.insert(q);
            } else if (had && !has) {
                hold.erase(q);
            }
        }
        st.scratch.unload(other);
    }
    for (const auto& e : row) st.holders[static_cast<std::size_t>(e.first)].insert(p);
    st.tails[static_cast<std::size_t>(p)] = std::move(row);
    st.has_tail[static_cast<std::size_t>(p)] = 1;
    return work;
}

using IntRow = std::vector<std::pair<std::int64_t, std::int64_t>>;

// The primitive integer row proportional to ``row`` with a positive entry at
// ``leading`` (its smallest column), sorted by column: certificate_basis._primitive.
IntRow primitive(const Row& row, std::int64_t leading) {
    std::int64_t scale = 1;
    for (const auto& e : row) {
        if (e.second.d != 1) scale = checked_mul(scale / std::gcd(scale, e.second.d), e.second.d);
    }
    IntRow out;
    out.reserve(row.size());
    std::int64_t divisor = 0;
    std::int64_t lead_value = 0;
    for (const auto& e : row) {
        const std::int64_t value = checked_mul(e.second.n, scale / e.second.d);
        out.emplace_back(e.first, value);
        divisor = positive_gcd(divisor, value);
        if (e.first == leading) lead_value = value;
    }
    if (lead_value == 0) throw Decline();  // never reached: the leading entry is nonzero
    if (lead_value < 0) divisor = checked_neg(divisor);
    for (auto& e : out) e.second /= divisor;
    std::sort(out.begin(), out.end());
    return out;
}

struct Inputs {
    std::vector<py::array_t<std::int64_t>> arrays;  // keep the buffers alive
    std::vector<const std::int64_t*> cols;
    py::ssize_t n = 0;
    std::int64_t width = 0;
};

Inputs read_mapped(const py::tuple& mapped, std::int64_t count) {
    if (mapped.size() < 1) throw py::type_error("mapped must be a nonempty tuple of intp arrays");
    if (count < 0) throw py::value_error("count must be nonnegative");
    Inputs in;
    if (count > std::numeric_limits<std::int32_t>::max()) throw Decline{};
    in.width = count;
    for (const auto& item : mapped) {
        auto array = py::cast<py::array>(item);
        if (array.dtype().kind() != 'i') {
            throw py::type_error("each mapped array must have a signed integer dtype");
        }
        py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> values(array);
        if (values.ndim() != 1) throw py::type_error("each mapped array must be one-dimensional");
        if (in.arrays.empty()) {
            in.n = values.shape(0);
        } else if (values.shape(0) != in.n) {
            throw py::value_error("mapped arrays differ in length");
        }
        const std::int64_t* data = values.data();
        for (py::ssize_t i = 0; i < in.n; ++i) {
            if (data[i] >= std::numeric_limits<std::int32_t>::max()) throw Decline{};
            if (data[i] >= in.width) in.width = data[i] + 1;
        }
        in.cols.push_back(data);
        in.arrays.push_back(std::move(values));
    }
    if (in.width > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw py::value_error("too many incidence columns for the compiled loop");
    }
    return in;
}

struct Outcome {
    bool ok = false;
    const char* reason = "invalid_head";
    std::int64_t work = 0;
    std::vector<std::int64_t> first_of;
    State st{0};
    // audit mode: the canonical rows at creation
    std::vector<std::int64_t> head_offset;  // count + 1 offsets into head_cols/head_vals
    std::vector<std::int64_t> head_cols, head_vals;
    std::vector<IntRow> later_rows;  // by pivot column
};

// The loop itself; ``budget < 0`` means unbounded (audit mode emits the canonical rows).
Outcome run(const Inputs& in, std::int64_t count, std::int64_t budget, bool audit) {
    Outcome out;
    out.first_of.assign(static_cast<std::size_t>(count), -1);
    out.st = State(in.width);
    std::vector<std::int64_t> head_begin, head_length;
    if (audit) {
        out.later_rows.resize(static_cast<std::size_t>(in.width));
        head_begin.assign(static_cast<std::size_t>(count), 0);
        head_length.assign(static_cast<std::size_t>(count), 0);
    }
    const std::size_t G = in.cols.size();
    Scratch& scratch = out.st.scratch;
    Row found, tail, head_row;
    std::vector<std::int64_t> later, anchor, present;
    try {
        for (py::ssize_t i = 0; i < in.n; ++i) {
            const std::int64_t head = in.cols[0][i];
            if (head < 0 || head >= count) return out;  // Python returns False
            later.clear();
            for (std::size_t k = 1; k < G; ++k) {
                if (in.cols[k][i] >= 0) later.push_back(in.cols[k][i]);
            }
            const std::int64_t previous = out.first_of[static_cast<std::size_t>(head)];
            out.work = checked_add(out.work, static_cast<std::int64_t>(later.size()));
            if (previous < 0) {
                out.first_of[static_cast<std::size_t>(head)] = static_cast<std::int64_t>(i);
                if (audit) {
                    // the anchor row, reduced by the later pivots existing now
                    head_row.clear();
                    scratch.add(head_row, head, Q{1, 1});
                    for (const std::int64_t q : later) {
                        if (out.st.has_tail[static_cast<std::size_t>(q)]) {
                            for (const auto& e : out.st.tails[static_cast<std::size_t>(q)]) {
                                scratch.add(head_row, e.first, q_neg(e.second));
                            }
                        } else {
                            scratch.add(head_row, q, Q{1, 1});
                        }
                    }
                    scratch.unload(head_row);
                    const IntRow canonical = primitive(head_row, head);
                    head_begin[static_cast<std::size_t>(head)] =
                        static_cast<std::int64_t>(out.head_cols.size());
                    head_length[static_cast<std::size_t>(head)] =
                        static_cast<std::int64_t>(canonical.size());
                    for (const auto& e : canonical) {
                        out.head_cols.push_back(e.first);
                        out.head_vals.push_back(e.second);
                    }
                }
            } else {
                anchor.clear();
                for (std::size_t k = 1; k < G; ++k) {
                    if (in.cols[k][previous] >= 0) anchor.push_back(in.cols[k][previous]);
                }
                found.clear();
                for (const std::int64_t c : later) {
                    if (!scratch.has(c)) scratch.add(found, c, Q{1, 1});  // dict.fromkeys
                }
                for (const std::int64_t c : anchor) scratch.add(found, c, Q{-1, 1});
                out.work = checked_add(out.work, checked_add(static_cast<std::int64_t>(anchor.size()),
                    reduce_by(found, out.st, present)));
                if (!found.empty()) {
                    std::int64_t pivot = found[0].first;
                    for (const auto& e : found) pivot = std::min(pivot, e.first);
                    if (audit) out.later_rows[static_cast<std::size_t>(pivot)] = primitive(found, pivot);
                    const Q lead = scratch.pop(found, pivot);
                    tail.clear();
                    for (const auto& e : found) tail.emplace_back(e.first, q_div(e.second, lead));
                    scratch.unload(found);
                    out.work = checked_add(out.work, checked_add(static_cast<std::int64_t>(found.size()),
                        new_tail(pivot, tail, out.st)));
                } else {
                    scratch.unload(found);
                }
            }
            if (budget >= 0 && out.work > budget) { out.reason = "work_limit"; return out; }  // Python returns None
        }
    } catch (const Decline&) {
        out.reason = "arithmetic_range";
        return out;
    }
    for (const std::int64_t f : out.first_of) {
        if (f < 0) { out.reason = "missing_head"; return out; }  // Python returns False
    }
    if (audit) {
        // rows were stored in creation order; the output follows head order
        out.head_offset.assign(static_cast<std::size_t>(count) + 1, 0);
        std::vector<std::int64_t> cols, vals;
        cols.reserve(out.head_cols.size());
        vals.reserve(out.head_vals.size());
        for (std::int64_t h = 0; h < count; ++h) {
            const auto begin = static_cast<std::size_t>(head_begin[static_cast<std::size_t>(h)]);
            const auto length = static_cast<std::size_t>(head_length[static_cast<std::size_t>(h)]);
            out.head_offset[static_cast<std::size_t>(h) + 1] =
                out.head_offset[static_cast<std::size_t>(h)] + static_cast<std::int64_t>(length);
            for (std::size_t j = begin; j < begin + length; ++j) {
                cols.push_back(out.head_cols[j]);
                vals.push_back(out.head_vals[j]);
            }
        }
        out.head_cols.swap(cols);
        out.head_vals.swap(vals);
    }
    out.ok = true;
    return out;
}

py::object q_to_python(const Q& value, const py::object& fraction) {
    if (value.d == 1) return py::int_(value.n);
    return fraction(py::int_(value.n), py::int_(value.d));
}

py::array_t<std::int64_t> first_of_array(const std::vector<std::int64_t>& first_of) {
    py::array_t<std::int64_t> array(static_cast<py::ssize_t>(first_of.size()));
    std::copy(first_of.begin(), first_of.end(), array.mutable_data());
    return array;
}

// [(pivot, {column: value}), ...] in increasing pivot order
py::list pivot_list(const State& st) {
    py::object fraction = py::module_::import("fractions").attr("Fraction");
    py::list pivots;
    for (std::int64_t p = 0; p < st.width; ++p) {
        if (!st.has_tail[static_cast<std::size_t>(p)]) continue;
        py::dict tail;
        for (const auto& e : st.tails[static_cast<std::size_t>(p)]) {
            tail[py::int_(e.first)] = q_to_python(e.second, fraction);
        }
        pivots.append(py::make_tuple(py::int_(p), std::move(tail)));
    }
    return pivots;
}

py::tuple int_row(const std::int64_t* cols, const std::int64_t* vals, std::size_t length) {
    py::tuple row(static_cast<py::ssize_t>(length));
    for (std::size_t j = 0; j < length; ++j) {
        PyTuple_SET_ITEM(row.ptr(), static_cast<py::ssize_t>(j),
                         py::make_tuple(py::int_(cols[j]), py::int_(vals[j])).release().ptr());
    }
    return row;
}

py::tuple attempt_from_mapped(const py::tuple& mapped, std::int64_t count, std::int64_t budget) {
    Inputs in;
    try {
        in = read_mapped(mapped, count);
    } catch (const Decline&) {
        return py::make_tuple(py::none(), py::none(), "column_capacity");
    }
    Outcome out;
    {
        py::gil_scoped_release release;
        out = run(in, count, budget < 0 ? -1 : budget, false);
    }
    if (!out.ok) return py::make_tuple(py::none(), py::int_(out.work), out.reason);
    return py::make_tuple(
        py::make_tuple(first_of_array(out.first_of), pivot_list(out.st), py::int_(out.work)),
        py::int_(out.work), py::none());
}

py::object tails_from_mapped(const py::tuple& mapped, std::int64_t count, std::int64_t budget) {
    return attempt_from_mapped(mapped, count, budget)[0];
}

py::object audit_from_mapped(const py::tuple& mapped, std::int64_t count) {
    Inputs in;
    try { in = read_mapped(mapped, count); }
    catch (const Decline&) { return py::none(); }
    Outcome out;
    {
        py::gil_scoped_release release;
        out = run(in, count, -1, true);
    }
    if (!out.ok) return py::none();
    // the canonical echelon: head rows in head order, then later rows by pivot column
    std::size_t later_count = 0;
    for (std::int64_t p = 0; p < out.st.width; ++p) {
        if (out.st.has_tail[static_cast<std::size_t>(p)]) ++later_count;
    }
    py::tuple echelon(static_cast<py::ssize_t>(count) + static_cast<py::ssize_t>(later_count));
    py::ssize_t at = 0;
    for (std::int64_t h = 0; h < count; ++h) {
        const auto begin = static_cast<std::size_t>(out.head_offset[static_cast<std::size_t>(h)]);
        const auto end = static_cast<std::size_t>(out.head_offset[static_cast<std::size_t>(h) + 1]);
        PyTuple_SET_ITEM(echelon.ptr(), at++,
                         int_row(out.head_cols.data() + begin, out.head_vals.data() + begin,
                                 end - begin).release().ptr());
    }
    std::vector<std::int64_t> cols, vals;
    for (std::int64_t p = 0; p < out.st.width; ++p) {
        if (!out.st.has_tail[static_cast<std::size_t>(p)]) continue;
        const IntRow& stored = out.later_rows[static_cast<std::size_t>(p)];
        cols.clear();
        vals.clear();
        for (const auto& e : stored) {
            cols.push_back(e.first);
            vals.push_back(e.second);
        }
        PyTuple_SET_ITEM(echelon.ptr(), at++,
                         int_row(cols.data(), vals.data(), stored.size()).release().ptr());
    }
    return py::make_tuple(first_of_array(out.first_of), pivot_list(out.st), std::move(echelon));
}

py::dict backend_info() {
    py::dict out;
    out["kernel"] = "operational_echelon";
#if defined(__SIZEOF_INT128__)
    out["arithmetic"] = "exact rationals: int64 values, int128 intermediates; declines past 64 bits";
#else
    out["arithmetic"] = "exact rationals: checked int64 values and intermediates; declines on overflow";
#endif
    out["python_fallback"] = "powermean.fe.certificate_basis._operational_echelon";
    out["input_ownership"] = "read_only_views";
    out["gil_policy"] = "release_around_core_loop";
    return out;
}

}  // namespace

PYBIND11_MODULE(_echelon_cpp, m) {
    m.doc() = "Compiled exact row reduction of the FE incidence (optional; Python fallback).";
    m.def("backend_info", &backend_info);
    m.def("attempt_from_mapped", &attempt_from_mapped, py::arg("mapped"), py::arg("count"),
          py::arg("budget"), "One attempt: (result or None, observed work or None, decline reason or None).");
    m.def("tails_from_mapped", &tails_from_mapped, py::arg("mapped"), py::arg("count"),
          py::arg("budget"),
          "tails_from_mapped(mapped, count, budget) -> (first_of, [(pivot, tail), ...], work) "
          "or None when the kernel declines (overflow, over budget, head out of range).");
    m.def("audit_from_mapped", &audit_from_mapped, py::arg("mapped"), py::arg("count"),
          "audit_from_mapped(mapped, count) -> (first_of, [(pivot, tail), ...], echelon) "
          "or None when the kernel declines.");
}
