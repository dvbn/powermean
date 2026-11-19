// Exact fresh type-identity label scans and the bound routes of the private label-check
// adapter.  CPython 3.12 C API, standard C++17, GIL held throughout.
//
// One scan per Python reference, each with that reference's own predicate:
//   one_exact_type(items, types)  powermean.results.retained_labels._one_exact_type
//   plain_kind(values)            powermean.results.retained_input_values._plain_kind
//   copy_free_labels(items)       powermean.results.retained_labels._copy_free_labels
// Types are compared by pointer identity (Py_TYPE), so no method of an item, of its
// class or of a metaclass runs, and a subclass is not its base.  Only exact list and
// tuple storage is read, through its own item array, which no Python code can resize
// during a scan.  Values are read only for exact floats (std::isfinite of the stored
// double, as math.isfinite reads it).  No Python code runs during a scan, and nothing
// is remembered between calls.
//
// A route (make_route) is the callable that the adapter binds in place of one reference.
// Being a C function, it adds no Python frame.  On every call it checks afresh that each
// binding its reference reads is still the expected original, scans, and only then
// returns the scan's answer, after entering, from its caller's own depth, the Python
// frames that the reference enters to reach that answer (counted by the scan; the frame
// notes are in label_adapter.py), through the adapter's ``enter`` function.  When the
// recursion limit stops those frames, when a binding is not the original, or for a call
// or input outside the scans, it calls the unchanged reference from its caller's depth
// with the same arguments, so the reference's own callbacks, result and exceptions,
// RecursionError included, are those of the call.  Every call appends the record of the
// implementation that ran to the activation's log.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cmath>

#ifdef Py_GIL_DISABLED
#error "label_checks reads list and tuple storage under the GIL; no free-threaded build"
#endif
#ifndef LABEL_CHECKS_SOURCE_SHA256
#error "build with LABEL_CHECKS_SOURCE_SHA256 set to this file's sha256 (tools/label_checks_build.py)"
#endif

namespace {

constexpr const char* kKernel = "powermean.label_checks/2";
// Exact tuples are followed this deep below the scanned tuple: labels that are tuples,
// and tuples inside those.  A deeper tuple is left to the recursive Python reference,
// so the C stack is bounded and that reference keeps its own recursion.
constexpr int kMaxNesting = 2;

// powermean.data.raw._IMMUTABLE_TYPES: int, float, complex, bool, str, bytes, None.
inline bool immutable_type(const PyTypeObject* t) noexcept {
    return t == &PyLong_Type || t == &PyFloat_Type || t == &PyComplex_Type ||
           t == &PyBool_Type || t == &PyUnicode_Type || t == &PyBytes_Type ||
           t == Py_TYPE(Py_None);
}

// powermean.results.retained_input_values._PLAIN_TYPES: bool, int, float, str.
inline bool plain_type(const PyTypeObject* t) noexcept {
    return t == &PyBool_Type || t == &PyLong_Type || t == &PyFloat_Type ||
           t == &PyUnicode_Type;
}

// The size and item array of an exact list or tuple; false for any other object,
// including a subclass, whose length and items could be a caller's code.
inline bool exact_items(PyObject* seq, Py_ssize_t* n, PyObject* const** items) noexcept {
    if (PyTuple_CheckExact(seq)) {
        *n = PyTuple_GET_SIZE(seq);
        *items = reinterpret_cast<PyTupleObject*>(seq)->ob_item;
        return true;
    }
    if (PyList_CheckExact(seq)) {
        *n = PyList_GET_SIZE(seq);
        *items = reinterpret_cast<PyListObject*>(seq)->ob_item;
        return true;
    }
    return false;
}

// Whether items[1:n] all have the exact type ``kind``.
inline bool rest_of_type(Py_ssize_t n, PyObject* const* items, const PyTypeObject* kind) noexcept {
    for (Py_ssize_t i = 1; i < n; ++i) {
        if (Py_TYPE(items[i]) != kind) {
            return false;
        }
    }
    return true;
}

// ``any(kind is t for t in types)`` for an exact tuple or frozenset (1 or 0), whose
// iteration runs no caller code; -1 for any other collection; -2 with an exception.
int identity_member(PyObject* types, PyObject* kind) {
    if (PyTuple_CheckExact(types)) {
        const Py_ssize_t m = PyTuple_GET_SIZE(types);
        for (Py_ssize_t j = 0; j < m; ++j) {
            if (PyTuple_GET_ITEM(types, j) == kind) {
                return 1;
            }
        }
        return 0;
    }
    if (!PyFrozenSet_CheckExact(types)) {
        return -1;
    }
    PyObject* it = PyObject_GetIter(types);
    if (it == nullptr) {
        return -2;
    }
    int found = 0;
    PyObject* t = nullptr;
    while ((t = PyIter_Next(it)) != nullptr) {
        const bool same = (t == kind);
        Py_DECREF(t);
        if (same) {
            found = 1;
            break;
        }
    }
    Py_DECREF(it);
    return PyErr_Occurred() ? -2 : found;
}

// Each scan below also sets *frames: the Python frames that its reference enters, from
// its caller, on the way to the answer that the scan gives.

// _one_exact_type(items, types): a new reference to the one exact type or None, or
// NotImplemented when ``types`` is not an exact tuple or frozenset (the reference then
// iterates that collection as it always did); nullptr with an exception.  One frame,
// and one more for the generator over ``types`` once an item is read.
PyObject* one_exact_type_scan(PyObject* seq, PyObject* types, int* frames) {
    Py_ssize_t n = 0;
    PyObject* const* items = nullptr;
    *frames = 1;
    if (!exact_items(seq, &n, &items) || n == 0) {
        Py_RETURN_NONE;
    }
    PyTypeObject* kind = Py_TYPE(items[0]);
    const int member = identity_member(types, reinterpret_cast<PyObject*>(kind));
    if (member == -2) {
        return nullptr;
    }
    if (member == -1) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    *frames = 2;
    if (member == 0 || !rest_of_type(n, items, kind)) {
        Py_RETURN_NONE;
    }
    Py_INCREF(kind);
    return reinterpret_cast<PyObject*>(kind);
}

// _plain_kind(values): a new reference to bool, int, float or str, or None: one exact
// plain type for every item of an exact nonempty list or tuple, and every float finite.
// One frame over those of _one_exact_type(values, _PLAIN_TYPES).
PyObject* plain_kind_scan(PyObject* values, int* frames) {
    Py_ssize_t n = 0;
    PyObject* const* items = nullptr;
    *frames = 2;
    if (!exact_items(values, &n, &items) || n == 0) {
        Py_RETURN_NONE;
    }
    *frames = 3;
    PyTypeObject* kind = Py_TYPE(items[0]);
    if (!plain_type(kind) || !rest_of_type(n, items, kind)) {
        Py_RETURN_NONE;
    }
    if (kind == &PyFloat_Type) {
        for (Py_ssize_t i = 0; i < n; ++i) {
            if (!std::isfinite(PyFloat_AS_DOUBLE(items[i]))) {
                Py_RETURN_NONE;
            }
        }
    }
    Py_INCREF(kind);
    return reinterpret_cast<PyObject*>(kind);
}

// _copy_free_labels of an exact tuple, label by label in order: 1 copy-free, 0 not
// (at the first label that is neither of an exact immutable type nor a copy-free exact
// tuple, where the reference stops too), -1 when a tuple deeper than kMaxNesting is
// reached before that decision.  Two frames (the function, then its generator or
// _windowed_copy_free), over the frames of the deepest exact tuple label reached.
int copy_free_scan(PyObject* tup, int depth, int* frames) noexcept {
    const Py_ssize_t n = PyTuple_GET_SIZE(tup);
    PyObject* const* items = reinterpret_cast<PyTupleObject*>(tup)->ob_item;
    int below = 0;
    for (Py_ssize_t i = 0; i < n; ++i) {
        const PyTypeObject* t = Py_TYPE(items[i]);
        if (immutable_type(t)) {
            continue;
        }
        if (t != &PyTuple_Type) {
            *frames = 2 + below;
            return 0;
        }
        if (depth >= kMaxNesting) {
            return -1;
        }
        int inner_frames = 0;
        const int inner = copy_free_scan(items[i], depth + 1, &inner_frames);
        if (inner < 0) {
            return inner;
        }
        below = inner_frames > below ? inner_frames : below;
        if (inner == 0) {
            *frames = 2 + below;
            return 0;
        }
    }
    *frames = 2 + below;
    return 1;
}

// ------------------------------------------------------------- the direct scans

// one_exact_type(items, types) -> the one exact type, None, or NotImplemented when
// ``types`` is not an exact tuple or frozenset.
PyObject* py_one_exact_type(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "one_exact_type(items, types) takes two arguments");
        return nullptr;
    }
    int frames = 0;
    return one_exact_type_scan(args[0], args[1], &frames);
}

// plain_kind(values) -> bool, int, float or str, or None.
PyObject* py_plain_kind(PyObject*, PyObject* values) {
    int frames = 0;
    return plain_kind_scan(values, &frames);
}

// copy_free_labels(items) -> True or False for an exact tuple decided within
// kMaxNesting, None otherwise (any other container, or a deeper tuple first).
PyObject* py_copy_free_labels(PyObject*, PyObject* items) {
    if (!PyTuple_CheckExact(items)) {
        Py_RETURN_NONE;
    }
    int frames = 0;
    const int found = copy_free_scan(items, 0, &frames);
    if (found < 0) {
        Py_RETURN_NONE;
    }
    return PyBool_FromLong(found);
}

// ------------------------------------------------------------------ the routes

// The state of a route, a tuple: its reference (the callable it replaces), its checks
// (a tuple of (namespace dict, name, expected original), ``absent`` for a name that
// must not be there), the modules whose attributes the reference reads (each must still
// be an exact module), ``enter``, the log list, the records (one per outcome below, None
// where a route has no such outcome), ``absent``, and for the batch route the batch
// class and the name of its labels attribute.
enum Slot : Py_ssize_t {
    kReference = 0,
    kChecks,
    kModules,
    kEnter,
    kLog,
    kRecords,
    kAbsent,
    kBatchClass,
    kBatchItems,
    kSlots
};
enum Outcome : Py_ssize_t {
    kDecided = 0,
    kRebound,
    kRecursion,
    kOtherTypes,
    kNotTuple,
    kTooDeep,
    kOtherBatch,
    kOtherCall,
    kOutcomes
};

inline PyObject* slot(PyObject* state, Slot s) noexcept { return PyTuple_GET_ITEM(state, s); }

// Append the record of ``outcome`` to the log; -1 with an exception.
int note(PyObject* state, Outcome outcome) {
    return PyList_Append(slot(state, kLog), PyTuple_GET_ITEM(slot(state, kRecords), outcome));
}

// The unchanged reference, with this call's own arguments, after the record of why.
PyObject* reference(PyObject* state, Outcome why, PyObject* const* args, Py_ssize_t nargs,
                    PyObject* kwnames) {
    if (note(state, why) < 0) {
        return nullptr;
    }
    return PyObject_Vectorcall(slot(state, kReference), args, static_cast<size_t>(nargs),
                               kwnames);
}

// 1 when every binding that the reference reads is its expected original, 0 when one is
// not, -1 with an exception.  Exact dictionaries are read by exact string keys.
int fresh(PyObject* state) {
    PyObject* checks = slot(state, kChecks);
    PyObject* absent = slot(state, kAbsent);
    const Py_ssize_t n = PyTuple_GET_SIZE(checks);
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* check = PyTuple_GET_ITEM(checks, i);
        PyObject* found =
            PyDict_GetItemWithError(PyTuple_GET_ITEM(check, 0), PyTuple_GET_ITEM(check, 1));
        if (found == nullptr && PyErr_Occurred()) {
            return -1;
        }
        PyObject* expected = PyTuple_GET_ITEM(check, 2);
        if (expected == absent ? found != nullptr : found != expected) {
            return 0;
        }
    }
    PyObject* modules = slot(state, kModules);
    const Py_ssize_t m = PyTuple_GET_SIZE(modules);
    for (Py_ssize_t j = 0; j < m; ++j) {
        if (!PyModule_CheckExact(PyTuple_GET_ITEM(modules, j))) {
            return 0;
        }
    }
    return 1;
}

// Enter ``frames`` nested Python frames from the caller's depth (``enter``): 1 entered,
// 0 stopped by the recursion limit (the RecursionError is dropped: the reference then
// runs and meets that limit itself), -1 with any other exception, which propagates.
int enter(PyObject* state, int frames) {
    PyObject* count = PyLong_FromLong(frames);
    if (count == nullptr) {
        return -1;
    }
    PyObject* done = PyObject_CallOneArg(slot(state, kEnter), count);
    Py_DECREF(count);
    if (done != nullptr) {
        Py_DECREF(done);
        return 1;
    }
    PyObject* error = PyErr_GetRaisedException();
    if (Py_IS_TYPE(error, reinterpret_cast<PyTypeObject*>(PyExc_RecursionError))) {
        Py_DECREF(error);
        return 0;
    }
    PyErr_SetRaisedException(error);
    return -1;
}

// The scan's answer ``found`` (a new reference) once the reference's ``frames`` are
// entered, else the reference itself.
PyObject* answer(PyObject* state, PyObject* found, int frames, PyObject* const* args,
                 Py_ssize_t nargs, PyObject* kwnames) {
    if (found == nullptr) {
        return nullptr;
    }
    const int entered = enter(state, frames);
    if (entered <= 0) {
        Py_DECREF(found);
        return entered < 0 ? nullptr : reference(state, kRecursion, args, nargs, kwnames);
    }
    if (note(state, kDecided) < 0) {
        Py_DECREF(found);
        return nullptr;
    }
    return found;
}

// A call of the reference's own positional form, with every binding it reads fresh:
// 1, 0 after running the reference into ``*result``, -1 with an exception.
int admitted(PyObject* state, Py_ssize_t positional, PyObject* const* args, Py_ssize_t nargs,
             PyObject* kwnames, PyObject** result) {
    if (nargs != positional || (kwnames != nullptr && PyTuple_GET_SIZE(kwnames) != 0)) {
        *result = reference(state, kOtherCall, args, nargs, kwnames);
        return 0;
    }
    const int ok = fresh(state);
    if (ok < 0) {
        return -1;
    }
    if (ok == 0) {
        *result = reference(state, kRebound, args, nargs, kwnames);
    }
    return ok;
}

PyObject* route_one_exact_type(PyObject* state, PyObject* const* args, Py_ssize_t nargs,
                               PyObject* kwnames) {
    PyObject* result = nullptr;
    const int ok = admitted(state, 2, args, nargs, kwnames, &result);
    if (ok <= 0) {
        return ok < 0 ? nullptr : result;
    }
    int frames = 0;
    PyObject* found = one_exact_type_scan(args[0], args[1], &frames);
    if (found == Py_NotImplemented) {
        Py_DECREF(found);
        return reference(state, kOtherTypes, args, nargs, kwnames);
    }
    return answer(state, found, frames, args, nargs, kwnames);
}

PyObject* route_plain_kind(PyObject* state, PyObject* const* args, Py_ssize_t nargs,
                           PyObject* kwnames) {
    PyObject* result = nullptr;
    const int ok = admitted(state, 1, args, nargs, kwnames, &result);
    if (ok <= 0) {
        return ok < 0 ? nullptr : result;
    }
    int frames = 0;
    PyObject* found = plain_kind_scan(args[0], &frames);
    return answer(state, found, frames, args, nargs, kwnames);
}

PyObject* route_copy_free_labels(PyObject* state, PyObject* const* args, Py_ssize_t nargs,
                                 PyObject* kwnames) {
    PyObject* result = nullptr;
    const int ok = admitted(state, 1, args, nargs, kwnames, &result);
    if (ok <= 0) {
        return ok < 0 ? nullptr : result;
    }
    if (!PyTuple_CheckExact(args[0])) {
        return reference(state, kNotTuple, args, nargs, kwnames);
    }
    int frames = 0;
    const int found = copy_free_scan(args[0], 0, &frames);
    if (found < 0) {
        return reference(state, kTooDeep, args, nargs, kwnames);
    }
    return answer(state, PyBool_FromLong(found), frames, args, nargs, kwnames);
}

// The labels of ``self`` when it is an exact batch whose class reads them as plain
// instance data, as the original getter would read them: no attribute hook, no class
// attribute of that name, the class's own method resolution order; nullptr without an
// exception for any other object (the original getter then reads them itself).
PyObject* batch_labels(PyObject* state, PyObject* self) {
    PyObject* cls = slot(state, kBatchClass);
    PyObject* name = slot(state, kBatchItems);
    auto* type = reinterpret_cast<PyTypeObject*>(cls);
    if (Py_TYPE(self) != type || type->tp_getattro != PyObject_GenericGetAttr) {
        return nullptr;
    }
    PyObject* mro = type->tp_mro;
    if (mro == nullptr || !PyTuple_CheckExact(mro) || PyTuple_GET_SIZE(mro) != 2 ||
        PyTuple_GET_ITEM(mro, 0) != cls ||
        PyTuple_GET_ITEM(mro, 1) != reinterpret_cast<PyObject*>(&PyBaseObject_Type)) {
        return nullptr;
    }
    PyObject* own = PyType_GetDict(type);
    if (own == nullptr) {
        PyErr_Clear();
        return nullptr;
    }
    PyObject* shadow = PyDict_GetItemWithError(own, name);
    Py_DECREF(own);
    if (shadow != nullptr || PyErr_Occurred()) {
        PyErr_Clear();
        return nullptr;
    }
    PyObject* labels = PyObject_GenericGetAttr(self, name);
    if (labels == nullptr) {
        PyErr_Clear();
    }
    return labels;
}

// The fget of RetainedLabelBatch._copy_free: the original getter's own frame counts.
PyObject* route_batch_copy_free(PyObject* state, PyObject* const* args, Py_ssize_t nargs,
                                PyObject* kwnames) {
    PyObject* result = nullptr;
    const int ok = admitted(state, 1, args, nargs, kwnames, &result);
    if (ok <= 0) {
        return ok < 0 ? nullptr : result;
    }
    PyObject* labels = batch_labels(state, args[0]);
    if (labels == nullptr) {
        return reference(state, kOtherBatch, args, nargs, kwnames);
    }
    if (!PyTuple_CheckExact(labels)) {
        Py_DECREF(labels);
        return reference(state, kNotTuple, args, nargs, kwnames);
    }
    int frames = 0;
    const int found = copy_free_scan(labels, 0, &frames);
    Py_DECREF(labels);
    if (found < 0) {
        return reference(state, kTooDeep, args, nargs, kwnames);
    }
    return answer(state, PyBool_FromLong(found), frames + 1, args, nargs, kwnames);
}

#define LABEL_ROUTE(name, fn, doc)                                                       \
    {name, reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)(void)>(fn)),           \
     METH_FASTCALL | METH_KEYWORDS, doc}

PyMethodDef g_routes[] = {
    LABEL_ROUTE("one_exact_type", route_one_exact_type,
                "Bound _one_exact_type: the exact scan once its frames enter, else the reference."),
    LABEL_ROUTE("plain_kind", route_plain_kind,
                "Bound _plain_kind: the exact scan once its frames enter, else the reference."),
    LABEL_ROUTE("copy_free_labels", route_copy_free_labels,
                "Bound _copy_free_labels: the exact scan once its frames enter, else the reference."),
    LABEL_ROUTE("batch_copy_free", route_batch_copy_free,
                "Bound RetainedLabelBatch._copy_free getter: the exact scan, else the getter."),
};
constexpr Py_ssize_t kRoutes = sizeof(g_routes) / sizeof(g_routes[0]);

// Whether ``checks`` is a tuple of (exact dict, exact str, object) triples.
bool valid_checks(PyObject* checks) {
    if (!PyTuple_CheckExact(checks)) {
        return false;
    }
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(checks); ++i) {
        PyObject* c = PyTuple_GET_ITEM(checks, i);
        if (!PyTuple_CheckExact(c) || PyTuple_GET_SIZE(c) != 3 ||
            !PyDict_CheckExact(PyTuple_GET_ITEM(c, 0)) ||
            !PyUnicode_CheckExact(PyTuple_GET_ITEM(c, 1))) {
            return false;
        }
    }
    return true;
}

// make_route(name, reference, checks, modules, enter, log, records, absent, batch_class,
// batch_items) -> the route of that name over this state (a C function object).
PyObject* py_make_route(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != kSlots + 1 || !PyUnicode_CheckExact(args[0])) {
        PyErr_SetString(PyExc_TypeError, "make_route(name, *state) takes a name and the state");
        return nullptr;
    }
    const PyMethodDef* def = nullptr;
    for (Py_ssize_t r = 0; r < kRoutes; ++r) {
        if (PyUnicode_CompareWithASCIIString(args[0], g_routes[r].ml_name) == 0) {
            def = &g_routes[r];
        }
    }
    PyObject* const* s = args + 1;
    const bool batch = def == &g_routes[kRoutes - 1];
    const bool ok =
        def != nullptr && PyCallable_Check(s[kReference]) && valid_checks(s[kChecks]) &&
        PyTuple_CheckExact(s[kModules]) && PyCallable_Check(s[kEnter]) &&
        PyList_CheckExact(s[kLog]) && PyTuple_CheckExact(s[kRecords]) &&
        PyTuple_GET_SIZE(s[kRecords]) == kOutcomes &&
        (batch ? PyType_Check(s[kBatchClass]) && PyUnicode_CheckExact(s[kBatchItems])
               : s[kBatchClass] == Py_None && s[kBatchItems] == Py_None);
    if (!ok) {
        PyErr_SetString(PyExc_TypeError, "make_route: an unknown route or a malformed state");
        return nullptr;
    }
    for (Py_ssize_t j = 0; j < PyTuple_GET_SIZE(s[kModules]); ++j) {
        if (!PyModule_CheckExact(PyTuple_GET_ITEM(s[kModules], j))) {
            PyErr_SetString(PyExc_TypeError, "make_route: modules must be exact modules");
            return nullptr;
        }
    }
    PyObject* state = PyTuple_New(kSlots);
    if (state == nullptr) {
        return nullptr;
    }
    for (Py_ssize_t k = 0; k < kSlots; ++k) {
        PyTuple_SET_ITEM(state, k, Py_NewRef(s[k]));
    }
    PyObject* fn = PyCFunction_NewEx(const_cast<PyMethodDef*>(def), state, nullptr);
    Py_DECREF(state);
    return fn;
}

PyMethodDef g_methods[] = {
    {"one_exact_type",
     reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)(void)>(py_one_exact_type)),
     METH_FASTCALL,
     "Exact one-type scan of an exact list or tuple; NotImplemented for another types collection."},
    {"plain_kind", py_plain_kind, METH_O,
     "Exact plain kind (bool, int, finite float, str) of an exact list or tuple, or None."},
    {"copy_free_labels", py_copy_free_labels, METH_O,
     "Exact copy-free scan of an exact tuple within the nesting bound, else None."},
    {"make_route",
     reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)(void)>(py_make_route)),
     METH_FASTCALL, "The bound route of one reference over its state."},
    {nullptr, nullptr, 0, nullptr},
};

int label_checks_exec(PyObject* m) {
    if (PyModule_AddStringConstant(m, "KERNEL", kKernel) < 0 ||
        PyModule_AddIntConstant(m, "MAX_NESTING", kMaxNesting) < 0 ||
        PyModule_AddIntConstant(m, "OUTCOMES", kOutcomes) < 0 ||
        PyModule_AddStringConstant(m, "SOURCE_SHA256", LABEL_CHECKS_SOURCE_SHA256) < 0) {
        return -1;
    }
    return 0;
}

// Multi-phase initialization (PEP 489): loading from an explicit path creates a fresh
// module object and puts nothing in sys.modules or the single-phase extension cache.
PyModuleDef_Slot g_slots[] = {
    {Py_mod_exec, reinterpret_cast<void*>(label_checks_exec)},
    {0, nullptr},
};

PyModuleDef g_module = {
    PyModuleDef_HEAD_INIT,
    "label_checks",
    "Exact C++17 fresh type-identity label scans and bound routes (private, explicit).",
    0,
    g_methods,
    g_slots,
    nullptr,
    nullptr,
    nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit_label_checks(void) { return PyModuleDef_Init(&g_module); }
