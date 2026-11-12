"""The fingerprint of an estimation sample: a digest of the actual ordered problem.

Each result records a fingerprint of the estimation sample.  The fingerprint identifies
the sample that an estimation ran on, so that a replication script, a validation report
or a later covariance request can tell whether two results concern the same rows and
values.  It is the SHA-256 digest of a canonical byte stream formed from the validated
numeric record of the retained rows (:class:`~powermean.data.inputs.RegressionData`), in
this order, each part tagged and length-prefixed:

1. the version of this schema (:data:`FINGERPRINT_VERSION`);
2. the row labels of the record: the original positions of the retained rows, in the
   order of the sample, so that a permutation, an omission or another selection of the
   same values changes the digest;
3. the column names, in order;
4. the raw outcome and the unscaled trained design, with their shapes;
5. the weights: their stated kind and their values, or the absence of weights;
6. the offset, or the exposure where one was given, so that the provenance of the
   index is part of the identity (an exposure and its logarithm as an offset are two
   inputs).

Encoding (a routine choice).  Doubles are written as IEEE little-endian bytes in C
order, after ``+ 0.0``, which turns ``-0.0`` into ``0.0`` and changes no other finite
double, so that the two zeros are one value; the record holds no NaN.  Shapes and
lengths are 8-byte little-endian integers; text is UTF-8; an integer label is its
decimal digits.  Nothing of Python's ``hash``, which is randomized per process, or of
``repr`` of an object enters the stream.  The digest is deterministic for equal values
on every platform; where a design column is the output of a transcendental function (a
logarithm, a spline), its last bits may differ across platforms or library versions, and
so may the digest: no bitwise cross-platform identity is claimed for such columns.

The fingerprint is the identity of the sample, not of the estimator: the mean order,
the working variance, the solver and its options are the provenance of a result and
are recorded beside it, never mixed into this digest.  A sample that ``positive_only``
conditions on positive outcomes differs by its rows, and so by its digest.
"""

from __future__ import annotations

import hashlib
import operator
from collections.abc import Iterable
from itertools import repeat

import numpy as np

from powermean.data.inputs import Label, RegressionData
from powermean.model.variance import FloatArray

FINGERPRINT_VERSION = "powermean.sample/1"
# Widths of the length and shape fields of the stream, in bytes.
WIDTH = 8


def _chunk(tag: str, payload: bytes) -> bytes:
    """``tag``, the length of ``payload`` and ``payload``: one part of the stream."""
    name = tag.encode("ascii")
    return len(name).to_bytes(1, "little") + name + _sized(payload)


def _sized(payload: bytes) -> bytes:
    """``payload`` after its length, so that no two parts can run into each other."""
    return len(payload).to_bytes(WIDTH, "little") + payload


def _floats(values: FloatArray) -> bytes:
    """The shape and the little-endian bytes of an array of finite doubles.

    ``+ 0.0`` maps ``-0.0`` to ``0.0`` and leaves every other double unchanged.
    """
    canonical = np.ascontiguousarray(values, dtype=np.float64) + 0.0
    shape = b"".join(int(n).to_bytes(WIDTH, "little") for n in canonical.shape)
    rank = len(canonical.shape).to_bytes(1, "little")
    return rank + shape + canonical.astype("<f8").tobytes(order="C")


def _label(label: Label) -> bytes:
    """An integer label as its decimal digits, a string label as its UTF-8 text."""
    if isinstance(label, int):
        return _chunk("int", str(label).encode("ascii"))
    return _chunk("str", label.encode("utf-8"))


def _labels(labels: Iterable[Label]) -> bytes:
    """The same ordered stream, in bulk for exact machine-range integer labels.

    Other iterables, mixed labels and unbounded Python integers keep the original
    scalar encoder. No caller conversion or label method decides eligibility.
    """
    if (type(labels) is tuple or type(labels) is list) and labels:
        if all(map(operator.is_, map(type, labels), repeat(int))):
            try:
                values = np.fromiter(labels, dtype=np.int64, count=len(labels))
            except OverflowError:
                return b"".join(_label(label) for label in labels)
            return _int_labels(values)
    return b"".join(_label(label) for label in labels)


def _int_labels(values: np.ndarray) -> bytes:
    """Encode exact int64 labels as the unchanged tagged decimal byte stream.

    Adapted from the delivered A4 integer-label encoder. Every row has the tag
    ``b'\\x03int'``, an eight-byte little-endian text length and signed decimal
    digits. Offsets preserve input order, including duplicates and negative rows.
    Unsigned magnitude also represents ``-2**63`` without signed overflow.
    """
    if not values.size:
        return b""
    negative = values < 0
    magnitude = values.astype(np.uint64)
    magnitude[negative] = np.uint64(0) - magnitude[negative]
    digits = np.ones(values.size, dtype=np.intp)
    for power in range(1, len(str(int(magnitude.max())))):
        digits += magnitude >= np.uint64(10**power)
    text = digits + negative
    sizes = 12 + text
    offsets = np.empty(values.size, dtype=np.intp)
    offsets[0] = 0
    np.cumsum(sizes[:-1], out=offsets[1:])
    buffer = np.zeros(int(sizes.sum()), dtype=np.uint8)
    for place, byte in enumerate(b"\x03int"):
        buffer[offsets + place] = byte
    buffer[offsets + 4] = text
    for sign in (False, True):
        for width in np.unique(digits[negative == sign]).tolist():
            rows = np.flatnonzero((digits == width) & (negative == sign))
            start = offsets[rows] + 12
            if sign:
                buffer[start] = 45
                start = start + 1
            quotient = magnitude[rows]
            for place in reversed(range(width)):
                quotient, digit = np.divmod(quotient, np.uint64(10))
                buffer[start + place] = digit.astype(np.uint8) + 48
    return buffer.tobytes()


def _names(names: Iterable[str]) -> bytes:
    """The column names, in order."""
    return b"".join(_chunk("name", name.encode("utf-8")) for name in names)


def _weights(data: RegressionData) -> bytes:
    """The kind and the values of the weights, or the absence of weights."""
    if data.weights is None:
        return b"none"
    kind = _chunk("kind", data.weights.kind.encode("ascii"))
    return kind + _chunk("values", _floats(data.weights.values))


def sample_fingerprint(data: RegressionData) -> str:
    """The hexadecimal SHA-256 fingerprint of the sample held by ``data``.

    Parameters
    ----------
    data : RegressionData
        The validated numeric record of the retained rows, whose row labels are their
        original positions.

    Returns
    -------
    str
        Sixty-four hexadecimal digits, deterministic for equal values.

    Raises
    ------
    TypeError
        ``data`` is not a ``RegressionData``.
    """
    if type(data) is not RegressionData:
        raise TypeError("data must be a RegressionData record")
    digest = hashlib.sha256()
    digest.update(_chunk("version", FINGERPRINT_VERSION.encode("ascii")))
    digest.update(_chunk("rows", _labels(data.row_labels)))
    digest.update(_chunk("columns", _names(data.column_names)))
    digest.update(_chunk("y", _floats(data.y)))
    digest.update(_chunk("design", _floats(data.design)))
    digest.update(_chunk("weights", _weights(data)))
    if data.exposure is not None:
        digest.update(_chunk("exposure", _floats(data.exposure)))
    else:
        digest.update(_chunk("offset", _floats(data.offset)))
    return digest.hexdigest()
