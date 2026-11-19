"""The fingerprint of an estimation sample: a digest of the actual ordered problem.

``design/02_computation.md``, Section 9 ("each result records ... a fingerprint of the
estimation sample"); ``design/11_engineering.md``, Section 5, rule 9; decisions D51 and
D55.  The fingerprint identifies the sample that an estimation ran on, so that a
replication script, a validation report or a later covariance request can tell whether
two results concern the same rows and values.  It is the SHA-256 digest of a canonical
byte stream formed from the validated numeric record of the retained rows
(:class:`~powermean.data.inputs.RegressionData`), in this order, each part tagged and
length-prefixed:

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

Encoding (decision D51, a routine choice).  Doubles are written as IEEE little-endian
bytes in C order, after ``+ 0.0``, which turns ``-0.0`` into ``0.0`` and changes no
other finite double, so that the two zeros are one value; the record holds no NaN.
Shapes and lengths are 8-byte little-endian integers; text is UTF-8; an integer label
is its decimal digits.  Nothing of Python's ``hash``, which is randomized per process,
or of ``repr`` of an object enters the stream.  The digest is deterministic for equal
values on every platform; where a design column is the output of a transcendental
function (a logarithm, a spline), its last bits may differ across platforms or library
versions, and so may the digest: no bitwise cross-platform identity is claimed for such
columns (file 13, Section 6).

The fingerprint is the identity of the sample, not of the estimator: the mean order,
the working variance, the solver and its options are the provenance of a result and
are recorded beside it, never mixed into this digest.  A sample that ``positive_only``
conditions on positive outcomes differs by its rows, and so by its digest.
"""

from __future__ import annotations

import hashlib
import operator
from collections.abc import Iterable, Sequence
from itertools import repeat
from typing import Protocol

import numpy as np

from powermean.data.inputs import Label, RegressionData
from powermean.model.variance import FloatArray

FINGERPRINT_VERSION = "powermean.sample/1"
# Widths of the length and shape fields of the stream, in bytes.
WIDTH = 8
# The tag of an integer label, as ``_chunk("int", ...)`` writes it.
_INT_TAG = np.frombuffer(b"\x03int", dtype=np.uint8)

# The pieces of one part of the stream, in order: bytes or C-contiguous byte arrays,
# whose concatenation is the part.
Pieces = Sequence[bytes | np.ndarray]


class _Digest(Protocol):
    """What the fingerprint needs of a ``hashlib`` digest."""

    def update(self, data: bytes | np.ndarray, /) -> None:
        """Append ``data`` to the digested stream."""
        ...


def _chunk(tag: str, payload: bytes) -> bytes:
    """``tag``, the length of ``payload`` and ``payload``: one part of the stream."""
    name = tag.encode("ascii")
    return len(name).to_bytes(1, "little") + name + _sized(payload)


def _sized(payload: bytes) -> bytes:
    """``payload`` after its length, so that no two parts can run into each other."""
    return len(payload).to_bytes(WIDTH, "little") + payload


def _head(tag: str, pieces: Pieces) -> bytes:
    """What :func:`_chunk` writes before a payload made of ``pieces``."""
    name = tag.encode("ascii")
    size = sum(memoryview(piece).nbytes for piece in pieces)
    return len(name).to_bytes(1, "little") + name + size.to_bytes(WIDTH, "little")


def _update(digest: _Digest, tag: str, pieces: Pieces) -> None:
    """Digest ``_chunk(tag, b"".join(pieces))`` piece by piece, without joining.

    SHA-256 of a stream does not depend on how the stream is cut, so the digest is
    that of the joined bytes.
    """
    digest.update(_head(tag, pieces))
    for piece in pieces:
        digest.update(piece)


def _joined(pieces: Pieces) -> bytes:
    """The bytes of ``pieces``, joined in order."""
    return b"".join(bytes(memoryview(piece)) for piece in pieces)


def _float_pieces(values: FloatArray) -> list[bytes | np.ndarray]:
    """The pieces of :func:`_floats`: the rank and shape, then the doubles as bytes.

    The doubles are the C-order little-endian bytes of the canonical array, read in
    place rather than copied out.
    """
    canonical = np.ascontiguousarray(values, dtype=np.float64) + 0.0
    shape = b"".join(int(n).to_bytes(WIDTH, "little") for n in canonical.shape)
    rank = len(canonical.shape).to_bytes(1, "little")
    stored = np.ascontiguousarray(canonical.astype("<f8", copy=False))
    return [rank + shape, stored.reshape(-1).view(np.uint8)]


def _floats(values: FloatArray) -> bytes:
    """The shape and the little-endian bytes of an array of finite doubles.

    ``+ 0.0`` maps ``-0.0`` to ``0.0`` and leaves every other double unchanged.
    """
    return _joined(_float_pieces(values))


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
    return _joined(_label_pieces(labels))


def _label_pieces(labels: Iterable[Label]) -> list[bytes | np.ndarray]:
    """The pieces of :func:`_labels`, whose concatenation is its stream."""
    if (
        (type(labels) is tuple or type(labels) is list)
        and labels
        and all(map(operator.is_, map(type, labels), repeat(int)))
    ):
        try:
            values = np.fromiter(labels, dtype=np.int64, count=len(labels))
        except OverflowError:
            return [b"".join(_label(label) for label in labels)]
        return _int_label_pieces(values)
    return [b"".join(_label(label) for label in labels)]


def _int_labels(values: np.ndarray) -> bytes:
    r"""Encode exact int64 labels as the unchanged tagged decimal byte stream.

    Adapted from the delivered A4 integer-label encoder. Every row has the tag
    ``b'\x03int'``, an eight-byte little-endian text length and signed decimal
    digits. Offsets preserve input order, including duplicates and negative rows.
    Unsigned magnitude also represents ``-2**63`` without signed overflow.
    """
    return _joined(_int_label_pieces(values))


def _int_label_pieces(values: np.ndarray) -> list[bytes | np.ndarray]:
    """The pieces of :func:`_int_labels`: by blocks for sorted nonnegative labels.

    Nonnegative labels in nondecreasing order, such as the positions of a sample,
    fall into consecutive blocks of one number of digits, where every row has the
    same length; each block is written as a matrix of its rows.  Every other array
    takes the row-offset encoder.  Both write the same bytes for every row.
    """
    if not values.size:
        return [b""]
    if values[0] >= 0 and bool((values[1:] >= values[:-1]).all()):
        return _sorted_label_blocks(values)
    return [_offset_labels(values)]


def _sorted_label_blocks(values: np.ndarray) -> list[bytes | np.ndarray]:
    """The rows of nonnegative nondecreasing int64 labels, one block per width.

    The block of ``width`` digits holds the labels from ``10**(width - 1)`` (from
    zero for one digit) to below ``10**width``; a row is the tag, the width as an
    eight-byte little-endian length and the digits, most significant first.
    """
    top = len(str(int(values[-1])))
    pieces: list[bytes | np.ndarray] = []
    start = 0
    for width in range(1, top + 1):
        stop = values.size if width == top else int(np.searchsorted(values, 10**width))
        if stop > start:
            pieces.append(_label_block(values[start:stop], width))
        start = stop
    return pieces


def _label_block(values: np.ndarray, width: int) -> np.ndarray:
    """The rows of nonnegative labels of ``width`` digits each, as one byte array."""
    rows = np.zeros((values.size, 12 + width), dtype=np.uint8)
    rows[:, :4] = _INT_TAG
    rows[:, 4] = width
    quotient = values.astype(np.uint64)
    for place in reversed(range(width)):
        quotient, digit = np.divmod(quotient, np.uint64(10))
        rows[:, 12 + place] = digit.astype(np.uint8) + 48
    return rows.reshape(-1)


def _offset_labels(values: np.ndarray) -> np.ndarray:
    """The rows of any nonempty int64 labels, placed by offsets, as one byte array."""
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
    return buffer


def _names(names: Iterable[str]) -> bytes:
    """The column names, in order."""
    return b"".join(_chunk("name", name.encode("utf-8")) for name in names)


def _weights(data: RegressionData) -> bytes:
    """The kind and the values of the weights, or the absence of weights."""
    return _joined(_weight_pieces(data))


def _weight_pieces(data: RegressionData) -> list[bytes | np.ndarray]:
    """The pieces of :func:`_weights`, whose concatenation is its bytes."""
    if data.weights is None:
        return [b"none"]
    kind = _chunk("kind", data.weights.kind.encode("ascii"))
    values = _float_pieces(data.weights.values)
    return [kind, _head("values", values), *values]


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
    return _digest(data, _label_pieces(data.row_labels))


def _positions_fingerprint(data: RegressionData, positions: np.ndarray) -> str:
    """:func:`sample_fingerprint` of ``data`` labeled by the integers ``positions``.

    For a record whose row labels are ``positions.tolist()``, as the sample derivation
    forms it: the bulk encoder of the labels reads them back as these integers, so the
    rows part is encoded from the array itself, and the digest is that of
    :func:`sample_fingerprint`.  ``ValueError`` when the numbers of rows differ.
    """
    if type(data) is not RegressionData:
        raise TypeError("data must be a RegressionData record")
    if len(data.row_labels) != positions.size:
        raise ValueError("positions must label every row of the record")
    return _digest(data, _int_label_pieces(positions.astype(np.int64)))


def _digest(data: RegressionData, rows: Pieces) -> str:
    """The fingerprint of ``data`` whose row labels are encoded as ``rows``."""
    digest = hashlib.sha256()
    digest.update(_chunk("version", FINGERPRINT_VERSION.encode("ascii")))
    _update(digest, "rows", rows)
    digest.update(_chunk("columns", _names(data.column_names)))
    _update(digest, "y", _float_pieces(data.y))
    _update(digest, "design", _float_pieces(data.design))
    _update(digest, "weights", _weight_pieces(data))
    if data.exposure is not None:
        _update(digest, "exposure", _float_pieces(data.exposure))
    else:
        _update(digest, "offset", _float_pieces(data.offset))
    return digest.hexdigest()
