"""Array algebra: special arrays, contraction by index specification, entry-by-entry operations.

Ported from the reference paclet (``src/ArrayAlgebra.wl``). The central operation is
:func:`array_multiply`, the contraction of several arrays according to an index
specification in the manner of Einstein summation (``"ij,jk->ik"``). An index may occur
in any number of the arrays and in the output too, so it expresses diagonals and
multi-way contractions that pairwise tensor contraction cannot.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np
from frontier import SourceRef, define

from hypermatrix.arrays import ArrayError, ArrayObject, as_array, tighten, to_ndarray
from hypermatrix.fields import FiniteField

_INT64_LIMIT = 2**62
_LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


# -- value domains for special arrays -----------------------------------


def _zero_one(domain: Any) -> tuple[Any, Any]:
    if isinstance(domain, FiniteField):
        return domain.zero, domain.one
    table = {
        "Boolean": (False, True),
        "Integer": (0, 1),
        "Rational": (0, 1),
        "Real": (0.0, 1.0),
        "Complex": (0, 1),
    }
    if domain not in table:
        raise ArrayError("ZeroArray::dom", f"{domain!r} is not a recognized value domain")
    return table[domain]


def _is_int(x: Any) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


def _shape_and_domain(head: str, args: tuple) -> tuple[list[int], Any]:
    """``(dims)``, ``(dims, dom)``, ``(n, d)`` or ``(n, d, dom)``: the reference's calling forms."""
    if args and isinstance(args[0], (list, tuple)):
        dims, rest = list(args[0]), args[1:]
        if not all(_is_int(k) and k > 0 for k in dims):
            raise ArrayError(f"{head}::spec", f"{args!r} is not a valid array shape")
    elif len(args) >= 2 and _is_int(args[0]) and _is_int(args[1]) and args[0] >= 0 and args[1] >= 1:
        dims, rest = [int(args[1])] * int(args[0]), args[2:]
    else:
        raise ArrayError(f"{head}::spec", f"{args!r} is not a valid array shape")
    if len(rest) > 1:
        raise ArrayError(f"{head}::spec", f"{args!r} is not a valid array shape")
    return dims, rest[0] if rest else "Integer"


def _constant(head: str, which: int, args: tuple) -> ArrayObject:
    dims, domain = _shape_and_domain(head, args)
    value = _zero_one(domain)[which]
    arr = np.empty(dims, dtype=object)
    arr.reshape(-1)[:] = [value] * arr.size
    # a constant array is invariant under every permutation of its indices
    return ArrayObject._trusted(tighten(arr), "Symmetric" if len(set(dims)) <= 1 else "Rigid")


@define(
    "hypermatrix.zero_array",
    "Zero array",
    r"`ZeroArray(n, d)` is the $n$-array with every index of length $d$ and every entry $0$; "
    r"`ZeroArray(dims)` has the given dimensions; a final argument names the value domain.",
    maturity="working",
    depends_on=("hypermatrix.array",),
    sources=(SourceRef("hypergraph-paclet", "symbol:ZeroArray", "implementation"),),
)
def zero_array(*args: Any) -> ArrayObject:
    return _constant("ZeroArray", 0, args)


@define(
    "hypermatrix.one_array",
    "One array",
    r"`OneArray(n, d)` is the $n$-array with every index of length $d$ and every entry $1$.",
    maturity="working",
    depends_on=("hypermatrix.array",),
    sources=(SourceRef("hypergraph-paclet", "symbol:OneArray", "implementation"),),
)
def one_array(*args: Any) -> ArrayObject:
    return _constant("OneArray", 1, args)


def _partial_identity(head: str, n: Any, d: Any, positions: Any, domain: Any) -> ArrayObject:
    if not (_is_int(n) and n >= 1 and _is_int(d) and d >= 1):
        raise ArrayError(f"{head}::spec", f"{(n, d)!r} is not a valid array shape")
    pos = list(positions)
    if not (pos and all(_is_int(p) and 1 <= p <= n for p in pos) and len(set(pos)) == len(pos)):
        raise ArrayError(
            "PartialIdentityArray::pos", f"{positions!r} is not a list of distinct index positions between 1 and {n}"
        )
    zero, one = _zero_one(domain)
    arr = np.empty([d] * n, dtype=object)
    for idx in np.ndindex(*arr.shape):
        arr[idx] = one if len({idx[p - 1] for p in pos}) == 1 else zero
    return ArrayObject._trusted(tighten(arr), "Symmetric" if sorted(pos) == list(range(1, n + 1)) else "Rigid")


@define(
    "hypermatrix.identity_array",
    "Identity array",
    r"`IdentityArray(n, d)` is the $n$-array $\delta_{i_1 \dots i_n}$ with indices of length $d$: "
    r"$1$ where all $n$ indices agree and $0$ elsewhere. At $n = 2$ it is the identity matrix.",
    maturity="working",
    depends_on=("hypermatrix.array",),
    sources=(SourceRef("hypergraph-paclet", "symbol:IdentityArray", "implementation"),),
)
def identity_array(n: Any, d: Any, domain: Any = "Integer") -> ArrayObject:
    if not _is_int(n):
        raise ArrayError("IdentityArray::spec", f"{(n, d)!r} is not a valid array shape")
    return _partial_identity("IdentityArray", n, d, range(1, n + 1), domain)


@define(
    "hypermatrix.partial_identity_array",
    "Partial identity array",
    r"`PartialIdentityArray(n, d, P)` is the $n$-array whose entry is $1$ where the indices at the "
    r"positions $P$ agree and $0$ elsewhere. With $P$ all positions it is the identity array.",
    maturity="working",
    depends_on=("hypermatrix.array", "hypermatrix.identity_array"),
    sources=(SourceRef("hypergraph-paclet", "symbol:PartialIdentityArray", "implementation"),),
)
def partial_identity_array(n: Any, d: Any, positions: Any, domain: Any = "Integer") -> ArrayObject:
    return _partial_identity("PartialIdentityArray", n, d, positions, domain)


# -- index specifications ------------------------------------------------


def _split_keep_empty(text: str) -> list[list[str]]:
    # empty segments matter: a 0-array contributes no labels, and an empty output contracts everything
    return [list(part) for part in text.split(",")]


def parse_index_spec(spec: Any) -> tuple[list[list[Any]], list[Any] | None]:
    """``"ij,jk->ik"``, ``(ins, out)`` or ``ins`` -> (input label lists, output labels or None)."""
    if isinstance(spec, str):
        clean = "".join(spec.split())
        if clean.count("->") > 1:
            raise ArrayError("ArrayMultiply::spec", f"{spec!r} is not a valid index specification")
        if "->" in clean:
            ins, out = clean.split("->")
            return _split_keep_empty(ins), list(out)
        return _split_keep_empty(clean), None
    if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], (list, tuple)):
        ins, out = spec
        if all(isinstance(x, (list, tuple)) for x in ins) and isinstance(out, (list, tuple)):
            return [list(x) for x in ins], list(out)
    if isinstance(spec, (list, tuple)) and spec and all(isinstance(x, (list, tuple)) for x in spec):
        return [list(x) for x in spec], None
    raise ArrayError("ArrayMultiply::spec", f"{spec!r} is not a valid index specification")


def _resolve(arrays: list[Any], spec: Any):
    ins, out = parse_index_spec(spec)
    if len(ins) != len(arrays):
        raise ArrayError(
            "ArrayMultiply::count", f"the specification names {len(ins)} arrays but {len(arrays)} were given"
        )
    data = []
    for k, a in enumerate(arrays, 1):
        try:
            data.append(to_ndarray(a))
        except ArrayError:
            raise ArrayError("ArrayMultiply::array", f"argument {k} is not a rectangular array") from None
    dim_of: dict[Any, int] = {}
    for k, (labels, arr) in enumerate(zip(ins, data, strict=True), 1):
        if len(labels) != arr.ndim:
            raise ArrayError(
                "ArrayMultiply::rank", f"array {k} has {arr.ndim} indices but its specification names {len(labels)}"
            )
        for label, n in zip(labels, arr.shape, strict=True):
            if dim_of.setdefault(label, n) != n:
                raise ArrayError(
                    "ArrayMultiply::dims", f"index {label!r} is used with inconsistent lengths {[dim_of[label], n]}"
                )
    if out is None:  # Einstein convention: labels occurring exactly once survive, in sorted order
        counts = Counter(label for labels in ins for label in labels)
        out = sorted((label for label, c in counts.items() if c == 1), key=_label_key)
    for label in out:
        if label not in dim_of:
            raise ArrayError("ArrayMultiply::out", f"the output index {label!r} does not occur among the inputs")
    if len(set(out)) != len(out):
        raise ArrayError("ArrayMultiply::spec", f"output indices {out!r} repeat a label")
    return ins, out, data, dim_of


def _label_key(label: Any) -> tuple:
    return (0, label, "") if isinstance(label, (int, float)) else (1, 0, str(label))


def array_multiply_cost(arrays: list[Any], spec: Any) -> int:
    """Multiplications done by the naive contraction: one per array per assignment of all labels."""
    ins, out, data, dim_of = _resolve(arrays, spec)
    return max(1, len(data)) * math.prod(dim_of.values())


def _max_abs(arr: np.ndarray) -> int:
    return int(np.max(np.abs(arr))) if arr.size else 0


@define(
    "hypermatrix.array_multiply",
    "Contraction by index specification",
    r"`ArrayMultiply([A_1, ..., A_m], spec)` contracts the arrays according to an index specification "
    r"such as `\"ij,jk->ik\"`: each array is assigned a list of index labels, and the result is "
    r"$C_{\mathrm{out}} = \sum_{\mathrm{rest}} \prod_k (A_k)_{\mathrm{labels}_k}$, the sum running over "
    r"every label not in the output. A label may occur in any number of arrays and in the output, so "
    r"diagonals and multi-way contractions are expressed directly. Without an output part, the labels "
    r"occurring exactly once survive, in sorted order. The specification may also be given as index "
    r"lists, `[[1, 2], [2, 3]] -> [1, 3]`.",
    maturity="stable",
    depends_on=("hypermatrix.array",),
    sources=(
        SourceRef("hypergraph-paclet", "symbol:ArrayMultiply", "implementation"),
        SourceRef("heaps-of-fish", "section:Array Algebra", "definition"),
    ),
    tags=("array", "contraction"),
)
def array_multiply(arrays: list[Any], spec: Any) -> ArrayObject:
    ins, out, data, dim_of = _resolve(list(arrays), spec)
    letters = {}
    for label in dim_of:
        if len(letters) == len(_LETTERS):
            raise ArrayError("ArrayMultiply::spec", f"more than {len(_LETTERS)} distinct index labels")
        letters[label] = _LETTERS[len(letters)]
    subscripts = ",".join("".join(letters[x] for x in labels) for labels in ins)
    subscripts += "->" + "".join(letters[x] for x in out)
    # exact integers: stay in int64 only when no partial sum can overflow
    if any(np.issubdtype(a.dtype, np.integer) for a in data):
        summed = math.prod(n for label, n in dim_of.items() if label not in out)
        bound = summed * math.prod(max(1, _max_abs(a)) if a.dtype != object else _INT64_LIMIT for a in data)
        if bound >= _INT64_LIMIT:
            data = [a.astype(object) if np.issubdtype(a.dtype, np.integer) else a for a in data]
    if any(a.dtype == object for a in data):
        data = [a.astype(object) for a in data]
    result = np.einsum(subscripts, *data) if data else np.array(1)
    return ArrayObject._trusted(tighten(np.asarray(result, dtype=object)))


# -- entry-by-entry operations ------------------------------------------


def _elementwise(head: str, op: str, arrays: tuple) -> ArrayObject:
    if not arrays:
        raise ArrayError(f"{head}::args", "at least one array is expected")
    data = []
    for k, a in enumerate(arrays, 1):
        try:
            data.append(to_ndarray(a))
        except ArrayError:
            raise ArrayError(f"{head}::array", f"argument {k} is not a rectangular array") from None
    shapes = {a.shape for a in data if a.ndim > 0}
    if len(shapes) > 1:
        raise ArrayError(
            f"{head}::dims",
            "an entry-by-entry operation needs arrays of the same dimensions, apart from arrays of order zero; "
            f"the dimensions given were {[list(a.shape) for a in data]}",
        )
    ints = [a for a in data if np.issubdtype(a.dtype, np.integer)]
    if ints:
        mags = [max(1, _max_abs(a)) for a in ints]
        bound = sum(mags) if op == "add" else math.prod(mags)
        if bound >= _INT64_LIMIT or any(a.dtype == object for a in data):
            data = [a.astype(object) for a in data]
    result = data[0]
    for a in data[1:]:
        result = result + a if op == "add" else result * a
    # an entry-by-entry operation preserves any symmetry all its (non-scalar) arguments share
    syms = {
        (a.symmetry if isinstance(a, ArrayObject) else "Rigid") for a, d in zip(arrays, data, strict=True) if d.ndim > 0
    }
    sym = syms.pop() if len(syms) == 1 else "Rigid"
    return ArrayObject._trusted(tighten(np.asarray(result, dtype=object)), sym)


@define(
    "hypermatrix.array_add",
    "Entry-by-entry sum",
    r"`ArrayAdd(A, B, ...)` adds arrays entry by entry. The arrays must have equal dimensions, except "
    r"that a 0-array (a scalar) is added to every entry. `A + B` is the same.",
    maturity="working",
    depends_on=("hypermatrix.array",),
    sources=(SourceRef("hypergraph-paclet", "symbol:ArrayAdd", "implementation"),),
)
def array_add(*arrays: Any) -> ArrayObject:
    return _elementwise("ArrayAdd", "add", arrays)


@define(
    "hypermatrix.array_times",
    "Entry-by-entry product",
    r"`ArrayTimes(A, B, ...)` multiplies arrays entry by entry; a 0-array multiplies every entry, so "
    r"scalar multiplication is the same operation. `A * B` is the same.",
    maturity="working",
    depends_on=("hypermatrix.array",),
    sources=(SourceRef("hypergraph-paclet", "symbol:ArrayTimes", "implementation"),),
)
def array_times(*arrays: Any) -> ArrayObject:
    return _elementwise("ArrayTimes", "times", arrays)


__all__ = [
    "array_add",
    "array_multiply",
    "array_multiply_cost",
    "array_times",
    "as_array",
    "identity_array",
    "one_array",
    "parse_index_spec",
    "partial_identity_array",
    "zero_array",
]
