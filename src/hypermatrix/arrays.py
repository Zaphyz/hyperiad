"""Arrays of any order, the objects of hypermatrix algebra.

Ported from the reference paclet (``src/Hypermatrix.wl``): an :class:`ArrayObject`
holds rectangular data of any order -- a scalar is the array of order zero -- and
a declared index symmetry ("Rigid", "Cyclic" or "Symmetric") that is verified
against the entries. What the entries are made of is never declared; it is read
off them by :attr:`ArrayObject.domain`.

Entries may be numbers, elements of a prime field, SymPy expressions or any
other values. Storage is a NumPy array: a numeric dtype where the entries allow
it exactly, ``object`` otherwise, so arithmetic stays exact.
"""

from __future__ import annotations

import itertools
from fractions import Fraction
from numbers import Number
from typing import Any

import numpy as np
import sympy
from frontier import SourceRef, define

from hypermatrix.fields import FieldElement, FiniteField

SYMMETRIES = ("Symmetric", "Cyclic", "Rigid")  # most constrained first, the canonical order
DOMAINS = ("Boolean", "Integer", "Rational", "Real", "Complex")  # narrowest first
_INT64_LIMIT = 2**62


class ArrayError(ValueError):
    """An invalid array or array operation. ``code`` names the reference paclet's message."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


# -- data normalization -------------------------------------------------


def _shape(data: Any) -> tuple[int, ...] | None:
    """Dimensions of nested lists/tuples, or None if ragged. Non-lists are 0-arrays."""
    if isinstance(data, np.ndarray):
        return data.shape
    if not isinstance(data, (list, tuple)):
        return ()
    if not data:
        return (0,)
    shapes = {_shape(x) for x in data}
    if len(shapes) != 1 or None in shapes:
        return None
    return (len(data), *shapes.pop())


def _leaves(data: Any, depth: int):
    if depth == 0:
        yield data
    else:
        for x in data:
            yield from _leaves(x, depth - 1)


def _normalize_scalar(x: Any) -> Any:
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, Fraction) and x.denominator == 1:
        return int(x.numerator)
    if isinstance(x, sympy.Basic) and x.is_Integer:
        return int(x)
    return x


def tighten(arr: np.ndarray) -> np.ndarray:
    """Store entries in the narrowest exact NumPy dtype: bool, int64, float64, complex128 or object."""
    flat = [_normalize_scalar(x) for x in arr.reshape(-1).tolist()] if arr.dtype == object else None
    if flat is None:
        return arr
    if all(isinstance(x, bool) for x in flat):
        return np.array(flat, dtype=bool).reshape(arr.shape)
    if all(isinstance(x, int) and not isinstance(x, bool) for x in flat):
        if all(-_INT64_LIMIT < x < _INT64_LIMIT for x in flat):
            return np.array(flat, dtype=np.int64).reshape(arr.shape)
    elif all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in flat):
        return np.array(flat, dtype=np.float64).reshape(arr.shape)
    elif all(isinstance(x, (int, float, complex)) and not isinstance(x, bool) for x in flat):
        if any(isinstance(x, complex) for x in flat):
            return np.array(flat, dtype=np.complex128).reshape(arr.shape)
    out = np.empty(arr.shape, dtype=object)
    out.reshape(-1)[:] = flat if flat else []
    return out


def to_ndarray(data: Any) -> np.ndarray:
    """Rectangular data (nested lists, NumPy array, ArrayObject or scalar) as a NumPy array."""
    if isinstance(data, ArrayObject):
        return data.data
    if isinstance(data, np.ndarray):
        return tighten(data.astype(object)) if data.dtype != object else tighten(data)
    shape = _shape(data)
    if shape is None:
        raise ArrayError("ArrayObject::data", f"{data!r} is not a rectangular array")
    out = np.empty(shape, dtype=object)
    flat = list(_leaves(data, len(shape)))
    if shape == ():
        out[()] = flat[0]
    else:
        out.reshape(-1)[:] = flat
    return tighten(out)


def entries_of(arr: np.ndarray) -> Any:
    """Entries as nested Python lists (a bare value for a 0-array)."""
    if arr.dtype == object:
        if arr.ndim == 0:
            return _normalize_scalar(arr[()])
        return [entries_of(x) for x in arr] if arr.ndim > 1 else [_normalize_scalar(x) for x in arr]
    return arr.tolist()


# -- symmetry -----------------------------------------------------------


def normalize_symmetry(sym: str) -> str:
    s = sym.lower() if isinstance(sym, str) else None
    table = {
        "rigid": "Rigid",
        "none": "Rigid",
        "cyclic": "Cyclic",
        "symmetric": "Symmetric",
        "symmetrical": "Symmetric",
    }
    if s not in table:
        raise ArrayError(
            "ArrayObject::sym", f"{sym!r} is not a valid array symmetry; expected 'Rigid', 'Cyclic' or 'Symmetric'"
        )
    return table[s]


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.all(a == b))


def has_symmetry(arr: np.ndarray, sym: str) -> bool:
    """Invariance under the index permutations of ``sym``; a generating set suffices."""
    n = arr.ndim
    if sym == "Rigid" or n <= 1:
        return True
    if len(set(arr.shape)) != 1:
        return False
    rotation = np.transpose(arr, list(range(1, n)) + [0])
    if not _same(arr, rotation):
        return False
    if sym == "Cyclic":
        return True
    swap = np.transpose(arr, [1, 0, *range(2, n)])
    return _same(arr, swap)


def canonical_index(idx: tuple[int, ...], sym: str) -> tuple[int, ...]:
    """The least index tuple equivalent to ``idx`` under ``sym``."""
    if sym == "Rigid" or not idx:
        return idx
    if sym == "Symmetric":
        return tuple(sorted(idx))
    return min(idx[k:] + idx[:k] for k in range(len(idx)))


# -- value domains ------------------------------------------------------


def _domain_of(x: Any) -> str | FiniteField:
    """The narrowest domain of one entry (or 'Expression')."""
    if isinstance(x, (bool, np.bool_)):
        return "Boolean"
    if isinstance(x, FieldElement):
        return x.field
    if isinstance(x, (int, np.integer)):
        return "Integer"
    if isinstance(x, Fraction):
        return "Rational"
    if isinstance(x, (float, np.floating)):
        return "Real"
    if isinstance(x, (complex, np.complexfloating)):
        return "Complex"
    if isinstance(x, sympy.Basic) and x.is_number:
        if x.is_Integer:
            return "Integer"
        if x.is_Rational:
            return "Rational"
        if x.is_real:
            return "Real"
        return "Complex"
    return "Expression"


def infer_domain(arr: np.ndarray) -> str | FiniteField:
    """The narrowest recognized domain containing every entry, else 'Expression'."""
    if arr.dtype == bool:
        return "Boolean"
    if np.issubdtype(arr.dtype, np.integer):
        return "Integer"
    if np.issubdtype(arr.dtype, np.floating):
        return "Real"
    if np.issubdtype(arr.dtype, np.complexfloating):
        return "Complex"
    kinds = {_domain_of(x) for x in arr.reshape(-1).tolist()}
    if not kinds:
        return DOMAINS[0]  # vacuously, as in the reference
    if len(kinds) == 1:
        return kinds.pop()
    if kinds & {"Boolean", "Expression"} or any(isinstance(k, FiniteField) for k in kinds):
        return "Expression"
    return max(kinds, key=DOMAINS.index)


# -- the array object ---------------------------------------------------


@define(
    "hypermatrix.array",
    "Array (hypermatrix of order n)",
    r"An **$n$-array** is a rectangular family of entries $A = (a_{i_1 i_2 \dots i_n})$ with "
    r"$1 \le i_k \le d_k$; $(d_1, \dots, d_n)$ are its *dimensions* and $n$ its *order*. A scalar is the "
    r"array of order $0$. It is *regular* when all $d_k$ are equal. An array carries a declared index "
    r"symmetry: **rigid** (none), **cyclic** (invariant under cyclic permutations of its indices) or "
    r"**symmetric** (invariant under all permutations), verified against its entries; only regular arrays "
    r"can be cyclic or symmetric. Its value domain is read off the entries, never declared.",
    maturity="stable",
    sources=(
        SourceRef("hypergraph-paclet", "symbol:ArrayObject", "implementation"),
        SourceRef("heaps-of-fish", "section:Array Algebra", "definition"),
    ),
    tags=("array", "symmetry"),
)
class ArrayObject:
    """An array of any order with a verified index symmetry."""

    PROPERTIES = ("Order", "Dimensions", "Symmetry", "Domain", "Entries", "RegularQ")
    __slots__ = ("data", "symmetry")
    __hash__ = None  # mutable-looking value semantics; compare with ==

    def __init__(self, data: Any, symmetry: str = "Rigid") -> None:
        arr = to_ndarray(data)
        sym = normalize_symmetry(symmetry)
        if sym != "Rigid" and len(set(arr.shape)) > 1:
            raise ArrayError(
                "ArrayObject::irreg", f"an irregular array cannot be declared {sym}; only a regular array can"
            )
        if not has_symmetry(arr, sym):
            raise ArrayError(
                "ArrayObject::nosym", f"the given entries are not invariant under the index permutations of {sym}"
            )
        self.data = arr
        self.symmetry = sym

    @classmethod
    def _trusted(cls, arr: np.ndarray, symmetry: str = "Rigid") -> ArrayObject:
        """Wrap data already known to be valid with this symmetry (internal)."""
        obj = cls.__new__(cls)
        obj.data = tighten(arr) if arr.dtype == object else arr
        obj.symmetry = symmetry
        return obj

    # -- properties ------------------------------------------------------

    @property
    def order(self) -> int:
        return self.data.ndim

    @property
    def dims(self) -> list[int]:
        return list(self.data.shape)

    @property
    def regular(self) -> bool:
        return len(set(self.data.shape)) <= 1

    @property
    def domain(self) -> str | FiniteField:
        return infer_domain(self.data)

    @property
    def entries(self) -> Any:
        return entries_of(self.data)

    def prop(self, name: str) -> Any:
        values = {
            "Order": lambda: self.order,
            "Dimensions": lambda: self.dims,
            "Symmetry": lambda: self.symmetry,
            "Domain": lambda: self.domain,
            "Entries": lambda: self.entries,
            "RegularQ": lambda: self.regular,
            "Properties": lambda: list(self.PROPERTIES),
        }
        if name not in values:
            raise KeyError(f"unknown array property {name!r}; known: {', '.join(self.PROPERTIES)}")
        return values[name]()

    def __getitem__(self, key: str) -> Any:
        if isinstance(key, str):
            return self.prop(key)
        raise TypeError("array properties are accessed by name, e.g. a['Order']")

    # -- value semantics -------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ArrayObject):
            return NotImplemented
        return self.symmetry == other.symmetry and _same(self.data, other.data)

    def __repr__(self) -> str:
        sym = "" if self.symmetry == "Rigid" else f", {self.symmetry!r}"
        return f"ArrayObject({self.entries!r}{sym})"

    # -- arithmetic (entry by entry; see hypermatrix.algebra) --------------

    def __add__(self, other):
        from hypermatrix.algebra import array_add

        return array_add(self, other)

    def __radd__(self, other):
        from hypermatrix.algebra import array_add

        return array_add(other, self)

    def __mul__(self, other):
        from hypermatrix.algebra import array_times

        return array_times(self, other)

    def __rmul__(self, other):
        from hypermatrix.algebra import array_times

        return array_times(other, self)

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-1) * other if isinstance(other, ArrayObject) else self + _negate(other)

    def __rsub__(self, other):
        return (-1) * self + other

    def __truediv__(self, other):
        if isinstance(other, (ArrayObject, list, np.ndarray)):
            raise ArrayError("ArrayTimes::dims", "an array can be divided only by a scalar")
        return self * reciprocal(other)


def _negate(x: Any) -> Any:
    return [_negate(v) for v in x] if isinstance(x, list) else -x


def reciprocal(x: Any) -> Any:
    """1/x, exact for integers and fractions."""
    if isinstance(x, bool):
        raise TypeError("cannot divide by a boolean")
    if isinstance(x, (int, np.integer)):
        return Fraction(1, int(x))
    if isinstance(x, Fraction):
        return 1 / x
    if isinstance(x, (Number, sympy.Basic, FieldElement)):
        return 1 / x if not isinstance(x, sympy.Basic) else sympy.Integer(1) / x
    raise TypeError(f"cannot divide by {x!r}")


def as_array(x: Any) -> ArrayObject:
    return x if isinstance(x, ArrayObject) else ArrayObject(x)


@define(
    "hypermatrix.symbolic_array",
    "Symbolic array",
    r"`GenerateSymbolicArray(a, dims, sym)` is the array whose entry at $(i_1, \dots, i_n)$ is the "
    r"symbol $a_{i_1 \dots i_n}$, indices taken up to the declared symmetry, so that equivalent index "
    r"tuples name the same entry. With no dimensions it is the 0-array $a$.",
    maturity="working",
    depends_on=("hypermatrix.array",),
    sources=(SourceRef("hypergraph-paclet", "symbol:GenerateSymbolicArray", "implementation"),),
    tags=("array", "symbolic"),
)
def generate_symbolic_array(name: Any, dims: Any, symmetry: str = "Rigid") -> ArrayObject:
    if isinstance(name, sympy.Symbol):
        name = name.name
    if not isinstance(name, str) or not name:
        raise ArrayError("GenerateSymbolicArray::name", f"{name!r} is not a valid array name")
    d = [dims] if isinstance(dims, int) and not isinstance(dims, bool) else list(dims)
    if not all(isinstance(k, int) and not isinstance(k, bool) and k > 0 for k in d):
        raise ArrayError("GenerateSymbolicArray::dims", f"{dims!r} is not a list of positive integer index lengths")
    sym = normalize_symmetry(symmetry)
    if sym != "Rigid" and len(set(d)) > 1:
        raise ArrayError("GenerateSymbolicArray::irreg", f"an irregular array cannot be declared {sym}")
    if not d:
        return ArrayObject._trusted(np.array(sympy.Symbol(name), dtype=object))
    base = sympy.IndexedBase(name)
    arr = np.empty(d, dtype=object)
    for idx in itertools.product(*(range(1, k + 1) for k in d)):
        arr[tuple(i - 1 for i in idx)] = base[canonical_index(idx, sym)]
    return ArrayObject._trusted(arr, sym)
