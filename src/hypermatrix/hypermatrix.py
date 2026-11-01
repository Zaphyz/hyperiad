"""Hypermatrices: canonically ordered lists of arrays, ported from the paclet's ``src/Hypermatrix.wl``.

A hypermatrix holds its arrays sorted by order, then dimensions, then symmetry from the most
constrained to the least (Symmetric, Cyclic, Rigid); arrays alike in all three are put in a
fixed order of their entries, so a hypermatrix depends only on the set of arrays it was given.
"""

from __future__ import annotations

from typing import Any

from frontier.hypergraphs.core import order_key
from frontier.registry import SourceRef, define

from hypermatrix.arrays import ArrayError, ArrayObject

SYMMETRY_ORDER = ("Symmetric", "Cyclic", "Rigid")


def _flat(entries: Any) -> list:
    return [x for e in entries for x in _flat(e)] if isinstance(entries, list) else [entries]


def array_sort_key(a: ArrayObject) -> tuple:
    """Order, dimensions, symmetry rank, then the entries (the paclet's arraySortKey)."""
    return (a.order, tuple(a.dims), SYMMETRY_ORDER.index(a.symmetry), tuple(order_key(x) for x in _flat(a.entries)))


def _to_array(x: Any) -> ArrayObject:
    if isinstance(x, ArrayObject):
        return x
    try:
        return ArrayObject(x)
    except ArrayError as e:
        raise ArrayError("Hypermatrix::arrays", f"{x!r} is not an array: {e}") from None


@define(
    "hypermatrix.hypermatrix",
    "Hypermatrix",
    r"A hypermatrix is a finite list of arrays held in canonical order: by order, then dimensions, then "
    r"symmetry (Symmetric before Cyclic before Rigid). Arrays of different orders and symmetries sit side "
    r"by side, as the edges of different arities and symmetry types of a hypergraph do.",
    maturity="working",
    depends_on=("hypermatrix.array",),
    sources=(SourceRef("hypergraph-paclet", "symbol:Hypermatrix", "implementation"),),
    tags=("array", "hypermatrix"),
)
class Hypermatrix:
    """``Hypermatrix([a1, a2, ...])`` or ``Hypermatrix(a1, a2, ...)``; raw data is promoted to arrays."""

    PROPERTIES = ("Arrays", "Length", "Orders", "Dimensions", "Symmetries", "Domains", "Entries")
    __slots__ = ("arrays",)
    __hash__ = None

    def __init__(self, *args: Any) -> None:
        items = list(args[0]) if len(args) == 1 and isinstance(args[0], (list, tuple)) else list(args)
        arrays = [_to_array(x) for x in items]
        object.__setattr__(self, "arrays", tuple(sorted(arrays, key=array_sort_key)))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Hypermatrix is immutable")

    def __len__(self) -> int:
        return len(self.arrays)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Hypermatrix) and self.arrays == other.arrays

    def __repr__(self) -> str:
        return f"Hypermatrix({list(self.arrays)!r})"

    def normal(self) -> list:
        """The entries of each array, in order (the paclet's ``Normal``)."""
        return [a.entries for a in self.arrays]

    def prop(self, name: str) -> Any:
        table = {
            "Arrays": lambda: list(self.arrays),
            "Length": lambda: len(self.arrays),
            "Orders": lambda: [a.order for a in self.arrays],
            "Dimensions": lambda: [a.dims for a in self.arrays],
            "Symmetries": lambda: [a.symmetry for a in self.arrays],
            "Domains": lambda: [a.domain for a in self.arrays],
            "Entries": self.normal,
            "Properties": lambda: list(self.PROPERTIES),
        }
        if name not in table:
            raise KeyError(f"unknown hypermatrix property {name!r}; known: {', '.join(self.PROPERTIES)}")
        return table[name]()

    def __getitem__(self, key: str) -> Any:
        if isinstance(key, str):
            return self.prop(key)
        raise TypeError("hypermatrix properties are accessed by name, e.g. hm['Orders']")
