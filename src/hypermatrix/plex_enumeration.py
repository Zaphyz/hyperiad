"""Enumerating closed plex products: k plexes of order n, n free indices.

A closed plex product of order n and arity k is a plex diagram with k directed plexes of order n
(one per argument, in argument order) and exactly n free vertices: on regular n-arrays of size d
it is a k-ary operation ``A^k -> A`` with ``A`` the n-arrays over ``[d]``. Its combinatorial data
is a :class:`PlexShape`: a set partition of the ``k*n`` slots ``(argument j, position p)`` into
vertices (a slot's vertex is the index it carries), and the output sequence of n distinct
vertices. Contracted vertices are the others. Since the plexes are labelled by their arguments
and directed, two shapes give the same diagram only if they are equal, so enumerating shapes
needs no isomorphism test at all.

Groups act on shapes (``act``) and give the coarser equivalences:

* ``"strict"``      -- the shape itself: the same k-ary operation (for every size d >= 2);
* ``"isomorphism"`` -- positions permuted, the same permutation in every plex and the output:
  the algebras ``(A, mu)`` are isomorphic through transposing every array;
* ``"parastrophy"`` -- positions and arguments permuted: isomorphic up to reordering arguments;
* ``"hypergraph"``  -- arguments and output order permuted: isomorphism of the plex diagrams as
  labelled hypergraphs, with labels up to renaming and the output order forgotten (what
  :func:`frontier.hypergraphs.canonical_hypergraph` identifies).

On basis arrays a plex product sends basis tuples to one basis array or to 0: it is a partial
k-ary operation on the index set ``[d]^n`` (:func:`operation_hypergraph` draws it as a directed
hypergraph of arity k+1).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from frontier.hypergraphs import Edge, Hypergraph, Vertex
from frontier.registry import SourceRef, define

from hypermatrix.plex import PLEX_MARK

EQUIVALENCES = ("strict", "isomorphism", "parastrophy", "hypergraph")


def _relabel(blocks: tuple[int, ...], output: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Renumber vertices by first occurrence among the slots (restricted growth form)."""
    new: dict[int, int] = {}
    rgs = tuple(new.setdefault(b, len(new)) for b in blocks)
    return rgs, tuple(new[b] for b in output)


@dataclass(frozen=True, order=True)
class PlexShape:
    """A closed plex product: slot partition (restricted growth form) and output vertices."""

    order: int  # n: the order of each plex and of the result
    arity: int  # k: the number of plexes (arguments)
    blocks: tuple[int, ...]  # vertex of slot (j, p) at index j * n + p
    output: tuple[int, ...]  # the n free vertices, in output order

    @property
    def vertex_count(self) -> int:
        return max(self.blocks, default=-1) + 1

    @property
    def contracted(self) -> list[int]:
        return [b for b in range(self.vertex_count) if b not in self.output]

    def edges(self) -> list[tuple[int, ...]]:
        n = self.order
        return [self.blocks[j * n : (j + 1) * n] for j in range(self.arity)]

    def connected(self) -> bool:
        parent = list(range(self.vertex_count))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for e in self.edges():
            for v in e[1:]:
                parent[find(v)] = find(e[0])
        return len({find(v) for v in range(self.vertex_count)}) <= 1

    def diagonal(self) -> bool:
        """Whether some plex carries the same index twice (a diagonal)."""
        return any(len(set(e)) < len(e) for e in self.edges())

    def act(self, arguments: tuple[int, ...], outputs: tuple[int, ...], positions: tuple[int, ...]) -> PlexShape:
        """The shape with argument j taken from old argument ``arguments[j]``, output q from old output
        ``outputs[q]``, and position p of every plex and of the output from old position ``positions[p]``."""
        n = self.order
        blocks = tuple(self.blocks[arguments[j] * n + positions[p]] for j in range(self.arity) for p in range(n))
        out = tuple(self.output[positions[outputs[q]]] for q in range(n))
        rgs, out = _relabel(blocks, out)
        return PlexShape(n, self.arity, rgs, out)

    def canonical(self, equivalence: str = "strict") -> PlexShape:
        """The least shape in this shape's class under ``equivalence``."""
        n, k = self.order, self.arity
        ident_k, ident_n = tuple(range(k)), tuple(range(n))
        groups = {
            "strict": ([ident_k], [ident_n], [ident_n]),
            "isomorphism": ([ident_k], [ident_n], list(itertools.permutations(ident_n))),
            "parastrophy": (list(itertools.permutations(ident_k)), [ident_n], list(itertools.permutations(ident_n))),
            "hypergraph": (list(itertools.permutations(ident_k)), list(itertools.permutations(ident_n)), [ident_n]),
        }
        if equivalence not in groups:
            raise ValueError(f"unknown equivalence {equivalence!r}; one of {', '.join(EQUIVALENCES)}")
        lams, sigs, taus = groups[equivalence]
        return min(self.act(lam, sig, tau) for lam in lams for sig in sigs for tau in taus)

    def spec(self) -> tuple[list[list[int]], list[int]]:
        """The index specification of the product, for :func:`hypermatrix.array_multiply`."""
        return [list(e) for e in self.edges()], list(self.output)

    def diagram(self, names: list[Any] | None = None) -> Hypergraph:
        """The plex diagram: free vertices named 1..n in output order, contracted ones n+1, ...;
        plexes named ``x1, ..., xk`` in argument order (so :func:`plex_names` lists them in order)."""
        rename = {b: q + 1 for q, b in enumerate(self.output)}
        for b in self.contracted:
            rename[b] = len(rename) + 1
        names = names or [f"x{j + 1}" for j in range(self.arity)]
        vertices = [Vertex(rename[b], None if b in self.output else PLEX_MARK) for b in range(self.vertex_count)]
        edges = [Edge([rename[b] for b in e], "Directed", names[j]) for j, e in enumerate(self.edges())]
        return Hypergraph(vertices, edges)


def set_partitions(m: int) -> Iterator[tuple[int, ...]]:
    """All set partitions of ``m`` slots, as restricted growth strings."""
    if m == 0:
        yield ()
        return

    def grow(prefix: list[int], top: int) -> Iterator[tuple[int, ...]]:
        if len(prefix) == m:
            yield tuple(prefix)
            return
        for b in range(top + 2):
            prefix.append(b)
            yield from grow(prefix, max(top, b))
            prefix.pop()

    yield from grow([0], 0)


def all_plex_shapes(order: int, arity: int, *, connected: bool = True, diagonals: bool = False) -> Iterator[PlexShape]:
    """Every closed plex product of this order and arity (strict classes), filtered as asked."""
    for rgs in set_partitions(order * arity):
        count = max(rgs) + 1
        for out in itertools.permutations(range(count), order):
            shape = PlexShape(order, arity, rgs, out)
            if connected and not shape.connected():
                continue
            if not diagonals and shape.diagonal():
                continue
            yield shape


@define(
    "hypermatrix.plex_product_census",
    "Enumeration of plex products",
    r"The closed plex products of order $n$ and arity $k$ (operations $A^k \to A$ on regular $n$-arrays) "
    r"are the pairs (set partition of the $kn$ argument slots into indices, ordered choice of $n$ of "
    r"them as the output). They are enumerated up to equality of operation (`strict`), up to a common "
    r"permutation of index positions (`isomorphism`, which gives isomorphic algebras), additionally up to "
    r"permuting arguments (`parastrophy`), or up to isomorphism of the plex diagrams as labelled "
    r"hypergraphs (`hypergraph`).",
    maturity="stable",
    depends_on=("hypermatrix.plex_diagram", "hypermatrix.plex_product"),
    sources=(SourceRef("hypergraph-paclet", "path:reference/PlexAlgebra.wl#L279-L283", "implementation"),),
    tags=("plex", "enumeration"),
)
def enumerate_plex_products(
    order: int, arity: int, *, equivalence: str = "strict", connected: bool = True, diagonals: bool = False
) -> list[PlexShape]:
    """One representative (the least shape) per class, sorted."""
    reps = {s.canonical(equivalence) for s in all_plex_shapes(order, arity, connected=connected, diagonals=diagonals)}
    return sorted(reps)


def operation_table(shape: PlexShape, d: int) -> dict[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """The partial operation on the index set ``[d]^n``: argument indices -> result index, where defined.

    A plex product sends basis arrays ``e_{x_1}, ..., e_{x_k}`` to ``e_y`` or to 0; it is defined
    exactly when the slot values agree on every vertex, one product per assignment of values to
    the vertices.
    """
    out = {}
    for values in itertools.product(range(d), repeat=shape.vertex_count):
        args = tuple(tuple(values[b] for b in e) for e in shape.edges())
        out[args] = tuple(values[b] for b in shape.output)
    return out


def operation_hypergraph(shape: PlexShape, d: int) -> Hypergraph:
    """The partial operation as a directed hypergraph on ``[d]^n``: an edge ``(x_1, ..., x_k, y)`` per product.

    Two plex products induce algebras isomorphic through a permutation of the basis exactly when
    these hypergraphs are isomorphic.
    """
    points = list(itertools.product(range(d), repeat=shape.order))
    edges = [Edge([*args, y], "Directed") for args, y in operation_table(shape, d).items()]
    return Hypergraph(points, edges)
