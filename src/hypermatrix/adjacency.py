"""The hypergraph–hypermatrix correspondence, ported from the paclet's ``src/Adjacency.wl``.

A hypergraph is a multiset of edges, each with an arity and a symmetry type; an array has an
order and a symmetry. The two vocabularies line up exactly:

    an edge of arity k   <->  an array of order k
    "Directed"           <->  "Rigid"
    "Cyclic"             <->  "Cyclic"
    "Unordered"          <->  "Symmetric"

The edges are grouped by arity and symmetry type together, and each group becomes one array
whose indices run over the vertices in ``VertexList`` order. The entry at an index tuple is the
number of edges of the group sitting on it, read at the least tuple of its orbit under the
symmetry, so every tuple of an orbit reads the same value and the array carries its symmetry by
construction. With ``labeled=True`` the entries are the edge labels (``None`` for an unlabelled
edge, 0 where there is none), which needs the hypergraph to be simple.
"""

from __future__ import annotations

import itertools
from collections import Counter
from typing import Any

import numpy as np
from frontier.hypergraphs import Hypergraph, simple_hypergraph_q
from frontier.registry import Evidence, EvidenceKind, SourceRef, claim, define

from hypermatrix.arrays import ArrayError, ArrayObject, canonical_index
from hypermatrix.hypermatrix import Hypermatrix

EDGE_TO_ARRAY_SYMMETRY = {"Directed": "Rigid", "Cyclic": "Cyclic", "Unordered": "Symmetric"}
ARRAY_TO_EDGE_SYMMETRY = {v: k for k, v in EDGE_TO_ARRAY_SYMMETRY.items()}


def _group_array(k: int, sym: str, items: list[tuple[tuple[int, ...], Any]], n: int, labeled: bool) -> ArrayObject:
    values: dict = dict(items) if labeled else Counter(idx for idx, _ in items)
    if k == 0:  # an edge of arity zero has no vertices to index: the group is one value
        return ArrayObject(items[0][1] if labeled else len(items), sym)
    data = np.empty((n,) * k, dtype=object)
    for idx in itertools.product(range(n), repeat=k):
        data[idx] = values.get(canonical_index(idx, sym), 0)
    return ArrayObject(data, sym)


def adjacency_size(hg: Hypergraph) -> int:
    """Entries the adjacency hypermatrix has: n^k summed over its groups."""
    n = len(hg.vertices)
    return sum(n**k for k in {(e.arity, e.symmetry): e.arity for e in hg.edges}.values())


@define(
    "hypermatrix.adjacency_hypermatrix",
    "Adjacency hypermatrix",
    r"The adjacency hypermatrix of a hypergraph on vertices $v_1, \dots, v_n$ (in vertex-list order) has "
    r"one array per arity $k$ and symmetry type of its edges: an array of order $k$ over $\{1..n\}$, Rigid "
    r"for directed edges, Cyclic for cyclic, Symmetric for unordered, whose entry at $(i_1, \dots, i_k)$ "
    r"counts the edges of that group equivalent to $(v_{i_1}, \dots, v_{i_k})$ under the symmetry. "
    r"`Labeled = true` writes each edge's label instead of a count (simple hypergraphs only).",
    maturity="working",
    depends_on=("common.hypergraph", "hypermatrix.hypermatrix"),
    sources=(SourceRef("hypergraph-paclet", "symbol:AdjacencyHypermatrix", "implementation"),),
    tags=("hypergraph", "hypermatrix", "adjacency"),
)
def adjacency_hypermatrix(hg: Hypergraph, *, labeled: bool = False) -> Hypermatrix:
    if not isinstance(hg, Hypergraph):
        raise ArrayError("AdjacencyHypermatrix::hg", f"{hg!r} is not a hypergraph")
    if labeled and not simple_hypergraph_q(hg):
        raise ArrayError(
            "AdjacencyHypermatrix::nonsimple",
            "labels can be the entries of a hypermatrix only when no edge is repeated, since one entry cannot "
            "hold two labels; this hypergraph has a repeated edge, labels aside",
        )
    index = {v: k for k, v in enumerate(hg.vertex_ids)}
    groups: dict[tuple[int, str], list] = {}
    for e in hg.edges:
        if e.symmetry not in EDGE_TO_ARRAY_SYMMETRY:
            raise ArrayError(
                "AdjacencyHypermatrix::io",
                f"{e.symmetry} edges have no adjacency array yet: arrays carry no symmetry within blocks of indices",
            )
        sym = EDGE_TO_ARRAY_SYMMETRY[e.symmetry]
        groups.setdefault((e.arity, sym), []).append((canonical_index(tuple(index[i] for i in e.ids), sym), e.label))
    n = len(index)
    return Hypermatrix([_group_array(k, sym, items, n, labeled) for (k, sym), items in groups.items()])


def hypermatrix_edges(hm: Hypermatrix) -> list[tuple[tuple[int, ...], str]]:
    """Read the edge multiset back off a counting adjacency hypermatrix: (canonical index, edge symmetry)."""
    out = []
    for a in hm.arrays:
        sym = ARRAY_TO_EDGE_SYMMETRY[a.symmetry]
        if a.order == 0:
            out += [((), sym)] * int(a.entries)
            continue
        data = a.data
        for idx in itertools.product(*(range(d) for d in a.dims)):
            if idx == canonical_index(idx, a.symmetry):
                out += [(idx, sym)] * int(data[idx])
    return sorted(out)


claim(
    "hypermatrix.adjacency_round_trip",
    r"The counting adjacency hypermatrix determines the edge multiset of the skeleton: reading each array "
    r"at the least index of every orbit, with its multiplicity and the edge symmetry its array symmetry "
    r"names, gives back exactly the edges (as vertex positions).",
    about=("hypermatrix.adjacency_hypermatrix",),
    status="supported",
    evidence=[
        Evidence(
            EvidenceKind.NUMERICAL,
            "held on 300 random hypergraphs (arities 0-3, all three symmetry types, repeated edges)",
            source="hypermatrix-algebra/tests/test_adjacency.py",
        )
    ],
)

claim(
    "hypermatrix.adjacency_canonical_invariant",
    r"The adjacency hypermatrix depends on how the vertices are named (a renaming permutes the indices in "
    r"every slot at once), but composed with the canonical form of the skeleton it is an isomorphism "
    r"invariant of the skeleton; for a hypergraph with at least one edge it is a complete one.",
    about=("hypermatrix.adjacency_hypermatrix", "common.canonical_hypergraph"),
    status="supported",
    evidence=[
        Evidence(
            EvidenceKind.NUMERICAL,
            "200 random hypergraphs renamed at random agree after canonicalizing; the paclet's renaming cases",
            source="hypermatrix-algebra/tests/test_adjacency.py",
        ),
        Evidence(
            EvidenceKind.ARGUMENT,
            "the canonical skeleton names its vertices 1..n, so the round trip recovers it exactly from the "
            "edge positions and n (the common dimension); only n is lost when there are no edges",
        ),
    ],
)
