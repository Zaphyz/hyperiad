"""Composition of hypergraphs as hypermatrix algebra: the bridge.

A composition rule (``frontier.hypergraphs.composition``) whose output is one edge on its input's
vertices is a plex diagram: the input edges are plexes named by their factors, the input vertices
off the output edge are contracted, and the output edge lists the free indices in order. Composing
hypergraphs by the rule and taking the adjacency hypermatrix gives the plex product of the
factors' adjacency arrays:

* ``Count`` multiplicity, non-injective matching, directed output: exactly the plex product;
* an unordered (cyclic) output edge: the product summed over the orbit of each index tuple under
  the output's symmetry (for a matrix, C + Cᵀ − diag C);
* injective vertex matching: the sum restricted to pairwise distinct indices;
* ``Boolean`` multiplicity: the support of the product (the product over the Boolean semiring);
* ``Labels``: the product over the semiring of formal sums of labels (tested with
  noncommuting symbols).

The claims below record these, with tests in ``hypermatrix-algebra/tests/test_composition_bridge.py``.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping

import numpy as np
from frontier.hypergraphs import Edge, Hypergraph, HypergraphRule, Vertex
from frontier.hypergraphs.composition import composition_factors
from frontier.hypergraphs.core import is_io, symmetry_orderings
from frontier.hypergraphs.labels import _composite
from frontier.registry import Evidence, EvidenceKind, claim, define

from hypermatrix.adjacency import EDGE_TO_ARRAY_SYMMETRY, adjacency_hypermatrix
from hypermatrix.arrays import ArrayError, ArrayObject
from hypermatrix.plex import PLEX_MARK, plex_product

_SOURCE = "hypermatrix-algebra/tests/test_composition_bridge.py"


def _factor_name(label) -> str:
    parts = _composite(label)
    return parts[0] if parts is not None else str(label)


def _plex_name(name: str, e: Edge, mixed: bool) -> str:
    return f"{name}[{e.arity},{e.symmetry}]" if mixed else name


def _output_edge(rule: HypergraphRule) -> Edge:
    if len(rule.output.edges) != 1:
        raise ArrayError("CompositionPlexDiagram::output", "a composition rule with one output edge is a plex diagram")
    out = rule.output.edges[0]
    if len(set(out.ids)) != len(out.ids):
        raise ArrayError("CompositionPlexDiagram::output", "the output edge repeats a vertex: no plex output lists it")
    if rule.new_vertices:
        raise ArrayError("CompositionPlexDiagram::vertices", "the rule creates vertices")
    return out


@define(
    "hypermatrix.composition_plex_diagram",
    "Plex diagram of a composition rule",
    r"A composition rule $L \to R$ with one output edge $e$ on $L$'s vertices is the plex diagram whose plexes "
    r"are $L$'s edges, named by their factors (one name per factor, arity and symmetry type), whose contracted "
    r"vertices are $L$'s vertices off $e$, and whose output is $e$'s vertices in order.",
    maturity="sketch",
    depends_on=("common.hypergraph_composition", "hypermatrix.plex_diagram"),
    tags=("plex", "composition"),
)
def composition_plex_diagram(rule: HypergraphRule, names: list[str] | None = None) -> tuple[Hypergraph, list]:
    """The plex diagram of a composition rule and its output order. ``names`` are the factor names
    (by default, the plain names the input edges carry)."""
    out = _output_edge(rule)
    if names is None:
        names = sorted({_factor_name(e.label) for e in rule.input.edges})
    parts = composition_factors(rule, names)
    kinds: dict[str, set] = {}
    for e, name, _ in parts:
        kinds.setdefault(name, set()).add((e.arity, e.symmetry))
    edges = [Edge(e.ids, e.symmetry, _plex_name(name, e, len(kinds[name]) > 1)) for e, name, _ in parts]
    free = set(out.ids)
    vertices = [Vertex(v.id) if v.id in free else Vertex(v.id, PLEX_MARK) for v in rule.input.vertices]
    return Hypergraph(vertices, edges), list(out.ids)


def _factor_arrays(rule: HypergraphRule, factors: Mapping[str, Hypergraph], vertices: list) -> dict[str, ArrayObject]:
    names = [str(n) for n in factors]
    parts = composition_factors(rule, names)
    kinds: dict[str, set] = {}
    for e, name, _ in parts:
        kinds.setdefault(name, set()).add((e.arity, e.symmetry))
    n = len(vertices)
    arrays: dict[str, ArrayObject] = {}
    for e, name, _ in parts:
        key = _plex_name(name, e, len(kinds[name]) > 1)
        if key in arrays:
            continue
        if is_io(e.symmetry):  # no array symmetry for blocks yet: a rigid array filled on each edge's orbit
            arrays[key] = ArrayObject(
                _orbit_counts([g for g in factors[name].edges if g.symmetry == e.symmetry], vertices, e), "Rigid"
            )
            continue
        sym = EDGE_TO_ARRAY_SYMMETRY[e.symmetry]
        on_v = Hypergraph(
            [Vertex(v) for v in vertices],
            [g for g in factors[name].edges if (g.arity, g.symmetry) == (e.arity, e.symmetry)],
        )
        found = [a for a in adjacency_hypermatrix(on_v).arrays if a.order == e.arity] if on_v.edges else []
        arrays[key] = found[0] if found else ArrayObject(np.zeros((n,) * e.arity, dtype=np.int64), sym)
    return arrays


def _orbit(idx: tuple, symmetry: str) -> set[tuple]:
    """The index tuples a symmetry type identifies with ``idx`` (each block within itself, for input–output)."""
    return {tuple(idx[p] for p in perm) for perm in symmetry_orderings(symmetry, len(idx))}


def _orbit_counts(edges: list[Edge], vertices: list, like: Edge) -> np.ndarray:
    """Counting array of ``edges`` (all of ``like``'s type): every tuple of each edge's orbit counts it once."""
    index = {v: k for k, v in enumerate(vertices)}
    data = np.zeros((len(vertices),) * like.arity, dtype=object)
    for g in edges:
        for o in _orbit(tuple(index[i] for i in g.ids), g.symmetry):
            data[o] += 1
    return data


def _orbit_sum(c: np.ndarray, symmetry: str) -> np.ndarray:
    if symmetry == "Directed":
        return c
    out = np.zeros_like(c)
    for idx in itertools.product(range(c.shape[0]), repeat=c.ndim):
        out[idx] = sum(c[o] for o in _orbit(idx, symmetry))
    return out


def _distinct_sum(rule: HypergraphRule, arrays: dict[str, ArrayObject], names: list[str], n: int) -> np.ndarray:
    """The plex product restricted to injective assignments of the rule's vertices."""
    diagram, output = composition_plex_diagram(rule, names)
    ids = diagram.vertex_ids
    out = np.zeros((n,) * len(output), dtype=object)
    data = {k: np.asarray(a.data) for k, a in arrays.items()}
    for assignment in itertools.permutations(range(n), len(ids)):
        at = dict(zip(ids, assignment, strict=True))
        term = 1
        for e in diagram.edges:
            term = term * data[e.label][tuple(at[i] for i in e.ids)]
        out[tuple(at[i] for i in output)] += term
    return out


def composition_adjacency_product(
    rule: HypergraphRule,
    factors: Mapping[str, Hypergraph],
    *,
    multiplicity: str = "Count",
    distinct_vertices: bool = False,
) -> ArrayObject:
    """The hypermatrix side of a composition: the plex product of the factors' adjacency arrays (on the
    union of their vertices, in order), symmetrised for an unordered or cyclic output edge, restricted
    to distinct indices when vertices match injectively, its support for ``Boolean``."""
    out_edge = _output_edge(rule)
    names = [str(n) for n in factors]
    vertices = list(dict.fromkeys(v for hg in factors.values() for v in hg.vertex_ids))
    arrays = _factor_arrays(rule, factors, vertices)
    if distinct_vertices:
        c = _distinct_sum(rule, arrays, names, len(vertices))
    else:
        diagram, output = composition_plex_diagram(rule, names)
        c = np.asarray(plex_product(diagram, arrays, output=output).data, dtype=object)
    c = _orbit_sum(c, out_edge.symmetry)
    if multiplicity == "Boolean":
        c = (c != 0).astype(np.int64)
    elif multiplicity != "Count":
        raise ArrayError("CompositionAdjacency::multiplicity", "the array side is computed for Count or Boolean")
    return ArrayObject(c.astype(np.int64), "Rigid")


def composition_adjacency(result: Hypergraph, rule: HypergraphRule, vertices: list) -> np.ndarray:
    """The counting adjacency array of a composition's result for the rule's output edge type, read as a
    plain (rigid) array over ``vertices``."""
    out = _output_edge(rule)
    n = len(vertices)
    index = {v: k for k, v in enumerate(vertices)}
    a = np.zeros((n,) * out.arity, dtype=np.int64)
    for e in result.edges:
        if (e.arity, e.symmetry) != (out.arity, out.symmetry):
            continue
        for o in _orbit(tuple(index[i] for i in e.ids), out.symmetry):
            a[o] += 1
    return a


claim(
    "hypermatrix.composition_is_plex_product",
    r"For a composition rule with one output edge, the adjacency array of the composition of $G_a, G_b, \ldots$ "
    r"(multiplicity Count, non-injective matching) is the plex product, by the rule's plex diagram, of the "
    r"factors' adjacency arrays, summed over the orbit of each index under the output edge's symmetry (no sum "
    r"for a directed output). Input edges may be directed, cyclic, unordered or input–output: a match is an edge "
    r"choice with a vertex map, so an unordered input edge counts once per orientation, as a symmetric array does "
    r"(an input–output edge's array is filled on its orbit under the product of its blocks' symmetries).",
    about=("common.hypergraph_composition", "hypermatrix.composition_plex_diagram", "hypermatrix.plex_product"),
    status="supported",
    evidence=[
        Evidence(EvidenceKind.NUMERICAL, "random multi-hypergraphs with loops and repeated vertices: graph "
                 "composition, fish, BM, TriForce, unordered inputs and outputs, one factor used twice",
                 source=_SOURCE),
        Evidence(EvidenceKind.NUMERICAL, "input–output edges (2026-10-31): multicategory plugging and polycategory "
                 "cuts with directed, cyclic and unordered blocks, 20 random factor pairs each", source=_SOURCE),
        Evidence(EvidenceKind.ARGUMENT, "a non-injective match is an assignment of host vertices to the rule's "
                 "vertices together with a choice of host edge for each input edge; the number of such choices at a "
                 "given assignment is the product of the adjacency entries, which is the summand of the plex product; "
                 "an output edge of symmetry S at a tuple is counted at every tuple of its S-orbit"),
    ],
)  # fmt: skip

claim(
    "hypermatrix.composition_variants",
    r"Under injective vertex matching the composition's adjacency array is the plex product summed over "
    r"pairwise distinct indices only; with multiplicity Boolean it is the support of the plex product (the "
    r"product over the Boolean semiring); with multiplicity Labels the formal sum of labels on each tuple is "
    r"the plex product over the semiring of formal sums of label words.",
    about=("common.hypergraph_composition", "hypermatrix.composition_plex_diagram"),
    status="supported",
    evidence=[Evidence(EvidenceKind.NUMERICAL, "random multi-hypergraphs, all three variants", source=_SOURCE)],
)

__all__ = [
    "composition_adjacency",
    "composition_adjacency_product",
    "composition_plex_diagram",
]
