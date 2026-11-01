"""Canonical forms and isomorphism of hypergraphs.

Two hypergraphs are **isomorphic** when a bijection of vertex identifiers carries the
edge multiset of one onto the other, preserving each edge's symmetry type (so a
directed edge must keep its order, a cyclic one its order up to rotation). Labels are
significant only up to renaming, as in the reference paclet (``src/Hypergraph.wl``):
what must agree is which elements share a label. Vertex labels and edge labels are put
in correspondence independently, and an unlabelled element corresponds only to an
unlabelled one. With ``labeled=False`` the skeletons are compared instead.

The **canonical form** renames the vertices 1, 2, ... and replaces labels by tokens
``ℓ1, ℓ2, ...``. It is a function of the isomorphism class alone, so two hypergraphs
are isomorphic exactly when their canonical forms are equal.

The paclet minimizes over all vertex orderings within classes of a vertex invariant.
Here the minimum is taken over the leaves of an individualization-refinement search
(the method of nauty): colour refinement on the vertex-edge incidence structure,
then branching on the vertices of the first non-singleton colour class. Both are
canonical; they choose different representatives of a class, as the paclet's own two
methods do. ``method="bruteforce"`` (all n! orderings) is kept for cross-checks.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

from frontier.hypergraphs.core import (
    Edge,
    Hypergraph,
    HypergraphError,
    Vertex,
    is_io,
    order_key,
    symmetry_blocks,
    symmetry_rank,
)
from frontier.registry import SourceRef, define

MAX_LEAVES = 200_000


@dataclass(frozen=True, order=True)
class LabelToken:
    """The canonical stand-in for a label: labels count only up to renaming."""

    index: int

    def __str__(self) -> str:
        return f"ℓ{self.index}"

    def __repr__(self) -> str:
        return f"LabelToken({self.index})"

    def order_key(self) -> tuple:
        return (self.index,)


@dataclass
class CanonicalData:
    form: Hypergraph  # the canonical hypergraph
    ordering: list[Hashable]  # original vertex identifiers, in canonical order (vertex k+1 is ordering[k])
    vertex_tokens: dict[Hashable, LabelToken] = field(default_factory=dict)  # original vertex label -> token
    edge_tokens: dict[Hashable, LabelToken] = field(default_factory=dict)  # original edge label -> token
    leaves: int = 0  # orderings examined


# -- certificates ----------------------------------------------------------------


def _rotate_min(seq: tuple) -> tuple:
    return min((seq[k:] + seq[:k] for k in range(len(seq))), default=seq)


def _edge_signature(ids: tuple[int, ...], sym: str) -> tuple[int, ...]:
    if is_io(sym):  # each block in the canonical order of its own type
        return tuple(x for s, i, n in symmetry_blocks(sym, len(ids)) for x in _edge_signature(ids[i : i + n], s))
    if sym == "Unordered":
        return tuple(sorted(ids))
    if sym == "Cyclic":
        return _rotate_min(ids)
    return ids


def _tokens(labels: Sequence[Any], keys: Sequence[Any]) -> dict[Hashable, int]:
    """Rank labels by the sorted keys of the elements carrying them (the paclet's labelTokens)."""
    groups: dict[Hashable, list] = {}
    for lab, key in zip(labels, keys, strict=True):
        if lab is not None:
            groups.setdefault(lab, []).append(key)
    ranked = sorted(groups, key=lambda lab: sorted(groups[lab]))
    return {lab: k + 1 for k, lab in enumerate(ranked)}


def _certificate(hg: Hypergraph, ordering: Sequence[Hashable], labeled: bool):
    """The relabelled hypergraph as a comparable tuple, and the label tokens it used."""
    index = {vid: k + 1 for k, vid in enumerate(ordering)}
    label_of = {v.id: v.label for v in hg.vertices}
    sigs = [(_edge_signature(tuple(index[i] for i in e.ids), e.symmetry), symmetry_rank(e.symmetry)) for e in hg.edges]
    if not labeled:
        return (len(ordering), (), tuple(sorted((s, 0) for s in sigs))), {}, {}
    vtok = _tokens([label_of[vid] for vid in ordering], list(range(1, len(ordering) + 1)))
    etok = _tokens([e.label for e in hg.edges], sigs)
    vpart = tuple(vtok.get(label_of[vid], 0) for vid in ordering)
    epart = tuple(sorted((s, etok.get(e.label, 0)) for s, e in zip(sigs, hg.edges, strict=True)))
    return (len(ordering), vpart, epart), vtok, etok


# -- colour refinement --------------------------------------------------------------


def _block_role(sym: str, cols: tuple, positions: list[int]) -> tuple:
    """How a block of an input–output edge sees a vertex at ``positions`` within it (colours ``cols``)."""
    if not positions:
        return ()
    if sym == "Directed":
        return (0, *positions)
    if sym == "Unordered":
        return (1, len(positions))
    return (2, *min(cols[p:] + cols[:p] for p in positions))


def _io_role(e: Edge, vid: Hashable) -> tuple:
    """The positions of a vertex in each block of an input–output edge, up to that block's symmetry."""
    out = []
    for s, i, n in symmetry_blocks(e.symmetry, e.arity):
        block = e.ids[i : i + n]
        if s == "Directed":
            out.append(tuple(p for p, x in enumerate(block) if x == vid))
        else:
            out.append((block.count(vid),))
    return tuple(out)


def _ranks(signatures: dict[Hashable, tuple]) -> dict[Hashable, int]:
    distinct = sorted(set(signatures.values()))
    rank = {s: k for k, s in enumerate(distinct)}
    return {v: rank[s] for v, s in signatures.items()}


def _initial_colours(hg: Hypergraph, labeled: bool) -> dict[Hashable, int]:
    """Invariant start: a vertex's label multiplicity and the shapes of the edges at it."""
    vmult = Counter(v.label for v in hg.vertices if v.label is not None)
    emult = Counter(e.label for e in hg.edges if e.label is not None)
    sig: dict[Hashable, tuple] = {}
    for v in hg.vertices:
        own = vmult[v.label] if labeled and v.label is not None else 0
        incident = []
        for e in hg.edges:
            if v.id in e.ids:
                lab = emult[e.label] if labeled and e.label is not None else 0
                role = (
                    tuple(p for p, i in enumerate(e.ids) if i == v.id)
                    if e.symmetry == "Directed"
                    else _io_role(e, v.id)
                    if is_io(e.symmetry)
                    else e.ids.count(v.id)
                )
                incident.append((e.arity, symmetry_rank(e.symmetry), lab, role if isinstance(role, tuple) else (role,)))
        sig[v.id] = (own, tuple(sorted(incident)))
    return _ranks(sig)


def _refine(hg: Hypergraph, colour: dict[Hashable, int], labeled: bool, emult: Counter) -> dict[Hashable, int]:
    """Refine until stable: each vertex is told the coloured shapes of the edges at it."""
    while True:
        incident: dict[Hashable, list] = {vid: [] for vid in colour}
        for e in hg.edges:
            cols = tuple(colour[i] for i in e.ids)
            lab = emult[e.label] if labeled and e.label is not None else 0
            if e.symmetry == "Directed":
                ecol = cols
            elif e.symmetry == "Unordered":
                ecol = tuple(sorted(cols))
            elif is_io(e.symmetry):
                ecol = _edge_signature(cols, e.symmetry)
            else:
                ecol = _rotate_min(cols)
            head = (e.arity, symmetry_rank(e.symmetry), lab, ecol)
            for vid in set(e.ids):
                positions = tuple(p for p, i in enumerate(e.ids) if i == vid)
                if e.symmetry == "Directed":
                    role = positions
                elif is_io(e.symmetry):  # within each block, as that block's type reads it
                    role = tuple(
                        _block_role(s, cols[i : i + n], [p - i for p in positions if i <= p < i + n])
                        for s, i, n in symmetry_blocks(e.symmetry, e.arity)
                    )
                elif e.symmetry == "Unordered":
                    role = (len(positions),)
                else:  # the colours read round the edge from each occurrence of the vertex
                    role = min(cols[p:] + cols[:p] for p in positions)
                incident[vid].append((head, role))
        new = _ranks({vid: (colour[vid], tuple(sorted(incident[vid]))) for vid in colour})
        if len(set(new.values())) == len(set(colour.values())):
            return new
        colour = new


def _orbit_rep(vid: Hashable, automorphisms: list[dict], fixed: Sequence[Hashable]) -> Hashable:
    """The least element (in ``order_key``) of ``vid``'s orbit under the automorphisms fixing ``fixed``."""
    gens = [g for g in automorphisms if all(g[u] == u for u in fixed)]
    orbit, frontier = {vid}, [vid]
    while frontier:
        u = frontier.pop()
        for g in gens:
            w = g[u]
            if w not in orbit:
                orbit.add(w)
                frontier.append(w)
    return min(orbit, key=order_key)


def _search(hg: Hypergraph, labeled: bool, max_leaves: int):
    """Individualization-refinement with pruning by the automorphisms met along the way.

    The canonical certificate is the least over the leaves of the search tree. A leaf whose
    certificate equals the best one so far yields an automorphism (labels allowed to be
    renamed); a child of a node that such an automorphism, fixing the node's individualized
    vertices, carries onto an explored sibling is skipped, since its leaves are images of
    leaves already seen and have the same certificates.
    """
    emult = Counter(e.label for e in hg.edges if e.label is not None)
    best: list = [None, None, None, None]  # certificate, ordering, vertex tokens, edge tokens
    automorphisms: list[dict] = []
    leaves = 0

    def visit(colour: dict[Hashable, int], prefix: tuple) -> None:
        nonlocal leaves
        colour = _refine(hg, colour, labeled, emult)
        cells: dict[int, list] = {}
        for vid, c in colour.items():
            cells.setdefault(c, []).append(vid)
        target = next((cells[c] for c in sorted(cells) if len(cells[c]) > 1), None)
        if target is None:
            leaves += 1
            if leaves > max_leaves:
                raise HypergraphError(
                    "CanonicalHypergraph::large",
                    f"the search exceeded {max_leaves} orderings; the hypergraph is too symmetric",
                )
            ordering = sorted(colour, key=colour.__getitem__)
            cert, vtok, etok = _certificate(hg, ordering, labeled)
            if best[0] is None or cert < best[0]:
                best[:] = [cert, ordering, vtok, etok]
            elif cert == best[0]:
                automorphisms.append(dict(zip(ordering, best[1], strict=True)))
            return
        explored: list[Hashable] = []
        for vid in sorted(target, key=order_key):
            if explored:
                reps = {_orbit_rep(u, automorphisms, prefix) for u in explored}
                if _orbit_rep(vid, automorphisms, prefix) in reps:
                    continue
            explored.append(vid)
            # individualize: vid takes the lowest colour of its cell, the rest keep a colour above it
            branch = {u: 2 * c + (0 if u == vid else 1) if c == colour[vid] else 2 * c for u, c in colour.items()}
            visit(_ranks({u: (b,) for u, b in branch.items()}), (*prefix, vid))

    visit(_initial_colours(hg, labeled), ())
    return best, leaves


def _brute_force(hg: Hypergraph, labeled: bool, max_leaves: int):
    ids = hg.vertex_ids
    if math.factorial(len(ids)) > max_leaves:
        raise HypergraphError("CanonicalHypergraph::large", f"{len(ids)}! orderings exceed {max_leaves}")
    best: list = [None, None, None, None]
    for ordering in itertools.permutations(ids):
        cert, vtok, etok = _certificate(hg, ordering, labeled)
        if best[0] is None or cert < best[0]:
            best[:] = [cert, list(ordering), vtok, etok]
    return best, math.factorial(len(ids))


def canonical_data(
    hg: Hypergraph, *, labeled: bool = True, method: str = "automatic", max_leaves: int = MAX_LEAVES
) -> CanonicalData:
    """The canonical form of ``hg`` with the ordering and label tokens that produce it."""
    if not isinstance(hg, Hypergraph):
        raise HypergraphError("Hypergraph::spec", f"{hg!r} is not a hypergraph")
    m = method.lower()
    if m not in ("automatic", "refine", "bruteforce"):
        raise HypergraphError("CanonicalHypergraph::method", f"unknown method {method!r}: use Automatic or BruteForce")
    if not hg.vertices:
        return CanonicalData(Hypergraph([]), [], {}, {}, 1)
    (cert, ordering, vtok, etok), leaves = (
        _brute_force(hg, labeled, max_leaves) if m == "bruteforce" else _search(hg, labeled, max_leaves)
    )
    index = {vid: k + 1 for k, vid in enumerate(ordering)}
    label_of = {v.id: v.label for v in hg.vertices}
    vertices = [
        Vertex(index[vid], LabelToken(vtok[label_of[vid]]) if labeled and label_of[vid] is not None else None)
        for vid in ordering
    ]
    by_index = {v.id: v for v in vertices}
    edges = [
        Edge(
            [by_index[index[i]] for i in e.ids],
            e.symmetry,
            LabelToken(etok[e.label]) if labeled and e.label is not None else None,
        )
        for e in hg.edges
    ]
    return CanonicalData(
        Hypergraph(vertices, edges),
        list(ordering),
        {lab: LabelToken(k) for lab, k in vtok.items()},
        {lab: LabelToken(k) for lab, k in etok.items()},
        leaves,
    )


@define(
    "common.canonical_hypergraph",
    "Canonical hypergraph",
    r"`CanonicalHypergraph(hg)` is a representative of the isomorphism class of `hg`: vertices renamed "
    r"$1, \dots, n$ and labels replaced by tokens $\ell_1, \ell_2, \dots$ (labels count only up to "
    r"renaming). Two hypergraphs are isomorphic exactly when their canonical forms are equal. "
    r"`Labeled = false` canonicalizes the skeleton.",
    maturity="working",
    depends_on=("common.hypergraph", "common.hypergraph_isomorphism"),
    sources=(SourceRef("hypergraph-paclet", "symbol:CanonicalHypergraph", "implementation"),),
    tags=("hypergraph", "isomorphism"),
)
def canonical_hypergraph(hg: Hypergraph, *, labeled: bool = True, method: str = "automatic") -> Hypergraph:
    return canonical_data(hg, labeled=labeled, method=method).form


def _coarse(hg: Hypergraph, labeled: bool) -> tuple:
    """Cheap invariants, for early rejection."""
    profile = (
        (
            sum(v.label is None for v in hg.vertices),
            tuple(sorted(Counter(v.label for v in hg.vertices if v.label is not None).values())),
            sum(e.label is None for e in hg.edges),
            tuple(sorted(Counter(e.label for e in hg.edges if e.label is not None).values())),
        )
        if labeled
        else ()
    )
    return (len(hg.vertices), len(hg.edges), tuple(sorted(e.arity for e in hg.edges)),
            tuple(sorted(symmetry_rank(e.symmetry) for e in hg.edges)), profile)  # fmt: skip


@define(
    "common.hypergraph_isomorphism",
    "Hypergraph isomorphism",
    r"Hypergraphs $G, H$ are isomorphic when a bijection $\varphi$ of vertices carries the edge multiset "
    r"of $G$ onto that of $H$, each edge keeping its symmetry type (a directed edge its order, a cyclic "
    r"edge its order up to rotation, an unordered edge its multiset of vertices). Labels count only up "
    r"to renaming: there are bijections of vertex labels and of edge labels, chosen independently, "
    r"carried along by $\varphi$; an unlabelled element corresponds only to an unlabelled one.",
    maturity="working",
    depends_on=("common.hypergraph",),
    sources=(SourceRef("hypergraph-paclet", "symbol:IsomorphicHypergraphQ", "implementation"),),
    tags=("hypergraph", "isomorphism"),
)
def isomorphic(g: Hypergraph, h: Hypergraph, *, labeled: bool = True, method: str = "automatic") -> bool:
    if _coarse(g, labeled) != _coarse(h, labeled):
        return False
    return canonical_hypergraph(g, labeled=labeled, method=method) == canonical_hypergraph(
        h, labeled=labeled, method=method
    )


def _sorted_dict(d: dict) -> dict:
    return dict(sorted(d.items(), key=lambda kv: order_key(kv[0])))


def find_isomorphism(
    g: Hypergraph, h: Hypergraph, *, labeled: bool = True, method: str = "automatic", labels: bool = False
) -> dict:
    """A vertex map realizing an isomorphism, or ``{}``.

    With ``labels=True``: ``{"Vertices": ..., "VertexLabels": ..., "EdgeLabels": ...}``, the
    vertex map and the two label correspondences that go with it (empty when there is none).
    """
    empty: dict = {"Vertices": {}, "VertexLabels": {}, "EdgeLabels": {}} if labels else {}
    if _coarse(g, labeled) != _coarse(h, labeled):
        return empty
    d1 = canonical_data(g, labeled=labeled, method=method)
    d2 = canonical_data(h, labeled=labeled, method=method)
    if d1.form != d2.form:
        return empty
    vertex_map = _sorted_dict(dict(zip(d1.ordering, d2.ordering, strict=True)))
    if not labels:
        return vertex_map

    def correspondence(t1: dict, t2: dict) -> dict:
        inverse = {tok: lab for lab, tok in t2.items()}
        return _sorted_dict({lab: inverse[tok] for lab, tok in t1.items()})

    return {
        "Vertices": vertex_map,
        "VertexLabels": correspondence(d1.vertex_tokens, d2.vertex_tokens),
        "EdgeLabels": correspondence(d1.edge_tokens, d2.edge_tokens),
    }


def apply_vertex_map(
    g: Hypergraph, vertex_map: dict, vertex_labels: dict | None = None, edge_labels: dict | None = None
):
    """``g`` with its vertices (and optionally labels) renamed: to check an isomorphism carries g onto h."""
    vl, el = vertex_labels or {}, edge_labels or {}
    vertices = [Vertex(vertex_map[v.id], vl.get(v.label, v.label)) for v in g.vertices]
    by_id = {v.id: v for v in vertices}
    edges = [Edge([by_id[vertex_map[i]] for i in e.ids], e.symmetry, el.get(e.label, e.label)) for e in g.edges]
    return Hypergraph(vertices, edges)
