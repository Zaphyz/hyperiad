"""Plex diagrams: hypermatrix products specified by labelled hypergraphs (heaps-of-fish §4).

A **plex** is an array up to the order of its indices. A **plex diagram** is a hypergraph whose
edges are plexes and whose vertices are indices: two plexes sharing a vertex share that index,
and a marked vertex is contracted (summed over). The unmarked vertices are the free indices of
the product. The conventions below were settled with the user on 2026-10-27 (see
``themes/hypermatrix-higher-arity/BRIEF.md``):

* a plex diagram is an ordinary labelled :class:`Hypergraph`: edge labels name the plexes and the
  vertex label :data:`PLEX_MARK` marks a contracted index, so isomorphism, rewriting and drawing
  apply unchanged;
* an edge's symmetry type is the array's: a directed edge's vertex order is the array's index
  order, an unordered edge accepts only symmetric arrays and a cyclic edge only cyclic ones;
* the free indices of the result run over the unmarked vertices in vertex-list order, unless an
  explicit output sequence is given;
* a plex diagram P is also a rewrite rule (:func:`plex_rule`): its input is P's edges with label
  variables, its output one directed edge on P's free vertices labelled ``P(x, y, ...)``.
  Rewriting a larger diagram with it and evaluating the labels gives the same array as
  evaluating the diagram, provided the event really contracts P's marked vertices
  (:func:`plex_events` keeps exactly those events).
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np
import sympy
from frontier.hypergraphs import CONTRACTED_MARK, Edge, Event, Hypergraph, HypergraphRule, Term, Vertex, apply_rule
from frontier.hypergraphs.core import is_io, order_key, symmetry_orderings
from frontier.registry import Evidence, EvidenceKind, SourceRef, claim, define

from hypermatrix.algebra import array_multiply
from hypermatrix.arrays import ArrayError, ArrayObject, as_array, has_symmetry

PLEX_MARK = CONTRACTED_MARK  # the vertex label of a contracted index; drawn as a solid dark vertex
_ARRAY_SYMMETRY = {"Directed": "Rigid", "Cyclic": "Cyclic", "Unordered": "Symmetric"}


@define(
    "hypermatrix.plex_diagram",
    "Plex diagram",
    r"A plex diagram is a labelled hypergraph whose edges are plexes (arrays up to the order of their "
    r"indices), named by their edge labels, and whose vertices are indices; a vertex labelled "
    r"$\bullet$ is contracted, the others are free. An edge's symmetry type fixes how an array sits on "
    r"it: a directed edge's vertex order is the array's index order, an unordered edge takes a "
    r"symmetric array, a cyclic edge a cyclic one.",
    maturity="stable",
    depends_on=("common.hypergraph", "hypermatrix.array"),
    sources=(
        SourceRef("heaps-of-fish", "section:The Plex Formalism", "definition"),
        SourceRef("hypergraph-paclet", "path:reference/PlexAlgebra.wl#L103-L116", "implementation"),
    ),
    tags=("plex", "hypermatrix", "hypergraph"),
)
def plex_diagram(
    edges: Sequence[Any],
    contracted: Sequence[Hashable] = (),
    *,
    names: Sequence[Any] | None = None,
    symmetry: str = "Directed",
) -> Hypergraph:
    """The plex diagram with these edges (vertex lists or :class:`Edge` objects), contracting ``contracted``.

    Plain vertex lists become edges of the given ``symmetry``; ``names`` labels them in the order
    given (default ``"p1"``, ``"p2"``, ...). An :class:`Edge` keeps its own symmetry and label.
    """
    made = []
    for k, e in enumerate(edges):
        name = names[k] if names is not None else None
        if isinstance(e, Edge):
            made.append(Edge(e.ids, e.symmetry, name if name is not None else e.label))
        else:
            made.append(Edge(list(e), symmetry, name if name is not None else f"p{k + 1}"))
    ids = list(dict.fromkeys(i for e in made for i in e.ids))
    missing = [v for v in contracted if v not in ids]
    if missing:
        raise ArrayError("PlexDiagram::vertex", f"contracted vertices {missing!r} lie on no plex")
    marked = set(contracted)
    return Hypergraph([Vertex(v, PLEX_MARK) if v in marked else Vertex(v) for v in ids], made)


def plex_contracted(hg: Hypergraph) -> list[Hashable]:
    """The marked (contracted) vertices of a plex diagram, in vertex-list order."""
    return [v.id for v in hg.vertices if v.label == PLEX_MARK]


def plex_free(hg: Hypergraph) -> list[Hashable]:
    """The free vertices of a plex diagram (the indices of the product), in vertex-list order."""
    return [v.id for v in hg.vertices if v.label != PLEX_MARK]


def plex_names(hg: Hypergraph) -> list[Any]:
    """The distinct plex names (edge labels) of a diagram, in a fixed order."""
    return sorted({e.label for e in hg.edges}, key=order_key)


def _value(label: Any, env: Mapping, diagrams: Mapping) -> ArrayObject:
    if label in env:
        return as_array(env[label])
    if isinstance(label, Term) and label.head in diagrams:
        diagram = diagrams[label.head]
        values = [_value(a, env, diagrams) for a in label.args]
        return plex_product(diagram, dict(zip(plex_names(diagram), values, strict=True)), diagrams=diagrams)
    raise ArrayError("PlexProduct::name", f"no array is given for the plex {label!r}")


def plex_spec(hg: Hypergraph, output: Sequence[Hashable] | None = None) -> tuple[list[list[Hashable]], list[Hashable]]:
    """The index specification of the product: each edge's vertices, then the free vertices (or ``output``)."""
    free = plex_free(hg)
    if output is not None:
        out = list(output)
        if sorted(out, key=order_key) != sorted(free, key=order_key):
            raise ArrayError("PlexProduct::output", f"the output {out!r} must list the free vertices {free!r}")
    else:
        out = free
    return [list(e.ids) for e in hg.edges], out


@define(
    "hypermatrix.plex_product",
    "Plex product",
    r"The plex product of a plex diagram on arrays $A_e$ (one per plex name) is the array on the free "
    r"vertices $C_{\mathrm{free}} = \sum_{\mathrm{marked}} \prod_e (A_e)_{\mathrm{vertices}(e)}$, the "
    r"indices of each array read in its edge's vertex order; an unordered or cyclic edge requires its "
    r"array to have that symmetry, which makes the order immaterial. A label that is a term "
    r"$P(x, y, \ldots)$ of a named diagram $P$ is first evaluated as that plex product.",
    maturity="stable",
    depends_on=("hypermatrix.plex_diagram", "hypermatrix.array_multiply"),
    sources=(
        SourceRef("heaps-of-fish", "section:The Plex Formalism", "definition"),
        SourceRef("hypergraph-paclet", "path:reference/PlexAlgebra.wl#L197-L203", "implementation"),
    ),
    tags=("plex", "contraction"),
)
def plex_product(
    hg: Hypergraph,
    arrays: Mapping[Any, Any] | Sequence[Any],
    *,
    output: Sequence[Hashable] | None = None,
    diagrams: Mapping[str, Hypergraph] | None = None,
) -> ArrayObject:
    """Evaluate a plex diagram. ``arrays`` maps plex names to arrays, or lists them in :func:`plex_names` order."""
    if not isinstance(hg, Hypergraph):
        raise ArrayError("PlexProduct::diagram", f"{hg!r} is not a plex diagram")
    names = plex_names(hg)
    if isinstance(arrays, Mapping):
        env = dict(arrays)
    else:
        arrays = list(arrays)
        if len(arrays) != len(names):
            raise ArrayError(
                "PlexProduct::count", f"the diagram has {len(names)} plexes {names!r}, {len(arrays)} arrays given"
            )
        env = dict(zip(names, arrays, strict=True))
    diagrams = diagrams or {}
    values = []
    for e in hg.edges:
        a = _value(e.label, env, diagrams)
        if is_io(e.symmetry):  # an input–output plex: its array is invariant within each block of indices
            data = np.asarray(a.data, dtype=object)
            if data.ndim != e.arity or any(
                not np.array_equal(data, np.transpose(data, perm)) for perm in symmetry_orderings(e.symmetry, e.arity)
            ):
                raise ArrayError(
                    "PlexProduct::symmetry",
                    f"the plex {e.label!r} sits on a {e.symmetry} edge: its array must be invariant within each block",
                )
            values.append(a)
            continue
        sym = _ARRAY_SYMMETRY[e.symmetry]
        if sym != "Rigid" and not has_symmetry(a.data, sym):
            raise ArrayError(
                "PlexProduct::symmetry",
                f"the plex {e.label!r} sits on a {e.symmetry.lower()} edge, so its array must be {sym.lower()}",
            )
        values.append(a)
    return array_multiply(values, plex_spec(hg, output))


# -- plex diagrams as rewrite rules ---------------------------------------------------------


def _variable(name: Any) -> sympy.Symbol:
    """The label variable standing for the plex ``name`` in a rule: a symbol of the same name."""
    return sympy.Symbol(name if isinstance(name, str) and name.isidentifier() else f"plex_{order_key(name)}")


@define(
    "hypermatrix.plex_rule",
    "Plex diagram as a rewrite rule",
    r"A plex diagram $P$ named $p$ gives the rewrite rule whose input is $P$ with each plex name replaced "
    r"by a label variable (its marked vertices still marked, so they match only contracted indices) and "
    r"whose output is one directed edge on $P$'s free vertices, in output order, labelled "
    r"$p(x_1, \ldots, x_k)$ in the variables of its plex names.",
    maturity="stable",
    depends_on=("hypermatrix.plex_diagram", "common.hypergraph_rule"),
    tags=("plex", "rewriting"),
)
def plex_rule(hg: Hypergraph, name: str, *, output: Sequence[Hashable] | None = None) -> HypergraphRule:
    names = plex_names(hg)
    var = {n: _variable(n) for n in names}
    lhs_vertices = [Vertex(v.id, PLEX_MARK) if v.label == PLEX_MARK else Vertex(v.id) for v in hg.vertices]
    lhs = Hypergraph(lhs_vertices, [Edge(e.ids, e.symmetry, var[e.label]) for e in hg.edges])
    _, out = plex_spec(hg, output)
    rhs = Hypergraph([Vertex(v) for v in out], [Edge(out, "Directed", Term(name, [var[n] for n in names]))])
    return HypergraphRule(lhs, rhs)


def plex_events(rule: HypergraphRule, hg: Hypergraph) -> list[Event]:
    """The events of a plex rule on a plex diagram that really contract: every marked vertex of the
    rule's input is removed, i.e. no plex outside the match uses it."""
    marked = [v.id for v in rule.input.vertices if v.label == PLEX_MARK]
    out = []
    for ev in apply_rule(rule, hg):
        removed = {ev.rule_vertex_map[m] for m in marked}
        if removed <= set(ev.deleted_vertices):
            out.append(ev)
    return out


claim(
    "hypermatrix.plex_rewriting_evaluates",
    r"Rewriting a plex diagram $D$ with the rule of a plex diagram $P$ named $p$ (at an event that "
    r"contracts $P$'s marked vertices) and evaluating, with $p(\ldots)$ read as the plex product of $P$, "
    r"gives the same array as evaluating $D$. Rewriting is symbolic plex algebra; two events whose "
    r"results agree in value but not in label are an associativity-type identity.",
    about=("hypermatrix.plex_rule", "hypermatrix.plex_product"),
    status="supported",
    evidence=[
        Evidence(
            EvidenceKind.NUMERICAL,
            "every valid event on random plex diagrams with random integer arrays (directed and symmetric plexes)",
            source="hypermatrix-algebra/tests/test_plex.py",
        ),
        Evidence(
            EvidenceKind.ARGUMENT,
            "the sum over all marked indices of a product splits as a sum over the indices internal to the "
            "match of the matched factors, times the rest, because no other factor carries those indices "
            "(commutative semiring: associativity, commutativity, distributivity)",
        ),
    ],
)
