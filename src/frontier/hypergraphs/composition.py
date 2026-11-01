"""Rewriting applied all at once: one pass, composition products, saturation.

One matcher (:mod:`frontier.hypergraphs.rewriting`) serves every mode; they differ in what they do
with its matches.

* **One pass** (:func:`one_pass`): every match of a rule at once, on the original hypergraph,
  keeping only what the matches create (the paclet prototype's ``OnePassRewriting``).
* **Composition** (:func:`hypergraph_compose`): a product of hypergraphs on a fixed vertex set,
  generalising graph composition. Each input edge of the rule names the factor it is taken from;
  the matches over the factors' union give the product, one output edge per match. Vertices and
  edges match non-injectively by default, which makes the product the plex product of the
  factors' adjacency hypermatrices (``hypermatrix.composition``). ``multiplicity`` chooses the
  semiring: ``"Count"`` (a multiset of edges: counts in ℕ), ``"Boolean"`` (a set: relational
  composition) or ``"Labels"`` (labelled edges kept, one per match: formal sums of labels).
* **Saturation** (:func:`hypergraph_saturate`): generating rules add what they create, never
  removing what they matched; equation rules then rewrite (replace) to a normal form; edges form a
  set. Rounds repeat until nothing changes or a bound is reached. Labels are syntax: equalities
  between them are equation rules (settled 2026-10-30). ``identify_by_label`` merges vertices with
  equal labels after each round (one vertex per object, the category reading); by default a
  vertex is a position, as in a diagram.

Settled choices: ``themes/hypergraph-rewrite-algebras/BRIEF.md`` (2026-10-30).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from frontier.hypergraphs.core import Edge, Hypergraph, HypergraphError, Vertex
from frontier.hypergraphs.labels import Term, _composite, is_label_variable, match_label, substitute
from frontier.hypergraphs.rewriting import FormalV, HypergraphRule, apply_rule
from frontier.registry import Evidence, EvidenceKind, SourceRef, claim, define

MULTIPLICITIES = ("Count", "Boolean", "Labels")


def _edge_key(e: Edge) -> tuple:
    return (e.ids, e.symmetry, e.label)


def _fresh(avoid: set, start: int = 1):
    k = start
    while True:
        if FormalV(k) not in avoid:
            yield FormalV(k)
        k += 1


def _created(events, avoid: set) -> tuple[list[Edge], list[Vertex]]:
    """The edges the events create, their new vertices renamed apart across events (``FormalV``)."""
    names = _fresh(avoid)
    edges: list[Edge] = []
    vertices: list[Vertex] = []
    for ev in events:
        labels = {v.id: v.label for v in ev.hypergraph.vertices}
        rename = {old: next(names) for old in ev.new_vertices}
        vertices += [Vertex(rename[old], labels.get(old)) for old in ev.new_vertices]
        edges += [Edge([rename.get(i, i) for i in e.ids], e.symmetry, e.label) for e in ev.new_edges]
    return edges, vertices


@define(
    "common.hypergraph_one_pass",
    "One-pass rewriting",
    r"One-pass rewriting of $H$ by a rule $L \to R$ applies every match of $L$ in $H$ at once, to $H$ "
    r"itself: the result has $H$'s vertices (and any the matches create, named apart) and, for each "
    r"match, the edges $R$ creates there. What the matches consume is not removed: the result is what "
    r"the rule makes of $H$, not a rewritten $H$.",
    maturity="sketch",
    depends_on=("common.hypergraph_rule",),
    sources=(SourceRef("hypergraph-paclet", "path:reference/HypergraphAlgebras.wl#L13-L13", "implementation"),),
    tags=("hypergraph", "rewriting"),
)
def one_pass(rule: HypergraphRule, hg: Hypergraph, **match_options: Any) -> Hypergraph:
    """Every event of ``rule`` on ``hg`` at once: the edges they create, on ``hg``'s vertices."""
    events = apply_rule(rule, hg, **match_options)
    edges, created = _created(events, set(hg.vertex_ids))
    return Hypergraph(list(hg.vertices) + created, edges)


# -- composition -----------------------------------------------------------------------------------


def _factor_of(label: Any, names: Sequence[str]) -> tuple[str, Any] | None:
    """Which factor an input edge's label names, and the pattern for the factor edge's own label.

    ``a`` (a name or a string) is any edge of factor a; ``a(x)`` is an edge of factor a whose own
    label matches ``x``."""
    if isinstance(label, str) and label in names:
        return label, None
    if is_label_variable(label) and label.name in names:
        return label.name, None
    parts = _composite(label)
    if parts is not None and parts[0] in names and len(parts[1]) == 1:
        return parts[0], parts[1][0]
    return None


def composition_factors(rule: HypergraphRule, names: Sequence[str]) -> list[tuple[Edge, str, Any]]:
    """For each input edge of a composition rule (EdgeList order): the edge, its factor, its own-label pattern."""
    out = []
    for e in rule.input.edges:
        found = _factor_of(e.label, names)
        if found is None:
            raise HypergraphError(
                "HypergraphCompose::factor",
                f"the input edge {list(e.ids)} is labelled {e.label!r}, which names none of the factors {list(names)}: "
                "label it by a factor name, a, or a(x) to match the factor edge's own label",
            )
        out.append((e, *found))
    return out


def _vertex_set(factors: Mapping[str, Hypergraph], vertices: Sequence[Any] | None) -> list[Vertex]:
    if vertices is not None:
        labels = {v.id: v.label for hg in factors.values() for v in hg.vertices}
        given = [v if isinstance(v, Vertex) else Vertex(v, labels.get(v)) for v in vertices]
        ids = {v.id for v in given}
        stray = [i for hg in factors.values() for i in hg.vertex_ids if i not in ids]
        if stray:
            raise HypergraphError("HypergraphCompose::vertices", f"factor vertices {stray!r} are not in the vertex set")
        return given
    seen: dict = {}
    for hg in factors.values():
        for v in hg.vertices:
            seen.setdefault(v.id, v)
    return list(seen.values())


@define(
    "common.hypergraph_composition",
    "Composition of hypergraphs by a rule",
    r"Let $L \to R$ be a rule each of whose input edges names a factor ($a$, or $a(x)$ to match the factor "
    r"edge's own label), $R$ creating no vertices, and $G_a, G_b, \ldots$ hypergraphs on a common vertex set "
    r"$V$. Their composition is the hypergraph on $V$ with one edge of $R$ per match of $L$ in the union of the "
    r"factors, each input edge landing on an edge of the factor it names. Matching is non-injective on "
    r"vertices and edges by default. Multiplicity: `Count` keeps every match (a multiset of unlabelled "
    r"edges), `Boolean` keeps each edge once, `Labels` keeps the labels $R$ writes (one edge per match). "
    r"Graph composition is the rule $a{:}(1,2),\ b{:}(2,3) \to (1,3)$.",
    maturity="sketch",
    depends_on=("common.hypergraph_rule",),
    sources=(SourceRef("hypergraph-paclet", "path:reference/HypergraphAlgebras.wl#L25-L40", "implementation"),),
    tags=("hypergraph", "rewriting", "composition"),
)
def hypergraph_compose(
    rule: HypergraphRule,
    factors: Mapping[str, Hypergraph],
    *,
    multiplicity: str = "Count",
    vertices: Sequence[Any] | None = None,
    distinct_vertices: bool = False,
    distinct_edges: bool = False,
    labeled: bool = True,
    label_patterns: bool = True,
) -> Hypergraph:
    """The composition of the factors (``{name: hypergraph}``) by ``rule``, on their common vertex set."""
    if not isinstance(rule, HypergraphRule):
        raise HypergraphError("HypergraphCompose::rule", f"{rule!r} is not a rewrite rule")
    if not isinstance(factors, Mapping) or not all(isinstance(h, Hypergraph) for h in factors.values()):
        raise HypergraphError("HypergraphCompose::factors", "the factors are given as {name: hypergraph, ...}")
    if multiplicity not in MULTIPLICITIES:
        raise HypergraphError("HypergraphCompose::multiplicity", f"multiplicity is one of {', '.join(MULTIPLICITIES)}")
    if rule.new_vertices:
        raise HypergraphError(
            "HypergraphCompose::vertices",
            f"the rule creates vertices {rule.new_vertices!r}: a composition stays on the factors' vertex set",
        )
    names = [str(n) for n in factors]
    parts = composition_factors(rule, names)
    V = _vertex_set(factors, vertices)

    host = Hypergraph(V, [e for hg in factors.values() for e in hg.edges])
    # the constructor sorts edges: find each host edge's factor again, equal edges in turn
    pool: dict[tuple, list[str]] = {}
    for name, hg in factors.items():
        for e in hg.edges:
            pool.setdefault(_edge_key(e), []).append(str(name))
    owner = [pool[_edge_key(e)].pop(0) for e in host.edges]
    positions = {n: {p for p, o in enumerate(owner) if o == n} for n in names}

    lhs = Hypergraph(list(rule.input.vertices), [Edge(e.ids, e.symmetry, own) for e, _, own in parts])
    waiting: dict[tuple, list[str]] = {}
    for e, name, own in parts:
        waiting.setdefault((e.ids, e.symmetry, own), []).append(name)
    restrict = [positions[waiting[_edge_key(e)].pop(0)] for e in lhs.edges]

    events = apply_rule(
        HypergraphRule(lhs, rule.output),
        host,
        labeled=labeled,
        label_patterns=label_patterns,
        distinct_vertices=distinct_vertices,
        distinct_edges=distinct_edges,
        restrict=restrict,
    )
    made = [e for ev in events for e in ev.new_edges]
    if multiplicity == "Count":
        edges = [Edge(e.ids, e.symmetry) for e in made]
    elif multiplicity == "Boolean":
        edges = list({(e.ids, e.symmetry): Edge(e.ids, e.symmetry) for e in made}.values())
    else:
        edges = made
    return Hypergraph(V, edges)


# -- saturation ------------------------------------------------------------------------------------


@dataclass
class Saturation:
    """The rounds of a saturation: ``states[0]`` is the start (as a set, equations applied)."""

    states: list[Hypergraph]
    added: list[list[Edge]] = field(default_factory=list)  # per round, the edges it brought in
    saturated: bool = False  # True when a round changed nothing; False when a bound stopped it
    stopped_by: str = ""  # "MaxRounds" or "MaxEdges" when a bound stopped it

    @property
    def final(self) -> Hypergraph:
        return self.states[-1]

    @property
    def rounds(self) -> int:
        return len(self.states) - 1

    PROPERTIES = ("Final", "States", "Added", "Saturated", "Rounds", "StoppedBy")

    def prop(self, name: str) -> Any:
        table = {
            "Final": lambda: self.final,
            "States": lambda: list(self.states),
            "Added": lambda: [list(a) for a in self.added],
            "Saturated": lambda: self.saturated,
            "Rounds": lambda: self.rounds,
            "StoppedBy": lambda: self.stopped_by or None,
            "Properties": lambda: list(self.PROPERTIES),
        }
        if name not in table:
            raise KeyError(f"unknown property {name!r}; a saturation has {', '.join(self.PROPERTIES)}")
        return table[name]()


def _as_set(hg: Hypergraph) -> Hypergraph:
    return Hypergraph(list(hg.vertices), list({_edge_key(e): e for e in hg.edges}.values()))


def _identify(hg: Hypergraph) -> Hypergraph:
    """Merge vertices with equal labels (unlabelled vertices stay apart); the first in vertex order stays."""
    first: dict = {}
    rename: dict = {}
    for v in hg.vertices:
        if v.label is None:
            continue
        try:
            rename[v.id] = first.setdefault(v.label, v.id)
        except TypeError:  # an unhashable label identifies nothing
            continue
    if all(rename.get(v, v) == v for v in hg.vertex_ids):
        return hg
    vertices = [v for v in hg.vertices if rename.get(v.id, v.id) == v.id]
    edges = [Edge([rename.get(i, i) for i in e.ids], e.symmetry, e.label) for e in hg.edges]
    return Hypergraph(vertices, edges)


def _normalise(hg: Hypergraph, equations: Sequence[HypergraphRule], options: dict, max_steps: int) -> Hypergraph:
    """Rewrite with the equation rules (first event first) until none applies; edges kept as a set."""
    hg = _as_set(hg)
    for _ in range(max_steps):
        for eq in equations:
            events = apply_rule(eq, hg, limit=1, **options)
            if events:
                hg = _as_set(events[0].hypergraph)
                break
        else:
            return hg
    raise HypergraphError(
        "HypergraphSaturate::equations",
        f"the equation rules did not reach a normal form in {max_steps} steps: they may undo one another",
    )


def _label_equations(equations: Sequence[HypergraphRule]) -> list[tuple[Any, Any]]:
    """The equations that are equalities of labels: one edge to one edge on the same vertices, no vertex conditions."""
    out = []
    for eq in equations:
        if len(eq.input.edges) != 1 or len(eq.output.edges) != 1:
            continue
        a, b = eq.input.edges[0], eq.output.edges[0]
        if a.ids != b.ids or a.symmetry != b.symmetry or any(v.label is not None for v in eq.input.vertices):
            continue
        out.append((a.label, b.label))
    return out


def _rewrite_label(label: Any, laws: list[tuple[Any, Any]], budget: list[int]) -> Any:
    """``label`` with the label equations applied at every subterm, innermost first, to a normal form."""
    parts = _composite(label)
    if parts is not None:
        label = Term(parts[0], [_rewrite_label(a, laws, budget) for a in parts[1]])
    for pattern, result in laws:
        found = match_label(pattern, label, {})
        if found is not None:
            budget[0] -= 1
            if budget[0] < 0:
                raise HypergraphError(
                    "HypergraphSaturate::equations",
                    "the label equations did not reach a normal form: they may undo one another",
                )
            return _rewrite_label(substitute(result, found), laws, budget)
    return label


def _congruence(hg: Hypergraph, laws: list[tuple[Any, Any]], max_steps: int) -> Hypergraph:
    if not laws:
        return hg
    budget = [max_steps]
    return Hypergraph(
        list(hg.vertices), [Edge(e.ids, e.symmetry, _rewrite_label(e.label, laws, budget)) for e in hg.edges]
    )


def _signature(hg: Hypergraph) -> tuple:
    return (frozenset((v.id, v.label) for v in hg.vertices), frozenset(_edge_key(e) for e in hg.edges))


@define(
    "common.hypergraph_saturation",
    "Saturation by generating rules and equations",
    r"Saturating $H$ by generating rules and equation rules repeats rounds: every match of every generating "
    r"rule adds the edges (and vertices) its output creates, keeping what it matched; then the equation rules "
    r"rewrite, as ordinary rules, until none applies; edges form a set (an edge is its vertices, symmetry and "
    r"label). Optionally, vertices with equal labels are merged after each round. It stops when a round "
    r"changes nothing (saturated) or after a bound. Labels are syntax: equalities between them, such as the "
    r"unit and associativity laws of a category, are given as equation rules.",
    maturity="sketch",
    depends_on=("common.hypergraph_rule",),
    tags=("hypergraph", "rewriting", "category"),
)
def hypergraph_saturate(
    hg: Hypergraph,
    rules: Sequence[HypergraphRule],
    *,
    equations: Sequence[HypergraphRule] = (),
    max_rounds: int = 20,
    identify_by_label: bool = False,
    distinct_vertices: bool = False,
    distinct_edges: bool = False,
    labeled: bool = True,
    label_patterns: bool = True,
    max_equation_steps: int = 10_000,
    max_edges: int = 1_000,
    congruence: bool = False,
) -> Saturation:
    """Saturate ``hg``: generating ``rules`` add, ``equations`` rewrite, until nothing changes.

    Generating rules match non-injectively on vertices and edges by default (an endomorphism composes
    with itself); equation rules match non-injectively on vertices and injectively on edges (a
    replacement consumes what it matched). Saturation of a structure with an infinite closure (a cycle
    in a category, a heap) does not end: ``max_rounds`` and ``max_edges`` stop it, and the result says
    which bound did.

    With ``congruence``, an equation that rewrites one edge's label to another on the same vertices
    (and puts no condition on vertex labels) also applies inside labels, at every subterm: equal
    morphisms then have equal composites, as in a category. Without it, equations act on whole
    edges only, as hypergraph rules."""
    for r in [*rules, *equations]:
        if not isinstance(r, HypergraphRule):
            raise HypergraphError("HypergraphSaturate::rule", f"{r!r} is not a rewrite rule")
    gen_opts = {
        "labeled": labeled,
        "label_patterns": label_patterns,
        "distinct_vertices": distinct_vertices,
        "distinct_edges": distinct_edges,
    }
    eq_opts = {**gen_opts, "distinct_edges": True}

    laws = _label_equations(equations) if congruence else []

    def tidy(g: Hypergraph) -> Hypergraph:
        g = _normalise(_congruence(g, laws, max_equation_steps), equations, eq_opts, max_equation_steps)
        return _as_set(_identify(g)) if identify_by_label else g

    state = tidy(hg)
    result = Saturation([state])
    for _ in range(max_rounds):
        if len(state.edges) > max_edges:
            result.stopped_by = "MaxEdges"
            return result
        present = {_edge_key(e) for e in state.edges}
        events = [ev for rule in rules for ev in apply_rule(rule, state, **gen_opts)]
        made, created = _created(events, set(state.vertex_ids))
        fresh = [e for e in made if _edge_key(e) not in present]
        nxt = tidy(Hypergraph(list(state.vertices) + created, list(state.edges) + fresh))
        if _signature(nxt) == _signature(state):
            result.saturated = True
            return result
        before = {_edge_key(e) for e in state.edges}
        result.added.append([e for e in nxt.edges if _edge_key(e) not in before])
        result.states.append(nxt)
        state = nxt
    result.stopped_by = "MaxEdges" if len(state.edges) > max_edges else "MaxRounds"
    return result


def parallel_edges(hg: Hypergraph) -> dict[tuple, list[Any]]:
    """Edges sharing their vertices and symmetry, grouped: ``{(ids, symmetry): [labels]}``, groups of two or more.

    In a saturated diagram, a group with distinct labels is a pair of paths not (yet) made equal."""
    groups: dict[tuple, list[Any]] = {}
    for e in hg.edges:
        groups.setdefault((e.ids, e.symmetry), []).append(e.label)
    return {k: v for k, v in groups.items() if len(v) > 1}


_IN_USE = (
    "themes/hypergraph-rewrite-algebras/questions/q02-unified-rewriting-engine/experiments/2026-10-30-engine-in-use.py"
)

claim(
    "common.saturation_free_category",
    r"Saturating a finite acyclic labelled digraph (objects on vertices, distinct morphism labels on edges) by "
    r"composition $x{:}X\to Y,\ y{:}Y\to Z \Rightarrow y\circ x{:}X \to Z$ and identities, with the two unit laws "
    r"and associativity as equation rules applied inside labels (congruence), ends with the free category: one "
    r"identity per object and, from $u$ to $w \ne u$, exactly $\sum_k (A^k)_{uw}$ morphisms. Applied to whole "
    r"labels only, the equations leave distinct bracketings of one morphism apart.",
    about=("common.hypergraph_saturation",),
    status="supported",
    evidence=[
        Evidence(EvidenceKind.NUMERICAL, "40 random DAGs on 5 objects with parallel morphisms; without congruence a "
                 "counterexample within 9 cases", source=_IN_USE),
    ],
)  # fmt: skip

__all__ = [
    "MULTIPLICITIES",
    "Saturation",
    "composition_factors",
    "hypergraph_compose",
    "hypergraph_saturate",
    "one_pass",
    "parallel_edges",
]
