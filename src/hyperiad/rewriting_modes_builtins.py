"""Hyperiad builtins for rewriting applied all at once: one pass, composition, saturation.

The same matcher as ``HypergraphRuleApply`` (``frontier.hypergraphs.rewriting``), applied in other
ways (``frontier.hypergraphs.composition``), and the bridge from composition to hypermatrix algebra
(``hypermatrix.composition``). A saturation displays as frames, one per round, what each round added
in amber, on one layout.
"""

from __future__ import annotations

from typing import Any

import sympy
from frontier.hypergraphs import (
    MULTIPLICITIES,
    Hypergraph,
    HypergraphError,
    HypergraphRule,
    Saturation,
    hypergraph_compose,
    hypergraph_saturate,
    one_pass,
    parallel_edges,
    symmetry_size,
)
from frontier.hypergraphs.pictures import hypergraph_positions
from frontier.hypergraphs.svg import hypergraph_svg
from hypermatrix import ArrayError, composition_adjacency_product, composition_plex_diagram

from hyperiad.display import register_display, steady_frames
from hyperiad.errors import HyperiadError
from hyperiad.kernel import builtin
from hyperiad.values import format_value

MAX_FRAMES = 40  # a saturation shows at most this many rounds drawn


def _rule(x: Any, name: str) -> HypergraphRule:
    if not isinstance(x, HypergraphRule):
        raise HyperiadError(f"{name} takes a rewrite rule, got {format_value(x, 60)}")
    return x


def _rules(x: Any, name: str) -> list[HypergraphRule]:
    items = x if isinstance(x, list) else [x]
    return [_rule(r, name) for r in items]


def _hypergraph(x: Any, name: str) -> Hypergraph:
    if not isinstance(x, Hypergraph):
        raise HyperiadError(f"{name} takes a hypergraph, got {format_value(x, 60)}")
    return x


def _factors(x: Any, name: str) -> dict[str, Hypergraph]:
    if not isinstance(x, dict) or not all(isinstance(h, Hypergraph) for h in x.values()):
        raise HyperiadError(f'{name} takes the factors as {{"a": hypergraph, ...}}, got {format_value(x, 60)}')
    return {str(k.name if isinstance(k, sympy.Symbol) else k): h for k, h in x.items()}


def _automatic(x: Any) -> bool:
    return x is None or (isinstance(x, sympy.Symbol) and x.name == "Automatic")


def _guard(f, *args, **kw):
    try:
        return f(*args, **kw)
    except (HypergraphError, ArrayError) as e:
        raise HyperiadError(str(e)) from None


def _match_options(Labeled, LabelPatterns, DistinctVertices, DistinctEdges) -> dict:  # noqa: N803
    return {
        "labeled": bool(Labeled),
        "label_patterns": bool(LabelPatterns),
        "distinct_vertices": bool(DistinctVertices),
        "distinct_edges": bool(DistinctEdges),
    }


def _pattern_cost(rule: HypergraphRule, host_edges: int, host_vertices: int) -> float:
    cost = 1.0
    for e in rule.input.edges:
        orderings = symmetry_size(e.symmetry, e.arity)
        cost *= max(1, host_edges) * orderings
    on_edges = {v.id for e in rule.input.edges for v in e.vertices}
    cost *= max(1, host_vertices) ** sum(1 for v in rule.input.vertices if v.id not in on_edges)
    return cost * (len(rule.input.edges) + len(rule.output.edges) + 1)


def _one_pass_cost(rule, hg, **_):
    try:
        return _pattern_cost(rule, len(hg.edges), len(hg.vertices))
    except AttributeError:
        return 1.0


def _compose_cost(rule, factors, **_):
    try:
        edges = sum(len(h.edges) for h in factors.values())
        vertices = len({v for h in factors.values() for v in h.vertex_ids})
        return _pattern_cost(rule, edges, vertices)
    except AttributeError:
        return 1.0


def _saturate_cost(hg, rules, MaxRounds=20, **_):  # noqa: N803
    try:
        items = rules if isinstance(rules, list) else [rules]
        per_round = sum(_pattern_cost(r, len(hg.edges) * 4, len(hg.vertices) * 2) for r in items)
        return per_round * max(1, int(MaxRounds))
    except (AttributeError, TypeError):
        return 1.0


# -- display ---------------------------------------------------------------------------------------


def _describe_saturation(s: Saturation) -> str:
    final = s.final
    how = (
        f"saturated after {s.rounds} rounds"
        if s.saturated
        else f"not saturated: {s.stopped_by or 'a bound'} stopped it after {s.rounds} rounds"
    )
    return f"{how} · {len(final.vertices)} vertices, {len(final.edges)} edges"


def _show_saturation(s: Saturation) -> list[dict]:
    states = s.states[: MAX_FRAMES + 1]
    try:
        coords = hypergraph_positions(s.states[len(states) - 1])
    except Exception:  # noqa: BLE001 - a layout failure only costs the shared layout
        coords = None
    svgs, labels = [], []
    for k, g in enumerate(states):
        new = set() if k == 0 else {(e.ids, e.symmetry, e.label) for e in s.added[k - 1]}
        highlight = [p for p, e in enumerate(g.edges) if (e.ids, e.symmetry, e.label) in new]
        at = None if coords is None else {v: coords[v] for v in g.vertex_ids if v in coords}
        svgs.append(hypergraph_svg(g, highlight=highlight, coordinates=at))
        labels.append("start" if k == 0 else f"round {k}: +{len(s.added[k - 1])} edges")
    return [
        {
            "type": "frames",
            "frames": steady_frames(svgs),
            "labels": labels,
            "caption": _describe_saturation(s),
            "autoplay": False,
            "interval": 1200,
        }
    ]


register_display(Saturation, _show_saturation)


# -- builtins --------------------------------------------------------------------------------------


@builtin(
    "HypergraphRuleOnePass",
    """HypergraphRuleOnePass(rule, hg) applies every match of the rule to hg at once: the result has hg's vertices
(and any the matches create, named apart) and, for each match, the edges the rule's output creates there. Options
as HypergraphRuleApply (Labeled, LabelPatterns, DistinctVertices, DistinctEdges).""",
    cost=_one_pass_cost,
)
def _one_pass(rule, hg, Labeled=True, LabelPatterns=True, DistinctVertices=True, DistinctEdges=True):  # noqa: N803
    r, h = _rule(rule, "HypergraphRuleOnePass"), _hypergraph(hg, "HypergraphRuleOnePass")
    return _guard(one_pass, r, h, **_match_options(Labeled, LabelPatterns, DistinctVertices, DistinctEdges))


@builtin(
    "HypergraphCompose",
    """HypergraphCompose(rule, {"a": g, "b": h, ...}) is the composition of hypergraphs by a rule, on their common
vertex set: each input edge of the rule names the factor it comes from ("a", or a(x) to match that edge's own
label as x), and every match over the factors gives one copy of the rule's output edges. Graph composition is
HypergraphCompose(HypergraphRule([Edge([1, 2], "Directed", "a"), Edge([2, 3], "Directed", "b")], [[1, 3]]),
{"a": g, "b": h}). Multiplicity = "Count" (a multiset: one edge per match), "Boolean" (each edge once) or
"Labels" (the labels the output writes, one edge per match: formal sums of labels). Matching is non-injective
on vertices and edges unless DistinctVertices / DistinctEdges = true; Vertices = [...] fixes the vertex set.
The adjacency hypermatrix of the result is the plex product CompositionAdjacencyProduct gives.""",
    cost=_compose_cost,
)
def _compose(
    rule,
    factors,
    Multiplicity="Count",  # noqa: N803
    Vertices=None,  # noqa: N803
    DistinctVertices=False,  # noqa: N803
    DistinctEdges=False,  # noqa: N803
    Labeled=True,  # noqa: N803
    LabelPatterns=True,  # noqa: N803
):
    r = _rule(rule, "HypergraphCompose")
    fs = _factors(factors, "HypergraphCompose")
    if Multiplicity not in MULTIPLICITIES:
        raise HyperiadError(f"Multiplicity is one of {', '.join(repr(m) for m in MULTIPLICITIES)}")
    vertices = None if _automatic(Vertices) else list(Vertices)
    return _guard(
        hypergraph_compose,
        r,
        fs,
        multiplicity=str(Multiplicity),
        vertices=vertices,
        **_match_options(Labeled, LabelPatterns, DistinctVertices, DistinctEdges),
    )


@builtin(
    "HypergraphSaturate",
    """HypergraphSaturate(hg, rules) saturates hg: in each round every match of every generating rule adds what the
rule's output creates (keeping what it matched), then the equation rules (Equations = [...]) rewrite, as ordinary
rules, until none applies; edges form a set. It stops when a round changes nothing, or after MaxRounds = 20.
Labels are syntax: equalities between them (units, associativity, a commuting square) are equation rules.
IdentifyByLabel = true merges vertices with equal labels after each round (one vertex per object). Generating
rules match non-injectively (DistinctVertices, DistinctEdges = false); equations consume distinct edges.
Congruence = true also applies each equation that rewrites one edge's label on the same vertices (with no
condition on vertex labels) inside labels, at every subterm: equal morphisms then have equal composites.
A structure whose closure is infinite (a cycle, a heap) does not saturate: MaxRounds and MaxEdges = 1000 stop it.
Shown as one frame per round, what it added in amber; s["Final"], ["States"], ["Added"], ["Saturated"],
["Rounds"], ["StoppedBy"].""",
    cost=_saturate_cost,
)
def _saturate(
    hg,
    rules,
    Equations=None,  # noqa: N803
    MaxRounds=20,  # noqa: N803
    MaxEdges=1000,  # noqa: N803
    Congruence=False,  # noqa: N803
    IdentifyByLabel=False,  # noqa: N803
    DistinctVertices=False,  # noqa: N803
    DistinctEdges=False,  # noqa: N803
    Labeled=True,  # noqa: N803
    LabelPatterns=True,  # noqa: N803
):
    h = _hypergraph(hg, "HypergraphSaturate")
    gens = _rules(rules, "HypergraphSaturate")
    eqs = [] if Equations is None else _rules(Equations, "HypergraphSaturate")
    if not isinstance(MaxRounds, int) or isinstance(MaxRounds, bool) or MaxRounds < 0:
        raise HyperiadError("MaxRounds is a nonnegative integer")
    return _guard(
        hypergraph_saturate,
        h,
        gens,
        equations=eqs,
        max_rounds=MaxRounds,
        max_edges=int(MaxEdges),
        congruence=bool(Congruence),
        identify_by_label=bool(IdentifyByLabel),
        **_match_options(Labeled, LabelPatterns, DistinctVertices, DistinctEdges),
    )


@builtin(
    "ParallelEdges",
    """ParallelEdges(hg) lists the groups of edges on the same vertices with the same symmetry, as
{"Vertices": [...], "Labels": [...]}: in a saturated diagram, a group with different labels is a pair of paths
that no equation has made equal (the diagram does not commute there).""",
)
def _parallel(hg):
    found = parallel_edges(_hypergraph(hg, "ParallelEdges"))
    return [{"Vertices": list(ids), "Labels": labels} for (ids, _), labels in found.items()]


@builtin(
    "CompositionPlexDiagram",
    """CompositionPlexDiagram(rule) is the plex diagram of a composition rule with one output edge: its input edges
as plexes named by their factors, the input vertices off the output edge contracted, the output edge's vertices
as the free indices in order.""",
)
def _composition_plex(rule):
    diagram, _ = _guard(composition_plex_diagram, _rule(rule, "CompositionPlexDiagram"))
    return diagram


@builtin(
    "CompositionAdjacencyProduct",
    """CompositionAdjacencyProduct(rule, {"a": g, ...}) is the hypermatrix side of HypergraphCompose: the plex product,
by CompositionPlexDiagram(rule), of the factors' adjacency arrays, summed over the orbit of each index under the
output edge's symmetry; Multiplicity = "Boolean" gives its support, DistinctVertices = true the sum over distinct
indices. It equals the adjacency array of HypergraphCompose(rule, factors) with the same options.""",
    cost=_compose_cost,
)
def _composition_array(rule, factors, Multiplicity="Count", DistinctVertices=False):  # noqa: N803
    r = _rule(rule, "CompositionAdjacencyProduct")
    fs = _factors(factors, "CompositionAdjacencyProduct")
    return _guard(
        composition_adjacency_product, r, fs, multiplicity=str(Multiplicity), distinct_vertices=bool(DistinctVertices)
    )
