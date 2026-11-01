"""The builtin functions of Hyperiad.

Names follow the reference paclet where the function comes from it (``ArrayMultiply``,
``IdentityArray``, ...), so the correspondence with the reference stays visible.
"""

from __future__ import annotations

import fnmatch
import itertools
import math
from typing import Any

import sympy
from frontier.hypergraphs import (
    Edge,
    Event,
    FormalV,
    Hypergraph,
    HypergraphRule,
    LabelToken,
    Verbatim,
    Vertex,
    apply_rule,
    canonical_hypergraph,
    find_isomorphism,
    isomorphic,
    rule_matches,
    simple_hypergraph_q,
    symmetry_size,
)
from frontier.hypergraphs.causality import CausalEdge, RewritingSystem, causal_graph
from frontier.hypergraphs.pictures import hypergraph_embedding
from frontier.hypergraphs.svg import event_svg, hypergraph_svg, rule_svg
from hypermatrix import (
    ArrayObject,
    FiniteField,
    Hypermatrix,
    adjacency_hypermatrix,
    adjacency_size,
    array_add,
    array_multiply,
    array_multiply_cost,
    array_nesting,
    array_svg,
    array_times,
    as_array,
    enumerate_plex_products,
    generate_symbolic_array,
    identity_array,
    one_array,
    partial_identity_array,
    plex_contracted,
    plex_diagram,
    plex_events,
    plex_free,
    plex_names,
    plex_product,
    plex_rule,
    zero_array,
)

from hyperiad import arith
from hyperiad.errors import HyperiadError
from hyperiad.expr import Expr, Symbol
from hyperiad.kernel import BUILTINS, Scope, Session, builtin, special
from hyperiad.values import Frames, Graphic, RuleValue, causal_edge_value, format_value, match_value


def _spec(spec: Any) -> Any:
    """``[[1, 2], [2, 3]] -> [1, 3]`` arrives as a rule value; the toolkit takes a pair."""
    return (spec.lhs, spec.rhs) if isinstance(spec, RuleValue) else spec


def _arrays(arrays: Any) -> list[Any]:
    if not isinstance(arrays, list):
        raise HyperiadError('ArrayMultiply takes a list of arrays: ArrayMultiply([a, b], "ij,jk->ik")')
    return arrays


# -- arrays --------------------------------------------------------------


@builtin(
    "ArrayObject",
    """ArrayObject(data) is the array with the given entries; ArrayObject(data, sym) declares the index
symmetry sym: "Rigid" (default), "Cyclic" or "Symmetric", checked against the entries.
A scalar is the array of order zero.""",
)
def _array_object(data, symmetry="Rigid"):
    return ArrayObject(data, symmetry)


@builtin(
    "ZeroArray",
    "ZeroArray(n, d) is the n-array with indices of length d and entries 0. "
    "ZeroArray(dims) gives the dimensions; a last argument names the domain.",
)
def _zero_array(*args):
    return zero_array(*args)


@builtin(
    "OneArray", "OneArray(n, d) is the n-array with indices of length d and entries 1. ZeroArray's other forms apply."
)
def _one_array(*args):
    return one_array(*args)


@builtin(
    "IdentityArray",
    "IdentityArray(n, d) is 1 where all n indices (of length d) agree, 0 elsewhere. IdentityArray(n, d, domain).",
)
def _identity_array(n, d, domain="Integer"):
    return identity_array(n, d, domain)


@builtin(
    "PartialIdentityArray", "PartialIdentityArray(n, d, positions) is 1 where the indices at the given positions agree."
)
def _partial_identity_array(n, d, positions, domain="Integer"):
    return partial_identity_array(n, d, positions, domain)


@builtin(
    "ArrayMultiply",
    """ArrayMultiply([a1, a2, ...], spec) contracts the arrays by an index specification, Einstein style:
"ij,jk->ik" (matrix product), "ii->" (trace), "ijp,qrp,qrk->ijk" (the fish product).
The spec may also be index lists: [[1, 2], [2, 3]] -> [1, 3]. Without "->", the indices
occurring once survive in sorted order.""",
    cost=lambda arrays, spec: array_multiply_cost(_arrays(arrays), _spec(spec)),
)
def _array_multiply(arrays, spec):
    return array_multiply(_arrays(arrays), _spec(spec))


@builtin("ArrayAdd", "ArrayAdd(a, b, ...) adds arrays entry by entry; a scalar adds to every entry. Same as a + b.")
def _array_add(*arrays):
    return array_add(*arrays)


@builtin(
    "ArrayTimes",
    "ArrayTimes(a, b, ...) multiplies arrays entry by entry; a scalar multiplies every entry. Same as a * b.",
)
def _array_times(*arrays):
    return array_times(*arrays)


@builtin("ArrayOrder", "ArrayOrder(a) is the number of indices of a.")
def _array_order(a):
    return as_array(a).order


@builtin("ArrayDimensions", "ArrayDimensions(a) is the list of index lengths of a.")
def _array_dimensions(a):
    return as_array(a).dims


@builtin("ArraySymmetry", 'ArraySymmetry(a) is the declared symmetry of a: "Rigid", "Cyclic" or "Symmetric".')
def _array_symmetry(a):
    return as_array(a).symmetry


@builtin("ArrayDomain", "ArrayDomain(a) is the narrowest value domain containing every entry, read off the entries.")
def _array_domain(a):
    return as_array(a).domain


@builtin("ArrayEntries", "ArrayEntries(a) is the entries of a as nested lists. Normal(a) is the same.")
def _array_entries(a):
    return as_array(a).entries


@builtin("Normal", "Normal(a) is the entries of the array a as nested lists; Normal(hm) those of each array of hm.")
def _normal(a):
    if isinstance(a, Hypermatrix):
        return a.normal()
    return as_array(a).entries


@builtin(
    "Hypermatrix",
    """Hypermatrix([a1, a2, ...]) (or Hypermatrix(a1, a2, ...)) holds the arrays in canonical order: by order, then
dimensions, then symmetry (Symmetric, Cyclic, Rigid). Raw entries become arrays. Properties: hm["Arrays"],
["Length"], ["Orders"], ["Dimensions"], ["Symmetries"], ["Domains"], ["Entries"]; hm[k] is the k-th array.""",
)
def _hypermatrix(*arrays):
    return Hypermatrix(*arrays)


@builtin("HypermatrixQ", "HypermatrixQ(x) is true if x is a hypermatrix.")
def _hypermatrix_q(x):
    return isinstance(x, Hypermatrix)


@builtin("HypermatrixArrays", "HypermatrixArrays(hm) is the list of arrays of hm, in canonical order.")
def _hypermatrix_arrays(hm):
    if not isinstance(hm, Hypermatrix):
        raise HyperiadError(f"HypermatrixArrays takes a hypermatrix, got {format_value(hm, 60)}")
    return list(hm.arrays)


def _adjacency_cost(hg, **_opts):
    return adjacency_size(_hypergraph(hg, "AdjacencyHypermatrix")) * 4


@builtin(
    "AdjacencyHypermatrix",
    """AdjacencyHypermatrix(hg) is the adjacency hypermatrix of hg: one array per arity and symmetry type of its
edges (Directed -> Rigid, Cyclic -> Cyclic, Unordered -> Symmetric), indexed by the vertices in VertexList
order, each entry counting the edges on that index tuple up to the symmetry. Labeled = true puts the edge
labels in instead (null for an unlabelled edge); that needs a simple hypergraph.""",
    cost=_adjacency_cost,
)
def _adjacency_hypermatrix(hg, Labeled=False):  # noqa: N803 - option names as written
    return adjacency_hypermatrix(_hypergraph(hg, "AdjacencyHypermatrix"), labeled=bool(Labeled))


@builtin("RegularArrayQ", "RegularArrayQ(a) is true if every index of a has the same length.")
def _regular(a):
    return as_array(a).regular


@builtin("ArrayObjectQ", "ArrayObjectQ(x) is true if x is an array object.")
def _array_q(x):
    return isinstance(x, ArrayObject)


@builtin(
    "GenerateSymbolicArray",
    "GenerateSymbolicArray(a, dims) has entries a[i, j, ...]; "
    "GenerateSymbolicArray(a, dims, sym) respects the symmetry sym.",
)
def _symbolic_array(name, dims, symmetry="Rigid"):
    return generate_symbolic_array(name, dims, symmetry)


@builtin("FiniteField", "FiniteField(p) is the field with p elements (p prime); FiniteField(p)(k) is the element k.")
def _finite_field(p):
    return FiniteField(p)


@builtin(
    "ArrayNesting", "ArrayNesting(n) is how the indices of an n-array are split into blocks of at most two for drawing."
)
def _array_nesting(n, b=2):
    return array_nesting(as_array(n).order if isinstance(n, (ArrayObject, list)) else n, b)


@builtin(
    "ArrayGraphics",
    "ArrayGraphics(a) draws the array a in the plane. Options: Nesting = [k1, k2, ...], Indices = false.",
)
def _array_graphics(a, Nesting=None, Indices=True):  # noqa: N803 - option names as written in Hyperiad
    arr = as_array(a)
    return Graphic(array_svg(arr, nesting=Nesting, indices=bool(Indices)), f"array of dimensions {arr.dims}")


# -- hypergraphs ----------------------------------------------------------


def _hypergraph(x: Any, name: str) -> Hypergraph:
    if not isinstance(x, Hypergraph):
        raise HyperiadError(f"{name} takes a hypergraph, got {format_value(x, 60)}")
    return x


def _edge(x: Any, name: str) -> Edge:
    if not isinstance(x, Edge):
        raise HyperiadError(f"{name} takes an edge, got {format_value(x, 60)}")
    return x


@builtin(
    "Vertex",
    "Vertex(id) is a vertex with identifier id; Vertex(id, label) gives it a label. "
    "Edges and hypergraphs also accept bare identifiers.",
)
def _vertex(id, label=None):  # noqa: A002 - the paclet's name
    return Vertex(id, label)


@builtin(
    "Edge",
    """Edge([v1, v2, ...]) is a directed hyperedge; Edge(vs, sym) has symmetry "Directed", "Unordered" or
"Cyclic"; Edge(vs, sym, label) labels it. Vertices are stored in the canonical order of the symmetry,
so Edge([3, 1, 2], "Cyclic") == Edge([1, 2, 3], "Cyclic"). An input–output edge is given by its two
nonempty blocks and their types: Edge([[a, b], [c, d, e]], ["Unordered", "Cyclic"], label), or
"InputOutput" for both directed; it is drawn as two lobes joined by a waist with an arrow from inputs to
outputs. Edge([[a], [b]], "InputOutput") is the directed edge Edge([a, b]).""",
)
def _edge_builtin(vertices, symmetry="Directed", label=None):
    return Edge(vertices, symmetry, label)


@builtin(
    "Hypergraph",
    """Hypergraph(edges) builds a hypergraph from edges: Edge objects or lists of vertices (directed).
Hypergraph(vertices, edges) adds vertices, possibly isolated or labelled; Hypergraph(vertices) is
edgeless. A hypergraph is a multiset of edges: order does not matter, multiplicity does.""",
)
def _hypergraph_builtin(*args):
    return Hypergraph(*args)


@builtin("HypergraphQ", "HypergraphQ(x) is true if x is a hypergraph.")
def _hypergraph_q(x):
    return isinstance(x, Hypergraph)


@builtin("VertexList", "VertexList(hg) is the list of vertices of hg, in canonical order.")
def _vertex_list(hg):
    return _hypergraph(hg, "VertexList")["VertexList"]


@builtin("EdgeList", "EdgeList(hg) is the list of edges of hg, in canonical order, repeats included.")
def _edge_list(hg):
    return _hypergraph(hg, "EdgeList")["EdgeList"]


@builtin("VertexCount", "VertexCount(hg) is the number of vertices of hg.")
def _vertex_count(hg):
    return _hypergraph(hg, "VertexCount")["VertexCount"]


@builtin("EdgeCount", "EdgeCount(hg) is the number of edges of hg, counted with multiplicity.")
def _edge_count(hg):
    return _hypergraph(hg, "EdgeCount")["EdgeCount"]


@builtin("VertexID", "VertexID(v) is the identifier of the vertex v.")
def _vertex_id(v):
    return v.id if isinstance(v, Vertex) else v


@builtin("VertexLabel", "VertexLabel(v) is the label of the vertex v, or null.")
def _vertex_label(v):
    return v.label if isinstance(v, Vertex) else None


@builtin("EdgeVertices", "EdgeVertices(e) is the list of vertices of the edge e.")
def _edge_vertices(e):
    return list(_edge(e, "EdgeVertices").vertices)


@builtin(
    "EdgeSymmetry",
    'EdgeSymmetry(e) is "Directed", "Unordered", "Cyclic" or, for an input–output edge, '
    '"InputOutput(S:a,T:b)": a inputs of type S and b outputs of type T.',
)
def _edge_symmetry(e):
    return _edge(e, "EdgeSymmetry").symmetry


@builtin(
    "EdgeBlocks",
    "EdgeBlocks(e) lists the vertices of e by block: [inputs, outputs] for an input–output edge, one block otherwise.",
)
def _edge_blocks(e):
    return [list(b) for b in _edge(e, "EdgeBlocks").block_ids]


@builtin("EdgeInputs", "EdgeInputs(e) lists the input vertices of an input–output (or directed binary) edge.")
def _edge_inputs(e):
    return list(_io_blocks(e, "EdgeInputs")[0])


@builtin("EdgeOutputs", "EdgeOutputs(e) lists the output vertices of an input–output (or directed binary) edge.")
def _edge_outputs(e):
    return list(_io_blocks(e, "EdgeOutputs")[1])


def _io_blocks(e: Any, name: str) -> tuple:
    e = _edge(e, name)
    if e.symmetry == "Directed" and e.arity == 2:  # one input and one output
        return (e.ids[:1], e.ids[1:])
    if len(e.block_ids) != 2:
        raise HyperiadError(f"{name} takes an input–output edge, got a {e.symmetry.lower()} edge")
    return e.block_ids


@builtin("EdgeLabel", "EdgeLabel(e) is the label of the edge e, or null.")
def _edge_label(e):
    return _edge(e, "EdgeLabel").label


@builtin("HypergraphSkeleton", "HypergraphSkeleton(hg) drops every vertex and edge label, keeping symmetry types.")
def _skeleton(hg):
    return _hypergraph(hg, "HypergraphSkeleton").skeleton()


@builtin(
    "SimpleHypergraphQ",
    "SimpleHypergraphQ(hg) is true if no edge of hg is repeated (labels set aside). "
    "A vertex repeated within one edge is allowed.",
)
def _simple_q(hg):
    return simple_hypergraph_q(hg)


def _method(m: Any, name: str) -> str:
    text = m.name if isinstance(m, Symbol) else m
    if not isinstance(text, str) or text.lower() not in ("automatic", "bruteforce"):
        raise HyperiadError(f'{name}: Method is Automatic or "BruteForce", got {format_value(m, 40)}')
    return text.lower()


def _iso_cost(*hgs, Method="Automatic", **_opts):  # noqa: N803
    """Refinement passes x incidences per leaf; brute force tries every ordering."""
    brute = _method(Method, "CanonicalHypergraph") == "bruteforce"
    cost = 0
    for x in hgs:
        h = _hypergraph(x, "CanonicalHypergraph")
        n, size = h["VertexCount"], h["EdgeCount"] + sum(h["Arity"]) + 1
        if brute:
            cost += math.factorial(n) * size
        else:
            cost += n * n * size * 10
    return cost


@builtin(
    "LabelToken",
    "LabelToken(k) is the k-th label of a canonical form, shown ℓk: canonical forms count labels only up to "
    "renaming, so they replace them by tokens.",
)
def _label_token(k):
    return LabelToken(_int(k, "LabelToken"))


@builtin(
    "CanonicalHypergraph",
    """CanonicalHypergraph(hg) is the representative of hg's isomorphism class: vertices renamed 1, ..., n and
labels replaced by tokens ℓ1, ℓ2, ... Isomorphic hypergraphs have equal canonical forms.
Options: Labeled = false canonicalizes the skeleton; Method = "BruteForce" tries every vertex ordering.""",
    cost=_iso_cost,
)
def _canonical_hypergraph(hg, Labeled=True, Method="Automatic"):  # noqa: N803 - option names as written
    h = _hypergraph(hg, "CanonicalHypergraph")
    return canonical_hypergraph(h, labeled=bool(Labeled), method=_method(Method, "CanonicalHypergraph"))


@builtin(
    "IsomorphicHypergraphQ",
    """IsomorphicHypergraphQ(g, h) is true if a renaming of vertices carries g onto h, each edge keeping its
symmetry type. Labels count up to renaming (vertex labels and edge labels independently).
Options: Labeled = false compares skeletons; Method = "BruteForce".""",
    cost=_iso_cost,
)
def _isomorphic_q(g, h, Labeled=True, Method="Automatic"):  # noqa: N803
    g, h = _hypergraph(g, "IsomorphicHypergraphQ"), _hypergraph(h, "IsomorphicHypergraphQ")
    return isomorphic(g, h, labeled=bool(Labeled), method=_method(Method, "IsomorphicHypergraphQ"))


@builtin(
    "FindHypergraphIsomorphism",
    """FindHypergraphIsomorphism(g, h) is a vertex map {v: w, ...} carrying g onto h, or {} if there is none.
Labels = true returns {"Vertices": ..., "VertexLabels": ..., "EdgeLabels": ...}, with the label
correspondences. Options Labeled and Method as for IsomorphicHypergraphQ.""",
    cost=_iso_cost,
)
def _find_isomorphism(g, h, Labeled=True, Method="Automatic", Labels=False):  # noqa: N803
    g, h = _hypergraph(g, "FindHypergraphIsomorphism"), _hypergraph(h, "FindHypergraphIsomorphism")
    return find_isomorphism(
        g, h, labeled=bool(Labeled), method=_method(Method, "FindHypergraphIsomorphism"), labels=bool(Labels)
    )


def _plot_cost(hg, **_opts):
    h = _hypergraph(hg, "HypergraphPlot")
    return 300 * (h["VertexCount"] + h["EdgeCount"]) ** 2  # spring layout: iterations × pairs


def _coordinates(c: Any) -> Any:
    """VertexCoordinates as given: {v: [x, y]} or a list of [x, y] in vertex order."""
    if c is None:
        return None

    def point(p):
        if not (isinstance(p, (list, tuple)) and len(p) == 2):
            raise HyperiadError(f"a vertex position is [x, y], got {format_value(p, 40)}")
        return (float(p[0]), float(p[1]))

    if isinstance(c, dict):
        return {k: point(p) for k, p in c.items()}
    if isinstance(c, list):
        return [point(p) for p in c]
    raise HyperiadError("VertexCoordinates is {vertex: [x, y], ...} or a list of positions in vertex order")


def _highlights(h: Hypergraph, items: Any, name: str) -> list[int]:
    """Highlight = [i, ...]: 1-based positions in EdgeList(hg), or edges given by value."""
    n, marked = h["EdgeCount"], []
    for i in items:
        if isinstance(i, Edge):
            marked += [k for k, e in enumerate(h.edges) if e == i or e.skeleton() == i]
        elif isinstance(i, int) and not isinstance(i, bool) and 1 <= i <= n:
            marked.append(i - 1)
        else:
            raise HyperiadError(f"{name}: Highlight positions are 1..{n} (or edges), got {format_value(i, 20)}")
    return marked


@builtin(
    "HypergraphPlot",
    """HypergraphPlot(hg) draws hg. Options: VertexLabels = true writes vertex identifiers;
Highlight = [i, ...] draws the i-th edges of EdgeList(hg) highlighted (or edges given by value);
HighlightVertices = [v, ...] highlights those vertices; VertexCoordinates = {v: [x, y], ...}
fixes positions (the rest settle among them); Title = "...".""",
    cost=_plot_cost,
)
def _hypergraph_plot(  # noqa: N803 - option names as written
    hg, VertexLabels=False, Highlight=(), HighlightVertices=(), VertexCoordinates=None, Title=""
):
    h = _hypergraph(hg, "HypergraphPlot")
    svg = hypergraph_svg(
        h,
        labels=bool(VertexLabels),
        highlight=_highlights(h, Highlight, "HypergraphPlot"),
        highlight_vertices=list(HighlightVertices),
        coordinates=_coordinates(VertexCoordinates),
        title=str(Title),
    )
    return Graphic(svg, Title)


@builtin(
    "HypergraphEmbedding",
    "HypergraphEmbedding(hg) is the list of vertex positions [x, y] HypergraphPlot uses, in vertex order. "
    "VertexCoordinates = ... fixes some, as for HypergraphPlot.",
    cost=_plot_cost,
)
def _hypergraph_embedding(hg, VertexCoordinates=None):  # noqa: N803
    h = _hypergraph(hg, "HypergraphEmbedding")
    return [list(p) for p in hypergraph_embedding(h, coordinates=_coordinates(VertexCoordinates))]


# -- rewriting -----------------------------------------------------------


def _rule(x: Any, name: str) -> HypergraphRule:
    if not isinstance(x, HypergraphRule):
        raise HyperiadError(f"{name} takes a rewrite rule, got {format_value(x, 60)}")
    return x


def _match_cost(rule, hg, **_opts):
    """Partial matches tried: per input edge, every host edge in every ordering; then the free vertices."""
    r, h = _rule(rule, "HypergraphRuleApply"), _hypergraph(hg, "HypergraphRuleApply")
    cost = 1.0
    for e in r.input.edges:
        orderings = symmetry_size(e.symmetry, e.arity)
        cost *= max(1, len(h.edges)) * orderings
    on_edges = {v.id for e in r.input.edges for v in e.vertices}
    for k in range(sum(1 for v in r.input.vertices if v.id not in on_edges)):
        cost *= max(1, len(h.vertices) - k)
    return cost * (len(r.input.edges) + len(r.output.edges) + 1)


@builtin(
    "HypergraphRule",
    """HypergraphRule(input, output) is the rewrite rule taking hypergraph input to output; either side may be
anything Hypergraph accepts. rule(hg) applies it. In the input, a label that is a bare name (x) is a variable
binding whatever it meets, no label matches anything, anything else ("f", 1) only itself; Verbatim(x) matches
x literally. A composite label holding names is a structured pattern: comp(x, id(X)) matches comp(f, id(A)),
binding x and X (syntactically: no equations between labels are assumed). Vertex and edge labels share their
bindings. The output's labels are written with the bindings put in, e.g. Row(["(", x, y, ")"]).
Properties: rule["Input"], ["Output"], ["SharedVertices"], ["NewVertices"], ["DeletedVertices"], ["Signature"].""",
)
def _hypergraph_rule(lhs, rhs):
    return HypergraphRule(lhs, rhs)


@builtin("HypergraphRuleQ", "HypergraphRuleQ(x) is true if x is a rewrite rule.")
def _hypergraph_rule_q(x):
    return isinstance(x, HypergraphRule)


@builtin(
    "HypergraphRuleApply",
    """HypergraphRuleApply(rule, hg) (or rule(hg)) is the list of events, one per way the rule's input matches hg.
An event draws as the match highlighted beside the result with its creations highlighted; ev["Hypergraph"] is the
result, and ev["MatchEdgePositions"], ["MatchVertices"], ["NewVertices"], ["NewEdges"], ["DeletedVertices"],
["EdgeOrigins"], ["RuleVertexMap"], ["LabelBindings"], ["Input"], ["Rule"] say what happened.
Options: Labeled = false ignores labels; LabelPatterns = false reads names in labels literally;
DistinctVertices = false lets two vertices of the rule land on one (an isolated one too); DistinctEdges = false
lets one edge of hg fill several edges of the rule's input.""",
    cost=_match_cost,
)
def _rule_apply(rule, hg, Labeled=True, LabelPatterns=True, DistinctVertices=True, DistinctEdges=True):  # noqa: N803
    r, h = _rule(rule, "HypergraphRuleApply"), _hypergraph(hg, "HypergraphRuleApply")
    return apply_rule(
        r,
        h,
        labeled=bool(Labeled),
        label_patterns=bool(LabelPatterns),
        distinct_vertices=bool(DistinctVertices),
        distinct_edges=bool(DistinctEdges),
    )


@builtin(
    "HypergraphRuleMatches",
    """HypergraphRuleMatches(rule, hg) is the list of ways the rule's input sits in hg, without rewriting:
{"MatchEdgePositions": ..., "RuleVertexMap": ..., "LabelBindings": ...} each. Options as HypergraphRuleApply.""",
    cost=_match_cost,
)
def _rule_matches(rule, hg, Labeled=True, LabelPatterns=True, DistinctVertices=True, DistinctEdges=True):  # noqa: N803
    r, h = _rule(rule, "HypergraphRuleMatches"), _hypergraph(hg, "HypergraphRuleMatches")
    found = rule_matches(
        r,
        h,
        labeled=bool(Labeled),
        label_patterns=bool(LabelPatterns),
        distinct_vertices=bool(DistinctVertices),
        distinct_edges=bool(DistinctEdges),
    )
    return [match_value(m) for m in found]


@builtin("Verbatim", "Verbatim(x), as a label in a rule's input, matches the label x literally instead of binding it.")
def _verbatim(x):
    return Verbatim(x)


@builtin("FormalV", "FormalV(k) is the k-th vertex name a rewrite gives to the vertices it creates, shown vₖ.")
def _formal_v(k):
    return FormalV(_int(k, "FormalV"))


def _rule_plot_cost(rule, **_opts):
    r = _rule(rule, "HypergraphRulePlot")
    return _plot_cost(r.input) + _plot_cost(r.output)


@builtin(
    "HypergraphRulePlot",
    """HypergraphRulePlot(rule) draws the rule: input, arrow, output, on a shared layout, with the vertices it
destroys and creates highlighted. Options: VertexLabels = true; Highlight = false; SharedCoordinates = false.""",
    cost=_rule_plot_cost,
)
def _rule_plot(rule, VertexLabels=False, Highlight=True, SharedCoordinates=True):  # noqa: N803
    r = _rule(rule, "HypergraphRulePlot")
    return Graphic(rule_svg(r, labels=bool(VertexLabels), highlight=bool(Highlight), shared=bool(SharedCoordinates)))


@builtin(
    "HypergraphEventPlot",
    """HypergraphEventPlot(event) draws an event: what it acted on with the match highlighted, an arrow, and the
result with the created vertices and edges highlighted. HypergraphEventPlot(rule, hg) draws every event.
Options: VertexLabels = true; SharedCoordinates = false.""",
)
def _event_plot(x, hg=None, VertexLabels=False, SharedCoordinates=True):  # noqa: N803
    if hg is not None:
        return [_event_plot(e, None, VertexLabels, SharedCoordinates) for e in _rule_apply(x, hg)]
    if not isinstance(x, Event):
        raise HyperiadError(
            f"HypergraphEventPlot takes an event (or a rule and a hypergraph), got {format_value(x, 60)}"
        )
    return Graphic(event_svg(x, labels=bool(VertexLabels), shared=bool(SharedCoordinates)))


# -- plex diagrams ---------------------------------------------------------


def _plex_env(arrays: Any, diagram: Hypergraph) -> Any:
    if isinstance(arrays, dict):
        return {k: as_array(v) for k, v in arrays.items()}
    if isinstance(arrays, list):
        return [as_array(a) for a in arrays]
    raise HyperiadError("PlexProduct takes the arrays as a list (in PlexNames order) or {name: array, ...}")


def _plex_cost(diagram, arrays, **_opts):
    d = _hypergraph(diagram, "PlexProduct")
    names = plex_names(d)
    given = arrays if isinstance(arrays, list) else []
    env = arrays if isinstance(arrays, dict) else dict(zip(names, given, strict=False))
    dims: dict = {}
    for e in d.edges:
        a = env.get(e.label)
        if a is None:
            continue
        shape = as_array(a).dims
        for v, n in zip(e.ids, shape, strict=False):
            dims[v] = n
    return max(1, len(d.edges)) * math.prod(dims.values() or [1])


@builtin(
    "PlexDiagram",
    """PlexDiagram(edges, contracted) is the plex diagram on these edges (vertex lists or Edge objects): a
labelled hypergraph whose edges are plexes, named p1, p2, ... (or Names = [...]), and whose vertices in
contracted are marked • (summed over); the rest are free. Symmetry = "Unordered" makes plain lists
unordered plexes (taking symmetric arrays). A directed edge's vertex order is its array's index order.""",
)
def _plex_diagram(edges, contracted=(), Names=None, Symmetry="Directed"):  # noqa: N803 - option names as written
    return plex_diagram(edges, list(contracted), names=Names, symmetry=Symmetry)


@builtin("PlexNames", "PlexNames(d) is the list of plex names of the diagram d, the order PlexProduct takes arrays in.")
def _plex_names(d):
    return plex_names(_hypergraph(d, "PlexNames"))


@builtin("PlexFreeVertices", "PlexFreeVertices(d) is the free (unmarked) vertices of d: the indices of its product.")
def _plex_free(d):
    return plex_free(_hypergraph(d, "PlexFreeVertices"))


@builtin("PlexContractedVertices", "PlexContractedVertices(d) is the marked vertices of d: the indices summed over.")
def _plex_contracted(d):
    return plex_contracted(_hypergraph(d, "PlexContractedVertices"))


@builtin(
    "PlexProduct",
    """PlexProduct(d, arrays) evaluates the plex diagram d: arrays is a list (in PlexNames order) or {name: array};
the result's indices are the free vertices in vertex order (or Output = [...]), summed over the marked ones.
A label that is a term P(x, y, ...) of a diagram in Diagrams = {"P": d2} is evaluated as that product first.""",
    cost=_plex_cost,
)
def _plex_product(d, arrays, Output=None, Diagrams=None):  # noqa: N803
    diagram = _hypergraph(d, "PlexProduct")
    return plex_product(diagram, _plex_env(arrays, diagram), output=Output, diagrams=Diagrams or {})


@builtin(
    "PlexRule",
    """PlexRule(d, "P") is the plex diagram d as a rewrite rule: its plexes, with variable labels, become one plex on
its free vertices labelled P(x, y, ...). PlexEvents(rule, d2) applies it where it really contracts.""",
)
def _plex_rule(d, name, Output=None):  # noqa: N803
    if not isinstance(name, str):
        raise HyperiadError(f"PlexRule: the product's name is a string, got {format_value(name, 40)}")
    return plex_rule(_hypergraph(d, "PlexRule"), name, output=Output)


@builtin(
    "PlexEvents",
    "PlexEvents(rule, d) is the events of a plex rule on the plex diagram d that contract all the rule's marked "
    "vertices (no plex outside the match uses them): exactly those keep the value of d.",
    cost=_match_cost,
)
def _plex_events(rule, d):
    return plex_events(_rule(rule, "PlexEvents"), _hypergraph(d, "PlexEvents"))


def _bell(m: int) -> int:
    row = [1]
    for _ in range(m):
        nxt = [row[-1]]
        for x in row:
            nxt.append(nxt[-1] + x)
        row = nxt
    return row[0]


def _enumerate_cost(order, arity, **opts):
    n, k = _int(order, "EnumeratePlexProducts"), _int(arity, "EnumeratePlexProducts")
    group = math.factorial(k) * math.factorial(n) if opts.get("Equivalence", "strict") != "strict" else 1
    return _bell(n * k) * (n * k) ** n * group


@builtin(
    "EnumeratePlexProducts",
    """EnumeratePlexProducts(n, k) is the list of closed plex products of order n and arity k: plex diagrams with
k directed plexes of order n (arguments x1, ..., xk) and n free indices (named 1..n in output order), one per
class. Equivalence = "strict" (one per distinct operation), "isomorphism" (up to a common permutation of index
positions, which gives isomorphic algebras), "parastrophy" (also permuting arguments) or "hypergraph" (up to
isomorphism of the diagrams). Connected = false and Diagonals = true widen the enumeration.""",
    cost=_enumerate_cost,
)
def _enumerate_plex(order, arity, Equivalence="strict", Connected=True, Diagonals=False):  # noqa: N803
    n, k = _int(order, "EnumeratePlexProducts"), _int(arity, "EnumeratePlexProducts")
    shapes = enumerate_plex_products(
        n, k, equivalence=str(Equivalence), connected=bool(Connected), diagonals=bool(Diagonals)
    )
    return [s.diagram() for s in shapes]


# -- rewriting systems and causality ---------------------------------------


def _system(x: Any, name: str) -> RewritingSystem:
    if not isinstance(x, RewritingSystem):
        raise HyperiadError(f"{name} takes a rewriting system, got {format_value(x, 60)}")
    return x


def _system_rules(spec: Any) -> Any:
    """Rules as given: one rule, an arrow lhs -> rhs, or a list of them."""
    items = spec if isinstance(spec, list) else [spec]
    out = []
    for r in items:
        if isinstance(r, RuleValue):
            r = HypergraphRule(r.lhs, r.rhs)
        out.append(r)
    return out if isinstance(spec, list) else out[0]


def _system_options(opts: dict) -> dict:
    """Options as written in Hyperiad: Automatic stands for the default behaviour."""
    out = {}
    for k, v in opts.items():
        if isinstance(v, sympy.Symbol) and v.name == "Automatic":
            v = None
        out[k] = v
    return out


def _shown(value: Any) -> Any:
    """A property's value as Hyperiad shows it."""
    if isinstance(value, list) and value and isinstance(value[0], CausalEdge):
        return [causal_edge_value(c) for c in value]
    if isinstance(value, dict) and "MaxStates" in value:  # the options in force
        return {k: sympy.Symbol("Automatic") if v is None else v for k, v in value.items()}
    return value


def _evolution_cost(system, prop="CausalGraph", n=1, **opts):
    """States the evolution reaches (capped by MaxStates), times the matching work on each.

    Generation by generation: each state branches as many ways as its rules match, which is
    the count on the initial state scaled by how the state has grown (to the power of the
    number of edges a rule's input has).
    """
    s = _system(system, "HypergraphRewritingSystem")
    if isinstance(prop, str) and prop in _STATIC_PROPS:
        return 0
    n = n if isinstance(n, int) and not isinstance(n, bool) else 1
    o = s.options_in_force(**_system_options(opts))
    worst = max((len(r.input.edges) for r in s.rules), default=1) or 1
    growth = max((len(r.output.edges) - len(r.input.edges) for r in s.rules), default=0)
    size0 = max(len(s.initial.edges), 1)
    branching = sum(len(rule_matches(r, s.initial)) for r in s.rules)
    states, frontier, size = 1, 1, size0
    for _ in range(n):
        size += max(growth, 0)
        frontier *= max(branching * (size / size0) ** worst, 0)
        states += frontier
        if states >= o["MaxStates"]:
            states = o["MaxStates"]
            break
    cost = states * (size**worst) * 10
    if prop in ("CausalGraph", "CausalEdges", "EvolutionCausalGraph"):
        cost += states**2
    return cost


_STATIC_PROPS = ("Rules", "RuleCount", "InitialState", "Options", "Properties")


@builtin(
    "HypergraphRewritingSystem",
    """HypergraphRewritingSystem(rules, init) is the rewriting system of the rules (a rule, lhs -> rhs, or a list)
started from the hypergraph init. sys("Prop", n) or sys["Prop", n] is a property of its evolution carried n
generations: "StateCount", "EventCount", "StateList", "States", "EventList", "CausalEdges", "CausalGraph",
"EvolutionGraph", "StatesGraph", "BranchialGraph", "EvolutionCausalGraph", "TokenEventGraph"; sys["Rules"],
["InitialState"], ["Options"]. Options held by the system or given to a property: CausalDependence =
"Elements" | "Edges" | "Vertices", TransitiveReduction, CanonicalizeStates, MaxStates, Labeled, LabelPatterns,
DistinctVertices, DistinctEdges, StateRendering, EventRendering, MaxDrawnVertices, PanelSize.""",
)
def _rewriting_system(rules, init, **options):
    return RewritingSystem(_system_rules(rules), init, _system_options(options))


@builtin("HypergraphRewritingSystemQ", "HypergraphRewritingSystemQ(x) is true if x is a rewriting system.")
def _rewriting_system_q(x):
    return isinstance(x, RewritingSystem)


@builtin(
    "RewritingSystemProperty",
    'RewritingSystemProperty(sys, "Prop", n) is sys("Prop", n): a property of the evolution carried n generations.',
    cost=_evolution_cost,
)
def _system_property(system, prop="Properties", n=1, **options):
    s = _system(system, "HypergraphRewritingSystem")
    if not isinstance(prop, str):
        raise HyperiadError(f"a rewriting system's properties are named by strings, got {format_value(prop, 40)}")
    return _shown(s.prop(prop, n, **_system_options(options)))


def _causal_cost(x, *args, **opts):
    if isinstance(x, RewritingSystem):
        return _evolution_cost(x, "CausalGraph", *args[:1], **opts)
    if len(args) >= 1 and isinstance(args[0], Hypergraph):
        return _evolution_cost(RewritingSystem(_system_rules(x), args[0]), "CausalGraph", *args[1:2], **opts)
    return 0


@builtin(
    "HypergraphCausalGraph",
    """HypergraphCausalGraph(sys, n) is the causal graph of sys evolved n generations: one vertex per event, an
edge from an event to a later one on the same path that touches some of the same elements.
HypergraphCausalGraph(rules, init, n) builds the system first. Options as for HypergraphRewritingSystem.""",
    cost=_causal_cost,
)
def _causal_graph(x, *args, **options):
    o = _system_options(options)
    if isinstance(x, RewritingSystem):
        return causal_graph(x, *args, **o)
    if not args:
        raise HyperiadError(f"HypergraphCausalGraph takes a rewriting system, got {format_value(x, 60)}")
    return causal_graph(_system_rules(x), *args, **o)


# -- lists and numbers ---------------------------------------------------


def _int(x: Any, what: str) -> int:
    if isinstance(x, int) and not isinstance(x, bool):
        return x
    raise HyperiadError(f"{what} must be an integer, got {format_value(x, 40)}")


@builtin("Range", "Range(n) is [1, ..., n]; Range(a, b) is [a, ..., b]; Range(a, b, step).")
def _range(a, b=None, step=1):
    if b is None:
        a, b = 1, a
    a, b, step = _int(a, "Range"), _int(b, "Range"), _int(step, "Range step")
    if step == 0:
        raise HyperiadError("Range step cannot be 0")
    return list(range(a, b + (1 if step > 0 else -1), step))


@builtin("Length", "Length(x) is the number of elements of a list (or the first dimension of an array).")
def _length(x):
    if isinstance(x, ArrayObject):
        return x.dims[0] if x.order else 0
    if isinstance(x, (list, str, dict)):
        return len(x)
    return 0


@builtin("Total", "Total(list) adds the elements of a list; Total(array) adds all entries of an array.")
def _total(x):
    if isinstance(x, ArrayObject):
        items = x.data.reshape(-1).tolist()
    elif isinstance(x, list):
        items = x
    else:
        raise HyperiadError("Total takes a list or an array")
    total: Any = 0
    for item in items:
        total = arith.plus(total, item)
    return total


@builtin("Map", "Map(f, list) applies f to each element: Map(x => x^2, [1, 2, 3]).", session=True)
def _map(session: Session, f, xs):
    if not isinstance(xs, list):
        raise HyperiadError("Map takes a list")
    return [session.apply(f, [x]) for x in xs]


@builtin("Select", "Select(list, test) keeps the elements for which test gives true.", session=True)
def _select(session: Session, xs, test):
    return [x for x in xs if arith.truth(session.apply(test, [x]), "Select")]


def _list(x: Any, name: str) -> list:
    if not isinstance(x, list):
        raise HyperiadError(f"{name} takes lists, got {format_value(x, 60)}")
    return x


@builtin("Join", "Join(l1, l2, ...) is the lists one after another; Join(hg1, hg2, ...) the hypergraphs' union.")
def _join(*xs):
    if xs and all(isinstance(x, Hypergraph) for x in xs):
        seen: dict = {}
        for x in xs:
            for v in x.vertices:
                seen.setdefault(v.id, v)
        return Hypergraph(list(seen.values()), [e for x in xs for e in x.edges])
    return [y for x in xs for y in _list(x, "Join")]


@builtin("Flatten", "Flatten(list) removes all nesting; Flatten(list, 1) one level.")
def _flatten(xs, levels=None):
    depth = None if levels is None else _int(levels, "Flatten")

    def go(x, d):
        if isinstance(x, list) and (depth is None or d < depth):
            return [z for y in x for z in go(y, d + 1)]
        return [x]

    return [z for y in _list(xs, "Flatten") for z in go(y, 0)]


@builtin("Union", "Union(l1, l2, ...) is the distinct elements of the lists, in first-seen order.")
def _union(*xs):
    out: list = []
    for x in xs:
        for y in _list(x, "Union"):
            if y not in out:
                out.append(y)
    return out


@builtin(
    "Tuples",
    "Tuples(list, n) is every list of n elements of list (with repetition); Tuples([l1, l2, ...]) one from each.",
    cost=lambda xs, n=None, **_: float(len(xs)) ** (n if isinstance(n, int) else 1) if isinstance(xs, list) else 1.0,
)
def _tuples(xs, n=None):
    if n is None:
        return [list(t) for t in itertools.product(*[_list(x, "Tuples") for x in _list(xs, "Tuples")])]
    return [list(t) for t in itertools.product(_list(xs, "Tuples"), repeat=_int(n, "Tuples"))]


@builtin(
    "Subsets",
    "Subsets(list) is every sublist (in order); Subsets(list, k) those of k elements.",
    cost=lambda xs, k=None, **_: 2.0 ** len(xs) if isinstance(xs, list) else 1.0,
)
def _subsets(xs, k=None):
    items = _list(xs, "Subsets")
    sizes = range(len(items) + 1) if k is None else [_int(k, "Subsets")]
    return [list(c) for size in sizes for c in itertools.combinations(items, size)]


@builtin("Fold", "Fold(f, x, list) is f(...f(f(x, l1), l2)..., ln): Fold((a, b) => a + b, 0, [1, 2, 3]) is 6.",
         session=True)  # fmt: skip
def _fold(session: Session, f, x, xs):
    for y in _list(xs, "Fold"):
        x = session.apply(f, [x, y])
    return x


@builtin(
    "RandomInteger",
    "RandomInteger([lo, hi]) is a random integer; RandomInteger([lo, hi], dims) a nested list of them.",
    session=True,
)
def _random_integer(session: Session, bounds, dims=None):
    lo, hi = (0, bounds) if isinstance(bounds, int) else bounds
    lo, hi = _int(lo, "RandomInteger"), _int(hi, "RandomInteger")

    def draw(ds):
        return session.rng.randint(lo, hi) if not ds else [draw(ds[1:]) for _ in range(ds[0])]

    shape = [] if dims is None else [dims] if isinstance(dims, int) else list(dims)
    return draw(shape)


@builtin("SeedRandom", "SeedRandom(n) makes the random numbers that follow reproducible.", session=True)
def _seed_random(session: Session, n):
    session.rng.seed(_int(n, "SeedRandom"))


def _entrywise(fn, x):
    if isinstance(x, ArrayObject):
        flat = [fn(v) for v in x.data.reshape(-1).tolist()]
        return ArrayObject(_reshape(flat, x.dims) if x.dims else flat[0])
    if isinstance(x, list):
        return [_entrywise(fn, v) for v in x]
    return fn(x)


def _reshape(flat: list, dims: list[int]) -> list:
    if len(dims) == 1:
        return flat
    step = len(flat) // dims[0]
    return [_reshape(flat[i * step : (i + 1) * step], dims[1:]) for i in range(dims[0])]


@builtin("Simplify", "Simplify(x) simplifies a symbolic expression (entry by entry on arrays and lists).")
def _simplify(x):
    return _entrywise(lambda v: arith.norm(sympy.simplify(v)), x)


@builtin("Expand", "Expand(x) multiplies out products and powers.")
def _expand(x):
    return _entrywise(lambda v: arith.norm(sympy.expand(v)), x)


@builtin("Factor", "Factor(x) factors a polynomial.")
def _factor(x):
    return _entrywise(lambda v: arith.norm(sympy.factor(v)), x)


@builtin("N", "N(x) is the numerical value of x; N(x, digits) to the given precision.")
def _n(x, digits=None):
    def num(v):
        r = sympy.N(v, digits) if digits else sympy.N(v)
        if r.is_real and r.is_Number and digits is None:
            return float(r)
        if r.is_Number and digits is None:
            return complex(r)
        return r

    return _entrywise(num, x)


@builtin("Print", "Print(x, ...) shows each value in the cell's output.", session=True)
def _print(session: Session, *xs):
    from hyperiad.display import display

    for x in xs:
        session.printed.extend(display(x))


# -- help and the research registry -------------------------------------


@builtin("Help", 'Help(name) describes a builtin: Help("ArrayMultiply") or Help(ArrayMultiply).')
def _help(name):
    key = name.name if hasattr(name, "name") else str(name)
    if key not in BUILTINS:
        raise HyperiadError(f"no builtin named {key}")
    return BUILTINS[key].usage


@builtin("Names", 'Names() lists the builtins; Names("Array*") those matching a pattern.')
def _names(pattern="*"):
    return sorted(n for n in BUILTINS if fnmatch.fnmatchcase(n, pattern))


@builtin(
    "Concept", 'Concept(key) shows a definition from the research registry, e.g. Concept("hypermatrix.array_multiply").'
)
def _concept(key):
    from frontier import REGISTRY

    try:
        c = REGISTRY.concept(key)
    except KeyError:
        hits = [x.key for x in REGISTRY.search(str(key))][:8]
        raise HyperiadError(f"no concept {key!r}" + (f"; did you mean {', '.join(hits)}?" if hits else "")) from None
    sources = "".join(f"\n\n*Source ({s.role}):* `{s.ref}` {s.locator}" for s in c.sources)
    md = f"**{c.name}** · `{c.key}` · *{c.maturity.value}*\n\n{c.definition}{sources}"
    return {"__display__": "concept", "markdown": md}


@builtin(
    "Concepts", 'Concepts(direction) lists the registry keys of a research direction, e.g. Concepts("hypermatrix").'
)
def _concepts(direction=None):
    from frontier import REGISTRY

    return [c.key for c in REGISTRY.concepts(direction)]


# -- held forms ------------------------------------------------------------


@special("If", "If(cond, a, b) is a when cond is true and b when it is false (b defaults to null).")
def _if(session: Session, args, scope: Scope):
    if len(args) not in (2, 3):
        raise HyperiadError("If takes a condition and one or two branches")
    if arith.truth(session.eval(args[0], scope), "If"):
        return session.eval(args[1], scope)
    return session.eval(args[2], scope) if len(args) == 3 else None


@special(
    "Table",
    "Table(expr, [i, n]) is [expr for i = 1..n]; iterators [i, a, b], [i, a, b, step] and [i, list]; "
    "several iterators nest, the first outermost.",
)
def _table(session: Session, args, scope: Scope):
    if len(args) < 2:
        raise HyperiadError("Table takes an expression and at least one iterator [i, n]")
    return _iterate(session, args[0], list(args[1:]), scope)


def _iterate(session: Session, body, iterators, scope: Scope):
    it = iterators[0]
    if not (isinstance(it, Expr) and it.head_name == "List" and it.args and isinstance(it.args[0], Symbol)):
        raise HyperiadError("an iterator is written [i, n], [i, a, b], [i, a, b, step] or [i, list]")
    var, bounds = it.args[0].name, [session.eval(a, scope) for a in it.args[1:]]
    if len(bounds) == 1 and isinstance(bounds[0], list):
        values = bounds[0]
    elif 1 <= len(bounds) <= 3:
        values = _range(*bounds) if len(bounds) > 1 else _range(bounds[0])
    else:
        raise HyperiadError("an iterator is written [i, n], [i, a, b], [i, a, b, step] or [i, list]")
    out = []
    for v in values:
        inner = Scope({var: v}, scope)
        out.append(session.eval(body, inner) if len(iterators) == 1 else _iterate(session, body, iterators[1:], inner))
    return out


# -- animations and controls ---------------------------------------------------


def _iterator_values(session: Session, it, scope: Scope) -> tuple[str, list]:
    if not (isinstance(it, Expr) and it.head_name == "List" and it.args and isinstance(it.args[0], Symbol)):
        raise HyperiadError("a control is written [x, list], [x, n] or [x, a, b], as a Table iterator")
    bounds = [session.eval(a, scope) for a in it.args[1:]]
    if len(bounds) == 1 and isinstance(bounds[0], list):
        return it.args[0].name, bounds[0]
    if 1 <= len(bounds) <= 3:
        return it.args[0].name, list(_range(*bounds) if len(bounds) > 1 else _range(bounds[0]))
    raise HyperiadError("a control is written [x, list], [x, n] or [x, a, b], as a Table iterator")


def _frames(values: list, labels: list[str], title: str, play: bool, interval: Any) -> Frames:
    """Frames of the pictures of ``values``; each label is followed by that picture's caption."""
    from hyperiad.display import value_picture

    if not values:
        raise HyperiadError("there is nothing to show: the list of frames is empty")
    if len(values) > MAX_FRAMES:
        raise HyperiadError(f"at most {MAX_FRAMES} frames, got {len(values)}")
    try:
        pictures = [value_picture(v) for v in values]
    except ValueError as e:
        raise HyperiadError(f"every frame must be something drawn: {e}") from None
    full = tuple(f"{label} · {cap}" if cap else label for label, (_, cap) in zip(labels, pictures, strict=False))
    return Frames(tuple(svg for svg, _ in pictures), full, str(title), play, int(interval))


MAX_FRAMES = 120


def _controlled(session: Session, args, scope: Scope, name: str, play: bool) -> Frames:
    opts = {}
    rest = []
    for a in args[1:]:
        if isinstance(a, Expr) and a.head_name == "Option":
            o = session.eval(a, scope)
            opts[o.name] = o.value
        else:
            rest.append(a)
    if len(args) < 2 or len(rest) != 1:
        raise HyperiadError(f"{name} takes an expression and one control [x, values]: {name}(expr, [x, values])")
    var, values = _iterator_values(session, rest[0], scope)
    shown = [session.eval(args[0], Scope({var: v}, scope)) for v in values[: MAX_FRAMES + 1]]
    labels = [f"{var} = {format_value(v, 40)}" for v in values]
    return _frames(shown, labels, opts.get("Title", ""), bool(opts.get("Play", play)), opts.get("Interval", 900))


@special(
    "Manipulate",
    "Manipulate(expr, [x, values]) draws expr for each value of x, with a slider and step buttons to choose x. "
    '[x, n] and [x, a, b] count as in Table. Options: Play = true starts it playing; Interval = ms; Title = "...".',
)
def _manipulate(session: Session, args, scope: Scope):
    return _controlled(session, args, scope, "Manipulate", play=False)


@special(
    "Animate",
    "Animate(expr, [x, values]) is Manipulate that starts playing: expr drawn for each value of x in turn. "
    'Options: Interval = ms (default 900); Title = "...".',
)
def _animate(session: Session, args, scope: Scope):
    return _controlled(session, args, scope, "Animate", play=True)


@builtin(
    "ListAnimate",
    "ListAnimate(list) shows the pictures of a list one after another. Options: Labels = [...], Interval = ms, "
    'Play = false for a stepper, Title = "...".',
)
def _list_animate(xs, Labels=(), Interval=900, Play=True, Title=""):  # noqa: N803 - option names as written
    if not isinstance(xs, list):
        raise HyperiadError("ListAnimate takes a list of things to draw")
    labels = [str(x) for x in Labels] or [f"{k} / {len(xs)}" for k in range(1, len(xs) + 1)]
    return _frames(xs, labels, Title, bool(Play), Interval)


@builtin("Mod", "Mod(a, n) is the remainder of a on division by n, between 0 and n - 1.")
def _mod(a, n):
    if isinstance(a, float) and a.is_integer():
        a = int(a)
    return a % n


@builtin("EvenQ", "EvenQ(n) is true for an even whole number.")
def _even_q(n):
    return isinstance(n, (int, float)) and not isinstance(n, bool) and float(n).is_integer() and int(n) % 2 == 0


@builtin("OddQ", "OddQ(n) is true for an odd whole number.")
def _odd_q(n):
    return isinstance(n, (int, float)) and not isinstance(n, bool) and float(n).is_integer() and int(n) % 2 == 1


from hyperiad import (  # noqa: E402,F401 - infrageometry builtins register
    calculus_builtins,
    geometry_builtins,
    plex_identity_builtins,  # noqa: E402,F401 - plex identity builtins register
    rewriting_modes_builtins,  # noqa: E402,F401 - one pass, composition, saturation
)
