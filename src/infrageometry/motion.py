r"""Tangent vectors, motions, and motion as rewriting.

Built on top of a spatial hypergraph $H$ is its **motion graph**: its nodes are the points of
the incidence space $X(H)$, vertices and edges; each incidence $v \in e$ gives a **departure**
$v \rightharpoonup e$ and an **arrival** $e \rightharpoonup v$. A departure is an infinitesimal
motion, a tangent vector at $v$. An integrated motion alternates: from a vertex to an edge,
from the edge to a vertex, and so on (the user, 2026-10-27).

Everything is discrete and is rewriting:

* a motion is a word in the points of $X(H)$; its **incidence homotopy class** is its normal form
  under the rewrite rule $x\,y\,x \to x$ (cancel a backtrack), which is terminating and confluent.
  The homotopy of motions proper is derived from the area homotopy of their step paths
  (:mod:`infrageometry.paths`; the user, 2026-10-29), which backtrack cancellation implies;
* a moving point is a **token** in a hypergraph rewriting system whose rules are the departures
  and arrivals; its multiway states graph is the motion graph up to the symmetries of $H$.
"""

from __future__ import annotations

import itertools
from collections import deque
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass

import networkx as nx
from frontier.hypergraphs.core import Edge, Hypergraph
from frontier.hypergraphs.rewriting import HypergraphRule
from frontier.registry import Evidence, EvidenceKind, claim, define, symbol

from infrageometry.space import (
    EXPERIMENTS,
    KEY,
    SpatialHypergraph,
    edge_key,
    edge_text,
    is_edge,
    point_key,
    point_text,
    sort_points,
)

symbol(f"{KEY}.departure", "v ⇀ e", r"v \rightharpoonup e", "a tangent vector: the departure from v into the edge e")
symbol(f"{KEY}.tangent_space_at", "T_v H", r"T_v H", "the tangent space of H at the point v")
symbol(f"{KEY}.tangent_bundle_of", "TH", r"TH", "the tangent bundle of H: all departures")
symbol(f"{KEY}.motion_graph_of", "M(H)", r"M(H)", "the motion graph of H")
symbol(f"{KEY}.interval", "I_n", r"I_n", "the interval of length n: the path graph on 0, 1, ..., n")

Point = Hashable


@define(
    f"{KEY}.tangent_vector",
    "Tangent vector",
    r"A tangent vector at a point $v$ is a **departure** $v \rightharpoonup e$ with $e \ni v$: the "
    r"infinitesimal motion from the position $v$ into the infinitesimal position $e$ (the user, "
    r"2026-10-27). The tangent space is $T_v H = \{v \rightharpoonup e : e \in \operatorname{St}(v)\}$, "
    r"a set of $\deg v$ elements. When $\{v\} \in E$, the departure $v \rightharpoonup \{v\}$ is the "
    r"**stationary** vector, the hypergraphic zero; nothing else supplies one. The tangent bundle "
    r"$TH = \bigsqcup_v T_vH$ is the set of incidences, projected to $V$ by "
    r"$v \rightharpoonup e \mapsto v$, with $|TH| = \sum_e |e|$.",
    maturity="working",
    notation=(f"{KEY}.departure", f"{KEY}.tangent_space_at", f"{KEY}.tangent_bundle_of"),
    depends_on=(f"{KEY}.spatial_hypergraph", f"{KEY}.incidence_space"),
    tags=("infrageometry", "tangent"),
)
@dataclass(frozen=True)
class TangentVector:
    base: Point
    edge: frozenset

    def __post_init__(self) -> None:
        if self.base not in self.edge:
            raise ValueError(f"{self.base!r} is not a vertex of {edge_text(self.edge)}")

    @property
    def stationary(self) -> bool:
        return self.edge == frozenset({self.base})

    def __repr__(self) -> str:
        return f"{point_text(self.base)} ⇀ {edge_text(self.edge)}"

    def sort_key(self) -> tuple:
        return (point_key(self.base), edge_key(self.edge))


def tangent_space(space: SpatialHypergraph, v: Point) -> list[TangentVector]:
    return [TangentVector(v, e) for e in sorted(space.star(v), key=edge_key)]


def tangent_bundle(space: SpatialHypergraph) -> list[TangentVector]:
    return [t for v in space.vertex_list() for t in tangent_space(space, v)]


@define(
    f"{KEY}.motion_graph",
    "Motion graph",
    r"The motion graph $M(H)$ is the directed graph on the points of $X(H)$ with a departure "
    r"$v \rightharpoonup e$ and an arrival $e \rightharpoonup v$ for each incidence $v \in e$. Its "
    r"departures are $TH$; forgetting directions gives the incidence (Levi) graph of $H$. It is the graph "
    r"of all one-step moves of a point that can sit at positions and at infinitesimal positions.",
    maturity="working",
    notation=(f"{KEY}.motion_graph_of",),
    depends_on=(f"{KEY}.tangent_vector",),
    tags=("infrageometry", "tangent"),
)
def motion_graph(space: SpatialHypergraph) -> nx.DiGraph:
    """Nodes are the points of $X(H)$ (vertex ids and frozenset edges); arcs carry ``kind``."""
    g = nx.DiGraph()
    g.add_nodes_from(space.points())
    for e in space.edge_list():
        for v in sort_points(e):
            g.add_edge(v, e, kind="departure")
            g.add_edge(e, v, kind="arrival")
    return g


@define(
    f"{KEY}.motion",
    "Motion",
    r"A motion is a walk in $M(H)$: a sequence of points of $X(H)$ alternating between positions and "
    r"infinitesimal positions, each incident to the next, e.g. "
    r"$\gamma = (v_0, e_1, v_1, \dots, e_n, v_n)$ with $v_{i-1}, v_i \in e_i$. Its length is its number "
    r"of **half-steps**; from a vertex to a vertex that is twice the number $n$ of **steps**. At a "
    r"position $v_i$ it has the **velocity** $v_i \rightharpoonup e_{i+1}$ and the **incoming "
    r"direction** $v_i \rightharpoonup e_i$. A step with $v_{i-1} = v_i$ is a *stall*. A **geodesic** "
    r"is a motion with fewest steps between its ends; its step count is the hop distance.",
    maturity="working",
    depends_on=(f"{KEY}.motion_graph", f"{KEY}.neighbourhood"),
    tags=("infrageometry", "path"),
)
class Motion:
    """``Motion(space, [x0, x1, ...])``: points of $X(H)$, each incident to the next.

    Edges may be given as any iterable of vertices; they are stored as frozensets.
    """

    __slots__ = ("space", "points")

    def __init__(self, space: SpatialHypergraph, points: Iterable) -> None:
        pts = tuple(frozenset(x) if isinstance(x, (list, set, frozenset)) else x for x in points)
        if not pts:
            raise ValueError("a motion has at least one point")
        for x in pts:
            if (is_edge(x) and x not in space.edges) or (not is_edge(x) and x not in space.vertices):
                raise ValueError(f"{point_text(x)} is not a point of the space")
        for a, b in itertools.pairwise(pts):
            if is_edge(a) == is_edge(b) or (b not in a if is_edge(a) else a not in b):
                raise ValueError(f"{point_text(a)} → {point_text(b)} is not a half-step (they are not incident)")
        object.__setattr__(self, "space", space)
        object.__setattr__(self, "points", pts)

    def __setattr__(self, name: str, value) -> None:
        raise AttributeError("Motion is immutable")

    @property
    def start(self) -> Point:
        return self.points[0]

    @property
    def end(self) -> Point:
        return self.points[-1]

    def half_steps(self) -> int:
        return len(self.points) - 1

    def vertex_positions(self) -> list[int]:
        return [i for i, x in enumerate(self.points) if not is_edge(x)]

    def steps(self) -> list[tuple[Point, frozenset, Point]]:
        """The steps vertex → edge → vertex, for a motion that starts at a vertex."""
        p = self.points
        first = 0 if not is_edge(p[0]) else 1
        return [(p[i], p[i + 1], p[i + 2]) for i in range(first, len(p) - 2, 2)]

    def velocity(self, i: int) -> TangentVector | None:
        """At the position ``points[i]``, the departure into the next point (None at the end)."""
        if is_edge(self.points[i]) or i + 1 >= len(self.points):
            return None
        return TangentVector(self.points[i], self.points[i + 1])

    def incoming(self, i: int) -> TangentVector | None:
        """At the position ``points[i]``, the departure back into the previous point (None at the start)."""
        if is_edge(self.points[i]) or i == 0:
            return None
        return TangentVector(self.points[i], self.points[i - 1])

    def is_closed(self) -> bool:
        return self.start == self.end

    def is_proper(self) -> bool:
        """No stall: no step that comes back to the vertex it left."""
        return all(a != c for a, _, c in self.steps())

    def reverse(self) -> Motion:
        return Motion(self.space, reversed(self.points))

    def then(self, other: Motion) -> Motion:
        if self.end != other.start:
            raise ValueError("the second motion must start where the first ends")
        return Motion(self.space, self.points + other.points[1:])

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Motion) and (self.space, self.points) == (other.space, other.points)

    def __hash__(self) -> int:
        return hash(("Motion", self.points))

    def __len__(self) -> int:
        return len(self.points) - 1

    def __repr__(self) -> str:
        return "Motion(" + " → ".join(point_text(x) for x in self.points) + ")"


def geodesic(space: SpatialHypergraph, a: Point, b: Point) -> Motion | None:
    """A motion from ``a`` to ``b`` with fewest half-steps (breadth first in $M(H)$, ties by point order)."""
    g = motion_graph(space)
    parent = {a: None}
    queue = deque([a])
    while queue and b not in parent:
        x = queue.popleft()
        for y in sorted(g.successors(x), key=lambda p: (is_edge(p), edge_key(p) if is_edge(p) else (0,), str(p))):
            if y not in parent:
                parent[y] = x
                queue.append(y)
    if b not in parent:
        return None
    path = [b]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    return Motion(space, reversed(path))


# -- intervals and curves -------------------------------------------------------------------


@define(
    f"{KEY}.interval",
    "Interval and curve",
    r"The interval $I_n$ is the path graph on $0, 1, \dots, n$ with edges $\{i, i+1\}$. A motion from a "
    r"vertex to a vertex with $n$ steps is the same thing as a map $\gamma: I_n \to H$ "
    r"(:mod:`infrageometry.maps`): $i \mapsto v_i$, $\{i-1, i\} \mapsto e_i$. Its velocity at $i$ is the "
    r"pushforward $\gamma_*(i \rightharpoonup \{i, i+1\})$ and its incoming direction "
    r"$\gamma_*(i \rightharpoonup \{i-1, i\})$: curves are maps from intervals, as in the continuum.",
    maturity="sketch",
    notation=(f"{KEY}.interval",),
    depends_on=(f"{KEY}.motion",),
    tags=("infrageometry", "path"),
)
def interval(n: int) -> SpatialHypergraph:
    if n < 0:
        raise ValueError("an interval has length n >= 0")
    return SpatialHypergraph([[i, i + 1] for i in range(n)], range(n + 1))


# -- homotopy as rewriting -----------------------------------------------------------------


def reduction_steps(m: Motion) -> list[Motion]:
    """The motion, then each result of cancelling the leftmost backtrack $x\\,y\\,x \\to x$, to the normal form."""
    out = [m]
    pts = list(m.points)
    while True:
        i = next((k for k in range(len(pts) - 2) if pts[k] == pts[k + 2]), None)
        if i is None:
            return out
        del pts[i + 1 : i + 3]
        out.append(Motion(m.space, pts))


@define(
    f"{KEY}.incidence_homotopy",
    "Incidence homotopy of motions",
    r"Two motions with the same ends are **incidence-homotopic** when they rewrite to the same normal form "
    r"under the single rule $x\,y\,x \to x$: a motion that steps into a point and straight back out the way "
    r"it came is the same as staying. The rule is terminating and confluent, so each motion has one "
    r"**reduced** form. Examples: a stall $v \to e \to v$ reduces to $v$; $u \to e \to w \to e \to x$ "
    r"reduces to $u \to e \to x$. This is homotopy in the topological space $X(H)$, where hyperedges are not "
    r"filled: its closed motions at $v$ form the free group $\pi_1(X(H), v)$ on $b_1(X(H))$ generators. It "
    r"refines the homotopy of motions, which is derived from area homotopy (the user, 2026-10-29).",
    maturity="sketch",
    depends_on=(f"{KEY}.motion", f"{KEY}.incidence_space"),
    tags=("infrageometry", "path", "rewriting"),
)
def reduce_motion(m: Motion) -> Motion:
    stack: list = []
    for x in m.points:
        if len(stack) >= 2 and stack[-2] == x:
            stack.pop()
        else:
            stack.append(x)
    return Motion(m.space, stack)


def incidence_homotopic(a: Motion, b: Motion) -> bool:
    return a.start == b.start and a.end == b.end and reduce_motion(a).points == reduce_motion(b).points


def incidence_group_generators(space: SpatialHypergraph, base: Point) -> list[Motion]:
    """Free generators of $\\pi_1(X(H), base)$: reduced closed motions, one per incidence outside a spanning
    tree of the component."""
    g = motion_graph(space).to_undirected()
    comp = nx.node_connected_component(g, base)
    tree = nx.bfs_tree(g.subgraph(comp), base)
    tree_edges = {frozenset(e) for e in tree.to_undirected().edges}
    parent = {b: a for a, b in tree.edges}

    def to_base(x):
        path = [x]
        while path[-1] != base:
            path.append(parent[path[-1]])
        return path

    gens = []
    for a, b in sorted(g.subgraph(comp).edges, key=lambda e: (str(e[0]), str(e[1]))):
        if frozenset((a, b)) not in tree_edges:
            loop = list(reversed(to_base(a))) + to_base(b)
            gens.append(reduce_motion(Motion(space, loop)))
    return gens


# -- motion as a rewriting system -----------------------------------------------------------

TOKEN = "•"


def _token_rules(arities: Iterable[int]) -> list[HypergraphRule]:
    """Departure and arrival rules, one pair per arity, as hypergraph rewrite rules.

    The token is a vertex ``t`` marked by a directed unary edge ``(t)``. At a position ``x`` it
    has the directed edge ``(t, x)``; inside an edge it is one more vertex of that (unordered) edge.
    """
    rules = []
    for k in sorted(set(arities)):
        ys = [f"y{i}" for i in range(2, k + 1)]
        free = Edge(["x", *ys], "Unordered")
        held = Edge(["t", "x", *ys], "Unordered")
        mark, at = Edge(["t"]), Edge(["t", "x"])
        rules.append(HypergraphRule([mark, at, free], [mark, held]))  # departure x ⇀ e
        rules.append(HypergraphRule([mark, held], [mark, at, free]))  # arrival e ⇀ x
    return rules


def token_state(space: SpatialHypergraph, at: Point) -> Hypergraph:
    """$H$ with the token at the point ``at`` of $X(H)$, as a hypergraph."""
    if TOKEN in space.vertices:
        raise ValueError(f"the vertex name {TOKEN!r} is reserved for the token")
    edges = [Edge([TOKEN])]
    for e in space.edge_list():
        edges.append(Edge([TOKEN, *sort_points(e)] if e == at else sort_points(e), "Unordered"))
    if not is_edge(at):
        edges.append(Edge([TOKEN, at]))
    return Hypergraph(space.vertex_list(), edges)


@define(
    f"{KEY}.token_rewriting",
    "Motion as rewriting",
    r"A moving point is a token in a hypergraph rewriting system. The state is $H$ with the token at a "
    r"point of $X(H)$: at a position $x$ it is attached by a directed edge $(t, x)$; at an infinitesimal "
    r"position $e$ it is one more vertex of $e$. For each edge size $k$ there are two rules, a "
    r"**departure** $\{(t), (t, x), \{x, y_2, \dots, y_k\}\} \to \{(t), \{t, x, y_2, \dots, y_k\}\}$ and "
    r"an **arrival**, its reverse. The multiway evolution from one state explores every motion at once, "
    r"and its states graph is the motion graph $M(H)$ with points identified by the symmetries of $H$.",
    maturity="sketch",
    depends_on=(f"{KEY}.motion_graph", "common.hypergraph_rule"),
    tags=("infrageometry", "rewriting"),
)
def token_rewriting_system(space: SpatialHypergraph, at: Point, **options):
    from frontier.hypergraphs.causality import RewritingSystem

    return RewritingSystem(_token_rules(len(e) for e in space.edges), token_state(space, at), options or None)


def automorphisms(space: SpatialHypergraph) -> list[dict]:
    """Every automorphism of a small spatial hypergraph, by brute force (for checks)."""
    vs = space.vertex_list()
    out = []
    for perm in itertools.permutations(vs):
        m = dict(zip(vs, perm, strict=True))
        if {frozenset(m[v] for v in e) for e in space.edges} == space.edges:
            out.append(m)
    return out


def point_orbits(space: SpatialHypergraph, autos: Sequence[dict] | None = None) -> list[frozenset]:
    """The orbits of the automorphism group on the points of $X(H)$."""
    autos = automorphisms(space) if autos is None else autos

    def image(m, x):
        return frozenset(m[v] for v in x) if is_edge(x) else m[x]

    seen: set = set()
    orbits = []
    for x in space.points():
        if x not in seen:
            orbit = frozenset(image(m, x) for m in autos)
            seen |= orbit
            orbits.append(orbit)
    return orbits


claim(
    f"{KEY}.departures_are_path_germs",
    r"The germs at $0$ of continuous paths $\gamma: [0, 1] \to X(H)$ with $\gamma(0) = v$ that are "
    r"constant on some $(0, \varepsilon)$ are exactly the constant germ and the germs "
    r"$\gamma|_{(0,\varepsilon)} = e$ with $e \in \operatorname{St}(v)$: the tangent vectors at $v$. "
    r"Motions are exactly the continuous paths that step through points of $X(H)$, with "
    r"$\gamma(i) = v_i$ and $\gamma = e_i$ on $(i-1, i)$.",
    about=(f"{KEY}.tangent_vector", f"{KEY}.motion", f"{KEY}.incidence_space"),
    status="supported",
    evidence=[
        Evidence(
            EvidenceKind.ARGUMENT,
            "the preimage of U_v is an open set containing 0, so the constant value lies in U_v = {v} ∪ St(v); "
            "conversely an open set containing v contains every edge through v (open sets are up-sets), so "
            "the stepped path is continuous at each integer time",
        )
    ],
)

claim(
    f"{KEY}.backtrack_rewriting_confluent",
    r"The rewrite rule $x\,y\,x \to x$ on motions is terminating and confluent; two motions with the same "
    r"ends are homotopic in $X(H)$ (relative to their ends) exactly when their normal forms agree.",
    about=(f"{KEY}.incidence_homotopy",),
    status="supported",
    evidence=[
        Evidence(
            EvidenceKind.ARGUMENT,
            "length decreases; the only critical pair x y x y reduces both ways to x y (free reduction of "
            "words); homotopy classes of paths in the graph to which X(H) is weakly equivalent are reduced "
            "walks",
        ),
        Evidence(
            EvidenceKind.NUMERICAL,
            "every order of cancellation gives the same normal form, 2000 random motions",
            source=f"{EXPERIMENTS}/2026-10-28-foundations-checks.py",
        ),
    ],
)

claim(
    f"{KEY}.token_states_graph",
    r"The states graph of the token rewriting system started at a point $x$ is the quotient of the "
    r"component of $x$ in $M(H)$ by the automorphisms of $H$: its states are the orbits of points, and "
    r"one orbit leads to another when some point of the first has an arc to some point of the second.",
    about=(f"{KEY}.token_rewriting", f"{KEY}.motion_graph"),
    status="supported",
    evidence=[
        Evidence(
            EvidenceKind.ARGUMENT,
            "the rules move the token along exactly one arc of M(H) and change nothing else; canonical forms "
            "identify states related by an automorphism of H fixing the token's role",
        ),
        Evidence(
            EvidenceKind.NUMERICAL,
            "states and edges agree with brute-force orbits for 40 random connected H",
            source=f"{EXPERIMENTS}/2026-10-28-foundations-checks.py",
        ),
    ],
)
