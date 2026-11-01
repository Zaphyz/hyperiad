r"""Spatial hypergraphs and their incidence spaces: points, infinitesimal positions, topology.

A **spatial hypergraph** $H = (V, E)$ is an undirected simple hypergraph: $E$ is a set of
nonempty subsets of $V$; an edge may have a single vertex. Vertices are the points of space;
edges are its connectivity, standing where the continuum has uncountably many points between
any two points.

The **incidence space** $X(H)$ (fixed by the user, 2026-10-28) puts vertices and edges side by
side as the points of one finite topological space, ordered by incidence ($v \le e$ iff
$v \in e$), whose open sets are the up-sets. Vertices are closed points, the *positions*; edges
are open points, the *infinitesimal positions*. The smallest open set around a vertex is the
vertex with its star, so every neighbourhood of a position contains the infinitesimal positions
around it. Every topological notion here (open, closed, closure, interior, frontier, subspace)
is taken in $X(H)$, and a :class:`Region` is a subset of it.

Nothing outside hypergraphs is used: no numbers beyond counting, no real-valued structure.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import Any

from frontier.hypergraphs.core import Edge, Hypergraph, order_key
from frontier.registry import Evidence, EvidenceKind, claim, define, symbol

KEY = "infrageometry"
EXPERIMENTS = "themes/hypergraph-geometry/questions/q01-foundations/experiments"

symbol(f"{KEY}.incidence_space_of", "X(H)", r"X(H)", "the incidence space of a spatial hypergraph H")
symbol(f"{KEY}.star_of", "St(v)", r"\operatorname{St}(v)", "the star of a vertex: the edges that contain it")
symbol(f"{KEY}.link_of", "Lk(v)", r"\operatorname{Lk}(v)", "the link of a vertex")
symbol(f"{KEY}.ball", "B_r(v)", r"B_r(v)", "the ball of radius r about a vertex, in hop distance")
symbol(f"{KEY}.closed_neighbourhood", "N[v]", r"N[v]", "the closed neighbourhood of a vertex")
symbol(f"{KEY}.minimal_open", "U_x", r"U_x", "the smallest open set of X(H) containing the point x")

Point = Hashable  # a vertex identifier, or a frozenset (an edge)


def sort_points(xs: Iterable[Any]) -> list:
    return sorted(xs, key=point_key)


def edge_key(e: frozenset) -> tuple:
    return (len(e), tuple(order_key(v) for v in sorted(e, key=order_key)))


def point_key(x: Any) -> tuple:
    """Vertices first (in identifier order), then edges by size and contents."""
    return (1, edge_key(x)) if isinstance(x, frozenset) else (0, order_key(x))


def is_edge(x: Any) -> bool:
    return isinstance(x, frozenset)


def edge_text(e: frozenset) -> str:
    return "{" + ", ".join(str(v) for v in sorted(e, key=order_key)) + "}"


def point_text(x: Any) -> str:
    return edge_text(x) if is_edge(x) else str(x)


@define(
    f"{KEY}.spatial_hypergraph",
    "Spatial hypergraph",
    r"A spatial hypergraph $H = (V, E)$ is an undirected simple hypergraph: a set $V$ of **points** "
    r"(vertices) and a set $E$ of nonempty subsets of $V$ (edges), so no edge repeats and no vertex "
    r"repeats within an edge. An edge $\{v\}$ with one vertex is allowed. Edges are the connectivity of "
    r"space: where the continuum has uncountably many points between two points, $H$ has the edges "
    r"that contain both. It is *finitary* when every vertex lies in finitely many edges and every edge "
    r"is finite. Settled by the user, 2026-10-27 and 2026-10-28.",
    maturity="working",
    depends_on=("common.hypergraph",),
    tags=("infrageometry", "space"),
)
class SpatialHypergraph:
    """``SpatialHypergraph(edges, vertices=(), coordinates=None)``: edges are iterables of vertex identifiers.

    Vertex identifiers may be any hashable value except a frozenset, which names an edge
    when points of the incidence space are mixed. ``coordinates`` ({vertex: (x, y)}) only says
    where to draw the vertices; it takes no part in equality.
    """

    __slots__ = ("vertices", "edges", "_star", "coordinates")

    def __init__(
        self,
        edges: Iterable[Iterable[Point]],
        vertices: Iterable[Point] = (),
        coordinates: dict | None = None,
    ) -> None:
        es = frozenset(frozenset(_vertex(v) for v in e) for e in edges)
        if frozenset() in es:
            raise ValueError("a spatial hypergraph has no empty edge")
        vs = frozenset(_vertex(v) for v in vertices).union(*es)
        star: dict[Point, set] = {v: set() for v in vs}
        for e in es:
            for v in e:
                star[v].add(e)
        object.__setattr__(self, "vertices", vs)
        object.__setattr__(self, "edges", es)
        object.__setattr__(self, "_star", {v: frozenset(s) for v, s in star.items()})
        coords = {v: tuple(map(float, coordinates[v])) for v in vs if coordinates and v in coordinates}
        object.__setattr__(self, "coordinates", coords if len(coords) == len(vs) and vs else None)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("SpatialHypergraph is immutable")

    @classmethod
    def from_hypergraph(cls, hg: Hypergraph) -> SpatialHypergraph:
        """Forget symmetry types and labels; refuse repeated edges or a repeated vertex in an edge."""
        edges = []
        for e in hg.edges:
            if len(set(e.ids)) != e.arity:
                raise ValueError(f"{e!r} repeats a vertex, so it is not an edge of a spatial hypergraph")
            edges.append(frozenset(e.ids))
        if len(set(edges)) != len(edges):
            raise ValueError("the hypergraph repeats an edge, so it is not simple")
        return cls(edges, hg.vertex_ids)

    def to_hypergraph(self) -> Hypergraph:
        return Hypergraph(self.vertex_list(), [Edge(sort_points(e), "Unordered") for e in self.edge_list()])

    # -- listing, in a fixed order --------------------------------------------

    def vertex_list(self) -> list[Point]:
        return sort_points(self.vertices)

    def edge_list(self) -> list[frozenset]:
        return sorted(self.edges, key=edge_key)

    def points(self) -> list[Point]:
        """The points of $X(H)$: vertices, then edges."""
        return self.vertex_list() + self.edge_list()

    def drawing_edges(self) -> list[list[Point]]:
        """The edges as vertex lists, in :meth:`edge_list` order, for the drawing functions."""
        return [sort_points(e) for e in self.edge_list()]

    def incidences(self) -> int:
        return sum(len(e) for e in self.edges)

    # -- points and their neighbourhoods --------------------------------------

    def star(self, v: Point) -> frozenset:
        r"""$\operatorname{St}(v)$: the edges containing ``v``."""
        return self._star[v]

    def degree(self, v: Point) -> int:
        return len(self._star[v])

    def neighbourhood(self, v: Point, closed: bool = True) -> frozenset:
        """$N[v]$, the vertices sharing an edge with ``v`` (with ``v``); $N(v)$ when not ``closed``."""
        nb = frozenset({v}).union(*self._star[v])
        return nb if closed else nb - {v}

    def distances(self, v: Point, radius: int | None = None) -> dict[Point, int]:
        """Hop distance from ``v``: crossing one edge counts one."""
        dist = {v: 0}
        queue = deque([v])
        while queue:
            u = queue.popleft()
            if radius is not None and dist[u] >= radius:
                continue
            for w in sort_points(self.neighbourhood(u)):
                if w not in dist:
                    dist[w] = dist[u] + 1
                    queue.append(w)
        return dist

    def distance(self, v: Point, w: Point) -> int | None:
        """The hop distance, or ``None`` between different components."""
        return self.distances(v).get(w)

    def ball(self, v: Point, r: int) -> frozenset:
        return frozenset(self.distances(v, r))

    def sphere(self, v: Point, r: int) -> frozenset:
        return frozenset(w for w, d in self.distances(v, r).items() if d == r)

    def components(self) -> list[frozenset]:
        seen: set = set()
        comps = []
        for v in self.vertex_list():
            if v not in seen:
                c = frozenset(self.distances(v))
                seen |= c
                comps.append(c)
        return comps

    def link(self, v: Point) -> SpatialHypergraph:
        r"""$\operatorname{Lk}(v)$: on $N(v)$, the edges $e \setminus \{v\}$ for $e \in \operatorname{St}(v)$."""
        return SpatialHypergraph((e - {v} for e in self._star[v] if len(e) > 1), self.neighbourhood(v, closed=False))

    # -- subspaces -------------------------------------------------------------

    def induced(self, vertices: Iterable[Point]) -> SpatialHypergraph:
        """The induced subspace on a vertex set: the edges lying inside it."""
        u = frozenset(vertices)
        return SpatialHypergraph((e for e in self.edges if e <= u), u, self.coordinates)

    def trace(self, vertices: Iterable[Point]) -> SpatialHypergraph:
        """The trace on a vertex set: each edge cut down to the set, when it meets it."""
        u = frozenset(vertices)
        return SpatialHypergraph((e & u for e in self.edges if e & u), u, self.coordinates)

    def region(self) -> Region:
        """The whole of $X(H)$ as a region."""
        return Region(self.vertices, self.edges)

    def relabel(self, mapping: dict) -> SpatialHypergraph:
        coords = {mapping[v]: p for v, p in self.coordinates.items()} if self.coordinates else None
        return SpatialHypergraph(
            ([mapping[v] for v in e] for e in self.edges), [mapping[v] for v in self.vertices], coords
        )

    def with_coordinates(self, coordinates: dict) -> SpatialHypergraph:
        return SpatialHypergraph(self.edges, self.vertices, coordinates)

    # -- value semantics -------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SpatialHypergraph) and (self.vertices, self.edges) == (other.vertices, other.edges)

    def __hash__(self) -> int:
        return hash(("SpatialHypergraph", self.vertices, self.edges))

    def __repr__(self) -> str:
        edges = self.drawing_edges()
        isolated = [v for v in self.vertex_list() if not self._star[v]]
        return f"SpatialHypergraph({edges!r})" if not isolated else f"SpatialHypergraph({edges!r}, {isolated!r})"


def _vertex(v: Any) -> Point:
    if isinstance(v, (frozenset, set)):
        raise ValueError(f"{v!r}: a vertex identifier may not be a set (sets name edges)")
    return tuple(v) if isinstance(v, list) else v


# -- regions and the incidence space ----------------------------------------------------


@dataclass(frozen=True)
class Region:
    """A subset of the incidence space $X(H)$: some vertices and some edges."""

    vertices: frozenset = frozenset()
    edges: frozenset = frozenset()

    @classmethod
    def of(cls, points: Iterable[Point]) -> Region:
        pts = list(points)
        return cls(frozenset(x for x in pts if not is_edge(x)), frozenset(x for x in pts if is_edge(x)))

    def points(self) -> list[Point]:
        return sort_points(self.vertices) + sorted(self.edges, key=edge_key)

    def __or__(self, other: Region) -> Region:
        return Region(self.vertices | other.vertices, self.edges | other.edges)

    def __and__(self, other: Region) -> Region:
        return Region(self.vertices & other.vertices, self.edges & other.edges)

    def __sub__(self, other: Region) -> Region:
        return Region(self.vertices - other.vertices, self.edges - other.edges)

    def __le__(self, other: Region) -> bool:
        return self.vertices <= other.vertices and self.edges <= other.edges

    def __contains__(self, x: Point) -> bool:
        return x in self.edges if is_edge(x) else x in self.vertices

    def __len__(self) -> int:
        return len(self.vertices) + len(self.edges)

    def __repr__(self) -> str:
        return "Region(" + ", ".join(point_text(x) for x in self.points()) + ")"


@define(
    f"{KEY}.incidence_space",
    "Incidence space",
    r"The incidence space $X(H)$ of a spatial hypergraph is the set $V \sqcup E$ ordered by "
    r"incidence, $v \le e$ iff $v \in e$, with the Alexandrov topology whose open sets are the "
    r"up-sets. Vertices are **closed points**, the *positions*; edges are **open points**, the "
    r"*infinitesimal positions*. The smallest open set around a vertex is "
    r"$U_v = \{v\} \cup \operatorname{St}(v)$, and around an edge it is $U_e = \{e\}$; the closure of an "
    r"edge is the edge with its vertices. An infinitesimal position thus lies *between* its vertices: "
    r"every neighbourhood of a position contains the infinitesimal positions around it. Fixed as the "
    r"topology of space by the user, 2026-10-28.",
    maturity="working",
    notation=(f"{KEY}.incidence_space_of", f"{KEY}.star_of", f"{KEY}.minimal_open"),
    depends_on=(f"{KEY}.spatial_hypergraph",),
    tags=("infrageometry", "topology"),
)
class IncidenceSpace:
    """Topological operations of $X(H)$ on :class:`Region` objects."""

    def __init__(self, space: SpatialHypergraph) -> None:
        self.space = space

    def __eq__(self, other: object) -> bool:
        return isinstance(other, IncidenceSpace) and self.space == other.space

    def __hash__(self) -> int:
        return hash(("IncidenceSpace", self.space))

    def below(self, x: Point) -> frozenset:
        """The points $\\le x$: an edge's vertices with the edge, or a vertex alone."""
        return frozenset(x) | {x} if is_edge(x) else frozenset({x})

    def above(self, x: Point) -> frozenset:
        """The points $\\ge x$: a vertex with its star, or an edge alone."""
        return frozenset({x}) if is_edge(x) else self.space.star(x) | {x}

    def minimal_open(self, x: Point) -> Region:
        """$U_x$: for a vertex, the vertex and its star; for an edge, the edge alone."""
        return Region.of(self.above(x))

    def is_open(self, r: Region) -> bool:
        return all(self.space.star(v) <= r.edges for v in r.vertices)

    def is_closed(self, r: Region) -> bool:
        return all(e <= r.vertices for e in r.edges)

    def open_hull(self, r: Region) -> Region:
        """The smallest open region containing ``r``: add the star of each vertex."""
        return Region(r.vertices, r.edges.union(*(self.space.star(v) for v in r.vertices)))

    def closure(self, r: Region) -> Region:
        """The smallest closed region containing ``r``: add the vertices of each edge."""
        return Region(r.vertices.union(*r.edges), r.edges)

    def interior(self, r: Region) -> Region:
        """The largest open region inside ``r``: drop the vertices whose star leaves ``r``."""
        return Region(frozenset(v for v in r.vertices if self.space.star(v) <= r.edges), r.edges)

    def frontier(self, r: Region) -> Region:
        return self.closure(r) - self.interior(r)

    def is_connected(self, r: Region) -> bool:
        """Connected in the subspace topology: linked by incidences inside ``r``."""
        pts = r.points()
        if not pts:
            return True
        seen, todo = {pts[0]}, [pts[0]]
        while todo:
            x = todo.pop()
            near = (x & r.vertices) if is_edge(x) else (self.space.star(x) & r.edges)
            for y in near:
                if y not in seen:
                    seen.add(y)
                    todo.append(y)
        return len(seen) == len(pts)

    def subspace(self, r: Region) -> SpatialHypergraph:
        r"""The subspace on ``r`` as a spatial hypergraph: the trace of its edges on its vertices.

        Defined when the edges of ``r`` meet its vertices in distinct nonempty sets; then
        $X(\text{result})$ is homeomorphic to ``r`` with the subspace topology.
        """
        traces = [e & r.vertices for e in r.edges]
        if any(not t for t in traces):
            raise ValueError("an edge of the region has none of its vertices in it (a free infinitesimal position)")
        if len(set(traces)) != len(traces):
            raise ValueError("two edges of the region meet it in the same vertices, so it is not simple")
        return SpatialHypergraph(traces, r.vertices)

    def first_betti(self) -> int:
        r"""Rank of $H_1(X(H))$, the cycle rank of the incidence graph: $|I| - |V| - |E| + c$."""
        s = self.space
        return s.incidences() - len(s.vertices) - len(s.edges) + len(s.components())


@define(
    f"{KEY}.neighbourhood",
    "Neighbourhood, distance and ball",
    r"The closed neighbourhood of a point is $N[v] = \{v\} \cup \bigcup \operatorname{St}(v)$, the "
    r"points sharing an edge with $v$; it is the vertex part of the closure of $U_v$. The **hop "
    r"distance** $d(v, w)$ is the least number of edges crossed by a motion from $v$ to $w$, and the "
    r"ball is $B_r(v) = \{w : d(v, w) \le r\}$, so $B_1(v) = N[v]$; the sphere $S_r(v)$ is where "
    r"$d = r$.",
    maturity="working",
    notation=(f"{KEY}.closed_neighbourhood", f"{KEY}.ball"),
    depends_on=(f"{KEY}.spatial_hypergraph", f"{KEY}.incidence_space"),
    tags=("infrageometry", "metric"),
)
def ball(space: SpatialHypergraph, v: Point, r: int) -> frozenset:
    return space.ball(v, r)


@define(
    f"{KEY}.link",
    "Link of a point",
    r"The link of $v$ is the spatial hypergraph $\operatorname{Lk}(v)$ on $N(v) = N[v] \setminus \{v\}$ "
    r"with edges $e \setminus \{v\}$ for $e \in \operatorname{St}(v)$, $|e| \ge 2$. Its edges are in "
    r"bijection with the tangent vectors at $v$ other than the stationary one $v \rightharpoonup \{v\}$: "
    r"the link is the discrete **unit sphere** of directions at $v$. For a triangulated surface given "
    r"by its triangles the link of a point is a cycle, a circle.",
    maturity="working",
    notation=(f"{KEY}.link_of",),
    depends_on=(f"{KEY}.spatial_hypergraph",),
    tags=("infrageometry", "local"),
)
def link(space: SpatialHypergraph, v: Point) -> SpatialHypergraph:
    return space.link(v)


@define(
    f"{KEY}.subspace",
    "Subspace",
    r"A subspace of $H$ is a subset $S$ of $X(H)$ with the subspace topology. It is again an incidence "
    r"space: $S \cong X(H_S)$ for the spatial hypergraph $H_S$ on the vertices of $S$ whose edges are the "
    r"traces $e \cap S$ of the edges of $S$, provided these are nonempty and distinct. Special cases: a "
    r"**closed** subspace is a sub-hypergraph; an **open** subspace holds, with each vertex, every edge "
    r"through it (edges may dangle); for a vertex set $U$, the *induced* subspace $H[U]$ keeps the edges "
    r"inside $U$, the *trace* $H|_U$ keeps every $e \cap U \neq \emptyset$, and the *open hull* "
    r"$U \cup \operatorname{St}(U)$ is the smallest open region containing $U$.",
    maturity="sketch",
    depends_on=(f"{KEY}.incidence_space",),
    tags=("infrageometry", "subspace"),
)
def subspace(space: SpatialHypergraph, region: Region) -> SpatialHypergraph:
    return IncidenceSpace(space).subspace(region)


@define(
    f"{KEY}.link_dimension",
    "Link dimension",
    r"The link dimension of $H$ at $v$ is defined through links: "
    r"$\dim_v H = 1 + \max_{w} \dim_w \operatorname{Lk}(v)$ over the vertices $w$ of the link, with the "
    r"maximum over no vertices equal to $-1$; so a point whose edges are at most $\{v\}$ has dimension $0$, "
    r"a point of a graph has dimension $1$ and a point inside a triangle has dimension $2$. It is a purely "
    r"local count. $\dim H$ is the maximum over the points.",
    maturity="sketch",
    depends_on=(f"{KEY}.link",),
    tags=("infrageometry", "dimension"),
)
def link_dimension(space: SpatialHypergraph, v: Point | None = None) -> int:
    if v is None:
        return max((link_dimension(space, w) for w in space.vertices), default=-1)
    lk = space.link(v)
    return 1 + max((link_dimension(lk, w) for w in lk.vertices), default=-1)


claim(
    f"{KEY}.closed_regions_are_subhypergraphs",
    r"The closed subsets of $X(H)$ are exactly the sub-hypergraphs of $H$ (edge sets with a vertex set "
    r"containing them), and the open subsets are exactly the regions holding the star of each of their "
    r"vertices.",
    about=(f"{KEY}.incidence_space", f"{KEY}.subspace"),
    status="supported",
    evidence=[
        Evidence(EvidenceKind.ARGUMENT, "open = up-set in the incidence order; closed = complement = down-set"),
        Evidence(
            EvidenceKind.NUMERICAL,
            "topology generated from the minimal open sets agrees, on every subset of X(H), for 300 random H",
            source=f"{EXPERIMENTS}/2026-10-27-incidence-space-checks.py",
        ),
    ],
)

claim(
    f"{KEY}.frontier_of_open_hull",
    r"For a vertex set $U$, the frontier in $X(H)$ of its open hull $U \cup \operatorname{St}(U)$ is the "
    r"outer vertex boundary $N[U] \setminus U$; the frontier of a closed subspace $S$ is the set of its "
    r"vertices at which some edge of $H$ leaves $S$ (nonempty normal set).",
    about=(f"{KEY}.incidence_space", f"{KEY}.subspace"),
    status="supported",
    evidence=[
        Evidence(EvidenceKind.ARGUMENT, "closure adds the vertices of the star; interior of an open set is itself"),
        Evidence(
            EvidenceKind.NUMERICAL,
            "every vertex set and sub-hypergraph of 300 random H",
            source=f"{EXPERIMENTS}/2026-10-27-incidence-space-checks.py",
        ),
    ],
)

claim(
    f"{KEY}.subspaces_are_traces",
    r"A subset $S \subseteq X(H)$ whose edges meet its vertices in distinct nonempty sets, with the "
    r"subspace topology, is homeomorphic to $X(H_S)$ where $H_S$ has the vertices of $S$ and the traces "
    r"$e \cap S$ of the edges of $S$. Without the proviso the subspace is the incidence space of a "
    r"hypergraph with repeated or empty edges.",
    about=(f"{KEY}.subspace", f"{KEY}.incidence_space"),
    status="supported",
    evidence=[
        Evidence(
            EvidenceKind.ARGUMENT,
            "in an Alexandrov space the subspace topology is the Alexandrov topology of the restricted order; "
            "restricted to S, v <= e iff v is in e ∩ S",
        ),
        Evidence(
            EvidenceKind.NUMERICAL,
            "every subset of X(H) satisfying the proviso, 200 random H: subspace opens = opens of X(H_S)",
            source=f"{EXPERIMENTS}/2026-10-28-foundations-checks.py",
        ),
    ],
)

claim(
    f"{KEY}.incidence_space_homotopy",
    r"$X(H)$ is weakly homotopy equivalent to the incidence (Levi) graph of $H$, a 1-dimensional complex; "
    r"so its fundamental group is free of rank $|I| - |V| - |E| + c$ ($I$ the incidences, $c$ the "
    r"components) and hyperedges are *not* filled in: an edge $\{a, b\}$ inside a hyperedge "
    r"$\{a, b, c\}$ makes a loop.",
    about=(f"{KEY}.incidence_space",),
    status="supported",
    evidence=[
        Evidence(
            EvidenceKind.LITERATURE,
            "McCord (1966): a finite T0 space is weakly equivalent to the order complex of its poset; here "
            "chains have length at most 1, so the order complex is the incidence graph",
        ),
        Evidence(
            EvidenceKind.NUMERICAL,
            "the q01 example: first Betti number 2 in X(H), 1 in the simplicial closure",
            source=f"{EXPERIMENTS}/2026-10-27-incidence-space-checks.py",
        ),
    ],
)
