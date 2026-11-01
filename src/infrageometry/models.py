r"""Local models and charts: when a spatial hypergraph looks everywhere like a model space.

A manifold is a space that looks locally like a fixed model. Here "looks like" is isomorphism of
pointed balls: the ball of radius $r$ about each point of $H$ is isomorphic, centre to centre, to
a ball of the model. The isomorphism is a **chart**. A second reading, closer to combinatorial
topology, asks that every link be a sphere; for a surface given by its triangles, a cycle.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable

from frontier.hypergraphs import Edge, Hypergraph, Vertex
from frontier.hypergraphs.canon import canonical_hypergraph, find_isomorphism
from frontier.registry import Evidence, EvidenceKind, claim, define

from infrageometry.maps import SpatialMap
from infrageometry.space import EXPERIMENTS, KEY, SpatialHypergraph, sort_points

Point = Hashable
CENTRE = "centre"


@define(
    f"{KEY}.pointed_ball",
    "Pointed ball",
    r"The pointed ball $(B_r(v), v)$ is the subspace of $H$ induced on $B_r(v)$, with $v$ marked as its "
    r"centre. Two pointed balls are isomorphic when an isomorphism of spatial hypergraphs takes centre to "
    r"centre.",
    maturity="sketch",
    depends_on=(f"{KEY}.neighbourhood", f"{KEY}.subspace"),
    tags=("infrageometry", "manifold"),
)
def pointed_ball(space: SpatialHypergraph, v: Point, r: int) -> SpatialHypergraph:
    return space.induced(space.ball(v, r))


def _marked(space: SpatialHypergraph, v: Point, r: int) -> Hypergraph:
    ball = pointed_ball(space, v, r)
    return Hypergraph(
        [Vertex(w, CENTRE if w == v else None) for w in ball.vertex_list()],
        [Edge(sort_points(e), "Unordered") for e in ball.edge_list()],
    )


def ball_type(space: SpatialHypergraph, v: Point, r: int) -> Hypergraph:
    """The isomorphism class of the pointed ball, as a canonical form."""
    return canonical_hypergraph(_marked(space, v, r))


@define(
    f"{KEY}.chart",
    "Chart and local model",
    r"Let $M$ be a model space and $C$ a set of its points (for a homogeneous model such as the grid "
    r"$\mathbb{Z}^n$, one point). A **chart** of $H$ at $v$ of radius $r$ is an isomorphism of pointed "
    r"balls $(B_r(v), v) \to (B_r(m), m)$ with $m \in C$. $H$ is **locally modelled** on $M$ at radius "
    r"$r$ when it has a chart at every point; the charts form an **atlas**. Where two balls overlap, "
    r"composing one chart with the inverse of the other gives a transition map between pieces of $M$.",
    maturity="sketch",
    depends_on=(f"{KEY}.pointed_ball", f"{KEY}.spatial_map"),
    tags=("infrageometry", "manifold"),
)
def chart(
    space: SpatialHypergraph, v: Point, model: SpatialHypergraph, centres: Iterable[Point], r: int
) -> SpatialMap | None:
    source = _marked(space, v, r)
    for m in centres:
        iso = find_isomorphism(source, _marked(model, m, r))
        if iso:
            ball, target = pointed_ball(space, v, r), pointed_ball(model, m, r)
            return SpatialMap(ball, target, iso, {e: frozenset(iso[w] for w in e) for e in ball.edges})
    return None


def atlas(
    space: SpatialHypergraph, model: SpatialHypergraph, centres: Iterable[Point], r: int
) -> dict[Point, SpatialMap | None]:
    """A chart at every point (``None`` where there is none)."""
    centres = list(centres)
    types = {}
    for m in centres:
        types.setdefault(ball_type(model, m, r), m)
    out = {}
    for v in space.vertex_list():
        m = types.get(ball_type(space, v, r))
        out[v] = None if m is None else chart(space, v, model, [m], r)
    return out


def locally_modelled(space: SpatialHypergraph, model: SpatialHypergraph, centres: Iterable[Point], r: int) -> bool:
    model_types = {ball_type(model, m, r) for m in centres}
    return all(ball_type(space, v, r) in model_types for v in space.vertices)


def is_cycle_graph(space: SpatialHypergraph) -> bool:
    return (
        len(space.vertices) >= 3
        and all(len(e) == 2 for e in space.edges)
        and all(space.degree(v) == 2 for v in space.vertices)
        and len(space.components()) == 1
    )


@define(
    f"{KEY}.combinatorial_surface",
    "Combinatorial surface",
    r"A spatial hypergraph whose edges are triples is a **combinatorial surface** when the link of every "
    r"point is a cycle: around each point the triangles close up into a disk. This is the link reading "
    r"of being a 2-manifold, the other being local models.",
    maturity="sketch",
    depends_on=(f"{KEY}.link",),
    tags=("infrageometry", "manifold"),
)
def is_combinatorial_surface(space: SpatialHypergraph) -> bool:
    return all(len(e) == 3 for e in space.edges) and all(is_cycle_graph(space.link(v)) for v in space.vertices)


# -- examples ------------------------------------------------------------------------------


def path(n: int) -> SpatialHypergraph:
    return SpatialHypergraph([[i, i + 1] for i in range(n)], range(n + 1))


def cycle(n: int) -> SpatialHypergraph:
    return SpatialHypergraph([[i, (i + 1) % n] for i in range(n)])


def grid(m: int, n: int) -> SpatialHypergraph:
    """The $m \\times n$ piece of $\\mathbb{Z}^2$, vertices $(i, j)$."""
    edges = [[(i, j), (i + 1, j)] for i in range(m - 1) for j in range(n)]
    edges += [[(i, j), (i, j + 1)] for i in range(m) for j in range(n - 1)]
    vs = [(i, j) for i in range(m) for j in range(n)]
    return SpatialHypergraph(edges, vs, {v: (float(v[1]), -float(v[0])) for v in vs})


def torus_grid(m: int, n: int) -> SpatialHypergraph:
    edges = [[(i, j), ((i + 1) % m, j)] for i in range(m) for j in range(n)]
    edges += [[(i, j), (i, (j + 1) % n)] for i in range(m) for j in range(n)]
    return SpatialHypergraph(edges)


def klein_grid(m: int, n: int) -> SpatialHypergraph:
    """A grid closed up as a Klein bottle: across the seam $j = n-1 \\to 0$ the row is reflected."""
    edges = [[(i, j), ((i + 1) % m, j)] for i in range(m) for j in range(n)]
    edges += [[(i, j), (i, j + 1)] for i in range(m) for j in range(n - 1)]
    edges += [[(i, n - 1), ((m - 1 - i) % m, 0)] for i in range(m)]
    return SpatialHypergraph(edges)


def octahedron() -> SpatialHypergraph:
    """The octahedron as a combinatorial sphere: its 8 triangles."""
    return SpatialHypergraph([[a, b, c] for a in (1, 2) for b in (3, 4) for c in (5, 6)])


def torus7() -> SpatialHypergraph:
    """The 7-point triangulation of the torus (Möbius, Császár): 14 triangles."""
    return SpatialHypergraph(
        [[i, (i + 1) % 7, (i + 3) % 7] for i in range(7)] + [[i, (i + 2) % 7, (i + 3) % 7] for i in range(7)]
    )


def with_unary(space: SpatialHypergraph) -> SpatialHypergraph:
    """$H$ with the unary edge $\\{v\\}$ added at every point: every point can rest."""
    return SpatialHypergraph([*space.edges, *([v] for v in space.vertices)], space.vertices, space.coordinates)


def triangle_example() -> SpatialHypergraph:
    """The running example of q01: a hyperedge, a parallel binary edge, a path and a second hyperedge."""
    return SpatialHypergraph([[1, 2, 3], [1, 2], [3, 4], [4, 5, 6], [2, 6]])


claim(
    f"{KEY}.closed_grids_locally_planar",
    r"The torus grid and the Klein-bottle grid $C_m \times C_n$ (with a reflected seam) with $m, n \ge 2r + 2$ "
    r"are locally modelled on $\mathbb{Z}^2$ at radius $r$ (the torus grid exactly then), while the "
    r"triangulated sphere and torus are not locally modelled on a grid. Local models at a fixed radius do "
    r"not see orientation.",
    about=(f"{KEY}.chart",),
    status="supported",
    evidence=[
        Evidence(
            EvidenceKind.NUMERICAL,
            "r = 1, 2 for 3 <= m, n <= 9; the Klein grid is also locally planar at a few smaller sizes with m even",
            source=f"{EXPERIMENTS}/2026-10-28-foundations-checks.py",
        )
    ],
)
