"""Hyperiad builtins for infrageometry: spatial hypergraphs as discrete spaces.

The mathematics is :mod:`infrageometry`; this module names it in Hyperiad and says how its
values display. Points of an incidence space are vertex identifiers (positions) and edges
(infinitesimal positions); an edge is written as a list of its vertices, ``[1, 2, 3]``, or as
``EdgePoint(1, 2, 3)`` where it must be a value that can key an association. Edges display as
``{1, 2, 3}``. Maps between spaces are given as associations ``{v: w}`` for vertices and as lists
of rules ``[[1, 2] -> [3, 4]]`` (or associations keyed by ``EdgePoint``) for edges.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any

from frontier.hypergraphs import Hypergraph
from frontier.hypergraphs.svg import scene_svg
from infrageometry.space import is_edge, point_text, sort_points

from hyperiad.display import register_display
from hyperiad.errors import HyperiadError
from hyperiad.kernel import builtin
from hyperiad.values import Frames, RuleValue, format_value
from infrageometry import (
    Flow,
    IncidenceSpace,
    Motion,
    Region,
    ScalarFunction,
    SpatialHypergraph,
    SpatialMap,
    TangentVector,
    Transit,
    VectorField,
    all_maps,
    ball_growth,
    curve,
    cycle,
    distance_function,
    geodesic,
    gradient_flow,
    grid,
    growth_dimension,
    homotopic,
    identity_map,
    incidence_group_generators,
    incidence_homotopic,
    interval,
    is_combinatorial_surface,
    klein_grid,
    link_dimension,
    locally_modelled,
    motion_of_curve,
    octahedron,
    path,
    reduce_motion,
    reduction_steps,
    scalar_line,
    tangent_bundle,
    tangent_space,
    token_rewriting_system,
    token_state,
    torus7,
    torus_grid,
    triangle_example,
    with_unary,
)
from infrageometry import chart as make_chart
from infrageometry import pictures as pic

Point = Hashable


# -- values that carry their space, so they can be drawn ------------------------------------


@dataclass(frozen=True)
class RegionValue:
    """A region of the incidence space of ``space``."""

    space: SpatialHypergraph
    region: Region
    title: str = ""

    def __repr__(self) -> str:
        return "Region(" + ", ".join(point_text(x) for x in self.region.points()) + ")"


class Tangents(list):
    """The tangent vectors at a point, as a list that knows its space and base point."""

    def __init__(self, space: SpatialHypergraph, base: Point) -> None:
        super().__init__(tangent_space(space, base))
        self.space, self.base = space, base


@dataclass(frozen=True)
class MotionGraphValue:
    space: SpatialHypergraph

    def __repr__(self) -> str:
        return f"MotionGraph({self.space!r})"


# -- conversions -------------------------------------------------------------------------


def _space(x: Any, name: str) -> SpatialHypergraph:
    if isinstance(x, SpatialHypergraph):
        return x
    if isinstance(x, IncidenceSpace):
        return x.space
    if isinstance(x, Hypergraph):
        try:
            return SpatialHypergraph.from_hypergraph(x)
        except ValueError as e:
            raise HyperiadError(f"{name}: {e}") from None
    if isinstance(x, list):
        return SpatialHypergraph([_edge(e, name) for e in x])
    raise HyperiadError(f"{name} takes a spatial hypergraph (or a list of edges), got {format_value(x, 60)}")


def _edge(x: Any, name: str) -> frozenset:
    if isinstance(x, (list, tuple, frozenset, set)):
        return frozenset(x)
    raise HyperiadError(f"{name}: an edge is a list of vertices, got {format_value(x, 40)}")


def _point(x: Any) -> Point:
    return frozenset(x) if isinstance(x, (list, tuple, frozenset, set)) else x


def _check_point(space: SpatialHypergraph, x: Point, name: str) -> Point:
    if (is_edge(x) and x not in space.edges) or (not is_edge(x) and x not in space.vertices):
        raise HyperiadError(f"{name}: {point_text(x)} is not a point of the space")
    return x


def _vertex(space: SpatialHypergraph, v: Any, name: str) -> Point:
    if is_edge(_point(v)) or v not in space.vertices:
        raise HyperiadError(f"{name}: {format_value(v, 40)} is not a vertex of the space")
    return v


def _pairs(spec: Any, name: str) -> list[tuple[Any, Any]]:
    """An association, or a list of rules ``a -> b``, as pairs."""
    if spec is None:
        return []
    if isinstance(spec, dict):
        return list(spec.items())
    if isinstance(spec, RuleValue):
        return [(spec.lhs, spec.rhs)]
    if isinstance(spec, list) and all(isinstance(r, RuleValue) for r in spec):
        return [(r.lhs, r.rhs) for r in spec]
    raise HyperiadError(f"{name}: give an association {{a: b, ...}} or a list of rules [a -> b, ...]")


def _region(x: Any, name: str) -> RegionValue:
    if isinstance(x, RegionValue):
        return x
    raise HyperiadError(f"{name} takes a region (from Region, MinimalOpen, Ball, ...), got {format_value(x, 60)}")


def _numbered(space: SpatialHypergraph) -> SpatialHypergraph:
    """The same space with its vertices renamed 1, 2, ... in their order (grids have pair names)."""
    return space.relabel({v: k for k, v in enumerate(space.vertex_list(), 1)})


def _motion(x: Any, name: str) -> Motion:
    if isinstance(x, Motion):
        return x
    raise HyperiadError(f"{name} takes a motion (from Motion or Geodesic), got {format_value(x, 60)}")


def _map(x: Any, name: str) -> SpatialMap:
    if isinstance(x, ScalarFunction):
        return x.map
    if isinstance(x, SpatialMap):
        return x
    raise HyperiadError(f"{name} takes a map (from SpatialMap), got {format_value(x, 60)}")


def _scalar(x: Any, name: str) -> ScalarFunction:
    if isinstance(x, ScalarFunction):
        return x
    raise HyperiadError(f"{name} takes a scalar function (from ScalarFunction), got {format_value(x, 60)}")


def _tangent(x: Any, name: str) -> TangentVector:
    if isinstance(x, TangentVector):
        return x
    raise HyperiadError(f"{name} takes a tangent vector (from Departure or TangentSpace), got {format_value(x, 60)}")


def _int(x: Any, name: str) -> int:
    if isinstance(x, bool) or not isinstance(x, int):
        raise HyperiadError(f"{name} needs a whole number, got {format_value(x, 30)}")
    return x


# -- display -------------------------------------------------------------------------------


ZOOM = 1.3  # these pictures carry small marks (squares, arrows): drawn a little larger


def _svg(scene) -> str:
    return scene_svg(scene, zoom=ZOOM)


def _plural(n: int, word: str, many: str | None = None) -> str:
    return f"{n} {word if n == 1 else many or word + 's'}"


def describe_space(h: SpatialHypergraph) -> str:
    sizes = sorted({len(e) for e in h.edges})
    parts = [
        "spatial hypergraph",
        _plural(len(h.vertices), "point"),
        _plural(len(h.edges), "edge") + (f" (sizes {', '.join(map(str, sizes))})" if sizes else ""),
    ]
    comps = len(h.components())
    if comps != 1:
        parts.append(_plural(comps, "component"))
    return " · ".join(parts)


def _show_space(h: SpatialHypergraph) -> list[dict]:
    if len(h.vertices) + len(h.edges) > 220:
        return [{"type": "text", "data": describe_space(h) + " (too large to draw)"}]
    return [{"type": "svg", "data": _svg(pic.space_scene(h, edge_points="none")), "caption": describe_space(h)}]


def _show_incidence(x: IncidenceSpace) -> list[dict]:
    h = x.space
    caption = (
        f"incidence space X(H) · {_plural(len(h.vertices), 'position')} (closed points, below) · "
        f"{_plural(len(h.edges), 'infinitesimal position')} (open points, above) · "
        f"{_plural(h.incidences(), 'incidence')} · first Betti number {x.first_betti()}"
    )
    return [{"type": "svg", "data": _svg(pic.incidence_scene(h)), "caption": caption}]


def _region_kind(r: RegionValue) -> str:
    x = IncidenceSpace(r.space)
    op, cl = x.is_open(r.region), x.is_closed(r.region)
    return "open and closed" if op and cl else "open" if op else "closed" if cl else "neither open nor closed"


def _show_region(r: RegionValue) -> list[dict]:
    pts = ", ".join(point_text(p) for p in r.region.points()) or "nothing"
    head = f"{r.title} · " if r.title else ""
    return [
        {
            "type": "svg",
            "data": _svg(pic.space_scene(r.space, region=r.region)),
            "caption": f"{head}{_plural(len(r.region), 'point')}, {_region_kind(r)}: {pts}",
        }
    ]


def _show_tangents(t: Tangents) -> list[dict]:
    caption = f"T_{point_text(t.base)}H: {_plural(len(t), 'tangent vector')}: " + ", ".join(map(repr, t))
    return [{"type": "svg", "data": _svg(pic.tangent_scene(t.space, t.base)), "caption": caption}]


def _show_motion_graph(g: MotionGraphValue) -> list[dict]:
    n = g.space.incidences()
    caption = (
        f"motion graph M(H) · {_plural(len(g.space.points()), 'node')} · {n} departures v ⇀ e and {n} arrivals "
        f"e ⇀ v (each line of the diagram is one of each)"
    )
    return [{"type": "svg", "data": _svg(pic.incidence_scene(g.space)), "caption": caption}]


def describe_motion(m: Motion) -> str:
    n = m.half_steps()
    steps = f" ({_plural(n // 2, 'step')})" if not is_edge(m.start) and not is_edge(m.end) else ""
    kind = [] if n == 0 else ["closed"] if m.is_closed() else []
    if any(a == c for a, _, c in m.steps()):
        kind.append("with stalls")
    return (
        " · ".join([f"motion {point_text(m.start)} to {point_text(m.end)}", _plural(n, "half-step") + steps, *kind])
        + ": "
        + " → ".join(point_text(x) for x in m.points)
    )


def _show_motion(m: Motion) -> list[dict]:
    return [{"type": "svg", "data": _svg(pic.motion_scene(m)), "caption": describe_motion(m)}]


def describe_map(f: SpatialMap) -> str:
    props = [
        name
        for name, ok in (
            ("isomorphism", f.is_isomorphism()),
            ("embedding", f.is_embedding() and not f.is_isomorphism()),
            ("local isomorphism", f.is_local_isomorphism() and not f.is_isomorphism()),
            ("immersion", f.is_immersion() and not f.is_local_isomorphism()),
            ("submersion", f.is_submersion() and not f.is_local_isomorphism()),
        )
        if ok
    ]
    return "map · " + " · ".join(props or ["neither immersion nor submersion"])


def _show_map(f: SpatialMap) -> list[dict]:
    return [{"type": "svg", "data": _svg(pic.map_scene(f)), "caption": describe_map(f)}]


def describe_scalar(fn: ScalarFunction) -> str:
    crit = fn.critical_points()
    kinds = {}
    for v, k in crit.items():
        kinds.setdefault(k, []).append(point_text(v))
    parts = [f"{k} at {', '.join(vs)}" for k, vs in sorted(kinds.items())]
    return "scalar function to the scalar line · " + (" · ".join(parts) if parts else "every point regular")


def _show_scalar(fn: ScalarFunction) -> list[dict]:
    return [{"type": "svg", "data": _svg(pic.scalar_scene(fn)), "caption": describe_scalar(fn)}]


def _flow_arrows(flow: Flow) -> list:
    arrows = []
    for v in flow.space.vertex_list():
        s = flow.step(v)
        if s is not None:
            arrows.append((v, s[0]))
            if s[1] != v:
                arrows.append((s[0], s[1]))
    return arrows


def _show_flow(flow: Flow) -> list[dict]:
    phi = flow.map()
    fates = {}
    for v in flow.space.vertex_list():
        fates.setdefault(flow.fate(v), []).append(point_text(v))
    caption = "flow φ: " + ", ".join(f"{point_text(v)}→{point_text(w)}" for v, w in phi.items())
    caption += " · integral curves " + "; ".join(f"from {', '.join(vs)}: {k}" for k, vs in fates.items())
    scene = pic.space_scene(flow.space, arrows=_flow_arrows(flow), arrow_color=pic.FLOW)
    return [{"type": "svg", "data": _svg(scene), "caption": caption}]


def _show_field(x: VectorField) -> list[dict]:
    scene = pic.space_scene(x.space, arrows=[(v, e) for v, e in x.choice.items()], arrow_color=pic.FLOW)
    return [
        {
            "type": "svg",
            "data": _svg(scene),
            "caption": "vector field: " + ", ".join(repr(x.at(v)) for v in sort_points(x.choice)),
        }
    ]


def _text(x: Any) -> list[dict]:
    return [{"type": "text", "data": repr(x)}]


register_display(SpatialHypergraph, _show_space)
register_display(IncidenceSpace, _show_incidence)
register_display(RegionValue, _show_region)
register_display(Tangents, _show_tangents)
register_display(MotionGraphValue, _show_motion_graph)
register_display(Motion, _show_motion)
register_display(SpatialMap, _show_map)
register_display(ScalarFunction, _show_scalar)
register_display(Flow, _show_flow)
register_display(VectorField, _show_field)
register_display(TangentVector, _text)
register_display(Transit, lambda t: [{"type": "text", "data": f"transit on {_plural(len(t.rule), 'edge')}"}])


# -- space and its points ----------------------------------------------------------------


@builtin(
    "SpatialHypergraph",
    """SpatialHypergraph(edges) is the space with these edges, each a list of vertices; an edge may have one
vertex. SpatialHypergraph(edges, vertices) adds isolated points. A hypergraph converts too (symmetry
types and labels are forgotten). Edges are undirected and none repeats.""",
)
def _spatial_hypergraph(edges, vertices=()):
    if isinstance(edges, (SpatialHypergraph, Hypergraph)):
        return _space(edges, "SpatialHypergraph")
    if not isinstance(edges, list):
        raise HyperiadError("SpatialHypergraph takes a list of edges, each a list of vertices")
    try:
        return SpatialHypergraph([_edge(e, "SpatialHypergraph") for e in edges], list(vertices))
    except ValueError as e:
        raise HyperiadError(f"SpatialHypergraph: {e}") from None


@builtin(
    "EdgePoint",
    "EdgePoint(1, 2, 3) is the edge {1, 2, 3} as a point of an incidence space (it can key an association).",
)
def _edge_point(*vs):
    return frozenset(vs)


@builtin("Points", "Points(h) lists the points of the incidence space X(h): vertices, then edges.")
def _points(h):
    return _space(h, "Points").points()


@builtin("Positions", "Positions(h) lists the vertices of h: the positions, closed points of X(h).")
def _positions(h):
    return _space(h, "Positions").vertex_list()


@builtin("InfinitesimalPositions", "InfinitesimalPositions(h) lists the edges of h: the open points of X(h).")
def _infinitesimal(h):
    return _space(h, "InfinitesimalPositions").edge_list()


@builtin("Star", "Star(h, v) lists the edges containing the vertex v.")
def _star(h, v):
    s = _space(h, "Star")
    return sorted(s.star(_vertex(s, v, "Star")), key=lambda e: (len(e), sort_points(e)))


@builtin("Degree", "Degree(h, v) is the number of edges containing v: the number of tangent vectors at v.")
def _degree(h, v):
    s = _space(h, "Degree")
    return s.degree(_vertex(s, v, "Degree"))


@builtin("Neighbourhood", "Neighbourhood(h, v) is N[v]: v and every vertex sharing an edge with it, as a region.")
def _neighbourhood(h, v):
    s = _space(h, "Neighbourhood")
    return RegionValue(s, Region(s.neighbourhood(_vertex(s, v, "Neighbourhood"))), f"N[{point_text(v)}]")


@builtin(
    "HopDistance", "HopDistance(h, v, w) is the least number of edges a motion from v to w crosses (null if none)."
)
def _hop_distance(h, v, w):
    s = _space(h, "HopDistance")
    return s.distance(_vertex(s, v, "HopDistance"), _vertex(s, w, "HopDistance"))


@builtin("Ball", "Ball(h, v, r) is the region of the vertices within r steps of v.")
def _ball(h, v, r):
    s = _space(h, "Ball")
    return RegionValue(s, Region(s.ball(_vertex(s, v, "Ball"), _int(r, "Ball"))), f"B_{r}({point_text(v)})")


@builtin("Sphere", "Sphere(h, v, r) is the region of the vertices exactly r steps from v.")
def _sphere(h, v, r):
    s = _space(h, "Sphere")
    return RegionValue(s, Region(s.sphere(_vertex(s, v, "Sphere"), _int(r, "Sphere"))), f"S_{r}({point_text(v)})")


@builtin("Link", "Link(h, v) is the link of v: on its neighbours, the edges e minus v for the edges e through v.")
def _link(h, v):
    s = _space(h, "Link")
    return s.link(_vertex(s, v, "Link"))


@builtin(
    "LinkDimension",
    "LinkDimension(h, v) is 1 + the largest link dimension inside the link of v (0 for a point on no edge but {v}); "
    "LinkDimension(h) is the largest over the points.",
)
def _link_dimension(h, v=None):
    s = _space(h, "LinkDimension")
    return link_dimension(s, None if v is None else _vertex(s, v, "LinkDimension"))


@builtin("InducedSubspace", "InducedSubspace(h, vertices) keeps the edges of h lying inside the given vertices.")
def _induced(h, vs):
    return _space(h, "InducedSubspace").induced(vs)


@builtin("Trace", "Trace(h, vertices) cuts every edge of h down to the given vertices (dropping those it misses).")
def _trace(h, vs):
    return _space(h, "Trace").trace(vs)


# -- the incidence space -------------------------------------------------------------------


@builtin(
    "IncidenceSpace",
    "IncidenceSpace(h) is X(h): vertices and edges as the points of one finite space, v below e when v is in e, "
    "the open sets the up-sets. Drawn as its diagram: positions below, infinitesimal positions above.",
)
def _incidence_space(h):
    return IncidenceSpace(_space(h, "IncidenceSpace"))


@builtin("Region", "Region(h, points) is a set of points of X(h): vertices and edges (edges as lists of vertices).")
def _make_region(h, points):
    s = _space(h, "Region")
    if not isinstance(points, list):
        raise HyperiadError("Region takes a list of points")
    return RegionValue(s, Region.of(_check_point(s, _point(p), "Region") for p in points))


@builtin(
    "MinimalOpen",
    "MinimalOpen(h, x) is U_x, the smallest open set containing the point x: a vertex with its star, or an edge alone.",
)
def _minimal_open(h, x):
    s = _space(h, "MinimalOpen")
    p = _check_point(s, _point(x), "MinimalOpen")
    return RegionValue(s, IncidenceSpace(s).minimal_open(p), f"U_{point_text(p)}")


def _region_op(name: str, op, title: str):
    def fn(r):
        r = _region(r, name)
        return RegionValue(r.space, op(IncidenceSpace(r.space), r.region), title)

    return fn


for _name, _op, _usage in (
    (
        "Closure",
        IncidenceSpace.closure,
        "Closure(region) adds the vertices of its edges: the smallest closed region containing it.",
    ),
    (
        "Interior",
        IncidenceSpace.interior,
        "Interior(region) drops the vertices whose star leaves it: the largest open region inside.",
    ),
    ("Frontier", IncidenceSpace.frontier, "Frontier(region) is its closure minus its interior."),
    (
        "OpenHull",
        IncidenceSpace.open_hull,
        "OpenHull(region) adds the star of each vertex: the smallest open region containing it.",
    ),
):
    builtin(_name, _usage)(_region_op(_name, _op, _name.lower()))


@builtin("OpenQ", "OpenQ(region) is true when the region holds the star of each of its vertices.")
def _open_q(r):
    r = _region(r, "OpenQ")
    return IncidenceSpace(r.space).is_open(r.region)


@builtin(
    "ClosedQ", "ClosedQ(region) is true when the region holds the vertices of each of its edges: a sub-hypergraph."
)
def _closed_q(r):
    r = _region(r, "ClosedQ")
    return IncidenceSpace(r.space).is_closed(r.region)


@builtin(
    "ConnectedQ",
    "ConnectedQ(region) is true when the region is connected in the subspace topology; ConnectedQ(h) for a space.",
)
def _connected_q(x):
    if isinstance(x, RegionValue):
        return IncidenceSpace(x.space).is_connected(x.region)
    return len(_space(x, "ConnectedQ").components()) <= 1


@builtin(
    "Subspace",
    "Subspace(region) is the region with the subspace topology, as a spatial hypergraph: its vertices, with each of "
    "its "
    "edges cut down to them.",
)
def _subspace(r):
    r = _region(r, "Subspace")
    return IncidenceSpace(r.space).subspace(r.region)


@builtin(
    "FirstBetti",
    "FirstBetti(h) is the number of independent loops of X(h): incidences − vertices − edges + components.",
)
def _first_betti(h):
    return IncidenceSpace(_space(h, "FirstBetti")).first_betti()


# -- tangent vectors and motions ------------------------------------------------------------


@builtin(
    "Departure",
    "Departure(v, edge) is the tangent vector v ⇀ edge at v: the infinitesimal motion from v into the edge.",
)
def _departure(v, e):
    return TangentVector(v, _edge(e, "Departure"))


@builtin(
    "TangentSpace",
    "TangentSpace(h, v) lists the tangent vectors at v, one departure v ⇀ e per edge e through v; drawn.",
)
def _tangent_space(h, v):
    s = _space(h, "TangentSpace")
    return Tangents(s, _vertex(s, v, "TangentSpace"))


@builtin("TangentBundle", "TangentBundle(h) lists every tangent vector of h: one per incidence.")
def _tangent_bundle(h):
    return tangent_bundle(_space(h, "TangentBundle"))


@builtin("StationaryQ", "StationaryQ(t) is true for a tangent vector v ⇀ {v}: the hypergraphic zero.")
def _stationary_q(t):
    return _tangent(t, "StationaryQ").stationary


@builtin(
    "MotionGraph", "MotionGraph(h) is the graph of all half-steps: v ⇀ e and e ⇀ v for each vertex v of each edge e."
)
def _motion_graph(h):
    return MotionGraphValue(_space(h, "MotionGraph"))


@builtin(
    "Motion",
    "Motion(h, [x0, x1, ...]) is a motion through points of X(h), alternating vertices and edges (edges as lists), "
    "each "
    "incident to the next: Motion(h, [1, [1, 2], 2]).",
)
def _make_motion(h, pts):
    s = _space(h, "Motion")
    if not isinstance(pts, list):
        raise HyperiadError("Motion takes a list of points")
    return Motion(s, [_point(p) for p in pts])


@builtin("MotionPoints", "MotionPoints(m) lists the points a motion passes through.")
def _motion_points(m):
    return list(_motion(m, "MotionPoints").points)


@builtin("HalfSteps", "HalfSteps(m) is the number of half-steps of a motion (twice its steps, between vertices).")
def _half_steps(m):
    return _motion(m, "HalfSteps").half_steps()


@builtin(
    "Velocity",
    "Velocity(m, i) is the tangent vector by which the motion leaves its i-th point (a vertex; counted from 1).",
)
def _velocity(m, i):
    m = _motion(m, "Velocity")
    t = m.velocity(_int(i, "Velocity") - 1)
    if t is None:
        raise HyperiadError("Velocity: the i-th point must be a vertex the motion leaves")
    return t


@builtin(
    "IncomingDirection",
    "IncomingDirection(m, i) is the tangent vector back into the edge the motion arrived from at its i-th point.",
)
def _incoming(m, i):
    m = _motion(m, "IncomingDirection")
    t = m.incoming(_int(i, "IncomingDirection") - 1)
    if t is None:
        raise HyperiadError("IncomingDirection: the i-th point must be a vertex the motion arrives at")
    return t


@builtin("Geodesic", "Geodesic(h, x, y) is a motion from x to y with fewest half-steps (null if there is none).")
def _geodesic(h, x, y):
    s = _space(h, "Geodesic")
    return geodesic(s, _check_point(s, _point(x), "Geodesic"), _check_point(s, _point(y), "Geodesic"))


@builtin("JoinMotions", "JoinMotions(m1, m2) is m1 followed by m2 (m2 starts where m1 ends).")
def _join(a, b):
    return _motion(a, "JoinMotions").then(_motion(b, "JoinMotions"))


@builtin("ReverseMotion", "ReverseMotion(m) runs a motion backwards.")
def _reverse(m):
    return _motion(m, "ReverseMotion").reverse()


@builtin(
    "ReduceMotion", "ReduceMotion(m) is the normal form of m under the rewrite x y x → x (cancel a step straight back)."
)
def _reduce(m):
    return reduce_motion(_motion(m, "ReduceMotion"))


@builtin(
    "ReductionSteps", "ReductionSteps(m) lists m and each motion after cancelling the leftmost backtrack x y x → x."
)
def _reduction_steps(m):
    return reduction_steps(_motion(m, "ReductionSteps"))


@builtin(
    "HomotopicQ",
    "HomotopicQ(m1, m2) is true when the step paths of the motions are area-homotopic (a derivation is found), false "
    'when their ends differ or a barrier separates them, and "undecided" otherwise.',
)
def _homotopic(a, b):
    from infrageometry.paths import StepPath

    def arg(x):
        return x if isinstance(x, StepPath) else _motion(x, "HomotopicQ")

    out = homotopic(arg(a), arg(b))
    return "undecided" if out is None else out


@builtin(
    "IncidenceHomotopicQ",
    "IncidenceHomotopicQ(m1, m2) is true when the motions have the same ends and the same reduced form under "
    "x y x → x: homotopy in the incidence space, where hyperedges are not filled.",
)
def _incidence_homotopic(a, b):
    return incidence_homotopic(_motion(a, "IncidenceHomotopicQ"), _motion(b, "IncidenceHomotopicQ"))


@builtin(
    "IncidenceGroupGenerators",
    "IncidenceGroupGenerators(h, v) lists reduced closed motions at v that freely generate π1 of the incidence space.",
)
def _pi1(h, v):
    s = _space(h, "IncidenceGroupGenerators")
    return incidence_group_generators(s, _vertex(s, v, "IncidenceGroupGenerators"))


@builtin("Interval", "Interval(n) is the path on 0, 1, ..., n: the domain of a curve with n steps.")
def _interval(n):
    return interval(_int(n, "Interval"))


@builtin("CurveOfMotion", "CurveOfMotion(m) is the map Interval(n) → h of a motion with n steps between vertices.")
def _curve(m):
    return curve(_motion(m, "CurveOfMotion"))


@builtin("MotionOfCurve", "MotionOfCurve(c) is the motion of a map from an interval.")
def _motion_of_curve(c):
    return motion_of_curve(_map(c, "MotionOfCurve"))


@builtin(
    "TokenState",
    "TokenState(h, x) is h with a token at the point x, as a hypergraph: the state of TokenRewritingSystem.",
)
def _token_state(h, at):
    s = _space(h, "TokenState")
    return token_state(s, _check_point(s, _point(at), "TokenState"))


def _token_cost(h, at, **_opts):
    return 1e4 * len(_space(h, "TokenRewritingSystem").points())


@builtin(
    "TokenRewritingSystem",
    "TokenRewritingSystem(h, x) is the hypergraph rewriting system of a token at the point x of X(h): one departure "
    "and "
    'one arrival rule per edge size. Ask it sys("StatesGraph", n) as any rewriting system.',
    cost=_token_cost,
)
def _token_system(h, at, **options):
    s = _space(h, "TokenRewritingSystem")
    return token_rewriting_system(s, _check_point(s, _point(at), "TokenRewritingSystem"), **options)


def _motion_frames(m: Motion, interval: Any) -> Frames:
    svgs, labels = [], []
    for k in range(len(m.points)):
        svgs.append(_svg(pic.motion_scene(m, k)))
        x = m.points[k]
        labels.append(f"{k}: at {point_text(x)}" + (" (infinitesimal position)" if is_edge(x) else ""))
    return Frames(tuple(svgs), tuple(labels), describe_motion(m), True, int(interval))


@builtin(
    "AnimateMotion",
    "AnimateMotion(m) plays a motion half-step by half-step: a token goes vertex → edge → vertex. Interval = ms.",
)
def _animate_motion(m, Interval=800):  # noqa: N803
    return _motion_frames(_motion(m, "AnimateMotion"), Interval)


@builtin("AnimateReduction", "AnimateReduction(m) steps through the cancellations x y x → x that reduce a motion.")
def _animate_reduction(m, Interval=1100):  # noqa: N803
    steps = reduction_steps(_motion(m, "AnimateReduction"))
    svgs = tuple(_svg(pic.motion_scene(s)) for s in steps)
    labels = tuple(f"{k}: {_plural(s.half_steps(), 'half-step')}" for k, s in enumerate(steps))
    return Frames(svgs, labels, "cancelling backtracks x y x → x", True, int(Interval))


# -- maps -----------------------------------------------------------------------------------


@builtin(
    "SpatialMap",
    """SpatialMap(h1, h2, vertexmap, edgemap) is the map sending vertices and edges together, keeping incidence:
vertexmap is {v: w, ...}; edgemap is [[1, 2] -> [5, 6], ...] (or keyed by EdgePoint), and [[1, 2] -> 5]
collapses the edge onto the vertex 5 (all its vertices must go there). An edge left out goes to the edge
spanned by its image when that is an edge of h2, else to the one edge of h2 containing its image.""",
)
def _spatial_map(h1, h2, vmap, emap=None):
    a, b = _space(h1, "SpatialMap"), _space(h2, "SpatialMap")
    v = {_point(k): _point(x) for k, x in _pairs(vmap, "SpatialMap")}
    e = {_edge(k, "SpatialMap"): _point(x) for k, x in _pairs(emap, "SpatialMap")}  # a vertex: a collapse
    return SpatialMap(a, b, v, e)


@builtin("IdentityMap", "IdentityMap(h) is the identity map of h.")
def _identity(h):
    return identity_map(_space(h, "IdentityMap"))


@builtin("ComposeMaps", "ComposeMaps(f, g) is g after f (first f, then g).")
def _compose(f, g):
    return _map(f, "ComposeMaps").then(_map(g, "ComposeMaps"))


@builtin("MapApply", "MapApply(f, x) is the image of a point (vertex or edge) under the map f.")
def _map_apply(f, x):
    return _map(f, "MapApply")(_point(x))


@builtin(
    "Pushforward",
    "Pushforward(f, t) is f_* t: the tangent vector f(v) ⇀ f(e) for t = v ⇀ e, or the point f(v) when e collapses.",
)
def _pushforward(f, t):
    return _map(f, "Pushforward").pushforward(_tangent(t, "Pushforward"))


@builtin(
    "Differential",
    "Differential(f, v) lists the tangent vectors at v with their images, t -> f_* t; for a scalar function, "
    "t -> ↑, ○ or ↓.",
)
def _differential(f, v):
    if isinstance(f, ScalarFunction):
        return [RuleValue(t, s) for t, s in f.differential(_vertex(f.space, v, "Differential")).items()]
    f = _map(f, "Differential")
    return [RuleValue(t, x) for t, x in f.differential(_vertex(f.source, v, "Differential")).items()]


@builtin("MapPlot", "MapPlot(f, v) draws the map f with the tangent vectors at v and their pushforwards in blue.")
def _map_plot(f, v):
    from hyperiad.values import Graphic

    f = _map(f, "MapPlot")
    return Graphic(_svg(pic.map_scene(f, highlight=_vertex(f.source, v, "MapPlot"))), f"f_* at {point_text(v)}")


for _name, _attr, _usage in (
    ("ImmersionQ", "is_immersion", "ImmersionQ(f) is true when f_* is injective at every point."),
    (
        "SubmersionQ",
        "is_submersion",
        "SubmersionQ(f) is true when f_* is onto the tangent space of the image at every point.",
    ),
    (
        "LocalIsomorphismQ",
        "is_local_isomorphism",
        "LocalIsomorphismQ(f) is true when f_* is a bijection at every point.",
    ),
    (
        "EmbeddingQ",
        "is_embedding",
        "EmbeddingQ(f) is true when f is injective and v lies in e whenever f(v) lies in f(e).",
    ),
    (
        "IsomorphismQ",
        "is_isomorphism",
        "IsomorphismQ(f) is true when f is a bijection of points keeping incidence both ways.",
    ),
):
    builtin(_name, _usage)((lambda attr, name: lambda f: getattr(_map(f, name), attr)())(_attr, _name))


@builtin("MapImage", "MapImage(f) is the image of f, a region of its target.")
def _map_image(f):
    f = _map(f, "MapImage")
    return RegionValue(f.target, f.image(), "image")


@builtin("PushMotion", "PushMotion(f, m) is the motion f(m) in the target.")
def _push_motion(f, m):
    return _map(f, "PushMotion").push_motion(_motion(m, "PushMotion"))


def _all_maps_cost(h1, h2):
    a, b = _space(h1, "AllMaps"), _space(h2, "AllMaps")
    return float(max(len(b.vertices), 1)) ** len(a.vertices) * max(len(b.edges), 1) ** len(a.edges)


@builtin("AllMaps", "AllMaps(h1, h2) lists every map h1 → h2 (small spaces only).", cost=_all_maps_cost)
def _all_maps(h1, h2):
    return list(all_maps(_space(h1, "AllMaps"), _space(h2, "AllMaps")))


# -- scalar functions -----------------------------------------------------------------------


@builtin(
    "ScalarLine", "ScalarLine(a, b) is the window a..b of the scalar line: the path with an edge {n} at each point."
)
def _scalar_line(a, b):
    return scalar_line(_int(a, "ScalarLine"), _int(b, "ScalarLine"))


@builtin(
    "ScalarFunction",
    """ScalarFunction(h, values) is the map h → the scalar line with values {v: n, ...} at the vertices; values on
one edge lie within one step. EdgeValues = [[1, 2] -> [5, 6], ...] sets an edge's value where the values of its
vertices leave a choice (an edge on which f is n may go to [n], [n - 1, n] or [n, n + 1]).""",
)
def _scalar_function(h, values, EdgeValues=None):  # noqa: N803
    s = _space(h, "ScalarFunction")
    vals = (
        dict(_pairs(values, "ScalarFunction"))
        if not isinstance(values, list)
        else dict(zip(s.vertex_list(), values, strict=True))
    )
    ev = {_edge(k, "ScalarFunction"): v for k, v in _pairs(EdgeValues, "ScalarFunction")}
    return ScalarFunction(s, vals, ev)


@builtin("DistanceFunction", "DistanceFunction(h, v) is the scalar function w ↦ HopDistance(h, v, w).")
def _distance_function(h, v):
    s = _space(h, "DistanceFunction")
    return distance_function(s, _vertex(s, v, "DistanceFunction"))


@builtin("Derivative", "Derivative(f, t) is the derivative of a scalar function along a tangent vector: ↑, ○ or ↓.")
def _derivative(f, t):
    return _scalar(f, "Derivative").derivative(_tangent(t, "Derivative"))


@builtin("PointKind", 'PointKind(f, v) is "regular", "maximum", "minimum" or "flat" for a scalar function f.')
def _point_kind(f, v):
    f = _scalar(f, "PointKind")
    return f.kind(_vertex(f.space, v, "PointKind"))


@builtin("CriticalPoints", "CriticalPoints(f) is {v: kind} for the points of f that are not regular.")
def _critical(f):
    return _scalar(f, "CriticalPoints").critical_points()


@builtin("LevelSet", "LevelSet(f, n) is f⁻¹(n): the vertices at n and the edges sent to {n} (a closed region).")
def _level_set(f, n):
    f = _scalar(f, "LevelSet")
    return RegionValue(f.space, f.level_set(_int(n, "LevelSet")), f"f⁻¹({n})")


@builtin(
    "Crossing",
    "Crossing(f, n) is f⁻¹({n, n+1}): the edges sent to the infinitesimal position between n and n + 1 (open).",
)
def _crossing(f, n):
    f = _scalar(f, "Crossing")
    return RegionValue(f.space, f.crossing(_int(n, "Crossing")), f"f⁻¹({{{n}, {n + 1}}})")


@builtin("Sublevel", "Sublevel(f, n) is the region where f is at most n.")
def _sublevel(f, n):
    f = _scalar(f, "Sublevel")
    return RegionValue(f.space, f.sublevel(_int(n, "Sublevel")), f"f ≤ {n}")


@builtin(
    "ScalarPlot", "ScalarPlot(f, v) draws f with the derivative at v in blue; ScalarPlot(f, region) marks a region."
)
def _scalar_plot(f, x=None):
    from hyperiad.values import Graphic

    f = _scalar(f, "ScalarPlot")
    if isinstance(x, RegionValue):
        return Graphic(_svg(pic.scalar_scene(f, region=x.region)), x.title)
    if x is None:
        return Graphic(_svg(pic.scalar_scene(f)), "")
    v = _vertex(f.space, x, "ScalarPlot")
    signs = ", ".join(f"{t!r}: {s}" for t, s in f.differential(v).items())
    return Graphic(_svg(pic.scalar_scene(f, highlight=v)), f"df at {point_text(v)}: {signs}")


# -- vector fields and flows ----------------------------------------------------------------


@builtin("VectorField", "VectorField(h, {v: edge, ...}) chooses a tangent vector v ⇀ edge at some points.")
def _vector_field(h, choice):
    s = _space(h, "VectorField")
    return VectorField(s, {_point(k): _edge(e, "VectorField") for k, e in _pairs(choice, "VectorField")})


@builtin("ExitField", "ExitField(h, [[1, 2] -> 2, ...]) leaves each edge by one vertex, whatever vertex entered it.")
def _exit_field(h, exits):
    s = _space(h, "ExitField")
    return Transit.exit_field(s, {_edge(e, "ExitField"): w for e, w in _pairs(exits, "ExitField")})


@builtin(
    "Transit",
    "Transit(h, [[1, 2, 3] -> {1: 3, 2: 1}, ...]) says, for each edge, by which vertex to leave given the entry.",
)
def _transit(h, rule):
    s = _space(h, "Transit")
    return Transit(s, {_edge(e, "Transit"): dict(_pairs(m, "Transit")) for e, m in _pairs(rule, "Transit")})


@builtin("Flow", "Flow(field, transit) is the flow: one step v → X(v) → leave X(v) as the transit says.")
def _flow(x, t):
    if not isinstance(x, VectorField) or not isinstance(t, Transit):
        raise HyperiadError("Flow takes a vector field and a transit (or exit field)")
    return Flow(x, t)


@builtin("FlowMap", "FlowMap(flow) is {v: φ(v)}: where one step of the flow takes each point.")
def _flow_map(flow):
    if not isinstance(flow, Flow):
        raise HyperiadError("FlowMap takes a flow")
    return flow.map()


@builtin("IntegralCurve", "IntegralCurve(flow, v) follows the flow from v until it ends, rests or closes up.")
def _integral_curve(flow, v):
    if not isinstance(flow, Flow):
        raise HyperiadError("IntegralCurve takes a flow")
    return flow.integral_curve(_vertex(flow.space, v, "IntegralCurve"))


@builtin(
    "GradientFlow",
    "GradientFlow(f) climbs a scalar function f: depart upward, leave at the highest vertex. "
    "Ascending = false descends.",
)
def _gradient_flow(f, Ascending=True):  # noqa: N803
    return gradient_flow(_scalar(f, "GradientFlow"), ascending=bool(Ascending))


@builtin(
    "AnimateFlow",
    "AnimateFlow(flow, n) moves a token from every point along the flow for n steps, half-step by half-step.",
)
def _animate_flow(flow, n=4, Interval=800):  # noqa: N803
    if not isinstance(flow, Flow):
        raise HyperiadError("AnimateFlow takes a flow")
    state = {v: (v, None) for v in flow.space.vertex_list()}  # each token: where it is, where it will exit
    arrows = _flow_arrows(flow)
    svgs, labels = [], []
    for k in range(2 * _int(n, "AnimateFlow") + 1):
        tokens = sorted({p for p, _ in state.values()}, key=lambda p: (is_edge(p), str(p)))
        svgs.append(_svg(pic.space_scene(flow.space, faint_arrows=arrows, tokens=tokens)))
        labels.append(f"half-step {k}: tokens at {_plural(len(tokens), 'point')}")
        for v, (p, exit_to) in state.items():
            if is_edge(p):
                state[v] = (exit_to, None)
            elif (step := flow.step(p)) is not None and step[1] != p:
                state[v] = (step[0], step[1])
    return Frames(tuple(svgs), tuple(labels), "every point carried by the flow", True, int(Interval))


# -- local models -------------------------------------------------------------------------


@builtin(
    "PointedBall", "PointedBall(h, v, r) is the ball of radius r about v with the edges inside it, centre v: a region."
)
def _pointed_ball(h, v, r):
    s = _space(h, "PointedBall")
    ball = s.induced(s.ball(_vertex(s, v, "PointedBall"), _int(r, "PointedBall")))
    return RegionValue(s, Region(ball.vertices, ball.edges), f"(B_{r}({point_text(v)}), {point_text(v)})")


@builtin(
    "LocallyModelledQ",
    "LocallyModelledQ(h, model, centres, r) is true when every pointed r-ball of h is isomorphic to the pointed r-ball "
    "of model about one of the centres.",
)
def _locally_modelled(h, model, centres, r):
    return locally_modelled(
        _space(h, "LocallyModelledQ"), _space(model, "LocallyModelledQ"), list(centres), _int(r, "LocallyModelledQ")
    )


@builtin(
    "Chart",
    "Chart(h, v, model, centres, r) is an isomorphism of pointed r-balls from h at v to model at a centre "
    "(null if none).",
)
def _chart(h, v, model, centres, r):
    s = _space(h, "Chart")
    return make_chart(s, _vertex(s, v, "Chart"), _space(model, "Chart"), list(centres), _int(r, "Chart"))


@builtin(
    "CombinatorialSurfaceQ", "CombinatorialSurfaceQ(h) is true when every edge is a triangle and every link a cycle."
)
def _surface_q(h):
    return is_combinatorial_surface(_space(h, "CombinatorialSurfaceQ"))


@builtin("BallGrowth", "BallGrowth(h, v, r) lists the sizes of the balls of radius 0, 1, ..., r about v.")
def _ball_growth(h, v, r):
    s = _space(h, "BallGrowth")
    return ball_growth(s, _vertex(s, v, "BallGrowth"), _int(r, "BallGrowth"))


@builtin(
    "GrowthDimension",
    "GrowthDimension(h, v, r) is a device (real-valued): the slope of log |B_r| against log r at scale r.",
)
def _growth_dimension(h, v, r):
    s = _space(h, "GrowthDimension")
    return growth_dimension(s, _vertex(s, v, "GrowthDimension"), _int(r, "GrowthDimension"))


# -- example spaces -----------------------------------------------------------------------


@builtin("PathSpace", "PathSpace(n) is the path on 0, 1, ..., n.")
def _path(n):
    return path(_int(n, "PathSpace"))


@builtin("CycleSpace", "CycleSpace(n) is the cycle on 0, ..., n − 1.")
def _cycle(n):
    return cycle(_int(n, "CycleSpace"))


@builtin("GridSpace", "GridSpace(m, n) is an m × n piece of the square grid, vertices numbered 1..mn row by row.")
def _grid(m, n):
    return _numbered(grid(_int(m, "GridSpace"), _int(n, "GridSpace")))


@builtin("TorusGrid", "TorusGrid(m, n) is the m × n grid closed up into a torus, vertices numbered row by row.")
def _torus_grid(m, n):
    return _numbered(torus_grid(_int(m, "TorusGrid"), _int(n, "TorusGrid")))


@builtin("KleinGrid", "KleinGrid(m, n) is the m × n grid closed up into a Klein bottle (one seam reflected).")
def _klein_grid(m, n):
    return _numbered(klein_grid(_int(m, "KleinGrid"), _int(n, "KleinGrid")))


@builtin("Octahedron", "Octahedron() is the octahedron as 8 triangles: a combinatorial sphere.")
def _octahedron():
    return octahedron()


@builtin("Torus7", "Torus7() is the 7-point torus: 14 triangles, every two points adjacent.")
def _torus7():
    return torus7()


@builtin("WithUnaryEdges", "WithUnaryEdges(h) adds the edge {v} at every vertex v, so every point can rest.")
def _with_unary(h):
    return with_unary(_space(h, "WithUnaryEdges"))


@builtin("ExampleSpace", "ExampleSpace() is the running example: edges {1,2,3}, {1,2}, {3,4}, {4,5,6}, {2,6}.")
def _example():
    return triangle_example()
