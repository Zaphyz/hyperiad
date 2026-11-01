r"""Scenes of the infrageometry objects: what is drawn, before a renderer paints it.

Everything is drawn on the shared hypergraph geometry (:func:`frontier.hypergraphs.geometry.hypergraph_scene`)
in the reference paclet's style, and both renderers paint these scenes (``scene_svg`` for Hyperiad
and web pages, ``frontier.viz.hypergraph.paint`` for figures). Colours have three roles
(the user, 2026-10-29):

* the **space** is the neutral blue drawing;
* what a picture **points at** (a region, a match, a moving point) is the desaturated amber
  :data:`~frontier.hypergraphs.geometry.HIGHLIGHT`;
* **objects drawn on the space** (motions, paths, tangent vectors, vector fields, the edges of a
  field) take the desaturated tones of :data:`~frontier.hypergraphs.geometry.FIELD_COLORS`:
  motions and paths green, tangent vectors and pushforwards blue, flows brown, and further
  fields red and violet.

An **infinitesimal position** (an edge as a point of $X(H)$) is a four-pointed star at the edge's
centre (for a unary edge, just above its vertex), told apart from the round vertices. A half-step
of a motion is an arrow between a vertex and such a star. The arcs of a vector field (departures
and arrivals) run between vertices and stars, its integrated steps and composites from vertex to
vertex, as curved arrows in the field's colour, bowing to their left so that opposite arcs part. The
unary edges of a scalar field are concentric disks at their vertex, one per copy.
The incidence space is drawn as its Hasse diagram: vertices below, edges (stars) above.

A space keeps one layout for every picture of it (fixed seed), so that frames of an animation and
pictures side by side line up.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Iterable, Mapping, Sequence
from functools import lru_cache

import numpy as np
from frontier.hypergraphs.geometry import (
    EDGE_LINE,
    HIGHLIGHT,
    HIGHLIGHT_RIM,
    RGBA,
    Area,
    Arrowhead,
    Curve,
    Dots,
    Label,
    Scene,
    field_color,
    graph_scene,
    hypergraph_layout,
    hypergraph_scene,
    scene_row,
    star_points,
)

from infrageometry.space import Region, SpatialHypergraph, edge_key, is_edge, point_text, sort_points

Point = Hashable
STAR = 4.2  # outer radius of an edge-point star, in points
TOKEN_RING = 7.5
FAINT = (*EDGE_LINE[:3], 0.55)
INK = (0.25, 0.25, 0.3, 1.0)
MOTION = field_color("green")  # motions and paths
TANGENT = field_color("blue")  # tangent vectors, derivatives, pushforwards
FLOW = field_color("brown")  # flows and vector fields drawn by their steps


def _pts_to_units(scene: Scene, pts: float) -> float:
    return pts * scene.span / 160.0


@lru_cache(maxsize=256)
def _cached_layout(space: SpatialHypergraph, seed: int) -> tuple:
    # unary edges do not pull on the layout, so a space keeps its layout when they are added
    edges = [e for e in space.drawing_edges() if len(e) > 1]
    pos = hypergraph_layout(edges, seed=seed, vertices=space.vertex_list())
    return tuple((v, tuple(p)) for v, p in pos.items())


def _crossings(space: SpatialHypergraph, pos: Mapping) -> int:
    """How many pairs of binary edges (straight segments) cross in a layout."""

    def side(p, q, r) -> float:
        return float(np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])))

    segs = [tuple(e) for e in space.edges if len(e) == 2]
    count = 0
    for i, (a, b) in enumerate(segs):
        for c, d in segs[i + 1 :]:
            if len({a, b, c, d}) == 4:
                pa, pb, pc, pd = (np.asarray(pos[x]) for x in (a, b, c, d))
                if side(pa, pb, pc) != side(pa, pb, pd) and side(pc, pd, pa) != side(pc, pd, pb):
                    count += 1
    return count


def layout_of(space: SpatialHypergraph, seed: int = 0) -> dict:
    """The positions of the vertices, the same for every picture of the space.

    For a small space a few seeds are tried and the first layout whose straight edges do not
    cross is kept (spring layouts occasionally fold a cycle into a bow-tie).
    """
    if space.coordinates:  # the space says where its vertices go
        return {v: np.asarray(p, dtype=float) for v, p in space.coordinates.items()}
    best = None
    tries = range(seed, seed + 5) if len(space.edges) <= 80 else [seed]
    for s in tries:
        pos = {v: np.asarray(p, dtype=float) for v, p in _cached_layout(space, s)}
        n = _crossings(space, pos) if len(tries) > 1 else 0
        if best is None or n < best[0]:
            best = (n, pos)
        if n == 0:
            break
    return best[1]


def point_position(scene: Scene, layout: Mapping, x: Point) -> np.ndarray:
    """Where a point of $X(H)$ is drawn: a vertex at its place, an edge at its centre."""
    if not is_edge(x):
        return np.asarray(layout[x], dtype=float)
    pts = np.array([layout[v] for v in x], dtype=float)
    if len(x) == 1:
        return pts[0] + np.array([0.0, _pts_to_units(scene, 9.0)])
    return pts.mean(axis=0)


def _star(scene: Scene, c: np.ndarray, color: RGBA, size: float = STAR, z: float = 3.5) -> Area:
    """The mark of an infinitesimal position: a four-pointed star."""
    rim = color[:3] + (1.0,) if color is not HIGHLIGHT else HIGHLIGHT_RIM
    return Area(star_points(c, _pts_to_units(scene, size)), color, rim, 0.6, z=z)


def _arrow(scene: Scene, a: np.ndarray, b: np.ndarray, color: RGBA, start=4.0, end=4.5, width=1.8, z=3.2):
    u = b - a
    n = float(np.linalg.norm(u))
    if n < 1e-9:
        return
    d = u / n
    tail = a + d * _pts_to_units(scene, start)
    tip = b - d * _pts_to_units(scene, end)
    if float(np.dot(tip - tail, d)) <= 0:
        return
    scene.shapes.append(Curve(np.array([tail, tip]), color, width, z=z))
    scene.shapes.append(Arrowhead(tip - 0.25 * (tip - tail), tip, 9.0 * width / 1.8, color, z=z + 0.05))


def _arc(
    scene: Scene,
    a: np.ndarray,
    b: np.ndarray,
    color: RGBA,
    bend: float = 0.18,
    width: float = 1.7,
    z: float = 3.3,
    trim_a: float = 4.5,
    trim_b: float = 4.5,
) -> None:
    """A directed field edge from ``a`` to ``b``: a curve bowing to its left, arrowhead at the end, trimmed
    ``trim_a`` and ``trim_b`` points short of its ends.

    Equal ends make a small loop (a stationary edge).
    """
    ta, tb = _pts_to_units(scene, trim_a), _pts_to_units(scene, trim_b)
    u = b - a
    n = float(np.linalg.norm(u))
    if n < 1e-9:  # a loop above the vertex
        r = _pts_to_units(scene, 8.0)
        t = np.linspace(0.15 * np.pi, 1.85 * np.pi, 40) - np.pi / 2
        centre = a + np.array([0.0, r * 1.25])
        loop = centre + r * np.column_stack([np.cos(t), np.sin(t)])
        scene.shapes.append(Curve(loop, color, width, z=z))
        scene.shapes.append(Arrowhead(loop[-4], loop[-1], 8.0 * width / 1.7, color, z=z + 0.05))
        return
    left = np.array([-u[1], u[0]]) / n
    mid = (a + b) / 2 + bend * n * left
    t = np.linspace(0.0, 1.0, 30)[:, None]
    curve = (1 - t) ** 2 * a + 2 * (1 - t) * t * mid + t**2 * b
    keep = (np.linalg.norm(curve - a, axis=1) > ta) & (np.linalg.norm(curve - b, axis=1) > tb)
    curve = curve[keep]
    if len(curve) < 3:
        return
    scene.shapes.append(Curve(curve, color, width, z=z, round_cap=True))
    scene.shapes.append(Arrowhead(curve[-3], curve[-1], 9.0 * width / 1.7, color, z=z + 0.05))


def space_scene(
    space: SpatialHypergraph,
    *,
    region: Region | None = None,
    red_edges: Iterable[frozenset] = (),
    red_vertices: Iterable[Point] = (),
    edge_points: str = "red",
    arrows: Sequence[tuple[Point, Point]] = (),
    arrow_color: RGBA = MOTION,
    faint_arrows: Sequence[tuple[Point, Point]] = (),
    token: Point | None = None,
    tokens: Sequence[Point] = (),
    fields: Sequence[tuple[Sequence[tuple[Point, Point]], RGBA]] = (),
    labels: bool = True,
    vertex_text: Mapping[Point, str] | None = None,
    title: str = "",
    seed: int = 0,
    layout: Mapping | None = None,
) -> Scene:
    """A spatial hypergraph with things pointed at highlighted and objects drawn on it.

    ``region`` (or ``red_vertices`` / ``red_edges``, the names kept from the first version) is
    highlighted. ``edge_points`` shows infinitesimal positions as stars: ``"all"``, ``"red"`` (the
    highlighted edges that dangle: in the region without all their vertices) or ``"none"``.
    ``arrows`` are half-steps between points of $X(H)$ in ``arrow_color``; ``faint_arrows`` faint
    ones. ``token`` (or several, ``tokens``) marks a moving point. ``fields`` are layers of
    vertex-to-vertex edges, each a list of ``(a, b)`` with its colour.
    """
    layout = layout or layout_of(space, seed)
    order = space.edge_list()
    reds = set(red_edges) | (set(region.edges) if region else set())
    hot_vertices = set(red_vertices) | (set(region.vertices) if region else set())
    scene = hypergraph_scene(
        space.drawing_edges(),
        layout=layout,
        labels=labels and vertex_text is None,
        vertex_text=vertex_text,
        title=title,
        highlight=[i for i, e in enumerate(order) if e in reds],
        highlight_vertices=[v for v in space.vertex_list() if v in hot_vertices],
        vertices=space.vertex_list(),
    )
    pts = np.array([layout[v] for v in space.vertex_list()]) if space.vertices else np.zeros((1, 2))
    scene.shapes.append(Dots(pts, TOKEN_RING + 2.4, (1.0, 1.0, 1.0, 0.0), None, 0.0, z=0.0))  # steady extent
    if edge_points == "all":
        shown = order
    elif edge_points == "red":  # an edge's star says it is in without all its vertices: it dangles
        shown = [e for e in order if e in reds and not e <= hot_vertices]
    else:
        shown = []
    for e in shown:
        scene.shapes.append(_star(scene, point_position(scene, layout, e), HIGHLIGHT if e in reds else FAINT))
    for a, b in faint_arrows:
        _arrow(scene, point_position(scene, layout, a), point_position(scene, layout, b), FAINT, width=1.2, z=2.6)
    for a, b in arrows:
        for x in (a, b):
            if is_edge(x) and x not in shown:
                scene.shapes.append(_star(scene, point_position(scene, layout, x), arrow_color))
        _arrow(scene, point_position(scene, layout, a), point_position(scene, layout, b), arrow_color)
    for k, (edges, color) in enumerate(fields):
        draw_field_edges(scene, layout, edges, color, layer=k)
    for t in ([token] if token is not None else []) + list(tokens):
        at = point_position(scene, layout, t)
        if is_edge(t):
            scene.shapes.append(_star(scene, at, HIGHLIGHT, size=6.5, z=4.2))
        else:
            scene.shapes.append(Dots(at[None, :], TOKEN_RING, None, HIGHLIGHT, 2.6, z=4.2))
    return scene


def draw_field_edges(scene: Scene, layout: Mapping, edges: Sequence[tuple[Point, Point]], color: RGBA, layer=0):
    """Edges of one field between points of $X(H)$ as curved arrows; repeats of one pair bow further out.

    A departure or an arrival runs between a vertex and the star of an edge; the star is drawn in the
    field's colour. Edges between two vertices (steps, composites) run from vertex to vertex.
    """

    def at(x):  # a unary edge's star stands higher than in motions, so that a stationary step shows
        if is_edge(x) and len(x) == 1:
            (v,) = x
            return np.asarray(layout[v], float) + np.array([0.0, _pts_to_units(scene, 17.0)])
        return point_position(scene, layout, x)

    seen: Counter = Counter()
    stars = set()
    for a, b in edges:
        seen[(a, b)] += 1
        stay = any(is_edge(x) and len(x) == 1 for x in (a, b))
        half = is_edge(a) or is_edge(b)
        bend = (0.5 if stay else 0.22 if half else 0.16) + 0.07 * layer + 0.12 * (seen[(a, b)] - 1)
        ta = STAR + 1.0 if is_edge(a) else 4.5
        tb = STAR + 1.5 if is_edge(b) else 4.5
        _arc(scene, at(a), at(b), color, bend=bend, trim_a=ta, trim_b=tb)
        stars |= {x for x in (a, b) if is_edge(x)}
    for e in sort_points(stars):
        scene.shapes.append(_star(scene, at(e), color, z=3.4))


QUANTUM = 5.6  # radius of the innermost unary edge of a scalar field, in points
QUANTUM_STEP = 2.6


def draw_quanta(scene: Scene, layout: Mapping, counts: Mapping[Point, int], color: RGBA) -> None:
    """The unary edges of a scalar field: at a vertex with multiplicity $n$, $n$ concentric disks."""
    fill = (*color[:3], 0.16)
    for v, n in counts.items():
        at = np.asarray(layout[v], float)[None, :]
        for k in range(n - 1, -1, -1):
            r = QUANTUM + QUANTUM_STEP * k
            scene.shapes.append(Dots(at, r, fill, (*color[:3], 0.85), 0.8, z=2.6 + 0.01 * (n - k)))


def incidence_scene(
    space: SpatialHypergraph,
    *,
    region: Region | None = None,
    arcs: Sequence[tuple[Point, Point]] = (),
    title: str = "",
    seed: int = 0,
) -> Scene:
    """The Hasse diagram of $X(H)$: positions below, infinitesimal positions (stars) above.

    Points of ``region`` are highlighted; ``arcs`` (half-steps of a motion) are green arrows.
    """
    layout = layout_of(space, seed)
    vs = sorted(space.vertex_list(), key=lambda v: (float(layout[v][0]), str(v)))
    xs = {v: float(i) for i, v in enumerate(vs)}
    es = sorted(space.edge_list(), key=lambda e: (float(np.mean([xs[v] for v in e])), edge_key(e)))
    width = max(len(vs) - 1, 1)
    step = width / max(len(es) - 1, 1) if len(es) > 1 else 0.0
    off = 0.0 if len(es) > 1 else width / 2
    height = max(1.4, 0.32 * width)
    pos: dict = {("v", v): (xs[v], 0.0) for v in vs}
    pos |= {("e", e): (off + i * step, height) for i, e in enumerate(es)}
    nodes = list(pos)
    links = [(("v", v), ("e", e)) for e in es for v in sort_points(e)]
    inside = set()
    if region is not None:
        inside = {("v", v) for v in region.vertices} | {("e", e) for e in region.edges}
    blue = EDGE_LINE[:3] + (1.0,)
    style = {n: ("disk" if n[0] == "v" else "star", HIGHLIGHT if n in inside else blue) for n in nodes}
    colours = {(a, b): HIGHLIGHT for a, b in links if a in inside and b in inside}
    scene = graph_scene(nodes, links, pos, node_style=style, edge_colors=colours, directed=False, title=title)
    for n in nodes:
        text = point_text(n[1])
        scene.shapes.append(Label(np.asarray(pos[n]), (0.0, -12.0 if n[0] == "v" else 13.0), text, size=8.0))
    for a, b in arcs:
        pa = np.asarray(pos[("e", a) if is_edge(a) else ("v", a)])
        pb = np.asarray(pos[("e", b) if is_edge(b) else ("v", b)])
        _graph_arrow(scene, pa, pb)
    return scene


def _graph_arrow(scene: Scene, a: np.ndarray, b: np.ndarray, color: RGBA = MOTION) -> None:
    unit = 1.0 / 52.0  # graph scenes: 52 points per layout unit
    u = b - a
    n = float(np.linalg.norm(u))
    d = u / n
    tail, tip = a + d * 7 * unit, b - d * 7 * unit
    scene.shapes.append(Curve(np.array([tail, tip]), color, 2.0, z=0.8))
    scene.shapes.append(Arrowhead(tip - 0.3 * (tip - tail), tip, 10.0, color, z=0.9))


def motion_scene(motion, upto: int | None = None, *, title: str = "", seed: int = 0, show_token: bool = False) -> Scene:
    """A motion as green half-step arrows; with ``upto``, only its first ``upto`` half-steps, and the
    token where it then is (the frames of an animation)."""
    space = motion.space
    pts = motion.points if upto is None else motion.points[: upto + 1]
    arrows = list(zip(pts, pts[1:], strict=False))
    rest = list(zip(motion.points[len(pts) - 1 :], motion.points[len(pts) :], strict=False)) if upto is not None else []
    ends = [motion.start] + ([motion.end] if upto is None else [])
    return space_scene(
        space,
        red_vertices=[x for x in ends if not is_edge(x)],
        arrows=arrows,
        faint_arrows=rest,
        token=pts[-1] if show_token or upto is not None else None,
        title=title,
        seed=seed,
    )


def tangent_scene(space: SpatialHypergraph, v: Point, *, title: str = "", seed: int = 0) -> Scene:
    """The tangent space at ``v``: its departures as blue arrows into the edges around it."""
    return space_scene(
        space, red_vertices=[v], arrows=[(v, e) for e in space.star(v)], arrow_color=TANGENT, title=title, seed=seed
    )


def map_scene(f, *, highlight: Point | None = None, title: str = "") -> Scene:
    """A map as its source beside its target. Each source vertex is labelled with its image; with
    ``highlight`` a vertex, its tangent vectors and their pushforwards are drawn in blue."""
    src = f.source
    text = {v: f"{point_text(v)}↦{point_text(f.vertex_map[v])}" for v in src.vertex_list()}
    left_kw: dict = {"vertex_text": text}
    right_kw: dict = {}
    if highlight is not None:
        left_kw |= {
            "red_vertices": [highlight],
            "arrows": [(highlight, e) for e in src.star(highlight)],
            "arrow_color": TANGENT,
        }
        w = f.vertex_map[highlight]
        right_kw |= {"red_vertices": [w], "arrows": [(w, f.edge_map[e]) for e in src.star(highlight)]}
        right_kw["arrow_color"] = TANGENT
    else:
        image = f.image()
        if image != f.target.region():  # an onto map would highlight everything
            right_kw |= {"region": image, "edge_points": "none"}
    left = space_scene(src, **left_kw)
    right = space_scene(f.target, **right_kw)
    return scene_row([left, right], title=title)


def scalar_scene(fn, *, highlight: Point | None = None, region: Region | None = None, title: str = "") -> Scene:
    """A scalar function drawn as a level picture: each vertex at the height of its value, each edge's
    point (a star) at the height of its value, half a level up for an edge $\\{n, n+1\\}$; the scalar
    line on the left. ``highlight`` draws the derivative at a vertex; ``region`` highlights points."""
    space = fn.space
    base = layout_of(space)
    levels: dict[int, list] = {}
    for v in space.vertex_list():
        levels.setdefault(fn.values[v], []).append(v)
    spread = max(len(vs) for vs in levels.values())
    gap = 1.0
    layout = {}
    for n, vs in levels.items():
        vs = sorted(vs, key=lambda v: (float(base[v][0]), str(v)))
        width = (len(vs) - 1) * gap
        for i, v in enumerate(vs):
            layout[v] = np.array([i * gap - width / 2, float(n)])
    span = max(spread * gap, max(levels) - min(levels) + 1.0)
    lo, hi = min(levels) - 1, max(levels) + 1
    x0 = -spread * gap / 2 - 1.2
    scene = space_scene(
        space,
        layout=layout,
        region=region,
        red_vertices=[highlight] if highlight is not None else (),
        edge_points="none",
        vertex_text={v: point_text(v) for v in space.vertex_list()},
        title=title,
    )
    for e in space.edge_list():
        val = fn.edge_values[e]
        y = float(min(val)) + (0.5 if len(val) == 2 else 0.0)
        cx = float(np.mean([layout[v][0] for v in e])) + (_pts_to_units(scene, 10.0) if len(e) == 1 else 0.0)
        hot = region is not None and e in region.edges
        derivative = highlight is not None and e in space.star(highlight)
        colour = HIGHLIGHT if hot else TANGENT if derivative else FAINT
        scene.shapes.append(_star(scene, np.array([cx, y]), colour))
        if derivative:  # the derivative: the departure to the edge's value
            _arrow(scene, layout[highlight], np.array([cx, y]), TANGENT)
    # the scalar line, on the left: points lo..hi, unary disks, the steps between them
    ys = np.arange(lo, hi + 1, dtype=float)
    line = np.column_stack([np.full_like(ys, x0), ys])
    scene.shapes.append(Curve(line, EDGE_LINE, 1.0, z=1.5))
    scene.shapes.append(Dots(line, 9.0, (*EDGE_LINE[:3], 0.18), None, 0.0, z=1.2))
    scene.shapes.append(Dots(line, 2.6, (1, 1, 1, 1), EDGE_LINE, 0.8, z=3.0))
    for y in ys:
        scene.shapes.append(Label(np.array([x0, y]), (-12.0, 0.0), str(int(y)), size=8.0, color=INK))
        scene.shapes.append(
            Curve(np.array([[x0 + 0.25, y], [x0 + spread * gap + 1.4, y]]), (0.85, 0.87, 0.93, 1.0), 0.5, z=0.2)
        )
    scene.points = np.vstack([scene.points, line, [[x0 + spread * gap + 1.4, lo]]])
    scene.span = max(scene.span, span)
    return scene


def incidence_and_space(space: SpatialHypergraph, region: Region | None = None, title: str = "") -> Scene:
    return scene_row(
        [space_scene(space, region=region), incidence_scene(space, region=region)], arrows=False, title=title
    )


# -- the rewriting calculus: paths, barriers, fields, rules ---------------------------------

PALETTE_NAMES = ("green", "blue", "brown", "red", "violet")


def _roles(kinds: Sequence[str]) -> dict[str, str]:
    """Colour names by role, so every picture of a square reads alike: the first field green, the second
    blue, further fields red and violet; a composite A;B brown when A came first, red otherwise; a bracket
    violet; a closure A+ (or any type derived from one field) in that field's colour."""
    kinds = list(dict.fromkeys(kinds))
    bases = [k for k in kinds if not any(c in k for c in ";[]+")]
    base_names = ["green", "blue", "red", "violet", "brown"]
    out = {k: base_names[i % len(base_names)] for i, k in enumerate(bases)}
    rank: dict[str, int] = {b: i for i, b in enumerate(bases)}  # fields in order of appearance
    for k in kinds:
        for part in k.strip("[]").replace(",", ";").split(";"):
            rank.setdefault(part.rstrip("+"), len(rank))
    for k in kinds:
        if k in out:
            continue
        if k.startswith("["):
            out[k] = "violet"
        elif ";" in k:
            first, second = (x.rstrip("+") for x in k.split(";", 1))
            out[k] = "brown" if rank[first] <= rank[second] else "red"
        else:
            root = k.rstrip("+")
            out[k] = out.get(root, "green")
    return out


def kind_colors(kinds: Sequence[str]) -> dict[str, RGBA]:
    """A colour for each field type of a state, by its role (see :func:`kind_color_names`)."""
    return {k: field_color(name) for k, name in _roles(kinds).items()}


def kind_color_names(kinds: Sequence[str]) -> dict[str, str]:
    return _roles(list(kinds))


def _barrier_marks(scene: Scene, layout: Mapping, pairs: Iterable[frozenset]) -> None:
    """A barrier pair as an amber bar across the step between its two positions."""
    half = _pts_to_units(scene, 7.0)
    for pair in pairs:
        a, b = (np.asarray(layout[v], float) for v in sort_points(pair))
        m, u = (a + b) / 2, b - a
        n = np.array([-u[1], u[0]]) / (np.linalg.norm(u) or 1.0)
        scene.shapes.append(Curve(np.array([m - half * n, m + half * n]), HIGHLIGHT, 3.2, z=3.6, round_cap=True))


def step_path_scene(
    path,
    *,
    others: Sequence[tuple[object, RGBA]] = (),
    window: Sequence[Point] = (),
    area: frozenset | None = None,
    barrier=None,
    color: RGBA = MOTION,
    title: str = "",
) -> Scene:
    """A step path as green arcs from position to position (repeated steps bow further out); ``others``
    are further paths with their colours; ``window`` (the positions a rewrite touched) and ``area``
    (the edge it deformed across) are highlighted; a ``barrier`` shows as amber bars."""
    space = path.space
    layers = [(path.steps(), color)] + [(p.steps(), c) for p, c in others]
    scene = space_scene(
        space,
        red_vertices=list(window) or [path.start, path.end],
        red_edges=[area] if area is not None else [],
        edge_points="none",
        fields=layers,
        title=title,
    )
    if barrier is not None:
        _barrier_marks(scene, layout_of(space), barrier.pairs)
    return scene


def field_state_scene(
    state, *, kinds: Sequence[str] | None = None, new=(), labels: bool = True, title: str = ""
) -> Scene:
    """Typed edges in the colours of their types, with their repeats: arcs between points of $X(H)$ as curved
    arrows, unary edges as concentric disks. ``new`` edges (just added by a rule) are highlighted."""
    shown = list(kinds) if kinds is not None else list(state.kinds)
    colours = kind_colors(list(state.kinds) + shown)
    fresh = set(new)
    scene = space_scene(state.space, labels=labels, edge_points="none", title=title)
    layout = layout_of(state.space)
    for k, kind in enumerate(shown):
        arcs = [(e.tail, e.head) for e in state.copies(kind) if not e.unary and e not in fresh]
        draw_field_edges(scene, layout, arcs, colours[kind], layer=k)
        quanta = Counter(e.tail for e in state.copies(kind) if e.unary and e not in fresh)
        draw_quanta(scene, layout, quanta, colours[kind])
    hot = [e for e in sorted(fresh) if e.kind in shown]
    draw_field_edges(scene, layout, [(e.tail, e.head) for e in hot if not e.unary], HIGHLIGHT, layer=len(shown))
    draw_quanta(scene, layout, Counter(e.tail for e in hot if e.unary), HIGHLIGHT)
    return scene


def _small_space(edges, coords) -> SpatialHypergraph:
    return SpatialHypergraph(edges, coordinates=coords)


def rule_picture(left: Scene, right: Scene, title: str = "") -> Scene:
    return scene_row([left, right], title=title)


def homotopy_rule_scenes() -> list[tuple[str, Scene]]:
    """The area-homotopy rules drawn: the path (green) before and after, the area highlighted."""
    tri = _small_space([["a", "b", "c"]], {"a": (0.0, 0.0), "b": (2.0, 0.0), "c": (1.0, 1.5)})
    seg = _small_space([["a", "b"]], {"a": (0.0, 0.0), "b": (2.0, 0.0)})
    stay = _small_space([["a"], ["a", "b"]], {"a": (0.0, 0.0), "b": (2.0, 0.0)})

    def pic(space, steps, hot=(), area=None):
        return space_scene(
            space, fields=[(steps, MOTION)], red_vertices=hot, red_edges=[area] if area else [], edge_points="none"
        )

    area = frozenset("abc")
    return [
        (
            "contract  a c b → a b, across an area",
            rule_picture(pic(tri, [("a", "c"), ("c", "b")], area=area), pic(tri, [("a", "b")], area=area)),
        ),
        (
            "expand  a b → a c b, across an area",
            rule_picture(pic(tri, [("a", "b")], area=area), pic(tri, [("a", "c"), ("c", "b")], area=area)),
        ),
        ("cancel  a b a → a", rule_picture(pic(seg, [("a", "b"), ("b", "a")]), pic(seg, [], hot=["a"]))),
        ("stall  a a → a", rule_picture(pic(stay, [("a", "a")]), pic(stay, [], hot=["a"]))),
    ]


def field_rule_scene(rule) -> Scene:
    """A field rule drawn on a small piece of grid: its inputs (in their colours) and what it adds."""
    from infrageometry.edgefields import FieldEdge, FieldState, step_arcs

    sq = _small_space(
        [["a", "b"], ["b", "c"], ["a", "d"], ["d", "c"]],
        {"a": (0.0, 0.0), "b": (1.6, 0.0), "c": (1.6, 1.6), "d": (0.0, 1.6)},
    )
    line = _small_space([["a", "b"], ["b", "c"]], {"a": (0.0, 0.0), "b": (1.6, 0.0), "c": (3.2, 0.0)})
    if rule.kind == "chain":
        needs = list(rule.needs)
        # walk along the line: two arcs of a base type make one step; a derived type is one edge a step long
        before, at, i = [], "a", 0
        names = {"a": "b", "b": "c", "c": "c"}
        while i < len(needs):
            t = needs[i]
            if i + 1 < len(needs) and needs[i + 1] == t and not any(c in t for c in ";[]+"):
                before += step_arcs(line, t, at, names[at])
                i += 2
            else:
                before.append(FieldEdge(t, at, names[at]))
                i += 1
            at = names[at]
        added = [FieldEdge(rule.gives, "a", at)]
        space = line
        kinds = list(dict.fromkeys([*needs, rule.gives]))
    else:
        xy, yx = rule.needs
        space = sq
        before = [FieldEdge(xy, "a", "c"), FieldEdge(yx, "a", "b")]
        added = step_arcs(sq, rule.gives, "b", "c")
        kinds = [xy, yx, rule.gives]
    left = FieldState(space, before, tuple(kinds))
    right = left.add(added)
    return rule_picture(field_state_scene(left, kinds=kinds), field_state_scene(right, kinds=kinds, new=added))
