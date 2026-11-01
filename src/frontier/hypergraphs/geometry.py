r"""The geometry of hypergraph drawings: layout, shapes, and the scene both renderers paint.

The style is the one of ``HypergraphPlot`` in the hypergraph paclet
(``references/cache/hypergraph-paclet/src/HypergraphPlot.wl``) and of the
example images it publishes:

* one homogeneous blue hue for everything;
* hyperedges of arity 3 or more are translucent concave regions whose sides bow
  in towards the centre of their vertices, the spiderweb look (the paclet's
  ``ConcavePolygon``);
* binary edges are straight lines, bending only to separate parallel edges;
* a unary edge is a translucent disk centred on its vertex, and repeated unary
  edges are concentric disks;
* vertices are small light disks with a dark rim, of fixed size on the page.

Edges carry a symmetry type, as in the paclet. ``"unordered"`` edges, the
default, show no direction. A ``"cyclic"`` edge gets a small circular arrow at
its centre. ``"directed"`` (or ``"ordered"``) edges show their vertex sequence:

* a binary edge has one arrowhead at its middle;
* an edge $(v_1, \dots, v_k)$ with $k \ge 3$ is a region bounded by the sides
  $v_1 \to v_2, \dots, v_{k-1} \to v_k$ and the closing side $v_k \to v_1$.
  Each side is drawn and carries a half-arrowhead: a barb on the outer side of
  the region, pointing along the side, all of one size. The one on the first
  side $v_1 \to v_2$ is hollow (white, outlined in the arrow colour), so the
  sequence is read from there by following the arrowheads. (Set by the user on
  2026-10-28, replacing the internal half-arrowheads growing along the sequence.)

An ``"inputoutput(S:a,T:b)"`` edge (the user, 2026-10-31) has an input block of $a$
vertices and an output block of $b$ vertices, each of type S or T. Each block is drawn
as a lobe of its own type (a region for three or more vertices, the side between them
for two, the vertex itself for one), and the two lobes are one translucent region
joined by a narrow **waist** prolonged from their sides, with one arrowhead at the
waist from inputs to outputs. A directed block draws all its sides with their
half-arrowheads (the first hollow), the closing side included; a cyclic block gets the
cyclic badge at its lobe. One input and one output is the directed binary edge.

:func:`hypergraph_scene` turns a hypergraph into a list of shapes: positions in
layout coordinates, sizes of marks in printer's points (fixed on the page). The
matplotlib renderer (:mod:`frontier.viz.hypergraph`) and the SVG renderer
(:mod:`frontier.hypergraphs.svg`) only paint scenes, so the two always agree.
This module needs NumPy and NetworkX only.
"""

from __future__ import annotations

import colorsys
import math
import re
from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

Hyperedge = Sequence[Hashable]
Layout = Mapping[Hashable, tuple[float, float]]
RGBA = tuple[float, float, float, float]


def _hue(h: float, s: float, v: float, alpha: float = 1.0) -> RGBA:
    return (*colorsys.hsv_to_rgb(h, s, v), alpha)


# The paclet's light-mode colours ($...Style in HypergraphPlot.wl).
EDGE_LINE = _hue(0.63, 0.7, 0.5, 0.7)  # binary edges, boundary arrows
EDGE_FILL = _hue(0.63, 0.66, 0.81, 0.05)  # hyperedge regions (measured from the published images)
UNARY_FILL = _hue(0.63, 0.66, 0.81, 0.1)  # unary disks
NULL_EDGE = _hue(0.63, 0.7, 0.5, 0.4)
VERTEX_FACE = _hue(0.63, 0.26, 0.89)
VERTEX_RIM = _hue(0.63, 0.7, 0.33, 0.95)
MARKED_VERTEX = _hue(0.63, 0.7, 0.33)  # a marked vertex (a contracted index): solid dark blue
# Highlights (a rewrite's match, what a cell points at) are a desaturated amber that stands out from
# the neutral blue drawing, not the paclet's saturated red (the user, 2026-10-29).
HIGHLIGHT = _hue(0.095, 0.62, 0.86)
HIGHLIGHT_RIM = _hue(0.085, 0.7, 0.62)
LABEL = (0.0, 0.0, 0.0, 1.0)

# Objects drawn on a space (paths, vector fields, the edges of a field) take desaturated tones from
# this palette, in this order, so that several can be told apart (the user, 2026-10-29).
FIELD_COLORS = {
    "green": _hue(0.37, 0.48, 0.56),
    "blue": _hue(0.56, 0.52, 0.64),
    "brown": _hue(0.075, 0.55, 0.52),
    "red": _hue(0.995, 0.5, 0.68),
    "violet": _hue(0.79, 0.32, 0.58),
}
FIELD_ORDER = tuple(FIELD_COLORS)

# Special arrays drawn as hyperedges keep the geometry of every hyperedge and only get a colour
# treatment (the user, 2026-10-30): white is a white body with black sides and marks; red, green and
# blue a light tint of the hue in the body and the solid hue on the sides and marks. The blue is
# lighter and greener than the neutral drawing blue so that the two can be told apart.
EDGE_TINTS = {  # tint name -> (line, fill)
    "white": ((0.1, 0.1, 0.1, 0.9), (1.0, 1.0, 1.0, 0.94)),
    "red": (_hue(0.995, 0.72, 0.78), _hue(0.995, 0.6, 0.95, 0.18)),
    "green": (_hue(0.37, 0.75, 0.55), _hue(0.37, 0.55, 0.85, 0.18)),
    "blue": (_hue(0.57, 0.85, 0.82), _hue(0.57, 0.6, 0.95, 0.18)),
    "purple": (_hue(0.78, 0.55, 0.62), _hue(0.78, 0.45, 0.9, 0.18)),  # the all-ones cubix
}
# The label of a white hyperedge is a white letter outlined in black (the user, 2026-10-30): (fill, outline).
EDGE_TINT_LABELS = {"white": ((1.0, 1.0, 1.0, 1.0), (0.1, 0.1, 0.1, 1.0))}


def field_color(k: int | str) -> RGBA:
    """The k-th colour of the field palette (cycling), or the colour named ``k``."""
    return FIELD_COLORS[k] if isinstance(k, str) else FIELD_COLORS[FIELD_ORDER[k % len(FIELD_ORDER)]]


def star_points(center, outer: float, inner: float | None = None, turn: float = 0.0) -> np.ndarray:
    """A four-pointed star about ``center`` (layout units): the mark of an edge seen as a point."""
    inner = 0.38 * outer if inner is None else inner
    c = np.asarray(center, dtype=float)
    angles = [turn + k * math.pi / 4 for k in range(8)]
    radii = [outer if k % 2 == 0 else inner for k in range(8)]
    return np.array([c + r * np.array([math.cos(a), math.sin(a)]) for a, r in zip(angles, radii, strict=True)])


# Sizes in printer's points, fixed on the page whatever the extent of the layout.
VERTEX_RADIUS = 2.6
HIGHLIGHT_VERTEX_RADIUS = 3.6  # the paclet's rewriting highlight: PointSize[0.02]
UNARY_RADIUS = 12.0  # the paclet: 0.03 of the plot size, rendered at 400 r / size points
UNARY_STEP = 4.0  # each repeated unary edge adds a ring this much wider
LINE_WIDTH = 1.25
ARROW_SIZES = (9.0, 12.0, 15.0)  # Small, Medium, Large along a hyperedge
LABEL_SIZE = 9.0
LABEL_OFFSET = 9.0  # from a vertex to the centre of its label, for a one-character label
LABEL_CHAR = 0.3 * LABEL_SIZE  # a longer label stands this much farther off per extra character, sideways
EDGE_LABEL_OFFSET = 7.0  # a binary edge's label stands off the line by this much

# Half-arrowheads on directed hyperedges, in units of the edge's own size: each is
# HALF_ARROW_UNIT of the mean side long (at most HALF_ARROW_MAX of the shortest side), and
# the barb stands off the side, outwards, by HALF_ARROW_WIDTH of its length.
HALF_ARROW_UNIT = 0.12
HALF_ARROW_MAX = 0.45
HALF_ARROW_WIDTH = 0.42
HALF_ARROW_FIRST_FILL = (1.0, 1.0, 1.0, 1.0)  # the first one is hollow: white, outlined

SYMMETRIES = {"unordered": "unordered", "undirected": "unordered", "directed": "directed", "ordered": "directed"}
SYMMETRIES["cyclic"] = "cyclic"
_IO = re.compile(r"^\s*inputoutput\s*\(\s*(\w+)\s*:\s*(\d+)\s*,\s*(\w+)\s*:\s*(\d+)\s*\)\s*$", re.IGNORECASE)


def _symmetry(name: str) -> str:
    found = _IO.match(name) if isinstance(name, str) else None
    if found:  # an input–output type: "inputoutput(s:a,t:b)", or "directed" for one input and one output
        s, a, t, b = _symmetry(found.group(1)), int(found.group(2)), _symmetry(found.group(3)), int(found.group(4))
        if "inputoutput" in s or "inputoutput" in t or a < 1 or b < 1:
            raise ValueError(f"invalid input–output symmetry {name!r}")
        return "directed" if a == b == 1 else f"inputoutput({s}:{a},{t}:{b})"
    try:
        return SYMMETRIES[name.lower()]
    except KeyError:
        raise ValueError(
            f"unknown edge symmetry {name!r}: use unordered, directed, cyclic or inputoutput(<type>:<a>,<type>:<b>)"
        ) from None


def io_blocks(symm: str) -> tuple[str, int, str, int] | None:
    """``(input type, a, output type, b)`` of a normalized input–output symmetry, None for the other types."""
    found = _IO.match(symm)
    return (found.group(1), int(found.group(2)), found.group(3), int(found.group(4))) if found else None


def _symmetries(symmetry: str | Sequence[str], n: int) -> list[str]:
    symms = [_symmetry(symmetry)] * n if isinstance(symmetry, str) else [_symmetry(s) for s in symmetry]
    if len(symms) != n:
        raise ValueError("give one symmetry per edge, or a single one for all")
    return symms


# -- layout ---------------------------------------------------------------


# Spring weight per unit of arity on a hyperedge's spokes. Chosen so that a ternary edge's
# sides come out about 0.8 of a binary edge, the proportion in the paclet's published images.
SPOKE_WEIGHT = 3.0
# Spring weight of the sequence cycle of a directed hyperedge (see layout_graph).
SEQUENCE_WEIGHT = 2.0


def layout_graph(edges: Sequence[Hyperedge], directed: Sequence[bool] = (), symmetries: Sequence[str] = ()) -> nx.Graph:
    """The graph the layout is computed on, built as the paclet builds it.

    Binary edges join their ends. Each hyperedge with distinct vertices gets a
    helper vertex joined to all of them, weighted by its arity, which holds the
    region open with its vertices spread around it. A directed hyperedge of arity
    4 or more is also tied in a cycle in sequence order, so that its vertices go
    round the region in that order and every side joins consecutive vertices. A
    hyperedge that repeats a vertex joins its consecutive vertices instead. An input–output
    edge (``symmetries``) holds each block open round its own helper (a one-vertex block is
    its vertex) and ties the two blocks together, so that inputs and outputs sit apart.
    """
    g = nx.Graph()
    for i, e in enumerate(edges):
        g.add_nodes_from(e)
        io = io_blocks(symmetries[i]) if i < len(symmetries) else None
        if io is not None:
            anchors = []
            for side, block, sym in (("in", e[: io[1]], io[0]), ("out", e[io[1] :], io[2])):
                distinct = list(dict.fromkeys(block))
                if len(distinct) == 1:
                    anchors.append(distinct[0])
                    continue
                helper = ("__edge__", (i, side))
                g.add_edges_from(((helper, v) for v in distinct), weight=SPOKE_WEIGHT * len(distinct))
                if sym == "directed" and len(distinct) == len(block) > 3:
                    g.graph.setdefault("sequences", {})[helper] = tuple(block)
                anchors.append(helper)
            if anchors[0] != anchors[1]:
                g.add_edge(anchors[0], anchors[1], weight=1.0)
            continue
        if len(e) == 2 and e[0] != e[1]:
            g.add_edge(e[0], e[1], weight=1.0)
        elif len(e) > 2 and len(set(e)) == len(e):
            g.add_edges_from(((("__edge__", i), v) for v in e), weight=SPOKE_WEIGHT * len(e))
            if len(e) > 3 and i < len(directed) and directed[i]:
                g.graph.setdefault("sequences", {})[("__edge__", i)] = tuple(e)
                for a, b in zip(e, (*e[1:], e[0]), strict=True):
                    w = g.get_edge_data(a, b, {"weight": 0.0})["weight"]
                    g.add_edge(a, b, weight=w + SEQUENCE_WEIGHT)
        elif len(e) > 2:
            g.add_edges_from(((a, b) for a, b in zip(e, e[1:], strict=False) if a != b), weight=1.0)
    return g


def hypergraph_layout(
    edges: Sequence[Hyperedge],
    *,
    symmetry: str | Sequence[str] = "unordered",
    seed: int = 0,
    vertices: Sequence[Hashable] = (),
    fixed: Mapping[Hashable, Sequence[float]] | None = None,
) -> dict[Hashable, tuple[float, float]]:
    """Positions of the vertices. Connected components are laid out apart and set side by side.

    ``vertices`` adds vertices that lie on no edge (isolated vertices). ``fixed`` gives some
    vertices their positions, as when the two sides of a rule or an event share a layout; the
    other vertices settle among them.
    """
    symms = _symmetries(symmetry, len(edges))
    g = layout_graph(edges, [s == "directed" for s in symms], symms)
    g.add_nodes_from(vertices)
    known = {v: np.asarray(fixed[v], dtype=float) for v in g if fixed and not _is_helper(v) and v in fixed}
    if known:
        return _settle_around(g, known, seed)
    comps = sorted(nx.connected_components(g), key=lambda c: (-len(c), str(sorted(map(str, c)))))
    pos: dict[Hashable, tuple[float, float]] = {}
    x0 = 0.0
    for comp in comps:
        sub = g.subgraph(comp)
        if len(comp) == 1:
            local = {next(iter(comp)): np.zeros(2)}
        else:
            local = nx.spring_layout(sub, seed=seed, iterations=300, weight="weight")
            for _ in range(3):  # settle, then end seated: the last pass only permutes slots
                if not _seat_in_sequence(sub, local):
                    break
                local = nx.spring_layout(sub, pos=local, iterations=60, weight="weight")
            _seat_in_sequence(sub, local)
        pts = np.array(list(local.values()), dtype=float)
        pts -= pts.mean(axis=0)
        if len(pts) >= 2:  # turn the long axis horizontal, as the paclet's layouts read
            _, _, vt = np.linalg.svd(pts, full_matrices=False)
            pts = pts @ vt.T
        extent = float(np.ptp(pts, axis=0).max()) if len(pts) > 1 else 0.0
        scale = math.sqrt(sum(1 for v in comp if not _is_helper(v))) / extent if extent else 1.0
        pts *= scale
        width = float(np.ptp(pts[:, 0])) if len(pts) > 1 else 0.0
        pts[:, 0] += x0 - pts[:, 0].min()
        for v, p in zip(local, pts, strict=True):
            if not _is_helper(v):
                pos[v] = (float(p[0]), float(p[1]))
        x0 += width + 1.0
    return pos


def _settle_around(g: nx.Graph, known: dict, seed: int) -> dict[Hashable, tuple[float, float]]:
    """Positions with the ``known`` vertices held in place and the rest laid out among them."""
    free = [v for v in g if v not in known and not _is_helper(v)]
    pos = {v: tuple(map(float, p)) for v, p in known.items()}
    if not free:
        return pos
    start = dict(known)
    rng = np.random.default_rng(seed)
    centre = np.mean(list(known.values()), axis=0)
    for v in g:  # helpers at the middle of what they hold; free vertices near what they touch
        if v not in start:
            near = [known[u] for u in g.neighbors(v) if u in known]
            start[v] = (np.mean(near, axis=0) if near else centre) + rng.normal(scale=0.3, size=2)
    settled = nx.spring_layout(g, pos=start, fixed=list(known), k=1.0, seed=seed, iterations=300, weight="weight")
    pos.update({v: (float(settled[v][0]), float(settled[v][1])) for v in free})
    return pos


def _seat_in_sequence(g: nx.Graph, local: dict) -> bool:
    """Put each directed hyperedge's vertices round its helper in sequence order.

    The vertices keep the angular slots the layout gave them, reassigned so that the
    sequence runs round them in order. Returns whether anything moved.
    """
    moved = False
    for helper, seq in g.graph.get("sequences", {}).items():
        if helper not in local or not all(v in local for v in seq):
            continue
        c = np.asarray(local[helper])
        rel = {v: np.asarray(local[v]) - c for v in seq}
        slots = sorted(rel.values(), key=lambda d: math.atan2(d[1], d[0]))
        start = min(range(len(slots)), key=lambda q: float(np.linalg.norm(slots[q] - rel[seq[0]])))
        slots = slots[start:] + slots[:start]
        for v, d in zip(seq, slots, strict=True):
            if not np.allclose(rel[v], d):
                local[v] = c + d
                moved = True
    return moved


def _is_helper(v: Hashable) -> bool:
    return isinstance(v, tuple) and len(v) == 2 and v[0] == "__edge__"


# -- geometry -------------------------------------------------------------


def _bspline(ctrl: np.ndarray, samples: int = 40) -> np.ndarray:
    """Clamped uniform B-spline through the end points (degree up to 3), as WL's BSplineCurve."""
    n = len(ctrl)
    k = min(3, n - 1)
    knots = np.concatenate([np.zeros(k), np.linspace(0, 1, n - k + 1), np.ones(k)])
    out = []
    for t in np.linspace(0, 1, samples):
        t = min(t, 1 - 1e-12)
        span = int(np.searchsorted(knots, t, side="right") - 1)
        d = [ctrl[j].copy() for j in range(span - k, span + 1)]
        for r in range(1, k + 1):
            for j in range(k, r - 1, -1):
                i = span - k + j
                denom = knots[i + k - r + 1] - knots[i]
                a = 0.0 if denom == 0 else (t - knots[i]) / denom
                d[j] = (1 - a) * d[j - 1] + a * d[j]
        out.append(d[k])
    out[-1] = ctrl[-1]
    return np.array(out)


def _cross(u, v) -> float:
    return float(u[0] * v[1] - u[1] * v[0])


def _hull_centroid(pts: np.ndarray) -> np.ndarray:
    """Centroid of the convex hull (the paclet's RegionCentroid @ ConvexHullRegion)."""
    uniq = np.unique(pts, axis=0)
    if len(uniq) < 3:
        return uniq.mean(axis=0)
    p = sorted(map(tuple, uniq))

    def half(seq):
        h: list[tuple[float, float]] = []
        for q in seq:
            while len(h) >= 2 and _cross(np.subtract(h[-1], h[-2]), np.subtract(q, h[-2])) <= 0:
                h.pop()
            h.append(q)
        return h

    hull = np.array(half(p)[:-1] + half(p[::-1])[:-1])
    x, y = hull[:, 0], hull[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area = cross.sum() / 2
    if abs(area) < 1e-12:
        return uniq.mean(axis=0)
    return np.array([((x + np.roll(x, -1)) * cross).sum(), ((y + np.roll(y, -1)) * cross).sum()]) / (6 * area)


def _concave_side(a: np.ndarray, b: np.ndarray, center: np.ndarray, n: int = 1) -> np.ndarray:
    """Side of a hyperedge region from a to b: interior control points pulled halfway to the centre."""
    ctrl = np.array([a + t * (b - a) for t in np.linspace(0, 1, n + 2)])
    ctrl[1:-1] = (ctrl[1:-1] + center) / 2
    return _bspline(ctrl)


def _bent_line(a: np.ndarray, b: np.ndarray, bend: float) -> np.ndarray:
    """Straight segment, or a symmetric arc bowed sideways by ``bend`` times its length."""
    if bend == 0:
        return np.array([a, b])
    d = b - a
    mid = (a + b) / 2 + bend * np.array([-d[1], d[0]])
    return _bspline(np.array([a, mid, b]))


def _loop(p: np.ndarray, radius: float, angle: float = math.pi / 2) -> np.ndarray:
    """A circular loop through p, lying on the side given by ``angle``, run counterclockwise."""
    c = p + radius * np.array([math.cos(angle), math.sin(angle)])
    t = angle + math.pi + np.linspace(0, 2 * math.pi, 60)
    return c + radius * np.stack([np.cos(t), np.sin(t)], axis=1)


def _ring_boundary(distinct: Sequence[Hashable], pos: Mapping, center: np.ndarray, n: int) -> np.ndarray:
    """Boundary of an unordered region: concave sides between angularly adjacent vertices."""
    ring = sorted(distinct, key=lambda v: math.atan2(*(pos[v] - center)[::-1]))
    return np.vstack(
        [_concave_side(pos[a], pos[b], center, n)[:-1] for a, b in zip(ring, ring[1:] + ring[:1], strict=True)]
    )


def _along(curve: np.ndarray, frac: float) -> tuple[np.ndarray, np.ndarray]:
    """The point at ``frac`` of the arc length of ``curve``, and the unit tangent there."""
    acc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    s = frac * acc[-1]
    i = int(np.clip(np.searchsorted(acc, s), 1, len(curve) - 1))
    a, b = curve[i - 1], curve[i]
    step = max(float(np.linalg.norm(b - a)), 1e-12)
    return a + (b - a) * (s - acc[i - 1]) / step, (b - a) / step


def ordered_marks(
    edge: Sequence[Hashable], pos: Mapping, center: np.ndarray, n: int = 1
) -> list[tuple[np.ndarray, np.ndarray, bool]]:
    r"""Sides and half-arrowheads of a directed hyperedge $(v_1, \dots, v_k)$.

    Returns, for $i = 1, \dots, k$, the side $v_i \to v_{i+1}$ (indices mod $k$) as a
    curve, its half-arrowhead as a triangle (tip, back on the side, back off the side
    away from the centre), and whether it is the first side $v_1 \to v_2$, whose
    half-arrowhead is drawn hollow. All half-arrowheads have one size. Sides joining a
    vertex to itself are skipped.
    """
    k = len(edge)
    pairs = [(i, edge[i], edge[(i + 1) % k]) for i in range(k) if edge[i] != edge[(i + 1) % k]]
    chords = [float(np.linalg.norm(pos[b] - pos[a])) for _, a, b in pairs]
    if not chords:
        return []
    length = min(HALF_ARROW_UNIT * float(np.mean(chords)), HALF_ARROW_MAX * min(chords))
    out = []
    for i, a, b in pairs:
        curve = _concave_side(pos[a], pos[b], center, n)
        mid, t = _along(curve, 0.5)
        outward = np.array([-t[1], t[0]])
        if np.dot(outward, center - mid) > 0:
            outward = -outward
        tip, back = mid + 0.5 * length * t, mid - 0.5 * length * t
        out.append((curve, np.array([tip, back, back + HALF_ARROW_WIDTH * length * outward]), i == 0))
    return out


def _densify(curve: np.ndarray, samples: int = 12) -> np.ndarray:
    """Points along a polyline, so a straight segment repels a label along its whole length."""
    if len(curve) > 2:
        return curve
    return np.linspace(curve[0], curve[-1], samples)


def _clearest(p: np.ndarray, obstacles: np.ndarray, reach: float) -> np.ndarray:
    """Of 16 directions, the one whose point at ``reach`` from p is farthest from everything drawn.

    Ties go to the upper right, where labels conventionally sit.
    """
    best, best_score = np.array([0.7, 0.7]), -1.0
    for k in range(16):
        t = math.pi / 4 + k * math.pi / 8
        u = np.array([math.cos(t), math.sin(t)])
        q = p + reach * u
        score = float(np.hypot(*(obstacles - q).T).min())
        if score > best_score + 1e-9:
            best, best_score = u, score
    return best


def _free_angle(p: np.ndarray, others: np.ndarray) -> float:
    away = _away(p, others)
    return math.atan2(away[1], away[0])


def _away(p: np.ndarray, others: np.ndarray) -> np.ndarray:
    """Direction pushing a label off the drawing: inverse-square repulsion from the other points."""
    d = p - others
    dist = np.hypot(d[:, 0], d[:, 1])
    near = dist > 1e-9
    push = (d[near] / dist[near, None] ** 3).sum(axis=0)
    norm = float(np.hypot(*push))
    return push / norm if norm > 1e-9 else np.array([0.7, 0.7])


# -- the scene ------------------------------------------------------------
#
# Positions are in layout coordinates (y up); widths, radii and arrow sizes are in points.
# Renderers paint shapes in list order within each z level, as matplotlib does.


@dataclass
class Area:
    """A filled and/or outlined closed region."""

    points: np.ndarray
    fill: RGBA | None
    stroke: RGBA | None = None
    width: float = 0.0
    z: float = 1.0


@dataclass
class Curve:
    points: np.ndarray
    color: RGBA
    width: float
    z: float = 2.0
    round_cap: bool = False


@dataclass
class Dots:
    """Disks of a fixed size on the page, centred at layout points."""

    centers: np.ndarray
    radius: float
    fill: RGBA | None
    stroke: RGBA | None = None
    width: float = 0.0
    z: float = 1.0


@dataclass
class Arrowhead:
    """A solid arrowhead whose tip is at ``head``, pointing from ``tail`` (matplotlib's ``-|>``)."""

    tail: np.ndarray
    head: np.ndarray
    size: float
    color: RGBA
    z: float = 2.5
    shrink: float = 0.0  # points trimmed off both ends of the arrow's path, as matplotlib's shrinkA/B


@dataclass
class ArcStroke:
    """A circular arc in layout units, angles in degrees."""

    center: np.ndarray
    radius: float
    theta1: float
    theta2: float
    color: RGBA
    width: float
    z: float = 2.5


@dataclass
class Label:
    """Text centred ``offset`` points away from ``anchor``."""

    anchor: np.ndarray
    offset: tuple[float, float]
    text: str
    size: float = LABEL_SIZE
    color: RGBA = LABEL
    z: float = 4.0
    outline: RGBA | None = None  # a halo around the letters (the white special array: white letter, black outline)


@dataclass
class Scene:
    shapes: list = field(default_factory=list)
    span: float = 1.0
    points: np.ndarray = field(default_factory=lambda: np.zeros((1, 2)))
    title: str = ""


def _arrowhead(curve: np.ndarray, size: float, color: RGBA) -> Arrowhead | None:
    """Arrowhead at the middle of ``curve``, pointing along it."""
    lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    if lengths[-1] == 0:
        return None
    half = lengths[-1] / 2
    i = int(np.clip(np.searchsorted(lengths, half), 1, len(curve) - 1))
    a, b = curve[i - 1], curve[i]
    tip = a + (b - a) * ((half - lengths[i - 1]) / max(lengths[i] - lengths[i - 1], 1e-12))
    direction = (b - a) / max(np.linalg.norm(b - a), 1e-12)
    eps = 1e-3 * lengths[-1]
    return Arrowhead(tip - eps * direction, tip + eps * direction, size, color)


def _cyclic_badge(at: np.ndarray, span: float, color: RGBA) -> list:
    """The paclet's badge for a cyclic edge: a small circular arrow at the edge's centre."""
    r = 0.045 * span
    end = at + r * np.array([math.cos(math.radians(396)), math.sin(math.radians(396))])
    tangent = np.array([-math.sin(math.radians(396)), math.cos(math.radians(396))])
    return [
        ArcStroke(at, r, 100, 396, color, 1.0),
        Arrowhead(end - 1e-3 * tangent, end, 7, color, shrink=2.0),
    ]


# The waist of an input–output edge: its half-width at the narrowest, as a fraction of the distance
# between the centres of the two blocks, and how far along the axis its sides run parallel.
WAIST_HALF_WIDTH = 0.07
WAIST_STRAIGHT = 0.18
TWO_VERTEX_BOW = 0.3  # a two-vertex lobe bows its side towards the waist by this much of the axis


def _lobe(block: Sequence[Hashable], pos: Mapping, front: np.ndarray, reach: float) -> tuple[list, np.ndarray]:
    """A block's distinct vertices in counterclockwise order starting just left of ``front`` (the
    direction of the waist), and the centre its sides bow towards."""
    distinct = list(dict.fromkeys(block))
    pts = np.array([pos[v] for v in distinct])
    c = _hull_centroid(pts)
    if len(distinct) == 2:
        c = pts.mean(axis=0) + TWO_VERTEX_BOW * reach * front
    elif len(distinct) == 1:
        c = pts[0]
    left = np.array([-front[1], front[0]])

    def turn(v):
        d = pos[v] - c
        return math.atan2(float(np.dot(d, left)), float(np.dot(d, front))) % (2 * math.pi)

    return sorted(distinct, key=turn), c


def _neck(p: np.ndarray, q: np.ndarray, at: np.ndarray, along: np.ndarray, reach: float) -> np.ndarray:
    """A side of the waist from p to q, through ``at`` and running parallel to ``along`` there."""
    d = WAIST_STRAIGHT * reach * along
    return _bspline(np.array([p, at - d, at, at + d, q]), samples=50)


def io_edge_shapes(
    e: Sequence[Hashable],
    io: tuple[str, int, str, int],
    pos: Mapping,
    span: float,
    *,
    line: RGBA,
    fill: RGBA,
    width: float,
    hot: bool,
    n: int = 1,
) -> tuple[list, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Shapes of an input–output edge: two lobes joined by a waist, its arrowhead and the blocks' marks.

    Returns the shapes, the boundary (for labels to keep clear of) and where the edge's text goes."""
    s_in, a, s_out, _ = io
    ins, outs = tuple(e[:a]), tuple(e[a:])
    ci = _hull_centroid(np.array([pos[v] for v in ins]))
    co = _hull_centroid(np.array([pos[v] for v in outs]))
    axis = co - ci
    reach = float(np.linalg.norm(axis))
    shapes: list = []
    if reach < 1e-6 * span:  # both blocks on the same place: a loop through them, its arrow along it
        distinct = list(dict.fromkeys(e))
        others = np.array([p for v, p in pos.items() if v not in distinct] or [ci + 1.0])
        loop = _loop(ci, 0.1 * span, _free_angle(ci, others))
        shapes += [Area(loop, fill, z=1.0), Curve(loop, line, width)]
        if head := _arrowhead(loop, ARROW_SIZES[1], line):
            shapes.append(head)
        return shapes, loop, (loop.mean(axis=0), np.zeros(2))
    u = axis / reach
    nrm = np.array([-u[1], u[0]])
    waist = (ci + co) / 2
    half = WAIST_HALF_WIDTH * reach
    ring_in, c_in = _lobe(ins, pos, u, reach)
    ring_out, c_out = _lobe(outs, pos, -u, reach)

    def sides(ring, c):
        return [_concave_side(pos[x], pos[y], c, n)[:-1] for x, y in zip(ring, ring[1:], strict=False)]

    boundary = np.vstack(
        [
            *sides(ring_in, c_in),
            _neck(pos[ring_in[-1]], pos[ring_out[0]], waist - half * nrm, u, reach)[:-1],
            *sides(ring_out, c_out),
            _neck(pos[ring_out[-1]], pos[ring_in[0]], waist + half * nrm, -u, reach)[:-1],
        ]
    )
    shapes.append(Area(boundary, fill, z=1.0))
    if hot:
        shapes.append(Area(boundary, None, line, 0.8 * width, z=2.0))
    shapes.append(Arrowhead(waist - 1e-3 * reach * u, waist + 1e-3 * reach * u, ARROW_SIZES[1], line))
    for block, sym, c in ((ins, s_in, c_in), (outs, s_out, c_out)):
        distinct = list(dict.fromkeys(block))
        if sym == "directed" and len(distinct) >= 2:
            for curve, barb, first in ordered_marks(block, pos, c, n):
                head = (
                    Area(barb, HALF_ARROW_FIRST_FILL, (*line[:3], 0.9), 0.8 * width, z=2.6)
                    if first
                    else Area(barb, (*line[:3], 0.9), z=2.5)
                )
                shapes += [Curve(curve, line, width), head]
        elif sym == "cyclic" and len(distinct) >= 2:
            shapes += _cyclic_badge(np.mean([pos[v] for v in distinct], axis=0), span, line)
    up = nrm if nrm[1] >= 0 else -nrm
    return shapes, boundary, (waist + half * up, EDGE_LABEL_OFFSET * up)


def hypergraph_scene(
    edges: Sequence[Hyperedge],
    *,
    layout: Layout | None = None,
    symmetry: str | Sequence[str] = "unordered",
    labels: bool = False,
    title: str = "",
    highlight: Sequence[int] = (),
    seed: int = 0,
    vertices: Sequence[Hashable] = (),
    vertex_text: Mapping[Hashable, str] | None = None,
    edge_text: Sequence[str | None] = (),
    highlight_vertices: Sequence[Hashable] = (),
    marked_vertices: Sequence[Hashable] = (),
    edge_tints: Sequence[str | None] = (),
) -> Scene:
    """The shapes of a hypergraph drawing.

    ``edges`` are vertex sequences; ``symmetry`` is one type for all edges or one per
    edge; ``highlight`` lists edge indices drawn highlighted (amber); ``vertices`` adds isolated
    vertices. ``labels`` writes vertex names beside the vertices (``vertex_text``
    replaces the names); ``edge_text`` puts one text per edge at its centre.
    ``highlight_vertices`` draws those vertices highlighted; ``marked_vertices`` draws those
    solid dark blue (the contracted indices of a plex diagram). ``edge_tints`` gives one tint name of
    :data:`EDGE_TINTS` (or None) per edge: the edge keeps its geometry and takes that colour treatment.
    """
    edges = [tuple(e) for e in edges]
    symms = _symmetries(symmetry, len(edges))
    layout = layout or hypergraph_layout(edges, symmetry=symms, seed=seed, vertices=vertices)
    pos = {v: np.asarray(p, dtype=float) for v, p in layout.items()}
    order = list(dict.fromkeys([*(v for e in edges for v in e), *vertices]))
    missing = [v for v in order if v not in pos]
    if missing:
        raise ValueError(f"layout gives no position for {missing!r}")
    pts = np.array([pos[v] for v in order]) if order else np.zeros((1, 2))
    span = float(np.ptp(pts, axis=0).max()) or 1.0
    scene = Scene(span=span, points=pts, title=title)
    shapes = scene.shapes

    pair_total = Counter(frozenset(e) for e in edges if len(e) == 2)
    pair_seen: Counter = Counter()
    region_seen: Counter = Counter()
    unary_seen: Counter = Counter()
    nulls = 0
    drawn: list[np.ndarray] = []  # curves that labels must keep clear of
    centers: list[np.ndarray | None] = []  # where each edge's text goes
    text_colors: list[RGBA] = []
    for i, (e, symm) in enumerate(zip(edges, symms, strict=True)):
        hot = i in highlight
        tint = edge_tints[i] if i < len(edge_tints) else None
        line = HIGHLIGHT if hot else EDGE_LINE
        fill = (*HIGHLIGHT[:3], 0.3) if hot else (UNARY_FILL if len(e) == 1 else EDGE_FILL)
        if tint and not hot:
            line, fill = EDGE_TINTS[tint]
        text_colors.append(
            EDGE_TINT_LABELS.get(tint, ((*line[:3], 1.0), None)) if tint and not hot else ((*line[:3], 1.0), None)
        )
        width = 2.2 * LINE_WIDTH if hot else LINE_WIDTH
        arrows = symm == "directed"
        distinct = list(dict.fromkeys(e))
        io = io_blocks(symm)

        if io is not None:
            region_seen[("io", frozenset(distinct))] += 1
            made, boundary, at = io_edge_shapes(
                e, io, pos, span, line=line, fill=fill, width=width, hot=hot, n=region_seen[("io", frozenset(distinct))]
            )
            shapes += made
            drawn.append(boundary)
            centers.append(at)
        elif not e:
            corner = pts.min(axis=0) + np.array([nulls * 0.12 * span, -0.15 * span])
            shapes.append(Dots(corner[None, :], UNARY_RADIUS, None, NULL_EDGE, 1.0))
            centers.append(corner)
            nulls += 1
        elif len(e) == 1:  # unary edge: a disk centred on its vertex
            v = e[0]
            unary_seen[v] += 1
            r = UNARY_RADIUS + (unary_seen[v] - 1) * UNARY_STEP
            shapes.append(Dots(pos[v][None, :], r, fill))
            centers.append(pos[v])
        elif len(distinct) == 1:  # (v, v, ...): a self-loop, larger for each repeat
            v = distinct[0]
            unary_seen[("loop", v)] += 1
            curve = _loop(pos[v], (0.07 + 0.03 * (unary_seen[("loop", v)] - 1)) * span, _free_angle(pos[v], pts))
            shapes.append(Curve(curve, line, width))
            drawn += [curve, curve.mean(axis=0, keepdims=True)]  # keep labels out of the loop too
            centers.append(curve.mean(axis=0))
            if arrows and (head := _arrowhead(curve, ARROW_SIZES[0], line)):
                shapes.append(head)
        elif len(distinct) == 2:  # binary edge, or longer edge on two vertices
            key = frozenset(distinct)
            pair_seen[key] += 1
            m, j = pair_total.get(key, 1), pair_seen[key]
            bend = 0.0 if m == 1 else (j - (m + 1) / 2) * 0.25
            first = min(distinct, key=lambda v: (str(type(v)), str(v)))
            mid = None
            normal = None
            for k, (a, b) in enumerate(zip(e, e[1:], strict=False)):
                if a == b:
                    continue
                # Bend relative to a fixed orientation of the pair, so (1,2) and (2,1) part ways.
                curve = _bent_line(pos[a], pos[b], bend if a == first else -bend)
                shapes.append(Curve(curve, line, width, round_cap=True))
                drawn.append(_densify(curve))
                if mid is None:
                    mid, t = _along(curve, 0.5)
                    normal = np.array([-t[1], t[0]])
                    normal = normal if normal[1] >= 0 else -normal  # the label goes above the line
                if arrows and (head := _arrowhead(curve, ARROW_SIZES[min(k, 2)], line)):
                    shapes.append(head)
            centers.append(None if mid is None else (mid, EDGE_LABEL_OFFSET * normal))
        else:  # hyperedge: concave region
            region_seen[frozenset(distinct)] += 1
            n = region_seen[frozenset(distinct)]
            coords = np.array([pos[v] for v in distinct])
            center = _hull_centroid(coords)
            sides: list = []
            if arrows:
                marks = ordered_marks(e, pos, center, n)
                if len(distinct) == len(e):  # the region is bounded by the sequence's own sides
                    boundary = np.vstack([curve[:-1] for curve, _, _ in marks])
                else:
                    boundary = _ring_boundary(distinct, pos, center, n)
                for curve, barb, first in marks:
                    head = (
                        Area(barb, HALF_ARROW_FIRST_FILL, (*line[:3], 0.9), 0.8 * width, z=2.6)
                        if first
                        else Area(barb, (*line[:3], 0.9), z=2.5)
                    )
                    sides += [Curve(curve, line, width), head]
            else:
                boundary = _ring_boundary(distinct, pos, center, n)
            shapes += sides
            drawn.append(boundary)
            shapes.append(Area(boundary, fill, z=1.0))
            if hot and not arrows:
                shapes.append(Area(boundary, None, line, 0.8 * width, z=2.0))
            if symm == "cyclic":
                shapes += _cyclic_badge(center, span, line)
            centers.append(center)
        if symm == "cyclic" and len(distinct) < 3 and e:
            shapes += _cyclic_badge(np.mean([pos[v] for v in distinct], axis=0), span, line)

    if order:
        shapes.append(Dots(pts, VERTEX_RADIUS, VERTEX_FACE, VERTEX_RIM, 0.9, z=3.0))
        marked = [pos[v] for v in order if v in set(marked_vertices)]
        if marked:
            shapes.append(Dots(np.array(marked), VERTEX_RADIUS, MARKED_VERTEX, VERTEX_RIM, 0.9, z=3.05))
        hot = [pos[v] for v in order if v in set(highlight_vertices)]
        if hot:
            shapes.append(Dots(np.array(hot), HIGHLIGHT_VERTEX_RADIUS, HIGHLIGHT, HIGHLIGHT_RIM, 0.9, z=3.1))
    if labels or vertex_text:
        obstacles = np.vstack([pts, *drawn]) if drawn else pts
        for v, p in zip(order, pts, strict=True):
            text = (vertex_text or {}).get(v, str(v) if labels else None)
            if text is None:
                continue
            away = _clearest(p, obstacles, 0.07 * span)
            reach = LABEL_OFFSET + LABEL_CHAR * (len(text) - 1) * abs(float(away[0]))
            shapes.append(Label(p, tuple(reach * away), text))
    for text, at, (color, outline) in zip(edge_text, centers, text_colors, strict=False):
        if text is not None and at is not None:
            anchor, offset = at if isinstance(at, tuple) else (at, np.zeros(2))
            shapes.append(
                Label(np.asarray(anchor), tuple(float(o) for o in offset), text, color=color, outline=outline)
            )
    return scene


# -- several drawings in a row --------------------------------------------------------------

# The reference paclet's rule arrow (RewritingGraphics.wl), a right-pointing filled curve.
RULE_ARROW = np.array(
    [
        (-1.0, 0.1848),
        (0.2991, 0.1848),
        (-0.1531, 0.6363),
        (0.109, 0.8982),
        (1.0, 0.0034),
        (0.109, -0.8982),
        (-0.1531, -0.6363),
        (0.2991, -0.1848),
        (-1.0, -0.1848),
    ]
)
RULE_ARROW_COLOR = (0.65, 0.65, 0.65, 1.0)
PANEL_GAP = 0.9  # between a panel and the arrow, in spans
ARROW_HALF = 0.11  # half the arrow's length, in spans


def _transformed(shape, d: np.ndarray, scale: float = 1.0, marks: float = 1.0):
    """A copy of ``shape`` with positions mapped ``p -> scale * p + d`` and marks (sizes in points) times ``marks``."""

    def at(p):
        return scale * np.asarray(p, dtype=float) + d

    if isinstance(shape, Area):
        return Area(at(shape.points), shape.fill, shape.stroke, marks * shape.width, shape.z)
    if isinstance(shape, Curve):
        return Curve(at(shape.points), shape.color, marks * shape.width, shape.z, shape.round_cap)
    if isinstance(shape, Dots):
        return Dots(at(shape.centers), marks * shape.radius, shape.fill, shape.stroke, marks * shape.width, shape.z)
    if isinstance(shape, Arrowhead):
        return Arrowhead(at(shape.tail), at(shape.head), marks * shape.size, shape.color, shape.z, marks * shape.shrink)
    if isinstance(shape, ArcStroke):
        return ArcStroke(
            at(shape.center),
            scale * shape.radius,
            shape.theta1,
            shape.theta2,
            shape.color,
            marks * shape.width,
            shape.z,
        )
    if isinstance(shape, Label):
        offset = (marks * shape.offset[0], marks * shape.offset[1])
        return Label(at(shape.anchor), offset, shape.text, marks * shape.size, shape.color, shape.z, shape.outline)
    raise TypeError(f"unknown shape {shape!r}")


def _shifted(shape, d: np.ndarray):
    return _transformed(shape, d)


def scene_row(panels: Sequence[Scene], *, arrows: bool = True, title: str = "") -> Scene:
    """Drawings side by side on one scale, with the paclet's rule arrow between them.

    Each panel's title is written above it. The panels share the largest span and are moved
    only sideways, so a vertex at the same layout position in two panels sits at the same
    height in both.
    """
    span = max((p.span for p in panels), default=1.0)
    out = Scene(span=span, title=title)
    points, x = [], 0.0
    for k, panel in enumerate(panels):
        lo, hi = panel.points.min(axis=0), panel.points.max(axis=0)
        d = np.array([x - lo[0], 0.0])  # only sideways: a shared layout keeps its heights
        out.shapes += [_shifted(s, d) for s in panel.shapes]
        points.append(panel.points + d)
        if panel.title:
            top = hi[1]
            out.shapes.append(Label(np.array([(lo[0] + hi[0]) / 2 + d[0], top]), (0.0, 16.0), panel.title, 10.0))
        x += hi[0] - lo[0]
        if k < len(panels) - 1:
            if arrows:
                mid = x + PANEL_GAP * span / 2
                level = np.mean([(q.points.min(axis=0)[1] + q.points.max(axis=0)[1]) / 2 for q in panels])
                out.shapes.append(Area(RULE_ARROW * ARROW_HALF * span + np.array([mid, level]), RULE_ARROW_COLOR))
            x += PANEL_GAP * span
    out.points = np.vstack(points) if points else np.zeros((1, 2))
    return out


# -- graphs whose vertices are pictures -----------------------------------------------------
#
# The graphs of a rewriting system (evolution, states, causal, branchial) are drawn in the
# colours of the Wolfram Physics Project, as the reference paclet draws them: states blue,
# events amber, causal edges red, branchial edges magenta, element tokens green. A vertex is
# either a plain mark or a small framed picture (a panel) of the state or event it stands for.

STATE_COLOR = _hue(0.62, 0.45, 0.87, 0.7)
STATE_EDGE_COLOR = _hue(0.75, 0.0, 0.35)
EVENT_COLOR = _hue(0.11, 1.0, 0.97)
CAUSAL_COLOR = _hue(0.0, 1.0, 0.56)
BRANCHIAL_COLOR = _hue(0.89, 0.97, 0.71)
TOKEN_COLOR = _hue(0.35, 0.5, 0.75, 0.75)
FRAME_COLOR = _hue(0.62, 0.45, 0.6, 0.6)
PANEL_GROUND = (0.99, 0.99, 0.99, 1.0)
GRAPH_POINTS_PER_UNIT = 52.0  # a panel one unit wide is about the paclet's 90-pixel panel
GRAPH_VERTEX_RADIUS = 5.0
GRAPH_EDGE_WIDTH = 1.1
PANEL_MARGIN = 1.25  # space claimed around a panel, as a multiple of its size


def scene_box(scene: Scene) -> tuple[np.ndarray, np.ndarray]:
    """The layout-unit box a scene occupies, padded for its marks: (lower-left, upper-right)."""
    pad = 0.12 * scene.span
    return scene.points.min(axis=0) - pad, scene.points.max(axis=0) + pad


def panel_shapes(scene: Scene, center: Sequence[float], width: float, height: float, outer_span: float):
    """``scene`` shrunk into a framed panel ``width`` by ``height`` (layout units) at ``center``.

    The drawing is fitted inside the frame and centred. Positions map into the outer scene's
    layout units and the marks (dot sizes, line widths, text) shrink with them, though not
    below :data:`PANEL_MIN_MARKS` of their size, so that a vertex stays visible in miniature.
    Returns the shapes and the panel's half-size.
    """
    lo, hi = scene_box(scene)
    inner = np.maximum(hi - lo, 1e-9)
    scale = PANEL_FILL * min(width / inner[0], height / inner[1])
    c = np.asarray(center, dtype=float)
    d = c - scale * (lo + hi) / 2
    marks = max(scale * scene.span / outer_span, PANEL_MIN_MARKS)
    half = np.array([width / 2, height / 2])
    frame = np.array([c + [-half[0], -half[1]], c + [half[0], -half[1]], c + half, c + [-half[0], half[1]]])
    shapes = [Area(frame, PANEL_GROUND, FRAME_COLOR, 0.8, z=0.8)]
    shapes += [_transformed(s, d, scale, marks) for s in scene.shapes]
    return shapes, half


PANEL_FILL = 0.9  # of the frame the drawing may take
PANEL_MIN_MARKS = 0.45


def layered_layout(nodes: Sequence[Hashable], edges: Sequence[tuple], *, dx: float = 1.0, dy: float = 1.0) -> dict:
    """Positions for a directed graph in layers, sources on top (the paclet's LayeredDigraphEmbedding).

    A vertex's layer is the length of the longest path reaching it (edges closing a cycle are
    set aside); the order within a layer is improved by a few barycentre sweeps.
    """
    dag = nx.DiGraph()
    dag.add_nodes_from(nodes)
    for a, b in edges:  # in order, setting aside any edge that would close a cycle
        if a != b and not nx.has_path(dag, b, a):
            dag.add_edge(a, b)
    layer: dict = {}
    for v in nx.topological_sort(dag):
        layer[v] = max((layer[u] + 1 for u in dag.predecessors(v)), default=0)
    rows: dict[int, list] = {}
    for v in nodes:
        rows.setdefault(layer[v], []).append(v)
    depth = max(rows, default=-1) + 1
    order = {v: k for row in rows.values() for k, v in enumerate(row)}
    for sweep in range(8):
        down = sweep % 2 == 0
        for r in range(1, depth) if down else range(depth - 2, -1, -1):
            row = rows[r]

            def bary(v, down=down):
                near = list(dag.predecessors(v) if down else dag.successors(v))
                return sum(order[u] for u in near) / len(near) if near else order[v]

            row.sort(key=lambda v: (bary(v), order[v]))
            order.update({v: k for k, v in enumerate(row)})
    pos = {}
    for r, row in rows.items():
        for k, v in enumerate(row):
            pos[v] = (dx * (k - (len(row) - 1) / 2), -dy * r)
    return pos


def _boundary(center: np.ndarray, half: np.ndarray | float, toward: np.ndarray) -> np.ndarray:
    """Where the segment from ``center`` toward ``toward`` leaves a box (or disk) of half-size ``half``."""
    u = toward - center
    if np.allclose(u, 0):
        return center
    if np.isscalar(half):
        return center + half * u / np.linalg.norm(u)
    t = min(half[0] / abs(u[0]) if u[0] else np.inf, half[1] / abs(u[1]) if u[1] else np.inf)
    return center + min(t, 1.0) * u


def graph_scene(
    nodes: Sequence[Hashable],
    edges: Sequence[tuple],
    pos: dict,
    *,
    node_style: dict,
    edge_colors: dict,
    directed: bool = True,
    panels: dict | None = None,
    title: str = "",
) -> Scene:
    """A graph drawn with plain marks or panels at ``pos`` (layout units).

    ``node_style[v]`` is ``("disk" | "square" | "star", color)``; ``panels[v]``, when given, is a scene
    drawn framed in the vertex's place, ``(scene, width)``. Edges end at the boundary of what
    they join and carry an arrowhead when ``directed``.
    """
    panels = panels or {}
    span = 160.0 / GRAPH_POINTS_PER_UNIT  # the renderers draw 160 points per span
    points = np.array([pos[v] for v in nodes], dtype=float) if nodes else np.zeros((1, 2))
    scene = Scene(span=span, points=points, title=title)
    unit = 1.0 / GRAPH_POINTS_PER_UNIT  # one point, in layout units
    half: dict = {}
    for v in nodes:
        c = np.asarray(pos[v], dtype=float)
        if v in panels:
            inner, width, height = panels[v]
            shapes, h = panel_shapes(inner, c, width, height, span)
            scene.shapes += shapes
            half[v] = h
        else:
            shape, color = node_style[v]
            r = GRAPH_VERTEX_RADIUS
            if shape == "star":  # an edge seen as a point
                pts = star_points(c, 1.45 * r * unit)
                scene.shapes.append(Area(pts, color, color[:3] + (1.0,), 0.6, z=3.0))
            elif shape == "square":
                q = r * unit
                sq = np.array([c + [-q, -q], c + [q, -q], c + [q, q], c + [-q, q]])
                scene.shapes.append(Area(sq, color, color[:3] + (1.0,), 0.6, z=3.0))
            else:
                scene.shapes.append(Dots(c[None, :], r, color, color[:3] + (1.0,), 0.6, z=3.0))
            half[v] = r * unit
    for a, b in edges:
        color = edge_colors.get((a, b), STATE_EDGE_COLOR)
        pa, pb = np.asarray(pos[a], dtype=float), np.asarray(pos[b], dtype=float)
        if a == b:  # a self-loop, above the vertex
            h = half[a] if np.isscalar(half[a]) else half[a][1]
            top = pa + np.array([0.0, h])
            loop = _loop(top, 0.18, math.pi / 2)
            scene.shapes.append(Curve(loop, color, GRAPH_EDGE_WIDTH, z=0.5))
            if directed and (head := _arrowhead(loop, 7.0, color)):
                scene.shapes.append(head)
            continue
        start, end = _boundary(pa, half[a], pb), _boundary(pb, half[b], pa)
        line = np.array([start, end])
        scene.shapes.append(Curve(line, color, GRAPH_EDGE_WIDTH, z=0.5))
        if directed:
            u = end - start
            norm = float(np.linalg.norm(u))
            if norm > 1e-9:
                scene.shapes.append(Arrowhead(end - 0.25 * u / norm * min(norm, 1.0), end, 8.0, color, z=0.6))
    return scene
